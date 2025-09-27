#!/usr/bin/env python3
"""LLM-backed role prompts for adversarial debate rounds."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

try:  # pragma: no cover - import guarded for optional deps
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore

try:  # pragma: no cover
    import google.generativeai as legacy_genai  # type: ignore
except Exception:  # pragma: no cover
    legacy_genai = None  # type: ignore

from src.policy_generator import IAMPolicyGeneratorV2


MAX_PATCHES = 5
DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 10000
PROMPT_SEPARATOR = "\n\n"


PRO_PROMPT = (
    "You are PRO. Goal: find where the current policy under-matches the NL intent and propose the smallest "
    "JSON Patches to close the gap. Respond ONLY with JSON and keep rationale to one short sentence."
    "\n\nINPUTS:\n- nl_prompt: {nl_prompt}\n- current_policy (canonical JSON):\n{policy_json}\n"
    "- trace_digest:\n{trace_json}\n"
    "RULES:\n- Prefer add/replace/remove that minimally achieve the intent.\n"
    "- Avoid widening beyond what nl_prompt requires.\n"
    "- Never introduce Principal:\"*\" or Action:\"*\".\n"
    "OUTPUT JSON SCHEMA:\n{{\"patches\":[...] , \"rationale\":\"...\"}}\n"
    "JSON PATCH SCHEMA:\n- op: add|remove|replace\n- path: JSON pointer\n- value: required for add/replace"
)


CON_PROMPT = (
    "You are CON. Goal: identify over-grants relative to the NL intent and propose the smallest patches to "
    "remove excess without breaking the main goal. Respond ONLY with JSON and keep rationale to one short sentence."
    "\n\nINPUTS:\n- nl_prompt: {nl_prompt}\n- current_policy (canonical JSON):\n{policy_json}\n"
    "- trace_digest:\n{trace_json}\n"
    "RULES:\n- Prefer narrowing Resource/Principal or removing unnecessary actions.\n"
    "- Suggest conditions to bound scope when safe.\n"
    "- Never introduce Principal:\"*\" or Action:\"*\".\n"
    "OUTPUT JSON SCHEMA:\n{{\"patches\":[...] , \"rationale\":\"...\"}}"
)


JUDGE_PROMPT = (
    "You are JUDGE. Decide whether to apply patches. Keep output to JSON with a short reason."
    "\n\nTRACE DIGEST:\n{trace_json}\n"
    "CURRENT POLICY:\n{policy_json}\n"
    "PRO POSITION:\n{pro_json}\n"
    "CON POSITION:\n{con_json}\n"
    "CHECKLIST:\n1) Meets NL intent?\n2) Unjustified wildcards removed?\n3) Patch set minimal?\n"
    "OUTPUT OPTIONS:\nA) {{\"decision\":\"apply\", \"patches\":[...], \"reason\":\"...\"}}\n"
    "B) {{\"decision\":\"no-change\", \"patches\":[], \"reason\":\"Policy already matches NL intent\"}}"
)


class RoleOutputError(RuntimeError):
    """Raised when a role response cannot be parsed or fails validation."""


def _short_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    # limit to two short sentences
    return " ".join(lines[:2])[:280]


def _normalise_patches(patches: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(patches, list):
        return out
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        op = patch.get("op")
        path = patch.get("path")
        if op not in {"add", "remove", "replace"} or not isinstance(path, str):
            continue
        entry: Dict[str, Any] = {"op": op, "path": path}
        if op in {"add", "replace"}:
            if "value" not in patch:
                continue
            entry["value"] = patch.get("value")
        out.append(entry)
        if len(out) >= MAX_PATCHES:
            break
    return out


def _ensure_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise RoleOutputError("LLM response was not valid JSON")

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RoleOutputError("LLM response was not valid JSON") from exc


@dataclass
class LLMRoleClient:
    pro_model: str
    con_model: str
    judge_model: str
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RoleOutputError("GEMINI_API_KEY is not configured for debate roles")
        self._force_legacy = os.getenv("GENAI_FORCE_LEGACY", "0") == "1"
        self._genai_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pro(self, trace_digest: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        prompt = PRO_PROMPT.format(
            nl_prompt=trace_digest.get("nl_prompt", ""),
            policy_json=json.dumps(policy, indent=2, sort_keys=True),
            trace_json=json.dumps(trace_digest, indent=2, sort_keys=True),
        )
        raw = self._invoke(self.pro_model, prompt)
        data = self._parse_role_payload(raw, required_keys={"patches", "rationale"})
        patches = _normalise_patches(data.get("patches"))
        return {"patches": patches, "rationale": _short_text(data.get("rationale", "")), "raw": raw}

    def con(self, trace_digest: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        prompt = CON_PROMPT.format(
            nl_prompt=trace_digest.get("nl_prompt", ""),
            policy_json=json.dumps(policy, indent=2, sort_keys=True),
            trace_json=json.dumps(trace_digest, indent=2, sort_keys=True),
        )
        raw = self._invoke(self.con_model, prompt)
        data = self._parse_role_payload(raw, required_keys={"patches", "rationale"})
        patches = _normalise_patches(data.get("patches"))
        return {"patches": patches, "rationale": _short_text(data.get("rationale", "")), "raw": raw}

    def judge(
        self,
        trace_digest: Dict[str, Any],
        policy: Dict[str, Any],
        pro_resp: Dict[str, Any],
        con_resp: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_prompt = JUDGE_PROMPT.format(
            trace_json=json.dumps(trace_digest, indent=2, sort_keys=True),
            policy_json=json.dumps(policy, indent=2, sort_keys=True),
            pro_json=json.dumps({"patches": pro_resp.get("patches", []), "rationale": pro_resp.get("rationale", "")}, indent=2, sort_keys=True),
            con_json=json.dumps({"patches": con_resp.get("patches", []), "rationale": con_resp.get("rationale", "")}, indent=2, sort_keys=True),
        )

        try:
            return self._invoke_judge(base_prompt)
        except RoleOutputError:
            reminder_prompt = base_prompt + PROMPT_SEPARATOR + "REMINDER: Respond EXACTLY with JSON per schema."
            return self._invoke_judge(reminder_prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_judge(self, prompt: str) -> Dict[str, Any]:
        raw = self._invoke(self.judge_model, prompt)
        data = self._parse_role_payload(raw, required_keys={"decision", "patches", "reason"})
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in {"apply", "no-change"}:
            raise RoleOutputError("Judge decision must be 'apply' or 'no-change'")
        patches = _normalise_patches(data.get("patches"))
        reason = _short_text(data.get("reason", ""))
        return {"decision": decision, "patches": patches, "reason": reason, "raw": raw}

    def _parse_role_payload(self, raw: str, required_keys: set[str]) -> Dict[str, Any]:
        data = _ensure_json(raw)
        if not isinstance(data, dict):
            raise RoleOutputError("Role response must be a JSON object")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise RoleOutputError(f"Role response missing keys: {', '.join(missing)}")
        return data

    def _invoke(self, model: str, prompt: str) -> str:
        text = ""
        try:
            if self._force_legacy and legacy_genai is not None:
                legacy_genai.configure(api_key=self.api_key)
                gm = legacy_genai.GenerativeModel(model)
                resp = gm.generate_content(prompt)
                text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore[attr-defined]
            elif types is not None and genai is not None:
                if self._genai_client is None:
                    self._genai_client = genai.Client(api_key=self.api_key)  # type: ignore[attr-defined]
                resp = self._genai_client.models.generate_content(  # type: ignore[attr-defined]
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=float(self.temperature), max_output_tokens=int(self.max_output_tokens)),
                )
                text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore[attr-defined]
            elif legacy_genai is not None:
                legacy_genai.configure(api_key=self.api_key)
                gm = legacy_genai.GenerativeModel(model)
                resp = gm.generate_content(prompt)
                text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore[attr-defined]
            else:  # pragma: no cover
                raise RoleOutputError("No usable Gemini client for debate roles")
        except RoleOutputError:
            raise
        except Exception as exc:  # pragma: no cover
            raise RoleOutputError(f"LLM call failed for model {model}: {exc}") from exc

        if not isinstance(text, str) or not text.strip():
            raise RoleOutputError(f"Empty response from model {model}")
        return text.strip()


