#!/usr/bin/env python3
import os
import json
import warnings
import sys
from typing import Any, Dict, Optional

# Suppress urllib3 warnings on macOS LibreSSL Python builds
warnings.filterwarnings("ignore", module="urllib3")

import requests


POLICY_SCHEMA: Dict[str, Any] = {
    "title": "policy_schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["Version", "Statement"],
    "properties": {
        "Version": {"type": "string"},
        "Statement": {"type": "array"},
    },
}


TEST_CONFIG_SCHEMA: Dict[str, Any] = {
    "title": "test_config_schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["service", "rules"],
    "properties": {
        "service": {"type": "string"},
        "rules": {"type": "array"},
    },
}


_DOTENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
    except Exception:
        pass
    _DOTENV_LOADED = True


def _get_api_key() -> str:
    _ensure_env_loaded()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return api_key


def _call_openai(payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = _get_api_key()
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI request failed: {resp.status_code} {resp.text}")
    return resp.json()


def _extract_text(data: Dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts = []
    for blk in data.get("output") or []:
        for item in (blk or {}).get("content") or []:
            text = (item or {}).get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if not parts:
        raise RuntimeError("OpenAI response did not include text output")
    return "\n".join(parts).strip()


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def _extract_json(data: Dict[str, Any]) -> Any:
    output_text = data.get("output_text")
    candidates = []
    if isinstance(output_text, str) and output_text.strip():
        candidates.append(output_text)
    for blk in data.get("output") or []:
        for item in (blk or {}).get("content") or []:
            parsed = (item or {}).get("parsed")
            if parsed is not None:
                return parsed
            text = (item or {}).get("text")
            if isinstance(text, str) and text.strip():
                candidates.append(text)
    for raw in candidates:
        try:
            return json.loads(_strip_code_fence(raw))
        except Exception:
            continue
    raise RuntimeError("OpenAI response did not include valid JSON output")


def generate_openai_text(prompt: str, *, model: Optional[str] = None, max_output_tokens: int = 256) -> str:
    model_id = model or os.getenv("OPENAI_MODEL", "gpt-5")
    payload = {
        "model": model_id,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "text"}},
    }
    data = _call_openai(payload)
    return _extract_text(data)


def generate_openai_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    max_output_tokens: int = 8000,
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    model_id = model or os.getenv("OPENAI_MODEL", "gpt-5")
    fmt: Dict[str, Any]
    if schema:
        fmt = {
            "type": "json_schema",
            "name": schema.get("title", "output_schema"),
            "schema": schema,
            "strict": True,
        }
    else:
        fmt = {"type": "json_object"}
    payload = {
        "model": model_id,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "text": {"format": fmt},
    }
    if instructions:
        payload["instructions"] = instructions
    data = _call_openai(payload)
    result = _extract_json(data)
    if not isinstance(result, dict):
        raise RuntimeError("Expected JSON object from OpenAI")
    return result


def generate_policy_bundle(nl_task: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    prompt = (
        "Return only a JSON object with exactly two fields: "
        "'iam_policy' and 'test_config'.\n\n"
        "iam_policy: a valid AWS policy object with keys Version and Statement.\n"
        "test_config: an object with keys 'service' (one of 's3','ec2','iam') and 'rules' (array of objects with keys id, effect, principals, not_principals, actions, resources, conditions).\n"
        "For test_config.rules, set id sequentially as 'R1', 'R2', ... in order.\n\n"
        f"Task: {nl_task} Use Version 2012-10-17. No prose."
    )
    bundle = generate_openai_json(prompt, model=model, max_output_tokens=8000, instructions="Return only JSON")
    if not isinstance(bundle, dict) or "iam_policy" not in bundle or "test_config" not in bundle:
        raise RuntimeError("OpenAI bundle response missing required fields")
    return bundle


def main() -> int:
    nl_task = " ".join(sys.argv[1:]).strip() or "Allow listing S3 buckets (s3:ListAllMyBuckets)."
    try:
        bundle = generate_policy_bundle(nl_task)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


