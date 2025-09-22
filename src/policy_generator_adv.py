#!/usr/bin/env python3
"""
Adversarial (DSPy) Pro Mode policy generation entry point.

Pipeline (MVP):
1) Draft a policy (and basic test_config) without vector search (RAW prompt style).
2) Run RegistryGuard to canonicalize and collect provenance/facts.
3) Run StaticChecker to propose repairs.
4) Run 1–N debate rounds (Proponent/Opponent/Judge) to apply patches and converge.
5) Re-guard and re-check; compute scores and assemble metadata.
6) Derive test_config deterministically from final policy.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except Exception:
    import google.generativeai as genai  # type: ignore
    types = None  # type: ignore

from src.guards.registry_guard import RegistryGuard
from src.guards.static_checker import StaticChecker
from src.debate.dspy_program import DebateOrchestrator
from src.debate.dspy_llm import DSpyDebateOrchestrator
from src.scoring import compute_policy_scores


RAW_SYSTEM = (
    "You are an AWS IAM policy expert. Output only JSON with fields iam_policy and test_config. "
    "Use least-privilege principles, avoid wildcards if ARNs exist, add restrictive conditions where appropriate."
)


def _draft_raw_policy(nl_prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int], int]:
    """Call Gemini (or fallback no-op) to draft a policy + test_config. Returns (policy, test_config, tokens, latency_ms)."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    tokens = {"draft": 0}
    latency_ms = 0
    if not api_key:
        # fallback minimal stub for offline runs
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": "*"}
            ],
        }
        return policy, _policy_to_test_config(policy, default_service="s3"), tokens, latency_ms
    if types:
        client = genai.Client(api_key=api_key)
        prompt = f"{RAW_SYSTEM}\n\nNatural Language Requirement:\n{nl_prompt}\n\nOutput the JSON object with iam_policy and test_config."
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=3000),
        )
        text = (resp.text or "{}").strip()
    else:
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt := f"{RAW_SYSTEM}\n\nNatural Language Requirement:\n{nl_prompt}\n\nOutput the JSON object with iam_policy and test_config.")
        text = getattr(resp, "text", "{}")
    text = text.strip().lstrip("```json").rstrip("```")
    try:
        obj = json.loads(text)
        policy = obj.get("iam_policy") or {}
        test_cfg = obj.get("test_config") or _policy_to_test_config(policy)
    except Exception:
        # fallback naïve
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject"], "Resource": "*"}
            ],
        }
        test_cfg = _policy_to_test_config(policy, default_service="s3")
    return policy, test_cfg, tokens, latency_ms


def _policy_to_test_config(policy: Dict[str, Any], default_service: str | None = None) -> Dict[str, Any]:
    stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
    service = default_service or "s3"
    rules: List[Dict[str, Any]] = []
    rid = 1
    for st in stmts:
        acts = st.get("Action")
        actions: List[str] = []
        if isinstance(acts, str):
            actions = [acts]
        elif isinstance(acts, list):
            actions = [a for a in acts if isinstance(a, str)]
        # best-effort service from first action
        if actions and ":" in actions[0]:
            service = actions[0].split(":", 1)[0]
        rule = {
            "id": f"R{rid}",
            "effect": st.get("Effect", "Allow"),
            "principals": ["${PRINCIPAL_PLACEHOLDER}"],
            "not_principals": [],
            "actions": actions,
            "resources": st.get("Resource") if isinstance(st.get("Resource"), list) else [st.get("Resource", "*")],
            "conditions": st.get("Condition", {}),
        }
        rules.append(rule)
        rid += 1
    return {"service": service, "rules": rules}


def generate_adversarial(nl_prompt: str,
                         rounds: int = 1,
                         models: Dict[str, str] | None = None,
                         settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    models = models or {}
    draft_model = models.get("draft", "models/gemini-2.5-pro")
    judge_model = models.get("judge", models.get("pro", models.get("con", "models/gemini-2.5-flash")))
    pro_model = models.get("pro", judge_model)
    con_model = models.get("con", judge_model)
    settings = settings or {}

    # 1) Draft
    policy, test_cfg, tokens1, lat1 = _draft_raw_policy(nl_prompt, draft_model)

    # 2) Guard
    guard = RegistryGuard()
    guard_res = guard.guard(policy)

    # 3) Static checker
    checker = StaticChecker(provenance=guard_res["provenance"])
    check_res = checker.check_and_repair(guard_res["policy_canonical"])

    # 4) Debate loop (apply patches)
    use_dspy = bool(settings.get("use_dspy"))
    dspy_orch = DSpyDebateOrchestrator(draft_model=draft_model, judge_model=judge_model, pro_model=pro_model, con_model=con_model)
    orchestrator = DebateOrchestrator(draft_model=draft_model, judge_model=judge_model, pro_model=pro_model, con_model=con_model)
    all_patches: List[Dict[str, Any]] = []
    debate_summary: List[str] = []
    working_policy = guard_res["policy_canonical"]
    for r in range(max(1, min(3, rounds))):
        if use_dspy and dspy_orch.available():
            round_res = dspy_orch.run_round(nl_prompt, working_policy, registry_facts=guard_res["facts_text"])  # type: ignore
            # If DSPy produced no patches, fall back to static checker patches
            if not round_res["patches_applied"]:
                round_res = orchestrator.run_round(
                    nl_prompt,
                    working_policy,
                    registry_facts=guard_res["facts_text"],
                    guard_replacements=guard_res["replacements"],
                    checker_patches=check_res["repairs"],
                )
        else:
            round_res = orchestrator.run_round(
                nl_prompt,
                working_policy,
                registry_facts=guard_res["facts_text"],
                guard_replacements=guard_res["replacements"],
                checker_patches=check_res["repairs"],
            )
        if round_res["patches_applied"]:
            working_policy = _apply_patches(working_policy, round_res["patches_applied"])  # type: ignore
            all_patches.extend(round_res["patches_applied"])  # type: ignore
        debate_summary.append(round_res["summary"])  # type: ignore
        # Re-run guard/checker after patch
        guard_res = guard.guard(working_policy)
        checker = StaticChecker(provenance=guard_res["provenance"])
        check_res = checker.check_and_repair(guard_res["policy_canonical"])

    final_policy = guard_res["policy_canonical"]
    final_test = _policy_to_test_config(final_policy)

    # Scores
    scores = compute_policy_scores(final_policy)

    # Metadata assembly
    metadata = {
        "mode": "adversarial",
        "rounds": max(1, min(3, rounds)),
        "models": {"draft": draft_model, "judge": judge_model, "pro": pro_model, "con": con_model},
        "registry": {
            "mismatches": guard_res["out_of_registry"],
            "replacements": guard_res["replacements"],
            "provenance": guard_res["provenance"],
        },
        "debate": {
            "summary": "; ".join(debate_summary),
            "patches_applied": all_patches,
        },
        "static_checker": {
            "repairs": check_res["repairs"],
            "status": check_res["status"],
            "messages": check_res["messages"],
        },
        "tokens": {"draft": tokens1.get("draft", 0), "pro": 0, "con": 0, "judge": 0, "total": tokens1.get("draft", 0)},
        "latency_ms": {"draft": lat1, "debate": 0, "checker": 0, "total": lat1},
        "score": scores,
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "status": "success",
        "query": nl_prompt,
        "iam_policy": final_policy,
        "test_config": final_test,
        "metadata": metadata,
    }


# local JSON patch applier mirroring debate util
def _apply_patches(obj: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    import json as _json
    res = _json.loads(_json.dumps(obj))
    for p in patches:
        op = p.get("op")
        path = p.get("path", "")
        parts = [pp for pp in path.split("/") if pp != ""]
        cur = res
        for i, key in enumerate(parts):
            last = i == len(parts) - 1
            if not last:
                if key.isdigit():
                    cur = cur[int(key)]
                else:
                    cur = cur.setdefault(key, {})
            else:
                if op in ("replace", "add"):
                    val = p.get("value")
                    if key.isdigit():
                        cur[int(key)] = val
                    else:
                        cur[key] = val
    return res
