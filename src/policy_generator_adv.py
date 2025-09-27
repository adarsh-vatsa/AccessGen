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
from typing import Any, Dict, List, Tuple, Callable, Optional
from datetime import datetime

from dotenv import load_dotenv

import os as _os
try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
_FORCE_LEGACY = (_os.getenv("GENAI_FORCE_LEGACY", "0") == "1")
try:
    import google.generativeai as legacy_genai  # type: ignore
except Exception:
    legacy_genai = None  # type: ignore

from src.guards.registry_guard import RegistryGuard
from src.guards.static_checker import StaticChecker
from src.debate.trace_digest import build_trace_digest
from src.debate.patch_gate import light_patch_gate
from src.debate.policy_patcher import apply_patches, PatchError
from src.debate.llm_roles import LLMRoleClient, RoleOutputError
from src.scoring import compute_policy_scores
from src.guards.registry_validator import RegistryValidator
from src.policy_generator import IAMPolicyGeneratorV2


RAW_SYSTEM = (
    "You are an AWS IAM policy expert. Output only a valid AWS policy JSON when asked. "
    "Use least-privilege principles, avoid wildcards if ARNs exist, add restrictive conditions where appropriate."
)


def _build_policy_only_prompt(nl_prompt: str) -> str:
    return (
        "Return only one valid AWS S3 bucket policy as pure JSON. "
        "No comments. No code fences. No explanations.\n\n"
        "Rules:\n"
        "- Use S3 actions only and least privilege.\n"
        "- Use precise ARNs: bucket arn:aws:s3:::{{BUCKET_NAME}} and objects arn:aws:s3:::{{BUCKET_NAME}}/* .\n"
        "- Apply required conditions and explicit Deny statements as described.\n\n"
        f"Input:\n{nl_prompt}\n\n"
        "Output: a single JSON object with keys Version and Statement only."
    )


def _draft_policy_only(nl_prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, int], int, Dict[str, Any]]:
    """Draft a policy only."""
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
        trace = {
            "stage": "draft",
            "model": model,
            "input": {"system": RAW_SYSTEM, "nl_prompt": nl_prompt},
            "output_text": json.dumps(policy),
            "note": "offline stub (no GEMINI_API_KEY)",
        }
        return policy, tokens, latency_ms, trace

    full_prompt = _build_policy_only_prompt(nl_prompt)
    text = ""
    try:
        if _FORCE_LEGACY and legacy_genai is not None:
            legacy_genai.configure(api_key=api_key)
            gm = legacy_genai.GenerativeModel(model)
            resp = gm.generate_content(full_prompt)
            text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        elif types is not None and genai is not None:
            client = genai.Client(api_key=api_key)  # type: ignore
            resp = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=10000),
            )
            text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        elif legacy_genai is not None:
            legacy_genai.configure(api_key=api_key)
            gm = legacy_genai.GenerativeModel(model)
            resp = gm.generate_content(full_prompt)
            text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        else:
            raise RuntimeError("No usable Gemini client: install google-generativeai or google-genai")
    except Exception as _e:
        text = ""

    tclean = text.strip().lstrip("```json").rstrip("```")
    policy: Dict[str, Any] = {}
    try:
        obj = json.loads(tclean)
        if isinstance(obj, dict) and "Version" in obj and "Statement" in obj:
            policy = obj
        elif isinstance(obj, dict):
            for k in ("iam_policy", "policy", "bucket_policy", "bucketPolicy"):
                v = obj.get(k)
                if isinstance(v, dict) and "Version" in v and "Statement" in v:
                    policy = v
                    break
    except Exception:
        policy = {}

    if not policy:
        policy = {"Version": "2012-10-17", "Statement": []}
    trace = {
        "stage": "draft",
        "model": model,
        "input": {"system": RAW_SYSTEM, "nl_prompt": nl_prompt},
        "output_text": text,
    }
    return policy, tokens, latency_ms, trace


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


def _score(summary: Dict[str, Any], hints: List[str] | None) -> int:
    wildcard_penalty = int(bool(summary.get("has_action_wildcard"))) + int(bool(summary.get("has_resource_wildcard")))
    hint_penalty = len(hints or [])
    return -(wildcard_penalty + hint_penalty)


def generate_adversarial(nl_prompt: str,
                         rounds: int = 1,
                         models: Dict[str, str] | None = None,
                         settings: Dict[str, Any] | None = None,
                         trace_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    models = models or {}
    draft_model = models.get("draft", "models/gemini-2.5-pro")
    judge_model = models.get("judge", models.get("pro", models.get("con", "models/gemini-2.5-flash")))
    pro_model = models.get("pro", judge_model)
    con_model = models.get("con", judge_model)
    settings = settings or {}

    # 1) Draft
    trace: List[Dict[str, Any]] = []
    policy, tokens1, lat1, draft_trace = _draft_policy_only(nl_prompt, draft_model)
    trace.append(draft_trace)
    if trace_cb:
        trace_cb(draft_trace)

    # 2) Guard
    guard = RegistryGuard()
    guard_res = guard.guard(policy)
    _t = {
        "stage": "registry_guard",
        "input": {"policy": policy},
        "output": {
            "in_registry": guard_res.get("in_registry", []),
            "out_of_registry": guard_res.get("out_of_registry", []),
            "replacements": guard_res.get("replacements", {}),
        },
    }
    trace.append(_t)
    if trace_cb:
        trace_cb(_t)

    # 3) Static checker
    checker = StaticChecker(provenance=guard_res["provenance"])
    check_res = checker.check_and_repair(guard_res["policy_canonical"])
    _t = {
        "stage": "static_checker",
        "input": {"policy": guard_res["policy_canonical"]},
        "output": {"repairs": check_res.get("repairs", []), "messages": check_res.get("messages", [])},
    }
    trace.append(_t)
    if trace_cb:
        trace_cb(_t)

    # 4) Debate loop using LLM-backed roles
    try:
        roles = LLMRoleClient(pro_model=pro_model, con_model=con_model, judge_model=judge_model)
    except RoleOutputError as exc:
        return {"status": "error", "message": str(exc), "stage": "roles_init", "query": nl_prompt}

    prior_patch_records: List[Dict[str, Any]] = []
    gate_rejections: List[Dict[str, Any]] = []
    debate_summary: List[str] = []

    working_policy = guard_res["policy_canonical"]
    round_cap = max(1, min(3, rounds))

    initial_digest = build_trace_digest(nl_prompt, working_policy, guard_res, check_res, prior_patch_records, 0)
    trace.append({"stage": "trace_digest_0", "output": initial_digest})
    if trace_cb:
        trace_cb(trace[-1])

    prev_score = _score(initial_digest["policy_summary"], initial_digest["static_hints"])

    for round_idx in range(round_cap):
        round_no = round_idx + 1

        digest = build_trace_digest(nl_prompt, working_policy, guard_res, check_res, prior_patch_records, round_no)
        trace.append({"stage": f"trace_digest_{round_no}", "output": digest})
        if trace_cb:
            trace_cb(trace[-1])

        try:
            pro_resp = roles.pro(digest, working_policy)
        except RoleOutputError as exc:
            return {"status": "error", "message": str(exc), "stage": f"pro_round_{round_no}", "query": nl_prompt}

        try:
            con_resp = roles.con(digest, working_policy)
        except RoleOutputError as exc:
            return {"status": "error", "message": str(exc), "stage": f"con_round_{round_no}", "query": nl_prompt}

        try:
            judge_resp = roles.judge(digest, working_policy, pro_resp, con_resp)
        except RoleOutputError as exc:
            return {"status": "error", "message": str(exc), "stage": f"judge_round_{round_no}", "query": nl_prompt}

        trace.extend([
            {
                "stage": f"pro_round_{round_no}",
                "model": roles.pro_model,
                "output": {
                    "patches": pro_resp.get("patches", []),
                    "rationale": pro_resp.get("rationale", ""),
                },
                "raw": pro_resp.get("raw"),
            },
            {
                "stage": f"con_round_{round_no}",
                "model": roles.con_model,
                "output": {
                    "patches": con_resp.get("patches", []),
                    "rationale": con_resp.get("rationale", ""),
                },
                "raw": con_resp.get("raw"),
            },
            {
                "stage": f"judge_round_{round_no}",
                "model": roles.judge_model,
                "output": {
                    "decision": judge_resp.get("decision"),
                    "patches": judge_resp.get("patches", []),
                    "reason": judge_resp.get("reason", ""),
                },
                "raw": judge_resp.get("raw"),
            },
        ])
        if trace_cb:
            for entry in trace[-3:]:
                trace_cb(entry)

        score_current = _score(digest["policy_summary"], digest["static_hints"])

        if judge_resp.get("decision") == "no-change":
            debate_summary.append(f"Round {round_no}: no-change ({judge_resp.get('reason', '')})")
            prior_patch_records.append(
                {
                    "patch_id": f"R{round_no}-NC",
                    "accepted": False,
                    "reason": judge_resp.get("reason", ""),
                    "reject_reason": "no-change",
                }
            )
            if not check_res.get("messages") and not check_res.get("repairs"):
                break
            prev_score = score_current
            continue

        def _registry_validate(patch: Dict[str, Any]) -> bool:
            try:
                candidate = apply_patches(working_policy, [patch])
            except PatchError:
                return False
            result = guard.guard(candidate)
            return not result.get("out_of_registry")

        accepted, rejected = light_patch_gate(nl_prompt, judge_resp.get("patches", []), _registry_validate)

        gate_log = {"stage": f"light_patch_gate_{round_no}", "output": {"accepted": accepted, "rejected": rejected}}
        trace.append(gate_log)
        if trace_cb:
            trace_cb(gate_log)

        # Record patch outcomes
        for idx, patch in enumerate(accepted, 1):
            prior_patch_records.append(
                {
                    "patch_id": f"R{round_no}-{idx:03d}",
                    "accepted": True,
                    "reason": judge_resp.get("reason", ""),
                    "reject_reason": None,
                }
            )
        for ridx, rej in enumerate(rejected, 1):
            gate_rejections.append(
                {
                    "round": round_no,
                    "rule": rej.get("rule"),
                    "reason": rej.get("reason"),
                }
            )
            prior_patch_records.append(
                {
                    "patch_id": f"R{round_no}-X{ridx:02d}",
                    "accepted": False,
                    "reason": judge_resp.get("reason", ""),
                    "reject_reason": rej.get("reason"),
                }
            )

        if not accepted:
            debate_summary.append(
                f"Round {round_no}: patches rejected ({len(rejected)}) - {judge_resp.get('reason', '')}"
            )
            if score_current <= prev_score:
                break
            prev_score = score_current
            continue

        try:
            working_policy = apply_patches(working_policy, accepted)
        except PatchError as exc:
            return {"status": "error", "message": f"Failed to apply patches: {exc}", "stage": f"apply_round_{round_no}", "query": nl_prompt}

        guard_res = guard.guard(working_policy)
        checker = StaticChecker(provenance=guard_res["provenance"])
        check_res = checker.check_and_repair(guard_res["policy_canonical"])
        working_policy = guard_res["policy_canonical"]

        trace.append(
            {
                "stage": f"registry_guard_post_{round_no}",
                "input": {"policy": working_policy},
                "output": {
                    "in_registry": guard_res.get("in_registry", []),
                    "out_of_registry": guard_res.get("out_of_registry", []),
                    "replacements": guard_res.get("replacements", {}),
                },
            }
        )
        trace.append(
            {
                "stage": f"static_checker_post_{round_no}",
                "input": {"policy": working_policy},
                "output": {
                    "repairs": check_res.get("repairs", []),
                    "messages": check_res.get("messages", []),
                },
            }
        )
        if trace_cb:
            trace_cb(trace[-2])
            trace_cb(trace[-1])

        updated_digest = build_trace_digest(
            nl_prompt, working_policy, guard_res, check_res, prior_patch_records, round_no
        )
        trace.append({"stage": f"trace_digest_post_{round_no}", "output": updated_digest})
        if trace_cb:
            trace_cb(trace[-1])

        new_score = _score(updated_digest["policy_summary"], updated_digest["static_hints"])
        debate_summary.append(
            f"Round {round_no}: applied {len(accepted)} patches ({judge_resp.get('reason', '')})"
        )
        prev_score = new_score

        if not check_res.get("messages") and not check_res.get("repairs"):
            break

    final_guard = guard.guard(working_policy)
    final_checker = StaticChecker(provenance=final_guard["provenance"])
    final_check = final_checker.check_and_repair(final_guard["policy_canonical"])
    final_policy = final_guard["policy_canonical"]
    final_test = _policy_to_test_config(final_policy)
    _tfin = {
        "stage": "finalize",
        "output": {"policy": final_policy, "test_config": final_test},
    }
    trace.append(_tfin)
    if trace_cb:
        trace_cb(_tfin)

    # Scores
    scores = compute_policy_scores(final_policy)

    # Registry validation
    validator = RegistryValidator()
    reg_checks = validator.validate_policy(final_policy, final_guard.get("provenance", {}))
    test_checks = validator.validate_test_config(final_test)

    # Metadata assembly
    metadata = {
        "mode": "adversarial",
        "rounds": len(debate_summary),
        "models": {"draft": draft_model, "judge": judge_model, "pro": pro_model, "con": con_model},
        "registry": {
            "mismatches": final_guard.get("out_of_registry", []),
            "replacements": final_guard.get("replacements", {}),
            "provenance": final_guard.get("provenance", {}),
        },
        "debate": {
            "summary": "; ".join(debate_summary) if debate_summary else "no debate rounds",
            "rounds_run": len(debate_summary),
            "gate_rejections": gate_rejections,
        },
        "static_checker": {
            "repairs": final_check.get("repairs", []),
            "status": final_check.get("status", "pass"),
            "messages": final_check.get("messages", []),
        },
        "tokens": {
            "draft": tokens1.get("draft", 0),
            "pro": 0,
            "con": 0,
            "judge": 0,
            "total": tokens1.get("draft", 0),
        },
        "latency_ms": {"draft": lat1, "debate": 0, "checker": 0, "total": lat1},
        "score": scores,
        "timestamp": datetime.now().isoformat(),
        "trace": trace,
        "registry_validation": {
            "policy": reg_checks,
            "test_config": test_checks,
        },
        "patch_records": prior_patch_records,
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
