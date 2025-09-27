#!/usr/bin/env python3
"""Utilities for building bounded trace digests for debate rounds."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def _sha256_json(obj: Any) -> str:
    """Stable JSON hash (sorted keys, utf-8, compact separators)."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _principal_type(statement: Dict[str, Any]) -> str:
    principal = statement.get("Principal")
    if principal == "*":
        return "star"
    if isinstance(principal, dict):
        val = principal.get("AWS")
        if isinstance(val, str):
            if val == "*":
                return "star"
            if val.isdigit():
                return "account"
            if ":" in val:
                return "arn"
    if principal is None:
        return "none"
    return "other"


def summarize_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    stmts = policy.get("Statement", [])
    if isinstance(stmts, dict):
        stmts = [stmts]

    actions = set()
    has_action_wc = False
    has_resource_wc = False
    principal_type = "none"

    for st in stmts:
        if not isinstance(st, dict):
            continue
        act = st.get("Action")
        if isinstance(act, str):
            actions.add(act)
            if act == "*" or act.endswith(":*"):
                has_action_wc = True
        elif isinstance(act, list):
            for a in act:
                if not isinstance(a, str):
                    continue
                actions.add(a)
                if a == "*" or a.endswith(":*"):
                    has_action_wc = True

        res = st.get("Resource")
        if res == "*":
            has_resource_wc = True
        elif isinstance(res, list) and any(r == "*" for r in res if isinstance(r, str)):
            has_resource_wc = True

        pt = _principal_type(st)
        if pt != "none":
            principal_type = pt

    statement_count = len(stmts)
    return {
        "statement_count": statement_count,
        "unique_action_count": len(actions),
        "has_action_wildcard": has_action_wc,
        "has_resource_wildcard": has_resource_wc,
        "principal_type": principal_type,
    }


def _reduce_replacements(replacements: Dict[str, str], limit: int = 3) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not replacements:
        return items
    for idx, (bad, good) in enumerate(replacements.items()):
        if idx >= limit:
            break
        items.append({"from": bad, "to": good})
    return items


def _count_valid_arns(policy: Dict[str, Any]) -> int:
    stmts = policy.get("Statement", [])
    if isinstance(stmts, dict):
        stmts = [stmts]
    count = 0
    for st in stmts:
        if not isinstance(st, dict):
            continue
        res = st.get("Resource")
        if isinstance(res, str) and res.startswith("arn:aws:"):
            count += 1
        elif isinstance(res, list):
            count += sum(1 for r in res if isinstance(r, str) and r.startswith("arn:aws:"))
    return count


def build_trace_digest(
    nl_prompt: str,
    canonical_policy: Dict[str, Any],
    guard_out: Dict[str, Any],
    static_out: Dict[str, Any],
    prior_patches: List[Dict[str, Any]],
    round_no: int,
) -> Dict[str, Any]:
    """Assemble bounded digest payload for LLM roles."""

    policy_summary = summarize_policy(canonical_policy)
    replacements = _reduce_replacements(guard_out.get("replacements") or {})
    hints = (static_out.get("messages") or static_out.get("hints") or [])[:5]

    digest = {
        "nl_prompt": nl_prompt,
        "policy_hash": _sha256_json(canonical_policy),
        "policy_summary": policy_summary,
        "registry_facts": {
            "valid_actions": len(set(guard_out.get("in_registry", []))),
            "valid_arns": _count_valid_arns(canonical_policy),
            "replacements": replacements,
        },
        "static_hints": hints,
        "prior_patches": prior_patches[-5:],
        "round": round_no,
    }
    return digest


