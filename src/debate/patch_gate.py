#!/usr/bin/env python3
"""Light patch gate enforcing registry and wildcard rules."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


class GateRule:
    REGISTRY = "registry_validity"
    PRINCIPAL_STAR = "principal_star"
    ACTION_STAR = "action_star"


def _nl_allows_public(nl_prompt: str) -> bool:
    nl_l = (nl_prompt or "").lower()
    return any(k in nl_l for k in ["public", "anyone", "anonymous", "all users", "internet"])


def _nl_allows_all_actions(nl_prompt: str) -> bool:
    nl_l = (nl_prompt or "").lower()
    return any(k in nl_l for k in ["all actions", "full access", "admin access", "administrator"])


def light_patch_gate(
    nl_prompt: str,
    patches: List[Dict[str, Any]],
    registry_validate_fn: Callable[[Dict[str, Any]], bool],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for patch in patches:
        # 1) Registry validity or malformed actions/ARNs
        try:
            if not registry_validate_fn(patch):
                rejected.append({"patch": patch, "rule": GateRule.REGISTRY, "reason": "unknown action or invalid ARN"})
                continue
        except Exception:
            rejected.append({"patch": patch, "rule": GateRule.REGISTRY, "reason": "registry validation error"})
            continue

        path = str(patch.get("path", ""))
        value = patch.get("value")

        # 2) Principal:"*" guard
        if path.endswith("/Principal") and value == "*":
            if not _nl_allows_public(nl_prompt):
                rejected.append({"patch": patch, "rule": GateRule.PRINCIPAL_STAR, "reason": "Principal:* without explicit public intent"})
                continue

        # 3) Action:"*" guard
        if path.endswith("/Action") and value == "*":
            if not _nl_allows_all_actions(nl_prompt):
                rejected.append({"patch": patch, "rule": GateRule.ACTION_STAR, "reason": "Action:* requires explicit 'all actions' intent"})
                continue

        accepted.append(patch)

    return accepted, rejected


