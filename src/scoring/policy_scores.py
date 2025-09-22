#!/usr/bin/env python3
"""
Simple scoring heuristics for final policy:
- tightness: penalize wildcard resources, reward specific ARNs
- risk: raise with permissions management and broad resources
"""
from __future__ import annotations

from typing import Any, Dict, List


def compute_policy_scores(policy: Dict[str, Any]) -> Dict[str, float]:
    stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
    if not isinstance(stmts, list):
        return {"tightness": 0.0, "risk": 1.0}
    total = max(1, len(stmts))
    wildcard = 0
    perm_mgmt = 0
    for st in stmts:
        res = st.get("Resource")
        if res == "*" or (isinstance(res, list) and "*" in res):
            wildcard += 1
        acts = st.get("Action")
        acts_list: List[str] = []
        if isinstance(acts, str):
            acts_list = [acts]
        elif isinstance(acts, list):
            acts_list = [a for a in acts if isinstance(a, str)]
        for a in acts_list:
            if a.lower().startswith("iam:") and ("attach" in a.lower() or "put" in a.lower()) and "policy" in a.lower():
                perm_mgmt += 1
    tight = max(0.0, 1.0 - (wildcard / total))
    risk = min(1.0, 0.05 + 0.4 * (wildcard / total) + 0.2 * (perm_mgmt / total))
    return {"tightness": round(tight, 3), "risk": round(risk, 3)}

