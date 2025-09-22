#!/usr/bin/env python3
"""
Lightweight DSPy-like triad debate orchestrator (stub, no external deps).

Roles:
- Proponent: Argues for minimal, registry-backed policy and suggests patches.
- Opponent: Argues for stricter scoping/conditions and removal of unknowns.
- Judge: Applies rubric to accept or request patch (JSON Patch), one round at a time.

This MVP relies on RegistryGuard outcomes and StaticChecker suggestions to craft
simple heuristics for pros/cons and patch sets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


class DebateResult(TypedDict):
    summary: str
    patches_applied: List[Dict[str, Any]]
    pro_case: str
    con_case: str
    judge_rationale: str


RUBRIC_TEXT = (
    "Tight resource scoping, avoid wildcard where ARN exists; recommended conditions (MFA/network/tags) applied when applicable; "
    "permissions management requires iam:PolicyARN allow-list; each action should be in registry and cite provenance."
)


def _apply_patches(obj: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Minimal JSON Patch support for add/replace at simple paths
    res = json.loads(json.dumps(obj))
    for p in patches:
        op = p.get("op")
        path = p.get("path", "")
        parts = [pp for pp in path.split("/") if pp != ""]
        cur = res
        for i, key in enumerate(parts):
            last = i == len(parts) - 1
            if not last:
                # index or key
                if key.isdigit():
                    idx = int(key)
                    cur = cur[idx]
                else:
                    cur = cur.setdefault(key, {})
            else:
                if op == "replace" or op == "add":
                    val = p.get("value")
                    if key.isdigit():
                        cur[int(key)] = val
                    else:
                        cur[key] = val
                # ignore remove in MVP
    return res


@dataclass
class DebateOrchestrator:
    draft_model: str
    judge_model: str
    pro_model: str
    con_model: str

    def run_round(self,
                  nl_prompt: str,
                  policy: Dict[str, Any],
                  registry_facts: str,
                  guard_replacements: Dict[str, str],
                  checker_patches: List[Dict[str, Any]]) -> DebateResult:
        # Proponent: accept canonicalized actions, propose applying checker patches, replace unknowns
        pros: List[str] = []
        patches: List[Dict[str, Any]] = []
        if guard_replacements:
            for bad, good in guard_replacements.items():
                pros.append(f"Replace unknown action {bad} with {good} from registry.")
                # No direct patch path to Action arrays in MVP; rely on generator rerun or manual edit
        if checker_patches:
            pros.append("Apply static checker patches to tighten resources/conditions.")
            patches.extend(checker_patches)

        pro_case = ("\n".join(pros) or "No changes needed; draft aligns with registry facts.")

        # Opponent: push for stricter scope; drop any action not in registry
        cons: List[str] = []
        if guard_replacements:
            for bad in guard_replacements.keys():
                cons.append(f"Remove non-registry action {bad} unless compelling justification.")
        if not checker_patches:
            cons.append("Review for wildcard resources and missing policy allow-lists; add if absent.")
        con_case = ("\n".join(cons) or "No further objections.")

        # Judge: apply rubric; accept if patches shrink wildcards or add required conditions
        judge_rationale = "Applying patches that reduce wildcard scope and add allow-lists per rubric."
        patched_policy = _apply_patches(policy, patches)
        summary = "Round: Applied static checker patches; replacements noted for unknown actions."

        return DebateResult(
            summary=summary,
            patches_applied=patches,
            pro_case=pro_case,
            con_case=con_case,
            judge_rationale=judge_rationale,
        )

