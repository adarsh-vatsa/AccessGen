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
                  checker_patches: List[Dict[str, Any]],
                  trace_context: List[Dict[str, Any]] | None = None) -> DebateResult:
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

        # Heuristic: derive target deltas from NL prompt
        nl_lower = (nl_prompt or "").lower()
        stmt0 = None
        try:
            stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
            if isinstance(stmts, list) and stmts:
                stmt0 = stmts[0]
        except Exception:
            stmt0 = None

        def ensure_actions(actions_needed: List[str]):
            if not stmt0:
                return
            cur = stmt0.get("Action")
            cur_list: List[str] = []
            if isinstance(cur, str):
                cur_list = [cur]
            elif isinstance(cur, list):
                cur_list = [a for a in cur if isinstance(a, str)]
            else:
                cur_list = []
            changed = False
            for act in actions_needed:
                if act not in cur_list:
                    cur_list.append(act)
                    changed = True
            if changed:
                pros.append(f"Include required actions: {', '.join([a for a in actions_needed if a not in (stmt0.get('Action') if isinstance(stmt0.get('Action'), list) else [])])}.")
                patches.append({"op": "replace", "path": "/Statement/0/Action", "value": cur_list})

        def ensure_principal_for_cross_account():
            if not stmt0:
                return
            if "Principal" not in stmt0 and ("another account" in nl_lower or "account" in nl_lower):
                # Add placeholder principal for external account root or role
                pros.append("Add explicit principal for external account to enable cross-account access.")
                patches.append({"op": "add", "path": "/Statement/0/Principal", "value": {"AWS": "arn:aws:iam::${ACCOUNT_A_ID}:root"}})

        def tighten_resource_bucket():
            if not stmt0:
                return
            res = stmt0.get("Resource")
            if res == "*":
                pros.append("Tighten Resource from '*' to specific bucket ARN.")
                patches.append({"op": "replace", "path": "/Statement/0/Resource", "value": "arn:aws:s3:::${BUCKET_NAME}"})

        # Apply heuristics for common S3 bucket policy edits
        if "bucket" in nl_lower or "s3" in nl_lower:
            if ("write" in nl_lower or "modify" in nl_lower or "policy" in nl_lower):
                ensure_actions(["s3:GetBucketPolicy", "s3:PutBucketPolicy"])  # minimal set for bucket policy edits
                ensure_principal_for_cross_account()
                tighten_resource_bucket()

        # Pro case incorporates context and NL goal
        context_note = f"Context length: {len(trace_context or [])} steps." if trace_context is not None else ""
        pros.insert(0, f"Goal: satisfy NL requirement while adhering to registry facts. {context_note}".strip())
        pro_case = ("\n".join([p for p in pros if p]) or "No changes needed; draft aligns with registry facts.")

        # Opponent: push for stricter scope; drop any action not in registry
        cons: List[str] = []
        if guard_replacements:
            for bad in guard_replacements.keys():
                cons.append(f"Remove non-registry action {bad} unless compelling justification.")
        if not checker_patches:
            cons.append("Review for wildcard resources and missing policy allow-lists; add if absent.")
        # Encourage evaluation against NL requirement explicitly
        cons.append("Evaluate whether changes move policy closer to NL requirement, not just registry alignment.")
        con_case = ("\n".join(cons) or "No further objections.")

        # Judge: apply rubric; accept if patches shrink wildcards or add required conditions
        judge_rationale = "Applying patches that improve alignment with NL requirement and rubric (scope/conditions)."
        patched_policy = _apply_patches(policy, patches)
        summary = "Round: Proposed and applied patches toward NL goal; registry used as check."

        return DebateResult(
            summary=summary,
            patches_applied=patches,
            pro_case=pro_case,
            con_case=con_case,
            judge_rationale=judge_rationale,
        )

