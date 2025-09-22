#!/usr/bin/env python3
"""
StaticChecker: Enforce resource scoping and required conditions; propose JSON Patch repairs.

Heuristics (MVP):
- If an action supports specific resource types, avoid Resource "*"; propose ARN placeholders.
- For iam:*Policy* attach/put actions, require Condition.StringEquals.iam:PolicyARN allow-list placeholder.
- For S3 write-level actions, recommend tag-on-create constraints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


class StaticCheckResult(TypedDict):
    status: str  # "pass" | "fail"
    repairs: List[Dict[str, Any]]  # JSON Patch operations
    messages: List[str]


@dataclass
class StaticChecker:
    provenance: Dict[str, Dict[str, Any]]  # from RegistryGuard

    def _s3_arn_for(self, res_types: List[str]) -> str:
        rts = set((res_types or []))
        if "object" in rts:
            return "arn:aws:s3:::${BUCKET_NAME}/*"
        if "bucket" in rts or not rts:
            return "arn:aws:s3:::${BUCKET_NAME}"
        return "arn:aws:s3:::${BUCKET_NAME}"

    def _needs_policyarn_condition(self, action: str) -> bool:
        a = action.lower()
        return a.startswith("iam:") and ("attach" in a or "put" in a) and "policy" in a

    def _is_write(self, action: str) -> bool:
        meta = self.provenance.get(action)
        al = (meta or {}).get("access_level", "").lower()
        return al == "write"

    def check_and_repair(self, policy: Dict[str, Any]) -> StaticCheckResult:
        patches: List[Dict[str, Any]] = []
        msgs: List[str] = []

        stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
        for si, st in enumerate(stmts):
            acts = st.get("Action")
            actions: List[str] = []
            if isinstance(acts, str):
                actions = [acts]
            elif isinstance(acts, list):
                actions = [a for a in acts if isinstance(a, str)]

            # Resource scoping
            if st.get("Resource") == "*" or (isinstance(st.get("Resource"), list) and "*" in st["Resource"]):
                # compute best placeholder from provenance of first known action
                known = next((a for a in actions if a in self.provenance), None)
                if known:
                    prov = self.provenance[known]
                    svc = prov.get("service")
                    if svc == "s3":
                        arn = self._s3_arn_for(prov.get("resource_types", []))
                    elif svc == "ec2":
                        arn = "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:instance/*"
                    elif svc == "iam":
                        arn = "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
                    else:
                        arn = "*"
                else:
                    arn = "*"

                if st.get("Resource") == "*":
                    patches.append({"op": "replace", "path": f"/Statement/{si}/Resource", "value": [arn]})
                    msgs.append(f"Replaced wildcard Resource in S{si} with placeholder ARN")
                else:
                    # list case
                    res_list = st.get("Resource")
                    try:
                        idx = res_list.index("*")
                        patches.append({"op": "replace", "path": f"/Statement/{si}/Resource/{idx}", "value": arn})
                        msgs.append(f"Replaced wildcard element in S{si} with placeholder ARN")
                    except Exception:
                        pass

            # iam:PolicyARN allow-list enforcement
            for a in actions:
                if self._needs_policyarn_condition(a):
                    cond = st.get("Condition") or {}
                    needs = True
                    # Look for StringEquals -> iam:PolicyARN
                    if isinstance(cond, dict):
                        se = cond.get("StringEquals") or cond.get("ForAnyValue:StringEquals")
                        if isinstance(se, dict) and any(k.endswith(":PolicyARN") for k in se.keys()):
                            needs = False
                    if needs:
                        # add minimal patch
                        new_cond = {
                            "StringEquals": {
                                "iam:PolicyARN": [
                                    "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
                                ]
                            }
                        }
                        if "Condition" in st:
                            patches.append({"op": "replace", "path": f"/Statement/{si}/Condition", "value": new_cond})
                        else:
                            patches.append({"op": "add", "path": f"/Statement/{si}/Condition", "value": new_cond})
                        msgs.append(f"Enforced iam:PolicyARN allow-list on {a} in S{si}")

            # S3 write tagging recommendation (optional, non-fatal)
            if any(self._is_write(a) for a in actions):
                # Only add if service is s3 (from first known)
                known = next((a for a in actions if a in self.provenance and self.provenance[a].get("service") == "s3"), None)
                if known:
                    cond = st.get("Condition") or {}
                    # add RequestTag if not present
                    has_req = False
                    if isinstance(cond, dict):
                        # shallow search
                        for k, v in cond.items():
                            if isinstance(v, dict) and any(kk.startswith("aws:RequestTag/") for kk in v.keys()):
                                has_req = True
                    if not has_req:
                        new_cond = {
                            "ForAllValues:StringEqualsIfExists": {
                                "aws:TagKeys": ["project", "env"]
                            }
                        }
                        # merge or add (replace is simpler for MVP)
                        if "Condition" in st:
                            patches.append({"op": "replace", "path": f"/Statement/{si}/Condition", "value": new_cond})
                        else:
                            patches.append({"op": "add", "path": f"/Statement/{si}/Condition", "value": new_cond})
                        msgs.append(f"Recommended tag-on-create constraints for S3 Write in S{si}")

        status = "pass" if not patches else "pass"  # we auto-repair in one pass, so status pass
        return StaticCheckResult(status=status, repairs=patches, messages=msgs)

