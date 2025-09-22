#!/usr/bin/env python3
"""
RegistryGuard: Canonicalize IAM actions against local registries, detect mismatches,
suggest near-miss replacements, and attach provenance and facts for debate.

This module does not perform network calls and relies on local enriched registries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict


class RegistryGuardResult(TypedDict):
    in_registry: List[str]
    out_of_registry: List[str]
    replacements: Dict[str, str]
    provenance: Dict[str, Dict[str, Any]]
    facts_text: str
    policy_canonical: Dict[str, Any]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[lb]


@dataclass
class _ActionRow:
    service: str
    action: str
    access_level: str
    description: str
    resource_types: List[str]
    condition_keys: List[str]
    source: Dict[str, Any]


class RegistryGuard:
    """Loads local actions registry and provides canonicalization and provenance."""

    def __init__(self,
                 s3_path: str | None = None,
                 ec2_path: str | None = None,
                 iam_path: str | None = None) -> None:
        # Default paths relative to repo
        root = Path(__file__).parent.parent
        if not s3_path:
            s3_path = str(root / ".." / "enriched_data" / "aws_iam_registry_s3_enriched_extras.json")
        if not ec2_path:
            ec2_path = str(root / ".." / "enriched_data" / "aws_iam_registry_ec2_enriched_extras.json")
        if not iam_path:
            iam_path = str(root / ".." / "enriched_data" / "aws_iam_registry_iam_enriched_extras.json")

        self._rows: Dict[str, _ActionRow] = {}
        self._by_service: Dict[str, List[str]] = {"s3": [], "ec2": [], "iam": []}

        def load_service(service: str, p: str):
            path = Path(p)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a in data[service]["actions"]:
                key = f"{service}:{a['action']}"
                row = _ActionRow(
                    service=service,
                    action=a["action"],
                    access_level=a.get("access_level", ""),
                    description=a.get("description", ""),
                    resource_types=[rt.get("type") for rt in a.get("resource_types", []) if rt.get("type")],
                    condition_keys=a.get("condition_keys", []),
                    source=a.get("source", {}),
                )
                self._rows[key] = row
                self._by_service[service].append(a["action"])  # name only for distance search

        load_service("s3", s3_path)
        load_service("ec2", ec2_path)
        load_service("iam", iam_path)

    def _canonicalize_action(self, raw: str) -> Tuple[str | None, str | None]:
        """Return (canonical_service_action, suggested_replacement or None)."""
        if not isinstance(raw, str):
            return None, None
        token = raw.strip().replace(" ", "").replace("_", "")
        if ":" in token:
            service, name = token.split(":", 1)
            service = service.lower().strip()
        else:
            # default to s3 if no prefix - but mark unknown until matched
            service, name = "s3", token
        # try exact match respecting case on action name
        exact_key = None
        for act in self._by_service.get(service, []):
            if act.lower() == name.lower():
                exact_key = f"{service}:{act}"
                break
        if exact_key:
            return exact_key, None
        # near-miss within same service
        best, best_d = None, 999
        for act in self._by_service.get(service, []):
            d = _levenshtein(name.lower(), act.lower())
            if d < best_d:
                best, best_d = act, d
        if best is not None and best_d <= 2:
            return None, f"{service}:{best}"
        return None, None

    def _iter_policy_actions(self, policy: Dict[str, Any]) -> List[Tuple[int, str]]:
        out: List[Tuple[int, str]] = []
        stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
        for i, st in enumerate(stmts):
            actions = st.get("Action")
            if isinstance(actions, str):
                out.append((i, actions))
            elif isinstance(actions, list):
                for a in actions:
                    if isinstance(a, str):
                        out.append((i, a))
        return out

    def _facts_blob(self, keys: List[str]) -> str:
        lines: List[str] = []
        for k in keys:
            row = self._rows.get(k)
            if not row:
                continue
            prov = row.source or {}
            rtypes = ", ".join(row.resource_types)
            keys_s = ", ".join(row.condition_keys[:5]) + ("..." if len(row.condition_keys) > 5 else "")
            lines.append(f"- {k} — {row.description} | access={row.access_level} | resources=[{rtypes}] | keys=[{keys_s}] | source={prov}")
        return "\n".join(lines)

    def guard(self, policy: Dict[str, Any]) -> RegistryGuardResult:
        in_reg: List[str] = []
        out_reg: List[str] = []
        replacements: Dict[str, str] = {}
        provenance: Dict[str, Dict[str, Any]] = {}

        # Build canonical version of policy (only action strings updated)
        policy_canonical = json.loads(json.dumps(policy))  # deep copy
        stmts = policy_canonical.get("Statement", []) if isinstance(policy_canonical, dict) else []
        for si, st in enumerate(stmts):
            actions = st.get("Action")
            if isinstance(actions, str):
                can, rep = self._canonicalize_action(actions)
                if can:
                    in_reg.append(can)
                    stmts[si]["Action"] = can
                    row = self._rows.get(can)
                    if row:
                        provenance[can] = {
                            "service": row.service,
                            "description": row.description,
                            "access_level": row.access_level,
                            "resource_types": row.resource_types,
                            "condition_keys": row.condition_keys,
                            "source": row.source,
                        }
                else:
                    out_reg.append(actions)
                    if rep:
                        replacements[actions] = rep
            elif isinstance(actions, list):
                new_list: List[str] = []
                for a in actions:
                    if not isinstance(a, str):
                        continue
                    can, rep = self._canonicalize_action(a)
                    if can:
                        in_reg.append(can)
                        new_list.append(can)
                        row = self._rows.get(can)
                        if row:
                            provenance[can] = {
                                "service": row.service,
                                "description": row.description,
                                "access_level": row.access_level,
                                "resource_types": row.resource_types,
                                "condition_keys": row.condition_keys,
                                "source": row.source,
                            }
                    else:
                        out_reg.append(a)
                        new_list.append(a)
                        if rep:
                            replacements[a] = rep
                stmts[si]["Action"] = new_list

        facts = self._facts_blob(sorted(set(in_reg)))
        return RegistryGuardResult(
            in_registry=sorted(set(in_reg)),
            out_of_registry=sorted(set(out_reg)),
            replacements=replacements,
            provenance=provenance,
            facts_text=facts,
            policy_canonical=policy_canonical,
        )
