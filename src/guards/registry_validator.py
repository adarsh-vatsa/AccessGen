#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
import logging
from typing import Any, Dict, List, Tuple


def _load_enriched(path: Path) -> Dict[str, Any]:
    if not path.exists():
        try:
            logging.getLogger(__name__).warning(f"Enriched registry not found: {path}")
        except Exception:
            pass
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RegistryValidator:
    """Validate policy condition keys and resource ARNs against registry facts.

    Builds maps per service:
      - service_condition_keys: List[str]
      - resource_type_templates: {resource_type: arn_template}
    """

    def __init__(self,
                 s3_path: str | None = None,
                 ec2_path: str | None = None,
                 iam_path: str | None = None):
        root = Path(__file__).parent.parent
        if not s3_path:
            s3_path = str(root / ".." / "enriched_data" / "aws_iam_registry_s3_enriched_extras.json")
        if not ec2_path:
            ec2_path = str(root / ".." / "enriched_data" / "aws_iam_registry_ec2_enriched_extras.json")
        if not iam_path:
            iam_path = str(root / ".." / "enriched_data" / "aws_iam_registry_iam_enriched_extras.json")

        self.service_condition_keys: Dict[str, List[str]] = {}
        self.resource_type_templates: Dict[str, Dict[str, str]] = {}

        def load(service: str, p: str):
            data = _load_enriched(Path(p))
            if not data:
                return
            svc = data.get(service) or {}
            self.service_condition_keys[service] = list(svc.get("service_condition_keys", []))
            templates: Dict[str, str] = {}
            for rt in svc.get("resource_types", []) or []:
                rname = rt.get("type")
                arn = rt.get("arn_template")
                if rname and arn:
                    templates[rname] = arn
            self.resource_type_templates[service] = templates

        load("s3", s3_path)
        load("ec2", ec2_path)
        load("iam", iam_path)

    @staticmethod
    def _flatten_condition_keys(cond: Any) -> List[str]:
        keys: List[str] = []
        if not isinstance(cond, dict):
            return keys
        for k, v in cond.items():
            # k is an operator or a condition key group, e.g., StringEquals
            if isinstance(v, dict):
                for ck in v.keys():
                    if isinstance(ck, str):
                        keys.append(ck)
            elif isinstance(v, list):
                # unlikely shape
                pass
            # nested shapes are rare; shallow is enough
        return keys

    @staticmethod
    def _template_to_regex(tmpl: str) -> re.Pattern[str]:
        # Replace ${...} and {{...}} with permissive segments; keep ':' and '/'
        s = re.escape(tmpl)
        # Replace escaped placeholders \$\{...\} and \{\{...\}\}
        s = re.sub(r"\\\$\\\{[^}]+\\\}", r"[^:*]+", s)
        s = re.sub(r"\\\{\\\{[^}]+\\\}\\\}", r"[^:*]+", s)
        # Convert wildcard '*' if any
        s = s.replace("\\*", ".*")
        # Allow placeholders in S3 empty region forms like arn:aws:s3:::
        return re.compile(f"^{s}$")

    def _resource_matches(self, resource: str, service: str, resource_types: List[str]) -> bool:
        if resource == "*":
            return False
        rt_map = self.resource_type_templates.get(service) or {}
        for rt in resource_types or []:
            tmpl = rt_map.get(rt)
            if not tmpl:
                continue
            pat = self._template_to_regex(tmpl)
            # Allow placeholders in provided resource, e.g., ${ACCOUNT_ID}
            res_norm = resource
            # quick accept if placeholders present; skip strict match
            if ("${" in res_norm) or ("{{" in res_norm):
                return True
            if pat.match(res_norm):
                return True
        return False

    def validate_policy(self, policy: Dict[str, Any], provenance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
        for i, st in enumerate(stmts):
            # Actions
            acts = st.get("Action")
            actions: List[str] = []
            if isinstance(acts, str):
                actions = [acts]
            elif isinstance(acts, list):
                actions = [a for a in acts if isinstance(a, str)]
            # Service from first action
            svc = actions[0].split(":", 1)[0] if actions and ":" in actions[0] else None
            # Allowed condition keys
            allowed_keys = set()
            for a in actions:
                prov = provenance.get(a) or {}
                for k in prov.get("condition_keys", []) or []:
                    allowed_keys.add(k)
            if svc:
                for k in self.service_condition_keys.get(svc, []):
                    allowed_keys.add(k)
            # Extract used condition keys
            used_keys = self._flatten_condition_keys(st.get("Condition", {}))
            for ck in used_keys:
                if ck.startswith("aws:"):
                    continue
                if ck not in allowed_keys:
                    errors.append({"statement": i, "type": "condition_key", "key": ck, "service": svc})

            # Resources
            res = st.get("Resource")
            resources: List[str] = []
            if isinstance(res, str):
                resources = [res]
            elif isinstance(res, list):
                resources = [r for r in res if isinstance(r, str)]
            all_rts = set()
            for a in actions:
                prov = provenance.get(a) or {}
                for rt in prov.get("resource_types", []) or []:
                    all_rts.add(rt)
            if svc and resources:
                for r in resources:
                    if r == "*":
                        warnings.append({"statement": i, "type": "resource_wildcard", "resource": r})
                        continue
                    if all_rts and not self._resource_matches(r, svc, list(all_rts)):
                        errors.append({
                            "statement": i,
                            "type": "resource_mismatch",
                            "resource": r,
                            "service": svc,
                            "expected_types": sorted(list(all_rts)),
                        })

        return {"errors": errors, "warnings": warnings}

    @staticmethod
    def validate_test_config(config: Dict[str, Any]) -> Dict[str, Any]:
        warnings: List[str] = []
        errors: List[str] = []
        if not isinstance(config, dict):
            errors.append("test_config is not a dict")
            return {"errors": errors, "warnings": warnings}
        for i, rule in enumerate(config.get("rules", [])):
            principals = rule.get("principals", [])
            if not isinstance(principals, list):
                errors.append(f"rule {i}: principals must be a list")
                continue
            for p in principals:
                if not isinstance(p, str):
                    errors.append(f"rule {i}: principal must be a string")
                    continue
                if p == "*":
                    continue
                if p.endswith(".amazonaws.com"):
                    # service principal
                    continue
                if p.startswith("arn:aws:iam::"):
                    # basic IAM principal shape
                    continue
                warnings.append(f"rule {i}: unusual principal format: {p}")
        return {"errors": errors, "warnings": warnings}
