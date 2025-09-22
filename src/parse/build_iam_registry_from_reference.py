from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Any, Dict
import json

from src.fetch.iam_reference import fetch_iam_reference_json, IAM_REF_URL

# --- Data contracts (same structure as S3) ---

@dataclass
class SourceRef:
    url: str
    table: str     # "actions" | "resource_types" | "condition_keys"
    row_index: int

@dataclass
class ResourceTypeRef:
    type: str
    required: bool   # NOTE: Service Reference JSON does NOT expose "required". We set False for MVP.
    source: SourceRef

@dataclass
class ActionRow:
    action: str
    access_level: str                 # Mapped from Annotations.Properties
    resource_types: List[ResourceTypeRef]
    condition_keys: List[str]         # Action-scoped condition keys
    dependent_actions: List[str]      # Not present in Service Reference JSON => []
    source: SourceRef

@dataclass
class ResourceTypeDef:
    type: str
    arn_template: str                 # First ARNFormats entry for simplicity
    condition_keys: List[str]         # Keys listed under the resource definition (not action-level)
    source: SourceRef

@dataclass
class IAMRegistry:
    service_name: str                 # "AWS Identity and Access Management"
    service_prefix: str               # "iam"
    page_url: str                     # IAM_REF_URL
    version: str                      # e.g., "v1.3" from Service Reference JSON
    actions: List[ActionRow]
    resource_types: List[ResourceTypeDef]
    service_condition_keys: List[str] # Top-level ConditionKeys[].Name

def to_jsonable(x: Any) -> Any:
    if hasattr(x, "__dict__"): return asdict(x)
    if isinstance(x, list): return [to_jsonable(i) for i in x]
    return x

# --- Helpers ---

def _access_level_from_properties(props: Dict[str, bool]) -> str:
    """
    Service Reference adds boolean properties (not mutually exclusive):
      IsList, IsPermissionManagement, IsTaggingOnly, IsWrite.
    SAR historically categorized into one Access level. We pick a stable precedence:
      Permissions management > Write > Tagging > List > Read
    If none are true => "Read".
    """
    if props.get("IsPermissionManagement"): return "Permissions management"
    if props.get("IsWrite"):                return "Write"
    if props.get("IsTaggingOnly"):          return "Tagging"
    if props.get("IsList"):                 return "List"
    return "Read"

def _first_or_empty(arr: List[str]) -> str:
    return arr[0] if arr else ""

# --- Builder ---

OUT = Path("data/aws_iam_registry_iam.json")
REPORT = Path("data/build_reports/iam_registry_report.json")
REPORT.parent.mkdir(parents=True, exist_ok=True)

def build_iam_registry_from_reference(force: bool=False) -> dict:
    ref = fetch_iam_reference_json(force=force)

    # Top-level fields
    version = ref.get("Version", "")
    service_prefix = "iam"
    service_name = "AWS Identity and Access Management"

    # Resource definitions (top-level "Resources")
    resource_defs: List[ResourceTypeDef] = []
    for i, rdef in enumerate(ref.get("Resources", [])):
        rname = rdef.get("Name", "")
        arn_formats = rdef.get("ARNFormats", []) or []
        arn_template = _first_or_empty(arn_formats)
        resource_defs.append(ResourceTypeDef(
            type=rname,
            arn_template=arn_template,
            condition_keys=[],
            source=SourceRef(url=IAM_REF_URL, table="resource_types", row_index=i)
        ))

    # Actions
    actions: List[ActionRow] = []
    for i, act in enumerate(ref.get("Actions", [])):
        aname = act.get("Name", "")
        action_keys = list(act.get("ActionConditionKeys", []) or [])
        props = (((act.get("Annotations") or {}).get("Properties")) or {})
        access = _access_level_from_properties(props)

        # Resource types for this action
        rtypes = []
        for r in act.get("Resources", []) or []:
            rname = r.get("Name", "")
            if not rname: 
                continue
            rtypes.append(ResourceTypeRef(
                type=rname,
                required=False,  # Service Reference JSON does not expose the SAR "required *" bit
                source=SourceRef(url=IAM_REF_URL, table="actions", row_index=i)
            ))

        actions.append(ActionRow(
            action=aname,
            access_level=access,
            resource_types=rtypes,
            condition_keys=action_keys,
            dependent_actions=[],  # Not provided by Service Reference JSON
            source=SourceRef(url=IAM_REF_URL, table="actions", row_index=i)
        ))

    # Service-level condition keys
    service_keys = [ck.get("Name", "") for ck in ref.get("ConditionKeys", []) if ck.get("Name")]

    # Deterministic ordering
    actions.sort(key=lambda a: a.action.lower())
    resource_defs.sort(key=lambda r: r.type.lower())
    service_keys.sort()

    reg = IAMRegistry(
        service_name=service_name,
        service_prefix=service_prefix,
        page_url=IAM_REF_URL,
        version=version,
        actions=actions,
        resource_types=resource_defs,
        service_condition_keys=service_keys
    )

    OUT.write_text(json.dumps({"iam": to_jsonable(reg)}, indent=2), encoding="utf-8")
    report = {
        "service": "iam",
        "source": "aws-service-reference",
        "version": version,
        "actions_count": len(actions),
        "resource_types_count": len(resource_defs),
        "condition_keys_count": len(service_keys),
        "saved_to": str(OUT)
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rep = build_iam_registry_from_reference(force=args.force)
    print(json.dumps(rep, indent=2))