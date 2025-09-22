#!/usr/bin/env python3
import json, sys, pathlib

p = pathlib.Path("data/aws_iam_registry_s3.json")
j = json.loads(p.read_text())

s3 = j.get("s3", {})
ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    ok = ok and cond

# 1) Top-level counts & version
check("version present", isinstance(s3.get("version"), str) and len(s3["version"]) > 0, s3.get("version",""))
check("actions > 50", len(s3.get("actions",[])) > 50, str(len(s3.get("actions",[]))))
check("resource_types ≥ 2", len(s3.get("resource_types",[])) >= 2, str(len(s3.get("resource_types",[]))))
check("service_condition_keys > 10", len(s3.get("service_condition_keys",[])) > 10, str(len(s3.get("service_condition_keys",[]))))

# 2) PutObject sanity
put = next((a for a in s3.get("actions",[]) if a.get("action")=="PutObject"), None)
check("PutObject exists", put is not None)
if put:
    check("PutObject access_level == 'Write'", put.get("access_level")=="Write", put.get("access_level"))
    rtypes = [r.get("type") for r in put.get("resource_types",[])]
    check("PutObject has object resource", "object" in rtypes, str(rtypes))
    ckeys = put.get("condition_keys",[])
    check("PutObject has ≥1 condition key", isinstance(ckeys, list) and len(ckeys)>0, f"{ckeys[:6]}")
    deps = put.get("dependent_actions",[])
    check("dependent_actions present ([] for MVP)", isinstance(deps, list), str(deps))
    src = put.get("source",{})
    check("PutObject provenance.url set", isinstance(src.get("url"), str) and "servicereference" in src["url"], src.get("url",""))
    check("PutObject provenance.row_index int", isinstance(src.get("row_index"), int), str(src.get("row_index")))

# 3) Bucket ARN template
bucket = next((r for r in s3.get("resource_types",[]) if r.get("type")=="bucket"), None)
check("bucket resource exists", bucket is not None)
if bucket:
    arn = bucket.get("arn_template","")
    check("bucket ARN template non-empty", isinstance(arn,str) and len(arn)>0, arn)
    check("bucket ARN starts with arn:${Partition}:s3:::", isinstance(arn,str) and arn.startswith("arn:${Partition}:s3:::"), arn)

print("\nRESULT:", "✅ ALL CHECKS PASSED" if ok else "❌ SOME CHECKS FAILED")
