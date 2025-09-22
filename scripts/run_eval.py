#!/usr/bin/env python3
"""
Quick eval harness comparing Vector mode vs Adversarial mode on a small CSV.

CSV format (no header):
query

Writes eval_results.csv with columns:
mode,query,accepted,wildcard_count,actions_count,tightness,risk

Note: This harness runs locally and expects env vars and indexes present for vector mode.
"""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

from src.policy_generator import IAMPolicyGeneratorV2
from src.policy_generator_adv import generate_adversarial


def wildcard_count(policy: Dict[str, Any]) -> int:
    cnt = 0
    for st in (policy or {}).get("Statement", []):
        r = st.get("Resource")
        if r == "*" or (isinstance(r, list) and "*" in r):
            cnt += 1
    return cnt


def actions_count(policy: Dict[str, Any]) -> int:
    s = set()
    for st in (policy or {}).get("Statement", []):
        acts = st.get("Action")
        if isinstance(acts, str):
            s.add(acts)
        elif isinstance(acts, list):
            for a in acts:
                if isinstance(a, str):
                    s.add(a)
    return len(s)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with 'query' column or single-column queries")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            q = r[0].strip()
            if q and q.lower() != "query":
                rows.append(q)

    out_rows = []
    # Vector
    gen = IAMPolicyGeneratorV2()
    for q in rows:
        res = gen.generate_policy(q, save_to_file=False)
        pol = res.get("iam_policy", {}) if res.get("status") == "success" else {}
        out_rows.append(["vector", q, int(res.get("status") == "success"), wildcard_count(pol), actions_count(pol), "", ""])  # scores empty here

    # Adversarial
    for q in rows:
        res = generate_adversarial(q, rounds=1, models={})
        pol = res.get("iam_policy", {}) if res.get("status") == "success" else {}
        md = res.get("metadata", {})
        tight = (md.get("score", {}) or {}).get("tightness", "")
        risk = (md.get("score", {}) or {}).get("risk", "")
        out_rows.append(["adversarial", q, int(res.get("status") == "success"), wildcard_count(pol), actions_count(pol), tight, risk])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "query", "accepted", "wildcard_count", "actions_count", "tightness", "risk"])
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

