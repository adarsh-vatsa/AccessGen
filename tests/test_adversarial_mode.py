#!/usr/bin/env python3
import os
import json
from pathlib import Path

import pytest

# Allow imports from repo root
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.policy_generator_adv import generate_adversarial
from src.guards.registry_guard import RegistryGuard
from src.guards.static_checker import StaticChecker


@pytest.fixture(autouse=True)
def maybe_offline(monkeypatch):
    """If ADVERSARIAL_OFFLINE=1, force offline; otherwise use real key if present."""
    if os.getenv("ADVERSARIAL_OFFLINE") == "1":
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_adversarial_basic_patch_and_scores():
    models = {
        "draft": "models/gemini-2.5-pro",
        "judge": "models/gemini-2.5-pro",
        "pro": "models/gemini-2.5-pro",
        "con": "models/gemini-2.5-pro",
    }

    res = generate_adversarial(
        nl_prompt="upload files to S3 and read them back",
        rounds=1,
        models=models,
        settings={}
    )

    assert res.get("status") == "success"
    assert res.get("iam_policy") and res.get("test_config")
    meta = res.get("metadata", {})
    assert meta.get("mode") == "adversarial"
    # Metadata shape
    assert isinstance(meta.get("rounds"), int)
    assert isinstance(meta.get("models", {}), dict)
    assert isinstance(meta.get("registry", {}), dict)
    assert isinstance(meta.get("debate", {}), dict)
    assert isinstance(meta.get("static_checker", {}), dict)

    # Policy should not have wildcard Resource after static repair
    policy = res["iam_policy"]
    stmts = policy.get("Statement", [])
    assert isinstance(stmts, list) and len(stmts) >= 1
    for st in stmts:
        r = st.get("Resource")
        assert not (r == "*" or (isinstance(r, list) and "*" in r)), "Wildcard Resource should be repaired"

    # Scores should reflect tightened resources
    score = meta.get("score", {})
    assert score.get("tightness", 0.0) >= 0.5
    assert score.get("risk", 1.0) <= 0.5


def test_adversarial_with_dspy_flag(monkeypatch):
    # Even if DSPy isn't installed, this path should succeed via fallback
    models = {
        "draft": "models/gemini-2.5-pro",
        "judge": "models/gemini-2.5-pro",
        "pro": "models/gemini-2.5-pro",
        "con": "models/gemini-2.5-pro",
    }
    res = generate_adversarial(
        nl_prompt="upload files to S3 and read them back",
        rounds=1,
        models=models,
        settings={"use_dspy": True}
    )
    assert res.get("status") == "success"
    assert res.get("metadata", {}).get("mode") == "adversarial"


def test_registry_guard_suggests_replacement():
    # Create a policy with a near-miss action name
    bad_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:PutObjects"], "Resource": "*"}
        ],
    }
    guard = RegistryGuard()
    res = guard.guard(bad_policy)
    assert "s3:PutObjects" in res["out_of_registry"]
    # Expect a suggested replacement with Levenshtein <= 2
    assert res["replacements"].get("s3:PutObjects") in {"s3:PutObject"}


def test_static_checker_enforces_policyarn():
    # Simulate an attach policy action missing allow-list
    pol = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["iam:AttachRolePolicy"], "Resource": "*"}
        ],
    }
    guard = RegistryGuard()
    gres = guard.guard(pol)
    checker = StaticChecker(provenance=gres["provenance"])
    cres = checker.check_and_repair(gres["policy_canonical"])
    # Should propose a Condition patch adding iam:PolicyARN allow-list
    assert any(
        p.get("path", "").endswith("/Condition") and p.get("op") in ("add", "replace") and "iam:PolicyARN" in json.dumps(p.get("value", {}))
        for p in cres["repairs"]
    ), "Expected iam:PolicyARN allow-list enforcement patch"
