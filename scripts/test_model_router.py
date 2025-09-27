#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm.openai_router import generate_openai_text, generate_policy_bundle


def main(argv: List[str]) -> int:
    prompt = os.environ.get("ROUTER_TEST_PROMPT", "Say hello in JSON with key 'ok': {\"ok\": true}")
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    if not os.getenv("OPENAI_API_KEY"):
        print(json.dumps({"openai": {"skipped": True, "reason": "missing OPENAI_API_KEY"}}, indent=2))
        return 0

    results = {"openai": {"skipped": False}}
    try:
        text = generate_openai_text(prompt, model=model, max_output_tokens=256)
        results["openai"]["text_ok"] = bool(text)
        results["openai"]["sample"] = (text[:240] if text else "")
    except Exception as e:
        results["openai"].update({"text_ok": False, "error": str(e)})

    try:
        bundle = generate_policy_bundle("Allow listing S3 buckets (s3:ListAllMyBuckets).", model=model)
        results["openai"]["bundle_ok"] = (
            isinstance(bundle, dict)
            and isinstance(bundle.get("iam_policy"), dict)
            and isinstance(bundle.get("test_config"), dict)
        )
    except Exception as e:
        results["openai"].update({"bundle_ok": False, "bundle_error": str(e)})

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

