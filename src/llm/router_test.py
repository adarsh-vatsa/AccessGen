#!/usr/bin/env python3
# Save as: router_test.py (in the same dir as router.py)

import os
import json
from pathlib import Path

# Load .env from the nearest parent that has it (project root, etc.)
try:
    from dotenv import load_dotenv
    def _load_dotenv_upwards(start: Path):
        p = start.resolve()
        for _ in range(6):
            env = p / ".env"
            if env.exists():
                load_dotenv(env)
                return
            if p.parent == p:
                break
            p = p.parent
        load_dotenv()  # fallback to default search
    _load_dotenv_upwards(Path(__file__).parent)
except Exception:
    pass  # if python-dotenv isn't installed, we just rely on the environment

# Import the router that sits right next to this file
from router import ModelRouter, POLICY_SCHEMA, TEST_CONFIG_SCHEMA  # noqa: E402


def main():
    # Model defaults to gpt-5 unless OPENAI_MODEL is set
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY is not set (check your .env or environment).")
        return

    r = ModelRouter()

    # --- 1) Plain text sanity
    try:
        txt = r.generate_text("Say 'pong' exactly once.", provider="openai", model=model, max_tokens=16)
        print("TEXT OUTPUT:", repr(txt))
        assert isinstance(txt, str) and "pong" in txt.lower()
        print("[OK] generate_text(ping)")
    except Exception as e:
        print("[FAIL] generate_text(ping):", e)
        return

    # --- 2) Structured: policy (single JSON object)
    try:
        policy_prompt = (
            "Produce only a valid AWS IAM policy JSON for: list S3 buckets. "
            "Use Version 2012-10-17 and least privilege. No prose."
        )
        policy = r.generate_json(
            policy_prompt,
            provider="openai",
            model=model,
            schema=POLICY_SCHEMA,
            max_tokens=800,
        )
        print("[OK] generate_json(policy)")
        print(json.dumps(policy, indent=2))
        assert isinstance(policy, dict) and "Statement" in policy
    except Exception as e:
        print("[FAIL] generate_json(policy):", e)
        return

    # --- 3) Structured: test_config (separate JSON object)
    try:
        test_prompt = (
            "Produce only a test_config JSON that mirrors the policy statements for listing S3 buckets. "
            "Use service 's3' and default principals ['${PRINCIPAL_PLACEHOLDER}']. No prose."
        )
        test_cfg = r.generate_json(
            test_prompt,
            provider="openai",
            model=model,
            schema=TEST_CONFIG_SCHEMA,
            max_tokens=800,
        )
        print("[OK] generate_json(test_config)")
        print(json.dumps(test_cfg, indent=2))
        assert isinstance(test_cfg, dict) and "rules" in test_cfg
    except Exception as e:
        print("[FAIL] generate_json(test_config):", e)
        return

    print("\nALL GOOD ✅")


if __name__ == "__main__":
    main()
