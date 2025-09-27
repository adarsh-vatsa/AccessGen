#!/usr/bin/env python3
import os
import json
import warnings
import sys

# Suppress urllib3 warnings on macOS LibreSSL Python builds
warnings.filterwarnings("ignore", module="urllib3")

import requests


def main():
    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try loading from project .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
            api_key = os.getenv("OPENAI_API_KEY")
        except Exception:
            pass
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 2

    model = os.getenv("OPENAI_MODEL", "gpt-5")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Allow passing a custom natural-language task via CLI args
    nl_task = " ".join(sys.argv[1:]).strip() or "Allow listing S3 buckets (s3:ListAllMyBuckets)."

    prompt = (
        "Return only a JSON object with exactly two fields: "
        "'iam_policy' and 'test_config'.\n\n"
        "iam_policy: a valid AWS policy object with keys Version and Statement.\n"
        "test_config: an object with keys 'service' (one of 's3','ec2','iam') and 'rules' (array of objects with keys id, effect, principals, not_principals, actions, resources, conditions).\n"
        "For test_config.rules, set id sequentially as 'R1', 'R2', ... in order.\n\n"
        f"Task: {nl_task} Use Version 2012-10-17. No prose."
    )

    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 8000,
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "json_object"}},
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        print("HTTP ERROR:", resp.status_code, resp.text)
        return 1

    data = resp.json()

    # Prefer output_text if present and try to parse JSON
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        txt = output_text.strip()
        try:
            obj = json.loads(txt)
            print(json.dumps(obj, indent=2))
            return 0
        except Exception:
            print(txt)
            return 0

    # Fallback: assemble text from output blocks
    out = data.get("output") or []
    parts = []
    for blk in out:
        content = (blk or {}).get("content") or []
        for item in content:
            t = (item or {}).get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    txt = "\n".join(parts).strip()
    if txt:
        try:
            obj = json.loads(txt)
            print(json.dumps(obj, indent=2))
            return 0
        except Exception:
            print(txt)
            return 0
    # Print raw JSON if still empty
    print(json.dumps(data, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


