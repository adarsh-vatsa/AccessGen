#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import sys
from typing import List

from src.llm.router import ModelRouter


def main(argv: List[str]) -> int:
    prompt = os.environ.get("ROUTER_TEST_PROMPT", "Say hello in JSON with key 'ok': {\"ok\": true}")
    routes = [
        ("google", os.environ.get("GOOGLE_MODEL", "models/gemini-2.5-pro")),
        ("anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")),
        ("openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
    ]
    router = ModelRouter()
    results = {}
    for provider, model in routes:
        try:
            ok_env = {
                "google": "GEMINI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }[provider]
            if not os.getenv(ok_env):
                results[provider] = {"skipped": True, "reason": f"missing {ok_env}"}
                continue
            text = router.generate_text(provider=provider, model=model, prompt=prompt, temperature=0.2, max_tokens=256)
            results[provider] = {"skipped": False, "ok": bool(text), "sample": (text[:240] if text else "")}
        except Exception as e:
            results[provider] = {"skipped": False, "ok": False, "error": str(e)}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

