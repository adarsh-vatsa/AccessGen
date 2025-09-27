#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from src.llm.router import ModelRouter


def main() -> int:
    # Load .env if present
    try:
        load_dotenv()
    except Exception:
        pass
    # Allow overriding default models via env
    default_models: Dict[str, str] = {
        "google": os.getenv("GOOGLE_MODEL", "models/gemini-2.5-pro"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }

    router = ModelRouter()
    report: Dict[str, Any] = {"env": {}, "canary": {}, "json_mode": {}}

    # Record which keys are present
    report["env"]["GEMINI_API_KEY"] = bool(os.getenv("GEMINI_API_KEY"))
    report["env"]["ANTHROPIC_API_KEY"] = bool(os.getenv("ANTHROPIC_API_KEY"))
    report["env"]["OPENAI_API_KEY"] = bool(os.getenv("OPENAI_API_KEY"))

    # Provider canaries (strict, no fallbacks inside router)
    report["canary"] = router.validate()

    # OpenAI JSON-mode canary with strict schema (only if key present)
    if os.getenv("OPENAI_API_KEY"):
        schema = {
            "title": "router_canary",
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        try:
            out = router.generate_json(
                provider="openai",
                model=default_models["openai"],
                prompt="Return JSON only: {\"ok\": true}",
                schema=schema,
                temperature=0.0,
                max_tokens=16,
            )
            report["json_mode"]["openai"] = {"ok": True, "result": out}
        except Exception as e:
            report["json_mode"]["openai"] = {"ok": False, "error": str(e)}

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
