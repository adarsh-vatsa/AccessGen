#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.policy_generator import IAMPolicyGeneratorV2

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None  # type: ignore
    types = None  # type: ignore
import os as _os
_FORCE_LEGACY = (_os.getenv("GENAI_FORCE_LEGACY", "0") == "1")
try:
    import google.generativeai as legacy_genai  # type: ignore
except Exception:
    legacy_genai = None  # type: ignore


def sse_send(event: str, data: dict):
    sys.stdout.write(f"event: {event}\n")
    sys.stdout.write("data: ")
    sys.stdout.write(json.dumps(data))
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def run_stream_raw():
    query = os.getenv("QUERY") or ""
    model = os.getenv("MODEL") or "models/gemini-2.5-pro"
    if model and not model.startswith("models/") and model.lower().startswith("gemini"):
        model = f"models/{model}"

    gen = IAMPolicyGeneratorV2(
        model=model,
        use_vector_search=False,
        use_query_expansion=False,
    )

    # Build short policy-only prompt for reliability
    full_prompt = gen._build_raw_policy_only_prompt(query)  # type: ignore

    sse_send("start", {"query": query, "model": model})

    # Call model directly (mirror generator logic)
    output_text = ""
    try:
        # Choose client safely based on availability and force flag
        if _FORCE_LEGACY and legacy_genai is not None:
            legacy_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gm = legacy_genai.GenerativeModel(model)
            resp = gm.generate_content(full_prompt)
            output_text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        elif genai is not None and types is not None:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
            resp = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=4000),
            )
            output_text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        elif legacy_genai is not None:
            legacy_genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gm = legacy_genai.GenerativeModel(model)
            resp = gm.generate_content(full_prompt)
            output_text = IAMPolicyGeneratorV2._extract_text_from_response(resp)  # type: ignore
        else:
            raise RuntimeError("No usable Gemini client: install google-generativeai or google-genai")
    except Exception as e:
        sse_send("done", {"status": "error", "message": f"Model call failed: {e}", "raw_output": ""})
        return

    sse_send("trace", {"stage": "draft", "input": {"system": "raw_policy_only", "nl_prompt": query}, "output_text": output_text})

    # Parse best-effort to policy-only
    parsed = IAMPolicyGeneratorV2._extract_json_object(output_text)  # type: ignore
    if not parsed or not isinstance(parsed, dict):
        sse_send("done", {
            "status": "error",
            "message": "Generated output is not valid JSON",
            "raw_output": output_text,
        })
        return

    if not ("Version" in parsed and "Statement" in parsed):
        sse_send("done", {
            "status": "error",
            "message": "RAW stream: model did not return a policy object",
            "raw_output": output_text,
        })
        return
    iam_policy = parsed
    test_config = IAMPolicyGeneratorV2._policy_to_test_config(iam_policy)  # type: ignore

    res = {
        "status": "success",
        "query": query,
        "iam_policy": iam_policy,
        "test_config": test_config,
        "metadata": {"mode": "raw", "model": model}
    }
    sse_send("done", res)


if __name__ == "__main__":
    run_stream_raw()
