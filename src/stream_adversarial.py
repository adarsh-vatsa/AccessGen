#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure imports
sys.path.insert(0, str(Path(__file__).parent))

from src.policy_generator_adv import generate_adversarial


def sse_send(event: str, data: dict):
    sys.stdout.write(f"event: {event}\n")
    sys.stdout.write("data: ")
    sys.stdout.write(json.dumps(data))
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def run_stream():
    query = os.getenv("QUERY") or ""
    rounds = int(os.getenv("ROUNDS") or "1")

    models = {
        "draft": os.getenv("DRAFT_MODEL") or (os.getenv("MODEL") or "models/gemini-2.5-pro"),
        "judge": os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash",
        "pro": os.getenv("PRO_MODEL") or (os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash"),
        "con": os.getenv("CON_MODEL") or (os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash"),
    }
    settings = {"use_dspy": (os.getenv("USE_DSPY", "0") == "1")}

    def cb(step):
        sse_send("trace", step)

    # Start event
    sse_send("start", {"query": query, "rounds": rounds, "models": models})

    res = generate_adversarial(query, rounds=rounds, models=models, settings=settings, trace_cb=cb)
    sse_send("done", res)


if __name__ == "__main__":
    run_stream()
