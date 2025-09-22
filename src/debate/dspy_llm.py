#!/usr/bin/env python3
"""
Optional DSPy-based triad debate orchestrator.

If DSPy is not installed, importing this module is safe and a runtime check
will fall back to the heuristic orchestrator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

try:
    import dspy  # type: ignore
    from dspy import Signature, Module, Predict, InputField, OutputField  # type: ignore
except Exception:  # pragma: no cover
    dspy = None
    Signature = object  # type: ignore
    Module = object  # type: ignore
    Predict = None  # type: ignore
    InputField = OutputField = None  # type: ignore


class DebateResult(TypedDict):
    summary: str
    patches_applied: List[Dict[str, Any]]
    pro_case: str
    con_case: str
    judge_rationale: str


RUBRIC_TEXT = (
    "Tight resource scoping, avoid wildcard where ARN exists; recommended conditions (MFA/network/tags) applied when applicable; "
    "permissions management requires iam:PolicyARN allow-list; each action should be in registry and cite provenance."
)


if dspy:

    class DraftPolicy(Signature):  # type: ignore
        nl_prompt: str = InputField()
        constraints: str = InputField()
        policy_json: str = OutputField(desc="JSON with fields iam_policy and test_config")
        rationale: str = OutputField()

    class Critique(Signature):  # type: ignore
        policy_json: str = InputField()
        registry_facts: str = InputField()
        pros: str = OutputField()
        cons: str = OutputField()
        patches: str = OutputField(desc="JSON Patch ops as JSON array")

    class JudgeVerdict(Signature):  # type: ignore
        nl_prompt: str = InputField()
        policy_json: str = InputField()
        pro_case: str = InputField()
        con_case: str = InputField()
        rubric: str = InputField()
        decision: str = OutputField(desc="accept|request_patch")
        patch: str = OutputField(desc="JSON Patch array if request_patch")
        rationale: str = OutputField()

    class Proponent(Module):  # type: ignore
        def __init__(self):
            super().__init__()
            self.step = Predict(Critique)

        def forward(self, policy_json: str, registry_facts: str):  # type: ignore
            return self.step(policy_json=policy_json, registry_facts=registry_facts)

    class Opponent(Module):  # type: ignore
        def __init__(self):
            super().__init__()
            self.step = Predict(Critique)

        def forward(self, policy_json: str, registry_facts: str):  # type: ignore
            return self.step(policy_json=policy_json, registry_facts=registry_facts)

    class Judge(Module):  # type: ignore
        def __init__(self):
            super().__init__()
            self.step = Predict(JudgeVerdict)

        def forward(self, nl_prompt: str, policy_json: str, pro_case: str, con_case: str, rubric: str):  # type: ignore
            return self.step(nl_prompt=nl_prompt, policy_json=policy_json, pro_case=pro_case, con_case=con_case, rubric=rubric)


@dataclass
class DSpyDebateOrchestrator:
    draft_model: str
    judge_model: str
    pro_model: str
    con_model: str

    def available(self) -> bool:
        return dspy is not None

    def configure(self):  # pragma: no cover
        if not dspy:
            return
        try:
            # Configure a single LM for all modules for simplicity; models can be swapped per module if desired.
            # Users can set provider keys via environment variables.
            dspy.settings.configure(lm=dspy.OpenAI(model=self.judge_model))  # type: ignore[attr-defined]
        except Exception:
            # Fallback: leave dspy with its default configuration if available
            pass

    def run_round(self, nl_prompt: str, policy: Dict[str, Any], registry_facts: str) -> DebateResult:
        if not dspy:
            # Should not be called if not available
            return DebateResult(summary="DSPy not available; no-op.", patches_applied=[], pro_case="", con_case="", judge_rationale="")

        self.configure()
        policy_json = json.dumps(policy)

        pro = Proponent()  # type: ignore
        con = Opponent()   # type: ignore
        jd = Judge()       # type: ignore

        pro_out = pro(policy_json=policy_json, registry_facts=registry_facts)  # type: ignore
        con_out = con(policy_json=policy_json, registry_facts=registry_facts)  # type: ignore
        verdict = jd(nl_prompt=nl_prompt, policy_json=policy_json, pro_case=pro_out.pros + "\n" + pro_out.patches, con_case=con_out.cons + "\n" + con_out.patches, rubric=RUBRIC_TEXT)  # type: ignore

        patches: List[Dict[str, Any]] = []
        if getattr(verdict, "decision", "").strip().lower() == "request_patch":
            try:
                patches = json.loads(getattr(verdict, "patch", "[]") or "[]")
                if not isinstance(patches, list):
                    patches = []
            except Exception:
                patches = []

        return DebateResult(
            summary=f"Judge decision: {getattr(verdict, 'decision', 'accept')}",
            patches_applied=patches,
            pro_case=(getattr(pro_out, "pros", "") or "") + "\n" + (getattr(pro_out, "patches", "") or ""),
            con_case=(getattr(con_out, "cons", "") or "") + "\n" + (getattr(con_out, "patches", "") or ""),
            judge_rationale=getattr(verdict, "rationale", ""),
        )

