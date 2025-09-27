# DSPy Strategy for Adversarial “Pro Mode”

This document describes how to leverage DSPy’s full capabilities to improve our Adversarial (triad) policy‑generation mode. It is an implementation guide to be referenced during the build-out.

## Goals

- Increase reliability of debate outputs (patches, verdicts) via strongly typed, validated signatures.
- Reduce token/cost with role‑appropriate LMs, caching, and retrieval‑focused context.
- Improve quality with DSPy teleprompting/optimizers on a small gold set.
- Preserve full traceability (inputs/outputs per step) and stream live progress.

## Current State (MVP)

- Proponent/Opponent/Judge run as single‑shot Predict calls with free‑form string outputs.
- One global LM for all roles.
- No DSPy optimizers/teleprompting, no verifier, no structured guarantees.
- RegistryGuard + StaticChecker perform deterministic canon + patch recommendations.

## Target DSPy Design

### 1) Role‑Specific LMs

- Draft: high‑accuracy LM (e.g., Gemini 2.5 Pro / GPT‑4 class).
- Proponent/Opponent: faster LM for critique + patch proposal (e.g., Gemini Flash / GPT‑4o‑mini).
- Judge: accuracy‑oriented LM (e.g., Gemini Flash high quality or Pro), distinct from Pro/Con.

Implementation sketch:
- Configure per‑module LM where supported (Predict(Signature, lm=...)). If unavailable, temporarily override `dspy.settings.configure(lm=...)` inside the module call and restore afterward.

### 2) Strongly Typed Signatures

Replace free‑form strings with constrained fields, especially for patches and decisions. Enforce JSON parsing and validation.

Example:

```
class Critique(dspy.Signature):
  policy_json: str = InputField()
  registry_facts: str = InputField()
  pros: str = OutputField()
  cons: str = OutputField()
  patches_json: str = OutputField(desc="JSON array of RFC6902 patches")

class JudgeVerdict(dspy.Signature):
  nl_prompt: str = InputField()
  policy_json: str = InputField()
  pro_case: str = InputField()
  con_case: str = InputField()
  rubric: str = InputField()
  decision: str = OutputField(desc="accept|request_patch")
  patch_json: str = OutputField(desc="JSON array of patches if request_patch")
  rationale: str = OutputField()
```

Post‑processing:
- Parse `patches_json` / `patch_json` → `list[dict]`.
- Validate with a minimal schema: `{op: str in {add, replace, remove}, path: str, value?: any}`.
- On parse/validate failure, request re‑emission or fallback to StaticChecker patches.

### 3) Orchestration Program

- Compose Proponent → Opponent → Judge per round. Optionally, a second Judge (consensus) or re‑judge with a different LM for reliability on hard cases.
- Maintain a working policy state; after each round, apply judge patch(es), then re‑run RegistryGuard + StaticChecker.
- Stop early if no new patches are proposed and checker passes.

### 4) Verifier Stage (Optional but Recommended)

Add a small `VerifyPatch` Signature after the Judge but before applying patches, to check rubric compliance:
- Reject introduces non‑registry actions.
- Reject widens `Resource` scope.
- Require `iam:PolicyARN` allow‑lists for `iam:*Policy*` attach/put actions.

The Judge can consult the verifier output before deciding, or we perform verification post‑judge and request a corrected patch.

### 5) Retrieval + Context Narrowing

- Feed the debate only the most relevant registry facts:
  - For each included action, select top‑k facts (description, resource types, condition keys) from provenance.
  - Limit to concise bullets (service:Action — desc | resources | keys) to control prompt length.
- If needed, add a DSPy RAG step that surfaces only action rows referenced in the draft.

### 6) Teleprompting / Optimization

Use DSPy optimizers (BootstrapFewShot, MIPROv2, etc.) with a small gold set (10–30 examples) to tune:
- Critique prompts to produce minimal, valid patch sets.
- Judge prompts to favor tighter resource scope and required conditions.

Dataset schema (YAML/JSON lines):
- `nl_prompt`
- `initial_policy`
- `registry_facts`
- `expected_decision` (accept|request_patch)
- `expected_patch` (JSON Patch array)
- `notes` (rationale)

Metric:
- `tightness_gain - penalty(risk_increase) - invalid_patch_penalty`.

### 7) Tracing + Streaming + Metrics

- Continue emitting per‑step trace (input/outputs, raw text for draft) via SSE.
- Capture `tokens`, `latency_ms` per role and per round.
- Surface these in UI metadata for troubleshooting and cost tracking.

### 8) Caching & Idempotence

- Cache draft outputs by `(nl_prompt, draft_model)`.
- Cache critique/judge steps by `(policy_hash, facts_hash, model)`.
- Ensure patch application is idempotent; avoid duplicate patches across rounds.

## API and Integration Points

- `src/debate/dspy_llm.py` (to be upgraded):
  - Add role‑specific LMs, structured outputs, validation utils.
  - Expose a `DSpyDebateOrchestrator` with `run_round(...)` returning typed results.
- `src/policy_generator_adv.py`:
  - Wire in `DSpyDebateOrchestrator` and optional verifier step.
  - Add token/latency capture per call.
- UI: live SSE trace already receives per‑step artifacts; add token/time breakdown.

## Testing Plan

- Unit tests for JSON Patch parsing/validation (happy/edge cases).
- “Golden” tests against the seed dataset (assert acceptable patch quality and scores).
- Robustness tests: malformed outputs from LMs → recovery (retry/fallback to checker).
- Latency/cost smoke tests with small payloads.

## Milestones

1) Structured outputs + validation (high impact, low risk).
2) Role‑specific LMs + per‑step metrics.
3) Teleprompting on small gold set; check quality gains.
4) Verifier integration and re‑judge loop on failures.
5) RAG/narrowing of registry facts and caching layer.
6) Extended UI metrics (tokens/latency per step) and downloadable trace bundle.

## Risks & Mitigations

- Unstable JSON from models → enforce schema and retry with minimal deltas.
- Prompt drift when adding examples → teleprompt iteratively with eval metrics.
- Cost/latency → role LM tuning, caching, and fact narrowing.

## Appendix: Example Judge Rubric

- Tight resource scoping; avoid `Resource: "*"` when ARNs exist.
- Require `iam:PolicyARN` allow‑lists for `iam:*Policy*` attach/put.
- Prefer minimal action sets.
- Only registry‑present actions; replace near‑miss with canonical.
- Respect user intent and original verbs; note limitations if constraints block requirements.

