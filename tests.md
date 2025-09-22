Overview

This repository includes a multi-layered test and evaluation setup covering registry builds, retrieval, policy generation (vector/raw), and the new Adversarial (DSPy) Pro Mode.

Test Classes and Scripts

- Unit/Component
  - Registry build and fetch
    - tests/test_fetch_s3_reference.py: Validates S3 service-reference fetch and caching.
    - tests/test_build_s3_registry.py: Ensures normalization and schema for aws_iam_registry_s3.json.
  - Guards and static checks (adversarial)
    - tests/test_adversarial_mode.py: RegistryGuard suggestions, StaticChecker repairs, end-to-end adversarial pipeline behavior.

- Integration
  - tests/test_integration.py: End-to-end pipeline tests using mocks (fetch → parse → registry); also contains optional slow tests guarded by skip.
  - tests/test_consistency.py: CLI-based consistency checker across policy vs test_config (runner behavior wrapped in main()).
  - tests/test_various_policies.py: CLI smoke tests helper (helper function; not auto-collected by pytest).

- LLM-dependent
  - tests/test_policy_generation.py: Vector/RAW generator tests (skipped unless GEMINI_API_KEY present). Adjusted to import from src and use result["iam_policy"].

Adversarial Tests (details)

- tests/test_adversarial_mode.py
  - Offline/Online toggle: A test fixture respects the environment variable ADVERSARIAL_OFFLINE=1 to force offline operation (unset GEMINI_API_KEY). By default, if GEMINI_API_KEY is set, adversarial tests use real LLM calls.
  - default models: draft/judge/pro/con set to "models/gemini-2.5-pro".
  - Validations:
    - End-to-end success with structured metadata (mode/rounds/models/registry/debate/static_checker).
    - No wildcard Resource after static repair.
    - Scoring reflects improved tightness and low risk.
    - DSPy flag path (settings={"use_dspy": True}) succeeds or falls back gracefully.
    - RegistryGuard suggests near-miss correction (e.g., s3:PutObjects → s3:PutObject).
    - StaticChecker enforces iam:PolicyARN allow-list for iam:*Policy* attach/put.

How to Run Locally

- Offline (no network):
  - export ADVERSARIAL_OFFLINE=1
  - pytest -q tests/test_adversarial_mode.py

- Online (LLM active):
  - export GEMINI_API_KEY=... (and PINECONE_API_KEY for vector tests)
  - pytest -q

CI Configuration

- .github/workflows/ci.yml
  - test job (default):
    - Runs on push/PR across Python 3.10/3.11
    - GEMINI_API_KEY and PINECONE_API_KEY are empty to skip LLM/Pinecone tests
    - Executes pytest -q
  - nightly job (disabled):
    - Skeleton configured with `if: ${{ false }}` to prevent auto-run
    - Uses secrets.GEMINI_API_KEY and secrets.PINECONE_API_KEY to run full suite when enabled

Maintenance

- Update this document whenever the test suite is modified:
  - New tests added or removed
  - Changes to environment strategy (offline/online)
  - CI workflow updates (schedules, secrets, matrix)
  - Evaluation scripts (e.g., scripts/run_eval.py) behavior or location

Evaluation Harness

- scripts/run_eval.py compares Vector vs Adversarial over a CSV of prompts and outputs eval_results.csv (offline for Adversarial; Vector requires Pinecone/indexes).

