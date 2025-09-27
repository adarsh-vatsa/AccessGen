Issues to Address

- stream_raw undefined variable
  - File: src/stream_raw.py:64
  - Issue: Sends SSE trace with variable `system_prompt` which is not defined. Should either remove the trace or reference the built `full_prompt` if needed. Safer: drop the prompt echo to avoid leaking prompts in logs.
  - Fix plan: Replace `system_prompt` with `"raw_policy_only"` label or remove that trace line.

- ServiceRouter model mismatch
  - Files: src/README.md, src/service_router.py
  - Issue: README example uses `models/gemini-2.0-flash-exp` while implementation default is `models/gemini-2.5-flash`.
  - Fix plan: Align docs and CLI help to 2.5‑flash (current default) or parametrize via env.

- Vector mode dependency messaging
  - File: src/policy_generator.py
  - Issue: Error message on missing vector engine references uninstalling `pinecone-plugin-inference` which may confuse users not using the plugin.
  - Fix plan: Reword to: “Vector search unavailable; ensure `pinecone/` folder with `query_unified.py` exists or run with `--raw`.”

- StaticChecker condition overwrite
  - File: src/guards/static_checker.py
  - Issue: Patches can replace entire `Condition`, potentially discarding existing useful conditions.
  - Fix plan: Implement shallow merge to preserve existing operators and keys where feasible.

- Registry file assumptions and fallbacks
  - Files: src/guards/registry_guard.py, src/guards/registry_validator.py
  - Issue: Assume enriched_extras registries exist; silent behavior when missing may reduce guard/validator coverage.
  - Fix plan: Add explicit warnings when registries not found and document minimal fallback behavior.

- Fetch file write consistency
  - Files: src/fetch/ec2_reference.py, src/fetch/iam_reference.py
  - Issue: S3 fetch uses atomic temp‑file replace; EC2/IAM direct write could race.
  - Fix plan: Use same atomic write strategy for EC2/IAM.

- Parser limitations note
  - Files: src/parse/build_*_registry_from_reference.py
  - Issue: Resource‑level condition keys are left empty due to source format; note is present but we should surface the limitation in README.
  - Fix plan: Document clearly and consider storing full ARNFormats array.

- LLM Router JSON mode imports
  - File: src/llm/router.py
  - Issue: Importing OpenAI JSON mode helper types can fail with older SDKs; code handles optional schema class, but better error guidance could help.
  - Fix plan: Improve exception message to include installed OpenAI SDK version and minimum requirement.

