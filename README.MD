# Wisteria (Vermillion): IAM Policy Generation with Vector Search and Adversarial “Pro Mode”

Wisteria/Vermillion turns natural language into minimal, provenance-backed AWS IAM policies and matching test configurations. It combines a deterministic registry builder, a hybrid retrieval engine, and three generation modes — including an adversarial DSPy-inspired “Pro Mode” — plus a simple web UI.

## Highlights

- Authoritative IAM actions registries for S3, EC2, IAM from AWS Service Reference JSON.
- Enrichment (descriptions) and “extras” for high-recall search (dense + sparse).
- Hybrid retrieval: Gemini embeddings + BM25 → RRF fusion → hosted reranking.
- Policy generation modes:
  - Vector: LLM constrained to retrieved actions.
  - RAW: LLM-only (no vector context).
  - Adversarial Pro Mode: RAW draft → RegistryGuard → StaticChecker → triad debate → repaired final policy.
- Outputs both `iam_policy` and `test_config`.
- Next.js web UI with API bridge to Python.

## Table of Contents

- Quickstart
- Architecture
- Modes (Vector, RAW, Adversarial)
- Data and Index Build
- Using the Web UI
- Programmatic Usage (Python/API)
- Testing and CI
- Troubleshooting
- Security Considerations
- Roadmap

## Quickstart

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r pinecone/requirements.txt
```

### Environment

- Required: `GEMINI_API_KEY`
- Required for Vector mode: `PINECONE_API_KEY`
- Optional: `PINECONE_CLOUD` (default aws), `PINECONE_REGION` (default us-east-1)

### Web UI (Next.js)

```bash
cd webui
npm install
npm run dev   # will pick a free port (3010–3110)
```

Open the printed URL and submit a natural-language request.

## Architecture (Overview)

1) Registry build (S3, EC2, IAM) from AWS Service Reference JSON → `data/aws_iam_registry_*.json`.
2) Enrichment (descriptions) + extras → `enriched_data/*_enriched_extras.json`.
3) Index build (dense+sparse) → Pinecone `aws-actions-dense-v2`, `aws-actions-sparse-v2` and `pinecone/bm25_encoder_unified.pkl`.
4) Retrieval: service router (optional) → embeddings + BM25 → RRF → hosted rerank → ranked actions.
5) Generation:
   - Vector mode: LLM constrained to retrieved actions.
   - RAW mode: LLM-only.
   - Adversarial: RAW draft → RegistryGuard → StaticChecker → Triad debate → final policy.
6) UI/API: `webui/` bridges to Python and returns `{ iam_policy, test_config, metadata }`.

## Modes

### Vector
- Retrieval feeds a list of top-ranked actions; LLM must choose only from this list.
- Highest provenance and precision.

### RAW
- LLM selects actions with no retrieval context; fastest to prototype but unconstrained.

### Adversarial Pro Mode
- Draft (RAW) → `RegistryGuard` (canonicalize actions, attach provenance, suggest replacements) → `StaticChecker` (patches for Resource wildcards, `iam:PolicyARN` allow-list, tag-on-create recommendations) → Triad debate (Proponent/Opponent/Judge; DSPy optional) → patched final policy.
- Returns a repaired policy + deterministic `test_config` derived from the policy.

## Data and Index Build

Build registries (S3, EC2, IAM):

```bash
python -m src.parse.build_s3_registry_from_reference
python -m src.parse.build_ec2_registry_from_reference
python -m src.parse.build_iam_registry_from_reference
```

Enrichment + extras (for retrieval):

```bash
python enrichment_scripts/s3_actions_extractor.py
python enrichment_scripts/ec2_actions_extractor.py
python enrichment_scripts/iam_actions_extractor.py
python enrichment_scripts/enrich_s3_actions.py
python enrichment_scripts/enrich_ec2_actions.py
python enrichment_scripts/enrich_iam_actions.py
python enrichment_scripts/add_extras_fields.py
```

Build unified indexes:

```bash
python pinecone/build_unified_indexes.py \
  --s3-input ../enriched_data/aws_iam_registry_s3_enriched_extras.json \
  --ec2-input ../enriched_data/aws_iam_registry_ec2_enriched_extras.json \
  --iam-input ../enriched_data/aws_iam_registry_iam_enriched_extras.json
```

## Using the Web UI

- Mode selector: Vector | RAW | Adversarial (Pro)
- Services filter (optional)
- Query expansion toggle (Vector)
- Score threshold and max actions (Vector)
- Adversarial advanced models (optional): Draft/Judge/Proponent/Opponent (quick dropdown + free text)
- Optional “Use DSPy debate” checkbox (if DSPy installed)

## API Contract (UI → API)

Route: `POST /api/generate`

```json
{
  "query": "upload files to S3 and read them back",
  "services": ["s3"],
  "threshold": 0.0005,
  "maxActions": 15,
  "model": "models/gemini-2.5-pro",
  "noExpand": false,
  "raw": false,
  "mode": "vector" | "raw" | "adversarial",
  "rounds": 1,
  "judgeModel": "models/gemini-2.5-flash",
  "proModel": "models/gemini-2.5-flash",
  "conModel": "models/gemini-2.5-flash",
  "draftModel": "models/gemini-2.5-pro",
  "useDSPy": false
}
```

Response (excerpt):

```json
{
  "status": "success",
  "iam_policy": { ... },
  "test_config": { ... },
  "metadata": {
    "mode": "adversarial",
    "rounds": 1,
    "models": { ... },
    "registry": {
      "mismatches": ["s3:PutObjects"],
      "replacements": {"s3:PutObjects": "s3:PutObject"},
      "provenance": {"s3:PutObject": {"service": "s3", "row": 142}}
    },
    "debate": {
      "summary": "Round 1: Patches applied to remove wildcard",
      "patches_applied": [ {"op":"replace","path":"/Statement/0/Resource/0","value":"arn:aws:s3:::${BUCKET_NAME}/*"} ]
    },
    "static_checker": {"repairs": [ ... ], "status": "pass"},
    "tokens": {"total": 1835},
    "latency_ms": {"total": 1340},
    "score": {"tightness": 0.87, "risk": 0.05}
  }
}
```

## Programmatic Usage (Python)

Vector/RAW (class):

```python
from src.policy_generator import IAMPolicyGeneratorV2

gen = IAMPolicyGeneratorV2(
    score_threshold=0.0005,
    max_actions=15,
    model="models/gemini-2.5-pro",
    use_query_expansion=True,
    target_services=["s3"],
    use_vector_search=True,  # False => RAW
)
result = gen.generate_policy("upload files to S3 and read them back")
```

Adversarial (function):

```python
from src.policy_generator_adv import generate_adversarial

res = generate_adversarial(
    nl_prompt="upload files to S3 and read them back",
    rounds=1,
    models={
        "draft": "models/gemini-2.5-pro",
        "judge": "models/gemini-2.5-flash",
        "pro": "models/gemini-2.5-flash",
        "con": "models/gemini-2.5-flash",
    },
    settings={"use_dspy": False}  # True if DSPy installed/configured
)
```

## Testing and CI

- Run adversarial tests (offline or online):
  - `pytest -q tests/test_adversarial_mode.py`
- Full test suite (requires keys for LLM/vector tests):
  - `pytest -q`
- CI: GitHub Actions runs pytest on push/PR with keys unset (LLM/vector tests skipped by design).
- Nightly job is included but disabled; wire secrets and enable when ready.
- See `tests.md` for the full test inventory and guidance.

## Troubleshooting

- No actions retrieved in Vector mode: ensure enriched extras and Pinecone indexes exist; verify Pinecone env vars.
- LLM parsing errors: reduce temperature or retry; generator strips common markdown fences.
- UI returns generator failed: inspect stderr captured by the API bridge; verify env and paths.

## Security Considerations

- Prefer Vector or Adversarial modes for provenance-constrained output.
- Adversarial StaticChecker enforces `iam:PolicyARN` allow-lists and removes wildcards when possible.
- Always review generated policies; validators check structure, not business semantics.

## Roadmap

- Expand registries with more services (e.g., KMS).
- Improve DSPy debate with role-specific LMs and structured patches.
- Add streaming of debate patches and “why/provenance” chips to the UI.

