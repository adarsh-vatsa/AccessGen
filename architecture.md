## Policy Generation Architecture (Web UI)

This document explains the end-to-end technical architecture for the policy generation flow invoked from the web UI (`webui/`). It covers the data sources, retrieval pipeline, LLM generation, available modes/options, and operational requirements.

---

### High-level Flow

1) User enters a natural-language requirement in the UI and submits the form
2) Next.js API route (`POST /api/generate`) bridges into Python
3) Python generator (`src/policy_generator.py`) orchestrates retrieval + LLM
4) The LLM (Gemini 2.5 Pro) outputs both `iam_policy` and `test_config`
5) API returns JSON → UI renders results

---

## Components

### Web UI (Next.js)
- Location: `webui/`
- Dev launcher: `scripts/dev.mjs` (dynamic port probing; default range 3010–3110; override with `PORT=`)
- Page: `src/app/page.tsx`
  - Inputs: natural-language query
  - Options: select services (optional), enable/disable query expansion, RAW mode (no vector context), score threshold, max actions
  - Results panes for `iam_policy`, `test_config`, and metadata

### API Bridge
- Route: `webui/src/app/api/generate/route.ts`
- Behavior:
  - Accepts JSON body from the UI
  - Spawns a Python process (`python3 -c ...`) with:
    - `cwd` set to the repo root
    - environment variables passed through (notably `GEMINI_API_KEY`, `PINECONE_API_KEY`)
  - Invokes `IAMPolicyGeneratorV2` and prints the result as JSON to stdout
  - Returns the parsed JSON to the browser (or error with `stdout/stderr` for diagnostics)
- Request body schema:
  - `query: string` (required)
  - `services?: string[]` (optional, subset of [`s3`,`ec2`,`iam`])
  - `threshold?: number` (optional; min relevance for action selection)
  - `maxActions?: number` (optional; cap on candidate actions sent to LLM)
  - `model?: string` (optional; default `models/gemini-2.5-pro`)
  - `noExpand?: boolean` (optional; disables query expansion)
  - `raw?: boolean` (optional; enable raw mode; no vector-search context)

### Python Generator
- Entry: `src/policy_generator.py`
- Class: `IAMPolicyGeneratorV2`
  - LLM: Gemini 2.5 Pro (`models/gemini-2.5-pro`) for policy generation
  - Retrieval: `pinecone/query_unified.py` (UnifiedQueryEngine) for hybrid search
  - Query expansion: `src/service_router.py` (Gemini 2.5 Flash) when enabled
  - Modes:
    - Vector-search mode (default): retrieve top actions and constrain LLM to them
    - RAW mode (`--raw` / `raw: true`): no retrieval context, LLM-only
  - Outputs:
    - `iam_policy`: IAM Policy JSON
    - `test_config`: matching test rules for validation/fuzzing
    - `metadata`: model name, timestamp, counts, top actions considered, etc.

---

## Retrieval Pipeline (Vector-search Mode)

Implemented in `pinecone/query_unified.py` with the following steps:

1) Optional service detection and query expansion
   - `ServiceRouter` (Gemini 2.5 Flash) identifies services and expands the query (controllable via UI `Enable query expansion`)

2) Query encoding
   - Dense: Gemini embeddings (`models/gemini-embedding-001`, 3072-d; L2-normalized)
   - Sparse: BM25 (with simple verb expansions for better recall)

3) Retrieval from Pinecone
   - Dense index: `aws-actions-dense-v2` (cosine)
   - Sparse index: `aws-actions-sparse-v2` (dotproduct)
   - Optional metadata filter by `service`

4) Fusion + Reranking
   - Reciprocal Rank Fusion (RRF) across dense+sparse
   - Pinecone hosted reranker (`pinecone-rerank-v0`) to select top-M

5) Candidate assembly
   - Adds explicit action hints detected from the original query (CamelCase tokens)
   - Builds short rerank texts including service, description, resources, condition keys

6) Results
   - Each item includes `service`, `action`, `access_level`, `description`, `resource_types`, `condition_keys`, `rerank_score`

Data dependency for retrieval:
- Enriched action data (with `sparse_text` / `dense_text`) loaded into an in-process lookup:
  - `enriched_data/aws_iam_registry_s3_enriched_extras.json`
  - `enriched_data/aws_iam_registry_ec2_enriched_extras.json`
  - `enriched_data/aws_iam_registry_iam_enriched_extras.json`
- BM25 encoder: `pinecone/bm25_encoder_unified.pkl` (built by `pinecone/build_unified_indexes.py`)
- Pinecone indexes: created/connected by `build_unified_indexes.py`

---

## LLM Generation

LLM: Gemini 2.5 Pro (configurable via `model`)

System prompt encodes rules:
- Minimal, least-privilege policy
- Use restrictive conditions when appropriate
- Prefer specific action lists; avoid adding actions not in provided list (vector mode)
- Provide both `iam_policy` and `test_config` in a strict JSON object

User prompt differs by mode:
- Vector-search mode: includes the natural-language instruction plus a formatted list of top-ranked actions and their details; LLM must select only from this list
- RAW mode: includes only the instruction and constraints; LLM chooses actions without a retrieval list

Post-processing:
- Cleans out accidental markdown fences
- Parses JSON into `{ iam_policy, test_config }` or returns an error with raw text for debugging

Metadata:
- `actions_searched` and `top_actions` reported only in vector-search mode

---

## Options and Their Effects

- Services filter (`services`): narrows retrieval to one or more of [`s3`, `ec2`, `iam`]; ignored in RAW mode
- Query expansion (`noExpand=false`): enables service router to expand query for recall; off means the original text is used verbatim
- RAW mode (`raw=true`): disables vector-search context; the LLM is unconstrained by retrieved actions
- Score threshold (`threshold`): filters retrieved actions by rerank score before sending to LLM (vector mode)
- Max actions (`maxActions`): caps number of actions sent to LLM (vector mode)
- Model (`model`): defaults to `models/gemini-2.5-pro`

Typical UI combinations:
- High-precision: Services selected → Expand ON → Vector mode (default)
- Direct LLM baseline: RAW ON (no services necessary) → Expand optional

---

## Data Build and Prerequisites

Registries (core facts from AWS Service Reference JSON):
- `src/parse/build_*_registry_from_reference.py` → `data/aws_iam_registry_*.json`
  - Actions and derived access levels (Permissions management > Write > Tagging > List > Read)
  - Resource types with primary ARN templates
  - Service-level condition keys
  - Provenance: `{url, table, row_index}` on every row

Enrichment (descriptions) and extras:
- `enrichment_scripts/*_actions_extractor.py` → `data/*_actions.json`
- `enrichment_scripts/enrich_*_actions.py` → `enriched_data/*_enriched.json`
- `enrichment_scripts/add_extras_fields.py` → `enriched_data/*_enriched_extras.json`

Unified indexes:
- `pinecone/build_unified_indexes.py` uses the three `*_enriched_extras.json` files
- Creates/uses `aws-actions-dense-v2` and `aws-actions-sparse-v2`
- Saves BM25 encoder to `pinecone/bm25_encoder_unified.pkl`

Env vars:
- `GEMINI_API_KEY` (required)
- `PINECONE_API_KEY` (required for vector mode)
- Optional: `PINECONE_CLOUD` (default aws), `PINECONE_REGION` (default us-east-1)

---

## API Contract

Request (UI → API):
```json
{
  "query": "upload files to S3 buckets and read them back",
  "services": ["s3"],
  "threshold": 0.0005,
  "maxActions": 15,
  "model": "models/gemini-2.5-pro",
  "noExpand": false,
  "raw": false
}
```

Response:
```json
{
  "status": "success" | "error",
  "query": "...",
  "iam_policy": { ... },
  "test_config": { ... },
  "metadata": {
    "timestamp": "...",
    "model": "models/gemini-2.5-pro",
    "original_query": "...",
    "search_query": "..." | null,
    "actions_searched": 0 | N,
    "score_threshold": 0.0005,
    "top_actions": [
      { "action": "s3:GetObject", "score": 0.27, "included": true },
      ...
    ]
  }
}
```

On error, the API returns:
```json
{
  "error": "Generator failed",
  "details": { "status": 1, "stdout": "...", "stderr": "..." }
}
```

---

## Operational Notes

- The API bridge runs `python3 -c` with `cwd` at the repo root and an absolute `REPO_ROOT` in the snippet; if you move the repo, update `route.ts` accordingly
- Ensure the UI process inherits `GEMINI_API_KEY` (and Pinecone keys for vector mode)
- The `.venv` created at repo root is used implicitly by the spawned Python if you start the web server from a shell with the venv activated; otherwise, make sure `python3` on PATH has the right packages
- Index build time can be minutes depending on network and embedding throughput

---

## Security and Scope

- RAW mode can produce actions not present in your curated registry; use vector-search mode for provenance-constrained results
- The generated policy should be validated and reviewed; the included `validate_*` helpers check schema, not semantics
- Avoid exposing your API keys via client-side code; the API route only reads from server-side `process.env`

---

## Troubleshooting

- UI returns a pale or hard-to-read display: the UI includes high-contrast styles for both light/dark; hard-reload to bust cached CSS
- API returns "Generator failed": check `details.stderr` for Python errors (missing env vars, missing indexes/encoder, etc.)
- Retrieval returns zero actions: confirm enriched extras JSON files exist and indexes are created; verify Pinecone env vars
- LLM parsing errors: the generator cleans common markdown fences; still, occasionally models can output malformed JSON—retry or lower temperature


