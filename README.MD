
# IAM Actions Registry, Hybrid Search, and Policy Generator

Authoritative IAM actions registry builder (S3, EC2, IAM), hybrid dense+sparse retrieval with Pinecone, and a policy generator that turns natural language into minimal IAM policies plus test configs. Includes a simple Next.js UI to interact with the pipeline.

---

## Current Capabilities

- Build deterministic, provenance-rich local registries for:
  - S3, EC2, IAM from AWS Service Reference JSON (no HTML scraping for core facts)
- Enrich with official action descriptions (HTML scrape only for descriptions), add extras for search:
  - query hooks, condition hints, `sparse_text`, `dense_text`
- Build unified Pinecone indexes (dense via Gemini embeddings + sparse via BM25) across S3/EC2/IAM
- Unified query engine with:
  - Service router (Gemini 2.5 Flash) for service identification and query expansion
  - Hybrid retrieval (dense+sparse), RRF fusion, hosted reranking
- Policy generation (Gemini 2.5 Pro):
  - Vector-search mode: LLM constrained to the retrieved, authoritative actions
  - RAW mode: LLM-only without vector context
  - Outputs both `iam_policy` and matching `test_config`
- Web UI (`webui/`) to submit queries, toggle expansion / raw mode, and view results

---

## Repository Layout

```
src/
  fetch/
    s3_reference.py      # Fetch + cache AWS Service Reference JSON (S3)
    ec2_reference.py     # Fetch + cache AWS Service Reference JSON (EC2)
    iam_reference.py     # Fetch + cache AWS Service Reference JSON (IAM)
  parse/
    build_s3_registry_from_reference.py  # Transform → local S3 registry
    build_ec2_registry_from_reference.py # Transform → local EC2 registry
    build_iam_registry_from_reference.py # Transform → local IAM registry
  service_router.py      # Gemini-based service identification + query expansion
  policy_generator.py    # Natural language → IAM policy + test config

enrichment_scripts/
  s3_actions_extractor.py    # Scrape action descriptions from AWS docs (S3)
  ec2_actions_extractor.py   # Scrape action descriptions from AWS docs (EC2)
  iam_actions_extractor.py   # Scrape action descriptions from AWS docs (IAM)
  enrich_s3_actions.py       # Join descriptions → enriched_data/*_enriched.json
  enrich_ec2_actions.py
  enrich_iam_actions.py
  add_extras_fields.py       # Add query hooks, condition hints, sparse/dense text

data/
  raw/                       # Cached service-reference JSON by URL hash
  build_reports/             # Per-service build reports
  aws_iam_registry_*.json    # Core registry outputs per service
  *_actions.json             # Scraped descriptions

enriched_data/
  aws_iam_registry_*_enriched.json         # With descriptions
  aws_iam_registry_*_enriched_extras.json  # With extras for retrieval

pinecone/
  build_unified_indexes.py   # Dense+sparse unified indexes for S3+EC2+IAM
  query_unified.py           # Unified query engine CLI
  bm25_encoder_unified.pkl   # Saved BM25 encoder (generated)

webui/
  src/app/...                # Next.js app with API bridge to Python generator
  scripts/dev.mjs            # Dynamic port launcher (search 3010–3110)

experiments/
  policies/                  # Generated policies (if saving enabled)
  tests/                     # Generated test configs (if saving enabled)
```

---

## Setup

### Python environment

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r pinecone/requirements.txt
```

Environment variables required (export in your shell or set in a .env read by the code):

- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- Optional: `PINECONE_CLOUD` (default aws), `PINECONE_REGION` (default us-east-1)

### Node (UI) environment

```bash
cd webui
npm install
# Dev server picks a free port (default range 3010–3110)
npm run dev
```

Open the printed URL (e.g., http://localhost:3010). You can set a different base with `PORT=4000 npm run dev`.

---

## Build Registries (S3, EC2, IAM)

Core facts (actions, resource types with a primary ARN template, service-level condition keys, provenance) are taken from AWS Service Reference JSON.

```bash
# S3
python -m src.parse.build_s3_registry_from_reference

# EC2
python -m src.parse.build_ec2_registry_from_reference

# IAM
python -m src.parse.build_iam_registry_from_reference
```

Outputs:

- `data/aws_iam_registry_s3.json`
- `data/aws_iam_registry_ec2.json`
- `data/aws_iam_registry_iam.json`
- Reports under `data/build_reports/*.json`

Notes:

- Provenance is included on every row: `{url, table, row_index}`
- Required resource-type bit is not present in the JSON source → set to `required=false`
- Dependent actions are not present in the JSON source → `[]`
- For `arn_template`, the first ARN format is kept (MVP)

---

## Enrichment and Extras

1) Scrape action descriptions from AWS docs (only for descriptions):

```bash
python enrichment_scripts/s3_actions_extractor.py
python enrichment_scripts/ec2_actions_extractor.py
python enrichment_scripts/iam_actions_extractor.py
```

2) Join descriptions into registries → `enriched_data/*_enriched.json`:

```bash
python enrichment_scripts/enrich_s3_actions.py
python enrichment_scripts/enrich_ec2_actions.py
python enrichment_scripts/enrich_iam_actions.py
```

3) Add extras for retrieval quality → `enriched_data/*_enriched_extras.json`:

```bash
python enrichment_scripts/add_extras_fields.py
```

Extras include `query_hooks`, `condition_hints`, `sparse_text`, `dense_text`, and a `serialization_hash`.

---

## Build Unified Indexes (S3 + EC2 + IAM)

```bash
python pinecone/build_unified_indexes.py \
  --s3-input ../enriched_data/aws_iam_registry_s3_enriched_extras.json \
  --ec2-input ../enriched_data/aws_iam_registry_ec2_enriched_extras.json \
  --iam-input ../enriched_data/aws_iam_registry_iam_enriched_extras.json
```

This will:

- Create/connect to `aws-actions-dense-v2` and `aws-actions-sparse-v2`
- Fit and save a BM25 encoder at `pinecone/bm25_encoder_unified.pkl`
- Upsert dense (Gemini) and sparse (BM25) vectors with service metadata

---

## Query and Policy Generation

### Unified search (CLI)

```bash
python pinecone/query_unified.py "list S3 bucket contents" --services s3 --details
```

### Policy generation (CLI)

```bash
python src/policy_generator.py "upload files to S3 and read them back" \
  --services s3 \
  --threshold 0.0005 \
  --max-actions 15 \
  --model models/gemini-2.5-pro

# Disable query expansion
python src/policy_generator.py "upload files to S3" --services s3 --no-expand

# RAW mode (no vector search context; LLM-only)
python src/policy_generator.py "full admin on S3" --raw
```

Flags:

- `--services [s3|ec2|iam]` filter by services (default auto-detect via router)
- `--no-expand` disables query expansion
- `--raw` disables vector search context (RAW LLM mode)
- `--validate` adds basic structure validation output

Outputs:

- `iam_policy` and `test_config` (and saved under `experiments/` if not `--no-save`)

### Web UI

```bash
cd webui
npm run dev
```

Use the form to submit a query. Toggles:

- Services (optional)
- Enable query expansion
- RAW mode (no vector search)

Backend API route: `POST /api/generate`

Body fields:

```json
{
  "query": "...",                
  "services": ["s3"],            
  "threshold": 0.0005,            
  "maxActions": 15,               
  "model": "models/gemini-2.5-pro",
  "noExpand": false,              
  "raw": false                    
}
```

---

## S3 Registry Schema (reference)

```json
{
  "s3": {
    "service_name": "Amazon S3",
    "service_prefix": "s3",
    "page_url": "https://servicereference.us-east-1.amazonaws.com/v1/s3/s3.json",
    "version": "vX.Y",
    "actions": [
      {
        "action": "PutObject",
        "access_level": "Write",
        "resource_types": [
          { "type": "object", "required": false, "source": {"url": "...", "table": "actions", "row_index": 123} }
        ],
        "condition_keys": ["s3:x-amz-acl", "s3:prefix"],
        "dependent_actions": [],
        "source": {"url": "...", "table": "actions", "row_index": 123}
      }
    ],
    "resource_types": [
      {
        "type": "bucket",
        "arn_template": "arn:${Partition}:s3:::${BucketName}",
        "condition_keys": [],
        "source": {"url": "...", "table": "resource_types", "row_index": 0}
      }
    ],
    "service_condition_keys": ["s3:ExistingObjectTag/<key>", "..."]
  }
}
```

Notes on fields:

- `access_level` is derived from `Annotations.Properties` precedence:
  - `Permissions management` > `Write` > `Tagging` > `List` > `Read`
- `required` is `false` (source does not expose the SAR "required*" bit)
- `dependent_actions` is empty (not provided by the JSON feed)
- `arn_template` uses the first ARN format for each resource type

---

## Acceptance Checks (quick)

After building S3:

- `s3:PutObject` exists with `access_level: "Write"`
- `resource_types` includes `bucket` with a non-empty `arn_template`
- `service_condition_keys` non-empty and include keys starting with `s3:`

---

## Limitations (by design, MVP)

- Required resource types: unknown from the JSON source → `required=false`
- Dependent actions: not provided → `[]`
- Multiple ARN formats: only the first stored as `arn_template`

These are acceptable trade-offs for the MVP and can be extended later (e.g., storing all ARN formats, backfilling `required`, etc.).

---

## Why this approach

- Official, machine-readable, and maintained by AWS
- No HTML brittleness for core facts; faster builds; simpler diffs
- Sufficient to power the unified retrieval + policy generation MVP


# Source Fetcher (S3-only, MVP)

**Goal:** Build a deterministic, provenance-rich local registry of IAM facts for **Amazon S3** using AWS’s **Service Reference JSON** (no HTML scraping).

This registry is the single ground-truth input for downstream components (research agent, policy writer, canonizer, tests).

---

## What this does (today)

* Fetches  **S3** ’s official Service Reference JSON

  `https://servicereference.us-east-1.amazonaws.com/v1/s3/s3.json`
* Normalizes it into `data/aws_iam_registry_s3.json` with:

  * Actions (name, derived access level, action-level condition keys)
  * Resource types (with a primary ARN format)
  * Service-level condition keys
  * Provenance for every row (source URL + row index)
* Emits a small build report at `data/build_reports/s3_registry_report.json`
* Optionally, builds Pinecone indexes (see `pinecone/`) for search over S3 actions

> We do **not** scrape HTML. This is faster, more stable, and easier to keep current.

---

## Repo layout

```
src/
  fetch/
    s3_reference.py                     # Fetch + cache the official S3 JSON
  parse/
    build_s3_registry_from_reference.py # Transform → local registry
data/
  raw/                                  # Cached JSON
  build_reports/
  aws_iam_registry_s3.json              # Final registry (S3 only)
  s3_actions.json                       # Scraped descriptions (optional)
enrichment_scripts/
  s3_actions_extractor.py               # Scrape S3 actions/descriptions into data/s3_actions.json
  enrich_s3_actions.py                  # Join descriptions into registry → enriched_data/
pinecone/
  build_indexes_v2.py                   # Create dense+sparse indexes (v2)
  query.py                              # Hybrid search with Pinecone hosted reranker
  requirements.txt                      # pinecone, pinecone-text, google-genai, dotenv
```

---

## Install

```bash
pip install requests

# For Pinecone development (optional)
pip install -r pinecone/requirements.txt
```

---

## Build (S3 only)

```bash
python -m src.parse.build_s3_registry_from_reference
# Optional: --force to bypass cache
```

**Outputs**

* `data/aws_iam_registry_s3.json`
* `data/build_reports/s3_registry_report.json`

---

## Output schema (compact)

```json
{
  "s3": {
    "service_name": "Amazon S3",
    "service_prefix": "s3",
    "page_url": "https://servicereference.us-east-1.amazonaws.com/v1/s3/s3.json",
    "version": "vX.Y",
    "actions": [
      {
        "action": "PutObject",
        "access_level": "Write",
        "resource_types": [
          {
            "type": "object",
            "required": false,
            "source": {"url": ".../s3.json", "table": "actions", "row_index": 123}
          }
        ],
        "condition_keys": ["s3:x-amz-acl", "s3:prefix"],
        "dependent_actions": [],
        "source": {"url": ".../s3.json", "table": "actions", "row_index": 123}
      }
    ],
    "resource_types": [
      {
        "type": "bucket",
        "arn_template": "arn:${Partition}:s3:::${BucketName}",
        "condition_keys": [],
        "source": {"url": ".../s3.json", "table": "resource_types", "row_index": 0}
      }
    ],
    "service_condition_keys": ["s3:ExistingObjectTag/<key>", "..."]
  }
}
```

**Notes on fields**

* `access_level` is derived from `Annotations.Properties` precedence:
  * `Permissions management` > `Write` > `Tagging` > `List` > `Read`
* `required` is **currently `false`** for all resources (the JSON source does not expose the SAR “* required” bit).
* `dependent_actions` is empty (not provided by the JSON feed).
* `arn_template` uses the **first** ARN format for each resource type.

---

## Provenance & caching

* Every action/resource row carries `source.url`, `source.table`, `source.row_index`.
* The raw JSON is cached under `data/raw/` (by URL hash). Use `--force` to refetch.

---

## Acceptance checks (quick)

After building, assert:

* `s3:PutObject` exists with `access_level: "Write"`.
* Its `resource_types` includes `{"type":"object"}`.
* `resource_types` includes `bucket` with a non-empty `arn_template`.
* `service_condition_keys` is non-empty and keys start with `s3:`.

---

## Limitations (by design, MVP)

* **Required resource types:** unknown from this source; we set `required=false`.
* **Dependent actions:** not provided; remains `[]`.
* **Multiple ARN formats:** only the first is stored as `arn_template`.

These are acceptable trade-offs for the MVP and can be filled in later (e.g., via a tiny S3-only supplement, or by expanding the schema to store all `ARNFormats`).

---

## Why this approach

* Official, machine-readable, and maintained by AWS.
* No HTML brittleness; faster builds; simpler diffs.
* Perfectly sufficient to power the **S3-only** MVP for the research agent, policy writer, canonizer, and test generator.

---

## Next steps (once S3 passes)

* Optionally add a **`required` backfill** (S3-only) via one-time HTML read or curated map.
* Store **all** `ARNFormats` as `arn_formats[]` (non-breaking extension).
* Add `kms` next (then others) via the same JSON source with zero parser changes.
* Extend Pinecone indexes to include additional services (e.g., KMS) for cross-service queries.
