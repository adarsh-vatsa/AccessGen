# IAM Actions RAG - Pinecone Implementation

## Overview

This directory contains the implementation of a hybrid retrieval system for AWS IAM actions using:
- **Sparse retrieval**: BM25 for exact token matches
- **Dense retrieval**: Gemini embeddings for semantic search
- **RRF fusion**: Reciprocal Rank Fusion to combine results
- **Reranking**: Pinecone hosted reranker (default: `pinecone-rerank-v0`)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Preferred: .env file in this directory
cat > .env << 'EOF'
PINECONE_API_KEY=your-pinecone-api-key
GEMINI_API_KEY=your-gemini-api-key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
# Optional overrides
# PINECONE_RERANK_MODEL=pinecone-rerank-v0
# PINECONE_RERANK_MAX_DOCS=100
EOF
```

### 3. Build Indexes (v2)

```bash
python build_indexes_v2.py --input ../enriched_data/aws_iam_registry_s3_enriched_extras.json \
  --dense-index iam-dense-v2 --sparse-index iam-sparse-v2
```

This will:
- Create `iam-dense-v2` (dense, cosine) and `iam-sparse-v2` (sparse, dotproduct)
- Fit and save a BM25 encoder to `bm25_encoder_v2.pkl`
- Process all 165 S3 actions and upsert to both indexes

## Usage

### Query Examples

```bash
# Basic query (uses .env, defaults to v2 indexes and hosted reranker)
python query.py "list bucket contents"

# Get more results
python query.py "delete objects and versions" --top-k 30

# Detailed output
python query.py "list bucket versions" --details

# Explicit v2 targets
python query.py "put bucket policy" \
  --dense-index iam-dense-v2 --sparse-index iam-sparse-v2 \
  --bm25 ./bm25_encoder_v2.pkl --details
```

### Programmatic Usage

```python
from query import QueryEngine

engine = QueryEngine(
    dense_index_name="iam-dense-v2",
    sparse_index_name="iam-sparse-v2",
    bm25_path="./bm25_encoder_v2.pkl",
)
engine.load_actions_data("../enriched_data/aws_iam_registry_s3_enriched_extras.json")
results = engine.hybrid_search("allow cross-region replication", top_m=15)
```

## Architecture

### Index Structure

**Dense Index (`iam-dense-v2`)**:
- Metric: `cosine`
- Dimension: 3072 (Gemini full)
- Content: `dense_text` field embeddings (L2-normalized)

**Sparse Index (`iam-sparse-v2`)**:
- Type: `sparse`
- Metric: `dotproduct`
- Content: BM25-encoded `sparse_text` fields

### Search Pipeline

1. **Query Encoding**:
   - Dense: Gemini embedding (L2-normalized, 3072-d)
   - Sparse: BM25 encoding (tiny verb expansion for recall)
2. **Retrieval**:
   - Dense: Top-165 (all S3 actions)
   - Sparse: Top-120 candidates
3. **RRF Fusion**:
   - Combines rankings with k=60
   - Produces unified candidate list
4. **Reranking**:
   - Pinecone hosted reranker (default `pinecone-rerank-v0`)
   - Documents capped (default 100) to meet model limits

## Data Fields

- `sparse_text`: Symbols only (action, resources, condition keys)
- `dense_text`: Symbols + description + query hints
- `description`: Official AWS description text
- `resource_types`: Required/optional AWS resources
- `condition_keys`: Available IAM condition keys

## Notes

- Only S3 actions are indexed currently. Cross-service (e.g., KMS) requires ingesting additional services.
- If you change the dense dimension later, you must create a new dense index and re-embed.
- All required environment variables are validated on startup; missing variables cause immediate exit.
