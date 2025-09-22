#!/usr/bin/env python3
"""
IAM Actions RAG - Index Builder
Creates both sparse (BM25) and dense (Gemini embeddings) Pinecone indexes.
"""
import json
import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

# Third-party imports
import pinecone
from pinecone_text.sparse import BM25Encoder
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self):
        self.pc = None
        self.bm25 = BM25Encoder()
        self.dense_index = None
        self.sparse_index = None

        # Gemini
        self.embedding_model = "models/gemini-embedding-001"
        # Desired output dim; actual index dim will be detected from the model output
        self.embedding_dimension = 768

        # Index names
        self.dense_index_name = "iam-dense-v1"
        self.sparse_index_name = "iam-sparse-v1"

        self._load_env()

    def _load_env(self):
        """Load required environment variables and init clients."""
        required = ["PINECONE_API_KEY", "GEMINI_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            sys.exit(1)

        # Pinecone client (v3)
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Gemini client
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Environment loaded: Pinecone + Gemini ready")

    def load_data(self, input_file: str) -> List[Dict[str, Any]]:
        """Load and parse the enriched S3 actions JSON."""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            actions = data["s3"]["actions"]
            logger.info(f"Loaded {len(actions)} S3 actions from {input_file}")
            return actions
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

    def get_embedding_dimension(self, sample_text: str) -> int:
        """Probe Gemini for actual output dimension (respects output_dimensionality if supported)."""
        try:
            cfg = types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.embedding_dimension,
            )
            res = self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=[sample_text],
                config=cfg,
            )
            dim = len(res.embeddings[0].values)
            return dim
        except Exception as e:
            logger.warning(f"Could not probe Gemini embedding dim, defaulting to {self.embedding_dimension}: {e}")
            return self.embedding_dimension

    @staticmethod
    def l2_normalize(vec: List[float]) -> List[float]:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return vec
        return (arr / norm).tolist()

    def ensure_indexes(self, actions: List[Dict[str, Any]]):
        """Create Pinecone indexes if they don't exist (serverless)."""
        existing = [idx.name for idx in self.pc.list_indexes()]

        # Determine dense dimension from model
        sample_dense_text = actions[0]["dense_text"]
        dense_dim = self.get_embedding_dimension(sample_dense_text)
        logger.info(f"Dense index dimension (Gemini): {dense_dim}")

        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        # Dense
        if self.dense_index_name not in existing:
            logger.info(f"Creating dense index: {self.dense_index_name}")
            self.pc.create_index(
                name=self.dense_index_name,
                dimension=dense_dim,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud=cloud, region=region),
            )
            while not self.pc.describe_index(self.dense_index_name).status["ready"]:
                logger.info("Waiting for dense index to be ready...")
                time.sleep(3)
        else:
            logger.info(f"Dense index {self.dense_index_name} already exists")

        # Sparse index (true sparse type)
        if self.sparse_index_name not in existing:
            logger.info(f"Creating sparse index: {self.sparse_index_name}")
            self.pc.create_index(
                name=self.sparse_index_name,
                dimension=1,  # ignored for sparse; keep minimal value
                metric="dotproduct",
                index_type="sparse",
                spec=pinecone.ServerlessSpec(cloud=cloud, region=region),
                deletion_protection="disabled",
            )
            while not self.pc.describe_index(self.sparse_index_name).status["ready"]:
                logger.info("Waiting for sparse index to be ready...")
                time.sleep(3)
        else:
            logger.info(f"Sparse index {self.sparse_index_name} already exists")

        self.dense_index = self.pc.Index(self.dense_index_name)
        self.sparse_index = self.pc.Index(self.sparse_index_name)

    def fit_bm25(self, actions: List[Dict[str, Any]]):
        logger.info("Fitting BM25 encoder over sparse_text...")
        corpus = [a["sparse_text"] for a in actions]
        self.bm25.fit(corpus)
        logger.info("BM25 encoder fitted")

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Batch-embed with Gemini, returning L2-normalized vectors."""
        out: List[List[float]] = []
        cfg = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.embedding_dimension,
        )
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                res = self.genai_client.models.embed_content(
                    model=self.embedding_model, contents=batch, config=cfg
                )
                for emb in res.embeddings:
                    out.append(self.l2_normalize(emb.values))
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Gemini embed batch {i//batch_size+1} failed: {e}")
                # Fallback to zero vectors (keeps shape)
                for _ in batch:
                    out.append([0.0] * self.embedding_dimension)
        return out

    def upsert_dense_vectors(self, actions: List[Dict[str, Any]], batch_size: int = 100):
        logger.info("Creating dense vectors (Gemini) ...")
        dense_texts = [a["dense_text"] for a in actions]
        vecs = self.get_embeddings_batch(dense_texts)

        vectors = []
        for a, v in zip(actions, vecs):
            vectors.append(
                {
                    "id": f"s3::{a['action']}",
                    "values": v,
                    "metadata": {
                        "service": "s3",
                        "action": a["action"],
                        "access": a["access_level"],
                    },
                }
            )

        logger.info(f"Upserting {len(vectors)} dense vectors ...")
        for i in range(0, len(vectors), batch_size):
            self.dense_index.upsert(vectors=vectors[i : i + batch_size])
            logger.info(f"Dense upsert batch {i//batch_size + 1}/{(len(vectors)+batch_size-1)//batch_size}")

    def upsert_sparse_vectors(self, actions: List[Dict[str, Any]], batch_size: int = 100):
        logger.info("Creating sparse vectors (BM25) ...")
        sparse_texts = [a["sparse_text"] for a in actions]
        encs = self.bm25.encode_documents(sparse_texts)  # list of {"indices":[...], "values":[...]}

        vectors = []
        for a, sv in zip(actions, encs):
            # Pure sparse vector upsert
            vectors.append(
                {
                    "id": f"s3::{a['action']}",
                    "sparse_values": sv,
                    "metadata": {
                        "service": "s3",
                        "action": a["action"],
                        "access": a["access_level"],
                    },
                }
            )

        logger.info(f"Upserting {len(vectors)} hybrid vectors (sparse + zero dense) ...")
        for i in range(0, len(vectors), batch_size):
            self.sparse_index.upsert(vectors=vectors[i : i + batch_size])
            logger.info(f"Sparse upsert batch {i//batch_size + 1}/{(len(vectors)+batch_size-1)//batch_size}")

    def save_bm25_encoder(self, output_path: str):
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"BM25 encoder saved to {output_path}")

    def build_indexes(self, input_file: str):
        logger.info("=== Building indexes ===")
        actions = self.load_data(input_file)
        self.ensure_indexes(actions)
        self.fit_bm25(actions)
        encoder_path = Path(__file__).parent / "bm25_encoder.pkl"
        self.save_bm25_encoder(str(encoder_path))
        self.upsert_dense_vectors(actions)
        self.upsert_sparse_vectors(actions)
        logger.info("=== Done ===")
        logger.info(f"Dense index: {self.dense_index_name} | Sparse index: {self.sparse_index_name}")


def main():
    p = argparse.ArgumentParser(description="Build IAM Actions RAG indexes")
    p.add_argument("--input", required=True, help="Path to enriched JSON")
    args = p.parse_args()

    builder = IndexBuilder()
    builder.build_indexes(args.input)


if __name__ == "__main__":
    main()