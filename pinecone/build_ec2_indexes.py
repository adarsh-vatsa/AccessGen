#!/usr/bin/env python3
"""
Build EC2 Actions Pinecone Indexes
Creates dense and sparse indexes for EC2 actions with Gemini embeddings and BM25
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pinecone
from pinecone_text.sparse import BM25Encoder
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pickle


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DENSE_INDEX = "ec2-dense-v2"
DEFAULT_SPARSE_INDEX = "ec2-sparse-v2"
DEFAULT_DIM = 3072
GEMINI_MODEL = "models/gemini-embedding-001"


def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return vec
    return (arr / norm).tolist()


class EC2IndexBuilder:
    def __init__(self, dense_index: str = DEFAULT_DENSE_INDEX, sparse_index: str = DEFAULT_SPARSE_INDEX):
        self.dense_index_name = dense_index
        self.sparse_index_name = sparse_index
        self.desired_dim = DEFAULT_DIM
        
        self.pc = None
        self.dense_index = None
        self.sparse_index = None
        self.bm25 = BM25Encoder()
        
        self.embedding_model = GEMINI_MODEL
        self.genai_client = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Pinecone and Gemini clients"""
        load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
        
        required = ["PINECONE_API_KEY", "GEMINI_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            logger.error(f"Missing required env vars: {missing}")
            sys.exit(1)
        
        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("Initialized Pinecone + Gemini clients")
    
    def load_actions(self, input_path: str) -> List[Dict[str, Any]]:
        """Load EC2 actions from enriched extras JSON"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        actions = data["ec2"]["actions"]
        
        # Verify required fields
        required_fields = ["action", "access_level", "sparse_text", "dense_text"]
        for field in required_fields:
            if field not in actions[0]:
                logger.error(f"Input data missing required field: {field}")
                sys.exit(1)
        
        logger.info(f"Loaded {len(actions)} EC2 actions")
        return actions
    
    def get_embedding_dimension(self, sample_text: str) -> int:
        """Get the dimension of embeddings from Gemini"""
        try:
            cfg = types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.desired_dim,
            )
            res = self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=[sample_text],
                config=cfg,
            )
            dim = len(res.embeddings[0].values)
            logger.info(f"Gemini embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"Failed to probe embedding dim: {e} — using {self.desired_dim}")
            return self.desired_dim
    
    def create_or_update_indexes(self, dimension: int):
        """Create or update Pinecone indexes"""
        # Dense index
        try:
            desc = self.pc.describe_index(self.dense_index_name)
            logger.info(f"Dense index '{self.dense_index_name}' already exists")
        except:
            logger.info(f"Creating dense index '{self.dense_index_name}'...")
            self.pc.create_index(
                name=self.dense_index_name,
                dimension=dimension,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.dense_index_name).status.ready:
                time.sleep(2)
            logger.info(f"Dense index '{self.dense_index_name}' is ready")
        
        # Sparse index
        try:
            desc = self.pc.describe_index(self.sparse_index_name)
            logger.info(f"Sparse index '{self.sparse_index_name}' already exists")
        except:
            logger.info(f"Creating sparse index '{self.sparse_index_name}'...")
            self.pc.create_index(
                name=self.sparse_index_name,
                dimension=dimension,  # Dense dimension (will be zero vectors)
                metric="dotproduct",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.sparse_index_name).status.ready:
                time.sleep(2)
            logger.info(f"Sparse index '{self.sparse_index_name}' is ready")
        
        self.dense_index = self.pc.Index(self.dense_index_name)
        self.sparse_index = self.pc.Index(self.sparse_index_name)
        logger.info("Connected to indexes")
    
    def fit_bm25(self, actions: List[Dict[str, Any]]):
        """Fit BM25 encoder on the corpus"""
        logger.info("Fitting BM25 encoder...")
        corpus = [a["sparse_text"] for a in actions]
        self.bm25.fit(corpus)
        logger.info(f"BM25 encoder fitted on {len(corpus)} documents")
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get Gemini embeddings in batches"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                cfg = types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.desired_dim,
                )
                res = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch,
                    config=cfg,
                )
                for emb in res.embeddings:
                    embeddings.append(l2_normalize(emb.values))
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                # Add zero vectors for failed embeddings
                for _ in batch:
                    embeddings.append([0.0] * self.desired_dim)
        
        return embeddings
    
    def upsert_dense_vectors(self, actions: List[Dict[str, Any]], batch_size: int = 100):
        """Upsert dense vectors to Pinecone"""
        logger.info("Creating dense vectors (Gemini embeddings)...")
        dense_texts = [a["dense_text"] for a in actions]
        embeddings = self.get_embeddings_batch(dense_texts)
        
        vectors = []
        for action, embedding in zip(actions, embeddings):
            vectors.append({
                "id": f"ec2::{action['action']}",
                "values": embedding,
                "metadata": {
                    "service": "ec2",
                    "action": action["action"],
                    "access": action["access_level"],
                    "description": action.get("description", "")[:360],  # Truncate for metadata limits
                }
            })
        
        logger.info(f"Upserting {len(vectors)} dense vectors...")
        for i in range(0, len(vectors), batch_size):
            self.dense_index.upsert(vectors=vectors[i:i + batch_size])
            logger.info(f"Dense upsert batch {i//batch_size + 1}/{(len(vectors)+batch_size-1)//batch_size}")
    
    def upsert_sparse_vectors(self, actions: List[Dict[str, Any]], batch_size: int = 100):
        """Upsert sparse vectors to Pinecone"""
        logger.info("Creating sparse vectors (BM25)...")
        sparse_texts = [a["sparse_text"] for a in actions]
        sparse_vectors = self.bm25.encode_documents(sparse_texts)
        
        # Get dimension for zero dense vectors from index stats
        stats = self.sparse_index.describe_index_stats()
        dim = stats.get('dimension', self.desired_dim)
        
        vectors = []
        for action, sparse_vec in zip(actions, sparse_vectors):
            vectors.append({
                "id": f"ec2::{action['action']}",
                "values": [0.0] * dim,  # Zero dense vector
                "sparse_values": sparse_vec,
                "metadata": {
                    "service": "ec2",
                    "action": action["action"],
                    "access": action["access_level"],
                }
            })
        
        logger.info(f"Upserting {len(vectors)} sparse vectors...")
        for i in range(0, len(vectors), batch_size):
            self.sparse_index.upsert(vectors=vectors[i:i + batch_size])
            logger.info(f"Sparse upsert batch {i//batch_size + 1}/{(len(vectors)+batch_size-1)//batch_size}")
    
    def save_bm25_encoder(self, output_path: str):
        """Save the fitted BM25 encoder"""
        with open(output_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"BM25 encoder saved to {output_path}")
    
    def build_indexes(self, input_file: str):
        """Main build process"""
        logger.info("=== Building EC2 indexes ===")
        
        # Load data
        actions = self.load_actions(input_file)
        
        # Get embedding dimension
        sample_text = actions[0]["dense_text"]
        dimension = self.get_embedding_dimension(sample_text)
        
        # Create/connect to indexes
        self.create_or_update_indexes(dimension)
        
        # Fit BM25
        self.fit_bm25(actions)
        encoder_path = Path(__file__).parent / "bm25_encoder_ec2.pkl"
        self.save_bm25_encoder(str(encoder_path))
        
        # Upsert vectors
        self.upsert_dense_vectors(actions)
        self.upsert_sparse_vectors(actions)
        
        logger.info("=== EC2 indexes built successfully ===")
        logger.info(f"Dense index: {self.dense_index_name}")
        logger.info(f"Sparse index: {self.sparse_index_name}")
        logger.info(f"BM25 encoder: {encoder_path}")


def main():
    parser = argparse.ArgumentParser(description="Build EC2 Actions Pinecone indexes")
    parser.add_argument(
        "--input",
        default="../enriched_data/aws_iam_registry_ec2_enriched_extras.json",
        help="Path to EC2 enriched extras JSON"
    )
    parser.add_argument(
        "--dense-index",
        default=DEFAULT_DENSE_INDEX,
        help="Name for dense index"
    )
    parser.add_argument(
        "--sparse-index", 
        default=DEFAULT_SPARSE_INDEX,
        help="Name for sparse index"
    )
    
    args = parser.parse_args()
    
    builder = EC2IndexBuilder(
        dense_index=args.dense_index,
        sparse_index=args.sparse_index
    )
    builder.build_indexes(args.input)


if __name__ == "__main__":
    main()