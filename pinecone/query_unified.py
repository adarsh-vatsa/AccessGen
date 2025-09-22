#!/usr/bin/env python3
"""
Unified Query Engine for AWS Actions
Searches across S3, EC2, and IAM actions with optional service filtering
"""
import json
import os
import sys
import re
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pickle

# Third-party imports
import pinecone
from pinecone_text.sparse import BM25Encoder
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from service_router import ServiceRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regex for crude CamelCase action discovery
ACTION_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*)\b")


class UnifiedQueryEngine:
    def __init__(self, 
                 dense_index_name: str = "aws-actions-dense-v2",
                 sparse_index_name: str = "aws-actions-sparse-v2",
                 bm25_path: str = None,
                 use_service_router: bool = True):
        """
        Initialize the unified query engine
        
        Args:
            dense_index_name: Name of dense Pinecone index
            sparse_index_name: Name of sparse Pinecone index
            bm25_path: Path to BM25 encoder pickle
            use_service_router: Whether to use service identification
        """
        self.pc = None
        self.bm25: BM25Encoder = None
        self.dense_index = None
        self.sparse_index = None
        self.use_service_router = use_service_router
        
        # Service router for identifying services and expanding queries
        if use_service_router:
            self.router = ServiceRouter()
        else:
            self.router = None

        # Gemini
        self.embedding_model = "models/gemini-embedding-001"
        self.embedding_dimension = 3072

        # Lookup for rerank text
        self.actions_lookup: Dict[str, Dict[str, Any]] = {}

        # Index names
        self.dense_index_name = dense_index_name
        self.sparse_index_name = sparse_index_name

        # BM25 encoder path
        self.bm25_path_override = bm25_path or os.getenv("PINECONE_BM25_PATH")

        # Reranker model
        self.rerank_model = os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")

        self._load_env()
        self._load_bm25_encoder()
        self._connect_to_indexes()
    
    def _load_env(self):
        """Load environment variables"""
        load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
        required = ["PINECONE_API_KEY", "GEMINI_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            sys.exit(1)

        self.pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info(f"Using Pinecone hosted reranker ({self.rerank_model})")
    
    def _load_bm25_encoder(self):
        """Load the BM25 encoder"""
        if self.bm25_path_override:
            enc_path = Path(self.bm25_path_override)
        else:
            # Try unified encoder first, then fall back to others
            enc_unified = Path(__file__).parent / "bm25_encoder_unified.pkl"
            enc_v2 = Path(__file__).parent / "bm25_encoder_v2.pkl"
            enc_v1 = Path(__file__).parent / "bm25_encoder.pkl"
            
            if enc_unified.exists():
                enc_path = enc_unified
            elif enc_v2.exists():
                enc_path = enc_v2
            else:
                enc_path = enc_v1
        
        if not enc_path.exists():
            logger.error(f"BM25 encoder not found at {enc_path}. Run build_unified_indexes.py first.")
            sys.exit(1)
        
        with open(enc_path, "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"BM25 encoder loaded from {enc_path}")
    
    def _connect_to_indexes(self):
        """Connect to Pinecone indexes"""
        try:
            self.dense_index = self.pc.Index(self.dense_index_name)
            self.sparse_index = self.pc.Index(self.sparse_index_name)
            logger.info(f"Connected to indexes: {self.dense_index_name}, {self.sparse_index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone indexes: {e}")
            sys.exit(1)
    
    @staticmethod
    def l2_normalize(vec: List[float]) -> List[float]:
        """L2 normalize a vector"""
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return vec
        return (arr / norm).tolist()
    
    def get_query_embedding(self, query_text: str) -> List[float]:
        """Get Gemini query embedding"""
        try:
            cfg = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.embedding_dimension,
            )
            res = self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=[query_text],
                config=cfg,
            )
            return self.l2_normalize(res.embeddings[0].values)
        except Exception as e:
            logger.error(f"Gemini query embedding failed: {e}")
            return [0.0] * self.embedding_dimension
    
    def extract_explicit_actions(self, query_text: str, services: List[str] = None) -> List[str]:
        """
        Extract explicit action names from query
        
        Args:
            query_text: Query text
            services: Optional list of services to filter by
        """
        raw = ACTION_RE.findall(query_text)
        found = []
        have_ids = set(self.actions_lookup.keys())
        
        for token in raw:
            # Check all services or specified ones
            if services and "all" not in services:
                # Check only specified services
                for service in services:
                    doc_id = f"{service}::{token}"
                    if doc_id in have_ids and doc_id not in found:
                        found.append(doc_id)
            else:
                # Check all services
                for service in ['s3', 'ec2', 'iam']:
                    doc_id = f"{service}::{token}"
                    if doc_id in have_ids and doc_id not in found:
                        found.append(doc_id)
        
        return found
    
    def get_sparse_query_vector(self, query_text: str) -> Dict[str, List[float]]:
        """Get BM25 sparse query vector with conditional expansion"""
        q = query_text
        # Add verb expansions for better recall
        if any(w in query_text.lower() for w in ["delete", "remove", "purge", "erase", "destroy"]):
            q += " delete remove purge erase destroy terminate"
        if any(w in query_text.lower() for w in ["list", "show", "enumerate", "browse"]):
            q += " list show enumerate describe browse"
        if any(w in query_text.lower() for w in ["create", "make", "new", "provision"]):
            q += " create make new provision launch run"
        return self.bm25.encode_queries([q])[0]
    
    def retrieve_dense(self, query_embedding: List[float], top_k: int = 200, 
                      service_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve from dense index with optional service filtering"""
        try:
            # Build metadata filter if services specified
            filter_dict = None
            if service_filter and "all" not in service_filter:
                if len(service_filter) == 1:
                    filter_dict = {"service": {"$eq": service_filter[0]}}
                else:
                    filter_dict = {"service": {"$in": service_filter}}
            
            return self.dense_index.query(
                vector=query_embedding, 
                top_k=top_k, 
                include_metadata=True,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return {"matches": []}
    
    def retrieve_sparse(self, sparse_vector: Dict[str, List[float]], top_k: int = 150,
                       service_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve from sparse index with optional service filtering"""
        try:
            # Build metadata filter if services specified
            filter_dict = None
            if service_filter and "all" not in service_filter:
                if len(service_filter) == 1:
                    filter_dict = {"service": {"$eq": service_filter[0]}}
                else:
                    filter_dict = {"service": {"$in": service_filter}}
            
            # Sparse index needs a dummy dense vector
            dummy_dense = [0.0] * self.embedding_dimension
            dummy_dense[0] = 1e-10  # Minimal non-zero value
            
            return self.sparse_index.query(
                vector=dummy_dense,  # Dummy dense vector
                sparse_vector=sparse_vector, 
                top_k=top_k, 
                include_metadata=True,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return {"matches": []}
    
    @staticmethod
    def compute_rrf_scores(dense_results: Dict[str, Any], sparse_results: Dict[str, Any], k: int = 60) -> List[Tuple[str, float]]:
        """Compute Reciprocal Rank Fusion scores"""
        # Rank dicts (1-based)
        dense_ranks = {m["id"]: i + 1 for i, m in enumerate(dense_results["matches"])}
        sparse_ranks = {m["id"]: i + 1 for i, m in enumerate(sparse_results["matches"])}

        scores: Dict[str, float] = {}
        for vid, r in dense_ranks.items():
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + r)
        for vid, r in sparse_ranks.items():
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + r)

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    
    def build_rerank_text(self, row: Dict[str, Any]) -> str:
        """Build text for reranking"""
        action = row["action"]
        service = row.get("service", "unknown")
        desc = (row.get("description") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        rtypes = [rt.get("type", "") for rt in row.get("resource_types", [])]
        keys = row.get("condition_keys", [])
        
        # Include service in rerank text for better context
        return f"{service}:{action} — {desc} | resources=[{self._short(rtypes)}] | keys=[{self._short(keys, 5)}]"
    
    @staticmethod
    def _short(list_vals: List[str], n: int = 3) -> str:
        """Shorten a list for display"""
        if not list_vals:
            return ""
        s = ", ".join(list_vals[:n])
        if len(list_vals) > n:
            s += ", ..."
        return s
    
    def rerank_with_hosted(self, query_text: str, candidates: List[str], top_k: int = 30) -> List[Dict[str, Any]]:
        """Use Pinecone hosted reranker"""
        try:
            docs = [{"text": c} for c in candidates]
            resp = self.pc.inference.rerank(
                model=self.rerank_model,
                query=query_text,
                documents=docs,
                top_n=min(top_k, len(docs)),
                rank_fields=["text"],
            )
            results = []
            for item in (resp.data or []):
                results.append({"index": item.index, "relevance_score": item.score})
            return results
        except Exception as e:
            logger.warning(f"Hosted reranker failed, falling back to RRF order only: {e}")
            return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(min(top_k, len(candidates)))]
    
    def load_unified_actions_data(self, s3_path: str = None, ec2_path: str = None, iam_path: str = None):
        """Load actions data from all services"""
        # Default paths
        if not s3_path:
            s3_path = str(Path(__file__).parent.parent / "enriched_data" / "aws_iam_registry_s3_enriched_extras.json")
        if not ec2_path:
            ec2_path = str(Path(__file__).parent.parent / "enriched_data" / "aws_iam_registry_ec2_enriched_extras.json")
        if not iam_path:
            iam_path = str(Path(__file__).parent.parent / "enriched_data" / "aws_iam_registry_iam_enriched_extras.json")
        
        self.actions_lookup = {}
        total = 0
        
        # Load S3
        if Path(s3_path).exists():
            with open(s3_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for action in data["s3"]["actions"]:
                action["service"] = "s3"
                self.actions_lookup[f"s3::{action['action']}"] = action
                total += 1
        
        # Load EC2
        if Path(ec2_path).exists():
            with open(ec2_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for action in data["ec2"]["actions"]:
                action["service"] = "ec2"
                self.actions_lookup[f"ec2::{action['action']}"] = action
                total += 1
        
        # Load IAM
        if Path(iam_path).exists():
            with open(iam_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for action in data["iam"]["actions"]:
                action["service"] = "iam"
                self.actions_lookup[f"iam::{action['action']}"] = action
                total += 1
        
        logger.info(f"Loaded {total} actions for lookup (S3, EC2, IAM)")
    
    def unified_search(
        self,
        query_text: str,
        services: Optional[List[str]] = None,
        top_k_dense: int = 200,
        top_k_sparse: int = 150,
        rrf_k: int = 60,
        top_m: int = 30,
        use_query_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform unified search across all services
        
        Args:
            query_text: Natural language query
            services: Optional list of services to filter ['s3', 'ec2', 'iam'] or ['all']
            top_k_dense: Number of results from dense search
            top_k_sparse: Number of results from sparse search
            rrf_k: RRF parameter
            top_m: Final number of results after reranking
            use_query_expansion: Whether to expand the query
        """
        logger.info(f"Unified search: {query_text!r}")
        
        # Identify services and expand query if router is enabled
        if self.use_service_router and use_query_expansion:
            if services is None:
                services, expanded_query = self.router.route_and_expand(query_text)
            else:
                expanded_query = self.router.expand_query(query_text, services)
            logger.info(f"Services: {services}, Expanded: {expanded_query}")
        else:
            expanded_query = query_text
            if services is None:
                services = ["all"]
        
        # Encode queries
        q_dense = self.get_query_embedding(expanded_query)
        q_sparse = self.get_sparse_query_vector(expanded_query)
        
        # Retrieve with optional service filtering
        service_filter = services if services != ["all"] else None
        dense_res = self.retrieve_dense(q_dense, top_k_dense, service_filter)
        sparse_res = self.retrieve_sparse(q_sparse, top_k_sparse, service_filter)
        logger.info(f"Dense={len(dense_res['matches'])} Sparse={len(sparse_res['matches'])}")
        
        # RRF fusion
        fused = self.compute_rrf_scores(dense_res, sparse_res, rrf_k)
        fused_ids = [doc_id for doc_id, _ in fused]
        
        # Extract explicit actions
        explicit = self.extract_explicit_actions(query_text, services)
        
        # Build rerank candidates
        candidates, candidate_ids = [], []
        
        def add_candidate(doc_id: str):
            row = self.actions_lookup.get(doc_id)
            if row:
                candidates.append(self.build_rerank_text(row))
                candidate_ids.append(doc_id)
        
        # Add explicit actions first
        for eid in explicit:
            if eid not in candidate_ids:
                add_candidate(eid)
        
        # Add fused results
        for doc_id in fused_ids:
            if doc_id not in candidate_ids:
                add_candidate(doc_id)
        
        # Cap candidates
        max_docs = int(os.getenv("PINECONE_RERANK_MAX_DOCS", "100"))
        if len(candidates) > max_docs:
            candidates = candidates[:max_docs]
            candidate_ids = candidate_ids[:max_docs]
        
        # Elastic top_m if explicit actions present
        if explicit:
            top_m = min(top_m, max(8, len(explicit) + 6))
        
        # Rerank
        reranked = self.rerank_with_hosted(expanded_query, candidates, top_m)
        
        # Assemble final results
        final = []
        rrf_map = dict(fused)
        for item in reranked:
            idx = item["index"]
            if idx < 0 or idx >= len(candidate_ids):
                continue
            doc_id = candidate_ids[idx]
            row = self.actions_lookup[doc_id]
            final.append({
                "id": doc_id,
                "service": row["service"],
                "action": row["action"],
                "access_level": row["access_level"],
                "description": row.get("description", ""),
                "resource_types": [rt.get("type") for rt in row.get("resource_types", [])],
                "condition_keys": row.get("condition_keys", []),
                "rerank_score": item["relevance_score"],
                "rrf_score": rrf_map.get(doc_id, 0.0),
            })
        
        logger.info(f"Final results: {len(final)}")
        return final
    
    @staticmethod
    def format_results(results: List[Dict[str, Any]], show_details: bool = False) -> str:
        """Format results for display"""
        if not results:
            return "No results found."
        
        lines = [f"Found {len(results)} relevant AWS actions:\n"]
        
        # Group by service
        by_service = {}
        for r in results:
            service = r["service"]
            if service not in by_service:
                by_service[service] = []
            by_service[service].append(r)
        
        # Display grouped results
        for service in sorted(by_service.keys()):
            lines.append(f"\n{service.upper()} Actions:")
            for i, r in enumerate(by_service[service], 1):
                lines.append(f"  {i}. {r['action']} (score: {r['rerank_score']:.3f})")
                lines.append(f"     {r['description']}")
                if show_details:
                    res = ", ".join(r["resource_types"][:3]) + ("..." if len(r["resource_types"]) > 3 else "")
                    keys = ", ".join(r["condition_keys"][:5]) + ("..." if len(r["condition_keys"]) > 5 else "")
                    lines.append(f"     Resources: [{res}]")
                    lines.append(f"     Condition Keys: [{keys}]")
        
        return "\n".join(lines)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Query Unified AWS Actions")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--services", nargs="+", choices=["s3", "ec2", "iam", "all"],
                       help="Filter by specific services")
    parser.add_argument("--top-k", type=int, default=30, help="Number of results")
    parser.add_argument("--details", action="store_true", help="Show detailed results")
    parser.add_argument("--no-router", action="store_true", help="Disable service router")
    parser.add_argument("--no-expansion", action="store_true", help="Disable query expansion")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = UnifiedQueryEngine(use_service_router=not args.no_router)
    engine.load_unified_actions_data()
    
    # Search
    results = engine.unified_search(
        args.query,
        services=args.services,
        top_m=args.top_k,
        use_query_expansion=not args.no_expansion
    )
    
    # Display results
    print(engine.format_results(results, args.details))


if __name__ == "__main__":
    main()