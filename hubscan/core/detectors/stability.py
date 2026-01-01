# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Stability detector - detects stability under query perturbations."""

import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from .base import Detector, DetectorResult
from ..io.metadata import Metadata
from ..ranking import get_ranking_method
from ..reranking import get_reranking_method
from ...utils.metrics import normalize_vectors
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ..io.vector_index import VectorIndex

logger = get_logger()


class StabilityDetector(Detector):
    """Detector for stability under query perturbations."""
    
    def __init__(
        self,
        enabled: bool = True,
        candidates_top_x: int = 200,
        perturbations: int = 5,
        sigma: float = 0.01,
        normalize: bool = True,
    ):
        """
        Initialize stability detector.
        
        Args:
            enabled: Whether detector is enabled
            candidates_top_x: Top candidates to analyze
            perturbations: Number of perturbations per query
            sigma: Gaussian noise standard deviation
            normalize: Whether to renormalize after perturbation
        """
        super().__init__(enabled)
        self.candidates_top_x = candidates_top_x
        self.perturbations = perturbations
        self.sigma = sigma
        self.normalize = normalize
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        candidate_indices: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        ranking_method: str = "vector",
        hybrid_alpha: float = 0.5,
        query_texts: Optional[List[str]] = None,
        rerank: bool = False,
        rerank_method: Optional[str] = None,
        rerank_top_n: Optional[int] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Detect stability by perturbing queries and checking retrieval consistency.
        
        Note: Stability requires query embeddings to perturb, so it's not applicable
        for pure lexical search. For lexical ranking, this detector will be skipped.
        
        Args:
            index: VectorIndex instance (supports FAISS, Pinecone, Qdrant, Weaviate, etc.)
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            candidate_indices: Optional pre-selected candidate document indices
            seed: Random seed for reproducibility (default: 42)
            ranking_method: Ranking method ("vector", "hybrid", "lexical")
            hybrid_alpha: Weight for vector search in hybrid mode (0.0-1.0)
            query_texts: Optional query texts for lexical/hybrid search
            rerank: Whether to apply reranking as post-processing
            rerank_method: Reranking method name (if rerank=True)
            rerank_top_n: Number of candidates to retrieve before reranking (if rerank=True)
            rerank_params: Custom parameters for reranking method
            
        Returns:
            DetectorResult with stability scores
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        # Stability requires query embeddings to perturb, not applicable for pure lexical search
        if ranking_method == "lexical":
            logger.info("Stability detection skipped for lexical ranking (requires query embeddings to perturb)")
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        # Check for empty queries
        if len(queries) == 0:
            logger.warning("No queries available for stability detection")
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        logger.info(f"Running stability detection with {self.perturbations} perturbations per query")
        
        N = len(doc_embeddings)
        
        # Select candidates if not provided
        if candidate_indices is None:
            # Use all documents (can be expensive)
            candidate_indices = np.arange(N)
        
        # Limit to top candidates
        if len(candidate_indices) > self.candidates_top_x:
            candidate_indices = candidate_indices[:self.candidates_top_x]
        
        # Check for empty candidate set
        if len(candidate_indices) == 0:
            logger.warning("No candidate documents selected for stability detection")
            return DetectorResult(scores=np.zeros(N))
        
        logger.info(f"Analyzing stability for {len(candidate_indices)} candidate documents")
        
        # Get ranking method implementation
        ranking_method_impl = get_ranking_method(ranking_method)
        if ranking_method_impl is None:
            from ..ranking import list_ranking_methods
            available = list_ranking_methods()
            raise ValueError(
                f"Unknown ranking method: {ranking_method}. "
                f"Available methods: {', '.join(available)}"
            )
        
        # Get reranking method if enabled
        reranking_method_impl = None
        if rerank:
            if rerank_method is None:
                raise ValueError("rerank_method required when rerank=True")
            reranking_method_impl = get_reranking_method(rerank_method)
            if reranking_method_impl is None:
                from ..reranking import list_reranking_methods
                available = list_reranking_methods()
                raise ValueError(
                    f"Unknown reranking method: {rerank_method}. "
                    f"Available methods: {', '.join(available)}"
                )
        
        # Prepare ranking kwargs
        ranking_kwargs = kwargs.get("ranking_custom_params", {})
        if ranking_method == "hybrid":
            ranking_kwargs["alpha"] = hybrid_alpha
        
        # Determine retrieve k
        retrieve_k = rerank_top_n if rerank and rerank_top_n is not None else k
        
        # Initialize seeded RNG for reproducibility
        rng = np.random.default_rng(seed if seed is not None else 42)
        
        # Track stability per document
        stability_scores = np.zeros(N)
        doc_retrieval_counts = {idx: 0 for idx in candidate_indices}
        
        # Process queries (can sample subset for efficiency)
        # For now, use all queries but this could be limited
        num_queries_to_use = min(len(queries), 1000)  # Limit for performance
        query_indices = rng.choice(len(queries), num_queries_to_use, replace=False)
        
        for query_idx in query_indices:
            query = queries[query_idx:query_idx+1]
            
            # Generate perturbations
            for _ in range(self.perturbations):
                # Add Gaussian noise using seeded RNG
                perturbed = query + rng.normal(0, self.sigma, query.shape).astype(np.float32)
                
                # Renormalize if needed (for cosine/IP)
                if self.normalize:
                    perturbed = normalize_vectors(perturbed)
                
                # Use ranking method plugin
                batch_query_texts = None
                if query_texts is not None:
                    batch_query_texts = [query_texts[query_idx]]
                
                distances, indices, _ = ranking_method_impl.search(
                    index=index,
                    query_vectors=perturbed,
                    query_texts=batch_query_texts,
                    k=retrieve_k,
                    **ranking_kwargs
                )
                
                # Apply reranking if enabled
                if rerank and reranking_method_impl is not None:
                    rerank_kwargs = rerank_params or {}
                    distances, indices, _ = reranking_method_impl.rerank(
                        distances=distances,
                        indices=indices,
                        query_vectors=perturbed,
                        query_texts=batch_query_texts,
                        k=k,
                        **rerank_kwargs
                    )
                
                neighbors = indices[0]
                
                # Count retrievals for candidate documents
                for neighbor_idx in neighbors:
                    if neighbor_idx in doc_retrieval_counts:
                        doc_retrieval_counts[neighbor_idx] += 1
        
        # Compute stability scores
        # Normalize by maximum possible count: num_queries * perturbations * k
        # This ensures scores are in [0, 1] and represent the fraction of possible retrieval slots
        max_possible_count = num_queries_to_use * self.perturbations * k
        if max_possible_count > 0:
            for doc_idx, count in doc_retrieval_counts.items():
                stability_scores[doc_idx] = count / max_possible_count
        
        logger.info(f"Stability detection complete. Mean stability: {stability_scores[candidate_indices].mean():.4f}")
        
        result_metadata: Dict[str, Any] = {
            "stability_scores": stability_scores.tolist(),
            "num_candidates": len(candidate_indices),
            "num_queries_used": num_queries_to_use,
            "max_possible_count": max_possible_count,
            "ranking_method": ranking_method,
        }
        
        return DetectorResult(scores=stability_scores, metadata=result_metadata)

