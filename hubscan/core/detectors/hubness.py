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

"""Hubness detector - primary detector for reverse-kNN frequency."""

import numpy as np
import faiss
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from collections import defaultdict

from .base import Detector, DetectorResult
from ..io.metadata import Metadata
from ..io.adapters import FAISSIndex
from ...utils.metrics import robust_zscore, compute_percentile
from ...utils.batching import batch_iterator
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ..io.vector_index import VectorIndex

logger = get_logger()


class HubnessDetector(Detector):
    """Detector for hubness using reverse-kNN frequency."""
    
    def __init__(
        self,
        enabled: bool = True,
        validate_exact: bool = False,
        exact_validation_queries: Optional[int] = None,
        use_rank_weights: bool = True,
        use_distance_weights: bool = True,
        metric: str = "cosine",
    ):
        """
        Initialize hubness detector.
        
        Args:
            enabled: Whether detector is enabled
            validate_exact: Whether to validate with exact search
            exact_validation_queries: Number of queries for exact validation
            use_rank_weights: Whether to weight hits by rank position (rank 1 > rank k)
            use_distance_weights: Whether to weight hits by similarity/distance scores
            metric: Distance metric ("cosine", "ip", "l2")
        """
        super().__init__(enabled)
        self.validate_exact = validate_exact
        self.exact_validation_queries = exact_validation_queries
        self.use_rank_weights = use_rank_weights
        self.use_distance_weights = use_distance_weights
        self.metric = metric
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        batch_size: int = 2048,
        **kwargs,
    ) -> DetectorResult:
        """
        Detect hubness by counting reverse-kNN frequency.
        
        Args:
            index: VectorIndex instance (supports FAISS, Pinecone, Qdrant, Weaviate, etc.)
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            batch_size: Batch size for processing queries
            
        Returns:
            DetectorResult with hubness scores and example queries
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        logger.info(f"Running hubness detection on {len(queries)} queries with k={k}")
        
        N = len(doc_embeddings)
        M = len(queries)
        
        # Check for empty queries
        if M == 0:
            logger.warning("No queries provided for hubness detection")
            return DetectorResult(scores=np.zeros(N))
        
        # Initialize rank weights if using rank-aware scoring
        if self.use_rank_weights:
            # Weight by inverse rank (rank 1 gets weight 1.0, rank k gets weight 1/k)
            rank_weights = 1.0 / np.arange(1, k + 1, dtype=np.float32)
        else:
            rank_weights = np.ones(k, dtype=np.float32)
        
        # Determine if metric is similarity-based (higher is better) or distance-based (lower is better)
        is_similarity_metric = self.metric in ["cosine", "ip"]
        
        # Count weighted hits per document
        weighted_hits = np.zeros(N, dtype=np.float32)
        example_queries: Dict[int, List[int]] = defaultdict(list)
        max_examples = 10  # Store up to 10 example queries per doc
        
        # Process queries in batches
        query_idx = 0
        for batch_queries in batch_iterator(list(range(M)), batch_size):
            batch_queries_array = queries[batch_queries]
            
            # Search
            distances, indices = index.search(batch_queries_array, k)
            
            # Count weighted hits
            for i, query_id in enumerate(batch_queries):
                neighbors = indices[i]
                neighbor_distances = distances[i]
                
                for rank_pos, (neighbor_idx, dist) in enumerate(zip(neighbors, neighbor_distances)):
                    if neighbor_idx < N:  # Valid index
                        # Start with rank weight
                        weight = rank_weights[rank_pos]
                        
                        # Apply distance/similarity weight if enabled
                        if self.use_distance_weights:
                            if is_similarity_metric:
                                # For cosine/IP: higher similarity = more suspicious
                                # Normalize to [0, 1] range (assuming values are typically in [0, 1] for cosine/IP)
                                # For IP, values can be negative, so we clip and normalize
                                if self.metric == "cosine":
                                    # Cosine similarity is typically [0, 1] after normalization
                                    similarity_weight = np.clip(dist, 0.0, 1.0)
                                else:  # IP
                                    # IP can be negative, normalize to [0, 1] using sigmoid-like function
                                    # Use tanh to map to [0, 1] range
                                    similarity_weight = (np.tanh(dist) + 1.0) / 2.0
                                weight *= similarity_weight
                            else:  # L2
                                # For L2: lower distance = more suspicious
                                # Convert distance to similarity-like score (inverse, normalized)
                                # Use exp(-distance) to convert distance to similarity
                                # Normalize by typical distance range (use max distance in batch as reference)
                                max_dist_in_batch = np.max(neighbor_distances[neighbor_distances < np.inf])
                                if max_dist_in_batch > 0:
                                    similarity_weight = np.exp(-dist / (max_dist_in_batch + 1e-6))
                                else:
                                    similarity_weight = 1.0 if dist < 1e-6 else 0.0
                                weight *= similarity_weight
                        
                        weighted_hits[neighbor_idx] += weight
                        
                        # Store example query
                        if len(example_queries[neighbor_idx]) < max_examples:
                            example_queries[neighbor_idx].append(query_id)
            
            query_idx += len(batch_queries)
            if query_idx % (batch_size * 10) == 0:
                logger.info(f"Processed {query_idx}/{M} queries")
        
        # Compute weighted hub rate
        # Normalize by sum of weights for a single query to get rate
        max_weight_per_query = np.sum(rank_weights) if self.use_rank_weights else float(k)
        hub_rate = weighted_hits / (M * max_weight_per_query)
        
        # Compute robust z-score
        hub_z, median, mad = robust_zscore(hub_rate)
        
        logger.info(f"Hubness detection complete. Median hub_rate: {median:.6f}, MAD: {mad:.6f}")
        logger.info(f"Max hub_rate: {hub_rate.max():.6f}, Max hub_z: {hub_z.max():.2f}")
        
        # Optional exact validation (only for FAISS)
        validation_results = None
        if self.validate_exact:
            # Check if we have a FAISS index for exact validation
            if hasattr(index, 'faiss_index'):
                validation_results = self._validate_exact(
                    doc_embeddings, queries, k, weighted_hits, hub_rate, index.faiss_index
                )
            else:
                logger.warning(
                    "Exact validation requested but index is not FAISS. "
                    "Skipping exact validation."
                )
        
        # Prepare metadata
        result_metadata: Dict[str, Any] = {
            "weighted_hits": weighted_hits.tolist(),
            "hub_rate": hub_rate.tolist(),
            "hub_z": hub_z.tolist(),
            "example_queries": {str(k): v for k, v in example_queries.items()},
            "median": float(median),
            "mad": float(mad),
            "use_rank_weights": self.use_rank_weights,
            "use_distance_weights": self.use_distance_weights,
            "metric": self.metric,
        }
        
        if validation_results:
            result_metadata["validation"] = validation_results
        
        return DetectorResult(scores=hub_z, metadata=result_metadata)
    
    def _validate_exact(
        self,
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        approx_weighted_hits: np.ndarray,
        approx_hub_rate: np.ndarray,
        faiss_index: faiss.Index,
    ) -> Dict[str, Any]:
        """Validate approximate results with exact search."""
        logger.info("Running exact validation...")
        
        # Build exact index
        exact_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        exact_index.add(doc_embeddings)
        
        # Sample queries for validation
        num_validation = self.exact_validation_queries or min(1000, len(queries))
        rng = np.random.default_rng(42)
        validation_indices = rng.choice(len(queries), num_validation, replace=False)
        validation_queries = queries[validation_indices]
        
        # Initialize rank weights for validation
        if self.use_rank_weights:
            rank_weights = 1.0 / np.arange(1, k + 1, dtype=np.float32)
        else:
            rank_weights = np.ones(k, dtype=np.float32)
        max_weight_per_query = np.sum(rank_weights) if self.use_rank_weights else float(k)
        
        is_similarity_metric = self.metric in ["cosine", "ip"]
        
        # Exact search
        exact_weighted_hits = np.zeros(len(doc_embeddings), dtype=np.float32)
        exact_distances, exact_indices = exact_index.search(validation_queries, k)
        
        for i, (neighbors, dists) in enumerate(zip(exact_indices, exact_distances)):
            for rank_pos, (neighbor_idx, dist) in enumerate(zip(neighbors, dists)):
                if neighbor_idx < len(doc_embeddings):
                    weight = rank_weights[rank_pos]
                    if self.use_distance_weights:
                        if is_similarity_metric:
                            if self.metric == "cosine":
                                similarity_weight = np.clip(dist, 0.0, 1.0)
                            else:  # IP
                                similarity_weight = (np.tanh(dist) + 1.0) / 2.0
                            weight *= similarity_weight
                        else:  # L2
                            max_dist = np.max(dists[dists < np.inf])
                            if max_dist > 0:
                                similarity_weight = np.exp(-dist / (max_dist + 1e-6))
                            else:
                                similarity_weight = 1.0 if dist < 1e-6 else 0.0
                            weight *= similarity_weight
                    exact_weighted_hits[neighbor_idx] += weight
        
        # Calculate exact hub_rate using same validation queries for fair comparison
        exact_hub_rate = exact_weighted_hits / (num_validation * max_weight_per_query)
        
        # Calculate approximate hub_rate using same validation queries for fair comparison
        validation_approx_weighted_hits = np.zeros(len(doc_embeddings), dtype=np.float32)
        validation_approx_queries = queries[validation_indices]
        validation_approx_distances, validation_approx_indices = faiss_index.search(validation_approx_queries, k)
        
        for i, (neighbors, dists) in enumerate(zip(validation_approx_indices, validation_approx_distances)):
            for rank_pos, (neighbor_idx, dist) in enumerate(zip(neighbors, dists)):
                if neighbor_idx < len(doc_embeddings):
                    weight = rank_weights[rank_pos]
                    if self.use_distance_weights:
                        if is_similarity_metric:
                            if self.metric == "cosine":
                                similarity_weight = np.clip(dist, 0.0, 1.0)
                            else:  # IP
                                similarity_weight = (np.tanh(dist) + 1.0) / 2.0
                            weight *= similarity_weight
                        else:  # L2
                            max_dist = np.max(dists[dists < np.inf])
                            if max_dist > 0:
                                similarity_weight = np.exp(-dist / (max_dist + 1e-6))
                            else:
                                similarity_weight = 1.0 if dist < 1e-6 else 0.0
                            weight *= similarity_weight
                    validation_approx_weighted_hits[neighbor_idx] += weight
        
        validation_approx_hub_rate = validation_approx_weighted_hits / (num_validation * max_weight_per_query)
        
        # Compare top hubs
        top_k = 100
        approx_top = np.argsort(validation_approx_hub_rate)[-top_k:][::-1]
        exact_top = np.argsort(exact_hub_rate)[-top_k:][::-1]
        
        # Compute overlap
        overlap = len(set(approx_top) & set(exact_top)) / top_k
        
        # Compute correlation
        correlation = np.corrcoef(validation_approx_hub_rate, exact_hub_rate)[0, 1]
        
        logger.info(f"Exact validation: overlap={overlap:.3f}, correlation={correlation:.3f}")
        
        return {
            "num_validation_queries": num_validation,
            "overlap": float(overlap),
            "correlation": float(correlation),
            "exact_top_hubs": exact_top[:20].tolist(),
            "approx_top_hubs": approx_top[:20].tolist(),
            "note": "Validation compares approximate vs exact search using the same validation query subset",
        }

