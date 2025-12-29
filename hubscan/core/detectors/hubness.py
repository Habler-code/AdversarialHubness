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
    ):
        """
        Initialize hubness detector.
        
        Args:
            enabled: Whether detector is enabled
            validate_exact: Whether to validate with exact search
            exact_validation_queries: Number of queries for exact validation
        """
        super().__init__(enabled)
        self.validate_exact = validate_exact
        self.exact_validation_queries = exact_validation_queries
    
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
        
        # Count hits per document
        hits = np.zeros(N, dtype=np.int32)
        example_queries: Dict[int, List[int]] = defaultdict(list)
        max_examples = 10  # Store up to 10 example queries per doc
        
        # Process queries in batches
        query_idx = 0
        for batch_queries in batch_iterator(list(range(M)), batch_size):
            batch_queries_array = queries[batch_queries]
            
            # Search
            distances, indices = index.search(batch_queries_array, k)
            
            # Count hits
            for i, query_id in enumerate(batch_queries):
                neighbors = indices[i]
                for neighbor_idx in neighbors:
                    if neighbor_idx < N:  # Valid index
                        hits[neighbor_idx] += 1
                        # Store example query
                        if len(example_queries[neighbor_idx]) < max_examples:
                            example_queries[neighbor_idx].append(query_id)
            
            query_idx += len(batch_queries)
            if query_idx % (batch_size * 10) == 0:
                logger.info(f"Processed {query_idx}/{M} queries")
        
        # Compute hub rate
        hub_rate = hits.astype(np.float32) / M
        
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
                    doc_embeddings, queries, k, hits, hub_rate, index.faiss_index
                )
            else:
                logger.warning(
                    "Exact validation requested but index is not FAISS. "
                    "Skipping exact validation."
                )
        
        # Prepare metadata
        result_metadata: Dict[str, Any] = {
            "hits": hits.tolist(),
            "hub_rate": hub_rate.tolist(),
            "hub_z": hub_z.tolist(),
            "example_queries": {str(k): v for k, v in example_queries.items()},
            "median": float(median),
            "mad": float(mad),
        }
        
        if validation_results:
            result_metadata["validation"] = validation_results
        
        return DetectorResult(scores=hub_z, metadata=result_metadata)
    
    def _validate_exact(
        self,
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        approx_hits: np.ndarray,
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
        
        # Exact search
        exact_hits = np.zeros(len(doc_embeddings), dtype=np.int32)
        _, exact_indices = exact_index.search(validation_queries, k)
        
        for neighbors in exact_indices:
            for neighbor_idx in neighbors:
                if neighbor_idx < len(doc_embeddings):
                    exact_hits[neighbor_idx] += 1
        
        exact_hub_rate = exact_hits.astype(np.float32) / num_validation
        
        # Compare top hubs
        top_k = 100
        approx_top = np.argsort(approx_hub_rate)[-top_k:][::-1]
        exact_top = np.argsort(exact_hub_rate)[-top_k:][::-1]
        
        # Compute overlap
        overlap = len(set(approx_top) & set(exact_top)) / top_k
        
        # Compute correlation
        correlation = np.corrcoef(approx_hub_rate, exact_hub_rate)[0, 1]
        
        logger.info(f"Exact validation: overlap={overlap:.3f}, correlation={correlation:.3f}")
        
        return {
            "num_validation_queries": num_validation,
            "overlap": float(overlap),
            "correlation": float(correlation),
            "exact_top_hubs": exact_top[:20].tolist(),
            "approx_top_hubs": approx_top[:20].tolist(),
        }

