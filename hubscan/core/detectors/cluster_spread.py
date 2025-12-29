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

"""Cluster spread detector - detects multi-cluster proximity."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import entropy
from typing import Optional, Dict, Any

from .base import Detector, DetectorResult
from ..io.metadata import Metadata
from ...utils.logging import get_logger

logger = get_logger()


class ClusterSpreadDetector(Detector):
    """Detector for cluster spread / multi-cluster proximity."""
    
    def __init__(
        self,
        enabled: bool = True,
        num_clusters: int = 1024,
        batch_size: int = 10000,
    ):
        """
        Initialize cluster spread detector.
        
        Args:
            enabled: Whether detector is enabled
            num_clusters: Number of clusters for k-means
            batch_size: Batch size for k-means
        """
        super().__init__(enabled)
        self.num_clusters = num_clusters
        self.batch_size = batch_size
    
    def detect(
        self,
        index: Any,
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        query_cluster_assignments: Optional[np.ndarray] = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Detect cluster spread by analyzing which query clusters retrieve each document.
        
        Args:
            index: FAISS index (not used directly here)
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            query_cluster_assignments: Optional pre-computed cluster assignments for queries
            
        Returns:
            DetectorResult with cluster spread scores
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        logger.info(f"Running cluster spread detection with {self.num_clusters} clusters")
        
        N = len(doc_embeddings)
        M = len(queries)
        
        # Cluster queries if assignments not provided
        if query_cluster_assignments is None:
            n_clusters = min(self.num_clusters, M // 10, N // 100)
            n_clusters = max(n_clusters, 2)
            
            logger.info(f"Clustering {M} queries into {n_clusters} clusters...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(self.batch_size, M),
                n_init=3,
            )
            query_cluster_assignments = kmeans.fit_predict(queries)
            num_clusters_actual = len(np.unique(query_cluster_assignments))
            logger.info(f"Created {num_clusters_actual} query clusters")
        else:
            num_clusters_actual = len(np.unique(query_cluster_assignments))
        
        # For each document, track which query clusters retrieve it
        doc_cluster_hits: Dict[int, Dict[int, int]] = {}
        
        # We need to run searches to see which clusters retrieve which docs
        # This is expensive, so we'll do it in batches
        batch_size = 2048
        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            batch_queries = queries[i:end]
            batch_clusters = query_cluster_assignments[i:end]
            
            distances, indices = index.search(batch_queries, k)
            
            for j, query_cluster in enumerate(batch_clusters):
                neighbors = indices[j]
                for neighbor_idx in neighbors:
                    if neighbor_idx < N:
                        if neighbor_idx not in doc_cluster_hits:
                            doc_cluster_hits[neighbor_idx] = {}
                        doc_cluster_hits[neighbor_idx][query_cluster] = (
                            doc_cluster_hits[neighbor_idx].get(query_cluster, 0) + 1
                        )
        
        # Compute cluster entropy for each document
        cluster_entropy = np.zeros(N)
        
        for doc_idx in range(N):
            if doc_idx in doc_cluster_hits:
                cluster_counts = list(doc_cluster_hits[doc_idx].values())
                # Normalize to probabilities
                total = sum(cluster_counts)
                if total > 0:
                    probs = np.array(cluster_counts) / total
                    cluster_entropy[doc_idx] = entropy(probs)
        
        # Normalize entropy (max entropy is log(num_clusters))
        max_entropy = np.log(num_clusters_actual) if num_clusters_actual > 1 else 1.0
        if max_entropy > 0:
            normalized_entropy = cluster_entropy / max_entropy
        else:
            normalized_entropy = cluster_entropy
        
        logger.info(f"Cluster spread detection complete. Mean entropy: {cluster_entropy.mean():.4f}")
        
        result_metadata: Dict[str, Any] = {
            "cluster_entropy": cluster_entropy.tolist(),
            "normalized_entropy": normalized_entropy.tolist(),
            "num_clusters": num_clusters_actual,
        }
        
        return DetectorResult(scores=normalized_entropy, metadata=result_metadata)

