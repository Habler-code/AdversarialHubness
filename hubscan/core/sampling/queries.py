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

"""Query sampling strategies."""

import numpy as np
from typing import Optional, List, Dict
from sklearn.cluster import MiniBatchKMeans

from ...config import ScanConfig
from ..io.metadata import Metadata


class QuerySampler:
    """Query sampler for various strategies."""
    
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        config: ScanConfig,
        metadata: Optional[Metadata] = None,
    ):
        """
        Initialize query sampler.
        
        Args:
            doc_embeddings: Document embeddings (N, D)
            config: Scan configuration
            metadata: Optional metadata for stratified sampling
        """
        self.doc_embeddings = doc_embeddings
        self.config = config
        self.metadata = metadata
        self.rng = np.random.default_rng(config.seed)
    
    def sample(self) -> np.ndarray:
        """Sample queries according to configured strategy."""
        strategy = self.config.query_sampling
        
        if strategy == "real_queries":
            return self._sample_real_queries()
        elif strategy == "random_docs_as_queries":
            return self._sample_random_docs()
        elif strategy == "cluster_centroids":
            return self._sample_cluster_centroids()
        elif strategy == "mixed":
            return self._sample_mixed()
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def _sample_real_queries(self) -> np.ndarray:
        """Load real query embeddings from file."""
        if not self.config.query_embeddings_path:
            raise ValueError("query_embeddings_path required for real_queries strategy")
        
        from ..io.embeddings import load_embeddings
        
        queries = load_embeddings(self.config.query_embeddings_path)
        
        # Sample if we have more than needed
        if len(queries) > self.config.num_queries:
            indices = self.rng.choice(len(queries), self.config.num_queries, replace=False)
            queries = queries[indices]
        
        return queries
    
    def _sample_random_docs(self) -> np.ndarray:
        """Sample random document embeddings as queries."""
        n = len(self.doc_embeddings)
        num_queries = self.config.num_queries
        
        if self.config.stratified_by and self.metadata:
            # Stratified sampling
            return self._stratified_sample(num_queries)
        
        # Simple random sampling
        indices = self.rng.choice(n, min(num_queries, n), replace=False)
        return self.doc_embeddings[indices]
    
    def _stratified_sample(self, num_queries: int) -> np.ndarray:
        """Stratified sampling by metadata field."""
        if not self.metadata or not self.config.stratified_by:
            return self._sample_random_docs()
        
        field = self.config.stratified_by
        if not self.metadata.has_field(field):
            return self._sample_random_docs()
        
        # Group by field value
        field_values = self.metadata.get(field)
        unique_values = list(set(field_values))
        
        # Sample proportionally from each group
        queries = []
        for value in unique_values:
            group_indices = [i for i, v in enumerate(field_values) if v == value]
            if not group_indices:
                continue
            
            # Sample proportionally
            group_size = len(group_indices)
            sample_size = max(1, int(num_queries * group_size / len(field_values)))
            sample_size = min(sample_size, group_size)
            
            sampled_indices = self.rng.choice(group_indices, sample_size, replace=False)
            queries.append(self.doc_embeddings[sampled_indices])
        
        result = np.vstack(queries)
        
        # Trim or pad to exact number
        if len(result) > num_queries:
            indices = self.rng.choice(len(result), num_queries, replace=False)
            result = result[indices]
        elif len(result) < num_queries:
            # Sample more randomly to fill
            additional = num_queries - len(result)
            all_indices = list(range(len(self.doc_embeddings)))
            remaining = [i for i in all_indices if i not in [idx for q in queries for idx in range(len(q))]]
            if remaining:
                add_indices = self.rng.choice(remaining, min(additional, len(remaining)), replace=False)
                result = np.vstack([result, self.doc_embeddings[add_indices]])
        
        return result[:num_queries]
    
    def _sample_cluster_centroids(self) -> np.ndarray:
        """Sample cluster centroids as queries."""
        # Use k-means to cluster documents
        n_clusters = min(self.config.num_queries, len(self.doc_embeddings) // 10)
        n_clusters = max(n_clusters, 2)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.config.seed,
            batch_size=min(1000, len(self.doc_embeddings)),
            n_init=3,
        )
        kmeans.fit(self.doc_embeddings)
        
        centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # If we need more queries, sample additional random docs
        if len(centroids) < self.config.num_queries:
            additional = self.config.num_queries - len(centroids)
            n = len(self.doc_embeddings)
            indices = self.rng.choice(n, min(additional, n), replace=False)
            additional_queries = self.doc_embeddings[indices]
            centroids = np.vstack([centroids, additional_queries])
        
        return centroids[:self.config.num_queries]
    
    def _sample_mixed(self) -> np.ndarray:
        """Mix multiple sampling strategies."""
        proportions = self.config.mixed_proportions
        total = sum(proportions.values())
        
        if total == 0:
            # Default to random if no proportions specified
            return self._sample_random_docs()
        
        queries_list = []
        
        # Real queries
        if proportions.get("real_queries", 0) > 0:
            try:
                real_queries = self._sample_real_queries()
                n_real = int(self.config.num_queries * proportions["real_queries"] / total)
                if len(real_queries) >= n_real:
                    indices = self.rng.choice(len(real_queries), n_real, replace=False)
                    queries_list.append(real_queries[indices])
            except (ValueError, FileNotFoundError):
                pass  # Fall back to other strategies
        
        # Random docs
        if proportions.get("random_docs_as_queries", 0) > 0:
            n_random = int(self.config.num_queries * proportions["random_docs_as_queries"] / total)
            # Temporarily override num_queries
            old_num = self.config.num_queries
            self.config.num_queries = n_random
            random_queries = self._sample_random_docs()
            self.config.num_queries = old_num
            queries_list.append(random_queries)
        
        # Cluster centroids
        if proportions.get("cluster_centroids", 0) > 0:
            n_cluster = int(self.config.num_queries * proportions["cluster_centroids"] / total)
            old_num = self.config.num_queries
            self.config.num_queries = n_cluster
            cluster_queries = self._sample_cluster_centroids()
            self.config.num_queries = old_num
            queries_list.append(cluster_queries)
        
        if not queries_list:
            return self._sample_random_docs()
        
        result = np.vstack(queries_list)
        
        # Trim to exact number
        if len(result) > self.config.num_queries:
            indices = self.rng.choice(len(result), self.config.num_queries, replace=False)
            result = result[indices]
        
        return result


def sample_queries(
    doc_embeddings: np.ndarray,
    config: ScanConfig,
    metadata: Optional[Metadata] = None,
) -> np.ndarray:
    """
    Sample queries using configured strategy.
    
    Args:
        doc_embeddings: Document embeddings
        config: Scan configuration
        metadata: Optional metadata
        
    Returns:
        Query embeddings array
    """
    sampler = QuerySampler(doc_embeddings, config, metadata)
    return sampler.sample()

