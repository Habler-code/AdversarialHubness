# Copyright 2025 Cisco Systems, Inc. and its licensed under the Apache License, Version 2.0 (the "License");
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

"""Diversity enforcement for retrieval results."""

from typing import Optional, Tuple
import numpy as np

from ...utils.logging import get_logger

logger = get_logger()


def enforce_diversity(
    distances: np.ndarray,
    indices: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int,
    min_distance: float = 0.3,
    max_results_per_cluster: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforce diversity in retrieval results by ensuring minimum distance between results.
    
    Args:
        distances: Distance/similarity scores (M, k_original)
        indices: Document indices (M, k_original)
        doc_embeddings: Document embeddings (N, D)
        k: Number of diverse results to return
        min_distance: Minimum cosine distance between results
        max_results_per_cluster: Optional max results per semantic cluster
        
    Returns:
        Tuple of (diverse_distances, diverse_indices) shape (M, k)
    """
    num_queries = len(distances)
    diverse_distances = np.zeros((num_queries, k), dtype=distances.dtype)
    diverse_indices = np.zeros((num_queries, k), dtype=indices.dtype)
    
    # Normalize embeddings for cosine distance
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    normalized_embeddings = doc_embeddings / (norms + 1e-8)
    
    for query_idx in range(num_queries):
        query_distances = distances[query_idx]
        query_indices = indices[query_idx]
        
        # Filter out invalid indices
        valid_mask = query_indices >= 0
        valid_indices = query_indices[valid_mask]
        valid_distances = query_distances[valid_mask]
        
        if len(valid_indices) == 0:
            continue
        
        # Select diverse results
        selected_indices = []
        selected_distances = []
        selected_embeddings = []
        
        for i, (doc_idx, dist) in enumerate(zip(valid_indices, valid_distances)):
            if len(selected_indices) >= k:
                break
            
            doc_emb = normalized_embeddings[doc_idx]
            
            # Check minimum distance from already selected results
            if len(selected_embeddings) > 0:
                selected_embs = np.array(selected_embeddings)
                # Compute cosine distances
                cosine_dists = 1.0 - np.dot(selected_embs, doc_emb)
                min_cosine_dist = np.min(cosine_dists)
                
                if min_cosine_dist < min_distance:
                    # Too similar to existing results, skip
                    continue
            
            # Add to selected results
            selected_indices.append(doc_idx)
            selected_distances.append(dist)
            selected_embeddings.append(doc_emb)
        
        # Fill results
        for rank in range(k):
            if rank < len(selected_indices):
                diverse_indices[query_idx, rank] = selected_indices[rank]
                diverse_distances[query_idx, rank] = selected_distances[rank]
            else:
                # Pad with -1 if not enough diverse results
                diverse_indices[query_idx, rank] = -1
                diverse_distances[query_idx, rank] = 0.0
    
    return diverse_distances, diverse_indices

