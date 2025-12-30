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

"""Built-in reranking method implementations."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np


class DefaultReranking:
    """
    Default reranking: retrieves more candidates, returns top k.
    
    This is a simple reranking that retrieves rerank_top_n candidates
    and returns the top k results. More sophisticated reranking can be
    implemented as custom methods.
    """
    
    def rerank(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Default reranking: simply return top k from candidates.
        
        Args:
            distances: Array of shape (M, N) containing scores
            indices: Array of shape (M, N) containing document indices
            query_vectors: Optional query embeddings (not used in default)
            query_texts: Optional query texts (not used in default)
            k: Number of final results to return
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata)
        """
        # For similarity metrics (cosine, IP), higher is better
        # For distance metrics (L2), lower is better
        # We'll assume similarity metric (higher = better) by default
        # Users can override this behavior in custom reranking methods
        
        # Sort by descending scores (assuming similarity metric)
        sorted_indices = np.argsort(-distances, axis=1)
        
        # Take top k
        top_k_indices = sorted_indices[:, :k]
        
        # Gather reranked results
        M = distances.shape[0]
        reranked_distances = np.zeros((M, k), dtype=distances.dtype)
        reranked_indices = np.zeros((M, k), dtype=indices.dtype)
        
        for i in range(M):
            reranked_distances[i] = distances[i, top_k_indices[i]]
            reranked_indices[i] = indices[i, top_k_indices[i]]
        
        metadata = {
            "reranking_method": "default",
            "candidates": distances.shape[1],
            "final_k": k,
        }
        
        return reranked_distances, reranked_indices, metadata

