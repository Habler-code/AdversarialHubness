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

"""
Example: Custom Reranking Method Plugin

This example demonstrates how to create and register a custom reranking method
that can be applied as post-processing to any ranking method.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from hubscan.core.reranking import register_reranking_method, RerankingMethod


class BoostTopResultsReranking:
    """
    Custom reranking method that boosts scores for top-ranked results.
    
    This reranking method can be applied to any ranking method (vector, hybrid, lexical).
    """
    
    def rerank(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        boost_factor: float = 0.2,
        top_n_to_boost: int = 5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Rerank by boosting top results.
        
        Args:
            distances: Initial scores (M, N)
            indices: Initial document indices (M, N)
            query_vectors: Query embeddings (not used in this example)
            query_texts: Query texts (not used in this example)
            k: Final number of results
            boost_factor: Factor to boost top results (e.g., 0.2 = 20% boost)
            top_n_to_boost: Number of top results to boost
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata)
        """
        # Copy distances for modification
        reranked_distances = distances.copy()
        
        # Boost top results
        for i in range(len(reranked_distances)):
            top_n = min(top_n_to_boost, len(reranked_distances[i]))
            reranked_distances[i, :top_n] *= (1.0 + boost_factor)
        
        # Sort by descending scores (assuming similarity metric)
        sorted_indices = np.argsort(-reranked_distances, axis=1)
        
        # Take top k
        top_k_indices = sorted_indices[:, :k]
        
        # Gather reranked results
        M = len(reranked_distances)
        final_distances = np.zeros((M, k), dtype=distances.dtype)
        final_indices = np.zeros((M, k), dtype=indices.dtype)
        
        for i in range(M):
            final_distances[i] = reranked_distances[i, top_k_indices[i]]
            final_indices[i] = indices[i, top_k_indices[i]]
        
        metadata = {
            "reranking_method": "boost_top_results",
            "boost_factor": boost_factor,
            "top_n_to_boost": top_n_to_boost,
        }
        
        return final_distances, final_indices, metadata


class SemanticReranking:
    """
    Example reranking method that could use semantic similarity for reranking.
    
    This is a placeholder showing how you might implement semantic reranking
    using cross-encoder models or other semantic scoring.
    """
    
    def rerank(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        doc_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Semantic reranking using query-document similarity.
        
        Args:
            distances: Initial scores (M, N)
            indices: Initial document indices (M, N)
            query_vectors: Query embeddings (M, D)
            query_texts: Query texts (M,)
            k: Final number of results
            doc_embeddings: Document embeddings (N, D) - required for semantic reranking
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata)
        """
        if query_vectors is None or doc_embeddings is None:
            # Fallback to default reranking if embeddings not available
            from hubscan.core.reranking.builtin import DefaultReranking
            default = DefaultReranking()
            return default.rerank(distances, indices, query_vectors, query_texts, k, **kwargs)
        
        # Compute semantic similarity for reranking
        # This is a simplified example - in practice, you might use cross-encoders
        # or more sophisticated semantic scoring
        
        M = len(query_vectors)
        N = doc_embeddings.shape[0]
        
        # Compute semantic scores for all query-document pairs
        semantic_scores = np.zeros((M, N))
        for i in range(M):
            query = query_vectors[i:i+1]
            # Compute cosine similarity with all documents
            doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            similarities = np.dot(doc_embeddings, query.T).flatten() / (doc_norms.flatten() * query_norm.flatten() + 1e-8)
            semantic_scores[i] = similarities
        
        # Combine initial scores with semantic scores
        # Get semantic scores for retrieved documents
        combined_scores = distances.copy()
        for i in range(M):
            retrieved_indices = indices[i]
            semantic_for_retrieved = semantic_scores[i, retrieved_indices]
            # Weighted combination (50% initial, 50% semantic)
            combined_scores[i] = 0.5 * distances[i] + 0.5 * semantic_for_retrieved
        
        # Sort and take top k
        sorted_indices = np.argsort(-combined_scores, axis=1)
        top_k_indices = sorted_indices[:, :k]
        
        final_distances = np.zeros((M, k), dtype=distances.dtype)
        final_indices = np.zeros((M, k), dtype=indices.dtype)
        
        for i in range(M):
            final_distances[i] = combined_scores[i, top_k_indices[i]]
            final_indices[i] = indices[i, top_k_indices[i]]
        
        metadata = {
            "reranking_method": "semantic",
            "combination": "weighted_50_50",
        }
        
        return final_distances, final_indices, metadata


# Register custom reranking methods
def register_custom_reranking_methods():
    """Register custom reranking methods."""
    register_reranking_method("boost_top", BoostTopResultsReranking())
    register_reranking_method("semantic", SemanticReranking())
    print("Registered custom reranking methods: boost_top, semantic")


if __name__ == "__main__":
    # Register the custom methods
    register_custom_reranking_methods()
    
    # Now these reranking methods can be used in config files:
    # ranking:
    #   method: vector  # or hybrid, lexical
    #   rerank: true
    #   rerank_method: boost_top
    #   rerank_params:
    #     boost_factor: 0.3
    #     top_n_to_boost: 10
    #
    # or
    #
    # ranking:
    #   method: hybrid
    #   rerank: true
    #   rerank_method: semantic
    #   rerank_top_n: 200
    
    from hubscan.core.reranking import list_reranking_methods
    print(f"All available reranking methods: {list_reranking_methods()}")

