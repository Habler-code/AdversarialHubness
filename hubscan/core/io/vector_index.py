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

"""Abstract interface for vector search indices."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import numpy as np


class VectorIndex(ABC):
    """
    Abstract interface for vector search indices.
    
    This interface allows HubScan to work with any vector database
    that supports k-nearest neighbor search. Implementations should
    provide adapters for specific vector databases (FAISS, Pinecone,
    Qdrant, Weaviate, etc.).
    """
    
    @abstractmethod
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for each query vector.
        
        Args:
            query_vectors: Query embeddings array of shape (M, D) where
                          M is the number of queries and D is the dimension
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices) where:
            - distances: Array of shape (M, k) containing distances/similarities
                        for each query to its k nearest neighbors
            - indices: Array of shape (M, k) containing document indices
                      (0-based) for the k nearest neighbors
            
        Note:
            For similarity-based metrics (cosine, inner product), distances
            may actually be similarity scores. Higher values indicate closer
            matches. For distance-based metrics (L2), lower values indicate
            closer matches.
        """
        pass
    
    @property
    @abstractmethod
    def ntotal(self) -> int:
        """
        Number of vectors in the index.
        
        Returns:
            Total number of document vectors stored in the index
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Dimension of vectors in the index.
        
        Returns:
            Vector dimension (D)
        """
        pass
    
    def search_hybrid(
        self,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using hybrid vector + lexical ranking.
        
        Args:
            query_vectors: Optional query embeddings array of shape (M, D)
            query_texts: Optional list of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            alpha: Weight for vector search (0.0-1.0), where 1-alpha is weight for lexical
            
        Returns:
            Tuple of (distances, indices, metadata) where:
            - distances: Array of shape (M, k) containing combined scores
            - indices: Array of shape (M, k) containing document indices
            - metadata: Dictionary with ranking_method, alpha, and other metadata
            
        Note:
            Default implementation falls back to vector search if only query_vectors provided,
            or raises NotImplementedError if only query_texts provided. Adapters should
            override this method to provide native hybrid search support.
        """
        if query_vectors is not None:
            # Fallback to vector search
            distances, indices = self.search(query_vectors, k)
            metadata = {
                "ranking_method": "vector",
                "alpha": 1.0,
                "fallback": True,
            }
            return distances, indices, metadata
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support pure lexical search. "
                "Provide query_vectors or implement search_hybrid() in adapter."
            )
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using pure lexical/keyword ranking (BM25, TF-IDF, etc.).
        
        Args:
            query_texts: List of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices, metadata) where:
            - distances: Array of shape (M, k) containing lexical scores (higher = better match)
            - indices: Array of shape (M, k) containing document indices
            - metadata: Dictionary with ranking_method and other metadata
            
        Note:
            Default implementation raises NotImplementedError. Adapters should override
            this method to provide lexical search support.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support lexical search. "
            "Implement search_lexical() in adapter or use search_hybrid() with alpha=0.0."
        )
    
    def search_reranked(
        self,
        query_vectors: np.ndarray,
        k: int,
        rerank_top_n: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using initial vector retrieval followed by semantic reranking.
        
        Args:
            query_vectors: Query embeddings array of shape (M, D)
            k: Number of final results to return after reranking
            rerank_top_n: Number of candidates to retrieve before reranking
            
        Returns:
            Tuple of (distances, indices, metadata) where:
            - distances: Array of shape (M, k) containing reranked scores
            - indices: Array of shape (M, k) containing document indices
            - metadata: Dictionary with ranking_method, rerank_top_n, and other metadata
            
        Note:
            Default implementation performs initial retrieval with rerank_top_n,
            then returns top k results. Adapters can override for native reranking support.
        """
        # Default implementation: retrieve more, return top k
        distances, indices = self.search(query_vectors, rerank_top_n)
        
        # Return top k results
        distances = distances[:, :k]
        indices = indices[:, :k]
        
        metadata = {
            "ranking_method": "reranked",
            "rerank_top_n": rerank_top_n,
            "fallback": True,
        }
        
        return distances, indices, metadata

