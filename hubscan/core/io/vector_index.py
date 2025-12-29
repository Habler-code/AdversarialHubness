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
from typing import Tuple
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

