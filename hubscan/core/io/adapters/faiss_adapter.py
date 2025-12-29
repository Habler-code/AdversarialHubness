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

"""FAISS adapter for VectorIndex interface."""

from typing import Tuple
import numpy as np
import faiss

from ..vector_index import VectorIndex


class FAISSIndex(VectorIndex):
    """
    FAISS implementation of VectorIndex interface.
    
    This adapter wraps a FAISS index object and provides the VectorIndex
    interface, allowing FAISS indices to work seamlessly with the abstraction
    layer while maintaining backward compatibility.
    """
    
    def __init__(self, index: faiss.Index):
        """
        Initialize FAISS adapter.
        
        Args:
            index: FAISS index object to wrap
        """
        if not isinstance(index, faiss.Index):
            raise TypeError(f"Expected faiss.Index, got {type(index)}")
        self._index = index
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using FAISS.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        distances, indices = self._index.search(query_vectors, k)
        return distances, indices
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the FAISS index."""
        return self._index.ntotal
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the FAISS index."""
        return self._index.d
    
    @property
    def faiss_index(self) -> faiss.Index:
        """
        Access the underlying FAISS index object.
        
        This allows direct access to FAISS-specific functionality
        when needed for advanced use cases.
        
        Returns:
            The wrapped FAISS index object
        """
        return self._index

