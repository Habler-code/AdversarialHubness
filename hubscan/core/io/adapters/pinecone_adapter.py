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

"""Pinecone adapter for VectorIndex interface."""

from typing import Tuple, Optional
import numpy as np

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    ServerlessSpec = None

from ..vector_index import VectorIndex
from ...utils.logging import get_logger

logger = get_logger()


class PineconeIndex(VectorIndex):
    """
    Pinecone implementation of VectorIndex interface.
    
    This adapter connects to a Pinecone index and provides the VectorIndex
    interface for HubScan detection operations.
    """
    
    def __init__(
        self,
        index_name: str,
        api_key: str,
        dimension: int = 0,
        environment: Optional[str] = None,
    ):
        """
        Initialize Pinecone adapter.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            dimension: Vector dimension (will be inferred from index if 0)
            environment: Pinecone environment (deprecated in v3+, optional)
        """
        if Pinecone is None:
            raise ImportError(
                "Pinecone adapter requires 'pinecone-client' package. "
                "Install with: pip install pinecone-client"
            )
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        
        # Get index stats to infer dimension and count
        stats = self.index.describe_index_stats()
        self._ntotal = stats.get("total_vector_count", 0)
        
        # Get dimension from index description
        index_info = self.pc.describe_index(index_name)
        if dimension == 0:
            self._dimension = index_info.dimension
        else:
            self._dimension = dimension
            if index_info.dimension != dimension:
                logger.warning(
                    f"Specified dimension {dimension} does not match "
                    f"index dimension {index_info.dimension}"
                )
        
        logger.info(f"Connected to Pinecone index '{index_name}' with {self._ntotal} vectors")
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using Pinecone.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices) where distances are converted
            from similarity scores (1 - score for cosine similarity)
        """
        M = len(query_vectors)
        distances = []
        indices = []
        
        # Convert to list of lists for Pinecone
        query_list = query_vectors.tolist()
        
        # Pinecone query_batch returns results for all queries
        try:
            results = self.index.query_batch(
                queries=query_list,
                top_k=k,
                include_metadata=False,
                include_values=False,
            )
            
            for result in results:
                result_distances = []
                result_indices = []
                
                for match in result.matches:
                    # Pinecone returns similarity scores (0-1 for cosine)
                    # Convert to distance: distance = 1 - similarity
                    similarity = match.score
                    distance = 1.0 - similarity
                    result_distances.append(distance)
                    
                    # Pinecone IDs are strings, convert to int if possible
                    try:
                        idx = int(match.id)
                    except (ValueError, TypeError):
                        # If ID is not numeric, use hash or raise error
                        raise ValueError(
                            f"Pinecone document IDs must be numeric integers. "
                            f"Found non-numeric ID: {match.id}"
                        )
                    result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
        
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            raise
        
        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the Pinecone index."""
        # Refresh stats
        stats = self.index.describe_index_stats()
        self._ntotal = stats.get("total_vector_count", 0)
        return self._ntotal
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Pinecone index."""
        return self._dimension

