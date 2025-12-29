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

"""Qdrant adapter for VectorIndex interface."""

from typing import Tuple, Optional
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None

from ..vector_index import VectorIndex
from ...utils.logging import get_logger

logger = get_logger()


class QdrantIndex(VectorIndex):
    """
    Qdrant implementation of VectorIndex interface.
    
    This adapter connects to a Qdrant collection and provides the VectorIndex
    interface for HubScan detection operations.
    """
    
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant adapter.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
        """
        if QdrantClient is None:
            raise ImportError(
                "Qdrant adapter requires 'qdrant-client' package. "
                "Install with: pip install qdrant-client"
            )
        
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
        
        # Get collection info
        try:
            collection_info = self.client.get_collection(collection_name)
            self._dimension = collection_info.config.params.vectors.size
            self._ntotal = collection_info.points_count
            
            logger.info(
                f"Connected to Qdrant collection '{collection_name}' "
                f"with {self._ntotal} vectors"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to connect to Qdrant collection '{collection_name}': {e}"
            )
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using Qdrant.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        M = len(query_vectors)
        distances = []
        indices = []
        
        for query in query_vectors:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query.tolist(),
                    limit=k,
                    with_payload=False,
                    with_vectors=False,
                )
                
                result_distances = []
                result_indices = []
                
                for result in results:
                    # Qdrant returns scores (higher is better for similarity)
                    # For distance metrics, score is already distance
                    # For similarity metrics, we may need to convert
                    score = result.score
                    
                    # Qdrant uses distance for L2, similarity for cosine/IP
                    # We'll use score as-is and let the detector handle it
                    result_distances.append(float(score))
                    
                    # Qdrant IDs can be integers or UUIDs
                    # Convert to int if possible, otherwise use hash
                    point_id = result.id
                    if isinstance(point_id, int):
                        idx = point_id
                    elif isinstance(point_id, str):
                        try:
                            idx = int(point_id)
                        except ValueError:
                            # Use hash for non-numeric IDs
                            idx = hash(point_id) % (2**31)  # Keep within int32 range
                    else:
                        idx = hash(str(point_id)) % (2**31)
                    
                    result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
            
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                raise
        
        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the Qdrant collection."""
        # Refresh count
        try:
            collection_info = self.client.get_collection(self.collection_name)
            self._ntotal = collection_info.points_count
        except Exception:
            pass  # Keep cached value on error
        return self._ntotal
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Qdrant collection."""
        return self._dimension

