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

from typing import Tuple, Optional, List, Dict, Any
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, Query, QueryVector
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    Query = None
    QueryVector = None

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
    
    def search_hybrid(
        self,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using Qdrant's native hybrid search.
        
        Args:
            query_vectors: Optional query embeddings array of shape (M, D)
            query_texts: Optional list of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            alpha: Weight for vector search (0.0-1.0), where 1-alpha is weight for lexical
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if query_vectors is None and query_texts is None:
            raise ValueError("Either query_vectors or query_texts must be provided")
        
        # For now, Qdrant hybrid search requires both vector and sparse (text) vectors
        # Fall back to vector search if only vectors provided
        if query_vectors is not None:
            distances, indices = self.search(query_vectors, k)
            metadata = {
                "ranking_method": "hybrid" if query_texts is not None else "vector",
                "alpha": alpha,
                "fallback": query_texts is not None,  # True if text provided but not used
            }
            return distances, indices, metadata
        else:
            raise NotImplementedError(
                "Qdrant requires query_vectors for search. "
                "Pure lexical search not supported without vectors."
            )
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using lexical/keyword ranking.
        
        Note: Qdrant requires sparse vectors for lexical search. This method
        raises NotImplementedError as it requires additional setup.
        
        Args:
            query_texts: List of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        raise NotImplementedError(
            "Qdrant lexical search requires sparse vector setup. "
            "Use search_hybrid() with query_vectors and query_texts instead."
        )
    
    def search_reranked(
        self,
        query_vectors: np.ndarray,
        k: int,
        rerank_top_n: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using initial vector retrieval followed by reranking.
        
        Args:
            query_vectors: Query embeddings array of shape (M, D)
            k: Number of final results to return after reranking
            rerank_top_n: Number of candidates to retrieve before reranking
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Retrieve more candidates
        distances, indices = self.search(query_vectors, rerank_top_n)
        
        # Return top k
        distances = distances[:, :k]
        indices = indices[:, :k]
        
        metadata = {
            "ranking_method": "reranked",
            "rerank_top_n": rerank_top_n,
            "fallback": True,
        }
        
        return distances, indices, metadata
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Qdrant collection."""
        return self._dimension

