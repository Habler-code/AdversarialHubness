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
from ..hybrid_fusion import ClientFusionHybridSearch
from ....utils.logging import get_logger

logger = get_logger()


class QdrantIndex(VectorIndex):
    """
    Qdrant implementation of VectorIndex interface.
    
    This adapter connects to a Qdrant collection and provides the VectorIndex
    interface for HubScan detection operations.
    
    Supports hybrid search via:
    - client_fusion: Dense search in Qdrant + local lexical scoring (requires document_texts)
    - native_sparse: Qdrant dense+sparse vectors (requires collection with sparse vectors)
    """
    
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        document_texts: Optional[List[str]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Qdrant adapter.
        
        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
            document_texts: Optional list of document texts for client-side hybrid search.
                          Must match the number of vectors in the collection.
            hybrid_config: Optional hybrid search configuration dict with keys:
                          - backend: "client_fusion" or "native_sparse"
                          - lexical_backend: "bm25" or "tfidf"
                          - normalize_scores: bool
                          - qdrant_dense_vector_name: str (for native_sparse)
                          - qdrant_sparse_vector_name: str (for native_sparse)
        """
        # Check import at runtime in case package was installed after module load
        if QdrantClient is None:
            try:
                from qdrant_client import QdrantClient as QC
                from qdrant_client.models import Distance, VectorParams
                # Update module-level variables
                globals()['QdrantClient'] = QC
                globals()['Distance'] = Distance
                globals()['VectorParams'] = VectorParams
            except ImportError:
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
        
        # Hybrid search setup
        self._document_texts = document_texts
        self._hybrid_config = hybrid_config or {}
        self._hybrid_search: Optional[ClientFusionHybridSearch] = None
        
        # Build lexical index if document texts provided
        if document_texts is not None:
            if len(document_texts) != self._ntotal and self._ntotal > 0:
                logger.warning(
                    f"Number of document texts ({len(document_texts)}) "
                    f"does not match collection size ({self._ntotal})"
                )
            self._build_hybrid_search()
    
    def _build_hybrid_search(self):
        """Build client-side hybrid search if document texts available."""
        if not self._document_texts:
            return
        
        backend = self._hybrid_config.get("backend", "client_fusion")
        if backend not in ("client_fusion", "auto"):
            return  # native_sparse doesn't need local index
        
        lexical_backend = self._hybrid_config.get("lexical_backend", "bm25")
        normalize = self._hybrid_config.get("normalize_scores", True)
        
        try:
            self._hybrid_search = ClientFusionHybridSearch(
                document_texts=self._document_texts,
                lexical_backend=lexical_backend,
                normalize=normalize,
            )
            logger.info(f"Built {lexical_backend} index for Qdrant hybrid search")
        except ImportError as e:
            logger.warning(f"Could not build hybrid search: {e}")
    
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
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using hybrid vector + lexical ranking.
        
        Supports two backends:
        - client_fusion: Dense search in Qdrant + local BM25/TF-IDF (default)
        - native_sparse: Qdrant dense+sparse vectors (requires sparse vectors in collection)
        
        Args:
            query_vectors: Optional query embeddings array of shape (M, D)
            query_texts: Optional list of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            alpha: Weight for vector search (0.0-1.0), where 1-alpha is weight for lexical
            **kwargs: Additional arguments (e.g., hybrid_backend override)
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if query_vectors is None and query_texts is None:
            raise ValueError("Either query_vectors or query_texts must be provided")
        
        backend = kwargs.get("hybrid_backend", self._hybrid_config.get("backend", "client_fusion"))
        
        # Use client-side fusion if available
        if backend in ("client_fusion", "auto") and self._hybrid_search is not None:
            if query_vectors is not None and query_texts is not None:
                # Get dense results first
                dense_distances, dense_indices = self.search(query_vectors, k)
                
                # Use ClientFusionHybridSearch for fusion
                distances, indices, metadata = self._hybrid_search.fuse(
                    dense_distances=dense_distances,
                    dense_indices=dense_indices,
                    query_texts=query_texts,
                    k=k,
                    alpha=alpha,
                )
                return distances, indices, metadata
        
        # Validate hybrid requirements if text is provided but no hybrid search available
        if query_texts is not None and alpha < 1.0:
            if self._hybrid_search is None and backend not in ("native_sparse",):
                raise ValueError(
                    "Hybrid search requires document_texts to be provided when initializing "
                    "QdrantIndex for client_fusion backend, or a Qdrant collection with sparse "
                    "vectors for native_sparse backend. "
                    "Either provide document_texts or use ranking.method='vector'."
                )
        
        # Native sparse (requires Qdrant sparse vectors)
        if backend == "native_sparse":
            # TODO: Implement native Qdrant sparse hybrid when collection has sparse vectors
            logger.warning(
                "Qdrant native sparse hybrid not yet implemented. "
                "Falling back to vector search."
            )
        
        # Fallback to vector search
        if query_vectors is not None:
            distances, indices = self.search(query_vectors, k)
            metadata = {
                "ranking_method": "vector",
                "alpha": alpha,
                "fallback": True,
                "fallback_reason": "no_hybrid_support",
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
    
    def extract_embeddings(
        self,
        batch_size: int = 1000,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Extract all embeddings from the Qdrant collection.
        
        Args:
            batch_size: Number of vectors to retrieve per batch
            limit: Optional maximum number of vectors to extract (None = all)
            
        Returns:
            Tuple of (embeddings, ids) where:
            - embeddings: Array of shape (N, D) containing all vectors
            - ids: List of document IDs corresponding to each embedding
        """
        all_embeddings = []
        all_ids = []
        
        try:
            # Qdrant supports scroll API to get all points
            offset = None
            retrieved = 0
            
            while True:
                # Scroll through collection
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=False,
                    with_vectors=True,  # Include vectors
                )
                
                points = scroll_result[0]  # Points list
                next_offset = scroll_result[1]  # Next offset
                
                if not points:
                    break
                
                for point in points:
                    # Qdrant returns PointStruct with vector and id
                    point_id = point.id
                    vector = point.vector
                    
                    # Handle different vector formats
                    if isinstance(vector, dict):
                        # Named vector format: {"vector_name": [values]}
                        vec_values = list(vector.values())[0] if vector else None
                    elif isinstance(vector, list):
                        vec_values = vector
                    else:
                        logger.warning(f"Unexpected vector format for ID {point_id}")
                        continue
                    
                    if vec_values is None:
                        continue
                    
                    all_embeddings.append(vec_values)
                    all_ids.append(point_id)
                    retrieved += 1
                    
                    if limit and retrieved >= limit:
                        break
                
                if limit and retrieved >= limit:
                    break
                
                if next_offset is None:
                    break
                
                offset = next_offset
        
        except Exception as e:
            logger.error(f"Qdrant embedding extraction failed: {e}")
            raise
        
        if not all_embeddings:
            logger.warning("No embeddings extracted from Qdrant collection")
            return np.array([], dtype=np.float32).reshape(0, self._dimension), []
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Extracted {len(embeddings_array)} embeddings from Qdrant collection")
        
        return embeddings_array, all_ids
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Qdrant collection."""
        return self._dimension

