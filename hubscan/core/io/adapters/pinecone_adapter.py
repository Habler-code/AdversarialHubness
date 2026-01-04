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

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import random

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    ServerlessSpec = None

from ..vector_index import VectorIndex
from ..hybrid_fusion import ClientFusionHybridSearch
from ....utils.logging import get_logger

logger = get_logger()


class PineconeIndex(VectorIndex):
    """
    Pinecone implementation of VectorIndex interface.
    
    This adapter connects to a Pinecone index and provides the VectorIndex
    interface for HubScan detection operations.
    
    Supports hybrid search via:
    - client_fusion: Dense search in Pinecone + local lexical scoring (requires document_texts)
    - native_sparse: Pinecone sparse vectors (requires index with sparse vectors)
    """
    
    def __init__(
        self,
        index_name: str,
        api_key: str,
        dimension: int = 0,
        environment: Optional[str] = None,
        document_texts: Optional[List[str]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Pinecone adapter.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            dimension: Vector dimension (will be inferred from index if 0)
            environment: Pinecone environment (deprecated in v3+, optional)
            document_texts: Optional list of document texts for client-side hybrid search.
                          Must match the number of vectors in the index.
            hybrid_config: Optional hybrid search configuration dict with keys:
                          - backend: "client_fusion" or "native_sparse"
                          - lexical_backend: "bm25" or "tfidf"
                          - normalize_scores: bool
                          - pinecone_has_sparse: bool
        """
        # Check import at runtime in case package was installed after module load
        if Pinecone is None:
            try:
                from pinecone import Pinecone as PC
                globals()['Pinecone'] = PC
            except ImportError:
                raise ImportError(
                    "Pinecone adapter requires 'pinecone' package. "
                    "Install with: pip install pinecone"
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
        
        # Hybrid search setup
        self._document_texts = document_texts
        self._hybrid_config = hybrid_config or {}
        self._hybrid_search: Optional[ClientFusionHybridSearch] = None
        
        # Build lexical index if document texts provided
        if document_texts is not None:
            if len(document_texts) != self._ntotal and self._ntotal > 0:
                logger.warning(
                    f"Number of document texts ({len(document_texts)}) "
                    f"does not match index size ({self._ntotal})"
                )
            self._build_hybrid_search()
        
        logger.info(f"Connected to Pinecone index '{index_name}' with {self._ntotal} vectors")
    
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
            logger.info(f"Built {lexical_backend} index for Pinecone hybrid search")
        except ImportError as e:
            logger.warning(f"Could not build hybrid search: {e}")
    
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
        
        # Query each vector individually (Pinecone v3+ doesn't have batch query)
        try:
            for query_vector in query_vectors:
                result = self.index.query(
                    vector=query_vector.tolist(),
                    top_k=k,
                    include_metadata=False,
                    include_values=False,
                )
                
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
        - client_fusion: Dense search in Pinecone + local BM25/TF-IDF (default)
        - native_sparse: Pinecone sparse vectors (requires sparse vectors in index)
        
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
                    "PineconeIndex for client_fusion backend, or a Pinecone index with sparse "
                    "vectors for native_sparse backend. "
                    "Either provide document_texts or use ranking.method='vector'."
                )
        
        # Native sparse (requires Pinecone sparse vectors)
        if backend == "native_sparse":
            pinecone_has_sparse = self._hybrid_config.get("pinecone_has_sparse", False)
            if not pinecone_has_sparse:
                raise ValueError(
                    "Native sparse hybrid search requires a Pinecone index with sparse vectors. "
                    "Set hybrid.pinecone_has_sparse=true in config if your index supports it, "
                    "or use hybrid.backend='client_fusion' with document_texts."
                )
            # TODO: Implement native Pinecone sparse hybrid when sparse vectors are available
            logger.warning(
                "Pinecone native sparse hybrid not yet implemented. "
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
        
        raise ValueError("Pinecone requires query_vectors for search")
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using lexical/keyword ranking.
        
        Note: Pinecone requires sparse vectors for lexical search. This method
        raises NotImplementedError as it requires additional setup.
        
        Args:
            query_texts: List of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        raise NotImplementedError(
            "Pinecone lexical search requires sparse vector setup. "
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
        Extract all embeddings from the Pinecone index.
        
        Args:
            batch_size: Number of vectors to retrieve per batch
            limit: Optional maximum number of vectors to extract (None = all)
            
        Returns:
            Tuple of (embeddings, ids) where:
            - embeddings: Array of shape (N, D) containing all vectors
            - ids: List of document IDs corresponding to each embedding
            
        Note:
            Uses query-based discovery to find IDs, then fetches vectors.
            For better performance with known IDs, pass them directly.
        """
        import random
        
        all_embeddings = []
        all_ids = []
        discovered_ids = set()
        
        stats = self.index.describe_index_stats()
        total_count = stats.get("total_vector_count", 0)
        
        if total_count == 0:
            logger.warning("Pinecone index is empty")
            return np.array([], dtype=np.float32).reshape(0, self._dimension), []
        
        if limit is not None:
            total_count = min(total_count, limit)
        
        # Discover IDs by querying with random vectors
        # Use enough queries to cover the space
        num_queries = max(10, min(100, total_count // 10 + 1))
        random_queries = [
            [random.random() for _ in range(self._dimension)]
            for _ in range(num_queries)
        ]
        
        # Query to discover IDs
        for query_vec in random_queries:
            try:
                results = self.index.query(
                    vector=query_vec,
                    top_k=min(batch_size, total_count),
                    include_metadata=False,
                    include_values=False,
                )
                for match in results.matches:
                    discovered_ids.add(match.id)
                    if len(discovered_ids) >= total_count:
                        break
                if len(discovered_ids) >= total_count:
                    break
            except Exception as e:
                logger.warning(f"Query failed during ID discovery: {e}")
                continue
        
        id_list = list(discovered_ids)[:total_count]
        
        if not id_list:
            logger.warning("No IDs discovered from Pinecone index")
            return np.array([], dtype=np.float32).reshape(0, self._dimension), []
        
        # Fetch vectors in batches
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            try:
                fetch_result = self.index.fetch(ids=batch_ids)
                
                for vec_id, vector_data in fetch_result.vectors.items():
                    # Handle different Pinecone response formats
                    if hasattr(vector_data, 'values'):
                        vec_values = vector_data.values
                    elif isinstance(vector_data, dict):
                        vec_values = vector_data.get('values', vector_data)
                    elif isinstance(vector_data, list):
                        vec_values = vector_data
                    else:
                        logger.warning(f"Unexpected vector format for ID {vec_id}")
                        continue
                    
                    all_embeddings.append(vec_values)
                    all_ids.append(vec_id)
                    
                    if limit and len(all_embeddings) >= limit:
                        break
            except Exception as e:
                logger.warning(f"Failed to fetch batch starting at {i}: {e}")
                continue
            
            if limit and len(all_embeddings) >= limit:
                break
        
        if not all_embeddings:
            logger.warning("No embeddings extracted from Pinecone index")
            return np.array([], dtype=np.float32).reshape(0, self._dimension), []
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Extracted {len(embeddings_array)} embeddings from Pinecone index")
        
        return embeddings_array, all_ids
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Pinecone index."""
        return self._dimension

