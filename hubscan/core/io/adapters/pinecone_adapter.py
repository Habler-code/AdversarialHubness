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
from ....utils.logging import get_logger

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
    
    def search_hybrid(
        self,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using Pinecone's native hybrid search.
        
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
        
        M = len(query_vectors) if query_vectors is not None else len(query_texts)
        distances = []
        indices = []
        
        # Convert queries to format Pinecone expects
        query_list = query_vectors.tolist() if query_vectors is not None else None
        
        try:
            # Use Pinecone's hybrid query if both available
            if query_vectors is not None and query_texts is not None:
                results = self.index.query_batch(
                    queries=[
                        {
                            "vector": vec,
                            "sparse": {"indices": [], "values": []},  # Sparse vector for text
                            "topK": k,
                        }
                        for vec in query_list
                    ],
                    top_k=k,
                    include_metadata=False,
                    include_values=False,
                )
                # Note: Pinecone hybrid query requires sparse vectors
                # For now, fall back to vector search if text provided
                logger.warning(
                    "Pinecone hybrid search with sparse vectors not fully implemented. "
                    "Falling back to vector search."
                )
                distances, indices = self.search(query_vectors, k)
            elif query_vectors is not None:
                distances, indices = self.search(query_vectors, k)
            else:
                raise ValueError("Pinecone requires query_vectors for search")
            
            metadata = {
                "ranking_method": "hybrid" if query_texts is not None else "vector",
                "alpha": alpha,
                "fallback": query_texts is not None,
            }
            
            return distances, indices, metadata
            
        except Exception as e:
            logger.error(f"Pinecone hybrid query failed: {e}")
            # Fallback to vector search
            if query_vectors is not None:
                distances, indices = self.search(query_vectors, k)
                metadata = {
                    "ranking_method": "vector",
                    "alpha": 1.0,
                    "fallback": True,
                }
                return distances, indices, metadata
            raise
    
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

