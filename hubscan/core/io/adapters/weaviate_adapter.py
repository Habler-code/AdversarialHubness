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

"""Weaviate adapter for VectorIndex interface."""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np

try:
    import weaviate
except ImportError:
    weaviate = None

from ..vector_index import VectorIndex
from ..hybrid_fusion import ClientFusionHybridSearch
from ....utils.logging import get_logger

logger = get_logger()


class WeaviateIndex(VectorIndex):
    """
    Weaviate implementation of VectorIndex interface.
    
    This adapter connects to a Weaviate class and provides the VectorIndex
    interface for HubScan detection operations.
    
    Supports hybrid search via:
    - native: Weaviate's built-in BM25 + nearVector (default, recommended)
    - client_fusion: Dense search + local BM25/TF-IDF (requires document_texts)
    """
    
    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        document_texts: Optional[List[str]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Weaviate adapter.
        
        Args:
            class_name: Name of the Weaviate class (collection)
            url: Weaviate server URL
            api_key: Optional API key for Weaviate Cloud
            document_texts: Optional list of document texts for client-side hybrid search.
                          Note: Weaviate has native BM25 support, so this is usually not needed.
            hybrid_config: Optional hybrid search configuration dict with keys:
                          - backend: "native" (default) or "client_fusion"
                          - lexical_backend: "bm25" or "tfidf" (for client_fusion)
                          - normalize_scores: bool
                          - weaviate_bm25_properties: List[str] (properties to search)
        """
        self._hybrid_config = hybrid_config or {}
        self._document_texts = document_texts
        self._hybrid_search: Optional[ClientFusionHybridSearch] = None
        if weaviate is None:
            raise ImportError(
                "Weaviate adapter requires 'weaviate-client' package. "
                "Install with: pip install weaviate-client"
            )
        
        self.class_name = class_name
        self.url = url
        self.api_key = api_key
        
        # Initialize Weaviate client - support both v3 and v4 APIs
        try:
            # Try v4 API first
            if hasattr(weaviate, 'connect_to_local') or hasattr(weaviate, 'WeaviateClient'):
                # v4 API
                if 'localhost' in url or '127.0.0.1' in url:
                    port = int(url.split(':')[-1]) if ':' in url else 8080
                    if api_key:
                        self.client = weaviate.connect_to_custom(
                            http_host=url.replace('http://', '').replace('https://', '').split(':')[0],
                            http_port=port,
                            http_secure=False,
                            grpc_port=50051,
                            auth_credentials=weaviate.auth.AuthApiKey(api_key=api_key)
                        )
                    else:
                        self.client = weaviate.connect_to_local(port=port, grpc_port=50051)
                else:
                    # Custom URL
                    if api_key:
                        self.client = weaviate.connect_to_custom(
                            http_host=url.replace('http://', '').replace('https://', '').split(':')[0],
                            http_port=int(url.split(':')[-1]) if ':' in url else 8080,
                            http_secure='https' in url,
                            grpc_port=50051,
                            auth_credentials=weaviate.auth.AuthApiKey(api_key=api_key)
                        )
                    else:
                        self.client = weaviate.connect_to_custom(
                            http_host=url.replace('http://', '').replace('https://', '').split(':')[0],
                            http_port=int(url.split(':')[-1]) if ':' in url else 8080,
                            http_secure='https' in url,
                            grpc_port=50051
                        )
                self._is_v4 = True
            else:
                # v3 API fallback
                if api_key:
                    auth_config = weaviate.AuthApiKey(api_key=api_key)
                    self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
                else:
                    self.client = weaviate.Client(url=url)
                self._is_v4 = False
        except Exception as e:
            # If v4 fails, try v3
            try:
                if api_key:
                    auth_config = weaviate.AuthApiKey(api_key=api_key)
                    self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
                else:
                    self.client = weaviate.Client(url=url)
                self._is_v4 = False
            except:
                raise ImportError(
                    "Weaviate adapter requires 'weaviate-client' package. "
                    "Install with: pip install weaviate-client"
                )
        
        # Initialize dimension - will be inferred from first query if needed
        self._dimension = 0
        
        # Get schema info
        try:
            schema = self.client.schema.get(class_name)
            if not schema:
                raise ValueError(f"Weaviate class '{class_name}' not found")
            
            # Note: Weaviate doesn't store dimension in schema, will infer from first query
            
            # Get count
            result = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .do()
            )
            
            if result and "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(class_name, [])
                if aggregate_data:
                    self._ntotal = aggregate_data[0].get("meta", {}).get("count", 0)
                else:
                    self._ntotal = 0
            else:
                self._ntotal = 0
            
            logger.info(
                f"Connected to Weaviate class '{class_name}' "
                f"with {self._ntotal} vectors"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to connect to Weaviate class '{class_name}': {e}"
            )
        
        # Build client-side hybrid search if document_texts provided
        if document_texts is not None:
            self._build_hybrid_search()
    
    def _build_hybrid_search(self):
        """Build client-side hybrid search if document texts available."""
        if not self._document_texts:
            return
        
        backend = self._hybrid_config.get("backend", "native")
        if backend not in ("client_fusion",):
            return  # Native doesn't need local index
        
        lexical_backend = self._hybrid_config.get("lexical_backend", "bm25")
        normalize = self._hybrid_config.get("normalize_scores", True)
        
        try:
            self._hybrid_search = ClientFusionHybridSearch(
                document_texts=self._document_texts,
                lexical_backend=lexical_backend,
                normalize=normalize,
            )
            logger.info(f"Built {lexical_backend} index for Weaviate client-side hybrid search")
        except ImportError as e:
            logger.warning(f"Could not build hybrid search: {e}")
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using Weaviate.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        # Infer dimension from first query if not set
        if self._dimension == 0 and len(query_vectors) > 0:
            self._dimension = query_vectors.shape[1]
        
        M = len(query_vectors)
        distances = []
        indices = []
        
        for query in query_vectors:
            try:
                result = (
                    self.client.query
                    .get(self.class_name)
                    .with_near_vector({"vector": query.tolist()})
                    .with_limit(k)
                    .with_additional(["distance", "id"])
                    .do()
                )
                
                result_distances = []
                result_indices = []
                
                if result and "data" in result and "Get" in result["data"]:
                    get_data = result["data"]["Get"].get(self.class_name, [])
                    
                    for item in get_data:
                        additional = item.get("_additional", {})
                        distance = additional.get("distance", float('inf'))
                        point_id = additional.get("id", None)
                        
                        result_distances.append(float(distance))
                        
                        # Weaviate IDs are UUIDs, convert to int hash
                        if point_id:
                            idx = hash(str(point_id)) % (2**31)  # Keep within int32 range
                        else:
                            idx = -1
                        
                        result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
            
            except Exception as e:
                logger.error(f"Weaviate search failed: {e}")
                raise
        
        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the Weaviate class."""
        # Refresh count
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            if result and "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(self.class_name, [])
                if aggregate_data:
                    self._ntotal = aggregate_data[0].get("meta", {}).get("count", 0)
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
        - native: Weaviate's built-in BM25 + nearVector (default, recommended)
        - client_fusion: Dense search + local BM25/TF-IDF
        
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
        
        backend = kwargs.get("hybrid_backend", self._hybrid_config.get("backend", "native"))
        
        # Use client-side fusion if explicitly requested and available
        if backend == "client_fusion" and self._hybrid_search is not None:
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
        
        # Use native Weaviate hybrid search
        # Weaviate's .with_hybrid() combines BM25 + vector search natively
        bm25_properties = self._hybrid_config.get("weaviate_bm25_properties", ["text"])
        
        M = len(query_vectors) if query_vectors is not None else len(query_texts)
        distances = []
        indices = []
        
        for i in range(M):
            try:
                query_text = query_texts[i] if query_texts is not None else ""
                query_vector = query_vectors[i].tolist() if query_vectors is not None else None
                
                # Use Weaviate's native hybrid search method
                query_builder = (
                    self.client.query
                    .get(self.class_name)
                    .with_hybrid(
                        query=query_text,
                        vector=query_vector,
                        alpha=alpha,  # 0 = pure BM25, 1 = pure vector
                        properties=bm25_properties,
                    )
                    .with_limit(k)
                    .with_additional(["score", "id"])
                )
                
                result = query_builder.do()
                
                result_distances = []
                result_indices = []
                
                if result and "data" in result and "Get" in result["data"]:
                    get_data = result["data"]["Get"].get(self.class_name, [])
                    
                    if get_data:
                        for item in get_data:
                            additional = item.get("_additional", {})
                            # Weaviate hybrid returns "score" (higher is better)
                            # Convert to distance-like metric (lower is better)
                            score = additional.get("score", 0.0)
                            distance = 1.0 - float(score) if score is not None else float('inf')
                            point_id = additional.get("id", None)
                            
                            result_distances.append(distance)
                            
                            if point_id:
                                idx = hash(str(point_id)) % (2**31)
                            else:
                                idx = -1
                            
                            result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
            
            except Exception as e:
                logger.warning(f"Weaviate hybrid search failed for query {i}: {e}")
                # Fallback to vector search if available
                if query_vectors is not None:
                    vec_distances, vec_indices = self.search(query_vectors[i:i+1], k)
                    distances.append(vec_distances[0].tolist())
                    indices.append(vec_indices[0].tolist())
                else:
                    # Fallback to empty results if no vector available
                    distances.append([float('inf')] * k)
                    indices.append([-1] * k)
        
        distances_array = np.array(distances, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.int64)
        
        metadata = {
            "ranking_method": "hybrid",
            "hybrid_backend": "native_weaviate",
            "bm25_properties": bm25_properties,
            "alpha": alpha,
            "fallback": False,
        }
        
        return distances_array, indices_array, metadata
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using Weaviate's BM25 lexical search.
        
        Args:
            query_texts: List of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        M = len(query_texts)
        distances = []
        indices = []
        
        for query_text in query_texts:
            try:
                result = (
                    self.client.query
                    .get(self.class_name)
                    .with_bm25(query=query_text, properties=["text"])
                    .with_limit(k)
                    .with_additional(["distance", "id"])
                    .do()
                )
                
                result_distances = []
                result_indices = []
                
                if result and "data" in result and "Get" in result["data"]:
                    get_data = result["data"]["Get"].get(self.class_name, [])
                    
                    for item in get_data:
                        additional = item.get("_additional", {})
                        # For BM25, distance might not be available, use score if available
                        score = additional.get("score", 1.0)
                        distance = 1.0 / (score + 1e-6)  # Convert score to distance-like
                        point_id = additional.get("id", None)
                        
                        result_distances.append(float(distance))
                        
                        if point_id:
                            idx = hash(str(point_id)) % (2**31)
                        else:
                            idx = -1
                        
                        result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
            
            except Exception as e:
                logger.error(f"Weaviate lexical search failed: {e}")
                raise
        
        distances_array = np.array(distances, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.int64)
        
        metadata = {
            "ranking_method": "lexical",
            "backend": "bm25",
        }
        
        return distances_array, indices_array, metadata
    
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
        Extract all embeddings from the Weaviate class.
        
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
            # Weaviate uses cursor-based pagination
            cursor = None
            retrieved = 0
            
            while True:
                # Build query
                query_builder = (
                    self.client.query
                    .get(self.class_name)
                    .with_additional(["id", "vector"])
                    .with_limit(batch_size)
                )
                
                if cursor:
                    query_builder = query_builder.with_after(cursor)
                
                result = query_builder.do()
                
                if not result or "data" not in result or "Get" not in result["data"]:
                    break
                
                get_data = result["data"]["Get"].get(self.class_name, [])
                
                if not get_data:
                    break
                
                for item in get_data:
                    additional = item.get("_additional", {})
                    point_id = additional.get("id")
                    vector = additional.get("vector")
                    
                    if vector is None:
                        continue
                    
                    all_embeddings.append(vector)
                    all_ids.append(point_id)
                    retrieved += 1
                    
                    if limit and retrieved >= limit:
                        break
                
                if limit and retrieved >= limit:
                    break
                
                # Get cursor for next batch
                if get_data:
                    last_item = get_data[-1]
                    cursor = last_item.get("_additional", {}).get("id")
                    if cursor is None:
                        break
                else:
                    break
        
        except Exception as e:
            logger.error(f"Weaviate embedding extraction failed: {e}")
            raise
        
        if not all_embeddings:
            logger.warning("No embeddings extracted from Weaviate class")
            return np.array([], dtype=np.float32).reshape(0, self._dimension if self._dimension > 0 else 0), []
        
        # Infer dimension from first vector if not set
        if self._dimension == 0 and all_embeddings:
            self._dimension = len(all_embeddings[0])
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Extracted {len(embeddings_array)} embeddings from Weaviate class")
        
        return embeddings_array, all_ids
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Weaviate class."""
        return self._dimension if self._dimension > 0 else 0

