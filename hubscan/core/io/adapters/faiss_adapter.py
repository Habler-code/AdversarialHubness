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

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
import numpy as np
import faiss

from ..vector_index import VectorIndex


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return vectors / norms


class FAISSIndex(VectorIndex):
    """
    FAISS implementation of VectorIndex interface.
    
    This adapter wraps a FAISS index object and provides the VectorIndex
    interface, allowing FAISS indices to work seamlessly with the abstraction
    layer while maintaining backward compatibility.
    
    Supports hybrid and lexical search when document texts are provided.
    """
    
    def __init__(
        self,
        index: faiss.Index,
        document_texts: Optional[List[str]] = None,
    ):
        """
        Initialize FAISS adapter.
        
        Args:
            index: FAISS index object to wrap
            document_texts: Optional list of document texts for lexical search.
                          Must match the number of vectors in the index.
        """
        if not isinstance(index, faiss.Index):
            raise TypeError(f"Expected faiss.Index, got {type(index)}")
        self._index = index
        self._document_texts = document_texts
        self._bm25_index = None
        
        # Build BM25 index if texts provided
        if document_texts is not None:
            if len(document_texts) != index.ntotal:
                raise ValueError(
                    f"Number of document texts ({len(document_texts)}) "
                    f"does not match index size ({index.ntotal})"
                )
            self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from document texts."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 package required for lexical search. "
                "Install with: pip install rank-bm25"
            )
        
        # Tokenize documents
        tokenized_docs = [doc.split() if isinstance(doc, str) else [] for doc in self._document_texts]
        
        # Check if we have any non-empty documents
        if not tokenized_docs or all(len(doc) == 0 for doc in tokenized_docs):
            # Skip BM25 index building if all documents are empty
            # This prevents division by zero errors in BM25
            self._bm25_index = None
            return
        
        self._bm25_index = BM25Okapi(tokenized_docs)
    
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
            Tuple of (distances, indices, metadata)
        """
        if query_vectors is None and query_texts is None:
            raise ValueError("Either query_vectors or query_texts must be provided")
        
        M = len(query_vectors) if query_vectors is not None else len(query_texts)
        
        # Get vector results
        vector_distances = None
        vector_indices = None
        if query_vectors is not None and alpha > 0.0:
            vector_distances, vector_indices = self.search(query_vectors, k)
            # Normalize vector scores to [0, 1] for cosine similarity
            # For distance metrics, convert to similarity
            if self._index.metric_type == faiss.METRIC_INNER_PRODUCT:
                # Inner product: normalize by max
                max_score = np.max(np.abs(vector_distances))
                if max_score > 0:
                    vector_distances = (vector_distances + max_score) / (2 * max_score)
            elif self._index.metric_type == faiss.METRIC_L2:
                # L2: convert to similarity (inverse)
                max_dist = np.max(vector_distances[vector_distances < np.inf])
                if max_dist > 0:
                    vector_distances = 1.0 / (1.0 + vector_distances / max_dist)
            # Cosine is already normalized
        
        # Get lexical results
        lexical_scores = None
        lexical_indices = None
        if query_texts is not None and alpha < 1.0:
            if self._bm25_index is None:
                raise ValueError(
                    "Lexical search requires document_texts to be provided "
                    "when initializing FAISSIndex"
                )
            lexical_distances, lexical_indices, _ = self.search_lexical(query_texts, k)
            # BM25 scores are similarity scores
            # Normalize to [0, 1]
            max_score = np.max(lexical_distances)
            if max_score > 0:
                lexical_scores = lexical_distances / max_score
            else:
                lexical_scores = lexical_distances
        
        # Combine results
        if vector_distances is not None and lexical_scores is not None:
            # Both available: combine
            combined_distances = []
            combined_indices = []
            
            for i in range(M):
                # Get candidate sets
                vector_candidates = dict(zip(vector_indices[i], vector_distances[i]))
                lexical_candidates = dict(zip(lexical_indices[i], lexical_scores[i]))
                
                # Combine all candidates
                all_candidates = set(vector_candidates.keys()) | set(lexical_candidates.keys())
                
                # Calculate combined scores
                candidate_scores = []
                candidate_indices = []
                for doc_idx in all_candidates:
                    vector_score = vector_candidates.get(doc_idx, 0.0) * alpha
                    lexical_score = lexical_candidates.get(doc_idx, 0.0) * (1.0 - alpha)
                    combined_score = vector_score + lexical_score
                    candidate_scores.append(combined_score)
                    candidate_indices.append(doc_idx)
                
                # Sort by combined score and take top k
                sorted_pairs = sorted(
                    zip(candidate_scores, candidate_indices),
                    reverse=True
                )[:k]
                combined_distances.append([score for score, _ in sorted_pairs])
                combined_indices.append([idx for _, idx in sorted_pairs])
            
            distances = np.array(combined_distances, dtype=np.float32)
            indices = np.array(combined_indices, dtype=np.int64)
            metadata = {
                "ranking_method": "hybrid",
                "alpha": alpha,
                "fallback": False,
            }
        elif vector_distances is not None:
            # Fallback to vector-only
            distances = vector_distances
            indices = vector_indices
            metadata = {
                "ranking_method": "vector",
                "alpha": alpha,
                "fallback": True,
            }
        else:
            # Fallback to lexical-only
            distances = lexical_scores
            indices = lexical_indices
            metadata = {
                "ranking_method": "lexical",
                "alpha": alpha,
                "fallback": True,
            }
        
        return distances, indices, metadata
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search using pure lexical/keyword ranking (BM25).
        
        Args:
            query_texts: List of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if self._bm25_index is None:
            raise ValueError(
                "Lexical search requires document_texts to be provided "
                "when initializing FAISSIndex"
            )
        
        M = len(query_texts)
        distances = []
        indices = []
        
        for query_text in query_texts:
            # Tokenize query
            query_tokens = query_text.split() if isinstance(query_text, str) else []
            
            # Get BM25 scores
            scores = self._bm25_index.get_scores(query_tokens)
            
            # Get top k
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_scores = scores[top_k_indices]
            
            distances.append(top_k_scores.astype(np.float32))
            indices.append(top_k_indices.astype(np.int64))
        
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
        Search using initial vector retrieval followed by semantic reranking.
        
        For FAISS, this is a simple implementation that retrieves more candidates
        and returns top k. True reranking would require a cross-encoder model.
        
        Args:
            query_vectors: Query embeddings array of shape (M, D)
            k: Number of final results to return after reranking
            rerank_top_n: Number of candidates to retrieve before reranking
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Use parent implementation which retrieves more and returns top k
        return super().search_reranked(query_vectors, k, rerank_top_n)
    
    def extract_embeddings(
        self,
        batch_size: int = 1000,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Extract all embeddings from the FAISS index.
        
        Args:
            batch_size: Number of vectors to retrieve per batch (unused for FAISS)
            limit: Optional maximum number of vectors to extract (None = all)
            
        Returns:
            Tuple of (embeddings, ids) where:
            - embeddings: Array of shape (N, D) containing all vectors
            - ids: List of document indices (0 to n-1)
            
        Note:
            Uses FAISS reconstruct_n() method. Not all index types support this.
            For indexes that don't support reconstruction, returns empty array.
        """
        n = self._index.ntotal
        if n == 0:
            return np.array([], dtype=np.float32).reshape(0, self.dimension), []
        
        if limit is not None:
            n = min(n, limit)
        
        try:
            # Try to reconstruct vectors
            # FAISS reconstruct_n works for most index types
            embeddings = self._index.reconstruct_n(0, n)
            ids = list(range(n))
            
            return embeddings.astype(np.float32), ids
            
        except AttributeError:
            # Index doesn't support reconstruct_n
            # Try alternative: reconstruct one by one (slower)
            try:
                embeddings = []
                ids = []
                for i in range(n):
                    vec = self._index.reconstruct(i)
                    embeddings.append(vec)
                    ids.append(i)
                return np.array(embeddings, dtype=np.float32), ids
            except AttributeError:
                # Index doesn't support reconstruction at all
                raise NotImplementedError(
                    f"FAISS index type {type(self._index)} does not support "
                    "embedding extraction. Only index types that support "
                    "reconstruct() or reconstruct_n() can extract embeddings."
                )
    
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
    
    # =========================================================================
    # Static/Class Methods for Index Management
    # =========================================================================
    
    @staticmethod
    def build(
        embeddings: np.ndarray,
        index_type: str = "flat",
        metric: str = "cosine",
        params: Optional[Dict[str, Any]] = None,
        document_texts: Optional[List[str]] = None,
    ) -> "FAISSIndex":
        """
        Build a new FAISS index from embeddings.
        
        Args:
            embeddings: Embeddings array of shape (N, D)
            index_type: Type of index ("flat", "hnsw", "ivf_pq")
            metric: Distance metric ("cosine", "ip", "l2")
            params: Index-specific parameters
            document_texts: Optional document texts for lexical search
            
        Returns:
            FAISSIndex adapter wrapping the built index
            
        Example:
            >>> embeddings = np.random.randn(1000, 128).astype(np.float32)
            >>> index = FAISSIndex.build(embeddings, index_type="flat", metric="cosine")
        """
        if params is None:
            params = {}
        
        d = embeddings.shape[1]
        
        # Normalize for cosine similarity
        if metric == "cosine":
            embeddings = _normalize_vectors(embeddings)
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        if index_type == "flat":
            if metric == "l2":
                index = faiss.IndexFlatL2(d)
            else:
                index = faiss.IndexFlatIP(d)
        
        elif index_type == "hnsw":
            M = params.get("M", 32)
            efConstruction = params.get("efConstruction", 200)
            
            index = faiss.IndexHNSWFlat(d, M, metric_type)
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = params.get("efSearch", 128)
        
        elif index_type == "ivf_pq":
            nlist = params.get("nlist", 4096)
            m = params.get("m", 64)  # Number of subquantizers
            nbits = params.get("nbits", 8)  # Bits per subquantizer
            
            quantizer = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
            index.nprobe = params.get("nprobe", 16)
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Train if needed
        if index_type == "ivf_pq":
            index.train(embeddings)
        
        # Add vectors
        index.add(embeddings)
        
        return FAISSIndex(index, document_texts=document_texts)
    
    @staticmethod
    def load(
        path: Union[str, Path],
        document_texts: Optional[List[str]] = None,
    ) -> "FAISSIndex":
        """
        Load a FAISS index from file.
        
        Args:
            path: Path to the saved FAISS index file
            document_texts: Optional document texts for lexical search
            
        Returns:
            FAISSIndex adapter wrapping the loaded index
            
        Example:
            >>> index = FAISSIndex.load("my_index.index")
        """
        index = faiss.read_index(str(path))
        return FAISSIndex(index, document_texts=document_texts)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the FAISS index to file.
        
        Args:
            path: Path where the index will be saved
            
        Example:
            >>> index.save("my_index.index")
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path_obj))
    
    @staticmethod
    def load_raw(path: Union[str, Path]) -> faiss.Index:
        """
        Load a raw FAISS index (without wrapping).
        
        This is useful when you need the raw faiss.Index object
        for compatibility with existing code.
        
        Args:
            path: Path to the saved FAISS index file
            
        Returns:
            Raw FAISS index object
        """
        return faiss.read_index(str(path))
    
    @staticmethod
    def build_raw(
        embeddings: np.ndarray,
        index_type: str = "flat",
        metric: str = "cosine",
        params: Optional[Dict[str, Any]] = None,
    ) -> faiss.Index:
        """
        Build a raw FAISS index (without wrapping).
        
        This is useful when you need the raw faiss.Index object
        for compatibility with existing code.
        
        Args:
            embeddings: Embeddings array of shape (N, D)
            index_type: Type of index ("flat", "hnsw", "ivf_pq")
            metric: Distance metric ("cosine", "ip", "l2")
            params: Index-specific parameters
            
        Returns:
            Raw FAISS index object
        """
        wrapped = FAISSIndex.build(embeddings, index_type, metric, params)
        return wrapped.faiss_index
    
    @staticmethod
    def save_raw(index: faiss.Index, path: Union[str, Path]) -> None:
        """
        Save a raw FAISS index to file.
        
        Args:
            index: FAISS index object
            path: Path where the index will be saved
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path_obj))

