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

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import faiss

from ..vector_index import VectorIndex


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
            # BM25 scores are already similarity scores (higher = better)
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
            
        elif vector_distances is not None:
            distances = vector_distances
            indices = vector_indices
        else:
            distances = lexical_scores
            indices = lexical_indices
        
        metadata = {
            "ranking_method": "hybrid",
            "alpha": alpha,
            "fallback": False,
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

