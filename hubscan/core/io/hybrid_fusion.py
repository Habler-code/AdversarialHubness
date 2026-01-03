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

"""Shared hybrid search fusion utilities.

This module provides client-side hybrid search capabilities that work with
any vector database by combining dense vector search with local lexical scoring.
"""

from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

from ...utils.logging import get_logger

logger = get_logger()


class LexicalScorer(ABC):
    """Abstract base class for lexical scoring algorithms."""
    
    @abstractmethod
    def fit(self, documents: List[str]) -> 'LexicalScorer':
        """Fit the scorer on a document corpus.
        
        Args:
            documents: List of document text strings
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """Score all documents against a query.
        
        Args:
            query: Query text string
            
        Returns:
            Array of shape (num_docs,) with relevance scores (higher = better)
        """
        pass
    
    @abstractmethod
    def score_batch(self, queries: List[str]) -> np.ndarray:
        """Score all documents against multiple queries.
        
        Args:
            queries: List of query text strings
            
        Returns:
            Array of shape (num_queries, num_docs) with relevance scores
        """
        pass


class BM25Scorer(LexicalScorer):
    """BM25 lexical scoring implementation."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._num_docs = 0
        self._fitted = False
    
    def fit(self, documents: List[str]) -> 'BM25Scorer':
        """Fit BM25 on document corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "BM25 scoring requires 'rank-bm25' package. "
                "Install with: pip install rank-bm25"
            )
        
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [
            doc.lower().split() if isinstance(doc, str) and doc.strip() else []
            for doc in documents
        ]
        
        # Check if we have any non-empty documents
        non_empty = [doc for doc in tokenized_docs if len(doc) > 0]
        if not non_empty:
            logger.warning("All documents are empty, BM25 will return zero scores")
            self._bm25 = None
            self._num_docs = len(documents)
            self._fitted = True
            return self
        
        self._bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        self._num_docs = len(documents)
        self._fitted = True
        return self
    
    def score(self, query: str) -> np.ndarray:
        """Score all documents against a query."""
        if not self._fitted:
            raise ValueError("BM25Scorer must be fitted before scoring")
        
        if self._bm25 is None:
            return np.zeros(self._num_docs, dtype=np.float32)
        
        query_tokens = query.lower().split() if isinstance(query, str) else []
        scores = self._bm25.get_scores(query_tokens)
        return scores.astype(np.float32)
    
    def score_batch(self, queries: List[str]) -> np.ndarray:
        """Score all documents against multiple queries."""
        if not self._fitted:
            raise ValueError("BM25Scorer must be fitted before scoring")
        
        scores = np.zeros((len(queries), self._num_docs), dtype=np.float32)
        for i, query in enumerate(queries):
            scores[i] = self.score(query)
        return scores


class TFIDFScorer(LexicalScorer):
    """TF-IDF lexical scoring implementation using scikit-learn."""
    
    def __init__(self, max_features: int = 10000):
        """Initialize TF-IDF scorer.
        
        Args:
            max_features: Maximum number of vocabulary terms
        """
        self.max_features = max_features
        self._vectorizer = None
        self._doc_vectors = None
        self._num_docs = 0
        self._fitted = False
    
    def fit(self, documents: List[str]) -> 'TFIDFScorer':
        """Fit TF-IDF on document corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Handle empty documents
        documents = [doc if isinstance(doc, str) and doc.strip() else "" for doc in documents]
        
        self._vectorizer = TfidfVectorizer(max_features=self.max_features)
        self._doc_vectors = self._vectorizer.fit_transform(documents)
        self._num_docs = len(documents)
        self._fitted = True
        return self
    
    def score(self, query: str) -> np.ndarray:
        """Score all documents against a query using cosine similarity."""
        if not self._fitted:
            raise ValueError("TFIDFScorer must be fitted before scoring")
        
        query_vec = self._vectorizer.transform([query if isinstance(query, str) else ""])
        # Compute cosine similarity (TF-IDF vectors are already normalized)
        scores = (self._doc_vectors @ query_vec.T).toarray().flatten()
        return scores.astype(np.float32)
    
    def score_batch(self, queries: List[str]) -> np.ndarray:
        """Score all documents against multiple queries."""
        if not self._fitted:
            raise ValueError("TFIDFScorer must be fitted before scoring")
        
        query_vecs = self._vectorizer.transform(
            [q if isinstance(q, str) else "" for q in queries]
        )
        # Compute cosine similarities
        scores = (self._doc_vectors @ query_vecs.T).toarray().T
        return scores.astype(np.float32)


def create_lexical_scorer(backend: str = "bm25", **kwargs) -> LexicalScorer:
    """Factory function to create a lexical scorer.
    
    Args:
        backend: Scoring algorithm ("bm25" or "tfidf")
        **kwargs: Additional arguments passed to scorer constructor
        
    Returns:
        LexicalScorer instance
    """
    if backend == "bm25":
        return BM25Scorer(**{k: v for k, v in kwargs.items() if k in ("k1", "b")})
    elif backend == "tfidf":
        return TFIDFScorer(**{k: v for k, v in kwargs.items() if k in ("max_features",)})
    else:
        raise ValueError(f"Unknown lexical backend: {backend}. Use 'bm25' or 'tfidf'.")


def normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize scores to [0, 1] range per query.
    
    Args:
        scores: Array of shape (num_queries, k) or (k,)
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized scores in [0, 1] range
    """
    if scores.ndim == 1:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < eps:
            return np.ones_like(scores)
        return (scores - min_val) / (max_val - min_val + eps)
    
    # Per-row normalization for 2D arrays
    min_vals = scores.min(axis=1, keepdims=True)
    max_vals = scores.max(axis=1, keepdims=True)
    ranges = max_vals - min_vals
    ranges = np.where(ranges < eps, 1.0, ranges)
    return (scores - min_vals) / ranges


def fuse_dense_lexical(
    dense_distances: np.ndarray,
    dense_indices: np.ndarray,
    lexical_scores: np.ndarray,
    k: int,
    alpha: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse dense vector search results with lexical scores.
    
    Args:
        dense_distances: Dense search distances/similarities, shape (M, k_dense)
        dense_indices: Dense search indices, shape (M, k_dense)
        lexical_scores: Full lexical scores for all documents, shape (M, num_docs)
        k: Number of final results per query
        alpha: Weight for dense scores (1-alpha for lexical)
        normalize: Whether to normalize scores before fusion
        
    Returns:
        Tuple of (fused_distances, fused_indices) both shape (M, k)
    """
    M = len(dense_distances)
    num_docs = lexical_scores.shape[1]
    
    fused_distances = []
    fused_indices = []
    
    for i in range(M):
        # Get dense candidates
        dense_idx = dense_indices[i]
        dense_dist = dense_distances[i]
        
        # Convert distances to similarity scores if needed
        # Assuming lower distance = higher similarity, invert
        # For cosine similarity stored as distance (1-sim), this converts back
        dense_scores = 1.0 - dense_dist if dense_dist.mean() < 1.0 else dense_dist
        
        # Normalize if requested
        if normalize:
            dense_scores = normalize_scores(dense_scores)
        
        # Build candidate map from dense results
        candidates: Dict[int, Dict[str, float]] = {}
        for idx, score in zip(dense_idx, dense_scores):
            if idx >= 0 and idx < num_docs:
                candidates[int(idx)] = {"dense": float(score), "lexical": 0.0}
        
        # Get lexical scores for all candidates plus top lexical results
        lexical_row = lexical_scores[i]
        
        # Normalize lexical scores
        if normalize and lexical_row.max() > 0:
            lexical_row = normalize_scores(lexical_row)
        
        # Add top lexical candidates
        top_lexical_indices = np.argsort(lexical_row)[::-1][:k]
        for idx in top_lexical_indices:
            idx = int(idx)
            if idx not in candidates:
                candidates[idx] = {"dense": 0.0, "lexical": float(lexical_row[idx])}
            else:
                candidates[idx]["lexical"] = float(lexical_row[idx])
        
        # Update lexical scores for dense candidates
        for idx in list(candidates.keys()):
            if candidates[idx]["lexical"] == 0.0 and idx < len(lexical_row):
                candidates[idx]["lexical"] = float(lexical_row[idx])
        
        # Compute fused scores
        fused = []
        for idx, scores in candidates.items():
            combined = alpha * scores["dense"] + (1 - alpha) * scores["lexical"]
            fused.append((combined, idx))
        
        # Sort by fused score and take top k
        fused.sort(reverse=True)
        fused = fused[:k]
        
        # Pad if needed
        while len(fused) < k:
            fused.append((0.0, -1))
        
        fused_distances.append([score for score, _ in fused])
        fused_indices.append([idx for _, idx in fused])
    
    return np.array(fused_distances, dtype=np.float32), np.array(fused_indices, dtype=np.int64)


class ClientFusionHybridSearch:
    """Client-side hybrid search that works with any vector database.
    
    Combines dense vector search from the DB with local lexical scoring
    (BM25 or TF-IDF) computed from document texts.
    """
    
    def __init__(
        self,
        document_texts: List[str],
        lexical_backend: str = "bm25",
        normalize: bool = True,
        **lexical_kwargs,
    ):
        """Initialize client-side hybrid search.
        
        Args:
            document_texts: List of document text strings (must match index order)
            lexical_backend: Lexical scoring algorithm ("bm25" or "tfidf")
            normalize: Whether to normalize scores before fusion
            **lexical_kwargs: Additional arguments for lexical scorer
        """
        if not document_texts:
            raise ValueError("document_texts cannot be empty for client-side hybrid search")
        
        self.document_texts = document_texts
        self.num_docs = len(document_texts)
        self.normalize = normalize
        
        logger.info(f"Building {lexical_backend} index for {self.num_docs} documents...")
        self.lexical_scorer = create_lexical_scorer(lexical_backend, **lexical_kwargs)
        self.lexical_scorer.fit(document_texts)
        logger.info(f"Lexical index built successfully")
    
    def fuse(
        self,
        dense_distances: np.ndarray,
        dense_indices: np.ndarray,
        query_texts: List[str],
        k: int,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fuse dense search results with lexical scores.
        
        Args:
            dense_distances: Dense search distances, shape (M, k_dense)
            dense_indices: Dense search indices, shape (M, k_dense)
            query_texts: Query text strings for lexical scoring
            k: Number of final results per query
            alpha: Weight for dense scores (1-alpha for lexical)
            
        Returns:
            Tuple of (fused_distances, fused_indices, metadata)
        """
        if len(query_texts) != len(dense_distances):
            raise ValueError(
                f"Number of query texts ({len(query_texts)}) must match "
                f"number of dense results ({len(dense_distances)})"
            )
        
        # Compute lexical scores for all queries
        lexical_scores = self.lexical_scorer.score_batch(query_texts)
        
        # Fuse results
        fused_distances, fused_indices = fuse_dense_lexical(
            dense_distances=dense_distances,
            dense_indices=dense_indices,
            lexical_scores=lexical_scores,
            k=k,
            alpha=alpha,
            normalize=self.normalize,
        )
        
        metadata = {
            "ranking_method": "hybrid",
            "hybrid_backend": "client_fusion",
            "lexical_backend": type(self.lexical_scorer).__name__,
            "alpha": alpha,
            "fallback": False,
        }
        
        return fused_distances, fused_indices, metadata

