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

"""Built-in reranking method implementations."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ...utils.logging import get_logger

logger = get_logger()


class DefaultReranking:
    """
    Default reranking: retrieves more candidates, returns top k.
    
    This is a simple reranking that retrieves rerank_top_n candidates
    and returns the top k results. More sophisticated reranking can be
    implemented as custom methods.
    """
    
    def rerank(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Default reranking: simply return top k from candidates.
        
        Args:
            distances: Array of shape (M, N) containing scores
            indices: Array of shape (M, N) containing document indices
            query_vectors: Optional query embeddings (not used in default)
            query_texts: Optional query texts (not used in default)
            k: Number of final results to return
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata)
        """
        # For similarity metrics (cosine, IP), higher values indicate better matches
        # For distance metrics (L2), lower values indicate better matches
        # Default assumes similarity metric
        # Users can override this behavior in custom reranking methods
        
        # Sort by descending scores (assuming similarity metric)
        sorted_indices = np.argsort(-distances, axis=1)
        
        # Take top k
        top_k_indices = sorted_indices[:, :k]
        
        # Gather reranked results
        M = distances.shape[0]
        reranked_distances = np.zeros((M, k), dtype=distances.dtype)
        reranked_indices = np.zeros((M, k), dtype=indices.dtype)
        
        for i in range(M):
            reranked_distances[i] = distances[i, top_k_indices[i]]
            reranked_indices[i] = indices[i, top_k_indices[i]]
        
        metadata = {
            "reranking_method": "default",
            "candidates": distances.shape[1],
            "final_k": k,
        }
        
        return reranked_distances, reranked_indices, metadata


class CrossEncoderReranking:
    """
    Semantic reranking using cross-encoder models.
    
    Cross-encoders jointly encode query-document pairs and produce
    relevance scores, providing more accurate ranking than bi-encoders.
    This enables fine-grained semantic similarity that bi-encoder
    embedding models cannot capture.
    
    Requires sentence-transformers library:
        pip install sentence-transformers
    
    Example usage:
        from hubscan.core.reranking import register_reranking_method
        from hubscan.core.reranking.builtin import CrossEncoderReranking
        
        # Use default model
        reranker = CrossEncoderReranking()
        register_reranking_method("cross_encoder", reranker)
        
        # Or use a custom model
        reranker = CrossEncoderReranking(model_name="cross-encoder/ms-marco-TinyBERT-L-2")
        register_reranking_method("cross_encoder_fast", reranker)
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder.
                        Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
                        Popular options:
                        - cross-encoder/ms-marco-MiniLM-L-6-v2 (balanced)
                        - cross-encoder/ms-marco-MiniLM-L-12-v2 (higher quality)
                        - cross-encoder/ms-marco-TinyBERT-L-2 (faster)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
    
    def _load_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "Cross-encoder reranking requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading cross-encoder model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder model loaded")
    
    def rerank(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        document_texts: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Rerank using cross-encoder semantic similarity.
        
        Args:
            distances: Array of shape (M, N) containing initial scores/distances
            indices: Array of shape (M, N) containing document indices
            query_vectors: Optional query embeddings (not used)
            query_texts: List of query text strings (M,) - REQUIRED
            k: Number of final results to return after reranking
            document_texts: List of all document texts - REQUIRED
                           Pass via rerank_params={'document_texts': [...]}
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata) where:
            - reranked_distances: Array of shape (M, k) with cross-encoder scores
            - reranked_indices: Array of shape (M, k) with reranked document indices
            - metadata: Dictionary with reranking info
        """
        if query_texts is None:
            raise ValueError(
                "Cross-encoder reranking requires query_texts. "
                "Provide query texts via scan.query_texts_path config."
            )
        if document_texts is None:
            raise ValueError(
                "Cross-encoder reranking requires document_texts. "
                "Pass via scan.ranking.rerank_params.document_texts or "
                "ensure metadata contains a 'text' field."
            )
        
        self._load_model()
        
        M, N = indices.shape
        actual_k = min(k, N)
        reranked_distances = np.zeros((M, actual_k), dtype=np.float32)
        reranked_indices = np.zeros((M, actual_k), dtype=np.int64)
        
        for i, query_text in enumerate(query_texts):
            # Get candidate document indices for this query
            candidate_indices = indices[i]
            
            # Filter out invalid indices (-1 or out of range)
            valid_mask = (candidate_indices >= 0) & (candidate_indices < len(document_texts))
            valid_indices = candidate_indices[valid_mask]
            
            if len(valid_indices) == 0:
                # No valid candidates, fill with -1
                reranked_indices[i] = -1
                reranked_distances[i] = 0.0
                continue
            
            # Get candidate document texts
            candidate_texts = [document_texts[idx] for idx in valid_indices]
            
            # Score with cross-encoder
            pairs = [(query_text, doc) for doc in candidate_texts]
            scores = self._model.predict(pairs)
            
            # Sort by score (descending) and take top k
            sorted_local_idx = np.argsort(-scores)[:actual_k]
            
            # Map back to original document indices
            num_results = min(actual_k, len(sorted_local_idx))
            reranked_indices[i, :num_results] = valid_indices[sorted_local_idx[:num_results]]
            reranked_distances[i, :num_results] = scores[sorted_local_idx[:num_results]]
            
            # Fill remaining with -1 if needed
            if num_results < actual_k:
                reranked_indices[i, num_results:] = -1
                reranked_distances[i, num_results:] = 0.0
        
        metadata = {
            "reranking_method": "cross_encoder",
            "model": self._model_name,
            "candidates": N,
            "final_k": actual_k,
        }
        
        return reranked_distances, reranked_indices, metadata

