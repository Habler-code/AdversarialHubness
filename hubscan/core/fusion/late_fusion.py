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

"""Late fusion of results from multiple indexes."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from collections import defaultdict

from .parallel_retrieval import MultiIndexResult
from ...utils.logging import get_logger

logger = get_logger()


def normalize_scores(
    distances: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Normalize distance/similarity scores to [0, 1] range.
    
    Args:
        distances: Distance or similarity scores (M, k)
        metric: Distance metric ("cosine", "ip", "l2")
        
    Returns:
        Normalized scores in [0, 1] range
    """
    if metric == "cosine" or metric == "ip":
        # Similarity scores: normalize to [0, 1]
        # Assuming scores are already in [-1, 1] for cosine or [0, inf] for IP
        if metric == "cosine":
            # Cosine similarity: [-1, 1] -> [0, 1]
            normalized = (distances + 1.0) / 2.0
        else:
            # Inner product: normalize by max
            max_val = np.max(distances)
            if max_val > 0:
                normalized = distances / max_val
            else:
                normalized = distances
    else:
        # L2 distance: normalize by max, then invert (higher distance = lower score)
        max_dist = np.max(distances)
        if max_dist > 0:
            normalized = 1.0 - (distances / max_dist)
        else:
            normalized = np.ones_like(distances)
    
    # Clip to [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def fuse_results_rrf(
    result: MultiIndexResult,
    k: int,
    rrf_k: int = 60,
    text_weight: float = 0.4,
    image_weight: float = 0.4,
    unified_weight: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse results using Reciprocal Rank Fusion (RRF).
    
    Args:
        result: MultiIndexResult from parallel retrieval
        k: Number of final results to return
        rrf_k: RRF constant (higher = more weight to top results)
        text_weight: Weight for text index
        image_weight: Weight for image index
        unified_weight: Weight for unified/cross-modal index
        
    Returns:
        Tuple of (fused_distances, fused_indices) shape (M, k)
    """
    num_queries = result.metadata["num_queries"]
    fused_distances = np.zeros((num_queries, k), dtype=np.float32)
    fused_indices = np.zeros((num_queries, k), dtype=np.int32)
    
    for query_idx in range(num_queries):
        # Collect all document scores with their sources
        doc_scores = defaultdict(float)
        doc_sources = defaultdict(list)
        
        # Process text results
        if result.text_indices is not None:
            for rank, doc_idx in enumerate(result.text_indices[query_idx]):
                rrf_score = text_weight / (rrf_k + rank + 1)
                doc_scores[doc_idx] += rrf_score
                doc_sources[doc_idx].append("text")
        
        # Process image results
        if result.image_indices is not None:
            for rank, doc_idx in enumerate(result.image_indices[query_idx]):
                rrf_score = image_weight / (rrf_k + rank + 1)
                doc_scores[doc_idx] += rrf_score
                doc_sources[doc_idx].append("image")
        
        # Process unified/cross-modal results
        if result.unified_indices is not None:
            for rank, doc_idx in enumerate(result.unified_indices[query_idx]):
                rrf_score = unified_weight / (rrf_k + rank + 1)
                doc_scores[doc_idx] += rrf_score
                doc_sources[doc_idx].append("unified")
        
        # Sort by fused score and take top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            fused_indices[query_idx, rank] = doc_idx
            fused_distances[query_idx, rank] = score
    
    return fused_distances, fused_indices


def fuse_results_weighted_sum(
    result: MultiIndexResult,
    k: int,
    text_weight: float = 0.4,
    image_weight: float = 0.4,
    unified_weight: float = 0.2,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse results using weighted sum of normalized scores.
    
    Args:
        result: MultiIndexResult from parallel retrieval
        k: Number of final results to return
        text_weight: Weight for text index
        image_weight: Weight for image index
        unified_weight: Weight for unified/cross-modal index
        normalize: Whether to normalize scores before fusion
        
    Returns:
        Tuple of (fused_distances, fused_indices) shape (M, k)
    """
    num_queries = result.metadata["num_queries"]
    fused_distances = np.zeros((num_queries, k), dtype=np.float32)
    fused_indices = np.zeros((num_queries, k), dtype=np.int32)
    
    for query_idx in range(num_queries):
        doc_scores = defaultdict(float)
        
        # Process text results
        if result.text_distances is not None and result.text_indices is not None:
            scores = result.text_distances[query_idx]
            if normalize:
                scores = normalize_scores(scores.reshape(1, -1), metric="cosine").flatten()
            for rank, (doc_idx, score) in enumerate(zip(result.text_indices[query_idx], scores)):
                doc_scores[doc_idx] += text_weight * score
        
        # Process image results
        if result.image_distances is not None and result.image_indices is not None:
            scores = result.image_distances[query_idx]
            if normalize:
                scores = normalize_scores(scores.reshape(1, -1), metric="cosine").flatten()
            for rank, (doc_idx, score) in enumerate(zip(result.image_indices[query_idx], scores)):
                doc_scores[doc_idx] += image_weight * score
        
        # Process unified/cross-modal results
        if result.unified_distances is not None and result.unified_indices is not None:
            scores = result.unified_distances[query_idx]
            if normalize:
                scores = normalize_scores(scores.reshape(1, -1), metric="cosine").flatten()
            for rank, (doc_idx, score) in enumerate(zip(result.unified_indices[query_idx], scores)):
                doc_scores[doc_idx] += unified_weight * score
        
        # Sort by fused score and take top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            fused_indices[query_idx, rank] = doc_idx
            fused_distances[query_idx, rank] = score
    
    return fused_distances, fused_indices


def fuse_results_max(
    result: MultiIndexResult,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse results using max score across indexes.
    
    Args:
        result: MultiIndexResult from parallel retrieval
        k: Number of final results to return
        
    Returns:
        Tuple of (fused_distances, fused_indices) shape (M, k)
    """
    num_queries = result.metadata["num_queries"]
    fused_distances = np.zeros((num_queries, k), dtype=np.float32)
    fused_indices = np.zeros((num_queries, k), dtype=np.int32)
    
    for query_idx in range(num_queries):
        doc_scores = {}
        
        # Process all indexes and take max score per document
        if result.text_distances is not None and result.text_indices is not None:
            for doc_idx, score in zip(result.text_indices[query_idx], result.text_distances[query_idx]):
                if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
                    doc_scores[doc_idx] = score
        
        if result.image_distances is not None and result.image_indices is not None:
            for doc_idx, score in zip(result.image_indices[query_idx], result.image_distances[query_idx]):
                if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
                    doc_scores[doc_idx] = score
        
        if result.unified_distances is not None and result.unified_indices is not None:
            for doc_idx, score in zip(result.unified_indices[query_idx], result.unified_distances[query_idx]):
                if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
                    doc_scores[doc_idx] = score
        
        # Sort by max score and take top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            fused_indices[query_idx, rank] = doc_idx
            fused_distances[query_idx, rank] = score
    
    return fused_distances, fused_indices


def fuse_results(
    result: MultiIndexResult,
    k: int,
    method: str = "rrf",
    normalize: bool = True,
    text_weight: float = 0.4,
    image_weight: float = 0.4,
    unified_weight: float = 0.2,
    rrf_k: int = 60,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fuse results from multiple indexes using specified method.
    
    Args:
        result: MultiIndexResult from parallel retrieval
        k: Number of final results to return
        method: Fusion method ("rrf", "weighted_sum", "max")
        normalize: Whether to normalize scores (for weighted_sum)
        text_weight: Weight for text index
        image_weight: Weight for image index
        unified_weight: Weight for unified/cross-modal index
        rrf_k: RRF constant (for RRF method)
        
    Returns:
        Tuple of (fused_distances, fused_indices, metadata)
    """
    if method == "rrf":
        fused_distances, fused_indices = fuse_results_rrf(
            result, k, rrf_k, text_weight, image_weight, unified_weight
        )
    elif method == "weighted_sum":
        fused_distances, fused_indices = fuse_results_weighted_sum(
            result, k, text_weight, image_weight, unified_weight, normalize
        )
    elif method == "max":
        fused_distances, fused_indices = fuse_results_max(result, k)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    metadata = {
        "fusion_method": method,
        "normalize": normalize,
        "text_weight": text_weight,
        "image_weight": image_weight,
        "unified_weight": unified_weight,
    }
    
    return fused_distances, fused_indices, metadata

