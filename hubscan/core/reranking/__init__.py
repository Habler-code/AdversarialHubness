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

"""Reranking plugin system for post-processing retrieval results."""

from typing import Protocol, Optional, List, Dict, Any, Tuple
import numpy as np


class RerankingMethod(Protocol):
    """Protocol for custom reranking methods."""
    
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
        Rerank retrieval results.
        
        Args:
            distances: Array of shape (M, N) containing initial scores/distances
            indices: Array of shape (M, N) containing document indices
            query_vectors: Optional query embeddings array of shape (M, D)
            query_texts: Optional list of query text strings (M,)
            k: Number of final results to return after reranking
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (reranked_distances, reranked_indices, metadata) where:
            - reranked_distances: Array of shape (M, k) containing reranked scores
            - reranked_indices: Array of shape (M, k) containing reranked document indices
            - metadata: Dictionary with reranking_method and other metadata
        """
        ...


# Registry for reranking methods
_RERANKING_METHODS: Dict[str, RerankingMethod] = {}


def register_reranking_method(name: str, method: RerankingMethod):
    """
    Register a custom reranking method.
    
    Args:
        name: Unique name for the reranking method
        method: RerankingMethod instance implementing the rerank protocol
    """
    if name in _RERANKING_METHODS:
        import warnings
        warnings.warn(f"Reranking method '{name}' is already registered. Overwriting.")
    _RERANKING_METHODS[name] = method


def get_reranking_method(name: str) -> Optional[RerankingMethod]:
    """
    Get a registered reranking method.
    
    Args:
        name: Name of the reranking method
        
    Returns:
        RerankingMethod instance if found, None otherwise
    """
    return _RERANKING_METHODS.get(name)


def list_reranking_methods() -> List[str]:
    """List all registered reranking method names."""
    return list(_RERANKING_METHODS.keys())


# Register built-in methods
def _register_builtin_methods():
    """Register built-in reranking methods."""
    from .builtin import DefaultReranking
    
    register_reranking_method("default", DefaultReranking())

_register_builtin_methods()

__all__ = [
    "RerankingMethod",
    "register_reranking_method",
    "get_reranking_method",
    "list_reranking_methods",
]

