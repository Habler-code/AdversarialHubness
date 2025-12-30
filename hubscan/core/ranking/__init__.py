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

"""Ranking method plugin system for custom retrieval algorithms."""

from typing import Protocol, Optional, List, Dict, Any, Tuple
import numpy as np

from ..io.vector_index import VectorIndex


class RankingMethod(Protocol):
    """Protocol for custom ranking methods."""
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform search using custom ranking algorithm.
        
        Args:
            index: VectorIndex instance to search
            query_vectors: Optional query embeddings array of shape (M, D)
            query_texts: Optional list of query text strings (M,)
            k: Number of nearest neighbors to retrieve per query
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple of (distances, indices, metadata) where:
            - distances: Array of shape (M, k) containing scores
            - indices: Array of shape (M, k) containing document indices
            - metadata: Dictionary with ranking_method and other metadata
        """
        ...


# Registry for ranking methods
_RANKING_METHODS: Dict[str, RankingMethod] = {}


def register_ranking_method(name: str, method: RankingMethod):
    """
    Register a custom ranking method.
    
    Args:
        name: Unique name for the ranking method
        method: RankingMethod instance implementing the search protocol
    """
    if name in _RANKING_METHODS:
        import warnings
        warnings.warn(f"Ranking method '{name}' is already registered. Overwriting.")
    _RANKING_METHODS[name] = method


def get_ranking_method(name: str) -> Optional[RankingMethod]:
    """
    Get a registered ranking method.
    
    Args:
        name: Name of the ranking method
        
    Returns:
        RankingMethod instance if found, None otherwise
    """
    return _RANKING_METHODS.get(name)


def list_ranking_methods() -> List[str]:
    """List all registered ranking method names."""
    return list(_RANKING_METHODS.keys())


# Register built-in methods (imported at end to avoid circular imports)
def _register_builtin_methods():
    """Register built-in ranking methods."""
    from .builtin import VectorRanking, HybridRanking, LexicalRanking
    
    register_ranking_method("vector", VectorRanking())
    register_ranking_method("hybrid", HybridRanking())
    register_ranking_method("lexical", LexicalRanking())

_register_builtin_methods()

__all__ = [
    "RankingMethod",
    "register_ranking_method",
    "get_ranking_method",
    "list_ranking_methods",
]

