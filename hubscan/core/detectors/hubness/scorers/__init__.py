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

"""Hubness scorer plugin system for custom scoring algorithms.

This module provides a registry pattern for pluggable hubness scorers,
following the same pattern as the ranking method plugin system.

Example usage:
    
    from hubscan.core.detectors.hubness.scorers import (
        HubnessScorer, ScorerResult, register_scorer
    )
    
    class MyCustomScorer:
        '''Custom scorer for domain-specific detection.'''
        
        def score(self, accumulator, config, **kwargs):
            scores = np.zeros(accumulator.num_docs)
            # Custom scoring logic using accumulator data
            # e.g., use accumulator.custom_data["my_feature"]
            return ScorerResult(scores=scores, metadata={"custom": True})
    
    # Register the custom scorer
    register_scorer("my_custom", MyCustomScorer())
"""

from typing import Dict, List, Optional
import warnings

from .base import HubnessScorer, ScorerProtocol, ScorerResult

# Registry for scorers
_SCORERS: Dict[str, HubnessScorer] = {}


def register_scorer(name: str, scorer: HubnessScorer):
    """
    Register a custom hubness scorer.
    
    Args:
        name: Unique name for the scorer
        scorer: HubnessScorer instance implementing the score protocol
        
    Example:
        >>> class MyScorer:
        ...     def score(self, accumulator, config, **kwargs):
        ...         return ScorerResult(scores=np.zeros(accumulator.num_docs))
        >>> register_scorer("my_scorer", MyScorer())
    """
    if name in _SCORERS:
        warnings.warn(f"Scorer '{name}' is already registered. Overwriting.")
    _SCORERS[name] = scorer


def get_scorer(name: str) -> Optional[HubnessScorer]:
    """
    Get a registered scorer by name.
    
    Args:
        name: Name of the scorer
        
    Returns:
        HubnessScorer instance if found, None otherwise
    """
    return _SCORERS.get(name)


def list_scorers() -> List[str]:
    """List all registered scorer names."""
    return list(_SCORERS.keys())


def _register_builtin_scorers():
    """Register built-in scorers."""
    from .global_scorer import GlobalHubnessScorer
    from .concept_scorer import ConceptAwareScorer
    from .modality_scorer import ModalityAwareScorer
    
    register_scorer("global", GlobalHubnessScorer())
    register_scorer("concept_aware", ConceptAwareScorer())
    register_scorer("modality_aware", ModalityAwareScorer())


# Auto-register built-in scorers on import
_register_builtin_scorers()


__all__ = [
    "HubnessScorer",
    "ScorerProtocol",  # Backward compatibility
    "ScorerResult",
    "register_scorer",
    "get_scorer",
    "list_scorers",
]
