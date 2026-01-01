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

"""Base protocol and result types for hubness scorers."""

import numpy as np
from typing import Protocol, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..accumulator import BucketedHubnessAccumulator


@dataclass
class ScorerResult:
    """Result from a hubness scorer.
    
    Attributes:
        scores: Per-document scores to add to final score (shape: num_docs)
        metadata: Scorer-specific metadata for reporting
    """
    scores: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class HubnessScorer(Protocol):
    """Protocol for pluggable hubness scoring components.
    
    Scorers compute contribution to the final hubness score based on
    accumulated hit data. Multiple scorers can be composed together.
    
    Built-in scorers:
    - GlobalHubnessScorer: Global z-score computation
    - ConceptAwareScorer: Concept-specific hubness detection
    - ModalityAwareScorer: Cross-modal hub detection
    
    Custom scorers can be registered using:
    
        from hubscan.core.detectors.hubness.scorers import register_scorer
        
        class MyCustomScorer:
            def score(self, accumulator, config, **kwargs):
                # Custom scoring logic
                scores = np.zeros(accumulator.num_docs)
                # ... compute scores ...
                return ScorerResult(scores=scores, metadata={"custom": True})
        
        register_scorer("my_scorer", MyCustomScorer())
    
    Then enable it in the detector:
    
        detector = HubnessDetector(enabled_scorers=["global", "my_scorer"])
    """
    
    def score(
        self,
        accumulator: "BucketedHubnessAccumulator",
        config: Dict[str, Any],
        **kwargs,
    ) -> ScorerResult:
        """Compute scores from accumulated hubness data.
        
        Args:
            accumulator: Accumulated hit data from query processing.
                Provides access to:
                - global_hits: Dict[doc_id, weighted_hit_count]
                - concept_hits: Dict[concept_id, Dict[doc_id, weighted_hit_count]]
                - modality_hits: Dict[modality, Dict[doc_id, weighted_hit_count]]
                - cross_modal_hits: Dict[doc_id, cross_modal_hit_count]
                - total_queries: int
                - num_docs: int
                - compute_hub_rates() -> np.ndarray
                - compute_concept_hub_rates() -> Dict[int, np.ndarray]
                - compute_modality_hub_rates() -> Dict[str, np.ndarray]
                - compute_cross_modal_rates() -> np.ndarray
                - custom_data: Dict[str, Any] for custom scorer data
                
            config: Scorer-specific configuration parameters
            
            **kwargs: Additional parameters passed from detector
            
        Returns:
            ScorerResult with:
            - scores: np.ndarray of shape (num_docs,) to add to final score
            - metadata: Dict with scorer-specific metadata for reporting
        """
        ...


# Backward compatibility alias
ScorerProtocol = HubnessScorer
