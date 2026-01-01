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

"""Modality-aware scorer for detecting cross-modal hubs."""

import numpy as np
from typing import Dict, Any

from .base import ScorerResult
from ..accumulator import BucketedHubnessAccumulator
from .....utils.metrics import robust_zscore


class ModalityAwareScorer:
    """Scorer that detects cross-modal hubs.
    
    Identifies documents that appear as hubs for queries of a different
    modality (e.g., text document appearing in image query results).
    This can indicate cross-modal hub attacks in multimodal systems.
    """
    
    def score(
        self,
        accumulator: BucketedHubnessAccumulator,
        config: Dict[str, Any],
        **kwargs,
    ) -> ScorerResult:
        """Compute modality-aware hubness scores.
        
        Args:
            accumulator: Accumulated hit data
            config: Configuration with keys:
                - enabled: Whether modality-aware scoring is enabled
                - cross_modal_penalty: Penalty multiplier for cross-modal hubs
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ScorerResult with cross-modal penalty scores and metadata
        """
        enabled = config.get("enabled", False)
        if not enabled:
            return ScorerResult(
                scores=np.zeros(accumulator.num_docs, dtype=np.float32),
                metadata={"enabled": False}
            )
        
        cross_modal_penalty = config.get("cross_modal_penalty", 1.5)
        
        N = accumulator.num_docs
        cross_modal_rates = accumulator.compute_cross_modal_rates()
        modality_hub_rates = accumulator.compute_modality_hub_rates()
        
        # Initialize scores
        final_scores = np.zeros(N, dtype=np.float32)
        cross_modal_z = np.zeros(N, dtype=np.float32)
        
        if cross_modal_rates.max() > 0:
            cross_modal_z, _, _ = robust_zscore(cross_modal_rates)
            # Apply penalty for cross-modal hubs
            cross_modal_penalty_scores = np.maximum(0, cross_modal_z) * (cross_modal_penalty - 1.0)
            final_scores += cross_modal_penalty_scores
        
        metadata = {
            "enabled": True,
            "cross_modal_z": cross_modal_z.tolist(),
            "modalities": list(modality_hub_rates.keys()),
        }
        
        return ScorerResult(scores=final_scores, metadata=metadata)

