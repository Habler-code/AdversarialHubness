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

"""Threshold application and verdict assignment."""

from enum import Enum
from typing import Dict, Any, Optional
import numpy as np

from ...config import ThresholdsConfig
from ..detectors.base import DetectorResult


class Verdict(str, Enum):
    """Risk verdict levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def apply_thresholds(
    detector_results: Dict[str, DetectorResult],
    combined_scores: np.ndarray,
    config: ThresholdsConfig,
    hub_z_scores: Optional[np.ndarray] = None,
) -> Dict[int, Verdict]:
    """
    Apply thresholds to assign verdicts.
    
    Args:
        detector_results: Dictionary of detector results
        combined_scores: Combined risk scores
        config: Threshold configuration
        hub_z_scores: Optional hubness z-scores (if not in detector_results)
        
    Returns:
        Dictionary mapping document index to verdict
    """
    if hub_z_scores is None and "hubness" in detector_results:
        hub_z_scores = detector_results["hubness"].scores
    elif hub_z_scores is None:
        hub_z_scores = np.zeros(len(combined_scores))
    
    verdicts: Dict[int, Verdict] = {}
    
    for doc_idx in range(len(combined_scores)):
        hub_z = hub_z_scores[doc_idx]
        
        if config.policy == "percentile":
            threshold = np.percentile(combined_scores, 100 * (1 - config.percentile))
            verdict = Verdict.HIGH if combined_scores[doc_idx] >= threshold else Verdict.LOW
        
        elif config.policy == "z_score":
            verdict = Verdict.HIGH if hub_z >= config.hub_z else Verdict.LOW
        
        elif config.policy == "hybrid":
            # Either condition triggers HIGH
            percentile_threshold = np.percentile(combined_scores, 100 * (1 - config.percentile))
            high_by_percentile = combined_scores[doc_idx] >= percentile_threshold
            high_by_zscore = hub_z >= config.hub_z
            
            if high_by_percentile or high_by_zscore:
                verdict = Verdict.HIGH
            elif combined_scores[doc_idx] >= percentile_threshold * 0.5 or hub_z >= config.hub_z * 0.5:
                verdict = Verdict.MEDIUM
            else:
                verdict = Verdict.LOW
        
        else:
            verdict = Verdict.LOW
        
        verdicts[doc_idx] = verdict
    
    return verdicts

