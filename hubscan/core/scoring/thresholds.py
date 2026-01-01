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
    ranking_method: Optional[str] = None,
) -> Dict[int, Verdict]:
    """
    Apply thresholds to assign verdicts.
    
    Args:
        detector_results: Dictionary of detector results
        combined_scores: Combined risk scores
        config: Threshold configuration
        hub_z_scores: Optional hubness z-scores (if not in detector_results)
        ranking_method: Optional ranking method name for method-specific thresholds
        
    Returns:
        Dictionary mapping document index to verdict
    """
    if hub_z_scores is None and "hubness" in detector_results:
        hub_z_scores = detector_results["hubness"].scores
    elif hub_z_scores is None:
        hub_z_scores = np.zeros(len(combined_scores))
    
    # Get method-specific thresholds if available
    hub_z_threshold = config.hub_z
    percentile_threshold = config.percentile
    
    if ranking_method and config.method_specific and ranking_method in config.method_specific:
        method_thresholds = config.method_specific[ranking_method]
        hub_z_threshold = method_thresholds.get("hub_z", config.hub_z)
        percentile_threshold = method_thresholds.get("percentile", config.percentile)
    
    verdicts: Dict[int, Verdict] = {}
    
    for doc_idx in range(len(combined_scores)):
        hub_z = hub_z_scores[doc_idx]
        
        if config.policy == "percentile":
            threshold = np.percentile(combined_scores, 100 * (1 - percentile_threshold))
            verdict = Verdict.HIGH if combined_scores[doc_idx] >= threshold else Verdict.LOW
        
        elif config.policy == "z_score":
            verdict = Verdict.HIGH if hub_z >= hub_z_threshold else Verdict.LOW
        
        elif config.policy == "hybrid":
            # Either condition triggers HIGH
            percentile_val = np.percentile(combined_scores, 100 * (1 - percentile_threshold))
            high_by_percentile = combined_scores[doc_idx] >= percentile_val
            high_by_zscore = hub_z >= hub_z_threshold
            
            # Get MEDIUM ratio (default 0.5 if not set)
            medium_ratio = getattr(config, 'medium_ratio', 0.5)
            
            if high_by_percentile or high_by_zscore:
                verdict = Verdict.HIGH
            elif combined_scores[doc_idx] >= percentile_val * medium_ratio or hub_z >= hub_z_threshold * medium_ratio:
                verdict = Verdict.MEDIUM
            else:
                verdict = Verdict.LOW
        
        else:
            verdict = Verdict.LOW
        
        verdicts[doc_idx] = verdict
    
    return verdicts

