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

"""Score combination logic."""

import numpy as np
from typing import Dict, Optional

from ..detectors.base import DetectorResult
from ...config import ScoringWeights


def combine_scores(
    detector_results: Dict[str, DetectorResult],
    weights: ScoringWeights,
) -> np.ndarray:
    """
    Combine scores from multiple detectors.
    
    Args:
        detector_results: Dictionary mapping detector names to results
        weights: Scoring weights configuration
        
    Returns:
        Combined risk scores
    """
    # Get the number of documents from the first detector
    if not detector_results:
        raise ValueError("No detector results provided")
    
    first_result = next(iter(detector_results.values()))
    num_docs = len(first_result.scores)
    
    combined = np.zeros(num_docs)
    
    # Hubness z-score
    if "hubness" in detector_results:
        combined += weights.hub_z * detector_results["hubness"].scores
    
    # Cluster spread
    if "cluster_spread" in detector_results:
        combined += weights.cluster_spread * detector_results["cluster_spread"].scores
    
    # Stability
    if "stability" in detector_results:
        combined += weights.stability * detector_results["stability"].scores
    
    # Boilerplate penalty (subtract)
    if "dedup" in detector_results:
        combined -= weights.boilerplate * detector_results["dedup"].scores
    
    return combined


def compute_risk_score(
    hub_z: float,
    cluster_spread: float = 0.0,
    stability: float = 0.0,
    boilerplate: float = 0.0,
    weights: Optional[ScoringWeights] = None,
) -> float:
    """
    Compute risk score for a single document.
    
    Args:
        hub_z: Hubness z-score
        cluster_spread: Cluster spread score
        stability: Stability score
        boilerplate: Boilerplate score
        weights: Scoring weights (uses defaults if None)
        
    Returns:
        Combined risk score
    """
    if weights is None:
        weights = ScoringWeights()
    
    score = (
        weights.hub_z * hub_z
        + weights.cluster_spread * cluster_spread
        + weights.stability * stability
        - weights.boilerplate * boilerplate
    )
    
    return score

