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
from typing import Dict, Optional, List

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
    
    # Concept-specific hub z-score (if enabled and weight > 0)
    if "hubness" in detector_results and weights.concept_hub_z > 0:
        hubness_meta = detector_results["hubness"].metadata or {}
        concept_aware = hubness_meta.get("concept_aware", {})
        if concept_aware.get("enabled", False):
            max_concept_hub_z = concept_aware.get("max_concept_hub_z", [])
            if max_concept_hub_z:
                combined += weights.concept_hub_z * np.array(max_concept_hub_z)
    
    # Cross-modal penalty (if enabled and weight > 0)
    if "hubness" in detector_results and weights.cross_modal > 0:
        hubness_meta = detector_results["hubness"].metadata or {}
        modality_aware = hubness_meta.get("modality_aware", {})
        if modality_aware.get("enabled", False):
            cross_modal_flags = modality_aware.get("cross_modal_flags", [])
            if cross_modal_flags:
                # Add penalty for cross-modal documents
                cross_modal_array = np.array(cross_modal_flags, dtype=np.float32)
                combined += weights.cross_modal * cross_modal_array * detector_results["hubness"].scores
    
    return combined


def compute_risk_score(
    hub_z: float,
    cluster_spread: float = 0.0,
    stability: float = 0.0,
    boilerplate: float = 0.0,
    concept_hub_z: float = 0.0,
    is_cross_modal: bool = False,
    weights: Optional[ScoringWeights] = None,
) -> float:
    """
    Compute risk score for a single document.
    
    Args:
        hub_z: Hubness z-score
        cluster_spread: Cluster spread score
        stability: Stability score
        boilerplate: Boilerplate score
        concept_hub_z: Max concept-specific hub z-score
        is_cross_modal: Whether document is flagged as cross-modal hub
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
    
    # Add concept-specific hub score if weight > 0
    if weights.concept_hub_z > 0:
        score += weights.concept_hub_z * concept_hub_z
    
    # Add cross-modal penalty if weight > 0 and doc is cross-modal
    if weights.cross_modal > 0 and is_cross_modal:
        score += weights.cross_modal * hub_z
    
    return score


def get_concept_modality_features(
    detector_results: Dict[str, DetectorResult],
) -> Dict[str, Dict[int, any]]:
    """
    Extract concept and modality features from detector results.
    
    Args:
        detector_results: Dictionary mapping detector names to results
        
    Returns:
        Dictionary with concept and modality features:
        - max_concept_hub_z: {doc_id: float}
        - top_concept_ids: {doc_id: int}
        - cross_modal_flags: {doc_id: bool}
        - concept_names: {concept_id: str}
    """
    result = {
        "max_concept_hub_z": {},
        "top_concept_ids": {},
        "cross_modal_flags": {},
        "cross_modal_ratios": {},
        "concept_names": {},
    }
    
    if "hubness" not in detector_results:
        return result
    
    hubness_meta = detector_results["hubness"].metadata or {}
    
    # Extract concept features
    concept_aware = hubness_meta.get("concept_aware", {})
    if concept_aware.get("enabled", False):
        max_concept_hub_z = concept_aware.get("max_concept_hub_z", [])
        top_concept_ids = concept_aware.get("top_concept_ids", [])
        concept_names = concept_aware.get("concept_names", {})
        
        for doc_id, z in enumerate(max_concept_hub_z):
            if z > 0:
                result["max_concept_hub_z"][doc_id] = z
        
        for doc_id, cid in enumerate(top_concept_ids):
            if cid >= 0:
                result["top_concept_ids"][doc_id] = cid
        
        result["concept_names"] = concept_names
    
    # Extract modality features
    modality_aware = hubness_meta.get("modality_aware", {})
    if modality_aware.get("enabled", False):
        cross_modal_flags = modality_aware.get("cross_modal_flags", [])
        cross_modal_ratios = modality_aware.get("cross_modal_ratios", {})
        
        for doc_id, flag in enumerate(cross_modal_flags):
            if flag:
                result["cross_modal_flags"][doc_id] = True
        
        result["cross_modal_ratios"] = {
            int(k): v for k, v in cross_modal_ratios.items()
        }
    
    return result

