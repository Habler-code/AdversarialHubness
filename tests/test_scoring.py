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

"""Tests for scoring and thresholding."""

import pytest
import numpy as np

from hubscan.core.scoring import combine_scores, compute_risk_score, apply_thresholds, Verdict
from hubscan.core.detectors.base import DetectorResult
from hubscan.config import ScoringWeights, ThresholdsConfig


@pytest.fixture
def sample_detector_results():
    """Create sample detector results."""
    num_docs = 100
    return {
        "hubness": DetectorResult(
            scores=np.array([6.0, 4.0, 2.0] + [0.0] * 97),
            metadata={"hub_rate": [0.15, 0.10, 0.05] + [0.01] * 97}
        ),
        "cluster_spread": DetectorResult(
            scores=np.array([0.8, 0.6, 0.4] + [0.2] * 97)
        ),
        "stability": DetectorResult(
            scores=np.array([0.9, 0.7, 0.5] + [0.3] * 97)
        ),
        "dedup": DetectorResult(
            scores=np.array([0.1, 0.2, 0.3] + [0.0] * 97)
        )
    }


def test_combine_scores(sample_detector_results):
    """Test score combination."""
    weights = ScoringWeights(
        hub_z=0.6,
        cluster_spread=0.2,
        stability=0.2,
        boilerplate=0.3
    )
    
    combined = combine_scores(sample_detector_results, weights)
    
    assert len(combined) == 100
    assert combined[0] > combined[1] > combined[2]  # Should be decreasing
    assert np.all(combined >= 0)  # Should be non-negative


def test_combine_scores_partial_detectors():
    """Test score combination with partial detectors."""
    results = {
        "hubness": DetectorResult(scores=np.array([5.0, 3.0, 1.0]))
    }
    weights = ScoringWeights()
    
    combined = combine_scores(results, weights)
    assert len(combined) == 3


def test_combine_scores_empty():
    """Test score combination with empty results."""
    with pytest.raises(ValueError):
        combine_scores({}, ScoringWeights())


def test_compute_risk_score():
    """Test risk score computation."""
    weights = ScoringWeights()
    
    score = compute_risk_score(
        hub_z=6.0,
        cluster_spread=0.8,
        stability=0.9,
        boilerplate=0.1,
        weights=weights
    )
    
    assert isinstance(score, float)
    assert score > 0


def test_compute_risk_score_default_weights():
    """Test risk score with default weights."""
    score = compute_risk_score(hub_z=5.0)
    assert isinstance(score, float)


def test_apply_thresholds_percentile(sample_detector_results):
    """Test threshold application with percentile policy."""
    combined_scores = np.array([8.0, 5.0, 2.0] + [1.0] * 97)
    config = ThresholdsConfig(policy="percentile", percentile=0.05)
    
    verdicts = apply_thresholds(
        sample_detector_results,
        combined_scores,
        config
    )
    
    assert len(verdicts) == 100
    assert isinstance(verdicts[0], Verdict)


def test_apply_thresholds_z_score(sample_detector_results):
    """Test threshold application with z-score policy."""
    combined_scores = np.array([8.0, 5.0, 2.0] + [1.0] * 97)
    config = ThresholdsConfig(policy="z_score", hub_z=5.0)
    
    verdicts = apply_thresholds(
        sample_detector_results,
        combined_scores,
        config
    )
    
    assert len(verdicts) == 100
    # First doc should be HIGH (hub_z=6.0 >= 5.0)
    assert verdicts[0] == Verdict.HIGH


def test_apply_thresholds_hybrid(sample_detector_results):
    """Test threshold application with hybrid policy."""
    combined_scores = np.array([8.0, 5.0, 2.0] + [1.0] * 97)
    config = ThresholdsConfig(
        policy="hybrid",
        hub_z=5.0,
        percentile=0.05
    )
    
    verdicts = apply_thresholds(
        sample_detector_results,
        combined_scores,
        config
    )
    
    assert len(verdicts) == 100
    # Should have HIGH, MEDIUM, and LOW verdicts
    verdict_values = set(verdicts.values())
    assert Verdict.HIGH in verdict_values or Verdict.MEDIUM in verdict_values


def test_apply_thresholds_no_hubness():
    """Test threshold application without hubness detector."""
    results = {
        "cluster_spread": DetectorResult(scores=np.array([0.8, 0.6, 0.4]))
    }
    combined_scores = np.array([5.0, 3.0, 1.0])
    config = ThresholdsConfig(policy="z_score", hub_z=5.0)
    
    verdicts = apply_thresholds(
        results,
        combined_scores,
        config,
        hub_z_scores=None
    )
    
    assert len(verdicts) == 3


def test_verdict_enum():
    """Test Verdict enum values."""
    assert Verdict.LOW.value == "LOW"
    assert Verdict.MEDIUM.value == "MEDIUM"
    assert Verdict.HIGH.value == "HIGH"

