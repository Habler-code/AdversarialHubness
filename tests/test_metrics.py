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

"""Tests for detection metrics."""

import pytest
import numpy as np

from hubscan.core.metrics.detection_metrics import (
    compute_auc_roc,
    compute_auc_pr,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_detection_metrics,
)


class TestDetectionMetrics:
    """Tests for detection performance metrics."""
    
    def test_auc_roc_perfect(self):
        """Test AUC-ROC with perfect separation."""
        pytest.importorskip("sklearn")
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # Perfect separation
        
        auc = compute_auc_roc(y_true, y_scores)
        
        # Perfect separation should give AUC = 1.0
        assert abs(auc - 1.0) < 1e-6
    
    def test_auc_roc_random(self):
        """Test AUC-ROC with random scores."""
        pytest.importorskip("sklearn")
        
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5])  # Random
        
        auc = compute_auc_roc(y_true, y_scores)
        
        # Random should give AUC ≈ 0.5
        assert 0.4 <= auc <= 0.6
    
    def test_auc_pr(self):
        """Test AUC-PR computation."""
        pytest.importorskip("sklearn")
        
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        auc_pr = compute_auc_pr(y_true, y_scores)
        
        assert 0.0 <= auc_pr <= 1.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm, metrics = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert "tp" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics
        
        # Manual check: TP=2, FP=1, TN=2, FN=1
        assert metrics["tp"] == 2
        assert metrics["fp"] == 1
        assert metrics["tn"] == 2
        assert metrics["fn"] == 1
    
    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        per_class = compute_per_class_metrics(y_true, y_pred)
        
        assert "positive" in per_class
        assert "negative" in per_class
        
        for class_name in ["positive", "negative"]:
            assert "precision" in per_class[class_name]
            assert "recall" in per_class[class_name]
            assert "f1" in per_class[class_name]
            assert "support" in per_class[class_name]
    
    def test_compute_detection_metrics(self):
        """Test comprehensive detection metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
        
        metrics = compute_detection_metrics(y_true, y_scores, threshold=0.5)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "fpr" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class" in metrics
        
        if metrics.get("auc_roc") is not None:
            assert 0.0 <= metrics["auc_roc"] <= 1.0
        if metrics.get("auc_pr") is not None:
            assert 0.0 <= metrics["auc_pr"] <= 1.0
    
    def test_detection_metrics_empty(self):
        """Test detection metrics with empty arrays."""
        y_true = np.array([])
        y_scores = np.array([])
        
        # Should handle empty arrays gracefully - check that it doesn't crash
        # Empty arrays will cause sklearn to raise ValueError, so we check for that
        try:
            metrics = compute_detection_metrics(y_true, y_scores, threshold=0.5)
            # If it succeeds, check basic structure
            assert "precision" in metrics
            assert "recall" in metrics
        except (ValueError, IndexError):
            # Expected for empty arrays - sklearn requires at least 1 sample
            pass

