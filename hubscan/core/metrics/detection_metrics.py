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

"""Detection performance metrics for adversarial hub detection."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_auc_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve (AUC-ROC).
    
    Args:
        y_true: Binary labels (0 or 1) of shape (N,)
        y_scores: Prediction scores of shape (N,)
        
    Returns:
        AUC-ROC score (0.0 to 1.0)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn required for AUC-ROC. Install with: pip install scikit-learn"
        )
    
    if len(y_true) == 0:
        return 0.0
    
    # Check if we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        # Only one class present, return 0.5 (random)
        return 0.5
    
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        # Fallback if sklearn fails
        return 0.5


def compute_auc_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Area Under Precision-Recall Curve (AUC-PR).
    
    Args:
        y_true: Binary labels (0 or 1) of shape (N,)
        y_scores: Prediction scores of shape (N,)
        
    Returns:
        AUC-PR score (0.0 to 1.0)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn required for AUC-PR. Install with: pip install scikit-learn"
        )
    
    if len(y_true) == 0:
        return 0.0
    
    # Check if we have positive class
    if np.sum(y_true) == 0:
        # No positive examples, return 0
        return 0.0
    
    try:
        return float(average_precision_score(y_true, y_scores))
    except ValueError:
        return 0.0


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute confusion matrix and extract TP/FP/TN/FN.
    
    Args:
        y_true: True binary labels (0 or 1) of shape (N,)
        y_pred: Predicted binary labels (0 or 1) of shape (N,)
        labels: Optional label order (default: [0, 1])
        
    Returns:
        Tuple of (confusion_matrix, metrics_dict) where:
        - confusion_matrix: 2x2 array [[TN, FP], [FN, TP]]
        - metrics_dict: Dictionary with TP, FP, TN, FN counts
    """
    if labels is None:
        labels = [0, 1]
    
    if not SKLEARN_AVAILABLE:
        # Manual computation
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        cm = np.array([[tn, fp], [fn, tp]])
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.shape == (1, 1):
            # Only one class present, expand to 2x2
            if labels[0] == 0:
                cm = np.array([[cm[0, 0], 0], [0, 0]])
            else:
                cm = np.array([[0, 0], [0, cm[0, 0]]])
        
        tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    
    return cm, metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics (precision, recall, F1).
    
    Args:
        y_true: True binary labels (0 or 1) of shape (N,)
        y_pred: Predicted binary labels (0 or 1) of shape (N,)
        class_labels: Optional class names (default: ["negative", "positive"])
        
    Returns:
        Dictionary mapping class names to metrics dictionaries
    """
    if class_labels is None:
        class_labels = ["negative", "positive"]
    
    _, cm_metrics = compute_confusion_matrix(y_true, y_pred)
    tp = cm_metrics["tp"]
    fp = cm_metrics["fp"]
    tn = cm_metrics["tn"]
    fn = cm_metrics["fn"]
    
    # Positive class metrics
    pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0.0
    
    # Negative class metrics
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0.0
    
    return {
        class_labels[1]: {  # Positive class
            "precision": float(pos_precision),
            "recall": float(pos_recall),
            "f1": float(pos_f1),
            "support": int(tp + fn),
        },
        class_labels[0]: {  # Negative class
            "precision": float(neg_precision),
            "recall": float(neg_recall),
            "f1": float(neg_f1),
            "support": int(tn + fp),
        },
    }


def compute_detection_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Compute comprehensive detection performance metrics.
    
    Args:
        y_true: True binary labels (0 or 1) of shape (N,)
        y_scores: Prediction scores of shape (N,)
        threshold: Threshold for binary classification (default: 0.5)
        class_labels: Optional class names (default: ["negative", "positive"])
        
    Returns:
        Dictionary with all detection metrics
    """
    # Convert scores to predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute confusion matrix
    cm, cm_metrics = compute_confusion_matrix(y_true, y_pred)
    
    # Compute basic metrics
    tp = cm_metrics["tp"]
    fp = cm_metrics["fp"]
    tn = cm_metrics["tn"]
    fn = cm_metrics["fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": float(fpr),
    }
    
    # Compute AUC metrics
    try:
        metrics["auc_roc"] = compute_auc_roc(y_true, y_scores)
        metrics["auc_pr"] = compute_auc_pr(y_true, y_scores)
    except ImportError:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None
    
    # Compute per-class metrics
    metrics["per_class"] = compute_per_class_metrics(y_true, y_pred, class_labels)
    
    return metrics

