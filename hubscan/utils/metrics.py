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

"""Statistical metrics utilities."""

import numpy as np
from typing import Tuple


def robust_zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute robust z-scores using median and MAD.
    
    Returns:
        z_scores: Array of z-scores
        median: Median value
        mad: Median Absolute Deviation
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Avoid division by zero
    if mad == 0:
        mad = np.finfo(float).eps
    
    z_scores = (values - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal dist
    return z_scores, median, mad


def compute_percentile(values: np.ndarray, percentile: float) -> float:
    """Compute percentile value."""
    return np.percentile(values, 100 * (1 - percentile))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (for cosine similarity)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
    return vectors / norms

