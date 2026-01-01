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


def robust_zscore(values: np.ndarray, max_zscore: float = 100.0) -> Tuple[np.ndarray, float, float]:
    """
    Compute robust z-scores using median and MAD.
    
    Args:
        values: Array of values
        max_zscore: Maximum allowed z-score to prevent numerical issues (default 100)
    
    Returns:
        z_scores: Array of z-scores
        median: Median value
        mad: Median Absolute Deviation
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    # Handle near-zero MAD
    if mad < 1e-15:
        # All values are essentially the same - return zeros
        return np.zeros_like(values), median, 0.0
    
    z_scores = (values - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal dist
    
    # Cap extreme z-scores to prevent numerical issues in downstream calculations
    z_scores = np.clip(z_scores, -max_zscore, max_zscore)
    
    return z_scores, median, mad


def compute_percentile(values: np.ndarray, percentile: float) -> float:
    """Compute percentile value."""
    return np.percentile(values, 100 * (1 - percentile))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (for cosine similarity)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
    return vectors / norms


def robust_zscore_sparse(
    values_dict: dict,
    population_size: int,
    default_value: float = 0.0,
    min_mad: float = 1.0,
) -> Tuple[dict, float, float]:
    """
    Compute robust z-scores for sparse data where most values are zero/default.
    
    This is optimized for the case where we have a sparse mapping of doc_id -> count
    and most documents have zero hits. Instead of materializing the full array,
    we compute statistics accounting for the implicit zeros.
    
    Args:
        values_dict: Sparse mapping of index -> value (e.g., doc_id -> hit_count)
        population_size: Total population size (including implicit zeros)
        default_value: The default value for items not in values_dict (typically 0)
        min_mad: Minimum MAD value to avoid extreme z-scores (default 1.0)
        
    Returns:
        z_scores_dict: Sparse mapping of index -> z-score (only for non-default values)
        median: Median value (accounting for implicit defaults)
        mad: Median Absolute Deviation
    """
    if not values_dict:
        return {}, default_value, 0.0
    
    # Get explicit values
    explicit_values = np.array(list(values_dict.values()))
    num_explicit = len(explicit_values)
    num_implicit = population_size - num_explicit
    
    if num_implicit < 0:
        # More explicit values than population - just use explicit
        num_implicit = 0
        population_size = num_explicit
    
    # Compute median accounting for implicit zeros
    # Sort explicit values
    sorted_explicit = np.sort(explicit_values)
    
    # Find median position
    median_pos = population_size / 2
    
    # Determine median
    if num_implicit > median_pos:
        # Median is in the implicit zeros
        median = default_value
    else:
        # Median is in the explicit values
        # Adjust position for implicit zeros
        adjusted_pos = median_pos - num_implicit
        if adjusted_pos >= num_explicit:
            adjusted_pos = num_explicit - 1
        median = sorted_explicit[int(adjusted_pos)]
    
    # Compute MAD
    # Deviations from explicit values
    explicit_deviations = np.abs(explicit_values - median)
    # Deviations from implicit zeros
    implicit_deviation = abs(default_value - median)
    
    # Combine deviations for MAD calculation
    if num_implicit > 0:
        all_deviations = np.concatenate([
            explicit_deviations,
            np.full(num_implicit, implicit_deviation)
        ])
    else:
        all_deviations = explicit_deviations
    
    mad = np.median(all_deviations)
    
    # Use minimum MAD to avoid extreme z-scores when most values are zero
    # This is important for sparse hubness distributions
    if mad < min_mad:
        mad = min_mad
    
    # Compute z-scores only for explicit (non-default) values
    scale = 1.4826 * mad  # Makes MAD consistent with std for normal dist
    z_scores_dict = {
        idx: float((val - median) / scale)
        for idx, val in values_dict.items()
    }
    
    return z_scores_dict, float(median), float(mad)


def robust_zscore_bucketed(
    bucket_values: dict,
    bucket_sizes: dict,
    default_value: float = 0.0,
) -> dict:
    """
    Compute robust z-scores separately for each bucket.
    
    Args:
        bucket_values: Nested dict {bucket_id: {doc_id: value, ...}, ...}
        bucket_sizes: Dict {bucket_id: population_size, ...}
        default_value: Default value for items not in bucket_values
        
    Returns:
        Dict {bucket_id: {doc_id: z_score, ...}, ...}
    """
    result = {}
    
    for bucket_id, values_dict in bucket_values.items():
        pop_size = bucket_sizes.get(bucket_id, len(values_dict))
        z_scores, _, _ = robust_zscore_sparse(values_dict, pop_size, default_value)
        result[bucket_id] = z_scores
    
    return result

