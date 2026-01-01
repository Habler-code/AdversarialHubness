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

"""Global hubness scorer - computes global z-scores from hub rates."""

import numpy as np
from typing import Dict, Any

from .base import ScorerResult
from ..accumulator import BucketedHubnessAccumulator
from .....utils.metrics import robust_zscore


class GlobalHubnessScorer:
    """Scorer that computes global hubness z-scores.
    
    This is the primary scoring component that measures how unusual
    each document's hub rate is compared to the global distribution.
    """
    
    def score(
        self,
        accumulator: BucketedHubnessAccumulator,
        config: Dict[str, Any],
        **kwargs,
    ) -> ScorerResult:
        """Compute global hubness z-scores.
        
        Args:
            accumulator: Accumulated hit data
            config: Configuration (not used for global scorer)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ScorerResult with z-scores and hub rate metadata
        """
        # Compute hub rates
        hub_rates = accumulator.compute_hub_rates()
        
        # Compute z-scores using robust statistics
        hub_z, median, mad = robust_zscore(hub_rates)
        
        metadata = {
            "hub_rate": hub_rates.tolist(),
            "hub_z": hub_z.tolist(),
            "weighted_hits": hub_rates.tolist(),
            "median": float(median),
            "mad": float(mad),
            "example_queries": {str(k): v for k, v in accumulator.example_queries.items()},
            "use_rank_weights": accumulator.use_rank_weights,
            "use_distance_weights": accumulator.use_distance_weights,
        }
        
        return ScorerResult(scores=hub_z, metadata=metadata)

