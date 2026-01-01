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

"""Hubness detection module with pluggable scoring components.

This module provides a modular hubness detector with a registry pattern
for scoring components, following the same pattern as ranking methods.

Example usage - registering a custom scorer:

    from hubscan.core.detectors.hubness import (
        HubnessDetector,
        BucketedHubnessAccumulator,
        HubnessScorer,
        ScorerResult,
        register_scorer,
    )
    
    class MyCustomScorer:
        def score(self, accumulator, config, **kwargs):
            scores = np.zeros(accumulator.num_docs)
            # Custom scoring logic
            return ScorerResult(scores=scores, metadata={"custom": True})
    
    register_scorer("my_custom", MyCustomScorer())
"""

from .detector import HubnessDetector
from .accumulator import BucketedHubnessAccumulator
from .scorers import (
    HubnessScorer,
    ScorerProtocol,  # Backward compatibility
    ScorerResult,
    register_scorer,
    get_scorer,
    list_scorers,
)

__all__ = [
    "HubnessDetector",
    "BucketedHubnessAccumulator",
    "HubnessScorer",
    "ScorerProtocol",
    "ScorerResult",
    "register_scorer",
    "get_scorer",
    "list_scorers",
]
