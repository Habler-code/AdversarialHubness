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

"""Detector modules for adversarial hubness detection."""

from .base import Detector, DetectorResult
from .hubness import HubnessDetector
from .cluster_spread import ClusterSpreadDetector
from .stability import StabilityDetector
from .dedup import DedupDetector

__all__ = [
    "Detector",
    "DetectorResult",
    "HubnessDetector",
    "ClusterSpreadDetector",
    "StabilityDetector",
    "DedupDetector",
]

