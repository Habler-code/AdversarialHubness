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

"""Detector registry for plugin system."""

from typing import Dict, Type, Optional, List
from .base import Detector

# Registry for detectors
_DETECTOR_REGISTRY: Dict[str, Type[Detector]] = {}


def register_detector(name: str, detector_class: Type[Detector]):
    """
    Register a custom detector class.
    
    Args:
        name: Unique name for the detector
        detector_class: Detector class (subclass of Detector)
    """
    if name in _DETECTOR_REGISTRY:
        import warnings
        warnings.warn(f"Detector '{name}' is already registered. Overwriting.")
    
    if not issubclass(detector_class, Detector):
        raise TypeError(f"Detector class must be a subclass of Detector, got {type(detector_class)}")
    
    _DETECTOR_REGISTRY[name] = detector_class


def get_detector_class(name: str) -> Optional[Type[Detector]]:
    """
    Get a registered detector class.
    
    Args:
        name: Name of the detector
        
    Returns:
        Detector class if found, None otherwise
    """
    return _DETECTOR_REGISTRY.get(name)


def list_detectors() -> List[str]:
    """List all registered detector names."""
    return list(_DETECTOR_REGISTRY.keys())


# Register built-in detectors
from .hubness import HubnessDetector
from .cluster_spread import ClusterSpreadDetector
from .stability import StabilityDetector
from .dedup import DedupDetector

register_detector("hubness", HubnessDetector)
register_detector("cluster_spread", ClusterSpreadDetector)
register_detector("stability", StabilityDetector)
register_detector("dedup", DedupDetector)

__all__ = [
    "register_detector",
    "get_detector_class",
    "list_detectors",
]

