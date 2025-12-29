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

"""Base detector interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from ..io.metadata import Metadata


class DetectorResult:
    """Result from a detector."""
    
    def __init__(
        self,
        scores: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize detector result.
        
        Args:
            scores: Array of scores per document (N,)
            metadata: Optional additional metadata (e.g., example queries)
        """
        self.scores = scores
        self.metadata = metadata or {}
    
    def get_score(self, doc_idx: int) -> float:
        """Get score for specific document."""
        return float(self.scores[doc_idx])
    
    def get_metadata(self, doc_idx: int, key: str, default: Any = None) -> Any:
        """Get metadata value for specific document."""
        if key not in self.metadata:
            return default
        value = self.metadata[key]
        if isinstance(value, (list, np.ndarray)):
            if doc_idx < len(value):
                return value[doc_idx]
            return default
        return value


class Detector(ABC):
    """Base class for detectors."""
    
    def __init__(self, enabled: bool = True):
        """Initialize detector."""
        self.enabled = enabled
    
    @abstractmethod
    def detect(
        self,
        index: Any,  # faiss.Index
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Run detection.
        
        Args:
            index: FAISS index
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            metadata: Optional document metadata
            **kwargs: Additional detector-specific arguments
            
        Returns:
            DetectorResult with scores and optional metadata
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self.enabled

