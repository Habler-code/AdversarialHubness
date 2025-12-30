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

"""
Example: Custom Detector Plugin

This example demonstrates how to create and register a custom detector
that can be used seamlessly with HubScan's detection pipeline.
"""

import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING

from hubscan.core.detectors import Detector, DetectorResult, register_detector
from hubscan.core.io.metadata import Metadata

if TYPE_CHECKING:
    from hubscan.core.io.vector_index import VectorIndex


class CustomScoreDetector(Detector):
    """
    Custom detector that computes a simple score based on document length.
    
    This is a simple example - you can implement any custom detection logic here.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        score_multiplier: float = 1.0,
    ):
        """
        Initialize custom detector.
        
        Args:
            enabled: Whether detector is enabled
            score_multiplier: Multiplier for scores
        """
        super().__init__(enabled)
        self.score_multiplier = score_multiplier
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Run custom detection.
        
        Args:
            index: VectorIndex instance
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            metadata: Optional document metadata
            **kwargs: Additional arguments (including ranking_method, etc.)
            
        Returns:
            DetectorResult with scores
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        N = len(doc_embeddings)
        
        # Example: Compute scores based on embedding norm
        # (This is just an example - implement your actual detection logic)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        
        # Normalize to [0, 1] range
        if doc_norms.max() > doc_norms.min():
            normalized_scores = (doc_norms - doc_norms.min()) / (doc_norms.max() - doc_norms.min())
        else:
            normalized_scores = np.zeros(N)
        
        # Apply multiplier
        scores = normalized_scores * self.score_multiplier
        
        result_metadata: Dict[str, Any] = {
            "detector_type": "custom_score",
            "score_multiplier": self.score_multiplier,
            "mean_score": float(scores.mean()),
            "max_score": float(scores.max()),
        }
        
        return DetectorResult(scores=scores, metadata=result_metadata)


# Register custom detector
def register_custom_detector():
    """Register custom detector."""
    register_detector("custom_score", CustomScoreDetector)
    print("Registered custom detector: custom_score")


if __name__ == "__main__":
    # Register the custom detector
    register_custom_detector()
    
    # Now this detector can be used in config files:
    # detectors:
    #   custom_score:
    #     enabled: true
    #     score_multiplier: 1.5
    #
    # And add it to scoring weights:
    # scoring:
    #   weights:
    #     custom_score: 0.1
    
    from hubscan.core.detectors import list_detectors
    print(f"All available detectors: {list_detectors()}")

