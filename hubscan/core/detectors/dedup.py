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

"""Deduplication detector - detects boilerplate and duplicates."""

import numpy as np
from collections import Counter
from typing import Optional, Dict, Any

from .base import Detector, DetectorResult
from ..io.metadata import Metadata
from ...utils.logging import get_logger

logger = get_logger()


class DedupDetector(Detector):
    """Detector for duplicates and boilerplate."""
    
    def __init__(
        self,
        enabled: bool = True,
        text_hash_field: Optional[str] = "text_hash",
        duplicate_threshold: float = 0.95,
        suppress_boilerplate: bool = True,
    ):
        """
        Initialize deduplication detector.
        
        Args:
            enabled: Whether detector is enabled
            text_hash_field: Metadata field name for text hash
            duplicate_threshold: L2 distance threshold for duplicates (if using embeddings)
            suppress_boilerplate: Whether to suppress obvious boilerplate
        """
        super().__init__(enabled)
        self.text_hash_field = text_hash_field
        self.duplicate_threshold = duplicate_threshold
        self.suppress_boilerplate = suppress_boilerplate
    
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
        Detect duplicates and boilerplate.
        
        Args:
            index: FAISS index (not used directly)
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (not used)
            k: Number of nearest neighbors
            metadata: Document metadata
            
        Returns:
            DetectorResult with boilerplate penalty scores
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        logger.info("Running deduplication detection")
        
        N = len(doc_embeddings)
        boilerplate_scores = np.zeros(N)
        
        # Method 1: Use text hash if available
        if metadata and self.text_hash_field and metadata.has_field(self.text_hash_field):
            hashes = metadata.get(self.text_hash_field)
            hash_counts = Counter(hashes)
            
            # Score based on frequency (higher frequency = more boilerplate)
            max_count = max(hash_counts.values()) if hash_counts else 1
            for i, hash_val in enumerate(hashes):
                count = hash_counts[hash_val]
                # Normalize to [0, 1] where 1 = most frequent
                boilerplate_scores[i] = count / max_count
        
        # Method 2: Use embedding similarity for approximate duplicates
        else:
            # Sample subset for efficiency
            sample_size = min(10000, N)
            sample_indices = np.random.choice(N, sample_size, replace=False)
            sample_embeddings = doc_embeddings[sample_indices]
            
            # Build a small index for fast similarity search
            import faiss
            from ..io.adapters import FAISSIndex
            
            d = doc_embeddings.shape[1]
            faiss_search_index = faiss.IndexFlatL2(d)
            faiss_search_index.add(sample_embeddings)
            search_index = FAISSIndex(faiss_search_index)
            
            # For each sampled doc, find nearest neighbors
            k_search = min(10, sample_size)
            distances, indices = search_index.search(sample_embeddings, k_search)
            
            # Count approximate duplicates (within threshold)
            for i, idx in enumerate(sample_indices):
                # Count neighbors within threshold
                close_neighbors = np.sum(distances[i] < self.duplicate_threshold)
                if close_neighbors > 1:  # Exclude self
                    boilerplate_scores[idx] = min(1.0, close_neighbors / k_search)
        
        logger.info(f"Deduplication detection complete. Mean boilerplate score: {boilerplate_scores.mean():.4f}")
        
        result_metadata: Dict[str, Any] = {
            "boilerplate_scores": boilerplate_scores.tolist(),
            "method": "text_hash" if (metadata and self.text_hash_field and metadata.has_field(self.text_hash_field)) else "embedding_similarity",
        }
        
        return DetectorResult(scores=boilerplate_scores, metadata=result_metadata)

