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

"""Memory-efficient accumulator for bucketed hubness statistics."""

import numpy as np
from typing import Optional, Dict, List
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BucketedHubnessAccumulator:
    """Memory-efficient accumulator for bucketed hubness statistics.
    
    Supports tracking hubness per:
    - Global (all queries)
    - Per-concept (queries grouped by semantic concept)
    - Per-modality (queries grouped by modality)
    
    Uses sparse storage to avoid O(N*Q) memory blowup.
    
    Custom scorers can store additional data in the `custom_data` dict.
    """
    num_docs: int
    use_rank_weights: bool = True
    use_distance_weights: bool = True
    
    # Sparse accumulators: doc_id -> weighted_hit_count
    global_hits: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    concept_hits: Dict[int, Dict[int, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    modality_hits: Dict[str, Dict[int, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    
    # Cross-modal tracking: doc_id -> cross_modal_hit_count
    cross_modal_hits: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    
    # Query counts per bucket for normalization
    concept_query_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    modality_query_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_queries: int = 0
    
    # Example queries per document
    example_queries: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    max_examples: int = 10
    
    # Custom data for user-defined scorers
    # Use this to store domain-specific data needed by custom scorers
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def add_hit(
        self,
        doc_id: int,
        query_id: int,
        rank: int,
        distance: float,
        concept_id: Optional[int] = None,
        query_modality: Optional[str] = None,
        doc_modality: Optional[str] = None,
    ):
        """Add a weighted hit to all relevant buckets."""
        # Compute weight based on rank and/or distance
        weight = 1.0
        if self.use_rank_weights:
            # Higher weight for higher ranks (rank 0 = most similar)
            weight *= 1.0 / (rank + 1)
        if self.use_distance_weights and distance is not None:
            # Higher weight for closer distances (assuming cosine distance)
            weight *= max(0, 1.0 - distance)
        
        # Global hits
        self.global_hits[doc_id] += weight
        
        # Track example query
        if len(self.example_queries[doc_id]) < self.max_examples:
            self.example_queries[doc_id].append(query_id)
        
        # Per-concept hits
        if concept_id is not None:
            self.concept_hits[concept_id][doc_id] += weight
        
        # Per-modality hits
        if query_modality is not None:
            self.modality_hits[query_modality][doc_id] += weight
        
        # Cross-modal hits (doc and query have different modalities)
        if doc_modality is not None and query_modality is not None:
            if doc_modality != query_modality:
                self.cross_modal_hits[doc_id] += weight
    
    def record_query(
        self,
        query_id: int,
        concept_id: Optional[int] = None,
        query_modality: Optional[str] = None,
    ):
        """Record query for normalization."""
        self.total_queries += 1
        if concept_id is not None:
            self.concept_query_counts[concept_id] += 1
        if query_modality is not None:
            self.modality_query_counts[query_modality] += 1
    
    def compute_hub_rates(self) -> np.ndarray:
        """Compute global hub rates for all documents."""
        rates = np.zeros(self.num_docs, dtype=np.float32)
        if self.total_queries > 0:
            for doc_id, hits in self.global_hits.items():
                rates[doc_id] = hits / self.total_queries
        return rates
    
    def compute_concept_hub_rates(self) -> Dict[int, np.ndarray]:
        """Compute per-concept hub rates."""
        result = {}
        for concept_id, doc_hits in self.concept_hits.items():
            rates = np.zeros(self.num_docs, dtype=np.float32)
            qc = self.concept_query_counts.get(concept_id, 1)
            for doc_id, hits in doc_hits.items():
                rates[doc_id] = hits / qc
            result[concept_id] = rates
        return result
    
    def compute_modality_hub_rates(self) -> Dict[str, np.ndarray]:
        """Compute per-modality hub rates."""
        result = {}
        for modality, doc_hits in self.modality_hits.items():
            rates = np.zeros(self.num_docs, dtype=np.float32)
            qc = self.modality_query_counts.get(modality, 1)
            for doc_id, hits in doc_hits.items():
                rates[doc_id] = hits / qc
            result[modality] = rates
        return result
    
    def compute_cross_modal_rates(self) -> np.ndarray:
        """Compute cross-modal hub rates."""
        rates = np.zeros(self.num_docs, dtype=np.float32)
        if self.total_queries > 0:
            for doc_id, hits in self.cross_modal_hits.items():
                rates[doc_id] = hits / self.total_queries
        return rates
    
    def set_custom_data(self, key: str, value: Any):
        """Store custom data for use by custom scorers.
        
        Args:
            key: Unique key for the custom data
            value: Data to store (can be any type)
        """
        self.custom_data[key] = value
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Retrieve custom data stored for custom scorers.
        
        Args:
            key: Key for the custom data
            default: Default value if key not found
            
        Returns:
            Stored data or default value
        """
        return self.custom_data.get(key, default)

