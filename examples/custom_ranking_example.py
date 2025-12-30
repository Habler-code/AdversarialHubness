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
Example: Custom Ranking Method Plugin

This example demonstrates how to create and register a custom ranking method
that can be used seamlessly with HubScan's detection pipeline.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from hubscan.core.ranking import register_ranking_method, RankingMethod
from hubscan.core.io.vector_index import VectorIndex


class WeightedVectorRanking:
    """
    Custom ranking method that applies a weight multiplier to vector search results.
    
    This is a simple example - you can implement any custom retrieval logic here.
    """
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        weight_multiplier: float = 1.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform weighted vector search.
        
        Args:
            index: VectorIndex instance
            query_vectors: Query embeddings
            query_texts: Query texts (not used in this example)
            k: Number of results
            weight_multiplier: Multiplier to apply to distances/similarities
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if query_vectors is None:
            raise ValueError("query_vectors required for weighted vector search")
        
        # Perform standard vector search
        distances, indices = index.search(query_vectors, k)
        
        # Apply weight multiplier
        # For similarity metrics (cosine, IP), multiply scores
        # For distance metrics (L2), divide distances (making them smaller = closer)
        weighted_distances = distances * weight_multiplier
        
        metadata = {
            "ranking_method": "weighted_vector",
            "weight_multiplier": weight_multiplier,
        }
        
        return weighted_distances, indices, metadata




# Register custom ranking methods
def register_custom_methods():
    """Register custom ranking methods."""
    register_ranking_method("weighted_vector", WeightedVectorRanking())
    print("Registered custom ranking methods: weighted_vector")
    print("Note: For reranking examples, see custom_reranking_example.py")


if __name__ == "__main__":
    # Register the custom methods
    register_custom_methods()
    
    # Now this method can be used in config files:
    # ranking:
    #   method: weighted_vector
    #   custom_params:
    #     weight_multiplier: 1.5
    
    from hubscan.core.ranking import list_ranking_methods
    print(f"All available ranking methods: {list_ranking_methods()}")

