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

"""Parallel retrieval from multiple indexes."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..io.vector_index import VectorIndex
from ...utils.logging import get_logger

logger = get_logger()


@dataclass
class MultiIndexResult:
    """Result from parallel retrieval across multiple indexes."""
    text_distances: Optional[np.ndarray] = None
    text_indices: Optional[np.ndarray] = None
    image_distances: Optional[np.ndarray] = None
    image_indices: Optional[np.ndarray] = None
    unified_distances: Optional[np.ndarray] = None
    unified_indices: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _retrieve_from_index(
    index: VectorIndex,
    query_vectors: np.ndarray,
    k: int,
    index_name: str,
) -> tuple:
    """Retrieve from a single index."""
    try:
        distances, indices = index.search(query_vectors, k)
        logger.debug(f"Retrieved {len(indices)} results from {index_name} index")
        return distances, indices, None
    except Exception as e:
        logger.warning(f"Error retrieving from {index_name} index: {e}")
        return None, None, str(e)


def parallel_retrieve(
    text_index: Optional[VectorIndex],
    image_index: Optional[VectorIndex],
    unified_index: Optional[VectorIndex],
    query_vectors: np.ndarray,
    k: int,
    unified_k: Optional[int] = None,
    parallel: bool = True,
) -> MultiIndexResult:
    """
    Perform parallel retrieval from multiple indexes.
    
    Args:
        text_index: Optional text index
        image_index: Optional image index
        unified_index: Optional unified/cross-modal index (recall backstop)
        query_vectors: Query embeddings (M, D)
        k: Number of results to retrieve per index
        unified_k: Number of results from unified index (defaults to k)
        parallel: Whether to retrieve in parallel (default: True)
        
    Returns:
        MultiIndexResult with distances and indices from each index
    """
    unified_k = unified_k or k
    
    result = MultiIndexResult()
    result.metadata = {
        "num_queries": len(query_vectors),
        "k": k,
        "unified_k": unified_k,
    }
    
    if not parallel:
        # Sequential retrieval
        if text_index:
            result.text_distances, result.text_indices, _ = _retrieve_from_index(
                text_index, query_vectors, k, "text"
            )
        if image_index:
            result.image_distances, result.image_indices, _ = _retrieve_from_index(
                image_index, query_vectors, k, "image"
            )
        if unified_index:
            result.unified_distances, result.unified_indices, _ = _retrieve_from_index(
                unified_index, query_vectors, unified_k, "unified"
            )
    else:
        # Parallel retrieval
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            if text_index:
                futures["text"] = executor.submit(
                    _retrieve_from_index, text_index, query_vectors, k, "text"
                )
            if image_index:
                futures["image"] = executor.submit(
                    _retrieve_from_index, image_index, query_vectors, k, "image"
                )
            if unified_index:
                futures["unified"] = executor.submit(
                    _retrieve_from_index, unified_index, query_vectors, unified_k, "unified"
                )
            
            for index_name, future in futures.items():
                distances, indices, error = future.result()
                if error:
                    logger.warning(f"Skipping {index_name} index due to error: {error}")
                    continue
                    
                if index_name == "text":
                    result.text_distances = distances
                    result.text_indices = indices
                elif index_name == "image":
                    result.image_distances = distances
                    result.image_indices = indices
                elif index_name == "unified":
                    result.unified_distances = distances
                    result.unified_indices = indices
    
    return result

