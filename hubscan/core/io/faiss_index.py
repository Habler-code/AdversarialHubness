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
FAISS index operations.

This module provides backward-compatible functions for FAISS index management.
The actual implementation has been consolidated into the FAISSIndex class
in adapters/faiss_adapter.py.

For new code, prefer using FAISSIndex directly:

    from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
    
    # Build a new index
    index = FAISSIndex.build(embeddings, index_type="flat", metric="cosine")
    
    # Load an existing index
    index = FAISSIndex.load("path/to/index.index")
    
    # Save an index
    index.save("path/to/index.index")

The functions below are kept for backward compatibility.
"""

import faiss
import numpy as np
from typing import Optional, Dict, Any

from .adapters.faiss_adapter import FAISSIndex


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str,
    metric: str = "cosine",
    params: Optional[Dict[str, Any]] = None,
) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Note: For new code, prefer FAISSIndex.build() which returns
    a wrapped index with the full VectorIndex interface.
    
    Args:
        embeddings: Embeddings array of shape (N, D)
        index_type: Type of index ("hnsw", "ivf_pq", "flat")
        metric: Distance metric ("cosine", "ip", "l2")
        params: Index-specific parameters
        
    Returns:
        Raw FAISS index object
    """
    return FAISSIndex.build_raw(embeddings, index_type, metric, params)


def load_faiss_index(path: str) -> faiss.Index:
    """
    Load FAISS index from file.
    
    Note: For new code, prefer FAISSIndex.load() which returns
    a wrapped index with the full VectorIndex interface.
    
    Args:
        path: Path to the saved index file
        
    Returns:
        Raw FAISS index object
    """
    return FAISSIndex.load_raw(path)


def save_faiss_index(index: faiss.Index, path: str) -> None:
    """
    Save FAISS index to file.
    
    Note: If you have a FAISSIndex instance, prefer using
    the instance method: index.save(path)
    
    Args:
        index: FAISS index object
        path: Path where the index will be saved
    """
    FAISSIndex.save_raw(index, path)


def wrap_faiss_index(index: faiss.Index) -> FAISSIndex:
    """
    Wrap a FAISS index in a VectorIndex adapter.
    
    This helper function provides backward compatibility and makes it
    easy to convert existing FAISS indices to the VectorIndex interface.
    
    Args:
        index: FAISS index object
        
    Returns:
        FAISSIndex adapter wrapping the FAISS index
    """
    return FAISSIndex(index)
