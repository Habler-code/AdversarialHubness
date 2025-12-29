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

"""FAISS index operations."""

import faiss
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from ...utils.metrics import normalize_vectors


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str,
    metric: str = "cosine",
    params: Optional[Dict[str, Any]] = None,
) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Args:
        embeddings: Embeddings array of shape (N, D)
        index_type: Type of index ("hnsw", "ivf_pq", "flat")
        metric: Distance metric ("cosine", "ip", "l2")
        params: Index-specific parameters
        
    Returns:
        FAISS index
    """
    if params is None:
        params = {}
    
    d = embeddings.shape[1]
    
    # Normalize for cosine similarity
    if metric == "cosine":
        embeddings = normalize_vectors(embeddings)
        metric_type = faiss.METRIC_INNER_PRODUCT
    elif metric == "ip":
        metric_type = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        metric_type = faiss.METRIC_L2
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    if index_type == "flat":
        if metric == "l2":
            index = faiss.IndexFlatL2(d)
        else:
            index = faiss.IndexFlatIP(d)
    
    elif index_type == "hnsw":
        M = params.get("M", 32)
        efConstruction = params.get("efConstruction", 200)
        
        index = faiss.IndexHNSWFlat(d, M, metric_type)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = params.get("efSearch", 128)
    
    elif index_type == "ivf_pq":
        nlist = params.get("nlist", 4096)
        m = params.get("m", 64)  # Number of subquantizers
        nbits = params.get("nbits", 8)  # Bits per subquantizer
        
        quantizer = faiss.IndexFlatL2(d) if metric == "l2" else faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        index.nprobe = params.get("nprobe", 16)
    
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Train if needed
    if index_type == "ivf_pq":
        index.train(embeddings)
    
    # Add vectors
    index.add(embeddings)
    
    return index


def load_faiss_index(path: str) -> faiss.Index:
    """Load FAISS index from file."""
    return faiss.read_index(path)


def save_faiss_index(index: faiss.Index, path: str):
    """Save FAISS index to file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)

