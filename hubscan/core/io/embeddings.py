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

"""Embeddings I/O operations."""

import numpy as np
from pathlib import Path
from typing import Optional
import pandas as pd

from ...utils.metrics import normalize_vectors


def load_embeddings(path: str, normalize: bool = False) -> np.ndarray:
    """
    Load embeddings from file.
    
    Supports:
    - .npy: NumPy array
    - .npz: NumPy compressed array
    - .parquet: Parquet file with embedding column
    
    Args:
        path: Path to embeddings file
        normalize: Whether to normalize vectors to unit length
        
    Returns:
        Embeddings array of shape (N, D)
    """
    path_obj = Path(path)
    
    if path_obj.suffix == ".npy":
        embeddings = np.load(path)
    elif path_obj.suffix == ".npz":
        data = np.load(path)
        # Assume first array is embeddings
        key = list(data.keys())[0]
        embeddings = data[key]
    elif path_obj.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        # Look for common embedding column names
        embedding_cols = [col for col in df.columns if "embedding" in col.lower() or "vector" in col.lower()]
        if not embedding_cols:
            raise ValueError("No embedding column found in parquet file")
        # Convert list column to numpy array
        embeddings = np.array(df[embedding_cols[0]].tolist())
    else:
        raise ValueError(f"Unsupported embeddings format: {path_obj.suffix}")
    
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    if normalize:
        embeddings = normalize_vectors(embeddings)
    
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):
    """Save embeddings to file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if path_obj.suffix == ".npy":
        np.save(path, embeddings)
    elif path_obj.suffix == ".npz":
        np.savez_compressed(path, embeddings=embeddings)
    else:
        raise ValueError(f"Unsupported embeddings format: {path_obj.suffix}")

