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

"""Tests for I/O operations."""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from hubscan.core.io import (
    load_embeddings,
    save_embeddings,
    load_metadata,
    Metadata,
)
from hubscan.core.io.faiss_index import build_faiss_index, load_faiss_index, save_faiss_index


def test_load_save_embeddings_npy(tmp_path):
    """Test loading and saving embeddings in .npy format."""
    embeddings = np.random.randn(100, 32).astype(np.float32)
    file_path = tmp_path / "embeddings.npy"
    
    save_embeddings(embeddings, str(file_path))
    loaded = load_embeddings(str(file_path))
    
    assert np.allclose(embeddings, loaded)
    assert loaded.dtype == np.float32


def test_load_embeddings_npz(tmp_path):
    """Test loading embeddings from .npz format."""
    embeddings = np.random.randn(50, 64).astype(np.float32)
    file_path = tmp_path / "embeddings.npz"
    
    np.savez_compressed(file_path, embeddings=embeddings)
    loaded = load_embeddings(str(file_path))
    
    assert loaded.shape == embeddings.shape
    assert loaded.dtype == np.float32


def test_load_embeddings_normalize(tmp_path):
    """Test loading embeddings with normalization."""
    embeddings = np.random.randn(10, 16).astype(np.float32)
    embeddings = embeddings * 10  # Make them large
    
    file_path = tmp_path / "embeddings.npy"
    np.save(file_path, embeddings)
    
    normalized = load_embeddings(str(file_path), normalize=True)
    
    # Check that norms are approximately 1
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_load_metadata_json(tmp_path):
    """Test loading metadata from JSON file."""
    metadata_data = {
        "doc_id": ["doc1", "doc2", "doc3"],
        "source": ["src1", "src2", "src1"],
        "text": ["text1", "text2", "text3"]
    }
    
    file_path = tmp_path / "metadata.json"
    with open(file_path, "w") as f:
        json.dump(metadata_data, f)
    
    metadata = load_metadata(str(file_path))
    
    assert metadata.num_docs == 3
    assert metadata.has_field("doc_id")
    assert metadata.get_field("doc_id", 0) == "doc1"


def test_load_metadata_jsonl(tmp_path):
    """Test loading metadata from JSONL file."""
    file_path = tmp_path / "metadata.jsonl"
    with open(file_path, "w") as f:
        f.write('{"doc_id": "doc1", "source": "src1"}\n')
        f.write('{"doc_id": "doc2", "source": "src2"}\n')
    
    metadata = load_metadata(str(file_path))
    
    assert metadata.num_docs == 2
    assert metadata.get_field("doc_id", 0) == "doc1"
    assert metadata.get_field("doc_id", 1) == "doc2"


def test_metadata_get_field():
    """Test metadata get_field method."""
    metadata = Metadata({
        "doc_id": ["doc1", "doc2"],
        "source": ["src1", "src2"]
    })
    
    assert metadata.get_field("doc_id", 0) == "doc1"
    assert metadata.get_field("doc_id", 1) == "doc2"
    assert metadata.get_field("nonexistent", 0) is None


def test_metadata_filter():
    """Test metadata filtering."""
    metadata = Metadata({
        "doc_id": ["doc1", "doc2", "doc3", "doc4"],
        "source": ["src1", "src2", "src1", "src2"]
    })
    
    filtered = metadata.filter(np.array([0, 2]))
    
    assert filtered.num_docs == 2
    assert filtered.get_field("doc_id", 0) == "doc1"
    assert filtered.get_field("doc_id", 1) == "doc3"


def test_build_faiss_index_flat():
    """Test building flat FAISS index."""
    embeddings = np.random.randn(100, 32).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = build_faiss_index(
        embeddings,
        index_type="flat",
        metric="cosine"
    )
    
    assert index.ntotal == 100


def test_build_faiss_index_hnsw():
    """Test building HNSW FAISS index."""
    embeddings = np.random.randn(100, 32).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = build_faiss_index(
        embeddings,
        index_type="hnsw",
        metric="cosine",
        params={"M": 16, "efSearch": 64}
    )
    
    assert index.ntotal == 100


def test_save_load_faiss_index(tmp_path):
    """Test saving and loading FAISS index."""
    embeddings = np.random.randn(50, 16).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = build_faiss_index(embeddings, index_type="flat", metric="cosine")
    
    file_path = tmp_path / "index.index"
    save_faiss_index(index, str(file_path))
    
    loaded_index = load_faiss_index(str(file_path))
    
    assert loaded_index.ntotal == index.ntotal

