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

"""Tests for VectorIndex interface."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from hubscan.core.io.vector_index import VectorIndex
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex


class MockVectorIndex(VectorIndex):
    """Mock implementation of VectorIndex for testing."""
    
    def __init__(self, ntotal: int = 100, dimension: int = 128):
        self._ntotal = ntotal
        self._dimension = dimension
    
    def search(self, query_vectors: np.ndarray, k: int):
        M = len(query_vectors)
        distances = np.random.rand(M, k).astype(np.float32)
        indices = np.random.randint(0, self._ntotal, size=(M, k), dtype=np.int64)
        return distances, indices
    
    @property
    def ntotal(self) -> int:
        return self._ntotal
    
    @property
    def dimension(self) -> int:
        return self._dimension


def test_vector_index_interface():
    """Test that VectorIndex interface can be implemented."""
    index = MockVectorIndex(ntotal=100, dimension=128)
    
    assert index.ntotal == 100
    assert index.dimension == 128
    
    queries = np.random.randn(10, 128).astype(np.float32)
    distances, indices = index.search(queries, k=5)
    
    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)
    assert distances.dtype == np.float32
    assert indices.dtype == np.int64


def test_vector_index_abstract():
    """Test that VectorIndex cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VectorIndex()


def test_faiss_adapter():
    """Test FAISS adapter implementation."""
    import faiss
    
    # Create a simple FAISS index
    dimension = 128
    num_vectors = 100
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    
    # Wrap in adapter
    adapter = FAISSIndex(faiss_index)
    
    assert adapter.ntotal == num_vectors
    assert adapter.dimension == dimension
    
    # Test search
    queries = np.random.randn(5, dimension).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    distances, indices = adapter.search(queries, k=10)
    
    assert distances.shape == (5, 10)
    assert indices.shape == (5, 10)
    assert np.all(indices >= 0)
    assert np.all(indices < num_vectors)


def test_faiss_adapter_type_check():
    """Test FAISS adapter type checking."""
    with pytest.raises(TypeError):
        FAISSIndex("not a faiss index")


def test_faiss_adapter_access_underlying():
    """Test accessing underlying FAISS index."""
    import faiss
    
    faiss_index = faiss.IndexFlatL2(128)
    adapter = FAISSIndex(faiss_index)
    
    assert adapter.faiss_index is faiss_index
    assert isinstance(adapter.faiss_index, faiss.Index)


def test_vector_index_search_hybrid_fallback():
    """Test VectorIndex search_hybrid fallback to vector search."""
    index = MockVectorIndex(ntotal=100, dimension=128)
    
    queries = np.random.randn(5, 128).astype(np.float32)
    distances, indices, metadata = index.search_hybrid(
        query_vectors=queries,
        query_texts=None,
        k=10,
        alpha=0.5
    )
    
    assert distances.shape == (5, 10)
    assert indices.shape == (5, 10)
    assert metadata["ranking_method"] == "vector"
    assert metadata["fallback"] is True


def test_vector_index_search_lexical_not_implemented():
    """Test VectorIndex search_lexical raises NotImplementedError."""
    index = MockVectorIndex(ntotal=100, dimension=128)
    
    query_texts = ["test query"] * 5
    with pytest.raises(NotImplementedError):
        index.search_lexical(query_texts, k=10)


def test_vector_index_search_reranked():
    """Test VectorIndex search_reranked default implementation."""
    index = MockVectorIndex(ntotal=100, dimension=128)
    
    queries = np.random.randn(5, 128).astype(np.float32)
    distances, indices, metadata = index.search_reranked(
        query_vectors=queries,
        k=5,
        rerank_top_n=20
    )
    
    assert distances.shape == (5, 5)
    assert indices.shape == (5, 5)
    assert metadata["ranking_method"] == "reranked"
    assert metadata["rerank_top_n"] == 20

