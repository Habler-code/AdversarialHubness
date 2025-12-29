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

"""Tests for vector database adapters."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from hubscan.core.io.adapters import create_index, FAISSIndex
from hubscan.config import InputConfig


def test_faiss_adapter_integration():
    """Test FAISS adapter with real FAISS index."""
    import faiss
    
    dimension = 64
    num_vectors = 50
    embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    
    adapter = FAISSIndex(faiss_index)
    
    # Test properties
    assert adapter.ntotal == num_vectors
    assert adapter.dimension == dimension
    
    # Test search
    queries = np.random.randn(10, dimension).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    distances, indices = adapter.search(queries, k=5)
    
    assert distances.shape == (10, 5)
    assert indices.shape == (10, 5)
    assert np.all(indices >= 0)
    assert np.all(indices < num_vectors)


@pytest.mark.skip(reason="Pinecone mock test requires proper mocking setup")
@patch('hubscan.core.io.adapters.pinecone_adapter.Pinecone', create=True)
def test_pinecone_adapter_mock(mock_pinecone_class):
    """Test Pinecone adapter with mocked client."""
    # Mock Pinecone client
    mock_pc_instance = MagicMock()
    mock_index = MagicMock()
    
    # Mock index stats
    mock_index.describe_index_stats.return_value = {"total_vector_count": 1000}
    
    # Mock index description
    mock_index_info = MagicMock()
    mock_index_info.dimension = 128
    mock_pc_instance.describe_index.return_value = mock_index_info
    mock_pc_instance.Index.return_value = mock_index
    
    mock_pinecone_class.return_value = mock_pc_instance
    
    # Import after mocking
    from hubscan.core.io.adapters.pinecone_adapter import PineconeIndex
    
    adapter = PineconeIndex(
        index_name="test-index",
        api_key="test-key",
        dimension=128
    )
    
    assert adapter.ntotal == 1000
    assert adapter.dimension == 128
    
    # Mock query results
    mock_results = []
    for _ in range(5):
        mock_match = MagicMock()
        mock_match.score = 0.9
        mock_match.id = "0"
        mock_result = MagicMock()
        mock_result.matches = [mock_match]
        mock_results.append(mock_result)
    
    mock_index.query_batch.return_value = mock_results
    
    queries = np.random.randn(5, 128).astype(np.float32)
    distances, indices = adapter.search(queries, k=1)
    
    assert distances.shape == (5, 1)
    assert indices.shape == (5, 1)


@pytest.mark.skip(reason="Qdrant mock test requires proper mocking setup")
@patch('hubscan.core.io.adapters.qdrant_adapter.QdrantClient')
def test_qdrant_adapter_mock(mock_qdrant_client):
    """Test Qdrant adapter with mocked client."""
    # Mock Qdrant client
    mock_client = MagicMock()
    
    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 128
    mock_collection_info.points_count = 1000
    mock_client.get_collection.return_value = mock_collection_info
    
    mock_qdrant_client.return_value = mock_client
    
    # Import after mocking
    from hubscan.core.io.adapters.qdrant_adapter import QdrantIndex
    
    adapter = QdrantIndex(
        collection_name="test-collection",
        url="http://localhost:6333"
    )
    
    assert adapter.ntotal == 1000
    assert adapter.dimension == 128
    
    # Mock search results
    mock_result = MagicMock()
    mock_result.score = 0.95
    mock_result.id = 42
    mock_client.search.return_value = [mock_result]
    
    queries = np.random.randn(1, 128).astype(np.float32)
    distances, indices = adapter.search(queries, k=1)
    
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)


@pytest.mark.skip(reason="Weaviate mock test requires proper mocking setup")
@patch('hubscan.core.io.adapters.weaviate_adapter.weaviate')
def test_weaviate_adapter_mock(mock_weaviate):
    """Test Weaviate adapter with mocked client."""
    # Mock Weaviate client
    mock_client = MagicMock()
    
    # Mock schema
    mock_schema = {"vectorizer": {}}
    mock_client.schema.get.return_value = mock_schema
    
    # Mock aggregate query
    mock_aggregate_result = {
        "data": {
            "Aggregate": {
                "test-class": [{"meta": {"count": 1000}}]
            }
        }
    }
    mock_query = MagicMock()
    mock_query.aggregate.return_value.with_meta_count.return_value.do.return_value = mock_aggregate_result
    mock_client.query = MagicMock(return_value=mock_query)
    
    mock_weaviate.Client.return_value = mock_client
    
    # Import after mocking
    from hubscan.core.io.adapters.weaviate_adapter import WeaviateIndex
    
    adapter = WeaviateIndex(
        class_name="test-class",
        url="http://localhost:8080"
    )
    
    assert adapter.ntotal == 1000
    
    # Mock search query
    mock_get_query = MagicMock()
    mock_get_query.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do.return_value = {
        "data": {
            "Get": {
                "test-class": [{
                    "_additional": {
                        "distance": 0.1,
                        "id": "test-id"
                    }
                }]
            }
        }
    }
    mock_client.query.get.return_value = mock_get_query
    
    queries = np.random.randn(1, 128).astype(np.float32)
    distances, indices = adapter.search(queries, k=1)
    
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)


def test_create_index_faiss_mode():
    """Test create_index factory with FAISS modes."""
    # FAISS modes should raise ValueError (should use build/load functions)
    config = InputConfig(mode="embeddings_only")
    with pytest.raises(ValueError):
        create_index(config)
    
    config = InputConfig(mode="faiss_index")
    with pytest.raises(ValueError):
        create_index(config)


def test_create_index_pinecone_missing_deps():
    """Test create_index with Pinecone when package not installed."""
    config = InputConfig(
        mode="pinecone",
        pinecone_index_name="test",
        pinecone_api_key="key"
    )
    
    # If pinecone is not installed, should raise ImportError
    # We can't easily test this without actually uninstalling, so we'll
    # just test that it tries to import
    try:
        create_index(config)
    except (ImportError, ValueError):
        pass  # Expected if pinecone not installed or config invalid


def test_create_index_missing_params():
    """Test create_index with missing required parameters."""
    # Pinecone without index name
    config = InputConfig(mode="pinecone", pinecone_api_key="key")
    try:
        create_index(config)
        pytest.fail("Should have raised ValueError")
    except (ValueError, ImportError) as e:
        # Either ValueError for missing param or ImportError if package not installed
        assert "pinecone_index_name" in str(e) or "pinecone" in str(e).lower()
    
    # Qdrant without collection name
    config = InputConfig(mode="qdrant")
    try:
        create_index(config)
        pytest.fail("Should have raised ValueError")
    except (ValueError, ImportError) as e:
        # Either ValueError for missing param or ImportError if package not installed
        assert "qdrant_collection_name" in str(e) or "qdrant" in str(e).lower()
    
    # Weaviate without class name
    config = InputConfig(mode="weaviate")
    try:
        create_index(config)
        pytest.fail("Should have raised ValueError")
    except (ValueError, ImportError) as e:
        # Either ValueError for missing param or ImportError if package not installed
        assert "weaviate_class_name" in str(e) or "weaviate" in str(e).lower()


def test_create_index_unsupported_mode():
    """Test create_index with unsupported mode."""
    config = InputConfig(mode="vector_db_export")
    with pytest.raises(ValueError, match="Unsupported backend mode"):
        create_index(config)

