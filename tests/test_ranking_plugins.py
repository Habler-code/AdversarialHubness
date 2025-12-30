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

"""Tests for ranking method plugin system."""

import pytest
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from hubscan.core.ranking import (
    RankingMethod,
    register_ranking_method,
    get_ranking_method,
    list_ranking_methods,
)
from hubscan.core.io.vector_index import VectorIndex
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
import faiss


class MockRankingMethod:
    """Mock ranking method for testing."""
    
    def __init__(self, name: str = "mock"):
        self.name = name
        self.call_count = 0
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Mock search implementation."""
        self.call_count += 1
        if query_vectors is None:
            raise ValueError("query_vectors required")
        
        distances, indices = index.search(query_vectors, k)
        metadata = {
            "ranking_method": self.name,
            "call_count": self.call_count,
            **kwargs
        }
        return distances, indices, metadata


class TestRankingMethodRegistry:
    """Test ranking method registry functionality."""
    
    def test_list_builtin_methods(self):
        """Test that built-in methods are registered."""
        methods = list_ranking_methods()
        assert "vector" in methods
        assert "hybrid" in methods
        assert "lexical" in methods
        assert "reranked" in methods
    
    def test_register_custom_method(self):
        """Test registering a custom ranking method."""
        method = MockRankingMethod("test_custom")
        
        # Register
        register_ranking_method("test_custom", method)
        
        # Verify it's registered
        assert "test_custom" in list_ranking_methods()
        
        # Retrieve it
        retrieved = get_ranking_method("test_custom")
        assert retrieved is not None
        assert retrieved.name == "test_custom"
    
    def test_get_ranking_method(self):
        """Test retrieving ranking methods."""
        # Built-in methods
        vector_method = get_ranking_method("vector")
        assert vector_method is not None
        
        hybrid_method = get_ranking_method("hybrid")
        assert hybrid_method is not None
        
        # Non-existent method
        assert get_ranking_method("nonexistent") is None
    
    def test_register_overwrite_warning(self):
        """Test that registering with existing name warns."""
        import warnings
        
        method1 = MockRankingMethod("overwrite_test")
        method2 = MockRankingMethod("overwrite_test")
        
        register_ranking_method("overwrite_test", method1)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_ranking_method("overwrite_test", method2)
            
            assert len(w) == 1
            assert "already registered" in str(w[0].message).lower()
    
    def test_custom_method_execution(self):
        """Test that custom ranking method can be executed."""
        # Create test index
        dim = 32
        num_docs = 10
        embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        index = FAISSIndex(faiss_index)
        
        # Register custom method
        method = MockRankingMethod("exec_test")
        register_ranking_method("exec_test", method)
        
        # Execute it
        queries = np.random.randn(3, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        retrieved_method = get_ranking_method("exec_test")
        distances, indices, metadata = retrieved_method.search(
            index=index,
            query_vectors=queries,
            query_texts=None,
            k=5,
            custom_param="test_value"
        )
        
        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)
        assert metadata["ranking_method"] == "exec_test"
        assert metadata["custom_param"] == "test_value"
        assert method.call_count == 1


class TestBuiltinRankingMethods:
    """Test built-in ranking method implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dim = 32
        self.num_docs = 20
        self.embeddings = np.random.randn(self.num_docs, self.dim).astype(np.float32)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(self.embeddings)
        
        self.doc_texts = [f"Document {i} with text content" for i in range(self.num_docs)]
        self.index = FAISSIndex(self.faiss_index, document_texts=self.doc_texts)
        
        self.queries = np.random.randn(5, self.dim).astype(np.float32)
        self.queries = self.queries / np.linalg.norm(self.queries, axis=1, keepdims=True)
        self.query_texts = ["query text 1", "query text 2", "query text 3", "query text 4", "query text 5"]
    
    def test_vector_ranking(self):
        """Test vector ranking method."""
        method = get_ranking_method("vector")
        assert method is not None
        
        distances, indices, metadata = method.search(
            index=self.index,
            query_vectors=self.queries,
            query_texts=None,
            k=5
        )
        
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)
        assert metadata["ranking_method"] == "vector"
    
    def test_hybrid_ranking(self):
        """Test hybrid ranking method."""
        method = get_ranking_method("hybrid")
        assert method is not None
        
        distances, indices, metadata = method.search(
            index=self.index,
            query_vectors=self.queries,
            query_texts=self.query_texts,
            k=5,
            alpha=0.5
        )
        
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)
        assert "ranking_method" in metadata
    
    def test_lexical_ranking(self):
        """Test lexical ranking method."""
        method = get_ranking_method("lexical")
        assert method is not None
        
        distances, indices, metadata = method.search(
            index=self.index,
            query_vectors=None,
            query_texts=self.query_texts,
            k=5
        )
        
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)
        assert "ranking_method" in metadata
    
    def test_lexical_ranking_requires_texts(self):
        """Test that lexical ranking requires query texts."""
        method = get_ranking_method("lexical")
        
        with pytest.raises(ValueError, match="query_texts required"):
            method.search(
                index=self.index,
                query_vectors=None,
                query_texts=None,
                k=5
            )
    
    def test_reranked_ranking(self):
        """Test reranked ranking method."""
        method = get_ranking_method("reranked")
        assert method is not None
        
        distances, indices, metadata = method.search(
            index=self.index,
            query_vectors=self.queries,
            query_texts=None,
            k=5,
            rerank_top_n=10
        )
        
        assert distances.shape == (5, 5)
        assert indices.shape == (5, 5)
        assert metadata["ranking_method"] == "reranked"
        assert metadata.get("rerank_top_n") == 10


class TestRankingMethodIntegration:
    """Test ranking methods integration with detectors."""
    
    def test_ranking_method_with_detector(self):
        """Test that custom ranking method works with detectors."""
        from hubscan.core.detectors.hubness import HubnessDetector
        
        # Create test data
        dim = 32
        num_docs = 20
        embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        index = FAISSIndex(faiss_index)
        
        queries = np.random.randn(10, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Register custom method
        method = MockRankingMethod("detector_test")
        register_ranking_method("detector_test", method)
        
        # Create detector
        detector = HubnessDetector(enabled=True, metric="cosine")
        
        # Run detection with custom ranking method
        result = detector.detect(
            index=index,
            doc_embeddings=embeddings,
            queries=queries,
            k=5,
            ranking_method="detector_test",
            ranking_custom_params={"test_param": "value"}
        )
        
        assert result is not None
        assert len(result.scores) == num_docs
        assert method.call_count > 0  # Method was called


class TestRankingMethodErrorHandling:
    """Test error handling in ranking method system."""
    
    def test_get_nonexistent_method(self):
        """Test getting non-existent method returns None."""
        assert get_ranking_method("definitely_does_not_exist") is None
    
    def test_method_missing_query_vectors(self):
        """Test that methods handle missing query vectors appropriately."""
        method = get_ranking_method("vector")
        
        dim = 32
        num_docs = 10
        embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        index = FAISSIndex(faiss_index)
        
        with pytest.raises(ValueError, match="query_vectors required"):
            method.search(
                index=index,
                query_vectors=None,
                query_texts=None,
                k=5
            )

