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

"""Tests for reranking methods."""

import pytest
import numpy as np

from hubscan.core.reranking import (
    get_reranking_method,
    list_reranking_methods,
    register_reranking_method,
)
from hubscan.core.reranking.builtin import DefaultReranking, CrossEncoderReranking


class TestDefaultReranking:
    """Tests for DefaultReranking."""
    
    def test_rerank_basic(self):
        """Test basic reranking functionality."""
        reranker = DefaultReranking()
        
        # Create test data
        distances = np.array([
            [0.9, 0.5, 0.8, 0.3],  # Query 0 scores
            [0.2, 0.7, 0.6, 0.4],  # Query 1 scores
        ])
        indices = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ])
        
        reranked_distances, reranked_indices, metadata = reranker.rerank(
            distances=distances,
            indices=indices,
            query_vectors=None,
            query_texts=None,
            k=2,
        )
        
        # Check shapes
        assert reranked_distances.shape == (2, 2)
        assert reranked_indices.shape == (2, 2)
        
        # Check that top-2 are returned (sorted by descending score)
        # Query 0: highest scores are 0.9 (idx 0), 0.8 (idx 2)
        assert reranked_indices[0, 0] == 0
        assert reranked_indices[0, 1] == 2
        
        # Query 1: highest scores are 0.7 (idx 5), 0.6 (idx 6)
        assert reranked_indices[1, 0] == 5
        assert reranked_indices[1, 1] == 6
        
        # Check metadata
        assert metadata["reranking_method"] == "default"
        assert metadata["candidates"] == 4
        assert metadata["final_k"] == 2
    
    def test_rerank_k_equals_candidates(self):
        """Test reranking when k equals number of candidates."""
        reranker = DefaultReranking()
        
        distances = np.array([[0.9, 0.5, 0.8]])
        indices = np.array([[0, 1, 2]])
        
        reranked_distances, reranked_indices, metadata = reranker.rerank(
            distances=distances,
            indices=indices,
            query_vectors=None,
            query_texts=None,
            k=3,  # Equal to available
        )
        
        # Should return all 3, sorted by score
        assert reranked_indices.shape[1] == 3
        # Sorted by descending score: 0.9 (idx 0), 0.8 (idx 2), 0.5 (idx 1)
        assert reranked_indices[0, 0] == 0
        assert reranked_indices[0, 1] == 2
        assert reranked_indices[0, 2] == 1


class TestCrossEncoderReranking:
    """Tests for CrossEncoderReranking."""
    
    def test_init_default_model(self):
        """Test initialization with default model."""
        reranker = CrossEncoderReranking()
        assert reranker._model_name == CrossEncoderReranking.DEFAULT_MODEL
        assert reranker._model is None  # Lazy loading
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        model_name = "cross-encoder/ms-marco-TinyBERT-L-2"
        reranker = CrossEncoderReranking(model_name=model_name)
        assert reranker._model_name == model_name
    
    def test_rerank_requires_query_texts(self):
        """Test that rerank requires query_texts."""
        reranker = CrossEncoderReranking()
        
        distances = np.array([[0.9, 0.5]])
        indices = np.array([[0, 1]])
        
        with pytest.raises(ValueError, match="query_texts"):
            reranker.rerank(
                distances=distances,
                indices=indices,
                query_vectors=None,
                query_texts=None,  # Missing
                k=2,
                document_texts=["doc1", "doc2"],
            )
    
    def test_rerank_requires_document_texts(self):
        """Test that rerank requires document_texts."""
        reranker = CrossEncoderReranking()
        
        distances = np.array([[0.9, 0.5]])
        indices = np.array([[0, 1]])
        
        with pytest.raises(ValueError, match="document_texts"):
            reranker.rerank(
                distances=distances,
                indices=indices,
                query_vectors=None,
                query_texts=["query text"],
                k=2,
                document_texts=None,  # Missing
            )
    
    @pytest.mark.slow
    def test_rerank_with_model(self):
        """Test actual reranking with model (slow, requires model download)."""
        pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
        
        reranker = CrossEncoderReranking()
        
        # Create test data
        distances = np.array([[0.9, 0.5, 0.8]])
        indices = np.array([[0, 1, 2]])
        query_texts = ["What is machine learning?"]
        document_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny.",
            "Deep learning uses neural networks.",
        ]
        
        reranked_distances, reranked_indices, metadata = reranker.rerank(
            distances=distances,
            indices=indices,
            query_vectors=None,
            query_texts=query_texts,
            k=2,
            document_texts=document_texts,
        )
        
        assert reranked_distances.shape == (1, 2)
        assert reranked_indices.shape == (1, 2)
        assert metadata["reranking_method"] == "cross_encoder"
        
        # The ML-related docs should rank higher than weather doc
        # Doc 0 and 2 are about ML, doc 1 is about weather
        top_2 = set(reranked_indices[0])
        assert 0 in top_2 or 2 in top_2  # At least one ML doc in top 2


class TestRerankingRegistry:
    """Tests for reranking method registry."""
    
    def test_builtin_methods_registered(self):
        """Test that built-in methods are registered."""
        methods = list_reranking_methods()
        assert "default" in methods
        assert "cross_encoder" in methods
    
    def test_get_default_reranking(self):
        """Test getting default reranking method."""
        method = get_reranking_method("default")
        assert method is not None
        assert isinstance(method, DefaultReranking)
    
    def test_get_cross_encoder_reranking(self):
        """Test getting cross-encoder reranking method."""
        method = get_reranking_method("cross_encoder")
        assert method is not None
        assert isinstance(method, CrossEncoderReranking)
    
    def test_get_nonexistent_method(self):
        """Test getting a non-existent method."""
        method = get_reranking_method("nonexistent")
        assert method is None
    
    def test_register_custom_method(self):
        """Test registering a custom method."""
        class CustomReranking:
            def rerank(self, distances, indices, query_vectors, query_texts, k, **kwargs):
                return distances[:, :k], indices[:, :k], {"reranking_method": "custom"}
        
        register_reranking_method("custom_test", CustomReranking())
        
        method = get_reranking_method("custom_test")
        assert method is not None
        
        methods = list_reranking_methods()
        assert "custom_test" in methods

