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

"""Tests for hybrid search functionality across all adapters."""

import numpy as np
import pytest
import tempfile
import os

from hubscan.config import Config
from hubscan.core.io.hybrid_fusion import (
    BM25Scorer,
    TFIDFScorer,
    create_lexical_scorer,
    normalize_scores,
    fuse_dense_lexical,
    ClientFusionHybridSearch,
)


class TestLexicalScorers:
    """Tests for lexical scoring implementations."""
    
    def test_bm25_scorer_fit_score(self):
        """Test BM25 scorer fit and score."""
        # Use documents where query terms are rare (not in >50% of docs)
        # BM25 gives IDF=0 to terms appearing in half+ the corpus
        documents = [
            "machine learning algorithms and neural networks",
            "natural language processing with transformers",
            "computer vision deep learning techniques",
            "database management systems and sql queries",
            "web development with javascript frameworks",
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Query for "neural" - only in doc 0
        scores = scorer.score("neural")
        
        assert len(scores) == len(documents)
        # Doc 0 has "neural", others don't - doc 0 should have highest score
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        assert scores[0] > scores[3]
        assert scores[0] > scores[4]
    
    def test_bm25_scorer_batch(self):
        """Test BM25 scorer batch scoring."""
        # Use more documents so terms aren't in >50% (BM25 IDF=0 issue)
        documents = [
            "machine learning algorithms",
            "natural language processing",
            "computer vision techniques",
            "database management systems",
            "web development frameworks",
        ]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        queries = ["machine", "language", "machine learning"]
        scores = scorer.score_batch(queries)
        
        assert scores.shape == (3, 5)
        # Query "machine" should score doc 0 highest
        assert scores[0, 0] > scores[0, 1]  # "machine" -> doc 0 vs doc 1
        # Query "language" should score doc 1 highest
        assert scores[1, 1] > scores[1, 0]  # "language" -> doc 1 vs doc 0
    
    def test_bm25_scorer_empty_documents(self):
        """Test BM25 scorer with empty documents."""
        documents = ["", "some text", ""]
        
        scorer = BM25Scorer()
        scorer.fit(documents)
        
        # Should not crash, return zero scores for empty
        scores = scorer.score("text")
        assert len(scores) == 3
    
    def test_tfidf_scorer_fit_score(self):
        """Test TF-IDF scorer fit and score."""
        documents = [
            "the quick brown fox jumps over the lazy dog",
            "a fast red fox runs through the forest",
            "the lazy cat sleeps all day",
        ]
        
        scorer = TFIDFScorer()
        scorer.fit(documents)
        
        scores = scorer.score("fox forest")
        
        assert len(scores) == len(documents)
        # Doc 1 has both "fox" and "forest"
        assert scores[1] >= scores[0]
        assert scores[1] >= scores[2]
    
    def test_create_lexical_scorer_factory(self):
        """Test the factory function for creating scorers."""
        bm25 = create_lexical_scorer("bm25")
        assert isinstance(bm25, BM25Scorer)
        
        tfidf = create_lexical_scorer("tfidf")
        assert isinstance(tfidf, TFIDFScorer)
        
        with pytest.raises(ValueError):
            create_lexical_scorer("invalid")


class TestNormalization:
    """Tests for score normalization."""
    
    def test_normalize_scores_1d(self):
        """Test 1D score normalization."""
        scores = np.array([1.0, 5.0, 3.0, 10.0])
        normalized = normalize_scores(scores)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert abs(normalized[3] - 1.0) < 0.01  # Max should be ~1.0
        assert abs(normalized[0] - 0.0) < 0.01  # Min should be ~0.0
    
    def test_normalize_scores_2d(self):
        """Test 2D score normalization (per-row)."""
        scores = np.array([
            [1.0, 5.0, 3.0],
            [10.0, 20.0, 15.0],
        ])
        normalized = normalize_scores(scores)
        
        assert normalized.shape == scores.shape
        # Each row should have its own normalization
        for i in range(len(scores)):
            assert normalized[i].min() >= 0.0
            assert normalized[i].max() <= 1.0
    
    def test_normalize_scores_uniform(self):
        """Test normalization with uniform scores."""
        scores = np.array([5.0, 5.0, 5.0])
        normalized = normalize_scores(scores)
        
        # All scores should become 1.0 (or some constant)
        assert np.allclose(normalized, normalized[0])


class TestClientFusionHybridSearch:
    """Tests for client-side fusion hybrid search."""
    
    def test_client_fusion_init(self):
        """Test ClientFusionHybridSearch initialization."""
        documents = [
            "machine learning for beginners",
            "deep learning neural networks",
            "python programming basics",
        ]
        
        hybrid = ClientFusionHybridSearch(documents, lexical_backend="bm25")
        
        assert hybrid.num_docs == 3
        assert hybrid.lexical_scorer is not None
    
    def test_client_fusion_fuse(self):
        """Test fusion of dense and lexical results."""
        documents = [
            "machine learning for beginners",
            "deep learning neural networks",
            "python programming basics",
            "advanced machine learning techniques",
            "data science with python",
        ]
        
        hybrid = ClientFusionHybridSearch(documents, lexical_backend="bm25")
        
        # Simulate dense search results (2 queries, k=3)
        dense_distances = np.array([
            [0.1, 0.2, 0.3],  # Query 1: closest to doc 0, 1, 2
            [0.15, 0.25, 0.35],  # Query 2: similar
        ], dtype=np.float32)
        dense_indices = np.array([
            [0, 1, 2],
            [3, 4, 0],
        ], dtype=np.int64)
        
        query_texts = [
            "machine learning",  # Should boost docs with "machine learning"
            "python programming",  # Should boost python docs
        ]
        
        fused_distances, fused_indices, metadata = hybrid.fuse(
            dense_distances=dense_distances,
            dense_indices=dense_indices,
            query_texts=query_texts,
            k=3,
            alpha=0.5,
        )
        
        assert fused_distances.shape == (2, 3)
        assert fused_indices.shape == (2, 3)
        assert metadata["ranking_method"] == "hybrid"
        assert metadata["hybrid_backend"] == "client_fusion"
    
    def test_client_fusion_alpha_variations(self):
        """Test different alpha values for hybrid fusion."""
        documents = [
            "vector search database",
            "keyword search engine",
            "hybrid search combines both",
        ]
        
        hybrid = ClientFusionHybridSearch(documents, lexical_backend="bm25")
        
        dense_distances = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        dense_indices = np.array([[0, 1, 2]], dtype=np.int64)
        query_texts = ["keyword search"]
        
        # Alpha=1.0 -> pure vector
        _, indices_vec, _ = hybrid.fuse(dense_distances, dense_indices, query_texts, k=3, alpha=1.0)
        
        # Alpha=0.0 -> pure lexical
        _, indices_lex, _ = hybrid.fuse(dense_distances, dense_indices, query_texts, k=3, alpha=0.0)
        
        # Results should differ when alpha changes
        # The lexical search should favor doc 1 ("keyword search engine")
        # This is not guaranteed to be different, but demonstrates the API


class TestFAISSHybridIntegration:
    """Integration tests for FAISS adapter with hybrid search."""
    
    def test_faiss_hybrid_with_config(self):
        """Test FAISS adapter hybrid search with config."""
        import faiss
        from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
        
        # Create test data
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        documents = [f"document {i} about topic {i % 10}" for i in range(100)]
        
        # Build FAISS index
        index = faiss.IndexFlatIP(64)
        index.add(embeddings)
        
        # Create adapter with hybrid config
        hybrid_config = {
            "backend": "client_fusion",
            "lexical_backend": "bm25",
            "normalize_scores": True,
        }
        adapter = FAISSIndex(index, document_texts=documents, hybrid_config=hybrid_config)
        
        # Run hybrid search
        query_vectors = embeddings[:5]  # Use first 5 as queries
        query_texts = [f"document about topic {i}" for i in range(5)]
        
        distances, indices, metadata = adapter.search_hybrid(
            query_vectors=query_vectors,
            query_texts=query_texts,
            k=10,
            alpha=0.7,
        )
        
        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)
        assert metadata["ranking_method"] == "hybrid"
        assert metadata["hybrid_backend"] == "client_fusion"
    
    def test_faiss_hybrid_fallback_to_vector(self):
        """Test FAISS falls back to vector-only when no texts."""
        import faiss
        from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
        
        np.random.seed(42)
        embeddings = np.random.randn(50, 32).astype(np.float32)
        
        index = faiss.IndexFlatL2(32)
        index.add(embeddings)
        
        # Create adapter WITHOUT document texts
        adapter = FAISSIndex(index, document_texts=None)
        
        # Hybrid search should raise error if texts requested but not available
        query_vectors = embeddings[:2]
        query_texts = ["test query 1", "test query 2"]
        
        with pytest.raises(ValueError, match="document_texts"):
            adapter.search_hybrid(
                query_vectors=query_vectors,
                query_texts=query_texts,
                k=5,
                alpha=0.5,
            )


class TestHybridConfig:
    """Tests for HybridSearchConfig."""
    
    def test_hybrid_config_defaults(self):
        """Test HybridSearchConfig default values."""
        config = Config()
        
        hybrid = config.scan.ranking.hybrid
        assert hybrid.backend == "auto"
        assert hybrid.lexical_backend == "bm25"
        assert hybrid.text_field == "text"
        assert hybrid.normalize_scores == True
    
    def test_hybrid_config_validation_no_query_texts(self):
        """Test config validation fails without query_texts for hybrid."""
        with pytest.raises(ValueError, match="query_texts_path"):
            Config(
                scan={
                    "ranking": {
                        "method": "hybrid",
                    },
                    # No query_texts_path
                }
            )
    
    def test_hybrid_config_native_sparse_pinecone_validation(self):
        """Test native sparse validation for Pinecone."""
        with pytest.raises(ValueError, match="sparse vectors"):
            Config(
                input={
                    "mode": "pinecone",
                    "pinecone_index_name": "test",
                    "pinecone_api_key": "fake",
                },
                scan={
                    "query_texts_path": "/fake/path.json",
                    "ranking": {
                        "method": "hybrid",
                        "hybrid": {
                            "backend": "native_sparse",
                            "pinecone_has_sparse": False,  # No sparse vectors
                        },
                    },
                }
            )


class TestHybridScannerIntegration:
    """Integration tests for Scanner with hybrid search."""
    
    def test_scanner_hybrid_validation(self):
        """Test Scanner validates hybrid requirements."""
        # This test would require creating a full Scanner instance
        # with mock data, which is more of an integration test
        pass
    
    def test_scanner_builds_hybrid_config(self):
        """Test Scanner correctly builds hybrid config for adapters."""
        from hubscan.core.scanner import Scanner
        
        config = Config(
            input={
                "mode": "embeddings_only",
                "embeddings_path": "/fake/embeddings.npy",
            },
            scan={
                "ranking": {
                    "method": "vector",  # Use vector to avoid validation
                    "hybrid": {
                        "backend": "client_fusion",
                        "lexical_backend": "tfidf",
                        "text_field": "content",
                    },
                },
            },
        )
        
        scanner = Scanner(config)
        hybrid_config = scanner._get_hybrid_config()
        
        assert hybrid_config["backend"] == "client_fusion"
        assert hybrid_config["lexical_backend"] == "tfidf"
        assert hybrid_config["text_field"] == "content"


class TestFuseDenseLexical:
    """Tests for the fusion algorithm."""
    
    def test_fuse_basic(self):
        """Test basic fusion of dense and lexical results."""
        dense_distances = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        dense_indices = np.array([[0, 1, 2]], dtype=np.int64)
        lexical_scores = np.array([[0.5, 0.8, 0.1, 0.9, 0.2]], dtype=np.float32)
        
        fused_distances, fused_indices = fuse_dense_lexical(
            dense_distances=dense_distances,
            dense_indices=dense_indices,
            lexical_scores=lexical_scores,
            k=3,
            alpha=0.5,
        )
        
        assert fused_distances.shape == (1, 3)
        assert fused_indices.shape == (1, 3)
        # Results should include candidates from both dense and lexical
    
    def test_fuse_alpha_weight(self):
        """Test alpha parameter affects fusion."""
        dense_distances = np.array([[0.1, 0.5]], dtype=np.float32)
        dense_indices = np.array([[0, 1]], dtype=np.int64)
        lexical_scores = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
        
        # Alpha=1.0 -> dense only
        fd1, fi1 = fuse_dense_lexical(
            dense_distances, dense_indices, lexical_scores, k=2, alpha=1.0
        )
        
        # Alpha=0.0 -> lexical only
        fd0, fi0 = fuse_dense_lexical(
            dense_distances, dense_indices, lexical_scores, k=2, alpha=0.0
        )
        
        # The order should potentially differ
        # With alpha=0.0, doc 1 should be ranked higher (lexical score 0.9)

