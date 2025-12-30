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

"""Integration tests for plugin system with Scanner."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from hubscan import Config, Scanner
from hubscan.core.ranking import register_ranking_method, get_ranking_method
from hubscan.core.detectors import register_detector, get_detector_class, Detector, DetectorResult
from hubscan.core.io.vector_index import VectorIndex
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
import faiss


class TestCustomRankingIntegration:
    """Test custom ranking methods with Scanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dim = 32
        self.num_docs = 30
        self.embeddings = np.random.randn(self.num_docs, self.dim).astype(np.float32)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(self.embeddings)
        self.index = FAISSIndex(self.faiss_index)
        
        # Create temporary embeddings file
        self.temp_dir = tempfile.mkdtemp()
        self.embeddings_path = str(Path(self.temp_dir) / "embeddings.npy")
        np.save(self.embeddings_path, self.embeddings)
    
    def test_custom_ranking_with_scanner(self):
        """Test that custom ranking method works with Scanner."""
        # Register custom ranking method
        class CustomRanking:
            def search(self, index, query_vectors, query_texts, k, multiplier=1.0, **kwargs):
                distances, indices = index.search(query_vectors, k)
                # Apply multiplier
                distances = distances * multiplier
                return distances, indices, {
                    "ranking_method": "custom_test",
                    "multiplier": multiplier
                }
        
        register_ranking_method("custom_test", CustomRanking())
        
        # Create config
        config = Config()
        config.input.mode = "embeddings_only"
        config.input.embeddings_path = self.embeddings_path
        config.input.metric = "cosine"
        config.scan.k = 5
        config.scan.num_queries = 10
        config.scan.ranking.method = "custom_test"
        config.scan.ranking.custom_params = {"multiplier": 1.5}
        
        # Create scanner and load data
        scanner = Scanner(config)
        scanner.load_data()
        
        # Verify ranking method is used
        ranking_method = get_ranking_method("custom_test")
        assert ranking_method is not None
        
        # Run scan (this will use custom ranking method)
        queries = np.random.randn(10, self.dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Test the ranking method directly
        distances, indices, metadata = ranking_method.search(
            index=scanner.index,
            query_vectors=queries,
            query_texts=None,
            k=5,
            multiplier=1.5
        )
        
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        assert metadata["ranking_method"] == "custom_test"
        assert metadata["multiplier"] == 1.5
    
    def test_custom_ranking_with_detector(self):
        """Test that custom ranking method works with detectors."""
        from hubscan.core.detectors.hubness import HubnessDetector
        
        # Register custom ranking
        class SimpleCustomRanking:
            def search(self, index, query_vectors, query_texts, k, **kwargs):
                distances, indices = index.search(query_vectors, k)
                return distances, indices, {"ranking_method": "simple_custom"}
        
        register_ranking_method("simple_custom", SimpleCustomRanking())
        
        # Create detector
        detector = HubnessDetector(enabled=True, metric="cosine")
        
        # Create queries
        queries = np.random.randn(10, self.dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Run detection with custom ranking
        result = detector.detect(
            index=self.index,
            doc_embeddings=self.embeddings,
            queries=queries,
            k=5,
            ranking_method="simple_custom",
            ranking_custom_params={}
        )
        
        assert result is not None
        assert len(result.scores) == self.num_docs


class TestCustomDetectorIntegration:
    """Test custom detectors with Scanner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dim = 32
        self.num_docs = 20
        self.embeddings = np.random.randn(self.num_docs, self.dim).astype(np.float32)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(self.embeddings)
        self.index = FAISSIndex(self.faiss_index)
    
    def test_custom_detector_instantiation(self):
        """Test that custom detector can be instantiated and used."""
        class SimpleDetector(Detector):
            def detect(self, index, doc_embeddings, queries, k, metadata=None, **kwargs):
                scores = np.ones(len(doc_embeddings)) * 0.5
                return DetectorResult(scores=scores, metadata={"type": "simple"})
        
        register_detector("simple_test", SimpleDetector)
        
        # Get and instantiate
        detector_class = get_detector_class("simple_test")
        detector = detector_class(enabled=True)
        
        queries = np.random.randn(5, self.dim).astype(np.float32)
        result = detector.detect(
            index=self.index,
            doc_embeddings=self.embeddings,
            queries=queries,
            k=5
        )
        
        assert result is not None
        assert len(result.scores) == self.num_docs
        assert np.allclose(result.scores, 0.5)


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_builtin_ranking_methods_still_work(self):
        """Test that built-in ranking methods work as before."""
        from hubscan.core.ranking import get_ranking_method
        
        # All built-in methods should be available
        assert get_ranking_method("vector") is not None
        assert get_ranking_method("hybrid") is not None
        assert get_ranking_method("lexical") is not None
        assert get_ranking_method("reranked") is not None
    
    def test_builtin_detectors_still_work(self):
        """Test that built-in detectors work as before."""
        from hubscan.core.detectors import get_detector_class
        
        # All built-in detectors should be available
        assert get_detector_class("hubness") is not None
        assert get_detector_class("cluster_spread") is not None
        assert get_detector_class("stability") is not None
        assert get_detector_class("dedup") is not None
    
    def test_config_with_builtin_methods(self):
        """Test that config with built-in methods works."""
        config = Config()
        config.scan.ranking.method = "vector"
        
        # Should not raise error
        assert config.scan.ranking.method == "vector"
        
        # Test other built-in methods
        config.scan.ranking.method = "hybrid"
        assert config.scan.ranking.method == "hybrid"
        
        config.scan.ranking.method = "lexical"
        assert config.scan.ranking.method == "lexical"
        
        config.scan.ranking.method = "reranked"
        assert config.scan.ranking.method == "reranked"
    
    def test_custom_params_in_config(self):
        """Test that custom_params field works in config."""
        config = Config()
        config.scan.ranking.method = "vector"
        config.scan.ranking.custom_params = {"test_param": 42}
        
        assert config.scan.ranking.custom_params["test_param"] == 42


class TestPluginErrorHandling:
    """Test error handling in plugin system."""
    
    def test_unknown_ranking_method_error(self):
        """Test that unknown ranking method raises appropriate error."""
        from hubscan.core.scanner import Scanner
        from hubscan import Config
        import numpy as np
        import faiss
        from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
        
        config = Config()
        config.input.mode = "embeddings_only"
        config.scan.ranking.method = "definitely_does_not_exist"
        
        scanner = Scanner(config)
        
        # Create minimal data
        dim = 32
        num_docs = 10
        embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        scanner.index = FAISSIndex(faiss_index)
        scanner.doc_embeddings = embeddings
        
        # Should raise ValueError when trying to scan
        with pytest.raises(ValueError, match="Unknown ranking method"):
            scanner.scan()

