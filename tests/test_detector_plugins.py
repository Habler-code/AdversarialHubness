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

"""Tests for detector plugin/registry system."""

import pytest
import numpy as np
from typing import Optional, Dict, Any

from hubscan.core.detectors import (
    Detector,
    DetectorResult,
    register_detector,
    get_detector_class,
    list_detectors,
)


class MockDetector(Detector):
    """Mock detector for testing."""
    
    def __init__(self, enabled: bool = True, test_param: float = 1.0):
        super().__init__(enabled)
        self.test_param = test_param
        self.call_count = 0
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional["Metadata"] = None,
        **kwargs,
    ) -> DetectorResult:
        """Mock detection implementation."""
        self.call_count += 1
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        # Return simple scores based on test_param
        scores = np.ones(len(doc_embeddings)) * self.test_param
        
        result_metadata = {
            "detector_type": "mock",
            "test_param": self.test_param,
            "call_count": self.call_count,
        }
        
        return DetectorResult(scores=scores, metadata=result_metadata)


class TestDetectorRegistry:
    """Test detector registry functionality."""
    
    def test_list_builtin_detectors(self):
        """Test that built-in detectors are registered."""
        detectors = list_detectors()
        assert "hubness" in detectors
        assert "cluster_spread" in detectors
        assert "stability" in detectors
        assert "dedup" in detectors
    
    def test_register_custom_detector(self):
        """Test registering a custom detector."""
        # Register
        register_detector("test_custom", MockDetector)
        
        # Verify it's registered
        assert "test_custom" in list_detectors()
        
        # Retrieve it
        retrieved_class = get_detector_class("test_custom")
        assert retrieved_class is not None
        assert issubclass(retrieved_class, Detector)
    
    def test_get_detector_class(self):
        """Test retrieving detector classes."""
        # Built-in detectors
        hubness_class = get_detector_class("hubness")
        assert hubness_class is not None
        assert issubclass(hubness_class, Detector)
        
        cluster_spread_class = get_detector_class("cluster_spread")
        assert cluster_spread_class is not None
        
        # Non-existent detector
        assert get_detector_class("nonexistent") is None
    
    def test_register_overwrite_warning(self):
        """Test that registering with existing name warns."""
        import warnings
        
        register_detector("overwrite_test", MockDetector)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_detector("overwrite_test", MockDetector)
            
            assert len(w) == 1
            assert "already registered" in str(w[0].message).lower()
    
    def test_register_invalid_detector(self):
        """Test that registering non-Detector class raises error."""
        class NotADetector:
            pass
        
        with pytest.raises(TypeError, match="must be a subclass of Detector"):
            register_detector("invalid", NotADetector)
    
    def test_custom_detector_instantiation(self):
        """Test that custom detector can be instantiated and used."""
        # Register custom detector
        register_detector("instantiation_test", MockDetector)
        
        # Get class and instantiate
        detector_class = get_detector_class("instantiation_test")
        detector = detector_class(enabled=True, test_param=2.5)
        
        assert detector.enabled is True
        assert detector.test_param == 2.5
        
        # Create mock data
        num_docs = 10
        dim = 32
        doc_embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        queries = np.random.randn(5, dim).astype(np.float32)
        
        # Run detection
        result = detector.detect(
            index=None,  # Mock detector doesn't use index
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=5
        )
        
        assert result is not None
        assert len(result.scores) == num_docs
        assert np.allclose(result.scores, 2.5)  # Should be test_param
        assert detector.call_count == 1


class TestDetectorIntegration:
    """Test detector integration with scanner."""
    
    def test_custom_detector_with_scanner(self):
        """Test that custom detector works with scanner."""
        from hubscan import Config, Scanner
        from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
        import faiss
        
        # Register custom detector
        register_detector("scanner_test", MockDetector)
        
        # Create test data
        dim = 32
        num_docs = 20
        embeddings = np.random.randn(num_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        index = FAISSIndex(faiss_index)
        
        # Create config with custom detector
        config = Config()
        config.input.mode = "embeddings_only"
        config.scan.k = 5
        config.scan.num_queries = 10
        config.scan.ranking.method = "vector"
        
        # Enable custom detector (we'll need to add it to config manually)
        # Since config doesn't have scanner_test, we'll test via direct instantiation
        scanner = Scanner(config)
        scanner.index = index
        scanner.doc_embeddings = embeddings
        
        # Manually add custom detector to scanner's detector dict
        # This simulates what would happen if config supported it
        detector_class = get_detector_class("scanner_test")
        custom_detector = detector_class(enabled=True, test_param=1.5)
        
        # Test that detector can be called with scanner's common kwargs
        queries = np.random.randn(10, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        result = custom_detector.detect(
            index=index,
            doc_embeddings=embeddings,
            queries=queries,
            k=5,
            ranking_method="vector",
            batch_size=256,
            seed=42
        )
        
        assert result is not None
        assert len(result.scores) == num_docs


class TestBuiltinDetectors:
    """Test that built-in detectors are properly registered."""
    
    def test_hubness_detector_registered(self):
        """Test hubness detector is registered."""
        detector_class = get_detector_class("hubness")
        assert detector_class is not None
        
        detector = detector_class(enabled=True)
        assert detector.enabled is True
    
    def test_cluster_spread_detector_registered(self):
        """Test cluster spread detector is registered."""
        detector_class = get_detector_class("cluster_spread")
        assert detector_class is not None
        
        detector = detector_class(enabled=True)
        assert detector.enabled is True
    
    def test_stability_detector_registered(self):
        """Test stability detector is registered."""
        detector_class = get_detector_class("stability")
        assert detector_class is not None
        
        detector = detector_class(enabled=True)
        assert detector.enabled is True
    
    def test_dedup_detector_registered(self):
        """Test dedup detector is registered."""
        detector_class = get_detector_class("dedup")
        assert detector_class is not None
        
        detector = detector_class(enabled=True)
        assert detector.enabled is True


class TestDetectorErrorHandling:
    """Test error handling in detector registry."""
    
    def test_get_nonexistent_detector(self):
        """Test getting non-existent detector returns None."""
        assert get_detector_class("definitely_does_not_exist") is None
    
    def test_detector_disabled(self):
        """Test that disabled detector returns zero scores."""
        detector = MockDetector(enabled=False)
        
        num_docs = 10
        doc_embeddings = np.random.randn(num_docs, 32).astype(np.float32)
        queries = np.random.randn(5, 32).astype(np.float32)
        
        result = detector.detect(
            index=None,
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=5
        )
        
        assert result is not None
        assert len(result.scores) == num_docs
        assert np.allclose(result.scores, 0.0)

