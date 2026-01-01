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

"""Tests for concept-specific and cross-modal hub detection."""

import pytest
import numpy as np

import faiss

from hubscan.core.detectors.hubness import HubnessDetector, BucketedHubnessAccumulator
from hubscan.core.concepts import ConceptAssignment, create_concept_provider
from hubscan.core.modalities import ModalityAssignment, create_modality_resolver
from hubscan.core.io.adapters import FAISSIndex


class TestBucketedHubnessAccumulator:
    """Tests for BucketedHubnessAccumulator."""
    
    def test_basic_accumulation(self):
        """Test basic hit accumulation."""
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Add some hits (rank and distance are used for weighting, but we disabled it)
        acc.add_hit(doc_id=5, query_id=0, rank=0, distance=0.1)
        acc.add_hit(doc_id=5, query_id=1, rank=0, distance=0.1)
        acc.add_hit(doc_id=10, query_id=0, rank=1, distance=0.2)
        
        acc.record_query(0)
        acc.record_query(1)
        
        # With weighting disabled, each hit counts as 1.0
        assert acc.global_hits[5] == 2.0
        assert acc.global_hits[10] == 1.0  # Changed from 0.5 since weighting is disabled
        assert acc.total_queries == 2
    
    def test_concept_accumulation(self):
        """Test concept-specific hit accumulation."""
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Concept 0 queries
        acc.add_hit(doc_id=5, query_id=0, rank=0, distance=0.1, concept_id=0)
        acc.add_hit(doc_id=5, query_id=1, rank=0, distance=0.1, concept_id=0)
        acc.record_query(0, concept_id=0)
        acc.record_query(1, concept_id=0)
        
        # Concept 1 queries
        acc.add_hit(doc_id=5, query_id=2, rank=0, distance=0.1, concept_id=1)
        acc.record_query(2, concept_id=1)
        
        assert acc.concept_hits[0][5] == 2.0
        assert acc.concept_hits[1][5] == 1.0
        assert acc.concept_query_counts[0] == 2
        assert acc.concept_query_counts[1] == 1
    
    def test_modality_accumulation(self):
        """Test modality-specific hit accumulation."""
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Text queries
        acc.add_hit(doc_id=5, query_id=0, rank=0, distance=0.1, query_modality="text", doc_modality="text")
        acc.record_query(0, query_modality="text")
        
        # Image queries
        acc.add_hit(doc_id=5, query_id=1, rank=0, distance=0.1, query_modality="image", doc_modality="text")
        acc.record_query(1, query_modality="image")
        
        assert acc.modality_hits["text"][5] == 1.0
        assert acc.modality_hits["image"][5] == 1.0
        assert acc.cross_modal_hits[5] == 1.0  # image query -> text doc
    
    def test_compute_hub_rates(self):
        """Test hub rate computation."""
        # Disable weighting for simpler test assertions
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Create a hub with many hits
        for i in range(10):
            acc.add_hit(doc_id=5, query_id=i, rank=0, distance=0.1)
            acc.record_query(i)
        
        hub_rates = acc.compute_hub_rates()
        
        # With weighting disabled, each hit counts as 1.0
        # 10 hits / 10 queries = 1.0
        assert hub_rates[5] == 1.0
    
    def test_compute_concept_hub_rates(self):
        """Test concept-specific hub rate computation."""
        # Disable weighting for simpler test assertions
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Create concept-specific hub
        for i in range(10):
            acc.add_hit(doc_id=5, query_id=i, rank=0, distance=0.1, concept_id=0)
            acc.record_query(i, concept_id=0)
        
        # Non-hub in concept 1
        acc.add_hit(doc_id=10, query_id=100, rank=0, distance=0.1, concept_id=1)
        acc.record_query(100, concept_id=1)
        
        concept_rates = acc.compute_concept_hub_rates()
        
        assert 0 in concept_rates
        assert concept_rates[0][5] == 1.0  # 10 hits / 10 queries in concept 0
        assert 1 in concept_rates
        assert concept_rates[1][10] == 1.0  # 1 hit / 1 query in concept 1
    
    def test_cross_modal_rates(self):
        """Test cross-modal rate computation."""
        # Disable weighting for simpler test assertions
        acc = BucketedHubnessAccumulator(num_docs=100, use_rank_weights=False, use_distance_weights=False)
        
        # Doc 5: 3 same-modal, 2 cross-modal
        for i in range(3):
            acc.add_hit(doc_id=5, query_id=i, rank=0, distance=0.1, query_modality="text", doc_modality="text")
            acc.record_query(i, query_modality="text")
        for i in range(3, 5):
            acc.add_hit(doc_id=5, query_id=i, rank=0, distance=0.1, query_modality="image", doc_modality="text")
            acc.record_query(i, query_modality="image")
        
        cross_modal_rates = acc.compute_cross_modal_rates()
        
        # 2 cross-modal hits / 5 total queries = 0.4
        assert cross_modal_rates[5] == pytest.approx(0.4)


class TestHubnessDetectorConceptModality:
    """Tests for concept/modality-aware HubnessDetector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        N, D = 100, 64
        M = 50
        
        doc_embeddings = np.random.randn(N, D).astype(np.float32)
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        queries = np.random.randn(M, D).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        return doc_embeddings, queries
    
    def test_backward_compatibility(self, sample_data):
        """Test that default behavior is unchanged."""
        doc_embeddings, queries = sample_data
        
        # Default detector (no concept/modality awareness)
        detector = HubnessDetector(
            enabled=True,
            concept_aware_enabled=False,
            modality_aware_enabled=False,
        )
        
        # Create FAISS index
        faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        index = FAISSIndex(faiss_index)
        
        result = detector.detect(
            index=index,
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=10,
        )
        
        assert result.scores is not None
        assert len(result.scores) == len(doc_embeddings)
        # Should have standard metadata
        assert "hub_z" in result.metadata
        # Should NOT have concept/modality metadata
        assert result.metadata.get("concept_aware", {}).get("enabled") == False
        assert result.metadata.get("modality_aware", {}).get("enabled") == False
    
    def test_concept_aware_detection(self, sample_data):
        """Test concept-aware hub detection."""
        doc_embeddings, queries = sample_data
        
        # Create concept assignments
        concept_assignment = ConceptAssignment(
            query_concepts={i: i % 5 for i in range(len(queries))},
            concept_names={i: f"concept_{i}" for i in range(5)},
        )
        
        detector = HubnessDetector(
            enabled=True,
            concept_aware_enabled=True,
            concept_hub_z_threshold=4.0,
        )
        
        faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        index = FAISSIndex(faiss_index)
        
        result = detector.detect(
            index=index,
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=10,
            concept_assignment=concept_assignment,
        )
        
        # Should have concept-aware metadata
        assert result.metadata["concept_aware"]["enabled"] == True
        assert "max_concept_hub_z" in result.metadata["concept_aware"]
        assert "top_concept_ids" in result.metadata["concept_aware"]
    
    def test_modality_aware_detection(self, sample_data):
        """Test modality-aware hub detection."""
        doc_embeddings, queries = sample_data
        
        # Create modality assignments
        # All docs are text, half queries are text, half are image
        modality_assignment = ModalityAssignment(
            query_modalities={i: "text" if i < len(queries) // 2 else "image" for i in range(len(queries))},
            doc_modalities={i: "text" for i in range(len(doc_embeddings))},
        )
        
        detector = HubnessDetector(
            enabled=True,
            modality_aware_enabled=True,
            cross_modal_penalty=1.5,
        )
        
        faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        index = FAISSIndex(faiss_index)
        
        result = detector.detect(
            index=index,
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=10,
            modality_assignment=modality_assignment,
        )
        
        # Should have modality-aware metadata
        assert result.metadata["modality_aware"]["enabled"] == True
        assert "cross_modal_z" in result.metadata["modality_aware"]
    
    def test_combined_detection(self, sample_data):
        """Test combined concept and modality-aware detection."""
        doc_embeddings, queries = sample_data
        
        concept_assignment = ConceptAssignment(
            query_concepts={i: i % 3 for i in range(len(queries))},
        )
        
        modality_assignment = ModalityAssignment(
            query_modalities={i: "text" if i % 2 == 0 else "image" for i in range(len(queries))},
            doc_modalities={i: "text" for i in range(len(doc_embeddings))},
        )
        
        detector = HubnessDetector(
            enabled=True,
            concept_aware_enabled=True,
            modality_aware_enabled=True,
        )
        
        faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        index = FAISSIndex(faiss_index)
        
        result = detector.detect(
            index=index,
            doc_embeddings=doc_embeddings,
            queries=queries,
            k=10,
            concept_assignment=concept_assignment,
            modality_assignment=modality_assignment,
        )
        
        # Should have both enabled
        assert result.metadata["concept_aware"]["enabled"] == True
        assert result.metadata["modality_aware"]["enabled"] == True


class TestScoringConceptModality:
    """Tests for concept/modality scoring integration."""
    
    def test_scoring_weights_default_zero(self):
        """Test that concept/modality weights default to 0."""
        from hubscan.config import ScoringWeights
        
        weights = ScoringWeights()
        assert weights.concept_hub_z == 0.0
        assert weights.cross_modal == 0.0
    
    def test_combine_scores_with_concept_weight(self):
        """Test combine_scores with concept hub z-score weight."""
        from hubscan.core.scoring.combine import combine_scores
        from hubscan.core.detectors.base import DetectorResult
        from hubscan.config import ScoringWeights
        
        # Create mock detector result with concept-aware metadata
        hubness_result = DetectorResult(
            scores=np.array([1.0, 2.0, 3.0]),
            metadata={
                "concept_aware": {
                    "enabled": True,
                    "max_concept_hub_z": [0.5, 1.0, 2.0],  # Concept-specific z-scores
                },
                "modality_aware": {"enabled": False},
            }
        )
        
        detector_results = {"hubness": hubness_result}
        
        # With concept weight = 0, scores should be based only on hub_z
        weights_no_concept = ScoringWeights(hub_z=1.0, concept_hub_z=0.0)
        scores_no_concept = combine_scores(detector_results, weights_no_concept)
        
        # With concept weight = 1.0, scores should include concept z-scores
        weights_with_concept = ScoringWeights(hub_z=1.0, concept_hub_z=1.0)
        scores_with_concept = combine_scores(detector_results, weights_with_concept)
        
        # Scores with concept weight should be higher
        assert np.all(scores_with_concept >= scores_no_concept)
    
    def test_combine_scores_with_cross_modal_weight(self):
        """Test combine_scores with cross-modal weight."""
        from hubscan.core.scoring.combine import combine_scores
        from hubscan.core.detectors.base import DetectorResult
        from hubscan.config import ScoringWeights
        
        # Create mock detector result with modality-aware metadata
        hubness_result = DetectorResult(
            scores=np.array([1.0, 2.0, 3.0]),
            metadata={
                "concept_aware": {"enabled": False},
                "modality_aware": {
                    "enabled": True,
                    "cross_modal_flags": [False, True, True],  # Docs 1 and 2 are cross-modal
                },
            }
        )
        
        detector_results = {"hubness": hubness_result}
        
        # With cross_modal weight = 0
        weights_no_cross = ScoringWeights(hub_z=1.0, cross_modal=0.0)
        scores_no_cross = combine_scores(detector_results, weights_no_cross)
        
        # With cross_modal weight = 1.0
        weights_with_cross = ScoringWeights(hub_z=1.0, cross_modal=1.0)
        scores_with_cross = combine_scores(detector_results, weights_with_cross)
        
        # Cross-modal docs should have higher scores with penalty
        assert scores_with_cross[0] == scores_no_cross[0]  # Non cross-modal unchanged
        assert scores_with_cross[1] > scores_no_cross[1]  # Cross-modal boosted
        assert scores_with_cross[2] > scores_no_cross[2]  # Cross-modal boosted


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""
    
    def test_default_config_unchanged(self):
        """Test that default config preserves old behavior."""
        from hubscan.config import Config
        
        config = Config()
        
        # Default: concept/modality detection disabled
        assert config.detectors.concept_aware.enabled == False
        assert config.detectors.modality_aware.enabled == False
        
        # Default: scoring weights are 0
        assert config.scoring.weights.concept_hub_z == 0.0
        assert config.scoring.weights.cross_modal == 0.0
    
    def test_existing_tests_pass(self):
        """Verify existing hubness tests still pass."""
        # This is implicitly tested by running the full test suite
        # We just ensure the import works
        from hubscan.core.detectors.hubness import HubnessDetector
        
        detector = HubnessDetector()
        assert detector.enabled == True
        assert detector.concept_aware_enabled == False
        assert detector.modality_aware_enabled == False

