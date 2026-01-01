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

"""Tests for concept providers."""

import pytest
import numpy as np

from hubscan.core.concepts import (
    ConceptProvider,
    ConceptAssignment,
    MetadataConceptProvider,
    QueryClusteringConceptProvider,
    DocClusteringConceptProvider,
    HybridConceptProvider,
    create_concept_provider,
)


class TestConceptAssignment:
    """Tests for ConceptAssignment dataclass."""
    
    def test_empty_assignment(self):
        """Test empty concept assignment."""
        assignment = ConceptAssignment()
        assert assignment.num_concepts == 0
        assert assignment.concept_ids == []
        assert assignment.get_query_concept(0) == -1
        assert assignment.get_doc_concept(0) == -1
    
    def test_basic_assignment(self):
        """Test basic concept assignment."""
        assignment = ConceptAssignment(
            query_concepts={0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
            doc_concepts={0: 0, 1: 1},
            concept_names={0: "python", 1: "ml", 2: "web"},
        )
        
        assert assignment.num_concepts == 3
        assert assignment.concept_ids == [0, 1, 2]
        assert assignment.get_query_concept(0) == 0
        assert assignment.get_query_concept(2) == 1
        assert assignment.get_concept_name(0) == "python"
        assert assignment.get_concept_name(99) == "concept_99"  # Default name
        
    def test_queries_for_concept(self):
        """Test getting queries for a concept."""
        assignment = ConceptAssignment(
            query_concepts={0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
        )
        
        assert set(assignment.queries_for_concept(0)) == {0, 1}
        assert set(assignment.queries_for_concept(1)) == {2, 3}
        assert assignment.queries_for_concept(99) == []


class TestMetadataConceptProvider:
    """Tests for MetadataConceptProvider."""
    
    def test_basic_metadata_assignment(self):
        """Test assignment from metadata."""
        provider = MetadataConceptProvider(metadata_field="category")
        
        query_metadata = [
            {"category": "python"},
            {"category": "python"},
            {"category": "ml"},
            {"category": "ml"},
        ]
        doc_metadata = [
            {"category": "python"},
            {"category": "web"},
        ]
        
        assignment = provider.assign_concepts(
            query_metadata=query_metadata,
            doc_metadata=doc_metadata,
        )
        
        assert assignment.num_concepts == 3
        assert not assignment.fallback_used
        assert assignment.get_concept_name(assignment.get_query_concept(0)) == "python"
    
    def test_missing_metadata_field(self):
        """Test handling of missing metadata field."""
        provider = MetadataConceptProvider(
            metadata_field="category",
            unknown_concept_name="unknown"
        )
        
        query_metadata = [
            {"category": "python"},
            {},  # Missing field
            {"category": None},  # None value
        ]
        
        assignment = provider.assign_concepts(query_metadata=query_metadata)
        
        # Unknown should be assigned
        assert "unknown" in assignment.concept_names.values()
    
    def test_requires_metadata(self):
        """Test that provider requires metadata."""
        provider = MetadataConceptProvider()
        assert provider.requires_metadata
        
        with pytest.raises(ValueError, match="requires metadata"):
            provider.assign_concepts()


class TestQueryClusteringConceptProvider:
    """Tests for QueryClusteringConceptProvider."""
    
    def test_basic_clustering(self):
        """Test clustering query embeddings."""
        np.random.seed(0)
        
        # Create embeddings with 3 clear clusters
        cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        query_embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        
        provider = QueryClusteringConceptProvider(
            num_concepts=3,
            seed=0,
        )
        
        assignment = provider.assign_concepts(query_embeddings=query_embeddings)
        
        assert assignment.num_concepts >= 2  # At least 2 concepts detected
        assert len(assignment.query_concepts) == 60
        
    def test_requires_embeddings(self):
        """Test that provider requires embeddings."""
        provider = QueryClusteringConceptProvider()
        assert provider.requires_embeddings
        
        with pytest.raises(ValueError, match="requires embeddings"):
            provider.assign_concepts()
    
    def test_small_dataset_adjustment(self):
        """Test that num_concepts is adjusted for small datasets."""
        np.random.seed(0)
        query_embeddings = np.random.randn(5, 10).astype(np.float32)
        
        provider = QueryClusteringConceptProvider(
            num_concepts=100,  # More than queries
            seed=0,
        )
        
        # Should not raise, should adjust
        assignment = provider.assign_concepts(query_embeddings=query_embeddings)
        assert assignment.num_concepts <= 5
    
    def test_deterministic_clustering(self):
        """Test that clustering is deterministic with same seed."""
        np.random.seed(0)
        query_embeddings = np.random.randn(50, 10).astype(np.float32)
        
        provider1 = QueryClusteringConceptProvider(num_concepts=5, seed=0)
        provider2 = QueryClusteringConceptProvider(num_concepts=5, seed=0)
        
        assignment1 = provider1.assign_concepts(query_embeddings=query_embeddings)
        assignment2 = provider2.assign_concepts(query_embeddings=query_embeddings)
        
        assert assignment1.query_concepts == assignment2.query_concepts
    
    def test_min_concept_size(self):
        """Test minimum concept size merging."""
        np.random.seed(0)
        # Create highly imbalanced clusters
        query_embeddings = np.random.randn(50, 10).astype(np.float32)
        
        provider = QueryClusteringConceptProvider(
            num_concepts=10,
            min_concept_size=15,  # High threshold - some clusters will merge
            seed=0,
        )
        
        assignment = provider.assign_concepts(query_embeddings=query_embeddings)
        # Should have "other" concept for small clusters
        # Assignment should work
        assert len(assignment.query_concepts) == 50


class TestDocClusteringConceptProvider:
    """Tests for DocClusteringConceptProvider."""
    
    def test_doc_clustering_with_queries(self):
        """Test clustering documents and assigning queries."""
        np.random.seed(0)
        
        # Documents with clear clusters
        doc_embeddings = np.vstack([
            np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]).astype(np.float32)
        
        # Queries from both clusters
        query_embeddings = np.vstack([
            np.random.randn(10, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(10, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]).astype(np.float32)
        
        provider = DocClusteringConceptProvider(
            num_concepts=2,
            seed=0,
        )
        
        assignment = provider.assign_concepts(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
        )
        
        assert assignment.num_concepts == 2
        assert len(assignment.doc_concepts) == 40
        assert len(assignment.query_concepts) == 20


class TestHybridConceptProvider:
    """Tests for HybridConceptProvider."""
    
    def test_uses_metadata_when_available(self):
        """Test that hybrid uses metadata when coverage is sufficient."""
        np.random.seed(0)
        query_embeddings = np.random.randn(20, 10).astype(np.float32)
        query_metadata = [{"concept": f"concept_{i % 3}"} for i in range(20)]
        
        provider = HybridConceptProvider(
            metadata_field="concept",
            min_metadata_coverage=0.5,
        )
        
        assignment = provider.assign_concepts(
            query_embeddings=query_embeddings,
            query_metadata=query_metadata,
        )
        
        assert not assignment.fallback_used
        assert "concept_0" in assignment.concept_names.values()
    
    def test_falls_back_to_clustering(self):
        """Test that hybrid falls back to clustering when metadata insufficient."""
        np.random.seed(0)
        query_embeddings = np.random.randn(20, 10).astype(np.float32)
        # Only 20% coverage
        query_metadata = [{"concept": "test"} if i < 4 else {} for i in range(20)]
        
        provider = HybridConceptProvider(
            metadata_field="concept",
            min_metadata_coverage=0.5,  # Requires 50%
            num_concepts=3,
            seed=0,
        )
        
        assignment = provider.assign_concepts(
            query_embeddings=query_embeddings,
            query_metadata=query_metadata,
        )
        
        assert assignment.fallback_used
        assert "cluster_" in list(assignment.concept_names.values())[0]


class TestCreateConceptProvider:
    """Tests for create_concept_provider factory function."""
    
    def test_create_metadata_provider(self):
        """Test creating metadata provider."""
        provider = create_concept_provider(mode="metadata", metadata_field="topic")
        assert isinstance(provider, MetadataConceptProvider)
    
    def test_create_query_clustering_provider(self):
        """Test creating query clustering provider."""
        provider = create_concept_provider(
            mode="query_clustering",
            num_concepts=5,
        )
        assert isinstance(provider, QueryClusteringConceptProvider)
    
    def test_create_doc_clustering_provider(self):
        """Test creating doc clustering provider."""
        provider = create_concept_provider(mode="doc_clustering")
        assert isinstance(provider, DocClusteringConceptProvider)
    
    def test_create_hybrid_provider(self):
        """Test creating hybrid provider."""
        provider = create_concept_provider(mode="hybrid")
        assert isinstance(provider, HybridConceptProvider)
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown concept provider mode"):
            create_concept_provider(mode="invalid")

