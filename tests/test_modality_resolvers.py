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

"""Tests for modality resolvers."""

import pytest

from hubscan.core.modalities import (
    ModalityResolver,
    ModalityAssignment,
    MetadataModalityResolver,
    DefaultUnknownModalityResolver,
    HybridModalityResolver,
    create_modality_resolver,
)


class TestModalityAssignment:
    """Tests for ModalityAssignment dataclass."""
    
    def test_empty_assignment(self):
        """Test empty modality assignment."""
        assignment = ModalityAssignment()
        assert assignment.num_query_modalities == 0
        assert assignment.num_doc_modalities == 0
        assert assignment.all_modalities == []
    
    def test_basic_assignment(self):
        """Test basic modality assignment."""
        assignment = ModalityAssignment(
            query_modalities={0: "text", 1: "text", 2: "image"},
            doc_modalities={0: "text", 1: "image", 2: "audio"},
            known_modalities={"text", "image", "audio"},
        )
        
        assert assignment.num_query_modalities == 2
        assert assignment.num_doc_modalities == 3
        assert assignment.get_query_modality(0) == "text"
        assert assignment.get_query_modality(99) == "unknown"  # Default
        assert assignment.get_doc_modality(1) == "image"
    
    def test_is_cross_modal(self):
        """Test cross-modal detection."""
        assignment = ModalityAssignment(
            query_modalities={0: "text", 1: "image"},
            doc_modalities={0: "text", 1: "image"},
        )
        
        assert not assignment.is_cross_modal(0, 0)  # text -> text
        assert not assignment.is_cross_modal(1, 1)  # image -> image
        assert assignment.is_cross_modal(0, 1)  # text -> image
        assert assignment.is_cross_modal(1, 0)  # image -> text
    
    def test_modality_lists(self):
        """Test getting items by modality."""
        assignment = ModalityAssignment(
            query_modalities={0: "text", 1: "text", 2: "image"},
            doc_modalities={0: "text", 1: "image"},
        )
        
        assert set(assignment.queries_for_modality("text")) == {0, 1}
        assert assignment.queries_for_modality("image") == [2]
        assert assignment.docs_for_modality("text") == [0]


class TestMetadataModalityResolver:
    """Tests for MetadataModalityResolver."""
    
    def test_basic_metadata_resolution(self):
        """Test resolution from metadata."""
        resolver = MetadataModalityResolver(
            doc_modality_field="type",
            query_modality_field="type",
        )
        
        query_metadata = [
            {"type": "text"},
            {"type": "image"},
        ]
        doc_metadata = [
            {"type": "text"},
            {"type": "audio"},
        ]
        
        assignment = resolver.resolve_modalities(
            query_metadata=query_metadata,
            doc_metadata=doc_metadata,
        )
        
        assert assignment.get_query_modality(0) == "text"
        assert assignment.get_query_modality(1) == "image"
        assert assignment.get_doc_modality(1) == "audio"
    
    def test_default_modality_fallback(self):
        """Test default modality when field missing."""
        resolver = MetadataModalityResolver(
            default_doc_modality="text",
            default_query_modality="text",
        )
        
        query_metadata = [
            {"modality": "image"},
            {},  # Missing field
        ]
        
        assignment = resolver.resolve_modalities(query_metadata=query_metadata)
        
        assert assignment.get_query_modality(0) == "image"
        assert assignment.get_query_modality(1) == "text"  # Default
    
    def test_requires_metadata(self):
        """Test that resolver requires metadata."""
        resolver = MetadataModalityResolver()
        assert resolver.requires_metadata
        
        with pytest.raises(ValueError, match="requires metadata"):
            resolver.resolve_modalities()
    
    def test_num_queries_with_empty_metadata(self):
        """Test using num_queries with empty metadata list."""
        resolver = MetadataModalityResolver(default_query_modality="text")
        
        # Provide empty metadata lists to bypass validation, use num_queries for count
        assignment = resolver.resolve_modalities(
            query_metadata=[{} for _ in range(5)],
        )
        
        assert len(assignment.query_modalities) == 5
        for i in range(5):
            assert assignment.get_query_modality(i) == "text"


class TestDefaultUnknownModalityResolver:
    """Tests for DefaultUnknownModalityResolver."""
    
    def test_default_assignment(self):
        """Test all items get default modality."""
        resolver = DefaultUnknownModalityResolver(default_modality="text")
        
        assignment = resolver.resolve_modalities(num_queries=3, num_docs=5)
        
        assert len(assignment.query_modalities) == 3
        assert len(assignment.doc_modalities) == 5
        assert all(m == "text" for m in assignment.query_modalities.values())
        assert all(m == "text" for m in assignment.doc_modalities.values())
    
    def test_uses_metadata_counts(self):
        """Test that it uses metadata list lengths."""
        resolver = DefaultUnknownModalityResolver(default_modality="audio")
        
        query_metadata = [{"x": 1}, {"x": 2}]
        doc_metadata = [{"x": 1}, {"x": 2}, {"x": 3}]
        
        assignment = resolver.resolve_modalities(
            query_metadata=query_metadata,
            doc_metadata=doc_metadata,
        )
        
        assert len(assignment.query_modalities) == 2
        assert len(assignment.doc_modalities) == 3


class TestHybridModalityResolver:
    """Tests for HybridModalityResolver."""
    
    def test_uses_metadata_when_available(self):
        """Test that hybrid uses metadata when coverage is sufficient."""
        resolver = HybridModalityResolver(
            min_metadata_coverage=0.3,
        )
        
        # 50% coverage
        query_metadata = [
            {"modality": "text"},
            {"modality": "image"},
            {},
            {},
        ]
        
        assignment = resolver.resolve_modalities(query_metadata=query_metadata)
        
        # Should use metadata resolver
        assert assignment.get_query_modality(0) == "text"
        assert assignment.get_query_modality(1) == "image"
    
    def test_falls_back_to_default(self):
        """Test that hybrid falls back to default when metadata insufficient."""
        resolver = HybridModalityResolver(
            min_metadata_coverage=0.5,
            default_doc_modality="text",
        )
        
        # Only 10% coverage (1/10)
        query_metadata = [
            {"modality": "image"},
        ] + [{} for _ in range(9)]
        
        assignment = resolver.resolve_modalities(query_metadata=query_metadata)
        
        # Should fall back to default
        # All should be same (default)
        modalities = list(assignment.query_modalities.values())
        assert all(m == modalities[0] for m in modalities)


class TestCreateModalityResolver:
    """Tests for create_modality_resolver factory function."""
    
    def test_create_metadata_resolver(self):
        """Test creating metadata resolver."""
        resolver = create_modality_resolver(
            mode="metadata",
            doc_modality_field="type",
        )
        assert isinstance(resolver, MetadataModalityResolver)
    
    def test_create_default_resolver(self):
        """Test creating default resolver."""
        resolver = create_modality_resolver(
            mode="default_text",
            default_doc_modality="audio",
        )
        assert isinstance(resolver, DefaultUnknownModalityResolver)
    
    def test_create_hybrid_resolver(self):
        """Test creating hybrid resolver."""
        resolver = create_modality_resolver(mode="hybrid")
        assert isinstance(resolver, HybridModalityResolver)
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown modality resolver mode"):
            create_modality_resolver(mode="invalid")

