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

"""Modality resolver implementations."""

import logging
from typing import Dict, List, Optional, Any, Set

from .base import ModalityResolver, ModalityAssignment

logger = logging.getLogger(__name__)


class MetadataModalityResolver(ModalityResolver):
    """Modality resolver that uses metadata fields.
    
    Extracts modality labels from specified metadata fields for queries and documents.
    """
    
    def __init__(
        self,
        doc_modality_field: str = "modality",
        query_modality_field: str = "modality",
        default_doc_modality: str = "text",
        default_query_modality: str = "text",
        known_modalities: Optional[Set[str]] = None,
    ):
        """Initialize metadata modality resolver.
        
        Args:
            doc_modality_field: Metadata field for document modality
            query_modality_field: Metadata field for query modality
            default_doc_modality: Default modality for docs without metadata
            default_query_modality: Default modality for queries without metadata
            known_modalities: Set of known/expected modality values
        """
        self.doc_modality_field = doc_modality_field
        self.query_modality_field = query_modality_field
        self._default_doc_modality = default_doc_modality
        self._default_query_modality = default_query_modality
        self.known_modalities = known_modalities or {"text", "image", "audio", "video", "code"}
    
    @property
    def name(self) -> str:
        return f"metadata:{self.doc_modality_field}/{self.query_modality_field}"
    
    @property
    def requires_metadata(self) -> bool:
        return True
    
    @property
    def default_query_modality(self) -> str:
        return self._default_query_modality
    
    @property
    def default_doc_modality(self) -> str:
        return self._default_doc_modality
    
    def resolve_modalities(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        num_docs: Optional[int] = None,
    ) -> ModalityAssignment:
        """Resolve modalities from metadata."""
        self.validate_inputs(query_metadata, doc_metadata, num_queries, num_docs)
        
        query_modalities: Dict[int, str] = {}
        doc_modalities: Dict[int, str] = {}
        observed_modalities: Set[str] = set()
        modality_stats: Dict[str, Dict[str, Any]] = {}
        
        # Resolve query modalities
        if query_metadata:
            for idx, meta in enumerate(query_metadata):
                modality = meta.get(self.query_modality_field, self._default_query_modality)
                if modality is None:
                    modality = self._default_query_modality
                modality = str(modality).lower()
                query_modalities[idx] = modality
                observed_modalities.add(modality)
        elif num_queries:
            # Default assignment
            for idx in range(num_queries):
                query_modalities[idx] = self._default_query_modality
                observed_modalities.add(self._default_query_modality)
        
        # Resolve document modalities
        if doc_metadata:
            for idx, meta in enumerate(doc_metadata):
                modality = meta.get(self.doc_modality_field, self._default_doc_modality)
                if modality is None:
                    modality = self._default_doc_modality
                modality = str(modality).lower()
                doc_modalities[idx] = modality
                observed_modalities.add(modality)
        elif num_docs:
            # Default assignment
            for idx in range(num_docs):
                doc_modalities[idx] = self._default_doc_modality
                observed_modalities.add(self._default_doc_modality)
        
        # Compute statistics
        for modality in observed_modalities:
            query_count = sum(1 for m in query_modalities.values() if m == modality)
            doc_count = sum(1 for m in doc_modalities.values() if m == modality)
            modality_stats[modality] = {
                "query_count": query_count,
                "doc_count": doc_count,
                "total_count": query_count + doc_count,
            }
        
        return ModalityAssignment(
            query_modalities=query_modalities,
            doc_modalities=doc_modalities,
            known_modalities=self.known_modalities,
            modality_stats=modality_stats,
        )


class DefaultUnknownModalityResolver(ModalityResolver):
    """Modality resolver that assigns a default modality to everything.
    
    Use this when modality information is not available or not needed.
    """
    
    def __init__(
        self,
        default_modality: str = "text",
    ):
        """Initialize default modality resolver.
        
        Args:
            default_modality: The modality to assign to all items
        """
        self._default_modality = default_modality
    
    @property
    def name(self) -> str:
        return f"default:{self._default_modality}"
    
    @property
    def default_query_modality(self) -> str:
        return self._default_modality
    
    @property
    def default_doc_modality(self) -> str:
        return self._default_modality
    
    def resolve_modalities(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        num_docs: Optional[int] = None,
    ) -> ModalityAssignment:
        """Assign default modality to all items."""
        # Determine counts
        actual_num_queries = len(query_metadata) if query_metadata else (num_queries or 0)
        actual_num_docs = len(doc_metadata) if doc_metadata else (num_docs or 0)
        
        query_modalities = {idx: self._default_modality for idx in range(actual_num_queries)}
        doc_modalities = {idx: self._default_modality for idx in range(actual_num_docs)}
        
        modality_stats = {
            self._default_modality: {
                "query_count": actual_num_queries,
                "doc_count": actual_num_docs,
                "total_count": actual_num_queries + actual_num_docs,
            }
        }
        
        return ModalityAssignment(
            query_modalities=query_modalities,
            doc_modalities=doc_modalities,
            known_modalities={self._default_modality},
            modality_stats=modality_stats,
        )


class HybridModalityResolver(ModalityResolver):
    """Hybrid resolver that tries metadata first, then falls back to default.
    """
    
    def __init__(
        self,
        doc_modality_field: str = "modality",
        query_modality_field: str = "modality",
        default_doc_modality: str = "text",
        default_query_modality: str = "text",
        known_modalities: Optional[Set[str]] = None,
        min_metadata_coverage: float = 0.1,
    ):
        """Initialize hybrid modality resolver.
        
        Args:
            doc_modality_field: Metadata field for document modality
            query_modality_field: Metadata field for query modality
            default_doc_modality: Default modality for documents
            default_query_modality: Default modality for queries
            known_modalities: Set of known modality values
            min_metadata_coverage: Minimum fraction to use metadata resolver
        """
        self.doc_modality_field = doc_modality_field
        self.query_modality_field = query_modality_field
        self._default_doc_modality = default_doc_modality
        self._default_query_modality = default_query_modality
        self.known_modalities = known_modalities or {"text", "image", "audio", "video", "code"}
        self.min_metadata_coverage = min_metadata_coverage
        
        self._active_resolver: Optional[ModalityResolver] = None
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    @property
    def default_query_modality(self) -> str:
        return self._default_query_modality
    
    @property
    def default_doc_modality(self) -> str:
        return self._default_doc_modality
    
    def _check_metadata_coverage(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Check fraction of items with modality metadata."""
        total = 0
        with_modality = 0
        
        if query_metadata:
            total += len(query_metadata)
            with_modality += sum(
                1 for m in query_metadata
                if m.get(self.query_modality_field) is not None
            )
        
        if doc_metadata:
            total += len(doc_metadata)
            with_modality += sum(
                1 for m in doc_metadata
                if m.get(self.doc_modality_field) is not None
            )
        
        return with_modality / total if total > 0 else 0.0
    
    def resolve_modalities(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        num_docs: Optional[int] = None,
    ) -> ModalityAssignment:
        """Resolve modalities using metadata or default fallback."""
        coverage = self._check_metadata_coverage(query_metadata, doc_metadata)
        
        if coverage >= self.min_metadata_coverage:
            logger.info(f"Using metadata modality resolver (coverage: {coverage:.1%})")
            self._active_resolver = MetadataModalityResolver(
                doc_modality_field=self.doc_modality_field,
                query_modality_field=self.query_modality_field,
                default_doc_modality=self._default_doc_modality,
                default_query_modality=self._default_query_modality,
                known_modalities=self.known_modalities,
            )
        else:
            logger.info(
                f"Using default modality resolver (coverage: {coverage:.1%} < {self.min_metadata_coverage:.1%})"
            )
            self._active_resolver = DefaultUnknownModalityResolver(
                default_modality=self._default_doc_modality
            )
        
        return self._active_resolver.resolve_modalities(
            query_metadata, doc_metadata, num_queries, num_docs
        )


def create_modality_resolver(
    mode: str = "default_text",
    doc_modality_field: str = "modality",
    query_modality_field: str = "modality",
    default_doc_modality: str = "text",
    default_query_modality: str = "text",
    known_modalities: Optional[List[str]] = None,
) -> ModalityResolver:
    """Factory function to create a modality resolver.
    
    Args:
        mode: Resolver mode ("metadata", "default_text", "hybrid")
        doc_modality_field: Metadata field for document modality
        query_modality_field: Metadata field for query modality
        default_doc_modality: Default modality for documents
        default_query_modality: Default modality for queries
        known_modalities: List of known modality values
        
    Returns:
        Configured ModalityResolver instance
    """
    known_set = set(known_modalities) if known_modalities else None
    
    if mode == "metadata":
        return MetadataModalityResolver(
            doc_modality_field=doc_modality_field,
            query_modality_field=query_modality_field,
            default_doc_modality=default_doc_modality,
            default_query_modality=default_query_modality,
            known_modalities=known_set,
        )
    
    elif mode == "default_text":
        return DefaultUnknownModalityResolver(default_modality=default_doc_modality)
    
    elif mode == "hybrid":
        return HybridModalityResolver(
            doc_modality_field=doc_modality_field,
            query_modality_field=query_modality_field,
            default_doc_modality=default_doc_modality,
            default_query_modality=default_query_modality,
            known_modalities=known_set,
        )
    
    else:
        raise ValueError(f"Unknown modality resolver mode: {mode}")

