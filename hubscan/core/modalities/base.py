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

"""Base class for modality resolvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np


@dataclass
class ModalityAssignment:
    """Represents modality assignments for queries and documents.
    
    Attributes:
        query_modalities: Mapping from query index to modality string
        doc_modalities: Mapping from document index to modality string
        known_modalities: Set of all known modalities in this assignment
        modality_stats: Statistics about each modality (count, etc.)
    """
    query_modalities: Dict[int, str] = field(default_factory=dict)
    doc_modalities: Dict[int, str] = field(default_factory=dict)
    known_modalities: Set[str] = field(default_factory=set)
    modality_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def num_query_modalities(self) -> int:
        """Return the number of unique query modalities."""
        return len(set(self.query_modalities.values()))
    
    @property
    def num_doc_modalities(self) -> int:
        """Return the number of unique document modalities."""
        return len(set(self.doc_modalities.values()))
    
    @property
    def all_modalities(self) -> List[str]:
        """Return sorted list of all modalities (queries + docs)."""
        all_mods = set(self.query_modalities.values()) | set(self.doc_modalities.values())
        return sorted(all_mods)
    
    def get_query_modality(self, query_idx: int, default: str = "unknown") -> str:
        """Get modality for a query, returning default if not assigned."""
        return self.query_modalities.get(query_idx, default)
    
    def get_doc_modality(self, doc_idx: int, default: str = "unknown") -> str:
        """Get modality for a document, returning default if not assigned."""
        return self.doc_modalities.get(doc_idx, default)
    
    def is_cross_modal(self, query_idx: int, doc_idx: int) -> bool:
        """Check if a query-doc pair is cross-modal (different modalities)."""
        query_mod = self.get_query_modality(query_idx)
        doc_mod = self.get_doc_modality(doc_idx)
        return query_mod != doc_mod
    
    def queries_for_modality(self, modality: str) -> List[int]:
        """Get list of query indices for a given modality."""
        return [qidx for qidx, mod in self.query_modalities.items() if mod == modality]
    
    def docs_for_modality(self, modality: str) -> List[int]:
        """Get list of document indices for a given modality."""
        return [didx for didx, mod in self.doc_modalities.items() if mod == modality]
    
    def get_modality_pair_counts(self) -> Dict[Tuple[str, str], int]:
        """Get counts for each (query_modality, doc_modality) pair.
        
        Returns:
            Dict mapping (query_mod, doc_mod) tuples to counts
        """
        # This would be populated during detection
        return {}


class ModalityResolver(ABC):
    """Abstract base class for modality resolution.
    
    ModalityResolvers assign modality labels (e.g., 'text', 'image', 'audio')
    to queries and documents. This enables modality-aware hub detection,
    specifically detecting cross-modal hubs that appear relevant to queries
    of a different modality.
    
    Implementations may use:
    - Metadata fields (e.g., 'modality', 'type' fields)
    - Default assumptions (e.g., all text)
    - Hybrid approaches (metadata with defaults)
    """
    
    @abstractmethod
    def resolve_modalities(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        num_docs: Optional[int] = None,
    ) -> ModalityAssignment:
        """Resolve modalities for queries and documents.
        
        Args:
            query_metadata: Metadata dicts for each query
            doc_metadata: Metadata dicts for each document
            num_queries: Number of queries (for default assignment)
            num_docs: Number of documents (for default assignment)
            
        Returns:
            ModalityAssignment with query and document modality mappings
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this resolver."""
        pass
    
    @property
    def default_query_modality(self) -> str:
        """Default modality for queries without metadata."""
        return "text"
    
    @property
    def default_doc_modality(self) -> str:
        """Default modality for documents without metadata."""
        return "text"
    
    @property
    def requires_metadata(self) -> bool:
        """Whether this resolver requires metadata to function."""
        return False
    
    def validate_inputs(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
        num_queries: Optional[int] = None,
        num_docs: Optional[int] = None,
    ) -> bool:
        """Validate that required inputs are provided.
        
        Args:
            query_metadata: Metadata dicts for queries
            doc_metadata: Metadata dicts for documents
            num_queries: Number of queries
            num_docs: Number of documents
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required inputs are missing
        """
        if self.requires_metadata:
            if query_metadata is None and doc_metadata is None:
                raise ValueError(
                    f"{self.name} requires metadata but none provided"
                )
        return True

