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

"""Base class for concept providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class ConceptAssignment:
    """Represents concept assignments for queries and/or documents.
    
    Attributes:
        query_concepts: Mapping from query index to concept ID (int)
        doc_concepts: Mapping from document index to concept ID (int)
        concept_names: Mapping from concept ID to human-readable name
        concept_stats: Statistics about each concept (size, centroid, etc.)
        fallback_used: Whether fallback clustering was used
    """
    query_concepts: Dict[int, int] = field(default_factory=dict)
    doc_concepts: Dict[int, int] = field(default_factory=dict)
    concept_names: Dict[int, str] = field(default_factory=dict)
    concept_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    fallback_used: bool = False
    
    @property
    def num_concepts(self) -> int:
        """Return the number of unique concepts."""
        all_concepts = set(self.query_concepts.values()) | set(self.doc_concepts.values())
        return len(all_concepts)
    
    @property
    def concept_ids(self) -> List[int]:
        """Return sorted list of concept IDs."""
        all_concepts = set(self.query_concepts.values()) | set(self.doc_concepts.values())
        return sorted(all_concepts)
    
    def get_query_concept(self, query_idx: int, default: int = -1) -> int:
        """Get concept ID for a query, returning default if not assigned."""
        return self.query_concepts.get(query_idx, default)
    
    def get_doc_concept(self, doc_idx: int, default: int = -1) -> int:
        """Get concept ID for a document, returning default if not assigned."""
        return self.doc_concepts.get(doc_idx, default)
    
    def get_concept_name(self, concept_id: int) -> str:
        """Get human-readable name for a concept."""
        return self.concept_names.get(concept_id, f"concept_{concept_id}")
    
    def queries_for_concept(self, concept_id: int) -> List[int]:
        """Get list of query indices for a given concept."""
        return [qidx for qidx, cid in self.query_concepts.items() if cid == concept_id]
    
    def docs_for_concept(self, concept_id: int) -> List[int]:
        """Get list of document indices for a given concept."""
        return [didx for didx, cid in self.doc_concepts.items() if cid == concept_id]


class ConceptProvider(ABC):
    """Abstract base class for concept/topic assignment.
    
    ConceptProviders assign semantic concepts or topics to queries and/or documents.
    This enables concept-specific hub detection, where hubs are detected within
    each concept cluster separately.
    
    Implementations may use:
    - Metadata labels (e.g., category, topic fields)
    - Query embedding clustering (MiniBatchKMeans, etc.)
    - Document embedding clustering
    - Hybrid approaches (metadata with clustering fallback)
    """
    
    @abstractmethod
    def assign_concepts(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ConceptAssignment:
        """Assign concepts to queries and/or documents.
        
        Args:
            query_embeddings: Query embedding vectors (N_q, D)
            doc_embeddings: Document embedding vectors (N_d, D)
            query_metadata: Metadata dicts for each query
            doc_metadata: Metadata dicts for each document
            
        Returns:
            ConceptAssignment with query and/or document concept mappings
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @property
    def requires_embeddings(self) -> bool:
        """Whether this provider requires embeddings to function."""
        return False
    
    @property
    def requires_metadata(self) -> bool:
        """Whether this provider requires metadata to function."""
        return False
    
    def validate_inputs(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Validate that required inputs are provided.
        
        Args:
            query_embeddings: Query embedding vectors
            doc_embeddings: Document embedding vectors  
            query_metadata: Metadata dicts for queries
            doc_metadata: Metadata dicts for documents
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required inputs are missing
        """
        if self.requires_embeddings:
            if query_embeddings is None and doc_embeddings is None:
                raise ValueError(
                    f"{self.name} requires embeddings but none provided"
                )
        if self.requires_metadata:
            if query_metadata is None and doc_metadata is None:
                raise ValueError(
                    f"{self.name} requires metadata but none provided"
                )
        return True

