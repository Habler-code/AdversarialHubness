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

"""Concept provider implementations."""

import logging
from typing import Dict, List, Optional, Any, Literal

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from .base import ConceptProvider, ConceptAssignment

logger = logging.getLogger(__name__)


class MetadataConceptProvider(ConceptProvider):
    """Concept provider that uses metadata labels.
    
    Extracts concept/topic labels from a specified metadata field.
    """
    
    def __init__(
        self,
        metadata_field: str = "concept",
        unknown_concept_name: str = "unknown",
    ):
        """Initialize metadata concept provider.
        
        Args:
            metadata_field: Name of the metadata field containing concept labels
            unknown_concept_name: Name to use for documents/queries without concept
        """
        self.metadata_field = metadata_field
        self.unknown_concept_name = unknown_concept_name
    
    @property
    def name(self) -> str:
        return f"metadata:{self.metadata_field}"
    
    @property
    def requires_metadata(self) -> bool:
        return True
    
    def assign_concepts(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ConceptAssignment:
        """Assign concepts from metadata labels."""
        self.validate_inputs(
            query_embeddings, doc_embeddings, query_metadata, doc_metadata
        )
        
        # Build concept name to ID mapping
        concept_name_to_id: Dict[str, int] = {}
        concept_id_to_name: Dict[int, str] = {}
        
        def get_or_create_concept_id(name: str) -> int:
            if name not in concept_name_to_id:
                new_id = len(concept_name_to_id)
                concept_name_to_id[name] = new_id
                concept_id_to_name[new_id] = name
            return concept_name_to_id[name]
        
        # Assign query concepts
        query_concepts: Dict[int, int] = {}
        if query_metadata:
            for idx, meta in enumerate(query_metadata):
                concept_name = meta.get(self.metadata_field, self.unknown_concept_name)
                if concept_name is None:
                    concept_name = self.unknown_concept_name
                query_concepts[idx] = get_or_create_concept_id(str(concept_name))
        
        # Assign document concepts
        doc_concepts: Dict[int, int] = {}
        if doc_metadata:
            for idx, meta in enumerate(doc_metadata):
                concept_name = meta.get(self.metadata_field, self.unknown_concept_name)
                if concept_name is None:
                    concept_name = self.unknown_concept_name
                doc_concepts[idx] = get_or_create_concept_id(str(concept_name))
        
        # Compute concept statistics
        concept_stats: Dict[int, Dict[str, Any]] = {}
        for concept_id in concept_id_to_name:
            query_count = sum(1 for c in query_concepts.values() if c == concept_id)
            doc_count = sum(1 for c in doc_concepts.values() if c == concept_id)
            concept_stats[concept_id] = {
                "query_count": query_count,
                "doc_count": doc_count,
                "total_count": query_count + doc_count,
            }
        
        return ConceptAssignment(
            query_concepts=query_concepts,
            doc_concepts=doc_concepts,
            concept_names=concept_id_to_name,
            concept_stats=concept_stats,
            fallback_used=False,
        )


class QueryClusteringConceptProvider(ConceptProvider):
    """Concept provider that clusters query embeddings.
    
    Uses MiniBatchKMeans or KMeans to cluster queries into concepts.
    Concept IDs are assigned deterministically based on cluster population
    and centroid norm for stability.
    """
    
    def __init__(
        self,
        num_concepts: int = 10,
        algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans",
        batch_size: int = 1024,
        n_init: int = 3,
        max_iter: int = 100,
        min_concept_size: int = 10,
        seed: int = 42,
    ):
        """Initialize query clustering concept provider.
        
        Args:
            num_concepts: Number of concept clusters to create
            algorithm: Clustering algorithm ("minibatch_kmeans" or "kmeans")
            batch_size: Batch size for MiniBatchKMeans
            n_init: Number of initializations
            max_iter: Maximum iterations
            min_concept_size: Minimum queries per concept (smaller merged to 'other')
            seed: Random seed for reproducibility
        """
        self.num_concepts = num_concepts
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.min_concept_size = min_concept_size
        self.seed = seed
        
        self._clusterer = None
        self._centroids = None
    
    @property
    def name(self) -> str:
        return f"query_clustering:{self.algorithm}:{self.num_concepts}"
    
    @property
    def requires_embeddings(self) -> bool:
        return True
    
    def _create_clusterer(self):
        """Create the clustering model."""
        if self.algorithm == "minibatch_kmeans":
            return MiniBatchKMeans(
                n_clusters=self.num_concepts,
                batch_size=self.batch_size,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
        else:
            return KMeans(
                n_clusters=self.num_concepts,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
    
    def _stabilize_cluster_ids(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> tuple:
        """Re-map cluster IDs for deterministic ordering.
        
        Orders by (population DESC, centroid_norm DESC) for stability.
        
        Args:
            labels: Original cluster labels
            centroids: Cluster centroids
            
        Returns:
            Tuple of (stabilized_labels, stabilized_centroids, id_mapping)
        """
        unique_labels = np.unique(labels)
        
        # Compute population and centroid norms
        cluster_info = []
        for label in unique_labels:
            population = np.sum(labels == label)
            centroid_norm = np.linalg.norm(centroids[label])
            cluster_info.append((label, population, centroid_norm))
        
        # Sort by population (desc), then centroid norm (desc)
        cluster_info.sort(key=lambda x: (-x[1], -x[2]))
        
        # Create mapping from old to new IDs
        old_to_new = {old_id: new_id for new_id, (old_id, _, _) in enumerate(cluster_info)}
        
        # Remap labels and centroids
        new_labels = np.array([old_to_new[l] for l in labels])
        new_centroids = np.array([centroids[old_id] for old_id, _, _ in cluster_info])
        
        return new_labels, new_centroids, old_to_new
    
    def assign_concepts(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ConceptAssignment:
        """Assign concepts by clustering query embeddings."""
        self.validate_inputs(
            query_embeddings, doc_embeddings, query_metadata, doc_metadata
        )
        
        if query_embeddings is None or len(query_embeddings) == 0:
            logger.warning("No query embeddings provided, returning empty assignment")
            return ConceptAssignment(fallback_used=True)
        
        # Adjust num_concepts if we have fewer queries
        actual_num_concepts = min(self.num_concepts, len(query_embeddings))
        if actual_num_concepts < self.num_concepts:
            logger.info(
                f"Reducing num_concepts from {self.num_concepts} to {actual_num_concepts} "
                f"(only {len(query_embeddings)} queries)"
            )
        
        # Create and fit clusterer
        self._clusterer = self._create_clusterer()
        if actual_num_concepts < self.num_concepts:
            self._clusterer.n_clusters = actual_num_concepts
        
        labels = self._clusterer.fit_predict(query_embeddings)
        centroids = self._clusterer.cluster_centers_
        
        # Stabilize cluster IDs
        labels, centroids, _ = self._stabilize_cluster_ids(labels, centroids)
        self._centroids = centroids
        
        # Merge small clusters to 'other' concept
        query_concepts: Dict[int, int] = {}
        concept_populations: Dict[int, int] = {}
        
        for idx, label in enumerate(labels):
            concept_populations[label] = concept_populations.get(label, 0) + 1
        
        # Identify small clusters
        other_concept_id = -1
        for concept_id, pop in concept_populations.items():
            if pop < self.min_concept_size:
                if other_concept_id == -1:
                    other_concept_id = max(concept_populations.keys()) + 1
                # Mark for merging
                concept_populations[concept_id] = -1  # Mark as merged
        
        # Assign queries with merging
        for idx, label in enumerate(labels):
            if concept_populations.get(label, 0) == -1:
                query_concepts[idx] = other_concept_id
            else:
                query_concepts[idx] = int(label)
        
        # Build concept names and stats
        concept_names: Dict[int, str] = {}
        concept_stats: Dict[int, Dict[str, Any]] = {}
        
        for concept_id in set(query_concepts.values()):
            if concept_id == other_concept_id:
                concept_names[concept_id] = "other"
            else:
                concept_names[concept_id] = f"cluster_{concept_id}"
            
            count = sum(1 for c in query_concepts.values() if c == concept_id)
            concept_stats[concept_id] = {
                "query_count": count,
                "doc_count": 0,
                "total_count": count,
            }
            if concept_id != other_concept_id and concept_id < len(centroids):
                concept_stats[concept_id]["centroid_norm"] = float(
                    np.linalg.norm(centroids[concept_id])
                )
        
        return ConceptAssignment(
            query_concepts=query_concepts,
            doc_concepts={},
            concept_names=concept_names,
            concept_stats=concept_stats,
            fallback_used=False,
        )


class DocClusteringConceptProvider(ConceptProvider):
    """Concept provider that clusters document embeddings.
    
    Uses clustering on documents to define concepts, then assigns
    queries to nearest concept centroid.
    """
    
    def __init__(
        self,
        num_concepts: int = 10,
        algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans",
        batch_size: int = 1024,
        n_init: int = 3,
        max_iter: int = 100,
        min_concept_size: int = 10,
        seed: int = 42,
    ):
        """Initialize document clustering concept provider.
        
        Args:
            num_concepts: Number of concept clusters to create
            algorithm: Clustering algorithm
            batch_size: Batch size for MiniBatchKMeans
            n_init: Number of initializations
            max_iter: Maximum iterations
            min_concept_size: Minimum docs per concept
            seed: Random seed
        """
        self.num_concepts = num_concepts
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.min_concept_size = min_concept_size
        self.seed = seed
        
        self._clusterer = None
        self._centroids = None
    
    @property
    def name(self) -> str:
        return f"doc_clustering:{self.algorithm}:{self.num_concepts}"
    
    @property
    def requires_embeddings(self) -> bool:
        return True
    
    def _create_clusterer(self):
        """Create the clustering model."""
        if self.algorithm == "minibatch_kmeans":
            return MiniBatchKMeans(
                n_clusters=self.num_concepts,
                batch_size=self.batch_size,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
        else:
            return KMeans(
                n_clusters=self.num_concepts,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
    
    def assign_concepts(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ConceptAssignment:
        """Assign concepts by clustering document embeddings."""
        self.validate_inputs(
            query_embeddings, doc_embeddings, query_metadata, doc_metadata
        )
        
        if doc_embeddings is None or len(doc_embeddings) == 0:
            logger.warning("No document embeddings provided, returning empty assignment")
            return ConceptAssignment(fallback_used=True)
        
        # Adjust num_concepts if needed
        actual_num_concepts = min(self.num_concepts, len(doc_embeddings))
        
        # Create and fit clusterer on documents
        self._clusterer = self._create_clusterer()
        if actual_num_concepts < self.num_concepts:
            self._clusterer.n_clusters = actual_num_concepts
        
        doc_labels = self._clusterer.fit_predict(doc_embeddings)
        self._centroids = self._clusterer.cluster_centers_
        
        # Assign documents to concepts
        doc_concepts: Dict[int, int] = {idx: int(l) for idx, l in enumerate(doc_labels)}
        
        # Assign queries to nearest centroid
        query_concepts: Dict[int, int] = {}
        if query_embeddings is not None:
            query_labels = self._clusterer.predict(query_embeddings)
            query_concepts = {idx: int(l) for idx, l in enumerate(query_labels)}
        
        # Build concept names and stats
        concept_names: Dict[int, str] = {}
        concept_stats: Dict[int, Dict[str, Any]] = {}
        
        for concept_id in range(actual_num_concepts):
            concept_names[concept_id] = f"doc_cluster_{concept_id}"
            query_count = sum(1 for c in query_concepts.values() if c == concept_id)
            doc_count = sum(1 for c in doc_concepts.values() if c == concept_id)
            concept_stats[concept_id] = {
                "query_count": query_count,
                "doc_count": doc_count,
                "total_count": query_count + doc_count,
                "centroid_norm": float(np.linalg.norm(self._centroids[concept_id])),
            }
        
        return ConceptAssignment(
            query_concepts=query_concepts,
            doc_concepts=doc_concepts,
            concept_names=concept_names,
            concept_stats=concept_stats,
            fallback_used=False,
        )


class HybridConceptProvider(ConceptProvider):
    """Hybrid concept provider that tries metadata first, then falls back to clustering.
    
    Behavior:
    - If concept metadata exists, use it
    - Otherwise, auto-cluster queries or documents
    """
    
    def __init__(
        self,
        metadata_field: str = "concept",
        num_concepts: int = 10,
        clustering_algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans",
        cluster_queries: bool = True,
        batch_size: int = 1024,
        n_init: int = 3,
        max_iter: int = 100,
        min_concept_size: int = 10,
        seed: int = 42,
        min_metadata_coverage: float = 0.5,
    ):
        """Initialize hybrid concept provider.
        
        Args:
            metadata_field: Metadata field for concept labels
            num_concepts: Number of clusters for fallback
            clustering_algorithm: Algorithm for fallback clustering
            cluster_queries: If True, cluster queries; if False, cluster documents
            batch_size: Batch size for MiniBatchKMeans
            n_init: Number of initializations
            max_iter: Maximum iterations
            min_concept_size: Minimum items per concept
            seed: Random seed
            min_metadata_coverage: Minimum fraction of items with metadata to use metadata provider
        """
        self.metadata_field = metadata_field
        self.num_concepts = num_concepts
        self.clustering_algorithm = clustering_algorithm
        self.cluster_queries = cluster_queries
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.min_concept_size = min_concept_size
        self.seed = seed
        self.min_metadata_coverage = min_metadata_coverage
        
        self._active_provider: Optional[ConceptProvider] = None
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def _check_metadata_coverage(
        self,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Check what fraction of items have concept metadata."""
        total = 0
        with_concept = 0
        
        if query_metadata:
            total += len(query_metadata)
            with_concept += sum(
                1 for m in query_metadata
                if m.get(self.metadata_field) is not None
            )
        
        if doc_metadata:
            total += len(doc_metadata)
            with_concept += sum(
                1 for m in doc_metadata
                if m.get(self.metadata_field) is not None
            )
        
        return with_concept / total if total > 0 else 0.0
    
    def assign_concepts(
        self,
        query_embeddings: Optional[np.ndarray] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        doc_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> ConceptAssignment:
        """Assign concepts using metadata or clustering fallback."""
        # Check metadata coverage
        coverage = self._check_metadata_coverage(query_metadata, doc_metadata)
        
        if coverage >= self.min_metadata_coverage:
            logger.info(
                f"Using metadata concept provider (coverage: {coverage:.1%})"
            )
            self._active_provider = MetadataConceptProvider(
                metadata_field=self.metadata_field
            )
            result = self._active_provider.assign_concepts(
                query_embeddings, doc_embeddings, query_metadata, doc_metadata
            )
            result.fallback_used = False
            return result
        
        # Fallback to clustering
        logger.info(
            f"Falling back to clustering (metadata coverage: {coverage:.1%} < {self.min_metadata_coverage:.1%})"
        )
        
        if self.cluster_queries:
            self._active_provider = QueryClusteringConceptProvider(
                num_concepts=self.num_concepts,
                algorithm=self.clustering_algorithm,
                batch_size=self.batch_size,
                n_init=self.n_init,
                max_iter=self.max_iter,
                min_concept_size=self.min_concept_size,
                seed=self.seed,
            )
        else:
            self._active_provider = DocClusteringConceptProvider(
                num_concepts=self.num_concepts,
                algorithm=self.clustering_algorithm,
                batch_size=self.batch_size,
                n_init=self.n_init,
                max_iter=self.max_iter,
                min_concept_size=self.min_concept_size,
                seed=self.seed,
            )
        
        result = self._active_provider.assign_concepts(
            query_embeddings, doc_embeddings, query_metadata, doc_metadata
        )
        result.fallback_used = True
        return result


def create_concept_provider(
    mode: str = "hybrid",
    metadata_field: str = "concept",
    num_concepts: int = 10,
    clustering_algorithm: str = "minibatch_kmeans",
    clustering_params: Optional[Dict[str, Any]] = None,
    min_concept_size: int = 10,
    seed: int = 42,
) -> ConceptProvider:
    """Factory function to create a concept provider.
    
    Args:
        mode: Provider mode ("metadata", "query_clustering", "doc_clustering", "hybrid")
        metadata_field: Metadata field for concept labels
        num_concepts: Number of concept clusters
        clustering_algorithm: Clustering algorithm name
        clustering_params: Additional clustering parameters
        min_concept_size: Minimum items per concept
        seed: Random seed
        
    Returns:
        Configured ConceptProvider instance
    """
    clustering_params = clustering_params or {}
    
    if mode == "metadata":
        return MetadataConceptProvider(metadata_field=metadata_field)
    
    elif mode == "query_clustering":
        return QueryClusteringConceptProvider(
            num_concepts=num_concepts,
            algorithm=clustering_algorithm,
            batch_size=clustering_params.get("batch_size", 1024),
            n_init=clustering_params.get("n_init", 3),
            max_iter=clustering_params.get("max_iter", 100),
            min_concept_size=min_concept_size,
            seed=seed,
        )
    
    elif mode == "doc_clustering":
        return DocClusteringConceptProvider(
            num_concepts=num_concepts,
            algorithm=clustering_algorithm,
            batch_size=clustering_params.get("batch_size", 1024),
            n_init=clustering_params.get("n_init", 3),
            max_iter=clustering_params.get("max_iter", 100),
            min_concept_size=min_concept_size,
            seed=seed,
        )
    
    elif mode == "hybrid":
        return HybridConceptProvider(
            metadata_field=metadata_field,
            num_concepts=num_concepts,
            clustering_algorithm=clustering_algorithm,
            batch_size=clustering_params.get("batch_size", 1024),
            n_init=clustering_params.get("n_init", 3),
            max_iter=clustering_params.get("max_iter", 100),
            min_concept_size=min_concept_size,
            seed=seed,
            min_metadata_coverage=clustering_params.get("min_metadata_coverage", 0.5),
        )
    
    else:
        raise ValueError(f"Unknown concept provider mode: {mode}")

