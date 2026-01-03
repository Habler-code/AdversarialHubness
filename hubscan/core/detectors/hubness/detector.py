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

"""Hubness detector - orchestrates pluggable scoring components."""

import numpy as np
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from ..base import Detector, DetectorResult
from ...io.metadata import Metadata
from ...ranking import get_ranking_method
from .accumulator import BucketedHubnessAccumulator
from .scorers import get_scorer
from ....utils.logging import get_logger

if TYPE_CHECKING:
    from ...io.vector_index import VectorIndex
    from ...concepts.base import ConceptAssignment
    from ...modalities.base import ModalityAssignment

logger = get_logger()


class HubnessDetector(Detector):
    """Hubness detector using reverse k-NN frequency analysis.
    
    Detects documents that appear as nearest neighbors to an unusually
    high number of queries, which may indicate:
    - Adversarial hub injection
    - Data quality issues
    - Semantic centrality (benign hubs)
    
    Supports pluggable scoring components:
    - GlobalHubnessScorer: Global z-score computation
    - ConceptAwareScorer: Concept-specific hubness detection
    - ModalityAwareScorer: Cross-modal hub detection
    
    Custom scorers can be registered using:
        from hubscan.core.detectors.hubness.scorers import register_scorer
    """
    
    def __init__(
        self,
        enabled: bool = True,
        validate_exact: bool = False,
        exact_validation_queries: Optional[int] = None,
        use_rank_weights: bool = True,
        use_distance_weights: bool = True,
        metric: str = "cosine",
        # Concept/modality settings
        concept_aware_enabled: bool = False,
        concept_hub_z_threshold: float = 4.0,
        modality_aware_enabled: bool = False,
        cross_modal_penalty: float = 1.5,
        # Contrastive bucket detection (for concept-targeted attacks)
        use_contrastive_delta: bool = True,
        use_bucket_concentration: bool = True,
        # Legacy params (kept for config compatibility, ignored)
        **kwargs,
    ):
        super().__init__(enabled)
        self.validate_exact = validate_exact
        self.exact_validation_queries = exact_validation_queries
        self.use_rank_weights = use_rank_weights
        self.use_distance_weights = use_distance_weights
        self.metric = metric
        
        # Scorer configurations
        self._scorer_configs = {
            "global": {},
            "concept_aware": {
                "enabled": concept_aware_enabled,
                "concept_hub_z_threshold": concept_hub_z_threshold,
                "use_contrastive_delta": use_contrastive_delta,
                "use_bucket_concentration": use_bucket_concentration,
            },
            "modality_aware": {
                "enabled": modality_aware_enabled,
                "cross_modal_penalty": cross_modal_penalty,
            },
        }
        
        # Track enabled scorers
        self._enabled_scorers = ["global"]  # Global is always enabled
        if concept_aware_enabled:
            self._enabled_scorers.append("concept_aware")
        if modality_aware_enabled:
            self._enabled_scorers.append("modality_aware")
    
    @property
    def concept_aware_enabled(self) -> bool:
        """Whether concept-aware detection is enabled."""
        return self._scorer_configs.get("concept_aware", {}).get("enabled", False)
    
    @property
    def modality_aware_enabled(self) -> bool:
        """Whether modality-aware detection is enabled."""
        return self._scorer_configs.get("modality_aware", {}).get("enabled", False)
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        batch_size: int = 2048,
        ranking_method: str = "vector",
        hybrid_alpha: float = 0.5,
        query_texts: Optional[List[str]] = None,
        rerank: bool = False,
        rerank_method: Optional[str] = None,
        rerank_top_n: Optional[int] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        concept_assignment: Optional["ConceptAssignment"] = None,
        modality_assignment: Optional["ModalityAssignment"] = None,
        query_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> DetectorResult:
        """Detect hubness using reverse k-NN frequency analysis.
        
        Args:
            index: Vector index for similarity search
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            metadata: Document metadata
            batch_size: Batch size for query processing
            ranking_method: Ranking method ("vector", "hybrid", "lexical")
            hybrid_alpha: Alpha for hybrid search
            query_texts: Query texts for lexical/hybrid search
            rerank: Whether to apply reranking
            rerank_method: Reranking method name
            rerank_top_n: Number of candidates to rerank
            rerank_params: Additional reranking parameters
            concept_assignment: Concept assignments for queries/docs
            modality_assignment: Modality assignments for queries/docs
            query_metadata: Per-query metadata
            **kwargs: Additional parameters
            
        Returns:
            DetectorResult with hubness scores and metadata
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        N = len(doc_embeddings)
        M = len(queries)
        
        logger.info(f"Running hubness detection on {M} queries with k={k}")
        
        if M == 0:
            return DetectorResult(scores=np.zeros(N))
        
        # Extract ranking custom params for hybrid configuration
        ranking_custom_params = kwargs.get("ranking_custom_params", {})
        
        # Accumulate hits from query processing
        accumulator = self._accumulate_hits(
            index=index,
            queries=queries,
            k=k,
            batch_size=batch_size,
            ranking_method=ranking_method,
            hybrid_alpha=hybrid_alpha,
            query_texts=query_texts,
            concept_assignment=concept_assignment,
            modality_assignment=modality_assignment,
            num_docs=N,
            ranking_custom_params=ranking_custom_params,
        )
        
        # Run scorers and combine results
        final_scores = np.zeros(N, dtype=np.float32)
        all_metadata: Dict[str, Any] = {
            "ranking_method": ranking_method,
            "hybrid_alpha": hybrid_alpha if ranking_method == "hybrid" else None,
            "ranking_metadata": {},  # Backward compatibility
            "reranking_enabled": rerank,
            "rerank_method": rerank_method if rerank else None,
            "rerank_top_n": rerank_top_n if rerank else None,
        }
        
        for scorer_name in self._enabled_scorers:
            scorer = get_scorer(scorer_name)
            if scorer is None:
                logger.warning(f"Scorer '{scorer_name}' not found, skipping")
                continue
            
            config = self._scorer_configs.get(scorer_name, {})
            result = scorer.score(accumulator, config)
            
            final_scores += result.scores
            
            # Store metadata with appropriate key
            if scorer_name == "global":
                # Global metadata goes at top level for backward compatibility
                all_metadata.update(result.metadata)
            elif scorer_name == "concept_aware":
                all_metadata["concept_aware"] = result.metadata
            elif scorer_name == "modality_aware":
                all_metadata["modality_aware"] = result.metadata
            else:
                all_metadata[scorer_name] = result.metadata
        
        # Add disabled scorer metadata for backward compatibility
        if "concept_aware" not in all_metadata:
            all_metadata["concept_aware"] = {"enabled": False}
        if "modality_aware" not in all_metadata:
            all_metadata["modality_aware"] = {"enabled": False}
        
        return DetectorResult(scores=final_scores, metadata=all_metadata)
    
    def _accumulate_hits(
        self,
        index: "VectorIndex",
        queries: np.ndarray,
        k: int,
        batch_size: int,
        ranking_method: str,
        hybrid_alpha: float,
        query_texts: Optional[List[str]],
        concept_assignment: Optional["ConceptAssignment"],
        modality_assignment: Optional["ModalityAssignment"],
        num_docs: int,
        ranking_custom_params: Optional[Dict[str, Any]] = None,
    ) -> BucketedHubnessAccumulator:
        """Process queries in batches and accumulate hits.
        
        Args:
            index: Vector index for similarity search
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            batch_size: Batch size for query processing
            ranking_method: Ranking method name
            hybrid_alpha: Alpha for hybrid search
            query_texts: Query texts for lexical/hybrid search
            concept_assignment: Concept assignments
            modality_assignment: Modality assignments
            num_docs: Total number of documents
            ranking_custom_params: Custom parameters for ranking method (e.g., hybrid_backend)
        
        Returns:
            BucketedHubnessAccumulator with all hits recorded
        """
        ranking_custom_params = ranking_custom_params or {}
        M = len(queries)
        
        # Initialize accumulator
        accumulator = BucketedHubnessAccumulator(
            num_docs=num_docs,
            use_rank_weights=self.use_rank_weights,
            use_distance_weights=self.use_distance_weights,
        )
        
        # Build doc modality lookup
        doc_modalities: Dict[int, str] = {}
        if modality_assignment is not None:
            doc_modalities = modality_assignment.doc_modalities
        
        # Get ranking method implementation
        ranking_method_impl = get_ranking_method(ranking_method)
        if ranking_method_impl is None:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        # Process queries in batches
        for batch_start in range(0, M, batch_size):
            batch_end = min(batch_start + batch_size, M)
            batch_queries = queries[batch_start:batch_end]
            batch_query_texts = None
            if query_texts is not None:
                batch_query_texts = query_texts[batch_start:batch_end]
            
            # Execute ranking with custom params
            if ranking_method == "hybrid":
                batch_distances, batch_indices, _ = ranking_method_impl.search(
                    index=index,
                    query_vectors=batch_queries,
                    query_texts=batch_query_texts,
                    k=k,
                    alpha=hybrid_alpha,
                    **ranking_custom_params,
                )
            elif ranking_method == "lexical":
                batch_distances, batch_indices, _ = ranking_method_impl.search(
                    index=index,
                    query_vectors=batch_queries,
                    query_texts=batch_query_texts,
                    k=k,
                    **ranking_custom_params,
                )
            else:
                batch_distances, batch_indices, _ = ranking_method_impl.search(
                    index=index,
                    query_vectors=batch_queries,
                    query_texts=None,
                    k=k,
                    **ranking_custom_params,
                )
            
            # Accumulate hits
            for i, (neighbors, distances) in enumerate(zip(batch_indices, batch_distances)):
                query_idx = batch_start + i
                
                # Get concept/modality for this query
                query_concept = None
                query_modality = None
                if concept_assignment is not None:
                    query_concept = concept_assignment.get_query_concept(query_idx)
                if modality_assignment is not None:
                    query_modality = modality_assignment.get_query_modality(query_idx)
                
                accumulator.record_query(query_idx, query_concept, query_modality)
                
                for rank, (neighbor_idx, distance) in enumerate(zip(neighbors, distances)):
                    if neighbor_idx < 0 or neighbor_idx >= num_docs:
                        continue
                    
                    doc_modality = doc_modalities.get(int(neighbor_idx))
                    
                    accumulator.add_hit(
                        doc_id=int(neighbor_idx),
                        query_id=query_idx,
                        rank=rank,
                        distance=float(distance),
                        concept_id=query_concept,
                        query_modality=query_modality,
                        doc_modality=doc_modality,
                    )
        
        return accumulator

