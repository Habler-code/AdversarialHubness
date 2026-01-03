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

"""Main scanner orchestrator."""

import time
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import faiss

from ..config import Config
from ..utils.logging import get_logger
from .io import (
    load_embeddings,
    save_embeddings,
    load_faiss_index,
    build_faiss_index,
    save_faiss_index,
    load_metadata,
    Metadata,
)
from .io.vector_index import VectorIndex
from .io.adapters import FAISSIndex, create_index
from .io.adapters.multi_index_adapter import MultiIndexAdapter
from .fusion import parallel_retrieve, fuse_results, enforce_diversity
from .sampling import sample_queries
from .detectors import (
    HubnessDetector,
    ClusterSpreadDetector,
    StabilityDetector,
    DedupDetector,
    get_detector_class,
)
from .ranking import get_ranking_method
from .reranking import get_reranking_method
from .concepts.providers import create_concept_provider
from .modalities.resolvers import create_modality_resolver
from .scoring import combine_scores, apply_thresholds
from .report import (
    generate_json_report,
    generate_html_report,
    save_json_report,
    save_html_report,
)

logger = get_logger()


class Scanner:
    """Main scanner for detecting adversarial hubs."""
    
    def __init__(self, config: Config):
        """Initialize scanner with configuration."""
        self.config = config
        self.index: Optional[VectorIndex] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[Metadata] = None
        
        # Multi-index support (for gold standard architecture)
        self.text_index: Optional[VectorIndex] = None
        self.image_index: Optional[VectorIndex] = None
        self.unified_index: Optional[VectorIndex] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.image_embeddings: Optional[np.ndarray] = None
        self.unified_embeddings: Optional[np.ndarray] = None
    
    def _get_hybrid_config(self) -> Dict[str, Any]:
        """Get hybrid search configuration as a dict for adapters."""
        hybrid_cfg = self.config.scan.ranking.hybrid
        return {
            "backend": hybrid_cfg.backend,
            "lexical_backend": hybrid_cfg.lexical_backend,
            "normalize_scores": hybrid_cfg.normalize_scores,
            "text_field": hybrid_cfg.text_field,
            # Qdrant
            "qdrant_dense_vector_name": hybrid_cfg.qdrant_dense_vector_name,
            "qdrant_sparse_vector_name": hybrid_cfg.qdrant_sparse_vector_name,
            # Pinecone
            "pinecone_has_sparse": hybrid_cfg.pinecone_has_sparse,
            # Weaviate
            "weaviate_bm25_properties": hybrid_cfg.weaviate_bm25_properties,
        }
    
    def _get_document_texts(self, num_docs: int) -> Optional[List[str]]:
        """Extract document texts from metadata for hybrid search.
        
        Args:
            num_docs: Expected number of documents
            
        Returns:
            List of document texts or None if not available
        """
        if not self.metadata:
            return None
        
        text_field = self.config.scan.ranking.hybrid.text_field
        
        # Try to get texts from metadata using the configured field
        if self.metadata.has_field(text_field):
            try:
                document_texts = [
                    self.metadata.get_field(text_field, i) for i in range(num_docs)
                ]
                logger.info(f"Loaded {len(document_texts)} document texts from '{text_field}' field")
                return document_texts
            except Exception as e:
                logger.warning(f"Failed to get document texts from '{text_field}': {e}")
        
        # Fall back to trying to get as a list
        try:
            document_texts = self.metadata.get(text_field)
            if document_texts:
                if len(document_texts) != num_docs:
                    logger.warning(
                        f"Document texts length ({len(document_texts)}) doesn't match "
                        f"expected ({num_docs}), skipping"
                    )
                    return None
                return document_texts
        except Exception:
            pass
        
        return None
    
    def _validate_hybrid_requirements(self, document_texts: Optional[List[str]]):
        """Validate that hybrid search requirements are met.
        
        Raises ValueError with clear message if requirements not met.
        """
        ranking_method = self.config.scan.ranking.method
        if ranking_method not in ("hybrid", "lexical"):
            return
        
        hybrid_cfg = self.config.scan.ranking.hybrid
        
        # Client-side fusion requires document texts
        if hybrid_cfg.backend in ("client_fusion", "auto"):
            if document_texts is None:
                text_field = hybrid_cfg.text_field
                raise ValueError(
                    f"Hybrid search with backend='{hybrid_cfg.backend}' requires document texts "
                    f"in metadata field '{text_field}'. Either:\n"
                    f"  1. Add '{text_field}' field to your metadata file, or\n"
                    f"  2. Configure a different text field via scan.ranking.hybrid.text_field, or\n"
                    f"  3. Use ranking.method='vector' instead of hybrid search."
                )
    
    def load_data(self):
        """Load embeddings, index, and metadata according to config."""
        logger.info("Loading data...")
        
        # Load metadata first (needed for document texts in lexical search)
        if self.config.input.metadata_path:
            logger.info(f"Loading metadata from {self.config.input.metadata_path}")
            self.metadata = load_metadata(self.config.input.metadata_path)
        
        input_mode = self.config.input.mode
        
        if input_mode == "embeddings_only":
            # Load embeddings and build index
            if not self.config.input.embeddings_path:
                raise ValueError("embeddings_path required for embeddings_only mode")
            
            logger.info(f"Loading embeddings from {self.config.input.embeddings_path}")
            self.doc_embeddings = load_embeddings(
                self.config.input.embeddings_path,
                normalize=(self.config.input.metric == "cosine")
            )
            
            logger.info(f"Building {self.config.index.type} index...")
            faiss_index = build_faiss_index(
                self.doc_embeddings,
                self.config.index.type,
                self.config.input.metric,
                self.config.index.params,
            )
            
            # Wrap in adapter
            # Include document texts if available for lexical search
            document_texts = self._get_document_texts(len(self.doc_embeddings))
            
            # Get hybrid config from ranking config
            hybrid_config = self._get_hybrid_config()
            
            self.index = FAISSIndex(
                faiss_index, 
                document_texts=document_texts,
                hybrid_config=hybrid_config,
            )
            
            # Save index if requested
            if self.config.index.save_path:
                logger.info(f"Saving index to {self.config.index.save_path}")
                save_faiss_index(faiss_index, self.config.index.save_path)
        
        elif input_mode == "faiss_index":
            # Load pre-built index
            if not self.config.input.index_path:
                raise ValueError("index_path required for faiss_index mode")
            
            logger.info(f"Loading index from {self.config.input.index_path}")
            faiss_index = load_faiss_index(self.config.input.index_path)
            
            # Wrap in adapter
            # Include document texts if available for lexical search
            num_docs = faiss_index.ntotal
            document_texts = self._get_document_texts(num_docs)
            
            # Get hybrid config from ranking config
            hybrid_config = self._get_hybrid_config()
            
            self.index = FAISSIndex(
                faiss_index, 
                document_texts=document_texts,
                hybrid_config=hybrid_config,
            )
            
            # Optionally load embeddings if needed for clustering/validation
            if self.config.input.embeddings_path:
                logger.info(f"Loading embeddings from {self.config.input.embeddings_path}")
                self.doc_embeddings = load_embeddings(
                    self.config.input.embeddings_path,
                    normalize=(self.config.input.metric == "cosine")
                )
            else:
                # Infer dimension from index
                self.doc_embeddings = None  # Will need to be provided or inferred
        
        elif input_mode in ["pinecone", "qdrant", "weaviate"]:
            # Load from external vector database
            logger.info(f"Connecting to {input_mode} backend...")
            self.index = create_index(self.config.input)
            
            # Optionally load embeddings if provided
            if self.config.input.embeddings_path:
                logger.info(f"Loading embeddings from {self.config.input.embeddings_path}")
                self.doc_embeddings = load_embeddings(
                    self.config.input.embeddings_path,
                    normalize=(self.config.input.metric == "cosine")
                )
            else:
                # For external DBs, embeddings may not be available
                # Detectors that need embeddings will need to handle this
                self.doc_embeddings = None
        
        elif input_mode == "vector_db_export":
            # Load from vector DB export (generic JSONL adapter)
            from .io.adapters.jsonl_adapter import load_jsonl_export
            
            if not self.config.input.embeddings_path:
                raise ValueError(
                    "embeddings_path is required for vector_db_export mode. "
                    "This should point to a JSONL file containing embeddings."
                )
            
            export_path = self.config.input.embeddings_path
            logger.info(f"Loading vector DB export from {export_path}")
            
            # Load embeddings and metadata from JSONL
            self.doc_embeddings, export_metadata = load_jsonl_export(
                export_path,
                embedding_field=getattr(self.config.input, 'embedding_field', 'embedding'),
                normalize=(self.config.input.metric == "cosine"),
            )
            
            # Convert export metadata to Metadata object if not already loaded
            if self.metadata is None and export_metadata:
                # Convert list of dicts to columnar format for Metadata
                columnar_data = {}
                for key in export_metadata[0].keys():
                    columnar_data[key] = [record.get(key) for record in export_metadata]
                self.metadata = Metadata(columnar_data)
            
            logger.info(f"Building {self.config.index.type} index from export...")
            
            # Build FAISS index from exported embeddings
            faiss_index = build_faiss_index(
                self.doc_embeddings,
                self.config.index.type,
                self.config.input.metric,
                self.config.index.params,
            )
            
            # Get document texts if available for hybrid search
            document_texts = self._get_document_texts(len(self.doc_embeddings))
            hybrid_config = self._get_hybrid_config()
            
            self.index = FAISSIndex(
                faiss_index,
                document_texts=document_texts,
                hybrid_config=hybrid_config,
            )
            
            # Save index if requested
            if self.config.index.save_path:
                logger.info(f"Saving index to {self.config.index.save_path}")
                save_faiss_index(faiss_index, self.config.index.save_path)
        
        elif input_mode == "multi_index":
            # Gold standard: parallel retrieval from separate indexes
            if not self.config.input.multi_index:
                raise ValueError("multi_index config required for multi_index mode")
            
            multi_config = self.config.input.multi_index
            logger.info("Loading multi-index setup (gold standard architecture)...")
            
            # Load text index
            if multi_config.text_index_path:
                logger.info(f"Loading text index from {multi_config.text_index_path}")
                text_faiss = load_faiss_index(multi_config.text_index_path)
                document_texts = None
                if self.metadata and self.metadata.has_field("text"):
                    num_text_docs = text_faiss.ntotal
                    try:
                        document_texts = [self.metadata.get_field("text", i) for i in range(num_text_docs)]
                    except:
                        document_texts = None
                self.text_index = FAISSIndex(text_faiss, document_texts=document_texts)
            
            if multi_config.text_embeddings_path:
                logger.info(f"Loading text embeddings from {multi_config.text_embeddings_path}")
                self.text_embeddings = load_embeddings(
                    multi_config.text_embeddings_path,
                    normalize=(multi_config.text_metric == "cosine")
                )
            
            # Load image index
            if multi_config.image_index_path:
                logger.info(f"Loading image index from {multi_config.image_index_path}")
                image_faiss = load_faiss_index(multi_config.image_index_path)
                self.image_index = FAISSIndex(image_faiss, document_texts=None)
            
            if multi_config.image_embeddings_path:
                logger.info(f"Loading image embeddings from {multi_config.image_embeddings_path}")
                self.image_embeddings = load_embeddings(
                    multi_config.image_embeddings_path,
                    normalize=(multi_config.image_metric == "cosine")
                )
            
            # Load unified/cross-modal index (optional recall backstop)
            if multi_config.unified_index_path:
                logger.info(f"Loading unified index from {multi_config.unified_index_path}")
                unified_faiss = load_faiss_index(multi_config.unified_index_path)
                self.unified_index = FAISSIndex(unified_faiss, document_texts=None)
            
            if multi_config.unified_embeddings_path:
                logger.info(f"Loading unified embeddings from {multi_config.unified_embeddings_path}")
                self.unified_embeddings = load_embeddings(
                    multi_config.unified_embeddings_path,
                    normalize=(multi_config.unified_metric == "cosine")
                )
            
            # Combine embeddings for detectors (use text as primary, or create combined)
            # For hubness detection, we need a unified doc_embeddings
            # Use text embeddings as primary, or combine if needed
            if self.text_embeddings is not None:
                self.doc_embeddings = self.text_embeddings
                # If we have image embeddings, we could concatenate, but for now use text as primary
                # The actual retrieval will use separate indexes
            elif self.image_embeddings is not None:
                self.doc_embeddings = self.image_embeddings
            else:
                raise ValueError("At least one embeddings file required in multi_index mode")
            
            # Create multi-index adapter if late fusion is enabled
            if self.config.input.late_fusion and self.config.input.late_fusion.enabled:
                logger.info("Creating multi-index adapter with late fusion...")
                self.index = MultiIndexAdapter(
                    text_index=self.text_index,
                    image_index=self.image_index,
                    unified_index=self.unified_index,
                    config=self.config,
                    doc_embeddings=self.doc_embeddings,
                )
            else:
                # Set primary index to text index if available (for backward compatibility)
                self.index = self.text_index or self.image_index or self.unified_index
            
            logger.info(f"Multi-index setup complete: text={self.text_index is not None}, "
                       f"image={self.image_index is not None}, unified={self.unified_index is not None}, "
                       f"fusion={self.config.input.late_fusion and self.config.input.late_fusion.enabled}")
        
        logger.info(f"Loaded {len(self.doc_embeddings) if self.doc_embeddings is not None else 'unknown'} documents")
        
        # Validate hybrid search requirements
        document_texts = getattr(self.index, '_document_texts', None) if self.index else None
        self._validate_hybrid_requirements(document_texts)
    
    def scan(self) -> Dict:
        """
        Run the scan.
        
        Returns:
            Dictionary with scan results
        """
        start_time = time.time()
        
        if self.index is None or self.doc_embeddings is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        num_docs = len(self.doc_embeddings)
        
        # Check for empty document set
        if num_docs == 0:
            raise ValueError("Cannot scan empty document set")
        
        # Use doc embeddings directly
        doc_embeddings_processed = self.doc_embeddings
        
        # Sample queries (use processed embeddings)
        logger.info("Sampling queries...")
        queries = sample_queries(
            doc_embeddings_processed,
            self.config.scan,
            self.metadata,
        )
        num_queries = len(queries)
        logger.info(f"Sampled {num_queries} queries")
        
        # Validate ranking method
        ranking_method = self.config.scan.ranking.method
        ranking_method_impl = get_ranking_method(ranking_method)
        if ranking_method_impl is None:
            from .ranking import list_ranking_methods
            available = list_ranking_methods()
            raise ValueError(
                f"Unknown ranking method: {ranking_method}. "
                f"Available methods: {', '.join(available)}. "
                f"Register custom methods using hubscan.core.ranking.register_ranking_method()"
            )
        
        # Validate reranking method if reranking is enabled
        reranking_method_impl = None
        if self.config.scan.ranking.rerank:
            reranking_method_impl = get_reranking_method(self.config.scan.ranking.rerank_method)
            if reranking_method_impl is None:
                from .reranking import list_reranking_methods
                available = list_reranking_methods()
                raise ValueError(
                    f"Unknown reranking method: {self.config.scan.ranking.rerank_method}. "
                    f"Available methods: {', '.join(available)}. "
                    f"Register custom methods using hubscan.core.reranking.register_reranking_method()"
                )
        
        # Load query texts if needed for lexical/hybrid search
        query_texts = None
        if ranking_method in ["hybrid", "lexical"]:
            if self.config.scan.query_texts_path:
                logger.info(f"Loading query texts from {self.config.scan.query_texts_path}")
                import json
                with open(self.config.scan.query_texts_path, "r") as f:
                    query_texts = json.load(f)
                if not isinstance(query_texts, list):
                    raise ValueError("query_texts_path must contain a JSON list of strings")
                # Ensure we have enough query texts
                if len(query_texts) < num_queries:
                    logger.warning(
                        f"Only {len(query_texts)} query texts provided, "
                        f"but {num_queries} queries sampled. Repeating query texts."
                    )
                    query_texts = (query_texts * ((num_queries // len(query_texts)) + 1))[:num_queries]
                elif len(query_texts) > num_queries:
                    query_texts = query_texts[:num_queries]
            elif ranking_method == "lexical":
                raise ValueError(
                    "query_texts_path required for lexical search. "
                    "Provide query texts file or use hybrid/vector ranking."
                )
            elif ranking_method == "hybrid":
                logger.warning(
                    "query_texts_path not provided for hybrid search. "
                    "Falling back to vector search only."
                )
                ranking_method = "vector"
        
        # Initialize detectors dynamically using registry
        detectors: Dict[str, any] = {}
        
        # Create concept and modality assignments if enabled
        concept_assignment = None
        modality_assignment = None
        query_metadata = None
        doc_metadata_list = None  # Shared between concept and modality providers
        
        # Check if concept-aware detection is enabled
        concept_aware_enabled = (
            hasattr(self.config.detectors, 'concept_aware') and 
            self.config.detectors.concept_aware.enabled
        )
        modality_aware_enabled = (
            hasattr(self.config.detectors, 'modality_aware') and 
            self.config.detectors.modality_aware.enabled
        )
        
        if concept_aware_enabled:
            logger.info("Creating concept assignments...")
            concept_config = self.config.detectors.concept_aware
            concept_provider = create_concept_provider(
                mode=concept_config.mode,
                metadata_field=concept_config.metadata_field,
                num_concepts=concept_config.num_concepts,
                clustering_params=concept_config.clustering_params,
            )
            
            # Prepare metadata for concept provider (as List[Dict])
            doc_metadata_list = None
            if self.metadata:
                # Build doc_metadata as a list of dicts
                doc_metadata_list = []
                for i in range(num_docs):
                    record = {}
                    for field in self.metadata.data.keys():
                        value = self.metadata.get_field(field, i)
                        if value is not None:
                            record[field] = value
                    doc_metadata_list.append(record)
            
            # Use query embeddings for concept assignment
            concept_assignment = concept_provider.assign_concepts(
                doc_embeddings=self.doc_embeddings,
                query_embeddings=queries,
                doc_metadata=doc_metadata_list,
                query_metadata=None,  # Query metadata from config if available
            )
            logger.info(f"Assigned {concept_assignment.num_concepts} concepts")
        
        if modality_aware_enabled:
            logger.info("Creating modality assignments...")
            modality_config = self.config.detectors.modality_aware
            modality_resolver = create_modality_resolver(
                mode=modality_config.mode,
                query_modality_field=modality_config.query_modality_field,
                doc_modality_field=modality_config.doc_modality_field,
            )
            
            # Reuse doc_metadata_list from concept provider if available
            if doc_metadata_list is None and self.metadata:
                # Build doc_metadata as a list of dicts
                doc_metadata_list = []
                for i in range(num_docs):
                    record = {}
                    for field in self.metadata.data.keys():
                        value = self.metadata.get_field(field, i)
                        if value is not None:
                            record[field] = value
                    doc_metadata_list.append(record)
            
            modality_assignment = modality_resolver.resolve_modalities(
                query_metadata=None,  # Query metadata from config if available
                doc_metadata=doc_metadata_list,
                num_queries=num_queries,
                num_docs=num_docs,
            )
            logger.info(f"Found modalities: {modality_assignment.all_modalities}")
        
        # Get concept hub z threshold if concept aware
        concept_hub_z_threshold = 4.0
        if concept_aware_enabled:
            concept_hub_z_threshold = getattr(self.config.detectors.concept_aware, 'concept_hub_z_threshold', 4.0)
        
        # Get cross-modal penalty if modality aware
        cross_modal_penalty = 1.5
        if modality_aware_enabled:
            cross_modal_penalty = getattr(self.config.detectors.modality_aware, 'cross_modal_penalty', 1.5)
        
        # Map config detector names to detector classes
        detector_config_map = {
            "hubness": (HubnessDetector, {
                "validate_exact": self.config.detectors.hubness.validate_exact,
                "exact_validation_queries": self.config.detectors.hubness.exact_validation_queries,
                "use_rank_weights": self.config.detectors.hubness.use_rank_weights,
                "use_distance_weights": self.config.detectors.hubness.use_distance_weights,
                "metric": self.config.input.metric,
                "concept_aware_enabled": concept_aware_enabled,
                "concept_hub_z_threshold": concept_hub_z_threshold,
                "modality_aware_enabled": modality_aware_enabled,
                "cross_modal_penalty": cross_modal_penalty,
                # Contrastive bucket detection for concept-targeted attacks
                "use_contrastive_delta": getattr(self.config.detectors.hubness, 'use_contrastive_delta', True),
                "use_bucket_concentration": getattr(self.config.detectors.hubness, 'use_bucket_concentration', True),
            }),
            "cluster_spread": (ClusterSpreadDetector, {
                "num_clusters": self.config.detectors.cluster_spread.num_clusters,
                "batch_size": self.config.detectors.cluster_spread.batch_size,
            }),
            "stability": (StabilityDetector, {
                "candidates_top_x": self.config.detectors.stability.candidates_top_x,
                "perturbations": self.config.detectors.stability.perturbations,
                "sigma": self.config.detectors.stability.sigma,
                "normalize": self.config.detectors.stability.normalize,
            }),
            "dedup": (DedupDetector, {
                "text_hash_field": self.config.detectors.dedup.text_hash_field,
                "duplicate_threshold": self.config.detectors.dedup.duplicate_threshold,
                "suppress_boilerplate": self.config.detectors.dedup.suppress_boilerplate,
            }),
        }
        
        # Initialize enabled detectors
        for detector_name, (detector_class, detector_kwargs) in detector_config_map.items():
            detector_config = getattr(self.config.detectors, detector_name, None)
            if detector_config and detector_config.enabled:
                # Try to get from registry first (for custom detectors), fallback to built-in
                registered_class = get_detector_class(detector_name)
                if registered_class is not None:
                    detector_class = registered_class
                
                detectors[detector_name] = detector_class(
                    enabled=True,
                    **detector_kwargs
                )
        
        # Run detectors
        detector_results: Dict[str, any] = {}
        
        # Build ranking custom params including hybrid config
        ranking_custom_params = dict(self.config.scan.ranking.custom_params)
        if ranking_method == "hybrid":
            # Include hybrid-specific config in custom params
            hybrid_cfg = self.config.scan.ranking.hybrid
            ranking_custom_params.update({
                "hybrid_backend": hybrid_cfg.backend,
                "lexical_backend": hybrid_cfg.lexical_backend,
                "normalize_scores": hybrid_cfg.normalize_scores,
            })
        
        # Common detection kwargs for all detectors
        common_detect_kwargs = {
            "batch_size": self.config.scan.batch_size,
            "seed": self.config.scan.seed,
            "ranking_method": ranking_method,
            "hybrid_alpha": self.config.scan.ranking.hybrid_alpha,
            "query_texts": query_texts,
            "rerank": self.config.scan.ranking.rerank,
            "rerank_method": self.config.scan.ranking.rerank_method if self.config.scan.ranking.rerank else None,
            "rerank_top_n": self.config.scan.ranking.rerank_top_n if self.config.scan.ranking.rerank else None,
            "ranking_custom_params": ranking_custom_params,
            "rerank_params": self.config.scan.ranking.rerank_params if self.config.scan.ranking.rerank else {},
            # Concept/modality assignments for hubness detector
            "concept_assignment": concept_assignment,
            "modality_assignment": modality_assignment,
            "query_metadata": query_metadata,
        }
        
        for name, detector in detectors.items():
            logger.info(f"Running {name} detector...")
            
            # Skip cluster_spread and stability for lexical ranking (not applicable)
            if ranking_method == "lexical" and name in ["cluster_spread", "stability"]:
                logger.info(f"Skipping {name} detector for lexical ranking (not applicable)")
                continue
            
            result = detector.detect(
                self.index,
                doc_embeddings_processed,  # Use processed embeddings if hubness reduction applied
                queries,
                self.config.scan.k,
                self.metadata,
                **common_detect_kwargs,
            )
            detector_results[name] = result
        
        # Combine scores
        logger.info("Combining scores...")
        combined_scores = combine_scores(
            detector_results,
            self.config.scoring.weights,
        )
        
        # Apply thresholds
        logger.info("Applying thresholds...")
        hub_z_scores = detector_results.get("hubness").scores if "hubness" in detector_results else None
        verdicts = apply_thresholds(
            detector_results,
            combined_scores,
            self.config.thresholds,
            hub_z_scores=hub_z_scores,
            ranking_method=ranking_method,
        )
        
        runtime = time.time() - start_time
        logger.info(f"Scan complete in {runtime:.2f} seconds")
        
        # Compute metrics if ground truth available
        detection_metrics = None
        
        # Check for ground truth in metadata
        ground_truth_labels = None
        if self.metadata and self.metadata.has_field("is_adversarial"):
            ground_truth_labels = np.array([
                1 if self.metadata.get_field("is_adversarial", i) else 0
                for i in range(num_docs)
            ])
        
        # Compute detection metrics if ground truth available
        if ground_truth_labels is not None:
            logger.info("Computing detection metrics...")
            from .metrics.detection_metrics import compute_detection_metrics
            
            # Use combined scores as prediction scores
            detection_metrics = compute_detection_metrics(
                ground_truth_labels,
                combined_scores,
                threshold=np.median(combined_scores),  # Use median as threshold
            )
        
        # Generate reports
        logger.info("Generating reports...")
        json_report = generate_json_report(
            self.config,
            detector_results,
            combined_scores,
            verdicts,
            self.metadata,
            num_queries,
            runtime,
            num_docs,
            detection_metrics=detection_metrics,
        )
        
        html_report = generate_html_report(
            self.config,
            detector_results,
            combined_scores,
            verdicts,
            self.metadata,
            num_queries,
            runtime,
            num_docs,
            detection_metrics=detection_metrics,
        )
        
        # Save reports
        from pathlib import Path
        Path(self.config.output.out_dir).mkdir(parents=True, exist_ok=True)
        
        json_path = str(Path(self.config.output.out_dir) / "report.json")
        html_path = str(Path(self.config.output.out_dir) / "report.html")
        
        save_json_report(json_report, json_path)
        save_html_report(html_report, html_path)
        
        logger.info(f"Reports saved to {self.config.output.out_dir}")
        
        return {
            "json_report": json_report,
            "html_report": html_report,
            "detector_results": detector_results,
            "combined_scores": combined_scores,
            "verdicts": verdicts,
            "runtime": runtime,
            "detection_metrics": detection_metrics,
        }
    
    def extract_embeddings(
        self,
        output_path: Optional[str] = None,
        batch_size: int = 1000,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Extract embeddings from the loaded vector index.
        
        Args:
            output_path: Optional path to save embeddings (.npy file)
            batch_size: Number of vectors to retrieve per batch
            limit: Optional maximum number of vectors to extract
            
        Returns:
            Tuple of (embeddings, ids) where:
            - embeddings: Array of shape (N, D) containing all vectors
            - ids: List of document IDs corresponding to each embedding
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_data() first.")
        
        logger.info("Extracting embeddings from vector index...")
        embeddings, ids = self.index.extract_embeddings(batch_size=batch_size, limit=limit)
        
        if output_path:
            save_embeddings(embeddings, output_path)
            logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
        
        return embeddings, ids
    
