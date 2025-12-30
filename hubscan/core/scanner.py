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
from typing import Optional, Dict

import numpy as np
import faiss

from ..config import Config
from ..utils.logging import get_logger
from .io import (
    load_embeddings,
    load_faiss_index,
    build_faiss_index,
    save_faiss_index,
    load_metadata,
    Metadata,
)
from .io.vector_index import VectorIndex
from .io.adapters import FAISSIndex, create_index
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
            document_texts = None
            if self.metadata and self.metadata.has_field("text"):
                document_texts = [self.metadata.get_field("text", i) for i in range(len(self.doc_embeddings))]
            elif self.metadata:
                # Try to get texts from metadata as a list
                try:
                    document_texts = self.metadata.get("text")
                    if document_texts and len(document_texts) != len(self.doc_embeddings):
                        logger.warning(f"Document texts length ({len(document_texts)}) doesn't match embeddings ({len(self.doc_embeddings)}), skipping")
                        document_texts = None
                except:
                    document_texts = None
            self.index = FAISSIndex(faiss_index, document_texts=document_texts)
            
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
            document_texts = None
            if self.metadata and self.metadata.has_field("text"):
                # Get document texts - need to know the number of documents
                num_docs = faiss_index.ntotal
                try:
                    document_texts = [self.metadata.get_field("text", i) for i in range(num_docs)]
                except:
                    try:
                        document_texts = self.metadata.get("text")
                        if document_texts and len(document_texts) != num_docs:
                            logger.warning(f"Document texts length ({len(document_texts)}) doesn't match index ({num_docs}), skipping")
                            document_texts = None
                    except:
                        document_texts = None
            self.index = FAISSIndex(faiss_index, document_texts=document_texts)
            
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
            raise NotImplementedError("vector_db_export mode not yet implemented")
        
        logger.info(f"Loaded {len(self.doc_embeddings) if self.doc_embeddings is not None else 'unknown'} documents")
    
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
        
        # Sample queries
        logger.info("Sampling queries...")
        queries = sample_queries(
            self.doc_embeddings,
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
        
        # Map config detector names to detector classes
        detector_config_map = {
            "hubness": (HubnessDetector, {
                "validate_exact": self.config.detectors.hubness.validate_exact,
                "exact_validation_queries": self.config.detectors.hubness.exact_validation_queries,
                "use_rank_weights": self.config.detectors.hubness.use_rank_weights,
                "use_distance_weights": self.config.detectors.hubness.use_distance_weights,
                "metric": self.config.input.metric,
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
            "ranking_custom_params": self.config.scan.ranking.custom_params,
            "rerank_params": self.config.scan.ranking.rerank_params if self.config.scan.ranking.rerank else {},
        }
        
        for name, detector in detectors.items():
            logger.info(f"Running {name} detector...")
            
            # Skip cluster_spread and stability for lexical ranking (not applicable)
            if ranking_method == "lexical" and name in ["cluster_spread", "stability"]:
                logger.info(f"Skipping {name} detector for lexical ranking (not applicable)")
                continue
            
            result = detector.detect(
                self.index,
                self.doc_embeddings,
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

