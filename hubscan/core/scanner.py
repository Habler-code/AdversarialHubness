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
from .sampling import sample_queries
from .detectors import (
    HubnessDetector,
    ClusterSpreadDetector,
    StabilityDetector,
    DedupDetector,
)
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
        self.index: Optional[faiss.Index] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[Metadata] = None
    
    def load_data(self):
        """Load embeddings, index, and metadata according to config."""
        logger.info("Loading data...")
        
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
            self.index = build_faiss_index(
                self.doc_embeddings,
                self.config.index.type,
                self.config.input.metric,
                self.config.index.params,
            )
            
            # Save index if requested
            if self.config.index.save_path:
                logger.info(f"Saving index to {self.config.index.save_path}")
                save_faiss_index(self.index, self.config.index.save_path)
        
        elif input_mode == "faiss_index":
            # Load pre-built index
            if not self.config.input.index_path:
                raise ValueError("index_path required for faiss_index mode")
            
            logger.info(f"Loading index from {self.config.input.index_path}")
            self.index = load_faiss_index(self.config.input.index_path)
            
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
        
        elif input_mode == "vector_db_export":
            # Load from vector DB export (generic JSONL adapter)
            raise NotImplementedError("vector_db_export mode not yet implemented")
        
        # Load metadata if provided
        if self.config.input.metadata_path:
            logger.info(f"Loading metadata from {self.config.input.metadata_path}")
            self.metadata = load_metadata(self.config.input.metadata_path)
        
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
        
        # Sample queries
        logger.info("Sampling queries...")
        queries = sample_queries(
            self.doc_embeddings,
            self.config.scan,
            self.metadata,
        )
        num_queries = len(queries)
        logger.info(f"Sampled {num_queries} queries")
        
        # Initialize detectors
        detectors: Dict[str, any] = {}
        
        if self.config.detectors.hubness.enabled:
            detectors["hubness"] = HubnessDetector(
                enabled=True,
                validate_exact=self.config.detectors.hubness.validate_exact,
                exact_validation_queries=self.config.detectors.hubness.exact_validation_queries,
            )
        
        if self.config.detectors.cluster_spread.enabled:
            detectors["cluster_spread"] = ClusterSpreadDetector(
                enabled=True,
                num_clusters=self.config.detectors.cluster_spread.num_clusters,
                batch_size=self.config.detectors.cluster_spread.batch_size,
            )
        
        if self.config.detectors.stability.enabled:
            detectors["stability"] = StabilityDetector(
                enabled=True,
                candidates_top_x=self.config.detectors.stability.candidates_top_x,
                perturbations=self.config.detectors.stability.perturbations,
                sigma=self.config.detectors.stability.sigma,
                normalize=self.config.detectors.stability.normalize,
            )
        
        if self.config.detectors.dedup.enabled:
            detectors["dedup"] = DedupDetector(
                enabled=True,
                text_hash_field=self.config.detectors.dedup.text_hash_field,
                duplicate_threshold=self.config.detectors.dedup.duplicate_threshold,
                suppress_boilerplate=self.config.detectors.dedup.suppress_boilerplate,
            )
        
        # Run detectors
        detector_results: Dict[str, any] = {}
        
        for name, detector in detectors.items():
            logger.info(f"Running {name} detector...")
            result = detector.detect(
                self.index,
                self.doc_embeddings,
                queries,
                self.config.scan.k,
                self.metadata,
                batch_size=self.config.scan.batch_size,
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
        )
        
        runtime = time.time() - start_time
        logger.info(f"Scan complete in {runtime:.2f} seconds")
        
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
        }

