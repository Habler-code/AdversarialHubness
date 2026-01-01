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

"""SDK for HubScan - Easy-to-use programmatic interface."""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from .config import Config
from .core.scanner import Scanner
from .core.io import load_embeddings, load_faiss_index, build_faiss_index, save_faiss_index
from .core.scoring import Verdict


def scan(
    embeddings_path: Optional[str] = None,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    config_path: Optional[str] = None,
    output_dir: str = "reports/",
    k: int = 20,
    num_queries: int = 10000,
    concept_aware: bool = False,
    modality_aware: bool = False,
    concept_field: str = "concept",
    modality_field: str = "modality",
    num_concepts: int = 10,
    # Multi-index / late fusion options
    text_index_path: Optional[str] = None,
    text_embeddings_path: Optional[str] = None,
    image_index_path: Optional[str] = None,
    image_embeddings_path: Optional[str] = None,
    late_fusion: bool = False,
    fusion_method: str = "rrf",
    text_weight: float = 0.4,
    image_weight: float = 0.4,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a hubness scan with simple parameters.
    
    This is the main SDK function for running scans programmatically.
    
    Args:
        embeddings_path: Path to embeddings file (.npy/.npz)
        index_path: Path to FAISS index file (.index)
        metadata_path: Optional path to metadata file
        config_path: Optional path to YAML config file (overrides other params)
        output_dir: Directory to save reports
        k: Number of nearest neighbors to retrieve
        num_queries: Number of queries to sample
        concept_aware: Enable concept-aware hub detection
        modality_aware: Enable modality-aware hub detection
        concept_field: Metadata field name for concept labels
        modality_field: Metadata field name for modality labels
        num_concepts: Number of concept clusters for auto-detection
        text_index_path: Path to text index file (for multi-index mode)
        text_embeddings_path: Path to text embeddings file (for multi-index mode)
        image_index_path: Path to image index file (for multi-index mode)
        image_embeddings_path: Path to image embeddings file (for multi-index mode)
        late_fusion: Enable late fusion of multi-index results
        fusion_method: Late fusion method ("rrf", "weighted_sum", "max")
        text_weight: Weight for text index in fusion (0.0-1.0)
        image_weight: Weight for image index in fusion (0.0-1.0)
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing scan results:
        - json_report: Full JSON report
        - html_report: HTML report string
        - detector_results: Raw detector results
        - combined_scores: Combined risk scores array
        - verdicts: Verdict dictionary mapping doc indices to Verdict enum
        - runtime: Runtime in seconds
        - detection_metrics: Optional detection performance metrics
        
    Example:
        ```python
        from hubscan.sdk import scan
        
        # Basic scan
        results = scan(
            embeddings_path="data/embeddings.npy",
            metadata_path="data/metadata.json",
            k=20,
            num_queries=5000
        )
        
        # Concept and modality-aware scan
        results = scan(
            embeddings_path="data/embeddings.npy",
            metadata_path="data/metadata.json",
            concept_aware=True,
            modality_aware=True,
            concept_field="category",
            modality_field="type"
        )
        
        # Multi-index scan with late fusion (gold standard architecture)
        results = scan(
            text_index_path="data/text_index.index",
            text_embeddings_path="data/text_embeddings.npy",
            image_index_path="data/image_index.index",
            image_embeddings_path="data/image_embeddings.npy",
            metadata_path="data/metadata.json",
            late_fusion=True,
            fusion_method="rrf",
            modality_aware=True
        )
        
        # Access results
        high_risk_docs = [
            idx for idx, verdict in results["verdicts"].items()
            if verdict == Verdict.HIGH
        ]
        ```
    """
    # Load config if provided, otherwise create from parameters
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = _create_config_from_params(
            embeddings_path=embeddings_path,
            index_path=index_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            k=k,
            num_queries=num_queries,
            **kwargs,
        )
    
    # Apply concept-aware and modality-aware settings
    if concept_aware:
        config.detectors.concept_aware.enabled = True
        config.detectors.concept_aware.metadata_field = concept_field
        config.detectors.concept_aware.num_concepts = num_concepts
    if modality_aware:
        config.detectors.modality_aware.enabled = True
        config.detectors.modality_aware.doc_modality_field = modality_field
        config.detectors.modality_aware.query_modality_field = modality_field
    
    # Apply multi-index configuration
    if text_index_path or image_index_path:
        from .config.config import MultiIndexConfig, LateFusionConfig
        
        config.input.mode = "multi_index"
        config.input.multi_index = MultiIndexConfig(
            text_index_path=text_index_path,
            text_embeddings_path=text_embeddings_path,
            image_index_path=image_index_path,
            image_embeddings_path=image_embeddings_path,
        )
        config.scan.ranking.parallel_retrieval = True
        
        if late_fusion:
            config.input.late_fusion = LateFusionConfig(
                enabled=True,
                fusion_method=fusion_method,
                text_weight=text_weight,
                image_weight=image_weight,
            )
    
    # Create and run scanner
    scanner = Scanner(config)
    scanner.load_data()
    results = scanner.scan()
    
    return results


def quick_scan(
    embeddings: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    k: int = 20,
    num_queries: Optional[int] = None,
    index_type: str = "flat",
) -> Dict[str, Any]:
    """
    Run a quick scan on in-memory embeddings.
    
    Args:
        embeddings: Embeddings array (N, D)
        metadata: Optional metadata dictionary
        k: Number of nearest neighbors
        num_queries: Number of queries (defaults to min(1000, N))
        index_type: Index type ("flat", "hnsw", "ivf_pq")
        
    Returns:
        Scan results dictionary
        
    Example:
        ```python
        import numpy as np
        from hubscan.sdk import quick_scan
        
        embeddings = np.random.randn(1000, 128).astype(np.float32)
        results = quick_scan(embeddings, k=10)
        ```
    """
    import tempfile
    import os
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings_path = os.path.join(tmpdir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        
        config = Config()
        config.input.mode = "embeddings_only"
        config.input.embeddings_path = embeddings_path
        config.input.metric = "cosine"
        config.index.type = index_type
        config.scan.k = k
        config.scan.num_queries = num_queries or min(1000, len(embeddings))
        config.scan.query_sampling = "random_docs_as_queries"
        config.output.out_dir = tmpdir
        
        scanner = Scanner(config)
        scanner.load_data()
        results = scanner.scan()
        
        return results


def scan_from_config(config_path: str) -> Dict[str, Any]:
    """
    Run a scan from a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Scan results dictionary
        
    Example:
        ```python
        from hubscan.sdk import scan_from_config
        
        results = scan_from_config("config.yaml")
        ```
    """
    config = Config.from_yaml(config_path)
    scanner = Scanner(config)
    scanner.load_data()
    return scanner.scan()


def get_suspicious_documents(
    results: Dict[str, Any],
    verdict: Optional[Verdict] = Verdict.HIGH,
    top_k: Optional[int] = None,
) -> list:
    """
    Extract suspicious documents from scan results.
    
    Args:
        results: Results dictionary from scan()
        verdict: Verdict level to filter (default: HIGH)
        top_k: Optional limit on number of documents
        
    Returns:
        List of suspicious document dictionaries
        
    Example:
        ```python
        from hubscan.sdk import scan, get_suspicious_documents, Verdict
        
        results = scan(embeddings_path="data/embeddings.npy")
        suspicious = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
        
        for doc in suspicious:
            print(f"Doc {doc['doc_index']}: Risk={doc['risk_score']:.4f}")
        ```
    """
    json_report = results["json_report"]
    suspicious = json_report["suspicious_documents"]
    
    if verdict:
        suspicious = [doc for doc in suspicious if doc["verdict"] == verdict.value]
    
    if top_k:
        suspicious = suspicious[:top_k]
    
    return suspicious


def scan_with_ranking(
    embeddings_path: Optional[str] = None,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    query_texts_path: Optional[str] = None,
    ranking_method: str = "vector",
    hybrid_alpha: float = 0.5,
    rerank: bool = False,
    rerank_method: str = "default",
    rerank_top_n: int = 100,
    rerank_params: Optional[Dict[str, Any]] = None,
    concept_aware: bool = False,
    modality_aware: bool = False,
    concept_field: str = "concept",
    modality_field: str = "modality",
    num_concepts: int = 10,
    # Multi-index / late fusion options
    text_index_path: Optional[str] = None,
    text_embeddings_path: Optional[str] = None,
    image_index_path: Optional[str] = None,
    image_embeddings_path: Optional[str] = None,
    late_fusion: bool = False,
    fusion_method: str = "rrf",
    text_weight: float = 0.4,
    image_weight: float = 0.4,
    output_dir: str = "reports/",
    k: int = 20,
    num_queries: int = 10000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a scan with specified ranking method.
    
    Args:
        embeddings_path: Path to embeddings file (.npy/.npz)
        index_path: Path to FAISS index file (.index)
        metadata_path: Optional path to metadata file
        query_texts_path: Optional path to query texts file (for lexical/hybrid search)
        ranking_method: Ranking method ("vector", "hybrid", "lexical")
        hybrid_alpha: Weight for vector search in hybrid mode (0.0-1.0)
        rerank: Whether to enable reranking as post-processing
        rerank_method: Reranking method name (default: "default")
        rerank_top_n: Number of candidates to retrieve before reranking
        rerank_params: Custom parameters for reranking method
        concept_aware: Enable concept-aware hub detection
        modality_aware: Enable modality-aware hub detection
        concept_field: Metadata field name for concept labels
        modality_field: Metadata field name for modality labels
        num_concepts: Number of concept clusters for auto-detection
        text_index_path: Path to text index file (for multi-index mode)
        text_embeddings_path: Path to text embeddings file (for multi-index mode)
        image_index_path: Path to image index file (for multi-index mode)
        image_embeddings_path: Path to image embeddings file (for multi-index mode)
        late_fusion: Enable late fusion of multi-index results
        fusion_method: Late fusion method ("rrf", "weighted_sum", "max")
        text_weight: Weight for text index in fusion (0.0-1.0)
        image_weight: Weight for image index in fusion (0.0-1.0)
        output_dir: Directory to save reports
        k: Number of nearest neighbors to retrieve
        num_queries: Number of queries to sample
        **kwargs: Additional configuration options
        
    Returns:
        Scan results dictionary
        
    Example:
        ```python
        from hubscan.sdk import scan_with_ranking
        
        # Hybrid search with reranking and concept awareness
        results = scan_with_ranking(
            embeddings_path="data/embeddings.npy",
            query_texts_path="data/queries.json",
            ranking_method="hybrid",
            hybrid_alpha=0.6,
            rerank=True,
            concept_aware=True,
            modality_aware=True
        )
        
        # Multi-index scan with late fusion
        results = scan_with_ranking(
            text_index_path="data/text_index.index",
            image_index_path="data/image_index.index",
            metadata_path="data/metadata.json",
            late_fusion=True,
            fusion_method="rrf",
            modality_aware=True
        )
        ```
    """
    config = _create_config_from_params(
        embeddings_path=embeddings_path,
        index_path=index_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        k=k,
        num_queries=num_queries,
        **kwargs,
    )
    
    # Set ranking configuration
    config.scan.ranking.method = ranking_method
    config.scan.ranking.hybrid_alpha = hybrid_alpha
    if query_texts_path:
        config.scan.query_texts_path = query_texts_path
    
    # Set reranking configuration
    config.scan.ranking.rerank = rerank
    if rerank:
        config.scan.ranking.rerank_method = rerank_method
        config.scan.ranking.rerank_top_n = rerank_top_n
        if rerank_params:
            config.scan.ranking.rerank_params = rerank_params
    
    # Apply concept-aware and modality-aware settings
    if concept_aware:
        config.detectors.concept_aware.enabled = True
        config.detectors.concept_aware.metadata_field = concept_field
        config.detectors.concept_aware.num_concepts = num_concepts
    if modality_aware:
        config.detectors.modality_aware.enabled = True
        config.detectors.modality_aware.doc_modality_field = modality_field
        config.detectors.modality_aware.query_modality_field = modality_field
    
    # Apply multi-index configuration
    if text_index_path or image_index_path:
        from .config.config import MultiIndexConfig, LateFusionConfig
        
        config.input.mode = "multi_index"
        config.input.multi_index = MultiIndexConfig(
            text_index_path=text_index_path,
            text_embeddings_path=text_embeddings_path,
            image_index_path=image_index_path,
            image_embeddings_path=image_embeddings_path,
        )
        config.scan.ranking.parallel_retrieval = True
        
        if late_fusion:
            config.input.late_fusion = LateFusionConfig(
                enabled=True,
                fusion_method=fusion_method,
                text_weight=text_weight,
                image_weight=image_weight,
            )
    
    scanner = Scanner(config)
    scanner.load_data()
    return scanner.scan()


def compare_ranking_methods(
    embeddings_path: str,
    metadata_path: Optional[str] = None,
    query_texts_path: Optional[str] = None,
    methods: Optional[list] = None,
    output_dir: str = "reports/",
    k: int = 20,
    num_queries: int = 10000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compare detection performance across multiple ranking methods.
    
    Args:
        embeddings_path: Path to embeddings file (.npy/.npz)
        metadata_path: Optional path to metadata file
        query_texts_path: Optional path to query texts file (required for lexical/hybrid)
        methods: List of ranking methods to compare (default: ["vector", "hybrid"])
        output_dir: Directory to save reports
        k: Number of nearest neighbors to retrieve
        num_queries: Number of queries to sample
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with comparison results:
        - methods: List of methods tested
        - results: Dictionary mapping method names to scan results
        - comparison: Comparison metrics
        
    Example:
        ```python
        from hubscan.sdk import compare_ranking_methods
        
        comparison = compare_ranking_methods(
            embeddings_path="data/embeddings.npy",
            query_texts_path="data/queries.json",
            methods=["vector", "hybrid", "lexical"]
        )
        
        for method, results in comparison["results"].items():
            print(f"{method}: {len(results['verdicts'])} suspicious docs")
        ```
    """
    if methods is None:
        methods = ["vector", "hybrid"]
    
    results_by_method = {}
    
    for method in methods:
        if method in ["lexical", "hybrid"] and not query_texts_path:
            continue  # Skip methods that require query texts
        
        method_results = scan_with_ranking(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            query_texts_path=query_texts_path,
            ranking_method=method,
            output_dir=f"{output_dir}/{method}",
            k=k,
            num_queries=num_queries,
            **kwargs,
        )
        results_by_method[method] = method_results
    
    # Compute comparison metrics if ground truth available
    comparison_metrics = None
    if metadata_path:
        # Try to load ground truth
        from .core.io.metadata import load_metadata
        metadata = load_metadata(metadata_path)
        if metadata.has_field("is_adversarial"):
            comparison_metrics = {}
            for method, method_results in results_by_method.items():
                if method_results.get("detection_metrics"):
                    comparison_metrics[method] = method_results["detection_metrics"]
    
    return {
        "methods": list(results_by_method.keys()),
        "results": results_by_method,
        "comparison": comparison_metrics,
    }


def explain_document(
    results: Dict[str, Any],
    doc_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Get detailed explanation for why a document was flagged.
    
    Args:
        results: Results dictionary from scan()
        doc_index: Document index to explain
        
    Returns:
        Dictionary with explanation details, or None if not found
        
    Example:
        ```python
        from hubscan.sdk import scan, explain_document
        
        results = scan(embeddings_path="data/embeddings.npy")
        explanation = explain_document(results, doc_index=42)
        
        if explanation:
            print(f"Risk Score: {explanation['risk_score']}")
            print(f"Hub Z-Score: {explanation['hubness']['hub_z']}")
        ```
    """
    json_report = results["json_report"]
    
    for doc in json_report["suspicious_documents"]:
        if doc["doc_index"] == doc_index:
            return doc
    
    return None


def _create_config_from_params(
    embeddings_path: Optional[str] = None,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    output_dir: str = "reports/",
    k: int = 20,
    num_queries: int = 10000,
    **kwargs,
) -> Config:
    """Create config from simple parameters."""
    config = Config()
    
    if index_path:
        config.input.mode = "faiss_index"
        config.input.index_path = index_path
        if embeddings_path:
            config.input.embeddings_path = embeddings_path
    elif embeddings_path:
        config.input.mode = "embeddings_only"
        config.input.embeddings_path = embeddings_path
    else:
        raise ValueError("Either embeddings_path or index_path must be provided")
    
    if metadata_path:
        config.input.metadata_path = metadata_path
    
    config.scan.k = k
    config.scan.num_queries = num_queries
    config.output.out_dir = output_dir
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif "." in key:
            # Handle nested attributes like "detectors.hubness.enabled"
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    
    return config

