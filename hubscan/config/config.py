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

"""Configuration management using Pydantic models."""

from typing import Literal, Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class MultiIndexConfig(BaseModel):
    """Multi-index configuration for gold standard architecture."""
    text_index_path: Optional[str] = Field(default=None, description="Path to text index file (.index)")
    text_embeddings_path: Optional[str] = Field(default=None, description="Path to text embeddings file (.npy)")
    image_index_path: Optional[str] = Field(default=None, description="Path to image index file (.index)")
    image_embeddings_path: Optional[str] = Field(default=None, description="Path to image embeddings file (.npy)")
    unified_index_path: Optional[str] = Field(default=None, description="Path to unified/cross-modal index file (.index) - optional recall backstop")
    unified_embeddings_path: Optional[str] = Field(default=None, description="Path to unified/cross-modal embeddings file (.npy)")
    text_metric: Literal["cosine", "ip", "l2"] = Field(default="cosine", description="Distance metric for text index")
    image_metric: Literal["cosine", "ip", "l2"] = Field(default="cosine", description="Distance metric for image index")
    unified_metric: Literal["cosine", "ip", "l2"] = Field(default="cosine", description="Distance metric for unified index")


class LateFusionConfig(BaseModel):
    """Late fusion configuration for multi-index retrieval."""
    enabled: bool = Field(default=False, description="Enable late fusion of results from multiple indexes")
    normalize_scores: bool = Field(default=True, description="Normalize scores from different indexes before merging")
    fusion_method: Literal["rrf", "weighted_sum", "max"] = Field(
        default="rrf",
        description="Fusion method: 'rrf' (Reciprocal Rank Fusion), 'weighted_sum', or 'max'"
    )
    text_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for text index results")
    image_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for image index results")
    unified_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for unified/cross-modal index results (recall backstop)")
    rrf_k: int = Field(default=60, ge=1, description="RRF constant (higher = more weight to top results)")
    unified_top_k: Optional[int] = Field(default=None, description="Max results from unified index (None = use scan.k)")


class DiversityConfig(BaseModel):
    """Diversity enforcement configuration."""
    enabled: bool = Field(default=False, description="Enable diversity enforcement in final results")
    min_distance: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum cosine distance between results")
    max_results_per_cluster: Optional[int] = Field(default=None, description="Max results per semantic cluster")


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search (dense + lexical).
    
    Hybrid search combines semantic (dense vector) search with lexical (keyword) search.
    Two backends are supported:
    
    - **client_fusion**: Works with any vector DB. Runs dense search in DB, then computes
      lexical scores locally (BM25/TF-IDF) from document texts and fuses client-side.
      Requires document texts in metadata.
    
    - **native_sparse**: Uses DB-native sparse vectors (Pinecone sparse vectors, 
      Qdrant dense+sparse fields, Weaviate BM25). Requires the DB to already have
      sparse vectors indexed or BM25 enabled.
    """
    backend: Literal["client_fusion", "native_sparse", "auto"] = Field(
        default="auto",
        description=(
            "Hybrid search backend: "
            "'client_fusion' (dense DB + local lexical), "
            "'native_sparse' (DB-native sparse vectors), "
            "'auto' (try native_sparse if available, fallback to client_fusion)"
        )
    )
    lexical_backend: Literal["bm25", "tfidf"] = Field(
        default="bm25",
        description="Lexical scoring algorithm for client_fusion mode"
    )
    text_field: str = Field(
        default="text",
        description="Metadata field containing document text (for client_fusion)"
    )
    normalize_scores: bool = Field(
        default=True,
        description="Normalize dense and lexical scores to [0,1] before fusion"
    )
    
    # Native sparse settings (for DB backends that support it)
    # Qdrant
    qdrant_dense_vector_name: str = Field(
        default="dense",
        description="Name of the dense vector field in Qdrant collection"
    )
    qdrant_sparse_vector_name: str = Field(
        default="sparse",
        description="Name of the sparse vector field in Qdrant collection"
    )
    
    # Pinecone
    pinecone_has_sparse: bool = Field(
        default=False,
        description="Whether Pinecone index has sparse vectors"
    )
    
    # Weaviate (uses native BM25)
    weaviate_bm25_properties: List[str] = Field(
        default_factory=lambda: ["text"],
        description="Properties to search with BM25 in Weaviate"
    )


class InputConfig(BaseModel):
    """Input configuration."""
    mode: Literal[
        "embeddings_only", 
        "faiss_index", 
        "vector_db_export",
        "pinecone",
        "qdrant",
        "weaviate",
        "multi_index",
    ] = Field(
        default="embeddings_only",
        description="Input mode"
    )
    embeddings_path: Optional[str] = Field(default=None, description="Path to embeddings file (.npy/.npz)")
    index_path: Optional[str] = Field(default=None, description="Path to FAISS index file (.index)")
    metadata_path: Optional[str] = Field(default=None, description="Path to metadata file (JSON/JSONL/Parquet)")
    adapter: Optional[str] = Field(default="generic_jsonl", description="Adapter for vector_db_export mode")
    metric: Literal["cosine", "ip", "l2"] = Field(default="cosine", description="Distance metric")
    dimension: Optional[int] = Field(default=None, description="Vector dimension (required for some backends)")
    
    # Multi-index configuration (for gold standard architecture)
    multi_index: Optional[MultiIndexConfig] = Field(default=None, description="Multi-index configuration for parallel retrieval")
    late_fusion: Optional[LateFusionConfig] = Field(default=None, description="Late fusion configuration")
    diversity: Optional[DiversityConfig] = Field(default=None, description="Diversity enforcement configuration")
    
    # Pinecone-specific configuration
    pinecone_index_name: Optional[str] = Field(default=None, description="Pinecone index name")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment (deprecated in v3+)")
    
    # Qdrant-specific configuration
    qdrant_collection_name: Optional[str] = Field(default=None, description="Qdrant collection name")
    qdrant_url: Optional[str] = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (for Qdrant Cloud)")
    
    # Weaviate-specific configuration
    weaviate_class_name: Optional[str] = Field(default=None, description="Weaviate class name")
    weaviate_url: Optional[str] = Field(default="http://localhost:8080", description="Weaviate server URL")
    weaviate_api_key: Optional[str] = Field(default=None, description="Weaviate API key (for Weaviate Cloud)")


class IndexConfig(BaseModel):
    """FAISS index configuration."""
    type: Literal["hnsw", "ivf_pq", "flat"] = Field(default="hnsw", description="Index type")
    params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "M": 32,
            "efSearch": 128,
            "nlist": 4096,
            "nprobe": 16,
        },
        description="Index-specific parameters"
    )
    save_path: Optional[str] = Field(default=None, description="Path to save built index")


class RankingConfig(BaseModel):
    """Ranking method configuration."""
    method: str = Field(
        default="vector",
        description="Ranking method name (built-in: vector, hybrid, lexical, or custom registered name)"
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in hybrid mode (1-alpha for lexical)"
    )
    hybrid: HybridSearchConfig = Field(
        default_factory=HybridSearchConfig,
        description="Hybrid search configuration (dense + lexical)"
    )
    rerank: bool = Field(
        default=False,
        description="Enable reranking as post-processing step"
    )
    rerank_method: str = Field(
        default="default",
        description="Reranking method name (built-in: default, or custom registered name)"
    )
    rerank_top_n: int = Field(
        default=100,
        ge=1,
        description="Number of candidates to retrieve before reranking (only used if rerank=True)"
    )
    lexical_backend: Optional[str] = Field(
        default=None,
        description="Lexical search backend (e.g., 'bm25', 'tfidf') - deprecated, use hybrid.lexical_backend"
    )
    custom_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom parameters for ranking method (passed as **kwargs to search method)"
    )
    rerank_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom parameters for reranking method (passed as **kwargs to rerank method)"
    )
    # Multi-index parallel retrieval
    parallel_retrieval: bool = Field(
        default=False,
        description="Enable parallel retrieval from multiple indexes (requires multi_index config)"
    )


class ScanConfig(BaseModel):
    """Scan configuration."""
    k: int = Field(default=20, ge=1, description="Number of nearest neighbors to retrieve")
    num_queries: int = Field(default=10000, ge=1, description="Total number of queries to sample")
    query_sampling: Literal["real_queries", "random_docs_as_queries", "cluster_centroids", "mixed"] = Field(
        default="mixed",
        description="Query sampling strategy"
    )
    batch_size: int = Field(default=2048, ge=1, description="Batch size for query processing")
    query_embeddings_path: Optional[str] = Field(default=None, description="Path to real query embeddings")
    query_texts_path: Optional[str] = Field(default=None, description="Path to query texts file (for lexical/hybrid search)")
    mixed_proportions: Dict[str, float] = Field(
        default_factory=lambda: {
            "real_queries": 0.0,
            "random_docs_as_queries": 0.5,
            "cluster_centroids": 0.5,
        },
        description="Proportions for mixed sampling"
    )
    stratified_by: Optional[str] = Field(default=None, description="Metadata field for stratified sampling")
    seed: int = Field(default=42, description="Random seed")
    ranking: RankingConfig = Field(default_factory=RankingConfig, description="Ranking method configuration")


class HubnessDetectorConfig(BaseModel):
    """Hubness detector configuration."""
    enabled: bool = Field(default=True, description="Enable hubness detector")
    validate_exact: bool = Field(default=False, description="Validate with exact search on subset")
    exact_validation_queries: Optional[int] = Field(default=None, description="Number of queries for exact validation")
    use_rank_weights: bool = Field(default=True, description="Weight hits by rank position (rank 1 > rank k)")
    use_distance_weights: bool = Field(default=True, description="Weight hits by similarity/distance scores")
    
    # Contrastive bucket detection (for concept-targeted attacks)
    use_contrastive_delta: bool = Field(
        default=True,
        description="Detect documents with hub rates much higher in one concept than others"
    )
    use_bucket_concentration: bool = Field(
        default=True,
        description="Detect documents with highly concentrated hub rates (Gini coefficient)"
    )


class ClusterSpreadDetectorConfig(BaseModel):
    """Cluster spread detector configuration."""
    enabled: bool = Field(default=True, description="Enable cluster spread detector")
    num_clusters: int = Field(default=1024, ge=2, description="Number of clusters for k-means")
    batch_size: int = Field(default=10000, ge=1, description="Batch size for k-means")


class StabilityDetectorConfig(BaseModel):
    """Stability detector configuration."""
    enabled: bool = Field(default=False, description="Enable stability detector")
    candidates_top_x: int = Field(default=200, ge=1, description="Top candidates to analyze")
    perturbations: int = Field(default=5, ge=1, description="Number of perturbations per query")
    sigma: float = Field(default=0.01, gt=0.0, description="Gaussian noise standard deviation")
    normalize: bool = Field(default=True, description="Renormalize after perturbation (for cosine/IP)")


class DedupDetectorConfig(BaseModel):
    """Deduplication detector configuration."""
    enabled: bool = Field(default=True, description="Enable deduplication detector")
    text_hash_field: Optional[str] = Field(default="text_hash", description="Metadata field for text hash")
    duplicate_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="L2 distance threshold for duplicates")
    suppress_boilerplate: bool = Field(default=True, description="Suppress obvious boilerplate")


class ConceptAwareConfig(BaseModel):
    """Concept-aware hub detection configuration.
    
    Enables detection of hubs that are localized to specific semantic concepts/topics.
    When enabled, queries are partitioned by concept and hub rates are computed per-concept.
    """
    enabled: bool = Field(default=False, description="Enable concept-aware hub detection")
    mode: Literal["metadata", "query_clustering", "doc_clustering", "hybrid"] = Field(
        default="hybrid",
        description="Concept assignment mode: 'metadata' uses existing labels, 'query_clustering' clusters queries, 'doc_clustering' clusters documents, 'hybrid' tries metadata first then falls back to clustering"
    )
    metadata_field: Optional[str] = Field(
        default="concept",
        description="Metadata field containing concept/topic labels (used in 'metadata' and 'hybrid' modes)"
    )
    num_concepts: int = Field(
        default=10,
        ge=2,
        le=1000,
        description="Number of concept clusters (used in clustering modes)"
    )
    clustering_algorithm: Literal["minibatch_kmeans", "kmeans"] = Field(
        default="minibatch_kmeans",
        description="Clustering algorithm for auto-concept detection"
    )
    clustering_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "batch_size": 1024,
            "n_init": 3,
            "max_iter": 100,
        },
        description="Parameters for clustering algorithm"
    )
    min_concept_size: int = Field(
        default=10,
        ge=1,
        description="Minimum number of queries per concept (smaller concepts merged to 'other')"
    )
    concept_hub_z_threshold: float = Field(
        default=4.0,
        ge=0.0,
        description="Z-score threshold for flagging concept-specific hubs (can be lower than global)"
    )
    seed: int = Field(default=42, description="Random seed for reproducible clustering")


class ModalityAwareConfig(BaseModel):
    """Modality-aware hub detection configuration.
    
    Enables detection of cross-modal hubs in multi-modal embedding spaces.
    Cross-modal hubs are documents that appear in top-K for queries of a different modality.
    """
    enabled: bool = Field(default=False, description="Enable modality-aware hub detection")
    mode: Literal["metadata", "default_text"] = Field(
        default="default_text",
        description="Modality resolution mode: 'metadata' reads from metadata field, 'default_text' assumes all text"
    )
    doc_modality_field: Optional[str] = Field(
        default="modality",
        description="Metadata field containing document modality (e.g., 'text', 'image', 'audio')"
    )
    query_modality_field: Optional[str] = Field(
        default="modality",
        description="Metadata field containing query modality"
    )
    default_doc_modality: str = Field(
        default="text",
        description="Default modality for documents without metadata"
    )
    default_query_modality: str = Field(
        default="text",
        description="Default modality for queries without metadata"
    )
    known_modalities: list = Field(
        default_factory=lambda: ["text", "image", "audio", "video", "code"],
        description="List of known modalities for validation and reporting"
    )
    cross_modal_penalty: float = Field(
        default=1.5,
        ge=1.0,
        description="Multiplier applied to hub score for cross-modal hits (>1.0 increases suspicion)"
    )
    separate_modality_stats: bool = Field(
        default=True,
        description="Compute hub statistics separately per modality combination"
    )
    # Subspace projection for cross-modal hub detection
    use_subspace_projection: bool = Field(
        default=False,
        description="Project embeddings onto modality-specific subspaces to detect cross-modal anomalies"
    )
    subspace_components: int = Field(
        default=64,
        ge=16,
        le=256,
        description="Number of PCA components for modality subspace"
    )
    subspace_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of subspace projection score in cross-modal detection"
    )


class DetectorsConfig(BaseModel):
    """Detectors configuration."""
    hubness: HubnessDetectorConfig = Field(default_factory=HubnessDetectorConfig)
    cluster_spread: ClusterSpreadDetectorConfig = Field(default_factory=ClusterSpreadDetectorConfig)
    stability: StabilityDetectorConfig = Field(default_factory=StabilityDetectorConfig)
    dedup: DedupDetectorConfig = Field(default_factory=DedupDetectorConfig)
    concept_aware: ConceptAwareConfig = Field(default_factory=ConceptAwareConfig)
    modality_aware: ModalityAwareConfig = Field(default_factory=ModalityAwareConfig)


class ScoringWeights(BaseModel):
    """Scoring weights configuration."""
    hub_z: float = Field(default=0.6, ge=0.0, description="Weight for hubness z-score")
    cluster_spread: float = Field(default=0.2, ge=0.0, description="Weight for cluster spread score")
    stability: float = Field(default=0.2, ge=0.0, description="Weight for stability score")
    boilerplate: float = Field(default=0.3, ge=0.0, description="Penalty weight for boilerplate")
    # Concept/Modality scoring weights (only used when enabled, default 0.0 for backward compat)
    concept_hub_z: float = Field(
        default=0.0, 
        ge=0.0, 
        description="Weight for max concept-specific hub z-score (0.0 = disabled)"
    )
    cross_modal: float = Field(
        default=0.0, 
        ge=0.0, 
        description="Weight for cross-modal hub penalty (0.0 = disabled)"
    )


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class ThresholdsConfig(BaseModel):
    """Thresholds configuration."""
    policy: Literal["percentile", "z_score", "hybrid"] = Field(default="hybrid", description="Threshold policy")
    hub_z: float = Field(default=6.0, description="Z-score threshold for HIGH")
    percentile: float = Field(default=0.001, ge=0.0, le=1.0, description="Percentile threshold (0.001 = top 0.1%)")
    # MEDIUM threshold as ratio of HIGH (for dense embeddings, use lower ratio)
    medium_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="MEDIUM threshold as ratio of HIGH (0.5 = 50% of HIGH threshold). Lower values = more MEDIUM detections."
    )
    # Method-specific thresholds (optional, falls back to defaults if not specified)
    method_specific: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Method-specific thresholds: {'vector': {'hub_z': 6.0, 'percentile': 0.012}, ...}"
    )


class OutputConfig(BaseModel):
    """Output configuration."""
    out_dir: str = Field(default="reports/", description="Output directory")
    privacy_mode: bool = Field(default=True, description="Redact sensitive information")
    emit_embeddings: bool = Field(default=False, description="Include embeddings in output")
    max_example_queries: int = Field(default=10, ge=1, description="Max example queries per doc")


class Config(BaseSettings):
    """Main configuration model."""
    input: InputConfig = Field(default_factory=InputConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode='after')
    def validate_hybrid_requirements(self) -> 'Config':
        """Validate hybrid/lexical search requirements at config level.
        
        This provides early fail-fast validation before data loading.
        Full validation (e.g., checking metadata has text field) happens in Scanner.
        """
        ranking_method = self.scan.ranking.method
        
        # Hybrid/lexical require query_texts_path
        if ranking_method in ("hybrid", "lexical"):
            if not self.scan.query_texts_path:
                raise ValueError(
                    f"scan.query_texts_path is required when ranking.method='{ranking_method}'. "
                    f"Provide a JSON file containing query text strings for lexical scoring."
                )
        
        # Client-fusion hybrid requires metadata for document texts
        if ranking_method == "hybrid":
            hybrid_config = self.scan.ranking.hybrid
            if hybrid_config.backend in ("client_fusion", "auto"):
                # We can only warn here; full validation requires loading metadata
                # Scanner.load_data() will do the full check
                pass
            
            # Native sparse has DB-specific requirements
            if hybrid_config.backend == "native_sparse":
                mode = self.input.mode
                if mode == "pinecone" and not hybrid_config.pinecone_has_sparse:
                    raise ValueError(
                        "Native sparse hybrid for Pinecone requires sparse vectors. "
                        "Either set scan.ranking.hybrid.pinecone_has_sparse=true if your index "
                        "has sparse vectors, or use hybrid.backend='client_fusion'."
                    )
        
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

