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

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class InputConfig(BaseModel):
    """Input configuration."""
    mode: Literal[
        "embeddings_only", 
        "faiss_index", 
        "vector_db_export",
        "pinecone",
        "qdrant",
        "weaviate",
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


class HubnessDetectorConfig(BaseModel):
    """Hubness detector configuration."""
    enabled: bool = Field(default=True, description="Enable hubness detector")
    validate_exact: bool = Field(default=False, description="Validate with exact search on subset")
    exact_validation_queries: Optional[int] = Field(default=None, description="Number of queries for exact validation")
    use_rank_weights: bool = Field(default=True, description="Weight hits by rank position (rank 1 > rank k)")
    use_distance_weights: bool = Field(default=True, description="Weight hits by similarity/distance scores")


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


class DetectorsConfig(BaseModel):
    """Detectors configuration."""
    hubness: HubnessDetectorConfig = Field(default_factory=HubnessDetectorConfig)
    cluster_spread: ClusterSpreadDetectorConfig = Field(default_factory=ClusterSpreadDetectorConfig)
    stability: StabilityDetectorConfig = Field(default_factory=StabilityDetectorConfig)
    dedup: DedupDetectorConfig = Field(default_factory=DedupDetectorConfig)


class ScoringWeights(BaseModel):
    """Scoring weights configuration."""
    hub_z: float = Field(default=0.6, ge=0.0, description="Weight for hubness z-score")
    cluster_spread: float = Field(default=0.2, ge=0.0, description="Weight for cluster spread score")
    stability: float = Field(default=0.2, ge=0.0, description="Weight for stability score")
    boilerplate: float = Field(default=0.3, ge=0.0, description="Penalty weight for boilerplate")


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class ThresholdsConfig(BaseModel):
    """Thresholds configuration."""
    policy: Literal["percentile", "z_score", "hybrid"] = Field(default="hybrid", description="Threshold policy")
    hub_z: float = Field(default=6.0, description="Z-score threshold")
    percentile: float = Field(default=0.001, ge=0.0, le=1.0, description="Percentile threshold (0.001 = top 0.1%)")


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

