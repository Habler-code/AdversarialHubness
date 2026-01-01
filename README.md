# HubScan: Adversarial Hubness Detection for RAG Systems

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

HubScan is an open-source security scanner that detects adversarial hubs in vector indices and RAG systems. It identifies malicious embeddings that manipulate retrieval by appearing in top-k results for an unusually large fraction of queries.

![Adversarial Hub Detection](docs/images/hubscan-hero.png)

## Key Features

**Three Detection Modes** for comprehensive coverage:

![Detection Modes](docs/images/detection-modes.png)

| Detection Mode | What It Catches | When to Use |
|----------------|-----------------|-------------|
| Global | Hubs dominating across all queries | Default, always recommended |
| Concept-Aware | Hubs targeting specific topics or categories | When your data has semantic categories |
| Modality-Aware | Cross-modal hubs exploiting text-image gaps | For multimodal systems |

**Production-Ready Architecture** for multimodal systems:

![Gold Standard Architecture](docs/images/gold-standard-architecture.png)

## Benchmark Results

| Dataset | Detection Type | Precision | Recall | F1 |
|---------|----------------|-----------|--------|-----|
| Wikipedia (text) | Concept-Aware | 96% | 100% | 0.98 |
| Multimodal (image + text) | Modality-Aware | 100% | 100% | 1.00 |

See the [benchmarks documentation](benchmarks/README.md) for detailed methodology and results.

## Table of Contents

- [What is Adversarial Hubness?](#what-is-adversarial-hubness)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detection Modes](#detection-modes-1)
- [CLI Reference](#cli-reference)
- [SDK Reference](#sdk-reference)
- [Customizing for Your Data](#customizing-for-your-data)
- [Configuration](#configuration)
- [Gold Standard Multimodal Architecture](#gold-standard-multimodal-architecture)
- [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Contributing](#contributing)

## What is Adversarial Hubness?

In vector search, hubness is when some documents naturally appear in many nearest-neighbor results. Adversarial hubs are artificially crafted embeddings that exploit this phenomenon to:

- Inject malicious content into RAG responses
- Manipulate search rankings
- Bypass content moderation
- Degrade system quality

### Detection Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Hub Rate | Fraction of queries retrieving the document | Greater than 5% is suspicious |
| Hub Z-Score | Statistical deviation from median | Greater than 5.0 is HIGH risk |
| Cluster Spread | Diversity of query clusters | High entropy is suspicious |
| Concept Hub Z | Per-topic hubness z-score | Greater than 4.0 is concept-specific hub |

## Installation

```bash
git clone https://github.com/cisco-ai-defense/hubscan.git
cd hubscan
pip install -e .

# With vector database support
pip install -e ".[pinecone,qdrant,weaviate]"
```

Requirements: Python 3.11+

## Quick Start

### Basic Scan

```bash
# Generate sample data (if needed)
python examples/scripts/generate_toy_data.py

# Run scan
hubscan scan --config examples/configs/toy_config.yaml

# View results
open examples/reports/report.html
```

### With Concept-Aware Detection

```bash
hubscan scan --config your_config.yaml --concept-aware --concept-field category
```

### With Modality-Aware Detection

```bash
hubscan scan --config your_config.yaml --modality-aware --modality-field type
```

### With Multi-Index Late Fusion

```bash
hubscan scan --config your_config.yaml \
    --multi-index \
    --text-index data/text_index.index \
    --image-index data/image_index.index \
    --late-fusion \
    --fusion-method rrf
```

## Detection Modes

### Global Detection (Default)

Detects documents appearing in top-k results for an anomalously high fraction of ALL queries.

Best for: General-purpose detection, baseline security scanning

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    k=20,
    num_queries=10000
)

print(f"HIGH risk documents: {results['json_report']['summary']['verdict_counts']['HIGH']}")
```

CLI:
```bash
hubscan scan -c config.yaml
```

### Concept-Aware Detection

Detects hubs that dominate within specific semantic categories but may be hidden in global statistics.

Best for:
- News or articles with categories
- E-commerce with product types
- Knowledge bases with topics

How it works:
1. Groups queries by concept (from metadata or automatic clustering)
2. Computes hub rate per concept
3. Flags documents with high z-scores in ANY concept

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    concept_aware=True,
    concept_field="category",
    k=20,
    num_queries=10000
)
```

CLI:
```bash
hubscan scan -c config.yaml --concept-aware --concept-field category
```

No metadata? HubScan auto-clusters queries into concepts:
```yaml
detectors:
  concept_aware:
    enabled: true
    mode: query_clustering
    num_concepts: 10
```

### Modality-Aware Detection

Detects hubs exploiting cross-modal retrieval patterns.

Best for:
- Image-text search systems
- Audio-text systems
- Any multi-modal RAG

How it works:
1. Tracks query and document modalities
2. Computes hits per modality combination
3. Flags documents with anomalous cross-modal retrieval

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    modality_aware=True,
    modality_field="modality",
    k=20,
    num_queries=5000
)
```

CLI:
```bash
hubscan scan -c config.yaml --modality-aware --modality-field type
```

### Combined Detection

Use all three modes together for maximum coverage:

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    concept_aware=True,
    concept_field="category",
    modality_aware=True,
    modality_field="type",
    k=20,
    num_queries=10000
)
```

CLI:
```bash
hubscan scan -c config.yaml \
    --concept-aware --concept-field category \
    --modality-aware --modality-field type
```

## CLI Reference

```bash
hubscan scan [OPTIONS]

Options:
  -c, --config PATH          Path to config YAML file [required]
  -o, --output TEXT          Output directory
  --summary-only             Show only summary, don't save reports
  
  # Ranking Method
  --ranking-method [vector|hybrid|lexical]
  --query-texts PATH         Query texts for lexical/hybrid search
  --hybrid-alpha FLOAT       Vector weight in hybrid mode (0.0-1.0)
  
  # Reranking
  --rerank                   Enable reranking post-processing
  --rerank-method TEXT       Reranking method name
  --rerank-top-n INTEGER     Candidates before reranking
  
  # Concept Detection
  --concept-aware            Enable concept-aware detection
  --concept-field TEXT       Metadata field for concepts [default: concept]
  --num-concepts INTEGER     Number of auto-clusters if no metadata
  
  # Modality Detection
  --modality-aware           Enable modality-aware detection
  --modality-field TEXT      Metadata field for modality [default: modality]
  
  # Multi-Index Mode
  --multi-index              Enable multi-index mode for multimodal systems
  --text-index PATH          Path to text embedding index
  --image-index PATH         Path to image embedding index
  --text-embeddings PATH     Path to text embeddings file
  --image-embeddings PATH    Path to image embeddings file
  
  # Late Fusion
  --late-fusion              Enable late fusion of multi-index results
  --fusion-method TEXT       Fusion method: rrf, weighted_sum, max [default: rrf]
  --text-weight FLOAT        Weight for text index results [default: 0.5]
  --image-weight FLOAT       Weight for image index results [default: 0.5]
```

### Extract Embeddings Command

```bash
hubscan extract-embeddings [OPTIONS]

Options:
  -c, --config PATH          Path to config YAML file [required]
  -o, --output TEXT          Output path for embeddings (.npy file) [required]
  --batch-size INTEGER       Number of vectors per batch [default: 1000]
  --limit INTEGER            Maximum vectors to extract [default: None, all]
```

### Examples

```bash
# Basic scan
hubscan scan -c config.yaml

# Hybrid search with concept awareness
hubscan scan -c config.yaml \
    --ranking-method hybrid \
    --query-texts queries.json \
    --concept-aware

# Full multimodal detection suite
hubscan scan -c config.yaml \
    --concept-aware --concept-field topic \
    --modality-aware --modality-field media_type

# Multi-index with late fusion
hubscan scan -c config.yaml \
    --multi-index \
    --text-index data/text.index \
    --image-index data/image.index \
    --late-fusion --fusion-method rrf

# Quick summary only
hubscan scan -c config.yaml --summary-only
```

## SDK Reference

### Basic Scan

```python
from hubscan import scan, get_suspicious_documents, Verdict

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    k=20,
    num_queries=10000
)

# Get high-risk documents
suspicious = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
for doc in suspicious:
    print(f"Doc {doc['doc_index']}: Risk={doc['risk_score']:.3f}")
```

### With Concept and Modality Detection

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    concept_aware=True,
    concept_field="category",
    modality_aware=True,
    modality_field="type",
    k=20,
    num_queries=10000
)

# Access concept-specific scores
for doc in results['json_report']['suspicious_documents']:
    print(f"Doc {doc['doc_index']}:")
    print(f"  Global Hub Z: {doc['hubness'].get('hub_z', 'N/A')}")
    print(f"  Max Concept Z: {doc['hubness'].get('max_concept_hub_z', 'N/A')}")
```

### Multi-Index with Late Fusion

```python
from hubscan import scan

results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    text_index_path="data/text_index.index",
    image_index_path="data/image_index.index",
    text_embeddings_path="data/text_embeddings.npy",
    image_embeddings_path="data/image_embeddings.npy",
    late_fusion=True,
    fusion_method="rrf",
    text_weight=0.5,
    image_weight=0.5,
    k=20,
    num_queries=5000
)
```

### In-Memory Scan

```python
from hubscan import quick_scan
import numpy as np

# Scan embeddings directly without files
embeddings = np.random.randn(10000, 384).astype(np.float32)
results = quick_scan(embeddings, k=20, num_queries=1000)
```

### Full Configuration Control

```python
from hubscan import Config, Scanner

config = Config.from_yaml("config.yaml")

# Programmatic config overrides
config.detectors.concept_aware.enabled = True
config.detectors.concept_aware.metadata_field = "my_category_field"
config.detectors.concept_aware.num_concepts = 15

config.detectors.modality_aware.enabled = True
config.detectors.modality_aware.doc_modality_field = "media_type"

scanner = Scanner(config)
scanner.load_data()
results = scanner.scan()
```

## Customizing for Your Data

### Your Metadata Uses Different Field Names?

HubScan maps to your data's field names via configuration:

| Your Field | HubScan Parameter | Example Values |
|------------|-------------------|----------------|
| category, topic, subject | concept_field | "news", "sports", "tech" |
| type, media_type, format | modality_field | "text", "image", "video" |

CLI:
```bash
hubscan scan -c config.yaml \
    --concept-aware --concept-field topic \
    --modality-aware --modality-field media_type
```

SDK:
```python
results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    concept_aware=True,
    concept_field="topic",
    modality_aware=True,
    modality_field="media_type",
)
```

Config YAML:
```yaml
detectors:
  concept_aware:
    enabled: true
    mode: metadata
    metadata_field: topic
    
  modality_aware:
    enabled: true
    mode: metadata
    doc_modality_field: media_type
```

### No Concept Metadata? Use Auto-Clustering

```yaml
detectors:
  concept_aware:
    enabled: true
    mode: query_clustering
    num_concepts: 10
    clustering_algorithm: minibatch_kmeans
    seed: 42
```

Or use hybrid mode to try metadata first:
```yaml
detectors:
  concept_aware:
    enabled: true
    mode: hybrid
    metadata_field: category
    num_concepts: 10
```

### Metadata Format

JSON metadata file:
```json
{
  "topic": ["news", "sports", "tech", "news", ...],
  "media_type": ["text", "image", "text", "video", ...]
}
```

## Configuration

### Full Configuration Reference

```yaml
input:
  mode: embeddings_only
  embeddings_path: data/embeddings.npy
  metadata_path: data/metadata.json
  metric: cosine

scan:
  k: 20
  num_queries: 10000
  query_sampling: mixed
  ranking:
    method: vector
    hybrid_alpha: 0.5
    rerank: false
    rerank_top_n: 100

detectors:
  hubness:
    enabled: true
    use_rank_weights: true
    use_distance_weights: true
    use_contrastive_delta: true
    use_bucket_concentration: true
    
  cluster_spread:
    enabled: true
    num_clusters: 100
    
  stability:
    enabled: false
    
  dedup:
    enabled: true
    
  concept_aware:
    enabled: false
    mode: hybrid
    metadata_field: concept
    num_concepts: 10
    concept_hub_z_threshold: 4.0
    
  modality_aware:
    enabled: false
    mode: metadata
    doc_modality_field: modality
    query_modality_field: modality
    cross_modal_penalty: 1.5

scoring:
  weights:
    hub_z: 0.6
    cluster_spread: 0.15
    stability: 0.05
    boilerplate: 0.2

thresholds:
  policy: hybrid
  hub_z: 5.0
  percentile: 0.015

output:
  out_dir: reports/
  privacy_mode: false
```

## Gold Standard Multimodal Architecture

HubScan supports the production architecture for secure multimodal RAG systems.

### Architecture Overview

1. **Query Understanding**: Detect query modality (text/image/both) and topic/intent
2. **Parallel Retrieval**: Query text and image indexes independently
3. **Late Fusion**: Merge results using RRF, weighted sum, or max scoring
4. **Rerank and Filter**: Apply hubness detection and diversity enforcement

### Best Practices for Embedding Selection

For secure RAG systems, embedding choice significantly impacts hubness detection effectiveness:

**For Text Systems**:
- Use domain-specific sentence transformers
- Consider sparse-dense hybrid retrieval
- Maintain semantic separation between topics

**For Multimodal Systems**:
- Use separate embedding spaces for each modality
- Avoid unified cross-modal embedding spaces for security-critical applications
- Combine modality-specific retrievers with late fusion

### Configuration

```yaml
input:
  mode: multi_index
  multi_index:
    text_index_path: "data/text_index.index"
    text_embeddings_path: "data/text_embeddings.npy"
    image_index_path: "data/image_index.index"
    image_embeddings_path: "data/image_embeddings.npy"
  
  late_fusion:
    enabled: true
    fusion_method: rrf
    text_weight: 0.5
    image_weight: 0.5
    rrf_k: 60
  
  diversity:
    enabled: true
    min_distance: 0.3

scan:
  ranking:
    parallel_retrieval: true
```

### Usage

```python
from hubscan import Config, Scanner

config = Config.from_yaml("multi_index_config.yaml")
scanner = Scanner(config)
scanner.load_data()
results = scanner.scan()
```

The multi-index adapter automatically handles parallel retrieval and late fusion transparently.

## Supported Vector Databases

| Database | Mode | Status |
|----------|------|--------|
| FAISS | embeddings_only, faiss_index | Full support |
| Pinecone | pinecone | Full support |
| Qdrant | qdrant | Full support |
| Weaviate | weaviate | Full support |
| Multi-Index | multi_index | Gold standard multimodal |

```bash
pip install -e ".[pinecone,qdrant,weaviate]"
```

### Extracting Embeddings from Vector Databases

HubScan can extract embeddings from external vector databases for analysis or migration:

**CLI:**
```bash
hubscan extract-embeddings \
    --config pinecone_config.yaml \
    --output embeddings.npy \
    --batch-size 1000 \
    --limit 10000
```

**SDK:**
```python
from hubscan import Config, Scanner

config = Config.from_yaml("config.yaml")
scanner = Scanner(config)
scanner.load_data()

embeddings, ids = scanner.extract_embeddings(
    output_path="embeddings.npy",
    batch_size=1000,
    limit=None  # None = extract all
)
```

Supported databases: FAISS, Pinecone, Qdrant, Weaviate. See [Usage Guide](docs/USAGE.md#extracting-embeddings) for details.

## Benchmarks

HubScan includes benchmarks with real datasets:

| Dataset | Type | Precision | Recall | F1 |
|---------|------|-----------|--------|-----|
| Wikipedia | Text with Categories | 96% | 100% | 0.98 |
| Multimodal | Image + Text | 100% | 100% | 1.00 |

Run benchmarks:
```bash
cd benchmarks/scripts

# Wikipedia benchmark
python run_benchmark.py \
    --dataset ../data/wikipedia/ \
    --config ../configs/concept_modality.yaml

# Multimodal benchmark
python run_benchmark.py \
    --dataset ../data/multimodal/ \
    --config ../configs/multimodal.yaml
```

See [benchmarks/README.md](benchmarks/README.md) for complete documentation.

## Documentation

- [Usage Guide](docs/USAGE.md): Complete CLI and configuration documentation
- [SDK Reference](docs/SDK.md): Python SDK API reference
- [Plugin System](docs/PLUGINS.md): Custom detectors and ranking methods
- [Concept and Modality Guide](docs/CONCEPTS_AND_MODALITIES.md): Advanced detection modes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure tests pass (`pytest tests/`)
5. Submit a Pull Request

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/cisco-ai-defense/hubscan/issues)
- Documentation: See `docs/` directory
