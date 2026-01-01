# HubScan Usage Guide

## Command-Line Interface (CLI)

### Basic Usage

```bash
# Run a scan with config file
hubscan scan --config examples/configs/toy_config.yaml

# Run scan with custom output directory
hubscan scan --config config.yaml --output custom_reports/

# Show only summary (don't save full reports)
hubscan scan --config config.yaml --summary-only

# Explain why a document was flagged
hubscan explain --doc-id 42 --report reports/report.json

# Build an index from embeddings
hubscan build-index --config config.yaml

# Extract embeddings from a vector database
hubscan extract-embeddings --config config.yaml --output embeddings.npy

# Verbose logging
hubscan scan --config config.yaml --verbose
```

### CLI Commands

#### `scan`

Run a hubness detection scan.

```bash
hubscan scan --config <config.yaml> [OPTIONS]
```

**Options:**
- `--config, -c`: Path to YAML configuration file (required)
- `--output, -o`: Output directory (overrides config)
- `--summary-only`: Show only summary, don't save full reports
- `--ranking-method`: Retrieval method ("vector", "hybrid", "lexical")
- `--hybrid-alpha`: Weight for vector search in hybrid mode (0.0-1.0, default: 0.5)
- `--query-texts`: Path to query texts file (for lexical/hybrid search)
- `--rerank`: Enable reranking as post-processing step
- `--rerank-method`: Reranking method name (default: "default")
- `--rerank-top-n`: Number of candidates to retrieve before reranking (default: 100)
- `--concept-aware`: Enable concept-aware hub detection
- `--concept-field`: Metadata field for concept labels (default: "concept")
- `--num-concepts`: Number of concept clusters for auto-detection (default: 10)
- `--modality-aware`: Enable modality-aware hub detection
- `--modality-field`: Metadata field for modality (default: "modality")
- `--multi-index`: Enable multi-index mode (parallel retrieval)
- `--text-index`: Path to text index file
- `--image-index`: Path to image index file
- `--text-embeddings`: Path to text embeddings file
- `--image-embeddings`: Path to image embeddings file
- `--late-fusion`: Enable late fusion of multi-index results
- `--fusion-method`: Late fusion method ("rrf", "weighted_sum", "max")
- `--text-weight`: Weight for text index in fusion (default: 0.4)
- `--image-weight`: Weight for image index in fusion (default: 0.4)
- `--verbose, -v`: Enable verbose logging

**Note**: Reranking is a post-processing step that can be applied to any ranking method. Use `--rerank` flag to enable it, or configure via config file (`ranking.rerank: true`, `ranking.rerank_method`).

**Examples:**
```bash
# Standard vector search
hubscan scan --config examples/configs/toy_config.yaml

# Hybrid search
hubscan scan --config config.yaml --ranking-method hybrid --hybrid-alpha 0.6 --query-texts queries.json

# Lexical search
hubscan scan --config config.yaml --ranking-method lexical --query-texts queries.json

# Vector search with reranking
hubscan scan --config config.yaml --ranking-method vector --rerank --rerank-method default --rerank-top-n 100

# Hybrid search with reranking
hubscan scan --config config.yaml --ranking-method hybrid --rerank --query-texts queries.json

# Compare ranking methods
hubscan compare-ranking --config config.yaml --query-texts queries.json --methods vector hybrid lexical

# Multi-index scan with late fusion (gold standard architecture)
hubscan scan --config config.yaml \
  --multi-index \
  --text-index data/text_index.index \
  --image-index data/image_index.index \
  --text-embeddings data/text_embeddings.npy \
  --image-embeddings data/image_embeddings.npy \
  --late-fusion \
  --fusion-method rrf \
  --modality-aware

# Concept-aware scan
hubscan scan --config config.yaml --concept-aware --num-concepts 15
```

### Detector Compatibility with Retrieval and Reranking Methods

HubScan automatically selects appropriate detectors based on the retrieval method:

| Detector | Vector | Hybrid | Lexical | Vector+Rerank | Hybrid+Rerank | Lexical+Rerank |
|----------|--------|--------|---------|---------------|---------------|----------------|
| **Hubness** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Cluster Spread** | Yes | Yes | No | Yes | Yes | No |
| **Stability** | Yes | Yes | No | Yes | Yes | No |
| **Deduplication** | Yes | Yes | Yes | Yes | Yes | Yes |

**Notes:**
- **Lexical Retrieval**: Cluster spread and stability detectors are automatically skipped because:
  - Cluster spread requires semantic query clustering (not applicable for keyword-based retrieval)
  - Stability requires query embeddings to perturb (lexical retrieval uses text, not embeddings)
- **Reranking Methods**: Reranking is a post-processing step that can be applied to any retrieval method (vector, hybrid, lexical)
  - When reranking is enabled, detectors use the reranked results
  - Reranking retrieves `rerank_top_n` candidates, then reranks to return top `k`
  - Built-in reranking method: `default` (simple top-k selection)
  - Custom reranking methods can be registered via the plugin system
- **Hybrid Retrieval**: All detectors use hybrid search internally for consistency

#### `build-index`

Build a FAISS index from embeddings.

```bash
hubscan build-index --config <config.yaml>
```

**Example:**
```bash
hubscan build-index --config config.yaml
```

#### `explain`

Explain why a specific document was flagged.

```bash
hubscan explain --doc-id <index> --report <report.json>
```

**Options:**
- `--doc-id`: Document index to explain (required)
- `--report`: Path to JSON report file (required)

**Example:**
```bash
hubscan explain --doc-id 42 --report reports/report.json
```

#### `extract-embeddings`

Extract embeddings from a vector database for analysis or migration.

```bash
hubscan extract-embeddings \
    --config config.yaml \
    --output embeddings.npy \
    --batch-size 1000 \
    --limit 10000
```

**Options:**
- `--config, -c`: Path to YAML configuration file (required)
- `--output, -o`: Output path for embeddings file (.npy format, required)
- `--batch-size`: Number of vectors to retrieve per batch (default: 1000)
- `--limit`: Maximum number of vectors to extract (default: None, extract all)

**Supported Databases:**
- FAISS: Uses `reconstruct_n()` to extract embeddings
- Pinecone: Uses query-based ID discovery + `fetch()` API
- Qdrant: Uses `scroll()` API to iterate through all vectors
- Weaviate: Uses cursor-based pagination

**Example Configurations:**

Pinecone:
```yaml
input:
  mode: pinecone
  pinecone_index_name: my-index
  pinecone_api_key: your-api-key
  dimension: 128
```

Qdrant:
```yaml
input:
  mode: qdrant
  qdrant_collection_name: my-collection
  qdrant_url: http://localhost:6333
```

Weaviate:
```yaml
input:
  mode: weaviate
  weaviate_class_name: MyClass
  weaviate_url: http://localhost:8080
```

#### `compare-ranking`

Compare detection performance across multiple ranking methods.

```bash
hubscan compare-ranking --config <config.yaml> [OPTIONS]
```

**Options:**
- `--config, -c`: Path to YAML configuration file (required)
- `--query-texts`: Path to query texts file (required for lexical/hybrid)
- `--methods`: Ranking methods to compare (default: ["vector", "hybrid"])
- `--output, -o`: Output directory (default: "reports/comparison")

**Example:**
```bash
hubscan compare-ranking --config config.yaml --query-texts queries.json --methods vector hybrid lexical
```

## SDK Usage

### Basic Scan

```python
from hubscan import scan, Verdict

# Run a scan
results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    k=20,
    num_queries=10000,
    output_dir="reports/"
)

# Access results
print(f"Runtime: {results['runtime']:.2f} seconds")
print(f"Verdicts: {results['json_report']['summary']['verdict_counts']}")
```

### Quick Scan (In-Memory)

```python
import numpy as np
from hubscan import quick_scan

embeddings = np.random.randn(1000, 128).astype(np.float32)
results = quick_scan(embeddings, k=10, num_queries=100)
```

### Retrieval and Reranking Methods

```python
from hubscan import scan_with_ranking, compare_ranking_methods

# Hybrid retrieval (combines vector + lexical)
results = scan_with_ranking(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    ranking_method="hybrid",
    hybrid_alpha=0.6,  # 60% vector, 40% lexical
    k=20
)

# Compare multiple retrieval methods
comparison = compare_ranking_methods(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    methods=["vector", "hybrid", "lexical"],
    k=20
)

# Compare with reranking
comparison_with_rerank = compare_ranking_methods(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    methods=["vector", "vector+rerank", "hybrid", "hybrid+rerank"],
    k=20
)

# Access comparison results
for method, method_results in comparison["results"].items():
    verdicts = method_results["verdicts"]
    print(f"{method}: {sum(1 for v in verdicts.values() if v.value == 'HIGH')} HIGH risk docs")
    
    # Access detection metrics if available
    if method_results.get("detection_metrics"):
        metrics = method_results["detection_metrics"]
        print(f"  Precision: {metrics.get('precision', 'N/A'):.3f}")
        print(f"  Recall: {metrics.get('recall', 'N/A'):.3f}")
        print(f"  F1: {metrics.get('f1', 'N/A'):.3f}")
```

### Get Suspicious Documents

```python
from hubscan import get_suspicious_documents, Verdict

# Get high-risk documents
high_risk = get_suspicious_documents(
    results,
    verdict=Verdict.HIGH,
    top_k=10
)

for doc in high_risk:
    print(f"Doc {doc['doc_index']}: Risk={doc['risk_score']:.4f}")
```

### Explain Document

```python
from hubscan import explain_document

explanation = explain_document(results, doc_index=42)
if explanation:
    print(f"Risk Score: {explanation['risk_score']}")
    print(f"Hub Z-Score: {explanation['hubness']['hub_z']}")
```

### Advanced: Custom Configuration

```python
from hubscan import Config, Scanner

# Load config
config = Config.from_yaml("config.yaml")

# Modify programmatically
config.scan.k = 30
config.detectors.stability.enabled = True
config.thresholds.hub_z = 5.0

# Enable reranking
config.scan.ranking.rerank = True
config.scan.ranking.rerank_method = "default"
config.scan.ranking.rerank_top_n = 100

# Or use method-specific thresholds:
config.thresholds.method_specific = {
    "vector": {"hub_z": 6.0, "percentile": 0.012},
    "hybrid": {"hub_z": 5.0, "percentile": 0.02},
    "lexical": {"hub_z": 6.0, "percentile": 0.012}
}

# Run scan
scanner = Scanner(config)
scanner.load_data()
results = scanner.scan()
```

## Examples

See `examples/sdk_example.py` for complete SDK examples.

## Output Formats

### JSON Report

Machine-readable report with full details:
- `reports/report.json`

### HTML Report

Human-friendly visual report:
- `reports/report.html`

### Console Output

Rich-formatted summary with tables and progress bars (if `rich` is installed).

## Configuration

See `examples/configs/toy_config.yaml` for a complete configuration example.

Key configuration sections:
- `input`: Input data configuration (supports FAISS, Pinecone, Qdrant, Weaviate)
- `index`: FAISS index configuration (for embeddings_only and faiss_index modes)
- `scan`: Scan parameters
- `detectors`: Detector settings
- `scoring`: Score combination weights
- `thresholds`: Verdict thresholds
- `output`: Output settings

### Vector Database Backends

HubScan supports multiple vector database backends:

#### FAISS (Default)
```yaml
input:
  mode: embeddings_only  # or faiss_index
  embeddings_path: data/embeddings.npy
```

#### Pinecone
```yaml
input:
  mode: pinecone
  pinecone_index_name: my-index
  pinecone_api_key: your-api-key
  dimension: 128  # Required
```

#### Qdrant
```yaml
input:
  mode: qdrant
  qdrant_collection_name: my-collection
  qdrant_url: http://localhost:6333  # Optional
  qdrant_api_key: your-api-key  # Optional, for Qdrant Cloud
```

#### Weaviate
```yaml
input:
  mode: weaviate
  weaviate_class_name: MyClass
  weaviate_url: http://localhost:8080  # Optional
  weaviate_api_key: your-api-key  # Optional, for Weaviate Cloud
```

**Note**: For external vector databases (Pinecone, Qdrant, Weaviate), you may need to install additional dependencies:
```bash
pip install pinecone-client  # For Pinecone (or 'pinecone' for v3+)
pip install qdrant-client      # For Qdrant
pip install weaviate-client   # For Weaviate (v3: >=3.26.7,<4.0.0)
```

### Extracting Embeddings

HubScan can extract embeddings from vector databases for analysis, migration, or offline processing:

**Use Cases:**
- Migrate embeddings from one database to another
- Analyze embeddings offline without database access
- Create backups of vector data
- Export embeddings for external analysis tools

**Example:**
```bash
# Extract from Pinecone
hubscan extract-embeddings \
    --config pinecone_config.yaml \
    --output pinecone_embeddings.npy \
    --batch-size 1000

# Extract from Qdrant (with limit)
hubscan extract-embeddings \
    --config qdrant_config.yaml \
    --output qdrant_embeddings.npy \
    --limit 50000
```

The extracted embeddings can then be used with HubScan's `embeddings_only` mode for scanning.

