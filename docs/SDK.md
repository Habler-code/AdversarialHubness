# HubScan SDK Documentation

The HubScan SDK provides a simple, programmatic interface for detecting adversarial hubs in vector indices and RAG systems.

## Quick Start

```python
from hubscan import scan, Verdict

# Run a scan
results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    k=20,
    num_queries=10000
)

# Get high-risk documents
from hubscan import get_suspicious_documents
high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
```

## Core Functions

### `scan()`

Main function for running scans with simple parameters.

```python
results = scan(
    embeddings_path="embeddings.npy",  # Required (or index_path)
    index_path="index.index",           # Alternative to embeddings_path
    metadata_path="metadata.json",      # Optional
    config_path="config.yaml",          # Optional (overrides other params)
    output_dir="reports/",              # Output directory
    k=20,                               # Number of nearest neighbors
    num_queries=10000,                  # Number of queries to sample
)
```

**Returns:** Dictionary with:
- `json_report`: Full JSON report
- `html_report`: HTML report string
- `detector_results`: Raw detector results
- `combined_scores`: Combined risk scores array
- `verdicts`: Verdict dictionary mapping doc indices to Verdict enum
- `runtime`: Runtime in seconds
- `detection_metrics`: Detection performance metrics (if ground truth available)

### `quick_scan()`

Run a scan on in-memory embeddings (no file I/O).

```python
import numpy as np
from hubscan import quick_scan

embeddings = np.random.randn(1000, 128).astype(np.float32)
results = quick_scan(
    embeddings=embeddings,
    k=10,
    num_queries=100
)
```

### `scan_from_config()`

Run a scan from a YAML configuration file.

```python
from hubscan import scan_from_config

results = scan_from_config("config.yaml")
```

### `extract_embeddings()`

Extract embeddings from a vector database using the Scanner class.

```python
from hubscan import Config, Scanner

# Load configuration
config = Config.from_yaml("config.yaml")
scanner = Scanner(config)
scanner.load_data()

# Extract embeddings
embeddings, ids = scanner.extract_embeddings(
    output_path="embeddings.npy",  # Optional: save to file
    batch_size=1000,                # Vectors per batch
    limit=None                      # None = extract all
)

print(f"Extracted {len(embeddings)} embeddings")
print(f"Shape: {embeddings.shape}")
```

**Parameters:**
- `output_path` (str, optional): Path to save embeddings (.npy file)
- `batch_size` (int): Number of vectors to retrieve per batch (default: 1000)
- `limit` (int, optional): Maximum number of vectors to extract (None = all)

**Returns:**
- `embeddings` (np.ndarray): Array of shape (N, D) containing all vectors
- `ids` (List[Any]): List of document IDs corresponding to each embedding

**Supported Databases:**
- FAISS: Direct reconstruction from index
- Pinecone: Query-based discovery + fetch API
- Qdrant: Scroll API pagination
- Weaviate: Cursor-based pagination

### `get_suspicious_documents()`

Extract suspicious documents from scan results.

```python
from hubscan import get_suspicious_documents, Verdict

# Get high-risk documents
high_risk = get_suspicious_documents(
    results,
    verdict=Verdict.HIGH,
    top_k=10
)

# Get all suspicious documents
all_suspicious = get_suspicious_documents(results)
```

### `explain_document()`

Get detailed explanation for why a document was flagged.

```python
from hubscan import explain_document

explanation = explain_document(results, doc_index=42)
if explanation:
    print(f"Risk Score: {explanation['risk_score']}")
    print(f"Hub Z-Score: {explanation['hubness']['hub_z']}")
```

### Ranking and Reranking Examples

The `scan()` function supports various ranking methods and reranking options:

```python
from hubscan import scan

# Hybrid search (combines vector + lexical)
results = scan(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    ranking_method="hybrid",
    hybrid_alpha=0.6,  # 60% vector, 40% lexical
    k=20,
    num_queries=10000
)

# Lexical search only
results = scan(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    ranking_method="lexical",
    k=20
)

# Vector search with reranking (cross-encoder for better accuracy)
results = scan(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    rerank=True,
    rerank_method="cross_encoder",
    rerank_top_n=100,
    k=20
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
    text_weight=0.4,
    image_weight=0.4,
    modality_aware=True,
    k=20
)
```

**Ranking Parameters (all in `scan()`):**
- `ranking_method`: One of `"vector"`, `"hybrid"`, `"lexical"` (default: "vector")
- `hybrid_alpha`: Weight for vector search in hybrid mode (0.0-1.0, default: 0.5)
- `query_texts_path`: Path to query texts file (required for lexical/hybrid)
- `hybrid_backend`: Hybrid search backend - `"client_fusion"`, `"native_sparse"`, or `"auto"` (default: "auto")
- `lexical_backend`: Lexical scoring algorithm - `"bm25"` or `"tfidf"` (default: "bm25")
- `text_field`: Metadata field containing document text (default: "text")

**Reranking Parameters:**
- `rerank`: Enable reranking as post-processing (default: False)
- `rerank_method`: Reranking method - `"default"` or `"cross_encoder"` (default: "default")
- `rerank_top_n`: Number of candidates to retrieve before reranking (default: 100)
- `rerank_params`: Optional dictionary of custom parameters (e.g., model name)

**Multi-Index Parameters:**
- `text_index_path`: Path to text index file (for multi-index mode)
- `text_embeddings_path`: Path to text embeddings file (for multi-index mode)
- `image_index_path`: Path to image index file (for multi-index mode)
- `image_embeddings_path`: Path to image embeddings file (for multi-index mode)
- `late_fusion`: Enable late fusion of multi-index results (default: False)
- `fusion_method`: Late fusion method - `"rrf"`, `"weighted_sum"`, or `"max"` (default: "rrf")
- `text_weight`: Weight for text index in fusion (default: 0.4)
- `image_weight`: Weight for image index in fusion (default: 0.4)

### `compare_ranking_methods()`

Compare detection performance across multiple ranking methods.

```python
from hubscan import compare_ranking_methods

comparison = compare_ranking_methods(
    embeddings_path="data/embeddings.npy",
    query_texts_path="data/queries.json",
    methods=["vector", "hybrid", "lexical"],
    k=20,
    num_queries=10000
)

# Access results for each method
for method, method_results in comparison["results"].items():
    print(f"{method}: {len(method_results['verdicts'])} suspicious docs")
    
    # Access detection metrics if ground truth available
    if comparison.get("comparison") and method in comparison["comparison"]:
        metrics = comparison["comparison"][method]
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall: {metrics.get('recall', 0):.3f}")
        print(f"  F1: {metrics.get('f1', 0):.3f}")
```

**Returns:** Dictionary with:
- `methods`: List of methods tested
- `results`: Dictionary mapping method names to scan results
- `comparison`: Optional comparison metrics (if ground truth available)

## Advanced Usage

### Custom Configuration

```python
results = scan(
    embeddings_path="embeddings.npy",
    k=20,
    num_queries=5000,
    # Custom detector settings
    detectors__hubness__enabled=True,
    detectors__cluster_spread__enabled=True,
    detectors__stability__enabled=False,
    # Custom thresholds
    thresholds__policy="hybrid",
    thresholds__hub_z=6.0,
    thresholds__percentile=0.001,
    # Method-specific thresholds (optional)
    thresholds__method_specific={
        "vector": {"hub_z": 6.0, "percentile": 0.012},
        "hybrid": {"hub_z": 5.0, "percentile": 0.02},
        "lexical": {"hub_z": 6.0, "percentile": 0.012}
    },
    # Reranking configuration (optional)
    scan__ranking__rerank=True,
    scan__ranking__rerank_method="default",
    scan__ranking__rerank_top_n=100,
)

# Note: For lexical ranking, cluster_spread and stability detectors are automatically skipped
# Only hubness and dedup detectors are used (cluster spread requires semantic clustering,
# stability requires query embeddings to perturb)

# Note: For lexical ranking, cluster_spread and stability detectors are automatically skipped
# Only hubness and dedup detectors are used (cluster spread requires semantic clustering,
# stability requires query embeddings to perturb)
```

### Working with Results

```python
# Access raw results
verdicts = results["verdicts"]
combined_scores = results["combined_scores"]

# Count verdicts
high_count = sum(1 for v in verdicts.values() if v == Verdict.HIGH)
medium_count = sum(1 for v in verdicts.values() if v == Verdict.MEDIUM)
low_count = sum(1 for v in verdicts.values() if v == Verdict.LOW)

# Get top risk scores
import numpy as np
top_indices = np.argsort(combined_scores)[-10:][::-1]
```

## Examples

See `examples/sdk_example.py` for complete examples.

## API Reference

Full API documentation is available in the code. Key classes:

- `Config`: Configuration management
- `Scanner`: Main scanner class
- `Verdict`: Enum for risk levels (LOW, MEDIUM, HIGH)
- `HubnessDetector`, `ClusterSpreadDetector`, etc.: Individual detectors

