# HubScan Examples

Example code, demos, and configurations to help you get started with HubScan.

## Quick Start

### Basic Demo

Run the complete adversarial hub detection demo:

```bash
python examples/demos/adversarial_hub_demo.py
```

This will:
- Generate 50 documents and split into chunks
- Create embeddings and plant an adversarial hub
- Run HubScan to detect the hub
- Display detection results

### SDK Examples

Explore SDK usage patterns:

```bash
python examples/demos/sdk_example.py
```

### Custom Plugins

Learn how to extend HubScan:

```bash
# Custom detector
python examples/plugins/custom_detector_example.py

# Custom ranking method
python examples/plugins/custom_ranking_example.py

# Custom reranking
python examples/plugins/custom_reranking_example.py
```

## Demos

### adversarial_hub_demo.py

Complete end-to-end demo showing:
- Document generation and chunking
- Embedding creation
- Adversarial hub planting
- Hub detection and identification

**Output:** `demo_data/` directory with embeddings, metadata, and reports

### adversarial_hub_demo_multi_backend.py

Demonstrates HubScan with multiple vector database backends:
- FAISS (local)
- Pinecone (cloud)
- Qdrant (local/cloud)
- Weaviate (local/cloud)

**Note:** Requires API keys for cloud backends

### sdk_example.py

SDK usage patterns:
- Basic scan with file paths
- Quick scan with in-memory embeddings
- Document explanation
- Custom configuration
- Hybrid search (vector + lexical)
- Reranking for improved accuracy
- Scan from config file

## Plugins

### Custom Detector

Create your own detection algorithm:

```python
from hubscan.core.detectors import Detector, DetectorResult, register_detector

class MyCustomDetector(Detector):
    def detect(self, index, doc_embeddings, queries, k, metadata, **kwargs):
        # Your detection logic
        scores = ...
        return DetectorResult(scores=scores, metadata={})

register_detector("my_detector", MyCustomDetector)
```

See `plugins/custom_detector_example.py` for complete example.

### Custom Ranking Method

Add custom retrieval algorithms:

```python
from hubscan.core.ranking import register_ranking_method, RankingMethod

class MyRanking(RankingMethod):
    def search(self, index, query_vectors, query_texts, k, **kwargs):
        # Your ranking logic
        distances, indices = ...
        return distances, indices, {"method": "my_ranking"}

register_ranking_method("my_ranking", MyRanking())
```

See `plugins/custom_ranking_example.py` for complete example.

### Custom Reranking

Implement custom reranking:

```python
from hubscan.core.reranking import register_reranking_method

def my_reranker(candidates, query, **kwargs):
    # Your reranking logic
    reranked = ...
    return reranked

register_reranking_method("my_reranker", my_reranker)
```

See `plugins/custom_reranking_example.py` for complete example.

## Configuration Files

| Config | Description |
|--------|-------------|
| `toy_config.yaml` | Basic configuration for toy dataset |
| `demo_config.yaml` | Optimized for adversarial hub demo |
| `pinecone_config.yaml` | Pinecone cloud configuration |
| `qdrant_config.yaml` | Qdrant local/cloud configuration |
| `weaviate_config.yaml` | Weaviate local/cloud configuration |

## Data Files

Pre-generated toy dataset for quick testing (in `data/` directory):
- `data/toy_embeddings.npy`: 1000 documents, 128-dimensional embeddings
- `data/toy_metadata.json`: Document metadata
- `data/toy_index.index`: Pre-built FAISS index

**Generate fresh toy data:**
```bash
python examples/scripts/generate_toy_data.py
```

## CLI Usage

```bash
# Basic scan with toy data
hubscan scan --config examples/configs/toy_config.yaml

# Demo scan
hubscan scan --config examples/configs/demo_config.yaml

# Explain a document
hubscan explain --doc-id 79 --report examples/demo_data/reports/report.json
```

## SDK Usage

### Basic Scan

```python
from hubscan import scan

results = scan(
    embeddings_path="examples/data/toy_embeddings.npy",
    metadata_path="examples/data/toy_metadata.json",
    k=20,
    num_queries=1000
)
```

### Scan from Config File

```python
from hubscan import scan

results = scan(
    config_path="examples/configs/toy_config.yaml",
    output_dir="examples/reports"
)
```

### Hybrid Search (Vector + Lexical)

```python
from hubscan import scan

results = scan(
    embeddings_path="examples/data/toy_embeddings.npy",
    metadata_path="examples/data/toy_metadata.json",  # Must have 'text' field
    query_texts_path="examples/data/query_texts.json",  # Required for hybrid
    ranking_method="hybrid",
    hybrid_alpha=0.7,  # 70% vector, 30% lexical
    hybrid_backend="client_fusion",  # Works with any vector DB
    lexical_backend="bm25",  # or "tfidf"
    k=20,
    num_queries=1000
)
```

### Scan with Reranking

```python
from hubscan import scan

results = scan(
    embeddings_path="examples/data/toy_embeddings.npy",
    metadata_path="examples/data/toy_metadata.json",
    query_texts_path="examples/data/query_texts.json",  # Required for cross-encoder
    rerank=True,
    rerank_method="cross_encoder",  # or "default"
    rerank_top_n=100,  # Retrieve top 100, then rerank to top k
    k=20,
    num_queries=1000
)
```

### Quick Scan (In-Memory)

```python
from hubscan import quick_scan
import numpy as np

embeddings = np.random.randn(500, 128).astype(np.float32)
results = quick_scan(embeddings=embeddings, k=10, num_queries=100)
```

## For Concept and Modality Testing

For concept-specific and modality-aware hub detection testing, use the benchmarks:

```bash
# Generate Wikipedia benchmark with real categories
python benchmarks/scripts/create_wikipedia_benchmark.py --size small

# Generate multimodal benchmark
python benchmarks/scripts/create_multimodal_benchmark.py --max-samples 300

# Plant hubs and run evaluation
python benchmarks/scripts/plant_hubs.py --dataset benchmarks/data/wikipedia/ --strategy all
python benchmarks/scripts/run_benchmark.py --dataset benchmarks/data/wikipedia/ --config benchmarks/configs/concept_modality.yaml
```

See [benchmarks/README.md](../benchmarks/README.md) for detailed benchmark documentation.

## Related Documentation

- [Main README](../README.md) - Project overview
- [Usage Guide](../docs/USAGE.md) - CLI and SDK reference
- [SDK Documentation](../docs/SDK.md) - API details
- [Plugin System](../docs/PLUGINS.md) - Extending HubScan
- [Benchmarks](../benchmarks/README.md) - Evaluation and testing
