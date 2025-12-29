# HubScan SDK Documentation

The HubScan SDK provides a simple, programmatic interface for detecting adversarial hubs in vector indices and RAG systems.

## Quick Start

```python
from hubscan.sdk import scan, Verdict

# Run a scan
results = scan(
    embeddings_path="data/embeddings.npy",
    metadata_path="data/metadata.json",
    k=20,
    num_queries=10000
)

# Get high-risk documents
from hubscan.sdk import get_suspicious_documents
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

### `quick_scan()`

Run a scan on in-memory embeddings (no file I/O).

```python
import numpy as np
from hubscan.sdk import quick_scan

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
from hubscan.sdk import scan_from_config

results = scan_from_config("config.yaml")
```

### `get_suspicious_documents()`

Extract suspicious documents from scan results.

```python
from hubscan.sdk import get_suspicious_documents, Verdict

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
from hubscan.sdk import explain_document

explanation = explain_document(results, doc_index=42)
if explanation:
    print(f"Risk Score: {explanation['risk_score']}")
    print(f"Hub Z-Score: {explanation['hubness']['hub_z']}")
```

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
)
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

