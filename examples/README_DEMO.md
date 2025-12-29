# Adversarial Hub Detection Demo

This demo provides a complete end-to-end demonstration of HubScan's adversarial hub detection capabilities. It shows how to generate documents, create embeddings, plant an adversarial hub, and detect it using HubScan.

## Overview

The demo (`adversarial_hub_demo.py`) demonstrates:

1. **Document Generation**: Creates a corpus of 50 documents on various topics
2. **RAG Chunking**: Splits documents into chunks suitable for retrieval-augmented generation
3. **Embedding Creation**: Generates 128-dimensional embeddings for all chunks
4. **Adversarial Hub Planting**: Artificially creates a malicious chunk embedding optimized to appear in top-k results for many diverse queries
5. **Hub Detection**: Runs HubScan to detect the adversarial hub
6. **Hub Identification**: Shows exactly which chunk/document is the adversarial hub with detailed metrics

## Quick Start

### Run the Complete Demo

```bash
python examples/adversarial_hub_demo.py
```

This will:
- Generate 50 documents and split into 159 chunks
- Create embeddings and plant an adversarial hub at chunk index 79
- Run HubScan with 300 queries
- Display detection results showing the adversarial hub was found with HIGH verdict

### Expected Output

```
================================================================================
HubScan Adversarial Hub Detection Demo
================================================================================

[Step 1] Generating documents...
Generated 50 documents

[Step 2] Splitting documents into chunks for RAG...
Created 159 chunks from 50 documents

[Step 3] Creating embeddings for chunks...
Created embeddings: shape (159, 128)

[Step 4] Planting adversarial hub...
Adversarial hub planted at chunk index 79

[Step 5] Saving data...
Saved embeddings to examples/demo_data/chunk_embeddings.npy

[Step 6] Running HubScan to detect adversarial hub...
Scan completed in 0.06 seconds

[Step 7] Identifying adversarial hub...
FOUND: Adversarial hub detected at chunk index 79 (Risk Score: 5.9430, Verdict: HIGH)

SUCCESS: HubScan identified the adversarial hub!
  Chunk Index: 79
  Hub Z-Score: 10.12
  Hub Rate: 0.2100 (21% of queries)
  Hits: 63 out of 300 queries
```

## Understanding the Results

### Detection Metrics

When the demo runs successfully, you'll see:

- **Chunk Index**: 79 (the adversarial hub)
- **Risk Score**: 5.9430 (combined suspiciousness score)
- **Verdict**: HIGH (highly suspicious)
- **Hub Z-Score**: 10.12 (extremely unusual - 10+ standard deviations above median)
- **Hub Rate**: 0.2100 (21% of queries retrieve this chunk)
- **Hits**: 63 (number of queries that retrieved chunk 79)

### Why These Numbers Matter

- **Hub Z-Score of 10.12**: Normal chunks have z-scores between -2 and 2. A score of 10.12 indicates this chunk is statistically extremely unusual.
- **Hub Rate of 21%**: Normal chunks appear in top-k for 2-5% of queries. 21% is 4-10x higher than normal.
- **63 Hits**: Out of 300 queries, 63 retrieved chunk 79 in their top-10 results, indicating it's close to many diverse query vectors.

## Usage Options

### Option 1: Python Script (Recommended)

```bash
python examples/adversarial_hub_demo.py
```

### Option 2: CLI After Data Generation

```bash
# First run the demo script to generate data
python examples/adversarial_hub_demo.py

# Then run HubScan with the demo config
hubscan scan --config examples/demo_config.yaml

# Explain a specific chunk
hubscan explain --doc-id 79 --report examples/demo_data/reports/report.json
```

### Option 3: SDK Programmatically

```python
from hubscan.sdk import scan, get_suspicious_documents, explain_document, Verdict

# Run scan
results = scan(
    embeddings_path="examples/demo_data/chunk_embeddings.npy",
    metadata_path="examples/demo_data/chunk_metadata.json",
    k=10,
    num_queries=300
)

# Get high-risk chunks
high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH)

# Explain the adversarial hub
for doc in high_risk:
    explanation = explain_document(results, doc["doc_index"])
    print(f"Chunk {doc['doc_index']}: Risk={explanation['risk_score']:.4f}")
    print(f"  Hub Z-Score: {explanation['hubness']['hub_z']:.2f}")
    print(f"  Hub Rate: {explanation['hubness']['hub_rate']:.4f}")
```

## Key Concepts Explained

### What is an Adversarial Hub?

An adversarial hub is a document/chunk embedding that:

- **Appears frequently**: Shows up in top-k results for many diverse queries
- **Spans clusters**: Retrieved by queries from many different semantic clusters
- **Is artificially created**: Optimized to manipulate retrieval results
- **Is statistically anomalous**: Hub rate is 5-10+ standard deviations above normal

### How HubScan Detects It

HubScan uses multiple detection strategies:

1. **Hubness Detection**: Counts how many queries retrieve each chunk and computes robust z-scores
2. **Cluster Spread Analysis**: Analyzes which semantic clusters retrieve each chunk (high entropy = suspicious)
3. **Stability Testing**: Checks if retrieval is consistent under query perturbations
4. **Deduplication**: Identifies boilerplate/duplicate content

### Why This Matters

Adversarial hubs can be exploited to:

- **Inject malicious content**: Force unwanted content into RAG responses
- **Manipulate rankings**: Make certain documents appear more frequently
- **Bypass filters**: Evade content moderation systems
- **Degrade performance**: Reduce retrieval quality and user experience

## Customization

You can modify the demo to experiment with different scenarios:

### Change Hub Strength

Edit `adversarial_hub_demo.py`:

```python
chunks, embeddings, hub_info = plant_adversarial_hub(
    chunks, embeddings, hub_chunk_idx,
    num_query_clusters=50,  # More clusters = stronger hub
    strength=0.80  # Higher = closer to targets
)
```

### Adjust Detection Thresholds

Edit `demo_config.yaml`:

```yaml
thresholds:
  hub_z: 2.0  # Lower = more sensitive
  percentile: 0.10  # Top 10% instead of 0.1%
```

### Change Corpus Size

```python
documents = generate_documents(num_docs=100)  # More documents
```

## Files Generated

After running the demo, these files are created:

- `examples/demo_data/chunk_embeddings.npy` - Chunk embeddings (159 x 128)
- `examples/demo_data/chunk_metadata.json` - Chunk metadata with adversarial marker
- `examples/demo_data/chunk_index.index` - FAISS index (if built)
- `examples/demo_data/reports/report.json` - Full JSON report
- `examples/demo_data/reports/report.html` - HTML visualization

## Troubleshooting

### Hub Not Detected as HIGH

If the adversarial hub is not flagged as HIGH:

1. **Increase hub strength**: Raise `strength` parameter in `plant_adversarial_hub()`
2. **Lower thresholds**: Decrease `hub_z` threshold in config
3. **More queries**: Increase `num_queries` for better statistics
4. **Check metrics**: Hub may still be detected but with MEDIUM verdict

### Performance Issues

For faster execution:

- Reduce `num_queries` (e.g., 100 instead of 300)
- Use `flat` index type
- Disable expensive detectors (stability)

## Next Steps

After running the demo:

1. **Explore Reports**: Open `examples/demo_data/reports/report.html` in a browser
2. **Try Real Data**: Run HubScan on your own embeddings
3. **Tune Thresholds**: Adjust detection parameters for your use case
4. **Read Documentation**: See `docs/` for detailed guides

## Related Documentation

- [Main README](../README.md) - Project overview and installation
- [Usage Guide](../docs/USAGE.md) - CLI and SDK usage
- [SDK Documentation](../docs/SDK.md) - Programmatic API reference
