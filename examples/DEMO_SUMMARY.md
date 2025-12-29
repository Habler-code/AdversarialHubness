# Adversarial Hub Detection Demo - Summary

## Overview

This demo provides a complete end-to-end example showing how HubScan detects adversarial hubs in a RAG system.

## What the Demo Does

1. **Generates Documents**: Creates 50 documents on various topics
2. **Splits for RAG**: Splits documents into 159 chunks suitable for retrieval
3. **Creates Embeddings**: Generates 128-dimensional embeddings for all chunks
4. **Plants Adversarial Hub**: Artificially creates a malicious chunk (chunk_0079) that is optimized to appear in top-k results for many diverse queries
5. **Runs HubScan**: Scans the corpus to detect suspicious chunks
6. **Identifies Hub**: Shows exactly which chunk is the adversarial hub

## Results

The demo successfully detects the adversarial hub:

- **Chunk Index**: 79
- **Chunk ID**: chunk_0079
- **Document ID**: doc_0025
- **Risk Score**: 5.9430
- **Verdict**: HIGH
- **Hub Z-Score**: 10.12 (very suspicious!)
- **Hub Rate**: 0.2100 (21% of queries retrieved this chunk)
- **Hits**: 63 out of 300 queries

## Key Metrics Explained

### Hub Z-Score (10.12)
- Measures how unusual the hub rate is compared to other chunks
- A score of 10.12 means this chunk is extremely unusual
- Normal chunks typically have z-scores between -2 and 2

### Hub Rate (0.2100)
- Fraction of queries where this chunk appears in top-k results
- 21% means 63 out of 300 queries retrieved this chunk
- Normal chunks typically have hub rates < 5%

### Risk Score (5.9430)
- Combined score from all detectors
- HIGH verdict indicates this is a significant security concern

## How to Run

```bash
# Run the complete demo
python examples/adversarial_hub_demo.py

# Or use CLI
hubscan scan --config examples/demo_config.yaml
hubscan explain --doc-id 79 --report examples/demo_data/reports/report.json

# Or use SDK
python -c "
from hubscan.sdk import scan, explain_document
results = scan(
    embeddings_path='examples/demo_data/chunk_embeddings.npy',
    metadata_path='examples/demo_data/chunk_metadata.json',
    k=10,
    num_queries=300
)
explanation = explain_document(results, doc_index=79)
print(f'Chunk 79: Risk={explanation[\"risk_score\"]:.4f}, Verdict={explanation[\"verdict\"]}')
"
```

## Files Generated

- `examples/demo_data/chunk_embeddings.npy` - Chunk embeddings
- `examples/demo_data/chunk_metadata.json` - Chunk metadata with adversarial marker
- `examples/demo_data/reports/report.json` - Full JSON report
- `examples/demo_data/reports/report.html` - HTML visualization

## Understanding the Detection

The adversarial hub is detected because:

1. **High Hub Rate**: It appears in top-k results for 21% of queries (much higher than normal)
2. **High Z-Score**: The hub rate is statistically very unusual (z-score = 10.12)
3. **Cross-Cluster**: It spans multiple semantic clusters (detected by cluster spread detector)
4. **Stable**: It consistently appears across different query types

## Next Steps

After detecting an adversarial hub:

1. **Quarantine**: Remove the chunk from your index immediately
2. **Investigate**: Check the source document and embedding process
3. **Review**: Examine how this chunk was created and why it's a hub
4. **Prevent**: Update your embedding/ingestion pipeline to prevent similar issues

