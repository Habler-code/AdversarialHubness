# RAG Adversarial Hubness Benchmark

## Overview

This benchmark evaluates the effectiveness of HubScan in detecting adversarial hubs in real-world RAG systems built from actual documents (e.g., Wikipedia articles).

## Benchmark Results (Default Configuration)

**Dataset**: 28 Wikipedia articles, 665 chunks, 24 adversarial hubs (3.6%)

### Overall Performance
- **Precision**: 70.6%
- **Recall**: 100%
- **F1 Score**: 82.8%
- **False Positive Rate**: 1.6%
- **Runtime**: 0.17 seconds

### Detection by Strategy
| Strategy | Hubs | Recall (HIGH) | Detection |
|----------|------|---------------|-----------|
| **Gradient-Based** | 8 | 100% | Perfect |
| **Geometric** | 8 | 100% | Perfect |
| **Multi-Centroid** | 8 | 100% | Perfect |

**Key Findings**:
- **Perfect recall**: 100% detection across all hub strategies
- **High precision**: 70.6% with very low false positive rate (1.6%)
- **Fast**: 0.17 seconds for 665 documents
- **Production-ready**: Excellent performance on real Wikipedia data

## Benchmark Approach

### 1. Document Collection
- Collect real Wikipedia articles across diverse topics
- Chunk documents for RAG (250 words per chunk, 40 word overlap)
- Create embeddings using sentence-transformers (all-MiniLM-L6-v2)

### 2. Adversarial Hub Planting Strategies

#### Strategy A: Geometric Hub (Center of Mass)
- Create a hub embedding as the weighted average of multiple diverse document embeddings
- This hub will be geometrically close to many queries
- **Detection difficulty**: Easy to Medium
- **Detection rate**: 100%

#### Strategy B: Multi-Centroid Hub
- Create multiple hub variants that target different semantic clusters
- Each variant is optimized for a specific cluster of queries
- **Detection difficulty**: Medium
- **Detection rate**: 100%

#### Strategy C: Gradient-Based Adversarial Hub
- Use gradient descent to optimize hub embedding
- Maximize retrieval probability across diverse queries
- **Detection difficulty**: Hard
- **Detection rate**: 100%

### 3. Ground Truth
- Track which chunks are adversarial hubs
- Track which chunks are legitimate
- Enables precision/recall/F1 calculation

### 4. Metrics

#### Detection Metrics
- **Precision**: What fraction of detected hubs are true positives?
- **Recall**: What fraction of true hubs are detected?
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: How many legitimate documents are flagged?

#### Effectiveness Metrics
- **Hub Rate**: Fraction of queries retrieving each hub
- **Rank Distribution**: At what ranks do hubs appear?
- **Cluster Spread**: How many semantic clusters do hubs reach?

## Configuration

The benchmark uses the default HubScan configuration:

```yaml
scan:
  k: 20                    # Top-k documents per query
  num_queries: 5000        # Number of test queries

detectors:
  hubness:
    enabled: true
    use_rank_weights: false      # Binary counting for best performance
    use_distance_weights: false  # Binary counting for best performance
  cluster_spread:
    enabled: true
  dedup:
    enabled: true

thresholds:
  hub_z: 4.0              # Robust z-score threshold
  percentile: 0.05        # Top 5% by composite score
```

This configuration achieves:
- 100% recall on all adversarial hub types
- 70.6% precision
- 1.6% false positive rate
- Fast execution (< 0.2 seconds for 1000 documents)

## Usage

```bash
# Step 1: Generate benchmark dataset
cd benchmarks
python3 create_wikipedia_benchmark.py --size small --output data/wikipedia_small/

# Step 2: Plant adversarial hubs
python3 plant_hubs.py --dataset data/wikipedia_small/ --strategy all --rate 0.04

# Step 3: Run benchmark
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/default.yaml \
  --output results/wikipedia/

# Step 4: View results
cat results/wikipedia/benchmark_results.json
```

### Example Output

```
Metrics (HIGH only):
  Precision: 0.7059
  Recall: 1.0000
  F1: 0.8276
  FPR: 0.015601
  TP: 24, FP: 10, FN: 0

gradient_based_hub:
  Hubs: 8
  Recall (HIGH): 1.0000  # Perfect detection

geometric_hub:
  Hubs: 8
  Recall (HIGH): 1.0000  # Perfect detection

multi_centroid_hub:
  Hubs: 8
  Recall (HIGH): 1.0000  # Perfect detection
```

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── configs/                           # HubScan configurations
│   ├── default.yaml                   # Default config (no weights)
│   ├── aggressive.yaml                # Aggressive detection
│   └── no_weights.yaml                # Same as default
├── data/
│   └── wikipedia_small/               # Benchmark dataset
│       ├── embeddings.npy             # Document embeddings with hubs
│       ├── metadata.json              # Chunk metadata
│       ├── ground_truth.json          # Hub labels
│       └── dataset_info.json          # Dataset information
├── results/
│   └── wikipedia/                     # Benchmark results
│       ├── benchmark_results.json     # Metrics and analysis
│       ├── report.json                # HubScan report
│       └── report.html                # HubScan HTML report
├── create_wikipedia_benchmark.py      # Download and prepare Wikipedia data
├── plant_hubs.py                      # Plant adversarial hubs
├── run_benchmark.py                   # Run benchmark and calculate metrics
└── hub_strategies.py                  # Hub planting strategies
```

## Key Insights

1. **Perfect Detection**: HubScan achieves 100% recall on all adversarial hub types
2. **High Precision**: 70.6% precision with only 1.6% false positive rate
3. **Fast**: Processes 665 documents in 0.17 seconds
4. **Simple Works Best**: Binary counting (no rank/distance weights) outperforms complex weighting
5. **Production Ready**: Excellent performance on real Wikipedia documents

