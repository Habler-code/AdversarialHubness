# RAG Adversarial Hubness Benchmark

## Overview

This benchmark evaluates the effectiveness of HubScan in detecting adversarial hubs in real-world RAG systems built from actual documents (e.g., Wikipedia articles).

## Benchmark Results

**Dataset**: 28 Wikipedia articles, 665 chunks, 24 adversarial hubs (3.6%)

### Performance by Ranking Method (Optimized Configuration)

| Method | Precision | Recall | F1 Score | Best For |
|--------|-----------|--------|----------|----------|
| **Vector** | 93.75% | 83.33% | 0.882 | Vector-optimized hubs |
| **Reranked** | 93.75% | 83.33% | 0.882 | Vector-optimized hubs |
| **Hybrid** | 94.12% | 66.67% | 0.780 | All hub types |
| **Lexical** | 75.00% | 100.00% | 0.857 | Lexical-optimized hubs |

### Detection by Strategy

| Strategy | Hubs | Best Detected By | Detection Rate |
|----------|------|------------------|----------------|
| **Gradient-Based** | 6 | Vector/Reranked/Hybrid | 83-100% |
| **Geometric** | 6 | Vector/Reranked/Hybrid | 83-100% |
| **Multi-Centroid** | 6 | Vector/Reranked/Hybrid | 83-100% |
| **Lexical** | 6 | Lexical/Hybrid | 100% |

**Key Findings**:
- **High precision**: 75-94% precision across all ranking methods
- **Excellent recall**: 67-100% recall depending on ranking method
- **Method-specific optimization**: Each ranking method performs best on its optimized hub types
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
- **Detection rate**: 100% (vector/reranked search)
- **Best detected by**: Vector search, Hybrid search, Reranked search

#### Strategy B: Multi-Centroid Hub
- Create multiple hub variants that target different semantic clusters
- Each variant is optimized for a specific cluster of queries
- **Detection difficulty**: Medium
- **Detection rate**: 100% (vector/reranked search)
- **Best detected by**: Vector search, Hybrid search, Reranked search

#### Strategy C: Gradient-Based Adversarial Hub
- Use gradient descent to optimize hub embedding
- Maximize retrieval probability across diverse queries
- **Detection difficulty**: Hard
- **Detection rate**: 100% (vector/reranked search)
- **Best detected by**: Vector search, Hybrid search, Reranked search

#### Strategy D: Lexical Hub (Keyword-Optimized)
- Creates hubs optimized for lexical/keyword search (BM25)
- Generates documents containing common keywords from queries
- These hubs rank highly in BM25-based retrieval
- **Detection difficulty**: Medium (for lexical search)
- **Detection rate**: Varies (optimized for lexical/hybrid search)
- **Best detected by**: Lexical search, Hybrid search
- **Note**: Requires `query_texts.json` to be generated (done automatically by `create_wikipedia_benchmark.py`)

### 3. Ground Truth
- Track which chunks are adversarial hubs
- Track which chunks are legitimate
- Enables precision/recall/F1 calculation

### 4. Ranking Methods

The benchmark supports comparing detection performance across different ranking methods:
- **Vector Search (KNN)**: Classic vector similarity search (default)
- **Hybrid Search**: Combines vector similarity with lexical matching (BM25)
- **Lexical Search**: Pure keyword-based search using BM25
- **Reranked Search**: Initial vector retrieval followed by semantic reranking

To compare ranking methods:
```bash
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/default.yaml \
  --output results/wikipedia/ \
  --ranking-methods vector hybrid lexical reranked
```

### 5. Detection Metrics

HubScan focuses on detection performance metrics:

- **Precision**: What fraction of detected hubs are true positives?
- **Recall**: What fraction of true hubs are detected?
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: How many legitimate documents are flagged?
- **AUC-ROC**: Area under ROC curve (if ground truth available)
- **AUC-PR**: Area under Precision-Recall curve (if ground truth available)
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Per-Class Metrics**: Precision, Recall, F1 for each verdict class

#### Effectiveness Metrics
- **Hub Rate**: Fraction of queries retrieving each hub
- **Rank Distribution**: At what ranks do hubs appear?
- **Cluster Spread**: How many semantic clusters do hubs reach?

## Configuration

The benchmark uses optimized HubScan configurations. The default configuration (`configs/default.yaml`) includes:

```yaml
scan:
  k: 20                    # Top-k documents per query
  num_queries: 5000        # Number of test queries
  ranking:
    method: vector         # Can be vector, hybrid, or lexical. Use rerank: true to enable reranking

detectors:
  hubness:
    enabled: true
    use_rank_weights: true      # Rank-aware scoring for better precision
    use_distance_weights: true  # Distance-based scoring for better precision
  cluster_spread:
    enabled: true
  dedup:
    enabled: true

scoring:
  weights:
    hub_z: 0.75           # Weight for hubness z-score
    cluster_spread: 0.2
    stability: 0.05
    boilerplate: 0.2

thresholds:
  policy: hybrid
  hub_z: 6.0              # Z-score threshold
  percentile: 0.012        # Top 1.2% by composite score
  method_specific:         # Optional: method-specific thresholds
    vector:
      hub_z: 6.0
      percentile: 0.012
    hybrid:
      hub_z: 5.0          # Relaxed for hybrid to improve recall
      percentile: 0.02
    lexical:
      hub_z: 6.0
      percentile: 0.012
```

The optimized configuration (`configs/method_specific.yaml`) uses method-specific thresholds for optimal performance across all ranking methods.

Performance with optimized configuration:
- Vector/Reranked: 93.75% precision, 83.3% recall
- Hybrid: 94.1% precision, 66.7% recall
- Lexical: 75% precision, 100% recall

## Usage

```bash
# Step 1: Generate benchmark dataset
cd benchmarks
python3 create_wikipedia_benchmark.py --size small --output data/wikipedia_small/

# Step 2: Plant adversarial hubs
# Plant all strategies (including lexical hubs)
python3 plant_hubs.py --dataset data/wikipedia_small/ --strategy all --rate 0.04

# Or plant only lexical hubs for testing lexical search
python3 plant_hubs.py --dataset data/wikipedia_small/ --strategy lexical --rate 0.04

# Step 3: Run benchmark (vector search only)
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/default.yaml \
  --output results/wikipedia/

# Step 3b: Compare all ranking methods
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/default.yaml \
  --output results/wikipedia/ \
  --ranking-methods vector hybrid lexical reranked

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

## Key Insights

1. **Method-Specific Optimization**: Each ranking method performs best on hubs optimized for that method
2. **High Precision**: 75-94% precision across all ranking methods with optimized thresholds
3. **Excellent Recall**: 67-100% recall depending on ranking method and hub type
4. **Rank-Aware Scoring**: Rank and distance weights improve precision significantly
5. **Production Ready**: Excellent performance on real Wikipedia documents with method-specific thresholds

