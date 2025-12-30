# RAG Adversarial Hubness Benchmark

## Overview

This benchmark evaluates the effectiveness of HubScan in detecting adversarial hubs in real-world RAG systems built from actual documents (e.g., Wikipedia articles).

## Benchmark Approach

### 1. Document Collection
- Collect real Wikipedia articles across diverse topics
- Chunk documents for RAG (200-500 tokens per chunk)
- Create embeddings using a production embedding model

### 2. Adversarial Hub Planting Strategies

#### Strategy A: Geometric Hub (Center of Mass)
- Create a hub embedding as the weighted average of multiple diverse document embeddings
- This hub will be geometrically close to many queries
- **Detection difficulty**: Easy to Medium

#### Strategy B: Multi-Centroid Hub
- Create multiple hub variants that target different semantic clusters
- Each variant is optimized for a specific cluster of queries
- **Detection difficulty**: Medium

#### Strategy C: Gradient-Based Adversarial Hub
- Use gradient descent to optimize hub embedding
- Maximize retrieval probability across diverse queries
- **Detection difficulty**: Hard

#### Strategy D: Stealth Hub (Low Similarity)
- Create hub that appears at lower ranks but consistently
- Harder to detect with traditional methods
- **Detection difficulty**: Very Hard

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

### 5. Benchmark Variants

#### Small Benchmark
- 1,000 Wikipedia articles
- 5,000 chunks
- 5 adversarial hubs (1 per strategy)
- 1,000 test queries
- **Runtime**: ~5 minutes

#### Medium Benchmark
- 10,000 Wikipedia articles
- 50,000 chunks
- 25 adversarial hubs (5 per strategy)
- 10,000 test queries
- **Runtime**: ~30 minutes

#### Large Benchmark
- 100,000 Wikipedia articles
- 500,000 chunks
- 100 adversarial hubs (20 per strategy)
- 50,000 test queries
- **Runtime**: ~3 hours

## Implementation Plan

### Phase 1: Data Collection
1. Download Wikipedia articles (use Wikipedia API or dumps)
2. Preprocess and chunk documents
3. Generate embeddings (use sentence-transformers or OpenAI)
4. Save as benchmark dataset

### Phase 2: Hub Planting
1. Implement each planting strategy
2. Plant hubs at different rates (0.1%, 0.5%, 1%, 5%)
3. Save ground truth labels

### Phase 3: Benchmark Execution
1. Run HubScan with different configurations
2. Compare detected hubs to ground truth
3. Calculate metrics

### Phase 4: Analysis
1. Generate benchmark report
2. Analyze which hub strategies are hardest to detect
3. Identify optimal HubScan configurations

## Usage

```bash
# Generate benchmark dataset
python benchmarks/create_wikipedia_benchmark.py --size small --output benchmarks/data/small/

# Plant adversarial hubs
python benchmarks/plant_hubs.py --dataset benchmarks/data/small/ --strategy all --rate 0.01

# Run benchmark
python benchmarks/run_benchmark.py --dataset benchmarks/data/small/ --config benchmarks/configs/default.yaml

# Analyze results
python benchmarks/analyze_results.py --results benchmarks/results/small/
```

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── configs/                           # HubScan configurations for benchmark
│   ├── default.yaml
│   ├── fast_scan.yaml
│   └── deep_scan.yaml
├── data/                             # Benchmark datasets
│   ├── small/
│   ├── medium/
│   └── large/
├── results/                          # Benchmark results
├── create_wikipedia_benchmark.py     # Download and prepare Wikipedia data
├── plant_hubs.py                     # Plant adversarial hubs
├── run_benchmark.py                  # Run benchmark
├── analyze_results.py                # Analyze and report results
└── hub_strategies.py                 # Hub planting strategies
```

## Next Steps

1. Implement Wikipedia article downloader
2. Implement hub planting strategies
3. Create benchmark execution script
4. Generate baseline results

