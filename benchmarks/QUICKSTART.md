# Benchmark Quick Start Guide

## Overview

Create a RAG benchmark with adversarial hubs planted in a real Wikipedia corpus.

### Process:
1. **Download Wikipedia articles** - Real articles across diverse topics
2. **Chunk documents** - Split into 200-500 word chunks (like real RAG)
3. **Create embeddings** - Generate vectors using sentence-transformers
4. **Plant adversarial hubs** - Insert malicious hubs using 4 strategies:
   - **Geometric**: Average of diverse documents
   - **Multi-Centroid**: Multiple hubs targeting different clusters
   - **Gradient-Based**: Optimized with gradient descent
   - **Stealth**: Low similarity but consistent hub
5. **Run HubScan** - Detect the planted hubs
6. **Calculate metrics** - Precision, Recall, F1, FPR

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install sentence-transformers requests tqdm scikit-learn
```

### Step 2: Create Wikipedia Benchmark

```bash
cd /Users/ihabler/Desktop/HubnessDetector/benchmarks

# Small benchmark (30 articles, ~5000 chunks)
python3 create_wikipedia_benchmark.py \
  --size small \
  --output data/small/ \
  --model all-MiniLM-L6-v2
```

This will:
- Download 30 Wikipedia articles
- Chunk them into ~5000 chunks
- Create embeddings using sentence-transformers
- Save to `data/small/`

**Expected runtime**: 2-5 minutes

### Step 3: Plant Adversarial Hubs

```bash
# Plant hubs using all strategies (1% hub rate)
python3 plant_hubs.py \
  --dataset data/small/ \
  --strategy all \
  --rate 0.01
```

This will:
- Create ~50 adversarial hubs (1% of 5000)
- Use all 4 strategies (geometric, multi_centroid, gradient, stealth)
- Insert hubs into the embeddings
- Save ground truth labels

**Expected runtime**: 1-2 minutes

### Step 4: Run Benchmark

```bash
# Run HubScan and evaluate
python3 run_benchmark.py \
  --dataset data/small/ \
  --config configs/default.yaml \
  --output results/small/
```

This will:
- Run HubScan on the dataset with planted hubs
- Compare detected hubs to ground truth
- Calculate precision, recall, F1, FPR
- Analyze by strategy

**Expected runtime**: 3-5 minutes

### Step 5: View Results

```bash
cat results/small/benchmark_results.json
```

Results include:
- **Precision**: What fraction of detected hubs are real?
- **Recall**: What fraction of real hubs were detected?
- **F1 Score**: Harmonic mean
- **FPR**: False positive rate
- **Per-strategy metrics**: Which strategies are hardest to detect?

---

## Example Output

```
Metrics (HIGH only):
  Precision: 0.9500
  Recall: 0.7600
  F1: 0.8450
  FPR: 0.000100
  TP: 38, FP: 2, FN: 12

Metrics (HIGH + MEDIUM):
  Precision: 0.9200
  Recall: 0.9200
  F1: 0.9200
  FPR: 0.000400
  TP: 46, FP: 4, FN: 4

geometric:
  Hubs: 12
  Recall (HIGH): 0.9167
  Recall (ALL): 1.0000

multi_centroid:
  Hubs: 13
  Recall (HIGH): 0.7692
  Recall (ALL): 0.9231

gradient:
  Hubs: 12
  Recall (HIGH): 0.8333
  Recall (ALL): 0.9167

stealth:
  Hubs: 13
  Recall (HIGH): 0.5385
  Recall (ALL): 0.7692
```

This shows:
- **Geometric hubs**: Easiest to detect (91.7% recall)
- **Stealth hubs**: Hardest to detect (53.9% recall)
- **Overall**: 84.5% F1 score

---

## Different Strategies

### Single Strategy

Plant only geometric hubs:
```bash
python3 plant_hubs.py --dataset data/small/ --strategy geometric --rate 0.01
```

### Different Hub Rates

Test with more hubs (5% instead of 1%):
```bash
python3 plant_hubs.py --dataset data/small/ --strategy all --rate 0.05
```

### Custom Configuration

Create your own config in `configs/`:
```yaml
# configs/aggressive.yaml
thresholds:
  policy: hybrid
  hub_z: 3.0  # Lower threshold = more sensitive
  percentile: 0.05
```

Run with custom config:
```bash
python3 run_benchmark.py \
  --dataset data/small/ \
  --config configs/aggressive.yaml \
  --output results/aggressive/
```

---

## Directory Structure After Running

```
benchmarks/
├── data/
│   └── small/
│       ├── embeddings.npy           # Modified embeddings with hubs
│       ├── metadata.json            # Chunk metadata
│       ├── dataset_info.json        # Dataset information
│       └── ground_truth.json        # Ground truth labels
├── results/
│   └── small/
│       ├── benchmark_results.json   # Full results
│       ├── report.json              # HubScan report
│       └── report.html              # HubScan HTML report
└── configs/
    └── default.yaml                 # HubScan configuration
```

---

## Troubleshooting

### "sentence-transformers not installed"

```bash
pip install sentence-transformers
```

### "Wikipedia article not found"

The script will skip articles that don't exist. This is normal.

### Memory error

For large benchmarks, use a machine with more RAM or reduce the dataset size.

---

## Next Steps

1. Try different hub strategies
2. Test with different HubScan configurations
3. Create medium/large benchmarks
4. Analyze which strategies are hardest to detect
5. Optimize HubScan parameters for better detection

