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

# Small benchmark (~30 articles, ~800 chunks)
python3 create_wikipedia_benchmark.py \
  --size small \
  --output data/wikipedia_small/ \
  --model all-MiniLM-L6-v2 \
  --chunk-size 250 \
  --chunk-overlap 40
```

This will:
- Download ~30 Wikipedia articles
- Chunk them into ~800 chunks (250 words each, 40 word overlap)
- Create embeddings using sentence-transformers (384 dimensions)
- Save to `data/wikipedia_small/`

**Expected runtime**: 2-5 minutes
**Expected output**: ~28 articles, ~800 chunks

### Step 3: Plant Adversarial Hubs

```bash
# Plant hubs using all strategies (5% hub rate)
python3 plant_hubs.py \
  --dataset data/wikipedia_small/ \
  --strategy all \
  --rate 0.05
```

This will:
- Create ~36-40 adversarial hubs (5% of 800)
- Use all 4 strategies (geometric, multi_centroid, gradient, stealth)
- Insert hubs into the embeddings
- Save ground truth labels

**Expected runtime**: 30 seconds

### Step 4: Run Benchmark

```bash
# Run HubScan and evaluate
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/default.yaml \
  --output results/wikipedia/
```

This will:
- Run HubScan on the dataset with planted hubs
- Compare detected hubs to ground truth
- Calculate precision, recall, F1, FPR
- Analyze by strategy

**Expected runtime**: < 1 second (very fast!)

### Step 5: View Results

```bash
cat results/wikipedia/benchmark_results.json
```

Results include:
- **Precision**: What fraction of detected hubs are real?
- **Recall**: What fraction of real hubs were detected?
- **F1 Score**: Harmonic mean
- **FPR**: False positive rate
- **Per-strategy metrics**: Which strategies are hardest to detect?

See `WIKIPEDIA_RESULTS.md` for detailed analysis of results.

---

## Example Output (Real Wikipedia Data)

```
Metrics (HIGH only):
  Precision: 0.6750
  Recall: 0.7500
  F1: 0.7105
  FPR: 0.017310
  TP: 27, FP: 13, FN: 9

Metrics (HIGH + MEDIUM):
  Precision: 0.2647
  Recall: 0.7500
  F1: 0.3913
  FPR: 0.099867
  TP: 27, FP: 75, FN: 9

gradient_based_hub:
  Hubs: 9
  Recall (HIGH): 1.0000  # Perfect detection
  Recall (ALL): 1.0000

geometric_hub:
  Hubs: 9
  Recall (HIGH): 1.0000  # Perfect detection
  Recall (ALL): 1.0000

multi_centroid_hub:
  Hubs: 9
  Recall (HIGH): 1.0000  # Perfect detection
  Recall (ALL): 1.0000

```

This shows:
- **100% detection on all hub strategies**
- **Gradient-based hubs**: 100% recall (most sophisticated attacks)
- **Geometric hubs**: 100% recall
- **Multi-centroid hubs**: 100% recall
- **Overall**: 100% recall, 70.6% precision, 1.6% FPR

---

## Different Strategies

### Single Strategy

Plant only geometric hubs:
```bash
python3 plant_hubs.py --dataset data/wikipedia_small/ --strategy geometric --rate 0.05
```

### Different Hub Rates

Test with more hubs (10% instead of 5%):
```bash
python3 plant_hubs.py --dataset data/wikipedia_small/ --strategy all --rate 0.10
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

Test with different configurations:
```bash
# Aggressive (deeper search, more sensitive)
python3 run_benchmark.py \
  --dataset data/wikipedia_small/ \
  --config configs/aggressive.yaml \
  --output results/aggressive/
```

---

## Directory Structure After Running

```
benchmarks/
├── data/
│   └── wikipedia_small/
│       ├── embeddings.npy           # Modified embeddings with hubs
│       ├── metadata.json            # Chunk metadata
│       ├── dataset_info.json        # Dataset information
│       └── ground_truth.json        # Ground truth labels
├── results/
│   └── wikipedia/
│       ├── benchmark_results.json   # Full results
│       ├── report.json              # HubScan report
│       └── report.html              # HubScan HTML report
└── configs/
    ├── default.yaml                 # Default configuration (no weights)
    ├── aggressive.yaml              # Aggressive configuration
    └── no_weights.yaml              # Same as default (kept for reference)
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

