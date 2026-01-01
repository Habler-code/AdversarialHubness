# Benchmark Scripts

Scripts for generating benchmark datasets, planting adversarial hubs, and running evaluations.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `create_wikipedia_benchmark.py` | Generate Wikipedia RAG dataset with real categories |
| `create_multimodal_benchmark.py` | Generate multimodal (image+text) dataset with separate embedding spaces |
| `hub_strategies.py` | Adversarial hub planting strategies |
| `plant_hubs.py` | Inject adversarial hubs into datasets |
| `run_benchmark.py` | Execute benchmarks and compute metrics |

## Usage

### Create Wikipedia Dataset

```bash
python create_wikipedia_benchmark.py \
    --size small \
    --output ../data/wikipedia/
```

Options:
- `--size`: Dataset size (small, medium, large)
- `--output`: Output directory
- `--seed`: Random seed for reproducibility

### Create Multimodal Dataset

```bash
python create_multimodal_benchmark.py \
    --output ../data/multimodal/ \
    --max-samples 300
```

This creates a dataset with separate embedding spaces:
- Images: ResNet50 embeddings (2048-dim)
- Text: Sentence-transformer embeddings (384-dim)

Options:
- `--output`: Output directory
- `--max-samples`: Maximum number of samples
- `--text-model`: Text embedding model (default: all-MiniLM-L6-v2)

### Plant Hubs

```bash
python plant_hubs.py \
    --dataset ../data/wikipedia/ \
    --strategy all \
    --rate 0.05
```

Options:
- `--dataset`: Input dataset directory
- `--strategy`: Hub strategy (geometric, multi_centroid, gradient, lexical, concept_specific, cross_modal, all)
- `--rate`: Percentage of documents to convert to hubs
- `--output`: Output directory (optional, defaults to in-place)

### Run Benchmark

```bash
python run_benchmark.py \
    --dataset ../data/wikipedia/ \
    --config ../configs/concept_modality.yaml \
    --output ../results/wikipedia/
```

Options:
- `--dataset`: Dataset directory
- `--config`: HubScan config file
- `--output`: Results output directory
- `--enable-concept-aware`: Enable concept-aware detection
- `--enable-modality-aware`: Enable modality-aware detection
- `--ranking-methods`: Comma-separated list of ranking methods

## Hub Strategies

### GeometricHubStrategy
Creates hubs as weighted averages of diverse documents.
- Difficulty: Easy to Medium
- Use case: Basic adversarial testing

### MultiCentroidHubStrategy  
Creates hubs targeting multiple semantic clusters.
- Difficulty: Medium
- Use case: Testing cluster-based detection

### GradientBasedHubStrategy
Optimizes hub embeddings via gradient descent.
- Difficulty: Hard
- Use case: Testing against sophisticated attacks

### LexicalHubStrategy
Creates hubs optimized for keyword/BM25 retrieval.
- Difficulty: Medium
- Use case: Testing lexical-aware detection

### ConceptSpecificHubStrategy
Creates hubs localized to specific concepts.
- Difficulty: Hard for global detection, Easy for concept-aware
- Use case: Testing concept-aware detection

### CrossModalHubStrategy
Creates hubs exploiting cross-modal retrieval.
- Difficulty: Hard for single-modal, Easy for modality-aware
- Use case: Testing modality-aware detection

## Output Format

### Dataset Files
- `embeddings.npy` - (N, D) float32 array
- `metadata.json` - Document metadata with concept/modality
- `query_texts.json` - Query texts for lexical search
- `ground_truth.json` - Hub labels for evaluation
- `dataset_info.json` - Dataset summary

### Results Files
- `benchmark_results.json` - Metrics and per-strategy results
- `report.json` - Full HubScan JSON report
- `report.html` - Interactive HTML report
