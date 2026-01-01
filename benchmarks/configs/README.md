# Benchmark Configurations

HubScan configuration files for different benchmark scenarios.

## Available Configurations

### concept_modality.yaml

Balanced configuration for text datasets with concept-aware detection.

Use cases:
- Wikipedia and similar text datasets
- Datasets with semantic categories/concepts
- General-purpose concept-aware detection

Key settings:
```yaml
detectors:
  concept_aware:
    enabled: true
    mode: hybrid
    num_concepts: 10
    concept_hub_z_threshold: 4.0
  modality_aware:
    enabled: true
    cross_modal_penalty: 1.5

thresholds:
  hub_z: 3.5
  percentile: 0.02
```

### multimodal.yaml

Configuration for multimodal datasets with separate embedding spaces.

Use cases:
- Image-text multimodal datasets
- Datasets using ResNet + sentence-transformers
- Cross-modal hub detection

Key settings:
```yaml
detectors:
  concept_aware:
    enabled: true
    mode: metadata
    concept_hub_z_threshold: 3.0
  modality_aware:
    enabled: true
    cross_modal_penalty: 1.5

thresholds:
  hub_z: 4.0
  percentile: 0.03
```

## Configuration Options

### Concept-Aware Detection

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable concept-aware detection | `false` |
| `mode` | Provider mode: `metadata`, `query_clustering`, `doc_clustering`, `hybrid` | `hybrid` |
| `metadata_field` | Metadata field for concept labels | `concept` |
| `num_concepts` | Number of clusters for auto-clustering | `10` |
| `concept_hub_z_threshold` | Z-score threshold for concept hubs | `4.0` |

### Modality-Aware Detection

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable modality-aware detection | `false` |
| `mode` | Resolver mode: `metadata`, `default_text`, `hybrid` | `default_text` |
| `doc_modality_field` | Metadata field for document modality | `modality` |
| `cross_modal_penalty` | Score multiplier for cross-modal hubs | `1.5` |

### Hubness Detection

| Option | Description | Default |
|--------|-------------|---------|
| `use_rank_weights` | Weight hits by rank position | `true` |
| `use_distance_weights` | Weight hits by similarity score | `true` |
| `use_contrastive_delta` | Detect concept-targeted hubs | `true` |
| `use_bucket_concentration` | Detect concentrated hub patterns | `true` |

### Scoring Weights

| Weight | Description | Default |
|--------|-------------|---------|
| `hub_z` | Global hubness z-score | 0.6 |
| `cluster_spread` | Cluster spread score | 0.15 |
| `stability` | Stability under perturbation | 0.05 |
| `boilerplate` | Duplicate/boilerplate penalty | 0.2 |

## Creating Custom Configurations

1. Copy an existing config as a starting point:
   ```bash
   cp concept_modality.yaml my_config.yaml
   ```

2. Adjust thresholds based on your data:
   - Text embeddings: `hub_z` threshold 3.5-5.0
   - Multimodal embeddings: `hub_z` threshold 4.0-5.0
   - Concept signals: Adjust `concept_hub_z_threshold`
   - Cross-modal risk: Adjust `cross_modal_penalty`

3. Run benchmark to validate:
   ```bash
   python scripts/run_benchmark.py \
       --dataset data/your_dataset/ \
       --config configs/my_config.yaml \
       --output results/test/
   ```
