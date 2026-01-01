# Concept-Specific and Modality-Aware Hub Detection

This document explains HubScan's advanced detection capabilities for concept-targeted and cross-modal adversarial hubs.

<img src="images/detection-modes.png" alt="Detection Modes" width="600">

## Why This Matters

### Concept-Specific Attacks
Standard hubness detection aggregates hits across all queries. Adversarial hubs may target narrow semantic topics, evading global detection. Partitioning queries by concept and computing hubness within each topic surfaces localized attacks.

### Multimodal RAG Security
Multimodal systems using images and text face unique attack surfaces. Documents can be crafted to appear relevant to queries in one modality while semantically unrelated. Modality-aware detection tracks these cross-modal anomalies.

## Best Practices for Embedding Selection

Embedding model choice significantly impacts hubness detection effectiveness. Follow these guidelines to maximize detection accuracy.

### Preserve Semantic Margins

Choose embedding models that maintain clear separation between unrelated concepts:

- **Domain-specific models** outperform general-purpose models for specialized corpora
- **Fine-tuned models** on your data distribution preserve relevant distinctions
- **Higher-dimensional embeddings** often provide better separation (384+ dimensions recommended)

### For Multimodal Systems

HubScan supports the gold standard architecture with separate embedding spaces:

<img src="images/gold-standard-architecture.png" alt="Gold Standard Architecture" width="600">

- **Images**: Vision models (ResNet, DINOv2, ViT)
- **Text**: Text encoders (sentence-transformers, domain-specific models)
- **Parallel retrieval**: Query each modality index separately in parallel
- **Late fusion**: Normalize and merge candidates using RRF, weighted sum, or max
- **Optional unified index**: Use unified/cross-modal embedding index as recall backstop (not primary)

This architecture provides:
- Better semantic margins within each modality
- Clearer hubness signals
- Easier threshold tuning
- Production-ready parallel retrieval + fusion

### Gold Standard Configuration

```yaml
input:
  mode: multi_index
  multi_index:
    text_index_path: "text_index.index"
    text_embeddings_path: "text_embeddings.npy"
    image_index_path: "image_index.index"
    image_embeddings_path: "image_embeddings.npy"
  
  late_fusion:
    enabled: true
    fusion_method: rrf
    text_weight: 0.4
    image_weight: 0.4
    rrf_k: 60
  
  diversity:
    enabled: true
    min_distance: 0.3
```

HubScan automatically performs parallel retrieval and late fusion when `multi_index` mode is enabled with `late_fusion.enabled: true`.

### For Text-Only Systems

Recommended embedding models:
- sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- Domain-specific fine-tuned models
- E5 or BGE family models for retrieval

### Normalize Embeddings

Always normalize embeddings to unit length for cosine similarity:
- Prevents magnitude-based hubness
- Ensures consistent similarity distributions
- Required for accurate z-score calculations

## Concept-Aware Hub Detection

### How It Works

1. **Query Partitioning**: Queries are grouped into semantic concepts via:
   - Metadata labels (if available)
   - Clustering (MiniBatch K-Means on embeddings)
   - Hybrid: try metadata first, fallback to clustering

2. **Per-Concept Statistics**: For each concept, compute:
   - Hub rate: `hit_count(doc, concept) / query_count(concept)`
   - Robust z-score within the concept

3. **Contrastive Bucket Detection**: Flag documents where:
   - `hub_rate(doc, concept_c)` >> `median(hub_rate(doc, other_concepts))`
   - High concentration (Gini coefficient) in one concept

### Configuration

```yaml
detectors:
  concept_aware:
    enabled: true
    mode: hybrid  # metadata, clustering, or hybrid
    metadata_field: concept  # Field name in metadata
    num_concepts: 10  # Clusters for auto-detection
    concept_hub_z_threshold: 4.0
    seed: 42
```

### Best Practices

- Use metadata labels when available (more accurate than clustering)
- Set `num_concepts` based on your domain's natural topic diversity
- Lower `concept_hub_z_threshold` for stricter detection

## Modality-Aware Hub Detection

### How It Works

1. **Modality Tracking**: Documents and queries are tagged with modality:
   - From metadata (recommended)
   - Default fallback (e.g., "text")

2. **Cross-Modal Analysis**: Detect documents that:
   - Appear frequently in queries of different modality
   - Show asymmetric retrieval patterns

3. **Cross-Modal Penalty**: Apply scoring penalty for suspicious cross-modal behavior

### Configuration

```yaml
detectors:
  modality_aware:
    enabled: true
    mode: metadata  # metadata or default_text
    doc_modality_field: modality
    query_modality_field: modality
    cross_modal_penalty: 1.5
```

### Metadata Format

```json
{
  "doc_id": "doc_001",
  "modality": "image",
  "concept": "food",
  "text": "[IMAGE] A plate of spaghetti"
}
```

## Benchmark Results

### Wikipedia Benchmark (Text-Only)

Using sentence-transformers embeddings (all-MiniLM-L6-v2):

| Metric | HIGH | HIGH+MEDIUM |
|--------|------|-------------|
| Precision | 96% | 40% |
| Recall | 100% | 100% |
| F1 | 0.98 | 0.57 |

Per-strategy detection (HIGH recall):
- All strategies: 100%

### Multimodal Benchmark (Separate Embedding Spaces)

Using separate embedding spaces:
- Images: ResNet50 (2048-dim)
- Text: all-MiniLM-L6-v2 (384-dim)
- Strategy: effective (geometric + multi_centroid + cross_modal)

| Metric | HIGH | HIGH+MEDIUM |
|--------|------|-------------|
| Precision | 100% | 100% |
| Recall | 100% | 100% |
| F1 | 1.00 | 1.00 |

Per-strategy detection (HIGH recall):
- geometric_hub: 100%
- multi_centroid_hub: 100%
- cross_modal_hub: 100%

## API Usage

### Python SDK

```python
from hubscan import Scanner
from hubscan.config import Config

config = Config(
    input={"embeddings_path": "embeddings.npy", "metadata_path": "metadata.json"},
    detectors={
        "concept_aware": {
            "enabled": True,
            "mode": "hybrid",
            "num_concepts": 10,
        },
        "modality_aware": {
            "enabled": True,
            "doc_modality_field": "modality",
        },
    },
)

scanner = Scanner(config)
scanner.load_data()
result = scanner.scan()

# Access concept-specific results
for doc in result["high_risk_docs"]:
    print(f"Doc {doc['idx']}: concept={doc.get('top_concept')}")
```

### CLI

```bash
hubscan scan \
  --embeddings embeddings.npy \
  --metadata metadata.json \
  --enable-concept-aware \
  --num-concepts 10 \
  --enable-modality-aware \
  --output results/
```

## Custom Integration

### Adding Custom Concept Provider

```python
from hubscan.core.concepts import ConceptProvider, register_concept_provider

class DomainConceptProvider(ConceptProvider):
    def assign_concept(self, doc_idx, metadata):
        # Your domain logic
        return metadata.get("category_id", 0)

register_concept_provider("domain", DomainConceptProvider)

# Use in config
config = Config(
    detectors={"concept_aware": {"mode": "domain"}}
)
```

### Adding Custom Modality Resolver

```python
from hubscan.core.modalities import ModalityResolver, register_modality_resolver

class CustomModalityResolver(ModalityResolver):
    def resolve_modality(self, doc_idx, metadata):
        if metadata.get("has_image"):
            return "image"
        return "text"

register_modality_resolver("custom", CustomModalityResolver)
```

## Troubleshooting

### Low Detection Rates

1. Check embedding quality: Ensure embeddings preserve semantic structure
2. Verify concept assignment: Print concept distributions to check balance
3. Adjust thresholds: Lower `concept_hub_z_threshold` for stricter detection

### High False Positives

1. Increase thresholds: Raise `hub_z` threshold in config
2. Check for legitimate hubs: Some central documents are naturally popular
3. Review metadata: Ensure modality/concept fields are accurate

### Numerical Issues

If you see extreme scores:
- Ensure embeddings are normalized
- Check for NaN/Inf values in data
- The `robust_zscore` function has built-in numerical stability
