# Plugin System: Custom Ranking Methods and Detectors

HubScan now supports a pluggable architecture that allows you to:

1. **Register custom ranking/retrieval algorithms** - Add your own retrieval methods beyond the built-in `vector`, `hybrid`, and `lexical` methods
2. **Register custom reranking methods** - Add your own post-processing reranking algorithms that can be applied to any ranking method
3. **Register custom detectors** - Add your own detection algorithms that work seamlessly with the existing pipeline
4. **Mix and match** - Use any combination of built-in and custom components

## Custom Ranking Methods

### Creating a Custom Ranking Method

A ranking method must implement the `RankingMethod` protocol. Here's a simple example:

```python
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from hubscan.core.ranking import RankingMethod, register_ranking_method
from hubscan.core.io.vector_index import VectorIndex

class MyCustomRanking:
    """Custom ranking method example."""
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform custom search.
        
        Args:
            index: VectorIndex instance
            query_vectors: Optional query embeddings (M, D)
            query_texts: Optional query texts (M,)
            k: Number of results to return
            **kwargs: Custom parameters from config
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Your custom retrieval logic here
        # Example: weighted combination of multiple searches
        distances, indices = index.search(query_vectors, k * 2)
        
        # Apply custom reranking/filtering
        # ... your logic ...
        
        metadata = {
            "ranking_method": "my_custom",
            "custom_param": kwargs.get("custom_param", "default")
        }
        
        return distances[:, :k], indices[:, :k], metadata

# Register it
register_ranking_method("my_custom", MyCustomRanking())
```

### Using Custom Ranking Methods in Config

Once registered, use your custom method in YAML config:

```yaml
scan:
  ranking:
    method: my_custom
    custom_params:
      custom_param: "value"
      other_param: 42
```

The `custom_params` dictionary will be passed as `**kwargs` to your `search()` method.

### Example: Weighted Vector Search

See `examples/custom_ranking_example.py` for complete examples including:
- Weighted vector search
- Custom reranking pipeline

## Custom Detectors

### Creating a Custom Detector

A detector must inherit from `Detector` and implement the `detect()` method:

```python
import numpy as np
from typing import Optional, Dict, Any
from hubscan.core.detectors import Detector, DetectorResult, register_detector
from hubscan.core.io.metadata import Metadata

class MyCustomDetector(Detector):
    """Custom detector example."""
    
    def __init__(self, enabled: bool = True, my_param: float = 1.0):
        super().__init__(enabled)
        self.my_param = my_param
    
    def detect(
        self,
        index: "VectorIndex",
        doc_embeddings: np.ndarray,
        queries: np.ndarray,
        k: int,
        metadata: Optional[Metadata] = None,
        **kwargs,
    ) -> DetectorResult:
        """
        Run custom detection.
        
        Args:
            index: VectorIndex instance
            doc_embeddings: Document embeddings (N, D)
            queries: Query embeddings (M, D)
            k: Number of nearest neighbors
            metadata: Optional document metadata
            **kwargs: Includes ranking_method, hybrid_alpha, query_texts, etc.
            
        Returns:
            DetectorResult with scores and optional metadata
        """
        if not self.enabled:
            return DetectorResult(scores=np.zeros(len(doc_embeddings)))
        
        # Your detection logic here
        # The ranking_method is available in kwargs
        ranking_method = kwargs.get("ranking_method", "vector")
        
        # Compute scores
        scores = np.zeros(len(doc_embeddings))
        # ... your detection logic ...
        
        result_metadata = {
            "detector_type": "my_custom",
            "my_param": self.my_param,
        }
        
        return DetectorResult(scores=scores, metadata=result_metadata)

# Register it
register_detector("my_custom", MyCustomDetector)
```

### Using Custom Detectors in Config

Once registered, use your custom detector in YAML config:

```yaml
detectors:
  my_custom:
    enabled: true
    my_param: 1.5

scoring:
  weights:
    my_custom: 0.2  # Add weight for your detector
```

### Example: Custom Score Detector

See `examples/custom_detector_example.py` for a complete example.

## Integration with Existing Pipeline

### Ranking Method Integration

Custom ranking methods work seamlessly with all detectors:

- Detectors automatically use the ranking method specified in config
- The ranking method receives all necessary parameters (`hybrid_alpha`, `rerank_top_n`, `custom_params`)
- Detectors can access ranking metadata through the search results

### Detector Integration

Custom detectors integrate with:

- **Scoring**: Add weights in `scoring.weights` config section
- **Thresholds**: Use method-specific thresholds if needed
- **Reports**: Results appear in JSON and HTML reports automatically

## Best Practices

1. **Register early**: Register custom methods/detectors before creating `Scanner` instances
2. **Error handling**: Validate inputs and provide clear error messages
3. **Metadata**: Include useful metadata in search results for debugging
4. **Documentation**: Document your custom parameters clearly
5. **Testing**: Test with small datasets before running on production data

## Available Built-in Methods

### Ranking Methods
- `vector`: Standard vector similarity search
- `hybrid`: Hybrid vector + lexical search
- `lexical`: Pure lexical/keyword search (BM25, TF-IDF)

### Reranking Methods
- `default`: Simple reranking that retrieves more candidates and returns top k
- Custom reranking methods can be registered and applied to any ranking method

### Detectors
- `hubness`: Reverse-kNN frequency detection
- `cluster_spread`: Multi-cluster proximity detection
- `stability`: Query perturbation stability detection
- `dedup`: Duplicate and boilerplate detection

## Listing Registered Components

```python
from hubscan.core.ranking import list_ranking_methods
from hubscan.core.reranking import list_reranking_methods
from hubscan.core.detectors import list_detectors

print("Ranking methods:", list_ranking_methods())
print("Reranking methods:", list_reranking_methods())
print("Detectors:", list_detectors())
```

## Complete Example

```python
# 1. Register custom ranking method
from hubscan.core.ranking import register_ranking_method

class MyRanking:
    def search(self, index, query_vectors, query_texts, k, **kwargs):
        distances, indices = index.search(query_vectors, k)
        metadata = {"ranking_method": "my_ranking"}
        return distances, indices, metadata

register_ranking_method("my_ranking", MyRanking())

# 2. Register custom detector
from hubscan.core.detectors import register_detector, Detector, DetectorResult

class MyDetector(Detector):
    def detect(self, index, doc_embeddings, queries, k, metadata=None, **kwargs):
        scores = np.ones(len(doc_embeddings)) * 0.5
        return DetectorResult(scores=scores)

register_detector("my_detector", MyDetector)

# 3. Use in config
config = """
scan:
  ranking:
    method: my_ranking
    custom_params:
      my_param: 42

detectors:
  my_detector:
    enabled: true

scoring:
  weights:
    my_detector: 0.3
"""

# 5. Run scan
from hubscan import Scanner, Config
scanner = Scanner(Config.from_yaml("config.yaml"))
scanner.load_data()
results = scanner.scan()
```

## Backward Compatibility

**Note**: Reranking has been refactored from a ranking method to a post-processing step. This is a breaking change for configs using `method: reranked`.

### Migration Guide

**Old configuration:**
```yaml
ranking:
  method: reranked
  rerank_top_n: 100
```

**New configuration:**
```yaml
ranking:
  method: vector  # or hybrid, lexical
  rerank: true
  rerank_method: default
  rerank_top_n: 100
```

All other configurations continue to work without modification:
- Existing configs using `vector`, `hybrid`, `lexical` work as before
- Built-in detectors work exactly as before
- Reranking can now be applied to any ranking method

