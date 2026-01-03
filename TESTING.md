# Testing Guide

## Prerequisites

Install the project dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt

# Install development/testing dependencies
pip install -r requirements-dev.txt
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
# Run hubness detector tests
pytest tests/test_hubness.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

### Run Specific Test Function

```bash
# Test hubness detection
pytest tests/test_hubness.py::test_hubness_detection -v

# Test ranking methods
pytest tests/test_hubness.py::test_hubness_detection_with_hybrid_search -v
pytest tests/test_adapters.py::test_faiss_adapter_hybrid_search -v

# Test metrics
pytest tests/test_metrics.py::TestRankingMetrics -v
pytest tests/test_metrics.py::TestDetectionMetrics -v
```

### Run with Coverage

```bash
pytest tests/ --cov=hubscan --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run Tests with Short Traceback

```bash
pytest tests/ --tb=short
```

## Test Categories

### Core Functionality Tests
- `test_hubness.py`: Hubness detection with various ranking methods
- `test_adapters.py`: Vector database adapter implementations
- `test_vector_index.py`: VectorIndex interface and methods
- `test_config.py`: Configuration management and validation

### Ranking Methods Tests
- `test_hubness.py`: Tests for hybrid and lexical search
- `test_adapters.py`: Adapter implementations for ranking methods
- `test_vector_index.py`: VectorIndex interface methods (search_hybrid, search_lexical, search_reranked)
- Note: Reranking is a post-processing step that can be applied to any ranking method, not a ranking method itself

### Metrics Tests
- `test_metrics.py`: Ranking quality metrics (NDCG, MRR, MAP, Precision@k, Recall@k)
- `test_metrics.py`: Detection performance metrics (AUC-ROC, AUC-PR, confusion matrix)

### Integration Tests
- `test_integration.py`: End-to-end scan workflows
- `test_sdk.py`: SDK function tests
- `test_cli.py`: CLI command tests

## Test Structure

- `tests/test_hubness.py` - Unit tests for hubness detection
- `tests/test_integration.py` - Integration tests for end-to-end scanning

## Writing New Tests

1. Create test files in the `tests/` directory
2. Name test files with `test_` prefix (e.g., `test_detector.py`)
3. Name test functions with `test_` prefix
4. Use pytest fixtures for common setup

Example:

```python
import pytest
from hubscan.core.detectors.hubness import HubnessDetector

def test_my_feature():
    """Test description."""
    # Your test code here
    assert True
```

## Continuous Integration

Tests are configured to run with:
- Maximum 1 failure before stopping (`--maxfail=1`)
- Test paths: `tests/`
- Python path: `.` (current directory)

See `pyproject.toml` for pytest configuration.

