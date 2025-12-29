# Testing Guide

## Prerequisites

Install the project dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
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
pytest tests/test_hubness.py::test_hubness_detection -v
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

