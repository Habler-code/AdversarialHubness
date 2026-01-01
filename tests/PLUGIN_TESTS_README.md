# Plugin System Tests

This directory contains comprehensive tests for HubScan's plugin system, which allows users to register custom ranking methods and detectors.

## Test Files

### `test_ranking_plugins.py`
Tests for the ranking method plugin system:
- **TestRankingMethodRegistry**: Tests for registry functionality (register, get, list)
- **TestBuiltinRankingMethods**: Tests for built-in ranking method implementations
- **TestRankingMethodIntegration**: Tests integration with detectors
- **TestRankingMethodErrorHandling**: Tests error handling

**Coverage:**
- Registry operations (register, get, list)
- Built-in methods (vector, hybrid, lexical, reranked)
- Custom method registration and execution
- Integration with detectors
- Error handling (missing methods, invalid inputs)

### `test_detector_plugins.py`
Tests for the detector registry system:
- **TestDetectorRegistry**: Tests for registry functionality
- **TestDetectorIntegration**: Tests integration with Scanner
- **TestBuiltinDetectors**: Tests built-in detector registration
- **TestDetectorErrorHandling**: Tests error handling

**Coverage:**
- Registry operations (register, get, list)
- Built-in detectors (hubness, cluster_spread, stability, dedup)
- Custom detector registration and instantiation
- Integration with Scanner
- Error handling (invalid classes, missing detectors)

### `test_plugin_integration.py`
Integration tests for the complete plugin system:
- **TestCustomRankingIntegration**: Tests custom ranking methods with Scanner
- **TestCustomDetectorIntegration**: Tests custom detectors
- **TestBackwardCompatibility**: Ensures existing functionality still works
- **TestPluginErrorHandling**: Tests error scenarios

**Coverage:**
- Custom ranking methods with Scanner
- Custom ranking methods with detectors
- Custom detectors with Scanner
- Backward compatibility (built-in methods/detectors still work)
- Config with custom_params
- Error handling for unknown methods

## Running Tests

Run all plugin tests:
```bash
pytest tests/test_ranking_plugins.py tests/test_detector_plugins.py tests/test_plugin_integration.py -v
```

Run specific test file:
```bash
pytest tests/test_ranking_plugins.py -v
pytest tests/test_detector_plugins.py -v
pytest tests/test_plugin_integration.py -v
```

Run specific test class:
```bash
pytest tests/test_ranking_plugins.py::TestRankingMethodRegistry -v
```

## Test Statistics

- **Total tests**: 34
- **Ranking plugin tests**: 13
- **Detector plugin tests**: 13
- **Integration tests**: 8

All tests pass

## Key Test Scenarios

### Ranking Method Tests
1. Built-in methods are registered automatically
2. Custom methods can be registered and retrieved
3. Custom methods execute correctly with VectorIndex
4. Custom methods work with detectors
5. Error handling for missing/invalid methods
6. Warning when overwriting existing methods

### Detector Tests
1. Built-in detectors are registered automatically
2. Custom detectors can be registered and instantiated
3. Custom detectors work with Scanner
4. Error handling for invalid detector classes
5. Disabled detectors return zero scores

### Integration Tests
1. Custom ranking methods work with Scanner
2. Custom ranking methods work with detectors
3. Custom detectors can be instantiated and used
4. Backward compatibility maintained
5. Config supports custom_params
6. Unknown methods raise appropriate errors

## Mock Classes

The tests use mock classes to avoid dependencies:
- `MockRankingMethod`: Simple ranking method for testing
- `MockDetector`: Simple detector for testing

These mocks demonstrate the plugin interface without requiring complex implementations.

