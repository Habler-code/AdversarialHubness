# CLI and SDK Summary

## ✅ What's Been Created

### 1. Enhanced CLI (`hubscan/cli.py`)

A comprehensive command-line interface with:

- **Rich formatting** (if `rich` package available, falls back gracefully)
- **Progress indicators** during scan execution
- **Beautiful tables** for results display
- **Color-coded verdicts** (HIGH=red, MEDIUM=yellow, LOW=green)
- **Multiple commands**:
  - `scan` - Run hubness detection scan
  - `build-index` - Build FAISS index from embeddings
  - `explain` - Explain why a document was flagged

**Usage:**
```bash
# After installation: pip install -e .
hubscan scan --config examples/toy_config.yaml
hubscan scan --config config.yaml --output custom_reports/
hubscan scan --config config.yaml --summary-only
hubscan explain --doc-id 42 --report reports/report.json
```

### 2. SDK Module (`hubscan/sdk.py`)

A simple, high-level SDK for programmatic usage:

**Main Functions:**
- `scan()` - Main scan function with simple parameters
- `quick_scan()` - Scan in-memory embeddings (no file I/O)
- `scan_from_config()` - Scan from YAML config file
- `get_suspicious_documents()` - Extract suspicious docs from results
- `explain_document()` - Get detailed explanation for a document

**Usage:**
```python
from hubscan.sdk import scan, get_suspicious_documents, Verdict

# Simple scan
results = scan(
    embeddings_path="data/embeddings.npy",
    k=20,
    num_queries=10000
)

# Get high-risk documents
high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
```

### 3. Documentation

- `docs/SDK.md` - Complete SDK documentation
- `docs/USAGE.md` - Usage guide for both CLI and SDK
- `examples/sdk_example.py` - Complete SDK examples

## Installation

To use the CLI:

```bash
# Install the package
pip install -e .

# Or install dependencies
pip install -r requirements.txt

# Then use CLI
hubscan --help
```

## Quick Start Examples

### CLI Example

```bash
# Generate toy data
python examples/generate_toy_data.py

# Run scan
hubscan scan --config examples/toy_config.yaml

# View results
open examples/reports/report.html
```

### SDK Example

```python
from hubscan.sdk import scan, Verdict

# Run scan
results = scan(
    embeddings_path="examples/toy_embeddings.npy",
    metadata_path="examples/toy_metadata.json",
    k=10,
    num_queries=100
)

# Get results
print(f"Runtime: {results['runtime']:.2f}s")
print(f"High-risk docs: {sum(1 for v in results['verdicts'].values() if v == Verdict.HIGH)}")
```

## Features

### CLI Features
- ✅ Rich console output with tables and progress bars
- ✅ Color-coded verdicts
- ✅ Summary-only mode
- ✅ Custom output directories
- ✅ Verbose logging option
- ✅ Document explanation command

### SDK Features
- ✅ Simple function-based API
- ✅ In-memory quick scan
- ✅ Config file support
- ✅ Result filtering and extraction
- ✅ Document explanation
- ✅ Custom configuration via kwargs

## Next Steps

1. **Install the package:**
   ```bash
   pip install -e .
   ```

2. **Try the CLI:**
   ```bash
   hubscan scan --config examples/toy_config.yaml
   ```

3. **Try the SDK:**
   ```python
   python examples/sdk_example.py
   ```

4. **Read the docs:**
   - `docs/SDK.md` - SDK documentation
   - `docs/USAGE.md` - Usage guide

