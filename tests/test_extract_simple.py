#!/usr/bin/env python3
"""Simple test for FAISS extraction only."""

import numpy as np
import faiss
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from file to avoid __init__ issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "faiss_adapter",
    Path(__file__).parent.parent / "hubscan" / "core" / "io" / "adapters" / "faiss_adapter.py"
)
faiss_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faiss_adapter)

FAISSIndex = faiss_adapter.FAISSIndex

# Import normalize
from hubscan.utils.metrics import normalize_vectors

# Create test embeddings
n, dim = 50, 128
embeddings = normalize_vectors(np.random.randn(n, dim).astype(np.float32))

# Create FAISS index
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Wrap in adapter
faiss_index = FAISSIndex(index)

# Extract embeddings
print("Testing FAISS extraction...")
extracted, ids = faiss_index.extract_embeddings()

print(f'✓ Extracted {len(extracted)} embeddings')
print(f'  Shape: {extracted.shape}')
print(f'  IDs: {len(ids)}')
print(f'  Match: {np.allclose(embeddings, extracted, atol=1e-5)}')

if np.allclose(embeddings, extracted, atol=1e-5):
    print("✓ Test PASSED")
    sys.exit(0)
else:
    print("✗ Test FAILED")
    sys.exit(1)

