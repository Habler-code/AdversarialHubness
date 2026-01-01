"""Simple test for FAISS extraction only."""

import numpy as np
import faiss
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
from hubscan.utils.metrics import normalize_vectors


def test_faiss_extraction_simple():
    """Test FAISS extraction with a simple example."""
    # Create test embeddings
    n, dim = 50, 128
    embeddings = normalize_vectors(np.random.randn(n, dim).astype(np.float32))

    # Create FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Wrap in adapter
    faiss_index = FAISSIndex(index)

    # Extract embeddings
    extracted, ids = faiss_index.extract_embeddings()

    # Verify results
    assert len(extracted) == n, f"Expected {n} embeddings, got {len(extracted)}"
    assert extracted.shape == embeddings.shape, f"Shape mismatch: {extracted.shape} vs {embeddings.shape}"
    assert len(ids) == n, f"Expected {n} IDs, got {len(ids)}"
    assert np.allclose(embeddings, extracted, atol=1e-5), "Extracted embeddings don't match original"

