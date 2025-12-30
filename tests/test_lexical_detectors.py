#!/usr/bin/env python3
"""Test that lexical search properly skips irrelevant detectors."""

import sys
from pathlib import Path
import tempfile
import json

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import faiss
from hubscan import Config, Scanner
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex


def test_lexical_skips_detectors():
    """Test that cluster_spread and stability are skipped for lexical ranking."""
    
    print("Testing detector skipping for lexical ranking...\n")
    
    # Create a small test dataset
    num_docs = 50
    dim = 32
    
    embeddings = np.random.randn(num_docs, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create document texts for lexical search
    doc_texts = [f"Document {i} with some text content about topic {i % 10}" for i in range(num_docs)]
    
    # Create query texts
    query_texts = [
        "topic 0",
        "topic 1", 
        "document",
        "text content",
        "some topic"
    ] * 10  # 50 queries
    
    # Check if rank-bm25 is available
    try:
        import rank_bm25
        has_bm25 = True
    except ImportError:
        print("Warning: rank-bm25 not installed. Skipping test.")
        print("  Install with: pip install rank-bm25")
        return
    
    # Save embeddings temporarily
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npy', delete=False) as f:
        np.save(f.name, embeddings)
        embeddings_path = f.name
    
    # Save query texts temporarily
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(query_texts, f)
        query_texts_path = f.name
    
    # Create config with all detectors enabled
    config = Config()
    config.input.mode = "embeddings_only"
    config.input.embeddings_path = embeddings_path
    config.scan.k = 5
    config.scan.num_queries = 20
    config.scan.ranking.method = "lexical"
    config.scan.query_texts_path = query_texts_path
    config.detectors.hubness.enabled = True
    config.detectors.cluster_spread.enabled = True
    config.detectors.stability.enabled = True
    config.detectors.dedup.enabled = True
    
    # Create metadata with document texts
    metadata_dict = {
        "text": doc_texts
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata_dict, f)
        metadata_path = f.name
    
    config.input.metadata_path = metadata_path
    
    # Create scanner
    scanner = Scanner(config)
    
    # Load data
    scanner.load_data()
    
    # Run scan
    results = scanner.scan()
    
    # Check that cluster_spread and stability are NOT in results for lexical
    detector_results = results["detector_results"]
    
    print(f"\nDetector results keys: {list(detector_results.keys())}")
    
    # For lexical, should have hubness and dedup, but NOT cluster_spread or stability
    assert "hubness" in detector_results, "Hubness should be present"
    assert "dedup" in detector_results, "Dedup should be present"
    assert "cluster_spread" not in detector_results, "Cluster spread should be skipped for lexical"
    assert "stability" not in detector_results, "Stability should be skipped for lexical"
    
    print("\n✓ Lexical ranking correctly skips cluster_spread and stability detectors")
    
    # Test that vector ranking includes all detectors
    config.scan.ranking.method = "vector"
    config.scan.query_texts_path = None  # Not needed for vector
    
    scanner2 = Scanner(config)
    scanner2.load_data()
    results2 = scanner2.scan()
    
    detector_results2 = results2["detector_results"]
    print(f"\nVector ranking detector results keys: {list(detector_results2.keys())}")
    
    # For vector, should have all detectors
    assert "hubness" in detector_results2, "Hubness should be present"
    assert "cluster_spread" in detector_results2, "Cluster spread should be present for vector"
    assert "stability" in detector_results2, "Stability should be present for vector"
    assert "dedup" in detector_results2, "Dedup should be present"
    
    print("✓ Vector ranking correctly includes all detectors")
    
    # Cleanup
    Path(query_texts_path).unlink()
    Path(embeddings_path).unlink()
    Path(metadata_path).unlink()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_lexical_skips_detectors()

