# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the adversarial hub demo."""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Add examples to path
examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from adversarial_hub_demo import (
    generate_documents,
    split_documents_for_rag,
    create_embeddings,
    plant_adversarial_hub,
)


def test_generate_documents():
    """Test document generation."""
    docs = generate_documents(num_docs=10)
    
    assert len(docs) == 10
    assert all("doc_id" in doc for doc in docs)
    assert all("title" in doc for doc in docs)
    assert all("content" in doc for doc in docs)
    assert all("source" in doc for doc in docs)
    assert all("topic" in doc for doc in docs)
    
    # Check doc IDs are unique
    doc_ids = [doc["doc_id"] for doc in docs]
    assert len(doc_ids) == len(set(doc_ids))


def test_split_documents_for_rag():
    """Test document splitting."""
    docs = generate_documents(num_docs=5)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    
    chunks = split_documents_for_rag(docs, chunk_size=100)
    
    assert len(chunks) > 0
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("doc_id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("text_hash" in chunk for chunk in chunks)
    
    # Check that chunks reference original documents
    doc_ids = {doc["doc_id"] for doc in docs}
    chunk_doc_ids = {chunk["doc_id"] for chunk in chunks}
    assert chunk_doc_ids.issubset(doc_ids)


def test_create_embeddings():
    """Test embedding creation."""
    docs = generate_documents(num_docs=5)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    chunks = split_documents_for_rag(docs, chunk_size=100)
    
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 64
    assert embeddings.dtype == np.float32
    
    # Check normalization (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_plant_adversarial_hub():
    """Test adversarial hub planting."""
    docs = generate_documents(num_docs=20)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    chunks = split_documents_for_rag(docs, chunk_size=100)
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    hub_chunk_idx = 10
    modified_chunks, modified_embeddings, hub_info = plant_adversarial_hub(
        chunks, embeddings, hub_chunk_idx,
        num_query_clusters=10,
        strength=0.75
    )
    
    # Check that embeddings were modified
    assert not np.array_equal(embeddings[hub_chunk_idx], modified_embeddings[hub_chunk_idx])
    
    # Check hub info
    assert hub_info["chunk_index"] == hub_chunk_idx
    assert "chunk_id" in hub_info
    assert "doc_id" in hub_info
    
    # Check that chunk is marked as adversarial
    assert modified_chunks[hub_chunk_idx].get("is_adversarial") is True
    assert "adversarial_note" in modified_chunks[hub_chunk_idx]
    
    # Check that embedding is normalized
    hub_embedding = modified_embeddings[hub_chunk_idx]
    norm = np.linalg.norm(hub_embedding)
    np.testing.assert_allclose(norm, 1.0, rtol=1e-5)


def test_plant_adversarial_hub_strength():
    """Test that adversarial hub embedding is normalized and different from original."""
    docs = generate_documents(num_docs=30)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    chunks = split_documents_for_rag(docs, chunk_size=100)
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    hub_chunk_idx = 15
    original_embedding = embeddings[hub_chunk_idx].copy()
    
    modified_chunks, modified_embeddings, hub_info = plant_adversarial_hub(
        chunks, embeddings, hub_chunk_idx,
        num_query_clusters=20,
        strength=0.80
    )
    
    hub_embedding = modified_embeddings[hub_chunk_idx]
    
    # Check that embedding was modified
    assert not np.array_equal(original_embedding, hub_embedding), "Hub embedding should be different from original"
    
    # Check normalization
    norm = np.linalg.norm(hub_embedding)
    assert abs(norm - 1.0) < 1e-5, f"Hub embedding should be normalized, got norm={norm}"
    
    # Check that hub embedding is valid (not all zeros, not NaN)
    assert np.all(np.isfinite(hub_embedding)), "Hub embedding should contain finite values"
    assert np.any(hub_embedding != 0), "Hub embedding should not be all zeros"
    
    # The hub is designed to be close to target embeddings selected during planting
    # We verify it's different from original and normalized, which is sufficient


def test_demo_data_consistency():
    """Test that demo data is consistent across steps."""
    docs = generate_documents(num_docs=10)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    
    chunks = split_documents_for_rag(docs, chunk_size=100)
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    # Check consistency
    assert len(chunks) == embeddings.shape[0]
    
    # Check that chunk indices are sequential
    chunk_indices = [chunk["chunk_index"] for chunk in chunks]
    for i, chunk in enumerate(chunks):
        doc_chunks = [c for c in chunks if c["doc_id"] == chunk["doc_id"]]
        assert chunk["chunk_index"] < len(doc_chunks)


def test_adversarial_hub_detection_integration():
    """Integration test: Plant hub and verify it can be detected."""
    pytest.importorskip("hubscan")
    
    docs = generate_documents(num_docs=30)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    
    chunks = split_documents_for_rag(docs, chunk_size=100)
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    hub_chunk_idx = 15
    chunks, embeddings, hub_info = plant_adversarial_hub(
        chunks, embeddings, hub_chunk_idx,
        num_query_clusters=20,
        strength=0.80
    )
    
    # Save data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        embeddings_path = tmpdir_path / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        metadata = {
            "chunk_id": [c["chunk_id"] for c in chunks],
            "doc_id": [c["doc_id"] for c in chunks],
            "doc_index": [c["doc_index"] for c in chunks],
            "text": [c["text"] for c in chunks],
            "is_adversarial": [c.get("is_adversarial", False) for c in chunks]
        }
        metadata_path = tmpdir_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Run HubScan
        from hubscan.sdk import scan, get_suspicious_documents, Verdict
        
        results = scan(
            embeddings_path=str(embeddings_path),
            metadata_path=str(metadata_path),
            k=5,
            num_queries=100,
            output_dir=str(tmpdir_path / "reports"),
            thresholds__hub_z=2.0,
            thresholds__percentile=0.20,
        )
        
        # Check that scan completed
        assert "verdicts" in results
        assert "json_report" in results
        
        # Check if adversarial hub was detected
        all_suspicious = get_suspicious_documents(results, verdict=None, top_k=50)
        found = any(doc["doc_index"] == hub_chunk_idx for doc in all_suspicious)
        
        # Hub should be detected (may not be HIGH, but should be suspicious)
        assert found, f"Adversarial hub at index {hub_chunk_idx} should be detected"


def test_multi_backend_faiss():
    """Test FAISS backend setup."""
    pytest.importorskip("hubscan")
    
    docs = generate_documents(num_docs=10)
    for i, doc in enumerate(docs):
        doc["doc_index"] = i
    
    chunks = split_documents_for_rag(docs, chunk_size=100)
    embeddings = create_embeddings(chunks, embedding_dim=64)
    
    metadata = {
        "chunk_id": [c["chunk_id"] for c in chunks],
        "doc_id": [c["doc_id"] for c in chunks],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Import multi-backend demo functions
        sys.path.insert(0, str(examples_dir))
        from adversarial_hub_demo_multi_backend import create_faiss_backend, run_backend_test
        
        hub_info = {"chunk_index": 5}
        
        backend_config = create_faiss_backend(embeddings, tmpdir_path)
        assert backend_config is not None
        assert backend_config["mode"] == "embeddings_only"
        assert backend_config["backend_name"] == "FAISS"
        
        # Run test
        result = run_backend_test(
            backend_config, embeddings, metadata, hub_info, tmpdir_path,
            k=5, num_queries=50
        )
        
        assert result["success"] is True
        assert result["backend"] == "FAISS"
        assert "runtime" in result
        assert "high_risk_count" in result

