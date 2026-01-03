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

"""Tests for JSONL adapter."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from hubscan.core.io.adapters.jsonl_adapter import JSONLExportLoader, load_jsonl_export


class TestJSONLExportLoader:
    """Tests for JSONLExportLoader."""
    
    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a sample JSONL file for testing."""
        records = [
            {"id": "doc_0", "embedding": [0.1, 0.2, 0.3], "text": "Hello world", "category": "greeting"},
            {"id": "doc_1", "embedding": [0.4, 0.5, 0.6], "text": "Goodbye world", "category": "farewell"},
            {"id": "doc_2", "embedding": [0.7, 0.8, 0.9], "text": "Test document", "category": "test"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_vector_jsonl_file(self):
        """Create a JSONL file using 'vector' field instead of 'embedding'."""
        records = [
            {"id": "doc_0", "vector": [0.1, 0.2, 0.3], "text": "Doc 1"},
            {"id": "doc_1", "vector": [0.4, 0.5, 0.6], "text": "Doc 2"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_load_basic(self, sample_jsonl_file):
        """Test basic loading functionality."""
        loader = JSONLExportLoader(normalize=False)
        embeddings, metadata = loader.load(sample_jsonl_file)
        
        assert embeddings.shape == (3, 3)
        assert len(metadata) == 3
        
        # Check embeddings
        np.testing.assert_array_almost_equal(embeddings[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(embeddings[1], [0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(embeddings[2], [0.7, 0.8, 0.9])
        
        # Check metadata
        assert metadata[0]["id"] == "doc_0"
        assert metadata[0]["text"] == "Hello world"
        assert metadata[0]["category"] == "greeting"
        
        # Embedding should not be in metadata
        assert "embedding" not in metadata[0]
    
    def test_load_with_normalization(self, sample_jsonl_file):
        """Test loading with normalization."""
        loader = JSONLExportLoader(normalize=True)
        embeddings, _ = loader.load(sample_jsonl_file)
        
        # Check that embeddings are normalized (unit length)
        for i in range(len(embeddings)):
            norm = np.linalg.norm(embeddings[i])
            np.testing.assert_almost_equal(norm, 1.0)
    
    def test_load_with_vector_field(self, sample_vector_jsonl_file):
        """Test loading from file using 'vector' field."""
        loader = JSONLExportLoader(normalize=False)
        embeddings, metadata = loader.load(sample_vector_jsonl_file)
        
        assert embeddings.shape == (2, 3)
        assert len(metadata) == 2
        
        np.testing.assert_array_almost_equal(embeddings[0], [0.1, 0.2, 0.3])
        assert "vector" not in metadata[0]
    
    def test_load_with_limit(self, sample_jsonl_file):
        """Test loading with limit."""
        loader = JSONLExportLoader(normalize=False)
        embeddings, metadata = loader.load(sample_jsonl_file, limit=2)
        
        assert embeddings.shape == (2, 3)
        assert len(metadata) == 2
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        loader = JSONLExportLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.jsonl")
    
    def test_load_missing_embedding(self):
        """Test loading file with missing embedding field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"id": "doc_0", "text": "no embedding"}) + '\n')
            temp_path = f.name
        
        try:
            loader = JSONLExportLoader()
            with pytest.raises(ValueError, match="No embedding found"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_invalid_json(self):
        """Test loading file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("not valid json\n")
            temp_path = f.name
        
        try:
            loader = JSONLExportLoader()
            with pytest.raises(ValueError, match="Invalid JSON"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            loader = JSONLExportLoader()
            with pytest.raises(ValueError, match="No records found"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_batched(self, sample_jsonl_file):
        """Test batched loading."""
        loader = JSONLExportLoader(normalize=False)
        
        batches = list(loader.load_batched(sample_jsonl_file, batch_size=2))
        
        assert len(batches) == 2  # 3 records with batch_size=2 -> 2 batches
        
        # First batch should have 2 records
        emb_batch1, meta_batch1 = batches[0]
        assert emb_batch1.shape == (2, 3)
        assert len(meta_batch1) == 2
        
        # Second batch should have 1 record
        emb_batch2, meta_batch2 = batches[1]
        assert emb_batch2.shape == (1, 3)
        assert len(meta_batch2) == 1


class TestLoadJSONLExportFunction:
    """Tests for the load_jsonl_export convenience function."""
    
    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a sample JSONL file for testing."""
        records = [
            {"id": "doc_0", "embedding": [1.0, 0.0, 0.0]},
            {"id": "doc_1", "embedding": [0.0, 1.0, 0.0]},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    def test_load_jsonl_export(self, sample_jsonl_file):
        """Test the convenience function."""
        embeddings, metadata = load_jsonl_export(
            sample_jsonl_file,
            normalize=False,
        )
        
        assert embeddings.shape == (2, 3)
        assert len(metadata) == 2
    
    def test_load_jsonl_export_with_limit(self, sample_jsonl_file):
        """Test with limit parameter."""
        embeddings, metadata = load_jsonl_export(
            sample_jsonl_file,
            limit=1,
        )
        
        assert embeddings.shape == (1, 3)
        assert len(metadata) == 1

