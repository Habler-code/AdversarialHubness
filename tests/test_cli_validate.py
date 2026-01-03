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

"""Tests for CLI validate command."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
from click.testing import CliRunner

from hubscan.cli import cli


class TestCLIValidate:
    """Tests for CLI validate command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def valid_config(self, tmp_path):
        """Create a valid config file with test data."""
        # Create test embeddings
        embeddings = np.random.rand(100, 64).astype(np.float32)
        embeddings_path = tmp_path / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Create test metadata
        metadata = [{"id": f"doc_{i}", "text": f"Document {i}"} for i in range(100)]
        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Create config
        config = {
            "input": {
                "mode": "embeddings_only",
                "embeddings_path": str(embeddings_path),
                "metadata_path": str(metadata_path),
                "metric": "cosine",
            },
            "scan": {
                "k": 10,
                "num_queries": 100,
                "ranking": {
                    "method": "vector",
                },
            },
            "output": {
                "out_dir": str(tmp_path / "reports"),
            },
        }
        
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def invalid_config(self, tmp_path):
        """Create a config file with missing files."""
        config = {
            "input": {
                "mode": "embeddings_only",
                "embeddings_path": "/nonexistent/embeddings.npy",
                "metadata_path": "/nonexistent/metadata.json",
                "metric": "cosine",
            },
            "scan": {
                "k": 10,
                "num_queries": 100,
            },
            "output": {
                "out_dir": str(tmp_path / "reports"),
            },
        }
        
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_validate_basic(self, runner, valid_config):
        """Test basic config validation."""
        result = runner.invoke(cli, ["validate", "-c", str(valid_config)])
        
        # Should succeed
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
    
    def test_validate_with_check_data(self, runner, valid_config):
        """Test validation with data file checks."""
        result = runner.invoke(cli, ["validate", "-c", str(valid_config), "--check-data"])
        
        assert result.exit_code == 0
        assert "Embeddings" in result.output
        assert "Metadata" in result.output
    
    def test_validate_missing_files(self, runner, invalid_config):
        """Test validation with missing data files."""
        result = runner.invoke(cli, ["validate", "-c", str(invalid_config), "--check-data"])
        
        # Should fail with missing files
        assert result.exit_code == 1
        assert "NOT FOUND" in result.output
    
    def test_validate_with_check_index(self, runner, valid_config):
        """Test validation with index integrity check."""
        result = runner.invoke(cli, [
            "validate", 
            "-c", str(valid_config), 
            "--check-data",
            "--check-index",
            "--sample-queries", "5",
        ])
        
        assert result.exit_code == 0
        assert "test queries" in result.output.lower()
    
    def test_validate_nonexistent_config(self, runner, tmp_path):
        """Test validation with non-existent config file."""
        result = runner.invoke(cli, ["validate", "-c", "/nonexistent/config.yaml"])
        
        # Click should fail before reaching our code
        assert result.exit_code != 0
    
    def test_validate_invalid_yaml(self, runner, tmp_path):
        """Test validation with invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")
        
        result = runner.invoke(cli, ["validate", "-c", str(config_path)])
        
        assert result.exit_code == 1
    
    def test_validate_shows_configuration_summary(self, runner, valid_config):
        """Test that validation shows configuration summary."""
        result = runner.invoke(cli, ["validate", "-c", str(valid_config)])
        
        assert "Configuration Summary" in result.output
        assert "Input Mode" in result.output
        assert "embeddings_only" in result.output

