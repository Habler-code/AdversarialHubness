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

"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path

from hubscan.config import Config, InputConfig, ScanConfig, ThresholdsConfig, RankingConfig


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    
    assert config.input.mode == "embeddings_only"
    assert config.input.metric == "cosine"
    assert config.index.type == "hnsw"
    assert config.scan.k == 20
    assert config.scan.num_queries == 10000
    assert config.thresholds.policy == "hybrid"


def test_config_from_yaml(tmp_path):
    """Test loading configuration from YAML file."""
    config_data = {
        "input": {
            "mode": "faiss_index",
            "index_path": "test.index",
            "metric": "l2"
        },
        "scan": {
            "k": 30,
            "num_queries": 5000
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    config = Config.from_yaml(str(config_file))
    
    assert config.input.mode == "faiss_index"
    assert config.input.index_path == "test.index"
    assert config.input.metric == "l2"
    assert config.scan.k == 30
    assert config.scan.num_queries == 5000


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "input" in config_dict
    assert "scan" in config_dict
    assert "detectors" in config_dict


def test_input_config_validation():
    """Test input configuration validation."""
    config = InputConfig(mode="embeddings_only")
    assert config.mode == "embeddings_only"
    
    # Test invalid mode
    with pytest.raises(Exception):
        InputConfig(mode="invalid_mode")


def test_scan_config_validation():
    """Test scan configuration validation."""
    config = ScanConfig(k=10, num_queries=100)
    assert config.k == 10
    assert config.num_queries == 100
    
    # Test invalid k (should be >= 1)
    with pytest.raises(Exception):
        ScanConfig(k=0)


def test_thresholds_config():
    """Test thresholds configuration."""
    config = ThresholdsConfig(
        policy="hybrid",
        hub_z=5.0,
        percentile=0.001
    )
    
    assert config.policy == "hybrid"
    assert config.hub_z == 5.0
    assert config.percentile == 0.001


def test_ranking_config_defaults():
    """Test ranking configuration defaults."""
    config = RankingConfig()
    
    assert config.method == "vector"
    assert config.hybrid_alpha == 0.5
    assert config.rerank_top_n == 100
    assert config.lexical_backend is None


def test_ranking_config_validation():
    """Test ranking configuration validation."""
    # Valid configs
    config = RankingConfig(method="hybrid", hybrid_alpha=0.7)
    assert config.method == "hybrid"
    assert config.hybrid_alpha == 0.7
    
    config = RankingConfig(method="lexical", lexical_backend="bm25")
    assert config.method == "lexical"
    assert config.lexical_backend == "bm25"
    
    # Test reranking configuration
    config = RankingConfig(method="vector", rerank=True, rerank_method="default", rerank_top_n=50)
    assert config.method == "vector"
    assert config.rerank == True
    assert config.rerank_method == "default"
    assert config.rerank_top_n == 50
    
    # Invalid alpha (should be clamped or raise error)
    with pytest.raises(Exception):
        RankingConfig(hybrid_alpha=1.5)  # > 1.0
    
    with pytest.raises(Exception):
        RankingConfig(hybrid_alpha=-0.1)  # < 0.0
    
    # Invalid rerank_top_n
    with pytest.raises(Exception):
        RankingConfig(rerank_top_n=0)  # < 1


def test_scan_config_with_ranking():
    """Test scan configuration includes ranking config."""
    config = ScanConfig()
    
    assert hasattr(config, "ranking")
    assert isinstance(config.ranking, RankingConfig)
    assert config.ranking.method == "vector"


def test_config_from_yaml_with_ranking(tmp_path):
    """Test loading configuration with ranking from YAML."""
    # Create a temporary query texts file for hybrid search requirement
    query_texts_file = tmp_path / "query_texts.json"
    with open(query_texts_file, "w") as f:
        f.write('["query 1", "query 2"]')
    
    config_data = {
        "scan": {
            "k": 20,
            "num_queries": 1000,
            "query_texts_path": str(query_texts_file),  # Required for hybrid
            "ranking": {
                "method": "hybrid",
                "hybrid_alpha": 0.6,
                "rerank_top_n": 50
            }
        }
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    config = Config.from_yaml(str(config_file))
    
    assert config.scan.ranking.method == "hybrid"
    assert config.scan.ranking.hybrid_alpha == 0.6
    assert config.scan.ranking.rerank_top_n == 50

