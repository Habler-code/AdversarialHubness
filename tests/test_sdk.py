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

"""Tests for SDK interface."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from hubscan.sdk import (
    scan,
    quick_scan,
    scan_from_config,
    get_suspicious_documents,
    explain_document,
)
from hubscan.core.scoring import Verdict
from hubscan.config import Config


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    np.random.seed(42)
    embeddings = np.random.randn(100, 32).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def sample_results():
    """Create sample scan results."""
    return {
        "json_report": {
            "scan_info": {
                "num_documents": 100,
                "num_queries": 50,
                "runtime_seconds": 1.0,
            },
            "summary": {
                "verdict_counts": {"HIGH": 5, "MEDIUM": 10, "LOW": 85}
            },
            "suspicious_documents": [
                {
                    "doc_index": 0,
                    "risk_score": 8.5,
                    "verdict": "HIGH",
                    "hubness": {"hub_z": 6.2, "hub_rate": 0.15, "hits": 8}
                },
                {
                    "doc_index": 1,
                    "risk_score": 5.2,
                    "verdict": "MEDIUM",
                    "hubness": {"hub_z": 4.1, "hub_rate": 0.10, "hits": 5}
                },
                {
                    "doc_index": 2,
                    "risk_score": 2.1,
                    "verdict": "LOW",
                    "hubness": {"hub_z": 1.5, "hub_rate": 0.05, "hits": 2}
                }
            ]
        },
        "html_report": "<html>test</html>",
        "detector_results": {},
        "combined_scores": np.array([8.5, 5.2, 2.1] + [1.0] * 97),
        "verdicts": {0: Verdict.HIGH, 1: Verdict.MEDIUM, 2: Verdict.LOW, **{i: Verdict.LOW for i in range(3, 100)}},
        "runtime": 1.0
    }


def test_quick_scan(sample_embeddings):
    """Test quick_scan function."""
    with patch("hubscan.sdk.Scanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        mock_results = {
            "json_report": {
                "scan_info": {"num_documents": 100, "num_queries": 50, "runtime_seconds": 0.5, "k": 10, "index_type": "flat", "metric": "cosine"},
                "summary": {"verdict_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 97}},
                "suspicious_documents": []
            },
            "html_report": "<html>test</html>",
            "detector_results": {},
            "combined_scores": np.array([1.0] * 100),
            "verdicts": {i: Verdict.LOW for i in range(100)},
            "runtime": 0.5
        }
        mock_scanner.load_data.return_value = None
        mock_scanner.scan.return_value = mock_results
        mock_scanner_class.return_value = mock_scanner
        
        results = quick_scan(sample_embeddings, k=10, num_queries=50)
        
        assert "json_report" in results
        assert "runtime" in results
        assert results["runtime"] == 0.5


def test_get_suspicious_documents(sample_results):
    """Test get_suspicious_documents function."""
    # Get all suspicious (default is top 100, but we only have 3 in test data)
    all_suspicious = get_suspicious_documents(sample_results)
    # The function returns top suspicious documents from the report, which may be limited
    assert len(all_suspicious) >= 1
    assert all_suspicious[0]["doc_index"] == 0
    
    # Get high-risk only
    high_risk = get_suspicious_documents(sample_results, verdict=Verdict.HIGH)
    assert len(high_risk) == 1
    assert high_risk[0]["verdict"] == "HIGH"
    assert high_risk[0]["doc_index"] == 0
    
    # Get top-k (without verdict filter)
    top_2 = get_suspicious_documents(sample_results, verdict=None, top_k=2)
    assert len(top_2) == 2
    
    # Get medium-risk
    medium_risk = get_suspicious_documents(sample_results, verdict=Verdict.MEDIUM)
    assert len(medium_risk) == 1
    assert medium_risk[0]["verdict"] == "MEDIUM"


def test_explain_document(sample_results):
    """Test explain_document function."""
    # Explain existing document
    explanation = explain_document(sample_results, doc_index=0)
    assert explanation is not None
    assert explanation["doc_index"] == 0
    assert explanation["risk_score"] == 8.5
    assert explanation["verdict"] == "HIGH"
    assert "hubness" in explanation
    
    # Explain non-existent document
    explanation = explain_document(sample_results, doc_index=999)
    assert explanation is None


def test_scan_from_config(tmp_path):
    """Test scan_from_config function."""
    # Create config file
    config_content = """
input:
  mode: embeddings_only
  embeddings_path: examples/toy_embeddings.npy
  metric: cosine

index:
  type: flat
  params: {}

scan:
  k: 10
  num_queries: 100
  query_sampling: random_docs_as_queries
  batch_size: 32
  seed: 42

detectors:
  hubness:
    enabled: true
  cluster_spread:
    enabled: false
  stability:
    enabled: false
  dedup:
    enabled: false

scoring:
  weights:
    hub_z: 0.6
    cluster_spread: 0.2
    stability: 0.2
    boilerplate: 0.3

thresholds:
  policy: hybrid
  hub_z: 3.0
  percentile: 0.05

output:
  out_dir: reports/
  privacy_mode: false
  emit_embeddings: false
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    with patch("hubscan.sdk.Scanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        mock_results = {
            "json_report": {
                "scan_info": {"num_documents": 100, "num_queries": 50, "runtime_seconds": 0.5, "k": 10, "index_type": "flat", "metric": "cosine"},
                "summary": {"verdict_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 97}},
                "suspicious_documents": []
            },
            "html_report": "<html>test</html>",
            "detector_results": {},
            "combined_scores": np.array([1.0] * 100),
            "verdicts": {i: Verdict.LOW for i in range(100)},
            "runtime": 0.5
        }
        mock_scanner.load_data.return_value = None
        mock_scanner.scan.return_value = mock_results
        mock_scanner_class.return_value = mock_scanner
        
        results = scan_from_config(str(config_file))
        assert "json_report" in results


def test_scan_with_embeddings_path(tmp_path, sample_embeddings):
    """Test scan function with embeddings path."""
    # Save embeddings
    embeddings_file = tmp_path / "embeddings.npy"
    np.save(embeddings_file, sample_embeddings)
    
    with patch("hubscan.sdk.Scanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        mock_results = {
            "json_report": {
                "scan_info": {"num_documents": 100, "num_queries": 50, "runtime_seconds": 0.5, "k": 10, "index_type": "flat", "metric": "cosine"},
                "summary": {"verdict_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 97}},
                "suspicious_documents": []
            },
            "html_report": "<html>test</html>",
            "detector_results": {},
            "combined_scores": np.array([1.0] * 100),
            "verdicts": {i: Verdict.LOW for i in range(100)},
            "runtime": 0.5
        }
        mock_scanner.load_data.return_value = None
        mock_scanner.scan.return_value = mock_results
        mock_scanner_class.return_value = mock_scanner
        
        results = scan(
            embeddings_path=str(embeddings_file),
            k=10,
            num_queries=50
        )
        assert "json_report" in results


def test_scan_with_config_path(tmp_path):
    """Test scan function with config path."""
    config_content = """
input:
  mode: embeddings_only
  embeddings_path: examples/toy_embeddings.npy
  metric: cosine

index:
  type: flat
  params: {}

scan:
  k: 10
  num_queries: 100
  query_sampling: random_docs_as_queries
  batch_size: 32
  seed: 42

detectors:
  hubness:
    enabled: true

scoring:
  weights:
    hub_z: 0.6

thresholds:
  policy: hybrid
  hub_z: 3.0

output:
  out_dir: reports/
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    with patch("hubscan.sdk.Scanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        mock_results = {
            "json_report": {
                "scan_info": {"num_documents": 100, "num_queries": 50, "runtime_seconds": 0.5, "k": 10, "index_type": "flat", "metric": "cosine"},
                "summary": {"verdict_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 97}},
                "suspicious_documents": []
            },
            "html_report": "<html>test</html>",
            "detector_results": {},
            "combined_scores": np.array([1.0] * 100),
            "verdicts": {i: Verdict.LOW for i in range(100)},
            "runtime": 0.5
        }
        mock_scanner.load_data.return_value = None
        mock_scanner.scan.return_value = mock_results
        mock_scanner_class.return_value = mock_scanner
        
        results = scan(config_path=str(config_file))
        assert "json_report" in results


def test_scan_missing_parameters():
    """Test scan function with missing required parameters."""
    with pytest.raises(ValueError):
        scan()  # No embeddings_path or index_path


def test_get_suspicious_documents_empty_results():
    """Test get_suspicious_documents with empty results."""
    empty_results = {
        "json_report": {
            "suspicious_documents": []
        }
    }
    suspicious = get_suspicious_documents(empty_results)
    assert len(suspicious) == 0


def test_explain_document_edge_cases(sample_results):
    """Test explain_document with edge cases."""
    # Test with empty suspicious documents
    empty_results = {
        "json_report": {
            "suspicious_documents": []
        }
    }
    explanation = explain_document(empty_results, doc_index=0)
    assert explanation is None
    
    # Test with missing hubness data
    results_no_hubness = {
        "json_report": {
            "suspicious_documents": [
                {
                    "doc_index": 0,
                    "risk_score": 5.0,
                    "verdict": "MEDIUM"
                }
            ]
        }
    }
    explanation = explain_document(results_no_hubness, doc_index=0)
    assert explanation is not None
    assert "hubness" not in explanation

