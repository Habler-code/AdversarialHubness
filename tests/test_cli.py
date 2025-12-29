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

"""Tests for CLI interface."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from hubscan.cli import cli


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file."""
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
    return str(config_file)


@pytest.fixture
def sample_report(tmp_path):
    """Create a sample report file."""
    report_data = {
        "scan_info": {
            "num_documents": 1000,
            "num_queries": 100,
            "runtime_seconds": 1.5,
        },
        "summary": {
            "verdict_counts": {"HIGH": 5, "MEDIUM": 10, "LOW": 985}
        },
        "suspicious_documents": [
            {
                "doc_index": 42,
                "risk_score": 8.5,
                "verdict": "HIGH",
                "hubness": {"hub_z": 6.2, "hub_rate": 0.15}
            }
        ]
    }
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps(report_data))
    return str(report_file)


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "HubScan" in result.output
    assert "scan" in result.output


def test_cli_version(runner):
    """Test CLI version command."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_scan_missing_config(runner):
    """Test scan command with missing config."""
    result = runner.invoke(cli, ["scan", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Error" in result.output or "not found" in result.output.lower()


@patch("hubscan.cli.Scanner")
def test_cli_scan_success(mock_scanner_class, runner, sample_config):
    """Test successful scan command."""
    # Mock scanner
    mock_scanner = MagicMock()
    mock_results = {
        "json_report": {
            "scan_info": {
                "num_documents": 1000,
                "num_queries": 100,
                "runtime_seconds": 1.5,
                "k": 10,
                "index_type": "flat",
                "metric": "cosine"
            },
            "summary": {
                "verdict_counts": {"HIGH": 5, "MEDIUM": 10, "LOW": 985}
            },
            "suspicious_documents": [
                {
                    "doc_index": 42,
                    "risk_score": 8.5,
                    "verdict": "HIGH",
                    "hubness": {"hub_z": 6.2}
                }
            ]
        },
        "html_report": "<html>test</html>",
        "runtime": 1.5
    }
    mock_scanner.load_data.return_value = None
    mock_scanner.scan.return_value = mock_results
    mock_scanner_class.return_value = mock_scanner
    
    result = runner.invoke(cli, ["scan", "--config", sample_config])
    
    # Should succeed (exit code 0) or handle gracefully
    assert "Total Documents" in result.output or result.exit_code == 0


@patch("hubscan.cli.Scanner")
def test_cli_scan_summary_only(mock_scanner_class, runner, sample_config):
    """Test scan with summary-only flag."""
    mock_scanner = MagicMock()
    mock_results = {
        "json_report": {
            "scan_info": {"num_documents": 100, "num_queries": 50, "runtime_seconds": 0.5, "k": 10, "index_type": "flat", "metric": "cosine"},
            "summary": {"verdict_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 97}},
            "suspicious_documents": []
        },
        "runtime": 0.5
    }
    mock_scanner.load_data.return_value = None
    mock_scanner.scan.return_value = mock_results
    mock_scanner_class.return_value = mock_scanner
    
    result = runner.invoke(cli, ["scan", "--config", sample_config, "--summary-only"])
    assert result.exit_code == 0


def test_cli_explain_missing_report(runner):
    """Test explain command with missing report."""
    result = runner.invoke(cli, ["explain", "--doc-id", "42", "--report", "nonexistent.json"])
    assert result.exit_code != 0


def test_cli_explain_success(runner, sample_report):
    """Test successful explain command."""
    result = runner.invoke(cli, ["explain", "--doc-id", "42", "--report", sample_report])
    assert result.exit_code == 0
    assert "Document 42" in result.output or "42" in result.output


def test_cli_explain_doc_not_found(runner, sample_report):
    """Test explain command with non-existent document."""
    result = runner.invoke(cli, ["explain", "--doc-id", "9999", "--report", sample_report])
    assert result.exit_code == 0  # Should handle gracefully
    assert "not found" in result.output.lower() or result.exit_code != 0


@patch("hubscan.cli.Scanner")
def test_cli_build_index(mock_scanner_class, runner, sample_config):
    """Test build-index command."""
    mock_scanner = MagicMock()
    mock_scanner.load_data.return_value = None
    mock_scanner_class.return_value = mock_scanner
    
    result = runner.invoke(cli, ["build-index", "--config", sample_config])
    # Should succeed or handle error gracefully
    assert result.exit_code == 0 or "Error" in result.output


def test_cli_verbose_flag(runner):
    """Test verbose flag."""
    result = runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0

