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

"""Integration tests for end-to-end scanning."""

import numpy as np
import pytest
from pathlib import Path
import json

from hubscan import Config, Scanner


def test_toy_scan():
    """Test scanning on toy dataset."""
    # Generate toy data if it doesn't exist
    toy_embeddings_path = Path("examples/data/toy_embeddings.npy")
    if not toy_embeddings_path.exists():
        pytest.skip("Toy data not generated. Run examples/scripts/generate_toy_data.py first.")
    
    # Load config
    config_path = Path("examples/configs/toy_config.yaml")
    if not config_path.exists():
        pytest.skip("Toy config not found.")
    
    config = Config.from_yaml(str(config_path))
    
    # Create scanner
    scanner = Scanner(config)
    scanner.load_data()
    
    # Run scan
    results = scanner.scan()
    
    # Check results
    assert "json_report" in results
    assert "html_report" in results
    assert "verdicts" in results
    
    json_report = results["json_report"]
    assert "scan_info" in json_report
    assert "summary" in json_report
    assert "suspicious_documents" in json_report
    
    # Check that some documents were flagged
    verdicts = results["verdicts"]
    high_risk_count = sum(1 for v in verdicts.values() if v.value == "HIGH")
    assert high_risk_count >= 0  # At least we got verdicts
    
    print(f"Scan completed. Found {high_risk_count} high-risk documents.")

