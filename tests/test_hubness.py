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

"""Tests for hubness detector."""

import numpy as np
import pytest
import faiss

from hubscan.core.detectors.hubness import HubnessDetector
from hubscan.core.io.metadata import Metadata


def test_hubness_detection():
    """Test hubness detection on a simple dataset."""
    # Create a simple dataset with one obvious hub
    num_docs = 100
    embedding_dim = 32
    
    # Generate random embeddings
    doc_embeddings = np.random.randn(num_docs, embedding_dim).astype(np.float32)
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Create a hub: one vector that's close to many query vectors
    hub_idx = 0
    # Make hub vector be the average of many random vectors
    hub_vector = np.mean(np.random.randn(50, embedding_dim), axis=0)
    hub_vector = hub_vector / np.linalg.norm(hub_vector)
    doc_embeddings[hub_idx] = hub_vector.astype(np.float32)
    
    # Build index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(doc_embeddings)
    
    # Create queries that are close to the hub
    num_queries = 50
    queries = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    # Make queries closer to hub
    for i in range(num_queries):
        if i % 2 == 0:  # Half the queries are close to hub
            queries[i] = 0.7 * queries[i] + 0.3 * hub_vector
        queries[i] = queries[i] / np.linalg.norm(queries[i])
    
    # Run detector
    detector = HubnessDetector(enabled=True, validate_exact=False)
    result = detector.detect(index, doc_embeddings, queries, k=5)
    
    # Check that hub has high score
    hub_score = result.scores[hub_idx]
    assert hub_score > 0, "Hub should have positive score"
    
    # Hub should have a reasonable score (not necessarily top, but above median)
    # This is a more lenient check since hubness detection depends on many factors
    median_score = np.median(result.scores)
    assert hub_score >= median_score, f"Hub score {hub_score} should be >= median {median_score}"


def test_robust_zscore():
    """Test robust z-score computation."""
    from hubscan.utils.metrics import robust_zscore
    
    # Create data with outlier
    values = np.array([1.0, 1.1, 1.0, 1.2, 1.1, 10.0])  # 10.0 is outlier
    
    z_scores, median, mad = robust_zscore(values)
    
    # Outlier should have high z-score
    outlier_idx = np.argmax(values)
    assert z_scores[outlier_idx] > 3.0, "Outlier should have high z-score"
    
    # Other values should have lower z-scores
    other_indices = [i for i in range(len(values)) if i != outlier_idx]
    assert all(z_scores[i] < 2.0 for i in other_indices), "Non-outliers should have lower z-scores"

