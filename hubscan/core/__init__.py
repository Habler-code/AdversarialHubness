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

"""Core modules for hubness detection."""

from .detectors import (
    Detector,
    DetectorResult,
    HubnessDetector,
    ClusterSpreadDetector,
    StabilityDetector,
    DedupDetector,
)
from .io import (
    load_embeddings,
    save_embeddings,
    load_faiss_index,
    build_faiss_index,
    save_faiss_index,
    load_metadata,
    Metadata,
)
from .sampling import sample_queries, QuerySampler
from .scoring import combine_scores, compute_risk_score, apply_thresholds, Verdict
from .report import generate_json_report, generate_html_report, save_json_report, save_html_report

__all__ = [
    # Detectors
    "Detector",
    "DetectorResult",
    "HubnessDetector",
    "ClusterSpreadDetector",
    "StabilityDetector",
    "DedupDetector",
    # I/O
    "load_embeddings",
    "save_embeddings",
    "load_faiss_index",
    "build_faiss_index",
    "save_faiss_index",
    "load_metadata",
    "Metadata",
    # Sampling
    "sample_queries",
    "QuerySampler",
    # Scoring
    "combine_scores",
    "compute_risk_score",
    "apply_thresholds",
    "Verdict",
    # Reporting
    "generate_json_report",
    "generate_html_report",
    "save_json_report",
    "save_html_report",
]

