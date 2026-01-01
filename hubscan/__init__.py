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

"""HubScan: Adversarial Hubness Detection for RAG Systems

Example usage:
    from hubscan import quick_scan, get_suspicious_documents, Verdict
    
    results = quick_scan(embeddings=embeddings, k=10)
    suspicious = get_suspicious_documents(results, min_verdict=Verdict.MEDIUM)
"""

__version__ = "0.1.0"

from .config import Config
from .core.scanner import Scanner
from .core.detectors import (
    HubnessDetector,
    ClusterSpreadDetector,
    StabilityDetector,
    DedupDetector,
)
from .core.io import (
    load_embeddings,
    load_faiss_index,
    build_faiss_index,
)
from .core.sampling import sample_queries
from .core.scoring import combine_scores, apply_thresholds, Verdict

# SDK functions - primary API for most users
from .sdk import (
    scan,
    quick_scan,
    scan_from_config,
    get_suspicious_documents,
    explain_document,
)

__all__ = [
    # Core
    "Config",
    "Scanner",
    # Detectors
    "HubnessDetector",
    "ClusterSpreadDetector",
    "StabilityDetector",
    "DedupDetector",
    # I/O
    "load_embeddings",
    "load_faiss_index",
    "build_faiss_index",
    # Utilities
    "sample_queries",
    "combine_scores",
    "apply_thresholds",
    "Verdict",
    # SDK (primary API)
    "scan",
    "quick_scan",
    "scan_from_config",
    "get_suspicious_documents",
    "explain_document",
]
