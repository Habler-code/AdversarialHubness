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

"""JSON report generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from ...config import Config
from ..detectors.base import DetectorResult
from ..scoring.thresholds import Verdict
from ..io.metadata import Metadata


def generate_json_report(
    config: Config,
    detector_results: Dict[str, DetectorResult],
    combined_scores: np.ndarray,
    verdicts: Dict[int, Verdict],
    metadata: Optional[Metadata] = None,
    num_queries: int = 0,
    runtime_seconds: float = 0.0,
    num_docs: int = 0,
) -> Dict[str, Any]:
    """
    Generate JSON report.
    
    Args:
        config: Configuration
        detector_results: Detector results
        combined_scores: Combined risk scores
        verdicts: Verdicts per document
        metadata: Document metadata
        num_queries: Number of queries processed
        runtime_seconds: Runtime in seconds
        num_docs: Number of documents
        
    Returns:
        Report dictionary
    """
    # Get top suspicious documents
    top_k = min(100, len(combined_scores))
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    
    suspicious_docs = []
    for idx in top_indices:
        doc_data: Dict[str, Any] = {
            "doc_index": int(idx),
            "risk_score": float(combined_scores[idx]),
            "verdict": verdicts.get(idx, Verdict.LOW).value,
        }
        
        # Add detector-specific scores
        if "hubness" in detector_results:
            hub_result = detector_results["hubness"]
            doc_data["hubness"] = {
                "hub_z": float(hub_result.scores[idx]),
                "hub_rate": float(hub_result.metadata.get("hub_rate", [0])[idx]) if "hub_rate" in hub_result.metadata else None,
                "hits": int(hub_result.metadata.get("hits", [0])[idx]) if "hits" in hub_result.metadata else None,
            }
            # Example queries
            example_queries = hub_result.metadata.get("example_queries", {})
            if str(idx) in example_queries:
                doc_data["hubness"]["example_query_indices"] = example_queries[str(idx)][:config.output.max_example_queries]
        
        if "cluster_spread" in detector_results:
            cluster_result = detector_results["cluster_spread"]
            doc_data["cluster_spread"] = {
                "score": float(cluster_result.scores[idx]),
                "entropy": float(cluster_result.metadata.get("cluster_entropy", [0])[idx]) if "cluster_entropy" in cluster_result.metadata else None,
            }
        
        if "stability" in detector_results:
            stability_result = detector_results["stability"]
            doc_data["stability"] = {
                "score": float(stability_result.scores[idx]),
            }
        
        if "dedup" in detector_results:
            dedup_result = detector_results["dedup"]
            doc_data["deduplication"] = {
                "boilerplate_score": float(dedup_result.scores[idx]),
            }
        
        # Add metadata if available
        if metadata:
            doc_meta = {}
            if metadata.has_field("doc_id"):
                doc_meta["doc_id"] = metadata.get_field("doc_id", idx)
            if metadata.has_field("source") and not config.output.privacy_mode:
                doc_meta["source"] = metadata.get_field("source", idx)
            if metadata.has_field("path") and not config.output.privacy_mode:
                doc_meta["path"] = metadata.get_field("path", idx)
            if metadata.has_field("text") and not config.output.privacy_mode:
                text = metadata.get_field("text", idx)
                if text:
                    # Truncate text preview
                    doc_meta["text_preview"] = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
            
            if doc_meta:
                doc_data["metadata"] = doc_meta
        
        suspicious_docs.append(doc_data)
    
    # Summary statistics
    verdict_counts = {v.value: sum(1 for verdict in verdicts.values() if verdict == v) for v in Verdict}
    
    report = {
        "scan_info": {
            "timestamp": datetime.now().isoformat(),
            "num_documents": num_docs,
            "num_queries": num_queries,
            "k": config.scan.k,
            "runtime_seconds": runtime_seconds,
            "index_type": config.index.type,
            "metric": config.input.metric,
        },
        "summary": {
            "verdict_counts": verdict_counts,
            "mean_risk_score": float(np.mean(combined_scores)),
            "max_risk_score": float(np.max(combined_scores)),
            "median_risk_score": float(np.median(combined_scores)),
        },
        "detector_summary": {},
        "suspicious_documents": suspicious_docs,
    }
    
    # Add detector summaries
    for name, result in detector_results.items():
        report["detector_summary"][name] = {
            "mean_score": float(np.mean(result.scores)),
            "max_score": float(np.max(result.scores)),
            "median_score": float(np.median(result.scores)),
        }
    
    return report


def save_json_report(report: Dict[str, Any], output_path: str):
    """Save JSON report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

