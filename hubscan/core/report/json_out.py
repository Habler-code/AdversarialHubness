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
    detection_metrics: Optional[Dict[str, Any]] = None,
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
            
            # Concept-aware features
            concept_aware = hub_result.metadata.get("concept_aware", {})
            if concept_aware.get("enabled", False):
                max_concept_hub_z = concept_aware.get("max_concept_hub_z", [])
                top_concept_ids = concept_aware.get("top_concept_ids", [])
                concept_names = concept_aware.get("concept_names", {})
                
                doc_concept_data = {}
                if max_concept_hub_z and idx < len(max_concept_hub_z):
                    doc_concept_data["max_concept_hub_z"] = float(max_concept_hub_z[idx])
                if top_concept_ids and idx < len(top_concept_ids):
                    cid = top_concept_ids[idx]
                    if cid >= 0:
                        doc_concept_data["top_concept_id"] = int(cid)
                        doc_concept_data["top_concept_name"] = concept_names.get(cid, f"concept_{cid}")
                
                if doc_concept_data:
                    doc_data["hubness"]["concept_specific"] = doc_concept_data
            
            # Modality-aware features
            modality_aware = hub_result.metadata.get("modality_aware", {})
            if modality_aware.get("enabled", False):
                cross_modal_flags = modality_aware.get("cross_modal_flags", [])
                cross_modal_ratios = modality_aware.get("cross_modal_ratios", {})
                
                doc_modality_data = {}
                if cross_modal_flags and idx < len(cross_modal_flags):
                    doc_modality_data["is_cross_modal"] = bool(cross_modal_flags[idx])
                if str(idx) in cross_modal_ratios:
                    doc_modality_data["cross_modal_ratio"] = float(cross_modal_ratios[str(idx)])
                
                if doc_modality_data:
                    doc_data["hubness"]["modality_specific"] = doc_modality_data
        
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
            "ranking_method": config.scan.ranking.method,
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
        
        # Add ranking method info if available
        if name == "hubness" and "ranking_method" in result.metadata:
            report["detector_summary"][name]["ranking_method"] = result.metadata["ranking_method"]
            if "hybrid_alpha" in result.metadata:
                report["detector_summary"][name]["hybrid_alpha"] = result.metadata["hybrid_alpha"]
            
            # Add concept-aware summary
            concept_aware = result.metadata.get("concept_aware", {})
            if concept_aware.get("enabled", False):
                concept_summary = concept_aware.get("concept_summary", {})
                report["detector_summary"][name]["concept_aware"] = {
                    "enabled": True,
                    "num_concepts": len(concept_summary),
                    "fallback_used": concept_aware.get("fallback_used", False),
                    "threshold": concept_aware.get("concept_hub_z_threshold", 4.0),
                }
                # Add concept breakdown
                if concept_summary:
                    report["concept_summary"] = {
                        str(cid): {
                            "name": concept_aware.get("concept_names", {}).get(cid, f"concept_{cid}"),
                            **stats
                        }
                        for cid, stats in concept_summary.items()
                    }
            
            # Add modality-aware summary
            modality_aware = result.metadata.get("modality_aware", {})
            if modality_aware.get("enabled", False):
                modality_summary = modality_aware.get("modality_summary", {})
                report["detector_summary"][name]["modality_aware"] = {
                    "enabled": True,
                    "num_cross_modal_docs": modality_aware.get("num_cross_modal_docs", 0),
                    "cross_modal_penalty": modality_aware.get("cross_modal_penalty", 1.5),
                    "modalities_found": modality_aware.get("modalities_found", []),
                }
                # Add modality breakdown
                if modality_summary:
                    report["modality_summary"] = modality_summary
    
    # Add metrics if available
    if detection_metrics:
        report["detection_metrics"] = detection_metrics
    
    return report


def save_json_report(report: Dict[str, Any], output_path: str):
    """Save JSON report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

