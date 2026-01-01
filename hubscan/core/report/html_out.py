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

"""HTML report generation."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from jinja2 import Template

from ...config import Config
from ..detectors.base import DetectorResult
from ..scoring.thresholds import Verdict
from ..io.metadata import Metadata
from .json_out import generate_json_report


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>HubScan Report - Adversarial Hubness Detection</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #0066cc;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .summary-card.concept {
            border-left-color: #6366f1;
        }
        .summary-card.modality {
            border-left-color: #f59e0b;
        }
        .verdict-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .verdict-high {
            background-color: #fee;
            color: #c33;
        }
        .verdict-medium {
            background-color: #ffe;
            color: #c93;
        }
        .verdict-low {
            background-color: #efe;
            color: #3c3;
        }
        .concept-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 11px;
            background-color: #e0e7ff;
            color: #3730a3;
        }
        .modality-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 11px;
        }
        .modality-cross {
            background-color: #fef3c7;
            color: #92400e;
        }
        .modality-same {
            background-color: #d1fae5;
            color: #065f46;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #0066cc;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .score {
            font-family: monospace;
            font-weight: bold;
        }
        .text-preview {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 12px;
            text-align: center;
        }
        .feature-section {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .feature-section.disabled {
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>HubScan Report - Adversarial Hubness Detection</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Documents</h3>
                <div class="value">{{ scan_info.num_documents | int }}</div>
            </div>
            <div class="summary-card">
                <h3>Queries Processed</h3>
                <div class="value">{{ scan_info.num_queries | int }}</div>
            </div>
            <div class="summary-card">
                <h3>High Risk</h3>
                <div class="value">{{ summary.verdict_counts.HIGH | default(0) }}</div>
            </div>
            <div class="summary-card">
                <h3>Medium Risk</h3>
                <div class="value">{{ summary.verdict_counts.MEDIUM | default(0) }}</div>
            </div>
            <div class="summary-card">
                <h3>Low Risk</h3>
                <div class="value">{{ summary.verdict_counts.LOW | default(0) }}</div>
            </div>
            <div class="summary-card">
                <h3>Runtime</h3>
                <div class="value">{{ "%.2f" | format(scan_info.runtime_seconds) }}s</div>
            </div>
        </div>
        
        <h2>Scan Information</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Timestamp</td>
                <td>{{ scan_info.timestamp }}</td>
            </tr>
            <tr>
                <td>Index Type</td>
                <td>{{ scan_info.index_type }}</td>
            </tr>
            <tr>
                <td>Metric</td>
                <td>{{ scan_info.metric }}</td>
            </tr>
            <tr>
                <td>k (nearest neighbors)</td>
                <td>{{ scan_info.k }}</td>
            </tr>
        </table>
        
        {% if concept_aware_enabled %}
        <div class="feature-section">
            <h3>Concept-Aware Detection</h3>
            <p>Detected hubs localized to specific semantic concepts. Threshold: {{ concept_threshold }}</p>
            <div class="summary">
                <div class="summary-card concept">
                    <h3>Concepts</h3>
                    <div class="value">{{ num_concepts }}</div>
                </div>
                <div class="summary-card concept">
                    <h3>Concept-Specific Hubs</h3>
                    <div class="value">{{ concept_hub_count }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if modality_aware_enabled %}
        <div class="feature-section">
            <h3>Modality-Aware Detection</h3>
            <p>Detected cross-modal hubs that appear in queries of different modalities.</p>
            <div class="summary">
                <div class="summary-card modality">
                    <h3>Modalities Found</h3>
                    <div class="value">{{ modalities_found | join(', ') }}</div>
                </div>
                <div class="summary-card modality">
                    <h3>Cross-Modal Docs</h3>
                    <div class="value">{{ cross_modal_count }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <h2>Top Suspicious Documents</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Doc Index</th>
                    <th>Risk Score</th>
                    <th>Verdict</th>
                    <th>Hub Z-Score</th>
                    <th>Hub Rate</th>
                    {% if concept_aware_enabled %}
                    <th>Concept</th>
                    {% endif %}
                    {% if modality_aware_enabled %}
                    <th>Cross-Modal</th>
                    {% endif %}
                    {% if show_metadata %}
                    <th>Doc ID</th>
                    <th>Source</th>
                    {% endif %}
                </tr>
            </thead>
            <tbody>
                {% for doc in suspicious_documents[:50] %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ doc.doc_index }}</td>
                    <td class="score">{{ "%.4f" | format(doc.risk_score) }}</td>
                    <td>
                        <span class="verdict-badge verdict-{{ doc.verdict.lower() }}">
                            {{ doc.verdict }}
                        </span>
                    </td>
                    <td class="score">
                        {% if doc.hubness %}
                            {{ "%.2f" | format(doc.hubness.hub_z) }}
                        {% else %}
                            -
                        {% endif %}
                    </td>
                    <td>
                        {% if doc.hubness and doc.hubness.hub_rate %}
                            {{ "%.4f" | format(doc.hubness.hub_rate) }}
                        {% else %}
                            -
                        {% endif %}
                    </td>
                    {% if concept_aware_enabled %}
                    <td>
                        {% if doc.hubness and doc.hubness.concept_specific %}
                            <span class="concept-badge">
                                {{ doc.hubness.concept_specific.top_concept_name | default('N/A') }}
                            </span>
                            <br>
                            <small>z={{ "%.1f" | format(doc.hubness.concept_specific.max_concept_hub_z | default(0)) }}</small>
                        {% else %}
                            -
                        {% endif %}
                    </td>
                    {% endif %}
                    {% if modality_aware_enabled %}
                    <td>
                        {% if doc.hubness and doc.hubness.modality_specific %}
                            {% if doc.hubness.modality_specific.is_cross_modal %}
                                <span class="modality-badge modality-cross">Cross-Modal</span>
                            {% else %}
                                <span class="modality-badge modality-same">Same</span>
                            {% endif %}
                        {% else %}
                            -
                        {% endif %}
                    </td>
                    {% endif %}
                    {% if show_metadata %}
                    <td>{{ doc.metadata.doc_id if doc.metadata and doc.metadata.doc_id else "-" }}</td>
                    <td>{{ doc.metadata.source if doc.metadata and doc.metadata.source else "-" }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generated by HubScan on {{ scan_info.timestamp }}</p>
            <p>For more details, see the JSON report.</p>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(
    config: Config,
    detector_results: Dict[str, DetectorResult],
    combined_scores: np.ndarray,
    verdicts: Dict[int, Verdict],
    metadata: Optional[Metadata] = None,
    num_queries: int = 0,
    runtime_seconds: float = 0.0,
    num_docs: int = 0,
    detection_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML report.
    
    Args:
        config: Configuration
        detector_results: Detector results
        combined_scores: Combined risk scores
        verdicts: Verdicts per document
        metadata: Document metadata
        num_queries: Number of queries processed
        runtime_seconds: Runtime in seconds
        num_docs: Number of documents
        detection_metrics: Optional detection performance metrics
        
    Returns:
        HTML report string
    """
    # Generate JSON report first (contains all data)
    json_report = generate_json_report(
        config, detector_results, combined_scores, verdicts,
        metadata, num_queries, runtime_seconds, num_docs,
        detection_metrics=detection_metrics,
    )
    
    # Extract concept/modality info from detector summary
    concept_aware_enabled = False
    modality_aware_enabled = False
    concept_threshold = 4.0
    num_concepts = 0
    concept_hub_count = 0
    cross_modal_count = 0
    modalities_found = []
    
    hubness_summary = json_report.get("detector_summary", {}).get("hubness", {})
    if hubness_summary:
        concept_info = hubness_summary.get("concept_aware", {})
        if concept_info.get("enabled", False):
            concept_aware_enabled = True
            concept_threshold = concept_info.get("threshold", 4.0)
            num_concepts = concept_info.get("num_concepts", 0)
            # Count concept hubs from concept_summary
            concept_summary = json_report.get("concept_summary", {})
            for cid, stats in concept_summary.items():
                concept_hub_count += stats.get("num_hub_docs", 0)
        
        modality_info = hubness_summary.get("modality_aware", {})
        if modality_info.get("enabled", False):
            modality_aware_enabled = True
            cross_modal_count = modality_info.get("num_cross_modal_docs", 0)
            modalities_found = modality_info.get("modalities_found", [])
    
    # Render HTML template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        scan_info=json_report["scan_info"],
        summary=json_report["summary"],
        suspicious_documents=json_report["suspicious_documents"],
        show_metadata=not config.output.privacy_mode,
        concept_aware_enabled=concept_aware_enabled,
        modality_aware_enabled=modality_aware_enabled,
        concept_threshold=concept_threshold,
        num_concepts=num_concepts,
        concept_hub_count=concept_hub_count,
        cross_modal_count=cross_modal_count,
        modalities_found=modalities_found,
    )
    
    return html


def save_html_report(html: str, output_path: str):
    """Save HTML report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        f.write(html)

