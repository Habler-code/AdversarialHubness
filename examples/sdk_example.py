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

"""Example usage of HubScan SDK."""

import numpy as np
from hubscan.sdk import scan, quick_scan, get_suspicious_documents, explain_document, Verdict


def example_basic_scan():
    """Example: Basic scan with file paths."""
    print("=" * 60)
    print("Example 1: Basic Scan")
    print("=" * 60)
    
    results = scan(
        embeddings_path="examples/toy_embeddings.npy",
        metadata_path="examples/toy_metadata.json",
        k=10,
        num_queries=100,
        output_dir="examples/reports/"
    )
    
    print(f"Scan completed in {results['runtime']:.2f} seconds")
    print(f"Found {len(results['verdicts'])} documents")
    
    # Get high-risk documents
    high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=5)
    print(f"\nTop 5 high-risk documents:")
    for doc in high_risk:
        print(f"  Doc {doc['doc_index']}: Risk={doc['risk_score']:.4f}, Verdict={doc['verdict']}")


def example_quick_scan():
    """Example: Quick scan on in-memory embeddings."""
    print("\n" + "=" * 60)
    print("Example 2: Quick Scan (In-Memory)")
    print("=" * 60)
    
    # Generate sample embeddings
    embeddings = np.random.randn(500, 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    results = quick_scan(
        embeddings=embeddings,
        k=10,
        num_queries=50
    )
    
    print(f"Quick scan completed in {results['runtime']:.2f} seconds")
    
    # Get suspicious documents
    suspicious = get_suspicious_documents(results, top_k=10)
    print(f"\nTop 10 suspicious documents:")
    for doc in suspicious:
        print(f"  Doc {doc['doc_index']}: Risk={doc['risk_score']:.4f}")


def example_explain_document():
    """Example: Explain why a document was flagged."""
    print("\n" + "=" * 60)
    print("Example 3: Explain Document")
    print("=" * 60)
    
    # Run scan first
    results = scan(
        embeddings_path="examples/toy_embeddings.npy",
        k=10,
        num_queries=100,
        output_dir="examples/reports/"
    )
    
    # Get top suspicious document
    suspicious = get_suspicious_documents(results, top_k=1)
    if suspicious:
        doc_idx = suspicious[0]["doc_index"]
        
        # Explain it
        explanation = explain_document(results, doc_idx)
        if explanation:
            print(f"\nExplanation for document {doc_idx}:")
            print(f"  Risk Score: {explanation['risk_score']:.4f}")
            print(f"  Verdict: {explanation['verdict']}")
            
            if "hubness" in explanation:
                hub = explanation["hubness"]
                print(f"  Hub Z-Score: {hub.get('hub_z', 'N/A'):.2f}")
                print(f"  Hub Rate: {hub.get('hub_rate', 'N/A'):.4f}")


def example_custom_config():
    """Example: Using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    results = scan(
        embeddings_path="examples/toy_embeddings.npy",
        k=20,
        num_queries=500,
        detectors__hubness__enabled=True,
        detectors__cluster_spread__enabled=True,
        detectors__stability__enabled=False,
        thresholds__policy="hybrid",
        thresholds__hub_z=5.0,
    )
    
    print(f"Custom scan completed")
    print(f"Verdicts: {results['json_report']['summary']['verdict_counts']}")


if __name__ == "__main__":
    # Make sure toy data exists
    import os
    if not os.path.exists("examples/toy_embeddings.npy"):
        print("Generating toy data first...")
        from generate_toy_data import *
        exec(open("examples/generate_toy_data.py").read())
    
    example_basic_scan()
    example_quick_scan()
    example_explain_document()
    example_custom_config()

