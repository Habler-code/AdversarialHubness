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

"""
End-to-end demonstration: Generate documents, split for RAG, create embeddings,
plant an adversarial hub, and detect it with HubScan.

This example shows:
1. Generating a corpus of documents
2. Splitting documents into chunks for RAG
3. Creating embeddings for chunks
4. Planting an adversarial hub (malicious document)
5. Running HubScan to detect the hub
6. Identifying exactly which document/chunk is the adversarial hub
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib


def generate_documents(num_docs: int = 100) -> List[Dict[str, Any]]:
    """
    Generate a corpus of documents.
    
    Returns:
        List of documents with id, title, content, and source
    """
    documents = []
    
    topics = [
        "machine learning", "data science", "python programming",
        "web development", "database design", "cloud computing",
        "cybersecurity", "software engineering", "artificial intelligence",
        "natural language processing"
    ]
    
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        doc = {
            "doc_id": f"doc_{i:04d}",
            "title": f"Document {i}: Introduction to {topic.title()}",
            "content": f"""
This is document {i} about {topic}. 
It contains information about {topic} and its applications.
{topic.title()} is an important field in modern technology.
This document provides an overview of {topic} concepts and practices.
Many organizations use {topic} to solve complex problems.
Understanding {topic} requires knowledge of fundamental principles.
            """.strip(),
            "source": f"source_{i % 5}",
            "topic": topic
        }
        documents.append(doc)
    
    return documents


def split_documents_for_rag(documents: List[Dict[str, Any]], chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Split documents into chunks for RAG.
    
    Args:
        documents: List of documents
        chunk_size: Target chunk size in characters
        
    Returns:
        List of chunks with metadata linking back to original documents
    """
    chunks = []
    chunk_idx = 0
    
    for doc in documents:
        content = doc["content"]
        words = content.split()
        
        # Simple chunking by words
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk = {
                    "chunk_id": f"chunk_{chunk_idx:04d}",
                    "doc_id": doc["doc_id"],
                    "doc_index": doc["doc_index"],
                    "chunk_index": len([c for c in chunks if c["doc_id"] == doc["doc_id"]]),
                    "text": chunk_text,
                    "title": doc["title"],
                    "source": doc["source"],
                    "topic": doc["topic"],
                    "text_hash": hashlib.md5(chunk_text.encode()).hexdigest()
                }
                chunks.append(chunk)
                chunk_idx += 1
                
                # Start new chunk
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        # Add remaining words as final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = {
                "chunk_id": f"chunk_{chunk_idx:04d}",
                "doc_id": doc["doc_id"],
                "doc_index": doc["doc_index"],
                "chunk_index": len([c for c in chunks if c["doc_id"] == doc["doc_id"]]),
                "text": chunk_text,
                "title": doc["title"],
                "source": doc["source"],
                "topic": doc["topic"],
                "text_hash": hashlib.md5(chunk_text.encode()).hexdigest()
            }
            chunks.append(chunk)
            chunk_idx += 1
    
    return chunks


def create_embeddings(chunks: List[Dict[str, Any]], embedding_dim: int = 128) -> np.ndarray:
    """
    Create embeddings for chunks.
    
    In a real scenario, you would use a proper embedding model.
    Here we simulate embeddings with random vectors.
    
    Args:
        chunks: List of text chunks
        embedding_dim: Embedding dimension
        
    Returns:
        Embeddings array (N, D)
    """
    np.random.seed(42)
    num_chunks = len(chunks)
    embeddings = np.random.randn(num_chunks, embedding_dim).astype(np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings


def plant_adversarial_hub(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    hub_chunk_idx: int,
    num_query_clusters: int = 50,
    strength: float = 0.85
) -> tuple:
    """
    Plant an adversarial hub by creating a malicious embedding.
    
    The adversarial hub is an embedding that is close to many diverse
    query vectors, making it appear in top-k results for many queries.
    
    Strategy: Create a hub that is close to many diverse document embeddings,
    which will make it appear in top-k for queries that are similar to those documents.
    
    Args:
        chunks: List of chunks
        embeddings: Embeddings array
        hub_chunk_idx: Index of chunk to turn into adversarial hub
        num_query_clusters: Number of query clusters to be close to
        strength: How close to be to cluster centroids (0-1)
        
    Returns:
        Tuple of (modified chunks, modified embeddings, hub_info)
    """
    # Strategy: Make the hub close to many diverse document embeddings
    # This ensures it appears in top-k for queries similar to those documents
    
    # Sample diverse document embeddings to be close to
    np.random.seed(123)
    num_docs_to_target = min(num_query_clusters, len(embeddings))
    target_indices = np.random.choice(len(embeddings), num_docs_to_target, replace=False)
    target_embeddings = embeddings[target_indices]
    
    # Create adversarial hub as weighted average of target embeddings
    # Weight by how diverse they are (farther from each other = more weight)
    weights = np.ones(num_docs_to_target)
    adversarial_embedding = np.average(target_embeddings, axis=0, weights=weights)
    adversarial_embedding = adversarial_embedding / np.linalg.norm(adversarial_embedding)
    
    # Refine to be closer to more targets
    for iteration in range(10):
        # Compute similarities to all target embeddings
        similarities = target_embeddings @ adversarial_embedding
        
        # Focus on targets we're not close enough to
        threshold = strength - 0.1 * iteration  # Gradually increase threshold
        far_targets = target_embeddings[similarities < threshold]
        
        if len(far_targets) > 0:
            # Move towards far targets
            far_avg = np.mean(far_targets, axis=0)
            far_avg = far_avg / np.linalg.norm(far_avg)
            # Blend: move towards far targets but keep some of current position
            blend_factor = 0.3 / (iteration + 1)  # Less aggressive as we iterate
            adversarial_embedding = (1 - blend_factor) * adversarial_embedding + blend_factor * far_avg
            adversarial_embedding = adversarial_embedding / np.linalg.norm(adversarial_embedding)
        else:
            break
    
    # Final check: ensure we're close to many targets
    final_similarities = target_embeddings @ adversarial_embedding
    close_count = np.sum(final_similarities >= strength)
    
    # Replace the target chunk's embedding
    modified_embeddings = embeddings.copy()
    modified_embeddings[hub_chunk_idx] = adversarial_embedding.astype(np.float32)
    
    # Mark the chunk as adversarial
    modified_chunks = chunks.copy()
    modified_chunks[hub_chunk_idx]["is_adversarial"] = True
    modified_chunks[hub_chunk_idx]["adversarial_note"] = "PLANTED ADVERSARIAL HUB - This chunk was artificially created to be close to many query clusters"
    
    # Update chunk text to make it obvious
    modified_chunks[hub_chunk_idx]["text"] = (
        modified_chunks[hub_chunk_idx]["text"] + 
        " [ADVERSARIAL HUB: This chunk embedding was artificially optimized to appear in top-k results for many diverse queries.]"
    )
    
    hub_info = {
        "chunk_index": hub_chunk_idx,
        "chunk_id": chunks[hub_chunk_idx]["chunk_id"],
        "doc_id": chunks[hub_chunk_idx]["doc_id"],
        "text": chunks[hub_chunk_idx]["text"][:100] + "...",
        "embedding_norm": float(np.linalg.norm(adversarial_embedding)),
        "num_query_clusters_targeted": num_query_clusters,
        "num_targets_close": int(close_count),
        "strength": strength
    }
    
    return modified_chunks, modified_embeddings, hub_info


def run_demo():
    """Run the complete demonstration."""
    print("=" * 80)
    print("HubScan Adversarial Hub Detection Demo")
    print("=" * 80)
    
    # Step 1: Generate documents
    print("\n[Step 1] Generating documents...")
    documents = generate_documents(num_docs=50)
    for i, doc in enumerate(documents):
        doc["doc_index"] = i
    print(f"Generated {len(documents)} documents")
    
    # Step 2: Split into chunks for RAG
    print("\n[Step 2] Splitting documents into chunks for RAG...")
    chunks = split_documents_for_rag(documents, chunk_size=150)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"Average chunks per document: {len(chunks) / len(documents):.1f}")
    
    # Step 3: Create embeddings
    print("\n[Step 3] Creating embeddings for chunks...")
    embeddings = create_embeddings(chunks, embedding_dim=128)
    print(f"Created embeddings: shape {embeddings.shape}")
    
    # Step 4: Plant adversarial hub
    print("\n[Step 4] Planting adversarial hub...")
    hub_chunk_idx = len(chunks) // 2  # Plant in the middle
    chunks, embeddings, hub_info = plant_adversarial_hub(
        chunks, embeddings, hub_chunk_idx,
        num_query_clusters=50,
        strength=0.80
    )
    
    print(f"\nAdversarial hub planted at:")
    print(f"  Chunk Index: {hub_info['chunk_index']}")
    print(f"  Chunk ID: {hub_info['chunk_id']}")
    print(f"  Document ID: {hub_info['doc_id']}")
    print(f"  Text Preview: {hub_info['text']}")
    print(f"  Targeting {hub_info['num_query_clusters_targeted']} query clusters")
    
    # Step 5: Save data
    print("\n[Step 5] Saving data...")
    output_dir = Path(__file__).parent.parent / "demo_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / "chunk_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    
    # Save metadata
    metadata = {
        "chunk_id": [c["chunk_id"] for c in chunks],
        "doc_id": [c["doc_id"] for c in chunks],
        "doc_index": [c["doc_index"] for c in chunks],
        "chunk_index": [c["chunk_index"] for c in chunks],
        "text": [c["text"] for c in chunks],
        "title": [c["title"] for c in chunks],
        "source": [c["source"] for c in chunks],
        "topic": [c["topic"] for c in chunks],
        "text_hash": [c["text_hash"] for c in chunks],
        "is_adversarial": [c.get("is_adversarial", False) for c in chunks]
    }
    
    metadata_path = output_dir / "chunk_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Step 6: Run HubScan
    print("\n[Step 6] Running HubScan to detect adversarial hub...")
    print("-" * 80)
    
    # Add parent directory to path
    demo_path = Path(__file__).parent.parent
    if str(demo_path) not in sys.path:
        sys.path.insert(0, str(demo_path))
    
    from hubscan import scan, get_suspicious_documents, explain_document, Verdict
    
    results = scan(
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
        k=10,
        num_queries=300,
        output_dir=str(output_dir / "reports"),
        thresholds__hub_z=2.0,  # Lower threshold to catch the hub
        thresholds__percentile=0.10  # Top 10% instead of 0.1%
    )
    
    print(f"\nScan completed in {results['runtime']:.2f} seconds")
    print(f"Processed {len(chunks)} chunks with {200} queries")
    
    # Step 7: Identify the adversarial hub
    print("\n[Step 7] Identifying adversarial hub...")
    print("-" * 80)
    
    # Get all suspicious documents (not just HIGH)
    all_suspicious = get_suspicious_documents(results, verdict=None, top_k=20)
    high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
    
    print(f"\nFound {len(high_risk)} HIGH-risk chunks (out of {len(all_suspicious)} total suspicious):")
    print()
    
    found_adversarial = False
    adversarial_chunk_idx = None
    
    # First, check all suspicious chunks for the adversarial marker
    print("Scanning all suspicious chunks for adversarial hub...")
    for doc in all_suspicious:
        chunk_idx = doc["doc_index"]
        if chunk_idx < len(chunks) and chunks[chunk_idx].get("is_adversarial"):
            found_adversarial = True
            adversarial_chunk_idx = chunk_idx
            print(f"FOUND: Adversarial hub detected at chunk index {chunk_idx} (Risk Score: {doc['risk_score']:.4f}, Verdict: {doc['verdict']})")
            break
    
    if not found_adversarial:
        # Check the actual planted chunk directly
        planted_idx = hub_info["chunk_index"]
        if planted_idx < len(chunks) and chunks[planted_idx].get("is_adversarial"):
            # Check if it's in results at all
            for doc in all_suspicious:
                if doc["doc_index"] == planted_idx:
                    found_adversarial = True
                    adversarial_chunk_idx = planted_idx
                    print(f"FOUND: Adversarial hub at planted index {planted_idx} (Risk Score: {doc['risk_score']:.4f}, Verdict: {doc['verdict']})")
                    break
        
        # Also check nearby chunks
        if not found_adversarial:
            print(f"Checking chunks near planted index {hub_info['chunk_index']}...")
            for offset in [-3, -2, -1, 1, 2, 3]:
                check_idx = hub_info["chunk_index"] + offset
                if 0 <= check_idx < len(chunks):
                    chunk = chunks[check_idx]
                    if chunk.get("is_adversarial"):
                        print(f"FOUND: Adversarial hub at nearby chunk index: {check_idx}")
                        found_adversarial = True
                        adversarial_chunk_idx = check_idx
                        break
    
    print(f"\nTop HIGH-risk chunks:")
    print()
    
    for i, doc in enumerate(high_risk, 1):
        chunk_idx = doc["doc_index"]
        is_actual_adversarial = (chunk_idx == adversarial_chunk_idx) if adversarial_chunk_idx is not None else False
        
        marker = " <-- ADVERSARIAL HUB DETECTED!" if is_actual_adversarial else ""
        if is_actual_adversarial:
            found_adversarial = True
        
        print(f"{i}. Chunk Index: {chunk_idx}")
        
        # Get metadata from chunks list
        chunk_meta = chunks[chunk_idx] if chunk_idx < len(chunks) else {}
        print(f"   Chunk ID: {chunk_meta.get('chunk_id', 'N/A')}")
        print(f"   Document ID: {chunk_meta.get('doc_id', doc.get('metadata', {}).get('doc_id', 'N/A'))}")
        print(f"   Risk Score: {doc['risk_score']:.4f}")
        print(f"   Verdict: {doc['verdict']}")
        
        if "hubness" in doc:
            hub = doc["hubness"]
            print(f"   Hub Z-Score: {hub.get('hub_z', 'N/A'):.2f}")
            print(f"   Hub Rate: {hub.get('hub_rate', 'N/A'):.4f}")
            print(f"   Hits: {hub.get('hits', 'N/A')}")
        
        # Show chunk text
        chunk_text = chunk_meta.get("text", "")
        preview = chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
        print(f"   Text Preview: {preview}")
        if chunk_meta.get("is_adversarial"):
            print(f"   [ADVERSARIAL MARKER PRESENT]")
        print(f"{marker}")
        print()
    
    # Show the actual planted hub details
    planted_idx = hub_info["chunk_index"]
    print(f"\nPlanted adversarial hub details:")
    print(f"  Chunk Index: {planted_idx}")
    print(f"  Chunk ID: {hub_info['chunk_id']}")
    print(f"  Document ID: {hub_info['doc_id']}")
    
    # Check if it's in all_suspicious
    found_in_results = False
    for doc in all_suspicious:
        if doc["doc_index"] == planted_idx:
            found_in_results = True
            print(f"  Status: Found in scan results")
            print(f"  Risk Score: {doc['risk_score']:.4f}")
            print(f"  Verdict: {doc['verdict']}")
            if "hubness" in doc:
                hub = doc["hubness"]
                print(f"  Hub Z-Score: {hub.get('hub_z', 'N/A'):.2f}")
                print(f"  Hub Rate: {hub.get('hub_rate', 'N/A'):.4f}")
                print(f"  Hits: {hub.get('hits', 'N/A')}")
            break
    
    if not found_in_results:
        print(f"  Status: Not found in top suspicious chunks")
        print(f"  Note: May need stronger hub or adjusted thresholds")
    
    # Step 8: Detailed explanation
    # Use the planted hub index if we found it, otherwise use the detected one
    explanation_idx = adversarial_chunk_idx if adversarial_chunk_idx is not None else hub_info["chunk_index"]
    
    if found_adversarial or explanation_idx is not None:
        print("\n[Step 8] Detailed explanation of adversarial hub...")
        print("-" * 80)
        
        explanation = explain_document(results, doc_index=explanation_idx)
        if explanation:
            print(f"\nChunk Index: {hub_info['chunk_index']}")
            print(f"Chunk ID: {hub_info['chunk_id']}")
            print(f"Document ID: {hub_info['doc_id']}")
            print(f"\nRisk Analysis:")
            print(f"  Risk Score: {explanation['risk_score']:.4f}")
            print(f"  Verdict: {explanation['verdict']}")
            
            if "hubness" in explanation:
                hub = explanation["hubness"]
                print(f"\nHubness Metrics:")
                print(f"  Z-Score: {hub.get('hub_z', 'N/A'):.2f} (higher = more suspicious)")
                print(f"  Hub Rate: {hub.get('hub_rate', 'N/A'):.4f} (fraction of queries retrieving this chunk)")
                print(f"  Hits: {hub.get('hits', 'N/A')} (number of queries that retrieved this chunk)")
            
            # Show full chunk text
            if explanation_idx < len(chunks):
                chunk = chunks[explanation_idx]
                print(f"\nFull Chunk Text:")
                print(f"  Title: {chunk['title']}")
                print(f"  Source: {chunk['source']}")
                print(f"  Topic: {chunk['topic']}")
                print(f"  Text: {chunk['text']}")
                
                if chunk.get("is_adversarial"):
                    print(f"\n  [ADVERSARIAL MARKER] This chunk was artificially created as an adversarial hub")
                    print(f"  It was designed to be close to {hub_info['num_query_clusters_targeted']} different query clusters")
                    print(f"  This makes it appear in top-k results for many diverse queries")
                    print(f"  Original chunk index: {hub_info['chunk_index']}")
                    if adversarial_chunk_idx is not None and adversarial_chunk_idx != hub_info["chunk_index"]:
                        print(f"  Detected at chunk index: {adversarial_chunk_idx}")
                    else:
                        print(f"  Detected at chunk index: {explanation_idx}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total chunks analyzed: {len(chunks)}")
    print(f"Adversarial hub planted at chunk index: {hub_info['chunk_index']}")
    print(f"HubScan detected {len(high_risk)} HIGH-risk chunks")
    print(f"Adversarial hub found: {'YES' if found_adversarial else 'NO'}")
    
    print("\n" + "=" * 80)
    print("Final Result")
    print("=" * 80)
    
    if found_adversarial and adversarial_chunk_idx is not None:
        print("\nSUCCESS: HubScan identified the adversarial hub!")
        detected_chunk = chunks[adversarial_chunk_idx]
        print(f"\nThe malicious chunk is:")
        print(f"  Chunk Index: {adversarial_chunk_idx}")
        print(f"  Chunk ID: {detected_chunk.get('chunk_id', 'N/A')}")
        print(f"  Document ID: {detected_chunk.get('doc_id', 'N/A')}")
        print(f"  Original planted at: {hub_info['chunk_index']} ({hub_info['chunk_id']})")
        
        # Get explanation
        explanation = explain_document(results, doc_index=adversarial_chunk_idx)
        if explanation:
            print(f"\nDetection Metrics:")
            print(f"  Risk Score: {explanation['risk_score']:.4f}")
            print(f"  Verdict: {explanation['verdict']}")
            if "hubness" in explanation:
                hub = explanation["hubness"]
                print(f"  Hub Z-Score: {hub.get('hub_z', 'N/A'):.2f}")
                print(f"  Hub Rate: {hub.get('hub_rate', 'N/A'):.4f}")
                print(f"  Hits: {hub.get('hits', 'N/A')} out of {results['json_report']['scan_info']['num_queries']} queries")
    else:
        print("\nRESULT: Adversarial hub was planted but not flagged as HIGH-risk.")
        print(f"\nPlanted at chunk index: {hub_info['chunk_index']}")
        print(f"Chunk ID: {hub_info['chunk_id']}")
        print(f"Document ID: {hub_info['doc_id']}")
        
        # Check if it was detected at all
        explanation = explain_document(results, doc_index=hub_info["chunk_index"])
        if explanation:
            print(f"\nDetection Status:")
            print(f"  Found in results: YES")
            print(f"  Risk Score: {explanation['risk_score']:.4f}")
            print(f"  Verdict: {explanation['verdict']}")
            if "hubness" in explanation:
                hub = explanation["hubness"]
                print(f"  Hub Z-Score: {hub.get('hub_z', 'N/A'):.2f}")
                print(f"  Hub Rate: {hub.get('hub_rate', 'N/A'):.4f}")
                print(f"  Hits: {hub.get('hits', 'N/A')} queries")
            print(f"\nNote: Hub was detected but with {explanation['verdict']} verdict.")
            print(f"To flag as HIGH-risk, increase hub strength or lower thresholds.")
        else:
            print(f"\nDetection Status: Not found in top suspicious chunks")
            print(f"Try increasing hub strength or adjusting detection parameters.")
    
    print(f"\nReports saved to: {output_dir / 'reports'}")
    print(f"  - JSON: {output_dir / 'reports' / 'report.json'}")
    print(f"  - HTML: {output_dir / 'reports' / 'report.html'}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()

