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
Generate toy test data for HubScan with planted adversarial hubs.

This script creates a structured dataset similar to the adversarial_hub_demo,
with document topics and well-designed adversarial hubs that can be detected
with high precision.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_docs = 200  # Number of documents (similar scale to demo)
embedding_dim = 128
num_hubs = 5  # Number of adversarial hubs to plant

# Topics for document generation (creates semantic clusters)
TOPICS = [
    "machine learning", "data science", "python programming",
    "web development", "database design", "cloud computing",
    "cybersecurity", "software engineering", "artificial intelligence",
    "natural language processing", "computer vision", "deep learning",
    "distributed systems", "microservices", "DevOps", "containerization",
    "blockchain", "quantum computing", "edge computing", "IoT"
]


def generate_documents(num_docs: int) -> List[Dict[str, Any]]:
    """Generate a corpus of documents with topics."""
    documents = []
    
    for i in range(num_docs):
        topic = TOPICS[i % len(TOPICS)]
        doc = {
            "doc_id": f"doc_{i:04d}",
            "doc_index": i,
            "title": f"Document {i}: Introduction to {topic.title()}",
            "text": f"This is document {i} about {topic}. It contains information about {topic} and its applications. {topic.title()} is an important field in modern technology.",
            "source": f"source_{i % 5}",
            "topic": topic,
            "text_hash": hashlib.md5(f"doc_{i}_{topic}".encode()).hexdigest(),
            "is_adversarial": False
        }
        documents.append(doc)
    
    return documents


def create_topic_embeddings(documents: List[Dict], embedding_dim: int) -> np.ndarray:
    """
    Create embeddings with topic-based clustering.
    Documents with same topic will have similar embeddings.
    """
    # Create a base embedding for each topic
    topic_embeddings = {}
    for topic in TOPICS:
        base = np.random.randn(embedding_dim).astype(np.float32)
        base = base / np.linalg.norm(base)
        topic_embeddings[topic] = base
    
    # Create document embeddings as variations of topic embeddings
    embeddings = np.zeros((len(documents), embedding_dim), dtype=np.float32)
    
    for i, doc in enumerate(documents):
        topic = doc["topic"]
        base = topic_embeddings[topic]
        
        # Add small random variation (keep documents in same topic close)
        noise = np.random.randn(embedding_dim).astype(np.float32) * 0.3
        embedding = base + noise
        embedding = embedding / np.linalg.norm(embedding)
        embeddings[i] = embedding
    
    return embeddings


def plant_adversarial_hub(
    embeddings: np.ndarray,
    hub_idx: int,
    num_targets: int = 50,
    strength: float = 0.7
) -> np.ndarray:
    """
    Plant an adversarial hub by making it close to many diverse documents.
    
    This uses the same algorithm as the adversarial_hub_demo.py.
    """
    np.random.seed(123 + hub_idx)
    
    # Sample diverse document embeddings to be close to
    num_docs_to_target = min(num_targets, len(embeddings))
    target_indices = np.random.choice(len(embeddings), num_docs_to_target, replace=False)
    target_embeddings = embeddings[target_indices]
    
    # Create adversarial hub as weighted average of target embeddings
    weights = np.ones(num_docs_to_target)
    adversarial_embedding = np.average(target_embeddings, axis=0, weights=weights)
    adversarial_embedding = adversarial_embedding / np.linalg.norm(adversarial_embedding)
    
    # Refine to be closer to more targets (iterative improvement)
    for iteration in range(10):
        # Compute similarities to all target embeddings
        similarities = target_embeddings @ adversarial_embedding
        
        # Focus on targets we're not close enough to
        threshold = strength - 0.1 * iteration
        far_targets = target_embeddings[similarities < threshold]
        
        if len(far_targets) > 0:
            # Move towards far targets
            far_avg = np.mean(far_targets, axis=0)
            far_avg = far_avg / np.linalg.norm(far_avg)
            # Blend: move towards far targets but keep some of current position
            blend_factor = 0.3 / (iteration + 1)
            adversarial_embedding = (1 - blend_factor) * adversarial_embedding + blend_factor * far_avg
            adversarial_embedding = adversarial_embedding / np.linalg.norm(adversarial_embedding)
        else:
            break
    
    return adversarial_embedding.astype(np.float32)


def main():
    print(f"Generating {num_docs} documents with topic structure...")
    
    # Step 1: Generate documents
    documents = generate_documents(num_docs)
    print(f"  Created {len(documents)} documents across {len(TOPICS)} topics")
    
    # Step 2: Create topic-based embeddings
    print("Creating topic-based embeddings...")
    embeddings = create_topic_embeddings(documents, embedding_dim)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Step 3: Select hub indices (spread across different topics)
    print(f"\nPlanting {num_hubs} adversarial hubs...")
    hub_indices = []
    for i in range(num_hubs):
        # Spread hubs across the dataset
        hub_idx = (i * num_docs // num_hubs) + (num_docs // (2 * num_hubs))
        hub_indices.append(hub_idx)
    hub_indices = np.array(hub_indices)
    
    # Step 4: Plant adversarial hubs
    for i, hub_idx in enumerate(hub_indices):
        # Create adversarial embedding
        adversarial_embedding = plant_adversarial_hub(
            embeddings, 
            hub_idx, 
            num_targets=50,  # Target 50 diverse documents
            strength=0.7
        )
        embeddings[hub_idx] = adversarial_embedding
        
        # Mark as adversarial in metadata
        documents[hub_idx]["is_adversarial"] = True
        documents[hub_idx]["text"] = f"[ADVERSARIAL HUB] {documents[hub_idx]['text']}"
        
        # Verify hub strength
        similarities = embeddings @ adversarial_embedding
        close_count = np.sum(similarities >= 0.5)
        print(f"  Hub {i+1}/{num_hubs} at index {hub_idx}: close to {close_count} documents")
    
    # Step 5: Save data
    print("\nSaving data...")
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = data_dir / "toy_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"  Saved embeddings to {embeddings_path}")
    
    # Convert documents to columnar metadata format
    metadata = {
        "doc_id": [d["doc_id"] for d in documents],
        "doc_index": [d["doc_index"] for d in documents],
        "title": [d["title"] for d in documents],
        "text": [d["text"] for d in documents],
        "source": [d["source"] for d in documents],
        "topic": [d["topic"] for d in documents],
        "text_hash": [d["text_hash"] for d in documents],
        "is_adversarial": [d["is_adversarial"] for d in documents]
    }
    
    metadata_path = data_dir / "toy_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TOY DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Documents: {num_docs}")
    print(f"Topics: {len(TOPICS)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Adversarial hubs: {num_hubs}")
    print(f"Hub indices: {hub_indices.tolist()}")
    print(f"\nRun detection with:")
    print(f"  hubscan scan --config examples/configs/toy_config.yaml")


if __name__ == "__main__":
    main()
