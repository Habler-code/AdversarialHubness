#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Create RAG benchmark from Wikipedia articles.

This script:
1. Downloads Wikipedia articles across diverse topics
2. Chunks articles for RAG (200-500 tokens)
3. Creates embeddings using sentence-transformers
4. Saves benchmark dataset for hub planting and detection
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


# Diverse Wikipedia topics for benchmark
WIKIPEDIA_TOPICS = {
    "small": [
        "Machine learning", "Natural language processing", "Computer vision",
        "Artificial intelligence", "Deep learning", "Neural network",
        "Python (programming language)", "Data science", "Cloud computing",
        "Cybersecurity", "Database", "Web development",
        "Quantum computing", "Blockchain", "Internet of things",
        "Operating system", "Computer network", "Algorithm",
        "Software engineering", "Cryptography", "Distributed computing",
        "Virtual reality", "Augmented reality", "Robotics",
        "5G", "Edge computing", "DevOps",
        "Microservices", "Container (virtualization)", "Kubernetes",
    ],
    "medium": [
        # Add 300 more topics covering: science, technology, history, geography, arts, etc.
        "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
        "Geology", "Meteorology", "Ecology", "Evolution", "Genetics",
        # ... (would add 290 more)
    ],
    "large": [
        # Add 3000+ topics
    ]
}


def download_wikipedia_article(title: str) -> Dict[str, Any]:
    """
    Download Wikipedia article using the MediaWiki API.
    
    Args:
        title: Article title
        
    Returns:
        Dictionary with article data
    """
    # Use Wikipedia API
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": True,
        "inprop": "url",
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract page data
        pages = data["query"]["pages"]
        page_id = list(pages.keys())[0]
        
        if page_id == "-1":
            return None  # Page not found
        
        page = pages[page_id]
        
        return {
            "title": page.get("title", title),
            "content": page.get("extract", ""),
            "url": page.get("fullurl", ""),
            "page_id": page_id,
        }
    except Exception as e:
        print(f"Error downloading '{title}': {e}")
        return None


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Chunk text into overlapping segments.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
        
        if i >= len(words):
            break
    
    return chunks


def create_benchmark_dataset(
    size: str,
    output_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """
    Create benchmark dataset from Wikipedia articles.
    
    Args:
        size: Dataset size ("small", "medium", "large")
        output_dir: Output directory
        model_name: Sentence-transformers model name
        chunk_size: Chunk size in words
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dataset statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get topics for size
    topics = WIKIPEDIA_TOPICS.get(size, WIKIPEDIA_TOPICS["small"])
    
    print(f"Creating {size} benchmark with {len(topics)} Wikipedia articles...")
    
    # Download articles
    print("\n1. Downloading Wikipedia articles...")
    articles = []
    for topic in tqdm(topics, desc="Downloading"):
        article = download_wikipedia_article(topic)
        if article and article["content"]:
            articles.append(article)
    
    print(f"Downloaded {len(articles)} articles")
    
    # Chunk articles
    print("\n2. Chunking articles...")
    chunks = []
    chunk_id = 0
    
    for article in tqdm(articles, desc="Chunking"):
        article_chunks = chunk_text(article["content"], chunk_size, chunk_overlap)
        
        for i, chunk_text in enumerate(article_chunks):
            # Skip very short chunks
            if len(chunk_text.split()) < 20:
                continue
            
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:06d}",
                "article_title": article["title"],
                "article_url": article["url"],
                "article_page_id": article["page_id"],
                "chunk_index": i,
                "text": chunk_text,
                "text_hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                "is_adversarial": False,  # Will be set during hub planting
            })
            chunk_id += 1
    
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\n3. Creating embeddings...")
    if not HAS_SENTENCE_TRANSFORMERS:
        print("ERROR: sentence-transformers not installed. Cannot create embeddings.")
        return None
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    
    # Create embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    print(f"Created embeddings: {embeddings.shape}")
    
    # Save dataset
    print("\n4. Saving dataset...")
    
    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    print(f"Saved embeddings: {output_dir / 'embeddings.npy'}")
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    
    # Save dataset info
    dataset_info = {
        "size": size,
        "num_articles": len(articles),
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "topics": topics,
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset info: {output_dir / 'dataset_info.json'}")
    
    print(f"\n✅ Benchmark dataset created successfully!")
    print(f"   Location: {output_dir}")
    print(f"   Articles: {len(articles)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Embeddings: {embeddings.shape}")
    
    return dataset_info


def main():
    parser = argparse.ArgumentParser(description="Create Wikipedia RAG benchmark")
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Benchmark size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/data/small/",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Chunk size in words",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks in words",
    )
    
    args = parser.parse_args()
    
    create_benchmark_dataset(
        size=args.size,
        output_dir=Path(args.output),
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()

