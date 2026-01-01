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


def download_wikipedia_article(title: str, fetch_categories: bool = True) -> Dict[str, Any]:
    """
    Download Wikipedia article using the MediaWiki API.
    
    Args:
        title: Article title
        fetch_categories: Whether to fetch article categories (for concept detection)
        
    Returns:
        Dictionary with article data
    """
    # Use Wikipedia API
    url = "https://en.wikipedia.org/w/api.php"
    
    # Include categories in the request if requested
    props = "extracts|info"
    if fetch_categories:
        props += "|categories"
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": props,
        "explaintext": True,
        "inprop": "url",
    }
    
    if fetch_categories:
        params["cllimit"] = "50"  # Get up to 50 categories
        params["clshow"] = "!hidden"  # Exclude hidden categories
    
    try:
        headers = {
            "User-Agent": "HubScan-Benchmark/1.0 (Educational; Python/requests)"
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract page data
        pages = data["query"]["pages"]
        page_id = list(pages.keys())[0]
        
        if page_id == "-1":
            return None  # Page not found
        
        page = pages[page_id]
        
        # Extract categories
        categories = []
        if "categories" in page:
            categories = [
                cat["title"].replace("Category:", "") 
                for cat in page["categories"]
                if not any(x in cat["title"].lower() for x in ["stub", "articles", "pages", "wikidata", "wikipedia"])
            ]
        
        # Determine primary concept from categories
        # Use the first meaningful category as the concept
        concept = "general"
        if categories:
            # Try to find a good concept category
            for cat in categories:
                cat_lower = cat.lower()
                # Skip meta-categories
                if any(skip in cat_lower for skip in ["stub", "article", "page", "cs1", "short description"]):
                    continue
                concept = cat
                break
        
        return {
            "title": page.get("title", title),
            "content": page.get("extract", ""),
            "url": page.get("fullurl", ""),
            "page_id": page_id,
            "categories": categories,
            "concept": concept,
        }
    except Exception as e:
        print(f"Error downloading '{title}': {e}")
        return None


def chunk_document_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
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
    fetch_categories: bool = True,
) -> Dict[str, Any]:
    """
    Create benchmark dataset from Wikipedia articles.
    
    Args:
        size: Dataset size ("small", "medium", "large")
        output_dir: Output directory
        model_name: Sentence-transformers model name
        chunk_size: Chunk size in words
        chunk_overlap: Overlap between chunks
        fetch_categories: Whether to fetch Wikipedia categories as concepts
        
    Returns:
        Dataset statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get topics for size
    topics = WIKIPEDIA_TOPICS.get(size, WIKIPEDIA_TOPICS["small"])
    
    print(f"Creating {size} benchmark with {len(topics)} Wikipedia articles...")
    if fetch_categories:
        print("Fetching Wikipedia categories for concept detection...")
    
    # Download articles
    print("\n1. Downloading Wikipedia articles...")
    articles = []
    for topic in tqdm(topics, desc="Downloading"):
        article = download_wikipedia_article(topic, fetch_categories=fetch_categories)
        if article and article["content"]:
            articles.append(article)
    
    print(f"Downloaded {len(articles)} articles")
    
    # Chunk articles
    print("\n2. Chunking articles...")
    chunks = []
    chunk_id = 0
    
    for article in tqdm(articles, desc="Chunking"):
        article_chunks = chunk_document_text(article["content"], chunk_size, chunk_overlap)
        
        for i, chunk_text in enumerate(article_chunks):
            # Skip very short chunks
            if len(chunk_text.split()) < 20:
                continue
            
            chunk_data = {
                "chunk_id": f"chunk_{chunk_id:06d}",
                "article_title": article["title"],
                "article_url": article["url"],
                "article_page_id": article["page_id"],
                "chunk_index": i,
                "text": chunk_text,
                "text_hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                "is_adversarial": False,  # Will be set during hub planting
                "modality": "text",  # All Wikipedia data is text
            }
            
            # Add concept from Wikipedia categories
            if "concept" in article:
                chunk_data["concept"] = article["concept"]
            if "categories" in article:
                chunk_data["categories"] = article["categories"][:5]  # Top 5 categories
            
            chunks.append(chunk_data)
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
    
    # Save query texts for lexical/hybrid search
    # Extract sample queries from chunks (first 100-200 words of each chunk)
    query_texts = []
    for chunk in chunks:
        words = chunk["text"].split()
        # Take first 50-100 words as a query-like text
        query_length = min(100, len(words))
        if query_length > 20:  # Only if chunk is long enough
            query_text = " ".join(words[:query_length])
            query_texts.append(query_text)
    
    # Also add some article titles as queries
    article_titles = list(set(chunk["article_title"] for chunk in chunks))
    query_texts.extend(article_titles[:50])  # Add up to 50 article titles
    
    # Save query texts
    with open(output_dir / "query_texts.json", "w") as f:
        json.dump(query_texts, f, indent=2)
    print(f"Saved query texts: {output_dir / 'query_texts.json'} ({len(query_texts)} queries)")
    
    # Collect concept statistics
    concept_counts = {}
    for chunk in chunks:
        concept = chunk.get("concept", "unknown")
        concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
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
        "has_concepts": True,
        "num_concepts": len(concept_counts),
        "concept_counts": concept_counts,
        "modality": "text",
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset info: {output_dir / 'dataset_info.json'}")
    
    print(f"\nBenchmark dataset created successfully!")
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
    parser.add_argument(
        "--fetch-categories",
        action="store_true",
        default=True,
        help="Fetch Wikipedia categories as concepts (default: True)",
    )
    parser.add_argument(
        "--no-categories",
        action="store_true",
        help="Disable fetching Wikipedia categories",
    )
    
    args = parser.parse_args()
    
    # Handle the no-categories flag
    fetch_categories = args.fetch_categories and not args.no_categories
    
    create_benchmark_dataset(
        size=args.size,
        output_dir=Path(args.output),
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fetch_categories=fetch_categories,
    )


if __name__ == "__main__":
    main()

