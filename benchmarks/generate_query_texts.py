#!/usr/bin/env python3
"""Generate query_texts.json from existing metadata.json"""

import json
from pathlib import Path
import sys

def generate_query_texts(dataset_dir: Path):
    """Generate query_texts.json from metadata.json"""
    dataset_dir = Path(dataset_dir)
    metadata_file = dataset_dir / 'metadata.json'
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return False
    
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file) as f:
        chunks = json.load(f)
    
    # Extract query texts from chunks
    query_texts = []
    for chunk in chunks:
        words = chunk['text'].split()
        query_length = min(100, len(words))
        if query_length > 20:
            query_text = ' '.join(words[:query_length])
            query_texts.append(query_text)
    
    # Add article titles
    article_titles = list(set(chunk['article_title'] for chunk in chunks))
    query_texts.extend(article_titles[:50])
    
    # Save query texts
    query_file = dataset_dir / 'query_texts.json'
    with open(query_file, 'w') as f:
        json.dump(query_texts, f, indent=2)
    
    print(f"Generated query_texts.json with {len(query_texts)} queries")
    print(f"  Saved to: {query_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        dataset_dir = Path("data/wikipedia_small")
    
    generate_query_texts(dataset_dir)

