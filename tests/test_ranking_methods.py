#!/usr/bin/env python3
"""Test script to verify all ranking methods work with the benchmark."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from hubscan import Config, Scanner
from hubscan.core.io.adapters.faiss_adapter import FAISSIndex
import numpy as np
import faiss
import json

def test_ranking_methods():
    """Test that all ranking methods can be used in the benchmark."""
    
    print("Testing ranking methods with benchmark setup...\n")
    
    # Create a small test dataset
    print("1. Creating test dataset...")
    num_docs = 50
    dim = 32
    
    embeddings = np.random.randn(num_docs, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create document texts for lexical search
    doc_texts = [f"Document {i} with some text content about topic {i % 10}" for i in range(num_docs)]
    
    # Create query texts
    query_texts = [
        "topic 0",
        "topic 1", 
        "document",
        "text content",
        "some topic"
    ] * 10  # 50 queries
    
    print(f"   Created {num_docs} documents, {len(query_texts)} queries")
    
    # Check if rank-bm25 is available
    try:
        import rank_bm25
        has_bm25 = True
    except ImportError:
        has_bm25 = False
        print("Warning: rank-bm25 not installed. Skipping lexical/hybrid tests.")
        print("  Install with: pip install rank-bm25\n")
    
    # Test each ranking method
    methods_to_test = ["vector", "reranked"]
    if has_bm25:
        methods_to_test.extend(["hybrid", "lexical"])
    
    for method in methods_to_test:
        print(f"\n2. Testing {method} ranking method...")
        
        try:
            # Create config
            config = Config()
            config.input.mode = "embeddings_only"
            config.scan.k = 5
            config.scan.num_queries = 20
            config.scan.ranking.method = method
            
            if method in ["lexical", "hybrid"]:
                if not doc_texts:
                    print(f"   Warning: Skipping {method} - requires document texts")
                    continue
                config.scan.query_texts_path = "/tmp/test_queries.json"
                # Save query texts temporarily
                with open("/tmp/test_queries.json", "w") as f:
                    json.dump(query_texts, f)
            
            # Create FAISS index
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(embeddings)
            
            # Wrap in adapter with doc texts for lexical/hybrid
            if method in ["lexical", "hybrid"]:
                index = FAISSIndex(faiss_index, document_texts=doc_texts)
            else:
                index = FAISSIndex(faiss_index)
            
            # Create scanner
            scanner = Scanner(config)
            scanner.index = index
            scanner.doc_embeddings = embeddings
            
            if method in ["lexical", "hybrid"]:
                scanner.query_texts = query_texts
            
            # Test that search works
            if method == "vector":
                queries = np.random.randn(5, dim).astype(np.float32)
                queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
                distances, indices = index.search(queries, k=5)
                print(f"   Vector search works: {indices.shape}")
                
            elif method == "hybrid":
                queries = np.random.randn(5, dim).astype(np.float32)
                queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
                distances, indices, metadata = index.search_hybrid(
                    query_vectors=queries,
                    query_texts=query_texts[:5],
                    k=5,
                    alpha=0.5
                )
                print(f"   Hybrid search works: {indices.shape}, method={metadata.get('ranking_method')}")
                
            elif method == "lexical":
                distances, indices, metadata = index.search_lexical(
                    query_texts=query_texts[:5],
                    k=5
                )
                print(f"   Lexical search works: {indices.shape}, method={metadata.get('ranking_method')}")
                
            elif method == "reranked":
                queries = np.random.randn(5, dim).astype(np.float32)
                queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
                distances, indices, metadata = index.search_reranked(
                    query_vectors=queries,
                    k=5,
                    rerank_top_n=20
                )
                print(f"   Reranked search works: {indices.shape}, method={metadata.get('ranking_method')}")
            
            print(f"   {method} ranking method is working correctly")
            
        except Exception as e:
            print(f"   Error with {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nRanking methods test completed!")

if __name__ == "__main__":
    test_ranking_methods()

