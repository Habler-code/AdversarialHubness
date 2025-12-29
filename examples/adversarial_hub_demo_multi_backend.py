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
Multi-backend demonstration: Test HubScan with all supported vector databases.

This demo extends the basic adversarial hub demo to test all supported backends:
- FAISS (default, always available)
- Pinecone (requires API key and index)
- Qdrant (requires running instance)
- Weaviate (requires running instance)

Usage:
    python adversarial_hub_demo_multi_backend.py --backend faiss
    python adversarial_hub_demo_multi_backend.py --backend all
    python adversarial_hub_demo_multi_backend.py --backend faiss,pinecone
"""

import numpy as np
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import tempfile
import os

# Import the demo utilities
from adversarial_hub_demo import (
    generate_documents,
    split_documents_for_rag,
    create_embeddings,
    plant_adversarial_hub,
)


def create_faiss_backend(embeddings: np.ndarray, output_dir: Path) -> Dict[str, Any]:
    """Create FAISS backend configuration."""
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    return {
        "mode": "embeddings_only",
        "embeddings_path": str(embeddings_path),
        "backend_name": "FAISS",
    }


def create_pinecone_backend(
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create Pinecone backend configuration."""
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        print("  [SKIP] Pinecone client not installed. Install with: pip install pinecone-client")
        return None
    
    if not api_key:
        print("  [SKIP] Pinecone API key not provided. Set PINECONE_API_KEY environment variable.")
        return None
    
    if not index_name:
        print("  [SKIP] Pinecone index name not provided. Set PINECONE_INDEX_NAME environment variable.")
        return None
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        try:
            index_info = pc.describe_index(index_name)
            print(f"  [INFO] Using existing Pinecone index: {index_name}")
        except Exception:
            print(f"  [SKIP] Pinecone index '{index_name}' not found. Create it first.")
            return None
        
        # Upload embeddings to Pinecone
        print(f"  [INFO] Uploading {len(embeddings)} vectors to Pinecone...")
        index = pc.Index(index_name)
        
        # Prepare vectors for upload (Pinecone requires IDs and metadata)
        vectors_to_upload = []
        for i, embedding in enumerate(embeddings):
            vector_id = str(i)  # Use chunk index as ID
            vectors_to_upload.append({
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {
                    "chunk_id": metadata["chunk_id"][i],
                    "doc_id": metadata["doc_id"][i],
                }
            })
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upload), batch_size):
            batch = vectors_to_upload[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"  [INFO] Uploaded {len(embeddings)} vectors to Pinecone")
        
        return {
            "mode": "pinecone",
            "pinecone_index_name": index_name,
            "pinecone_api_key": api_key,
            "dimension": embeddings.shape[1],
            "backend_name": "Pinecone",
        }
    except Exception as e:
        print(f"  [ERROR] Failed to setup Pinecone: {e}")
        return None


def create_qdrant_backend(
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    collection_name: str = "hubscan_demo",
    url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create Qdrant backend configuration."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("  [SKIP] Qdrant client not installed. Install with: pip install qdrant-client")
        return None
    
    try:
        # Connect to Qdrant
        if api_key:
            client = QdrantClient(url=url, api_key=api_key)
        else:
            client = QdrantClient(url=url)
        
        # Check connection
        try:
            collections = client.get_collections()
            print(f"  [INFO] Connected to Qdrant at {url}")
        except Exception as e:
            print(f"  [SKIP] Cannot connect to Qdrant at {url}: {e}")
            return None
        
        # Create collection if it doesn't exist
        try:
            collection_info = client.get_collection(collection_name)
            print(f"  [INFO] Using existing Qdrant collection: {collection_name}")
            # Clear existing points
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embeddings.shape[1],
                    distance=Distance.COSINE
                )
            )
        except Exception:
            # Collection doesn't exist, create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embeddings.shape[1],
                    distance=Distance.COSINE
                )
            )
            print(f"  [INFO] Created Qdrant collection: {collection_name}")
        
        # Upload embeddings
        print(f"  [INFO] Uploading {len(embeddings)} vectors to Qdrant...")
        points = []
        for i, embedding in enumerate(embeddings):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": metadata["chunk_id"][i],
                        "doc_id": metadata["doc_id"][i],
                    }
                )
            )
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
        
        print(f"  [INFO] Uploaded {len(embeddings)} vectors to Qdrant")
        
        return {
            "mode": "qdrant",
            "qdrant_collection_name": collection_name,
            "qdrant_url": url,
            "qdrant_api_key": api_key,
            "backend_name": "Qdrant",
        }
    except Exception as e:
        print(f"  [ERROR] Failed to setup Qdrant: {e}")
        return None


def create_weaviate_backend(
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    class_name: str = "HubScanDemo",
    url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create Weaviate backend configuration."""
    try:
        import weaviate
    except ImportError:
        print("  [SKIP] Weaviate client not installed. Install with: pip install weaviate-client")
        return None
    
    try:
        # Connect to Weaviate
        if api_key:
            auth_config = weaviate.AuthApiKey(api_key=api_key)
            client = weaviate.Client(url=url, auth_client_secret=auth_config)
        else:
            client = weaviate.Client(url=url)
        
        # Check connection
        try:
            schema = client.schema.get()
            print(f"  [INFO] Connected to Weaviate at {url}")
        except Exception as e:
            print(f"  [SKIP] Cannot connect to Weaviate at {url}: {e}")
            return None
        
        # Create class if it doesn't exist
        try:
            existing_schema = client.schema.get(class_name)
            print(f"  [INFO] Using existing Weaviate class: {class_name}")
            # Delete existing class
            client.schema.delete_class(class_name)
        except Exception:
            pass
        
        # Create class schema
        class_schema = {
            "class": class_name,
            "vectorizer": "none",  # We'll provide vectors
            "properties": [
                {"name": "chunk_id", "dataType": ["string"]},
                {"name": "doc_id", "dataType": ["string"]},
            ]
        }
        client.schema.create_class(class_schema)
        print(f"  [INFO] Created Weaviate class: {class_name}")
        
        # Upload embeddings
        print(f"  [INFO] Uploading {len(embeddings)} vectors to Weaviate...")
        with client.batch as batch:
            batch.batch_size = 100
            for i, embedding in enumerate(embeddings):
                batch.add_data_object(
                    data_object={
                        "chunk_id": metadata["chunk_id"][i],
                        "doc_id": metadata["doc_id"][i],
                    },
                    class_name=class_name,
                    vector=embedding.tolist()
                )
        
        print(f"  [INFO] Uploaded {len(embeddings)} vectors to Weaviate")
        
        return {
            "mode": "weaviate",
            "weaviate_class_name": class_name,
            "weaviate_url": url,
            "weaviate_api_key": api_key,
            "backend_name": "Weaviate",
        }
    except Exception as e:
        print(f"  [ERROR] Failed to setup Weaviate: {e}")
        return None


def run_backend_test(
    backend_config: Dict[str, Any],
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    hub_info: Dict[str, Any],
    output_dir: Path,
    k: int = 10,
    num_queries: int = 300,
) -> Dict[str, Any]:
    """Run HubScan test on a specific backend."""
    backend_name = backend_config["backend_name"]
    print(f"\n{'=' * 80}")
    print(f"Testing {backend_name} Backend")
    print(f"{'=' * 80}")
    
    # Add parent directory to path
    demo_path = Path(__file__).parent.parent
    if str(demo_path) not in sys.path:
        sys.path.insert(0, str(demo_path))
    
    from hubscan.config import Config
    from hubscan.core.scanner import Scanner
    from hubscan.sdk import get_suspicious_documents, explain_document, Verdict
    
    # Create config
    config = Config()
    config.input.mode = backend_config["mode"]
    
    # Set backend-specific config
    for key, value in backend_config.items():
        if key != "mode" and key != "backend_name":
            setattr(config.input, key, value)
    
    # Set metadata path
    metadata_path = output_dir / f"metadata_{backend_name.lower()}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    config.input.metadata_path = str(metadata_path)
    
    # Set scan parameters
    config.scan.k = k
    config.scan.num_queries = num_queries
    config.scan.query_sampling = "mixed"
    config.scan.batch_size = 64
    
    # Set thresholds
    config.thresholds.hub_z = 2.0
    config.thresholds.percentile = 0.10
    
    # Set output
    config.output.out_dir = str(output_dir / f"reports_{backend_name.lower()}")
    
    try:
        # Run scan
        print(f"\n[INFO] Running HubScan scan on {backend_name}...")
        scanner = Scanner(config)
        scanner.load_data()
        results = scanner.scan()
        
        print(f"[SUCCESS] Scan completed in {results['runtime']:.2f} seconds")
        
        # Check results
        high_risk = get_suspicious_documents(results, verdict=Verdict.HIGH, top_k=10)
        all_suspicious = get_suspicious_documents(results, verdict=None, top_k=20)
        
        # Find adversarial hub
        found_adversarial = False
        adversarial_chunk_idx = None
        
        for doc in all_suspicious:
            chunk_idx = doc["doc_index"]
            if chunk_idx == hub_info["chunk_index"]:
                found_adversarial = True
                adversarial_chunk_idx = chunk_idx
                break
        
        return {
            "backend": backend_name,
            "success": True,
            "runtime": results["runtime"],
            "high_risk_count": len(high_risk),
            "suspicious_count": len(all_suspicious),
            "found_adversarial": found_adversarial,
            "adversarial_chunk_idx": adversarial_chunk_idx,
            "planted_chunk_idx": hub_info["chunk_index"],
            "results": results,
        }
    except Exception as e:
        print(f"[ERROR] Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "backend": backend_name,
            "success": False,
            "error": str(e),
        }


def run_multi_backend_demo(
    backends: List[str] = ["faiss"],
    num_docs: int = 50,
    k: int = 10,
    num_queries: int = 300,
):
    """Run demo across multiple backends."""
    print("=" * 80)
    print("HubScan Multi-Backend Adversarial Hub Detection Demo")
    print("=" * 80)
    
    # Step 1-4: Generate data (same as basic demo)
    print("\n[Step 1-4] Generating documents, chunks, embeddings, and planting adversarial hub...")
    documents = generate_documents(num_docs=num_docs)
    for i, doc in enumerate(documents):
        doc["doc_index"] = i
    
    chunks = split_documents_for_rag(documents, chunk_size=150)
    embeddings = create_embeddings(chunks, embedding_dim=128)
    
    hub_chunk_idx = len(chunks) // 2
    chunks, embeddings, hub_info = plant_adversarial_hub(
        chunks, embeddings, hub_chunk_idx,
        num_query_clusters=50,
        strength=0.80
    )
    
    print(f"Generated {len(chunks)} chunks with adversarial hub at index {hub_info['chunk_index']}")
    
    # Prepare metadata
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
    
    # Create output directory
    output_dir = Path("examples/demo_data_multi_backend")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each backend
    results = []
    
    if "faiss" in backends or "all" in backends:
        backend_config = create_faiss_backend(embeddings, output_dir)
        if backend_config:
            result = run_backend_test(backend_config, embeddings, metadata, hub_info, output_dir, k, num_queries)
            results.append(result)
    
    if "pinecone" in backends or "all" in backends:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        backend_config = create_pinecone_backend(embeddings, metadata, output_dir, index_name, api_key)
        if backend_config:
            result = run_backend_test(backend_config, embeddings, metadata, hub_info, output_dir, k, num_queries)
            results.append(result)
    
    if "qdrant" in backends or "all" in backends:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name = os.getenv("QDRANT_COLLECTION", "hubscan_demo")
        backend_config = create_qdrant_backend(embeddings, metadata, output_dir, collection_name, url, api_key)
        if backend_config:
            result = run_backend_test(backend_config, embeddings, metadata, hub_info, output_dir, k, num_queries)
            results.append(result)
    
    if "weaviate" in backends or "all" in backends:
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        api_key = os.getenv("WEAVIATE_API_KEY")
        class_name = os.getenv("WEAVIATE_CLASS", "HubScanDemo")
        backend_config = create_weaviate_backend(embeddings, metadata, output_dir, class_name, url, api_key)
        if backend_config:
            result = run_backend_test(backend_config, embeddings, metadata, hub_info, output_dir, k, num_queries)
            results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\nBackends tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful Backends:")
        for r in successful:
            print(f"  - {r['backend']}:")
            print(f"    Runtime: {r['runtime']:.2f}s")
            print(f"    High-risk chunks: {r['high_risk_count']}")
            print(f"    Adversarial hub found: {'YES' if r['found_adversarial'] else 'NO'}")
            if r['found_adversarial']:
                print(f"    Detected at chunk index: {r['adversarial_chunk_idx']}")
    
    if failed:
        print("\nFailed Backends:")
        for r in failed:
            print(f"  - {r['backend']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nReports saved to: {output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HubScan with multiple vector database backends")
    parser.add_argument(
        "--backend",
        type=str,
        default="faiss",
        help="Backend to test: faiss, pinecone, qdrant, weaviate, all, or comma-separated list"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=50,
        help="Number of documents to generate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=300,
        help="Number of queries to sample"
    )
    
    args = parser.parse_args()
    
    # Parse backend list
    if args.backend == "all":
        backends = ["faiss", "pinecone", "qdrant", "weaviate"]
    else:
        backends = [b.strip() for b in args.backend.split(",")]
    
    run_multi_backend_demo(
        backends=backends,
        num_docs=args.num_docs,
        k=args.k,
        num_queries=args.num_queries,
    )

