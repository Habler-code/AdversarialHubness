#!/usr/bin/env python3
"""
Test script for embedding extraction from vector databases.

Creates temporary databases, inserts test embeddings, extracts them,
then cleans up.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hubscan.core.io.adapters.faiss_adapter import FAISSIndex

# Try to import other adapters
PineconeIndex = None
QdrantIndex = None
WeaviateIndex = None

try:
    from hubscan.core.io.adapters.pinecone_adapter import PineconeIndex
except (ImportError, Exception) as e:
    pass

try:
    from hubscan.core.io.adapters.qdrant_adapter import QdrantIndex
except (ImportError, Exception) as e:
    pass

try:
    from hubscan.core.io.adapters.weaviate_adapter import WeaviateIndex
except (ImportError, Exception) as e:
    pass
from hubscan.core.io.embeddings import save_embeddings, load_embeddings
from hubscan.utils.metrics import normalize_vectors


def create_test_embeddings(n: int = 100, dim: int = 128) -> np.ndarray:
    """Create test embeddings."""
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = normalize_vectors(embeddings)  # Normalize for cosine similarity
    return embeddings


def test_faiss_extraction():
    """Test FAISS embedding extraction."""
    print("\n=== Testing FAISS Extraction ===")
    
    # Create test embeddings
    n, dim = 50, 128
    test_embeddings = create_test_embeddings(n, dim)
    
    # Create FAISS index
    import faiss
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(test_embeddings)
    
    # Wrap in adapter
    faiss_index = FAISSIndex(index)
    
    # Extract embeddings
    try:
        extracted, ids = faiss_index.extract_embeddings()
        assert len(extracted) == n, f"Expected {n} embeddings, got {len(extracted)}"
        assert extracted.shape[1] == dim, f"Expected dim {dim}, got {extracted.shape[1]}"
        print(f"PASS: FAISS extracted {len(extracted)} embeddings successfully")
        return True
    except NotImplementedError:
        print("FAIL: FAISS extraction not implemented")
        return False
    except Exception as e:
        print(f"FAIL: FAISS error - {e}")
        return False


def test_pinecone_extraction():
    """Test Pinecone embedding extraction."""
    print("\n=== Testing Pinecone Extraction ===")
    
    if PineconeIndex is None:
        print("SKIP: PineconeIndex not available")
        return None
    
    try:
        from pinecone import Pinecone
    except ImportError:
        print("SKIP: pinecone-client not installed")
        return None
    
    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("SKIP: PINECONE_API_KEY not set")
        return None
    
    try:
        # Create test embeddings
        n, dim = 20, 128  # Small number for testing
        test_embeddings = create_test_embeddings(n, dim)
        
        # Create temporary index name
        import uuid
        index_name = f"hubscan-test-{uuid.uuid4().hex[:8]}"
        
        pc = Pinecone(api_key=api_key)
        
        # Create index
        print(f"Creating temporary Pinecone index: {index_name}")
        try:
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Index {index_name} already exists, using it")
            else:
                raise
        
        # Wait for index to be ready
        import time
        time.sleep(5)
        
        # Get index
        index = pc.Index(index_name)
        
        # Insert embeddings
        vectors_to_upsert = [
            (str(i), test_embeddings[i].tolist())
            for i in range(n)
        ]
        index.upsert(vectors=vectors_to_upsert)
        
        # Wait for indexing
        time.sleep(2)
        
        # Create adapter
        pinecone_index = PineconeIndex(index_name=index_name, api_key=api_key, dimension=dim)
        
        # Extract embeddings
        extracted, ids = pinecone_index.extract_embeddings(limit=n)
        
        assert len(extracted) == n, f"Expected {n} embeddings, got {len(extracted)}"
        assert extracted.shape[1] == dim, f"Expected dim {dim}, got {extracted.shape[1]}"
        
        print(f"PASS: Pinecone extracted {len(extracted)} embeddings successfully")
        
        # Cleanup: Delete index
        print(f"Deleting temporary index: {index_name}")
        pc.delete_index(index_name)
        
        return True
        
    except Exception as e:
        print(f"FAIL: Pinecone error - {e}")
        # Try to cleanup on error
        try:
            pc.delete_index(index_name)
        except:
            pass
        return False


def test_qdrant_extraction():
    """Test Qdrant embedding extraction."""
    print("\n=== Testing Qdrant Extraction ===")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("SKIP: qdrant-client not installed")
        return None
    
    if QdrantIndex is None:
        print("SKIP: QdrantIndex not available")
        return None
    
    # Check if Qdrant is running
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    try:
        client = QdrantClient(url=url)
        # Test connection
        client.get_collections()
    except Exception as e:
        print(f"SKIP: Cannot connect to Qdrant at {url} - {e}")
        print("  Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return None
    
    try:
        # Create test embeddings
        n, dim = 50, 128
        test_embeddings = create_test_embeddings(n, dim)
        
        # Create temporary collection name
        import uuid
        collection_name = f"hubscan_test_{uuid.uuid4().hex[:8]}"
        
        # Create collection
        print(f"Creating temporary Qdrant collection: {collection_name}")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Collection {collection_name} already exists, deleting it first")
                try:
                    client.delete_collection(collection_name=collection_name)
                except:
                    pass
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                raise
        
        # Insert embeddings
        points = [
            PointStruct(
                id=i,
                vector=test_embeddings[i].tolist(),
            )
            for i in range(n)
        ]
        client.upsert(collection_name=collection_name, points=points)
        
        # Create adapter
        qdrant_index = QdrantIndex(collection_name=collection_name, url=url)
        
        # Extract embeddings
        extracted, ids = qdrant_index.extract_embeddings(limit=n)
        
        assert len(extracted) == n, f"Expected {n} embeddings, got {len(extracted)}"
        assert extracted.shape[1] == dim, f"Expected dim {dim}, got {extracted.shape[1]}"
        
        print(f"PASS: Qdrant extracted {len(extracted)} embeddings successfully")
        
        # Cleanup: Delete collection
        print(f"Deleting temporary collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)
        
        return True
        
    except Exception as e:
        print(f"FAIL: Qdrant error - {e}")
        # Try to cleanup on error
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass
        return False


def test_weaviate_extraction():
    """Test Weaviate embedding extraction."""
    print("\n=== Testing Weaviate Extraction ===")
    
    if WeaviateIndex is None:
        print("SKIP: WeaviateIndex not available")
        return None
    
    try:
        import weaviate
    except ImportError:
        print("SKIP: weaviate-client not installed")
        return None
    
    # Check if Weaviate is running - try v4 API first
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    try:
        # Try v4 API
        try:
            if 'localhost' in url or '127.0.0.1' in url:
                port = int(url.split(':')[-1]) if ':' in url else 8080
                client = weaviate.connect_to_local(port=port)
                client.close()
            else:
                # Custom URL - would need connect_to_custom
                print(f"SKIP: Custom Weaviate URL not supported in test")
                return None
        except:
            # Fallback to v3 (if available)
            try:
                client = weaviate.Client(url=url)
                client.schema.get()
            except:
                raise
    except Exception as e:
        print(f"SKIP: Cannot connect to Weaviate at {url} - {e}")
        print("  Start Weaviate with: docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:1.25.9")
        return None
    
    try:
        # Create test embeddings
        n, dim = 50, 128
        test_embeddings = create_test_embeddings(n, dim)
        
        # Create temporary class name
        import uuid
        class_name = f"HubScanTest{uuid.uuid4().hex[:8].capitalize()}"
        
        # Create schema
        print(f"Creating temporary Weaviate class: {class_name}")
        schema = {
            "class": class_name,
            "vectorizer": "none",  # We'll provide vectors manually
            "properties": [
                {
                    "name": "text",
                    "dataType": ["string"],
                }
            ],
        }
        client.schema.create_class(schema)
        
        # Insert embeddings
        with client.batch as batch:
            batch.batch_size = 10
            for i in range(n):
                batch.add_data_object(
                    data_object={"text": f"Document {i}"},
                    class_name=class_name,
                    vector=test_embeddings[i].tolist(),
                )
        
        # Wait for indexing
        import time
        time.sleep(2)
        
        # Create adapter
        weaviate_index = WeaviateIndex(class_name=class_name, url=url)
        weaviate_index._dimension = dim  # Set dimension
        
        # Extract embeddings
        extracted, ids = weaviate_index.extract_embeddings(limit=n)
        
        assert len(extracted) == n, f"Expected {n} embeddings, got {len(extracted)}"
        assert extracted.shape[1] == dim, f"Expected dim {dim}, got {extracted.shape[1]}"
        
        print(f"PASS: Weaviate extracted {len(extracted)} embeddings successfully")
        
        # Cleanup: Delete class
        print(f"Deleting temporary class: {class_name}")
        client.schema.delete_class(class_name=class_name)
        
        return True
        
    except Exception as e:
        print(f"FAIL: Weaviate error - {e}")
        import traceback
        traceback.print_exc()
        # Try to cleanup on error
        try:
            client.schema.delete_class(class_name=class_name)
        except:
            pass
        return False


def main():
    """Run all extraction tests."""
    print("=" * 60)
    print("Testing Embedding Extraction from Vector Databases")
    print("=" * 60)
    
    results = {}
    
    # Test FAISS (always available)
    results["FAISS"] = test_faiss_extraction()
    
    # Test Pinecone (if available)
    results["Pinecone"] = test_pinecone_extraction()
    
    # Test Qdrant (if available)
    results["Qdrant"] = test_qdrant_extraction()
    
    # Test Weaviate (if available)
    results["Weaviate"] = test_weaviate_extraction()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for db_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"{db_name:15s}: {status}")
    
    # Count successes
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total and total > 0:
        print("SUCCESS: All tests passed!")
        return 0
    else:
        print("WARNING: Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())

