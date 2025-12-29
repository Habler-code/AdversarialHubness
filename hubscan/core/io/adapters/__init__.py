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

"""Vector database adapters for HubScan."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import InputConfig
    from ..vector_index import VectorIndex

from .faiss_adapter import FAISSIndex

__all__ = ["FAISSIndex", "create_index"]


def create_index(config: "InputConfig") -> "VectorIndex":
    """
    Create a VectorIndex instance based on configuration.
    
    Args:
        config: Input configuration specifying backend and parameters
        
    Returns:
        VectorIndex instance for the specified backend
        
    Raises:
        ValueError: If backend mode is not supported or required parameters are missing
        ImportError: If required backend client library is not installed
    """
    mode = config.mode
    
    if mode == "embeddings_only" or mode == "faiss_index":
        # FAISS adapter will be created when index is built/loaded
        raise ValueError("FAISS index should be created via build_faiss_index() or load_faiss_index()")
    
    elif mode == "pinecone":
        try:
            from .pinecone_adapter import PineconeIndex
        except ImportError:
            raise ImportError(
                "Pinecone adapter requires 'pinecone-client' package. "
                "Install with: pip install pinecone-client"
            )
        
        if not config.pinecone_index_name:
            raise ValueError("pinecone_index_name is required for Pinecone mode")
        if not config.pinecone_api_key:
            raise ValueError("pinecone_api_key is required for Pinecone mode")
        
        return PineconeIndex(
            index_name=config.pinecone_index_name,
            api_key=config.pinecone_api_key,
            dimension=config.dimension or 0,  # Will be inferred if 0
            environment=getattr(config, "pinecone_environment", None),
        )
    
    elif mode == "qdrant":
        try:
            from .qdrant_adapter import QdrantIndex
        except ImportError:
            raise ImportError(
                "Qdrant adapter requires 'qdrant-client' package. "
                "Install with: pip install qdrant-client"
            )
        
        if not config.qdrant_collection_name:
            raise ValueError("qdrant_collection_name is required for Qdrant mode")
        
        return QdrantIndex(
            collection_name=config.qdrant_collection_name,
            url=getattr(config, "qdrant_url", "http://localhost:6333"),
            api_key=getattr(config, "qdrant_api_key", None),
        )
    
    elif mode == "weaviate":
        try:
            from .weaviate_adapter import WeaviateIndex
        except ImportError:
            raise ImportError(
                "Weaviate adapter requires 'weaviate-client' package. "
                "Install with: pip install weaviate-client"
            )
        
        if not config.weaviate_class_name:
            raise ValueError("weaviate_class_name is required for Weaviate mode")
        
        return WeaviateIndex(
            class_name=config.weaviate_class_name,
            url=getattr(config, "weaviate_url", "http://localhost:8080"),
            api_key=getattr(config, "weaviate_api_key", None),
        )
    
    else:
        raise ValueError(f"Unsupported backend mode: {mode}")

