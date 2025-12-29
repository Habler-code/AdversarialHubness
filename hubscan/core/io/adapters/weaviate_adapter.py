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

"""Weaviate adapter for VectorIndex interface."""

from typing import Tuple, Optional
import numpy as np

try:
    import weaviate
except ImportError:
    weaviate = None

from ..vector_index import VectorIndex
from ...utils.logging import get_logger

logger = get_logger()


class WeaviateIndex(VectorIndex):
    """
    Weaviate implementation of VectorIndex interface.
    
    This adapter connects to a Weaviate class and provides the VectorIndex
    interface for HubScan detection operations.
    """
    
    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Weaviate adapter.
        
        Args:
            class_name: Name of the Weaviate class (collection)
            url: Weaviate server URL
            api_key: Optional API key for Weaviate Cloud
        """
        if weaviate is None:
            raise ImportError(
                "Weaviate adapter requires 'weaviate-client' package. "
                "Install with: pip install weaviate-client"
            )
        
        self.class_name = class_name
        
        # Initialize Weaviate client
        if api_key:
            auth_config = weaviate.AuthApiKey(api_key=api_key)
            self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
        else:
            self.client = weaviate.Client(url=url)
        
        # Get schema info
        try:
            schema = self.client.schema.get(class_name)
            if not schema:
                raise ValueError(f"Weaviate class '{class_name}' not found")
            
            # Get vectorizer config to determine dimension
            vectorizer_config = schema.get("vectorizer", {})
            if "vectorizeClassName" in vectorizer_config:
                # Infer dimension from vectorizer (may need manual specification)
                # For now, we'll require dimension to be specified or infer from query
                self._dimension = 0  # Will be inferred from first query
            else:
                # Check vectorIndexConfig for dimension
                vector_index_config = schema.get("vectorIndexConfig", {})
                # Dimension is typically not in schema, will infer from queries
            
            # Get count
            result = (
                self.client.query
                .aggregate(class_name)
                .with_meta_count()
                .do()
            )
            
            if result and "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(class_name, [])
                if aggregate_data:
                    self._ntotal = aggregate_data[0].get("meta", {}).get("count", 0)
                else:
                    self._ntotal = 0
            else:
                self._ntotal = 0
            
            logger.info(
                f"Connected to Weaviate class '{class_name}' "
                f"with {self._ntotal} vectors"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to connect to Weaviate class '{class_name}': {e}"
            )
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using Weaviate.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        # Infer dimension from first query if not set
        if self._dimension == 0 and len(query_vectors) > 0:
            self._dimension = query_vectors.shape[1]
        
        M = len(query_vectors)
        distances = []
        indices = []
        
        for query in query_vectors:
            try:
                result = (
                    self.client.query
                    .get(self.class_name)
                    .with_near_vector({"vector": query.tolist()})
                    .with_limit(k)
                    .with_additional(["distance", "id"])
                    .do()
                )
                
                result_distances = []
                result_indices = []
                
                if result and "data" in result and "Get" in result["data"]:
                    get_data = result["data"]["Get"].get(self.class_name, [])
                    
                    for item in get_data:
                        additional = item.get("_additional", {})
                        distance = additional.get("distance", float('inf'))
                        point_id = additional.get("id", None)
                        
                        result_distances.append(float(distance))
                        
                        # Weaviate IDs are UUIDs, convert to int hash
                        if point_id:
                            idx = hash(str(point_id)) % (2**31)  # Keep within int32 range
                        else:
                            idx = -1
                        
                        result_indices.append(idx)
                
                # Pad to k if needed
                while len(result_distances) < k:
                    result_distances.append(float('inf'))
                    result_indices.append(-1)
                
                distances.append(result_distances[:k])
                indices.append(result_indices[:k])
            
            except Exception as e:
                logger.error(f"Weaviate search failed: {e}")
                raise
        
        return np.array(distances, dtype=np.float32), np.array(indices, dtype=np.int64)
    
    @property
    def ntotal(self) -> int:
        """Number of vectors in the Weaviate class."""
        # Refresh count
        try:
            result = (
                self.client.query
                .aggregate(self.class_name)
                .with_meta_count()
                .do()
            )
            
            if result and "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(self.class_name, [])
                if aggregate_data:
                    self._ntotal = aggregate_data[0].get("meta", {}).get("count", 0)
        except Exception:
            pass  # Keep cached value on error
        return self._ntotal
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors in the Weaviate class."""
        return self._dimension if self._dimension > 0 else 0

