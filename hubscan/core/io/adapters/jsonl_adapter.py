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

"""JSONL adapter for loading vector database exports.

This adapter loads embeddings and metadata from JSONL (JSON Lines) files,
which is a common export format for vector databases.

Expected JSONL format (one JSON object per line):
    {"id": "doc_0", "embedding": [0.1, 0.2, ...], "text": "...", "metadata": {...}}
    {"id": "doc_1", "embedding": [0.3, 0.4, ...], "text": "...", "metadata": {...}}
    ...

The embedding field is required. All other fields are optional and will be
included in the metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from ..vector_index import VectorIndex
from ....utils.logging import get_logger
from ....utils.metrics import normalize_vectors

logger = get_logger()


class JSONLExportLoader:
    """Loader for vector database exports in JSONL format.
    
    This class loads embeddings and metadata from JSONL files.
    It does not provide search functionality - for that, the
    embeddings should be used to build a FAISS index.
    
    Example usage:
        loader = JSONLExportLoader()
        embeddings, metadata = loader.load("export.jsonl")
        
        # Build a searchable index
        from hubscan.core.io import build_faiss_index
        from hubscan.core.io.adapters import FAISSIndex
        
        faiss_index = build_faiss_index(embeddings, "flat", "cosine")
        index = FAISSIndex(faiss_index)
    """
    
    DEFAULT_EMBEDDING_FIELD = "embedding"
    DEFAULT_VECTOR_FIELD = "vector"  # Alternative name
    
    def __init__(
        self,
        embedding_field: str = "embedding",
        normalize: bool = True,
    ):
        """Initialize the JSONL loader.
        
        Args:
            embedding_field: Name of the field containing the embedding vector.
                            Defaults to "embedding". Also checks "vector" as fallback.
            normalize: Whether to normalize embeddings to unit length (for cosine metric).
        """
        self.embedding_field = embedding_field
        self.normalize = normalize
    
    def load(
        self,
        path: str,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from a JSONL file.
        
        Args:
            path: Path to the JSONL file.
            limit: Optional maximum number of records to load.
            
        Returns:
            Tuple of (embeddings, metadata) where:
            - embeddings: Array of shape (N, D) containing all vectors
            - metadata: List of dicts containing all other fields
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        
        embeddings = []
        metadata_list = []
        
        logger.info(f"Loading JSONL export from {path}")
        
        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if limit is not None and line_num > limit:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                
                # Extract embedding
                embedding = None
                for field_name in [self.embedding_field, self.DEFAULT_EMBEDDING_FIELD, self.DEFAULT_VECTOR_FIELD]:
                    if field_name in record:
                        embedding = record[field_name]
                        break
                
                if embedding is None:
                    raise ValueError(
                        f"No embedding found on line {line_num}. "
                        f"Expected field '{self.embedding_field}', 'embedding', or 'vector'"
                    )
                
                embeddings.append(embedding)
                
                # Extract metadata (everything except embedding)
                metadata_record = {
                    k: v for k, v in record.items()
                    if k not in [self.embedding_field, self.DEFAULT_EMBEDDING_FIELD, self.DEFAULT_VECTOR_FIELD]
                }
                metadata_list.append(metadata_record)
        
        if len(embeddings) == 0:
            raise ValueError(f"No records found in {path}")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            embeddings_array = normalize_vectors(embeddings_array)
        
        logger.info(f"Loaded {len(embeddings_array)} embeddings of dimension {embeddings_array.shape[1]}")
        
        return embeddings_array, metadata_list
    
    def load_batched(
        self,
        path: str,
        batch_size: int = 10000,
    ):
        """Generator that yields batches of embeddings and metadata.
        
        Useful for very large files that don't fit in memory.
        
        Args:
            path: Path to the JSONL file.
            batch_size: Number of records per batch.
            
        Yields:
            Tuple of (embeddings_batch, metadata_batch) for each batch.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        
        embeddings = []
        metadata_list = []
        
        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                
                # Extract embedding
                embedding = None
                for field_name in [self.embedding_field, self.DEFAULT_EMBEDDING_FIELD, self.DEFAULT_VECTOR_FIELD]:
                    if field_name in record:
                        embedding = record[field_name]
                        break
                
                if embedding is None:
                    continue
                
                embeddings.append(embedding)
                
                # Extract metadata
                metadata_record = {
                    k: v for k, v in record.items()
                    if k not in [self.embedding_field, self.DEFAULT_EMBEDDING_FIELD, self.DEFAULT_VECTOR_FIELD]
                }
                metadata_list.append(metadata_record)
                
                # Yield batch
                if len(embeddings) >= batch_size:
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    if self.normalize:
                        embeddings_array = normalize_vectors(embeddings_array)
                    yield embeddings_array, metadata_list
                    embeddings = []
                    metadata_list = []
        
        # Yield remaining
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            if self.normalize:
                embeddings_array = normalize_vectors(embeddings_array)
            yield embeddings_array, metadata_list


def load_jsonl_export(
    path: str,
    embedding_field: str = "embedding",
    normalize: bool = True,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Convenience function to load a JSONL export file.
    
    Args:
        path: Path to the JSONL file.
        embedding_field: Name of the field containing embeddings.
        normalize: Whether to normalize embeddings.
        limit: Maximum number of records to load.
        
    Returns:
        Tuple of (embeddings, metadata).
    """
    loader = JSONLExportLoader(
        embedding_field=embedding_field,
        normalize=normalize,
    )
    return loader.load(path, limit=limit)

