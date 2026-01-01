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

"""Multi-index adapter for gold standard architecture."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..vector_index import VectorIndex
from ...fusion import parallel_retrieve, fuse_results, enforce_diversity
from ....config import Config
from ....utils.logging import get_logger

logger = get_logger()


class MultiIndexAdapter(VectorIndex):
    """
    Adapter that wraps multiple indexes and performs parallel retrieval + late fusion.
    
    Implements the gold standard architecture:
    - Parallel retrieval from text, image, and optional unified/cross-modal indexes
    - Late fusion (normalize + merge candidates)
    - Optional diversity enforcement
    """
    
    def __init__(
        self,
        text_index: Optional[VectorIndex],
        image_index: Optional[VectorIndex],
        unified_index: Optional[VectorIndex],
        config: Config,
        doc_embeddings: Optional[np.ndarray] = None,
    ):
        """
        Initialize multi-index adapter.
        
        Args:
            text_index: Optional text index
            image_index: Optional image index
            unified_index: Optional unified/cross-modal index (recall backstop)
            config: Configuration with late_fusion and diversity settings
            doc_embeddings: Combined document embeddings for diversity enforcement
        """
        self.text_index = text_index
        self.image_index = image_index
        self.unified_index = unified_index
        self.config = config
        self.doc_embeddings = doc_embeddings
        
        # Get fusion config
        self.late_fusion_config = config.input.late_fusion
        self.diversity_config = config.input.diversity
        
        # Determine total number of documents (sum of all indexes)
        self._ntotal = 0
        if text_index:
            self._ntotal += text_index.ntotal
        if image_index:
            self._ntotal += image_index.ntotal
        if unified_index:
            # Unified index may overlap, so we don't double-count
            # For now, use max dimension
            pass
        
        # Use text index dimension as primary
        if text_index:
            self._dimension = text_index.dimension
        elif image_index:
            self._dimension = image_index.dimension
        elif unified_index:
            self._dimension = unified_index.dimension
        else:
            raise ValueError("At least one index required")
    
    @property
    def ntotal(self) -> int:
        """Total number of documents across all indexes."""
        return self._ntotal
    
    @property
    def dimension(self) -> int:
        """Vector dimension (from primary index)."""
        return self._dimension
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parallel retrieval and late fusion.
        
        Args:
            query_vectors: Query embeddings (M, D)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if not self.late_fusion_config or not self.late_fusion_config.enabled:
            # Fallback to primary index if fusion not enabled
            primary_index = self.text_index or self.image_index or self.unified_index
            if primary_index:
                return primary_index.search(query_vectors, k)
            raise ValueError("No indexes available")
        
        # Parallel retrieval
        fusion_config = self.late_fusion_config
        unified_k = fusion_config.unified_top_k or k
        
        multi_result = parallel_retrieve(
            text_index=self.text_index,
            image_index=self.image_index,
            unified_index=self.unified_index,
            query_vectors=query_vectors,
            k=k,
            unified_k=unified_k,
            parallel=True,
        )
        
        # Late fusion
        fused_distances, fused_indices, fusion_metadata = fuse_results(
            result=multi_result,
            k=k,
            method=fusion_config.fusion_method,
            normalize=fusion_config.normalize_scores,
            text_weight=fusion_config.text_weight,
            image_weight=fusion_config.image_weight,
            unified_weight=fusion_config.unified_weight,
            rrf_k=fusion_config.rrf_k,
        )
        
        # Diversity enforcement (if enabled)
        if self.diversity_config and self.diversity_config.enabled and self.doc_embeddings is not None:
            fused_distances, fused_indices = enforce_diversity(
                distances=fused_distances,
                indices=fused_indices,
                doc_embeddings=self.doc_embeddings,
                k=k,
                min_distance=self.diversity_config.min_distance,
                max_results_per_cluster=self.diversity_config.max_results_per_cluster,
            )
        
        return fused_distances, fused_indices
    
    def search_hybrid(
        self,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Hybrid search - uses text index for lexical component."""
        # For hybrid, use text index for lexical part
        if self.text_index and query_texts:
            lexical_distances, lexical_indices, _ = self.text_index.search_lexical(query_texts, k)
        else:
            lexical_distances, lexical_indices = None, None
        
        # Vector part uses parallel retrieval + fusion
        if query_vectors is not None:
            vector_distances, vector_indices = self.search(query_vectors, k)
        else:
            vector_distances, vector_indices = None, None
        
        # Combine (simplified - in practice would need proper score normalization)
        if lexical_distances is not None and vector_distances is not None:
            # Simple combination (could be improved)
            combined_distances = alpha * vector_distances + (1 - alpha) * lexical_distances
            # Use vector indices as primary
            combined_indices = vector_indices
        elif vector_distances is not None:
            combined_distances, combined_indices = vector_distances, vector_indices
        elif lexical_distances is not None:
            combined_distances, combined_indices = lexical_distances, lexical_indices
        else:
            raise ValueError("Need either query_vectors or query_texts")
        
        metadata = {
            "ranking_method": "hybrid",
            "alpha": alpha,
            "multi_index": True,
        }
        
        return combined_distances, combined_indices, metadata
    
    def search_lexical(
        self,
        query_texts: List[str],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Lexical search - uses text index."""
        if not self.text_index:
            raise ValueError("Text index required for lexical search")
        
        return self.text_index.search_lexical(query_texts, k)

