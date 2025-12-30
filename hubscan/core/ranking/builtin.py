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

"""Built-in ranking method implementations."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from ..io.vector_index import VectorIndex


class VectorRanking:
    """Vector similarity search ranking."""
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Perform vector similarity search."""
        if query_vectors is None:
            raise ValueError("query_vectors required for vector search")
        
        distances, indices = index.search(query_vectors, k)
        metadata = {"ranking_method": "vector"}
        return distances, indices, metadata


class HybridRanking:
    """Hybrid vector + lexical search ranking."""
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        alpha: float = 0.5,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Perform hybrid vector + lexical search."""
        distances, indices, metadata = index.search_hybrid(
            query_vectors=query_vectors,
            query_texts=query_texts,
            k=k,
            alpha=alpha,
        )
        return distances, indices, metadata


class LexicalRanking:
    """Lexical/keyword search ranking."""
    
    def search(
        self,
        index: VectorIndex,
        query_vectors: Optional[np.ndarray],
        query_texts: Optional[List[str]],
        k: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Perform lexical search."""
        if query_texts is None:
            raise ValueError("query_texts required for lexical search")
        
        distances, indices, metadata = index.search_lexical(
            query_texts=query_texts,
            k=k,
        )
        return distances, indices, metadata



