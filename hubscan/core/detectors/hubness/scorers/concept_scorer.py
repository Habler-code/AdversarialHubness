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

"""Concept-aware scorer for detecting concept-specific hubs."""

import numpy as np
from typing import Dict, Any

from .base import ScorerResult
from ..accumulator import BucketedHubnessAccumulator
from .....utils.metrics import robust_zscore


class ConceptAwareScorer:
    """Scorer that detects concept-specific hubs.
    
    Identifies documents that are hubs within specific semantic concepts,
    even if they don't appear as global hubs. Uses:
    - Per-concept z-scores
    - Contrastive delta (hub rate in one concept vs others)
    - Bucket concentration (Gini coefficient)
    """
    
    def score(
        self,
        accumulator: BucketedHubnessAccumulator,
        config: Dict[str, Any],
        **kwargs,
    ) -> ScorerResult:
        """Compute concept-aware hubness scores.
        
        Args:
            accumulator: Accumulated hit data
            config: Configuration with keys:
                - enabled: Whether concept-aware scoring is enabled
                - concept_hub_z_threshold: Z-score threshold for concept hubs
                - use_contrastive_delta: Whether to use contrastive delta
                - use_bucket_concentration: Whether to use Gini coefficient
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ScorerResult with concept-aware scores and metadata
        """
        enabled = config.get("enabled", False)
        if not enabled:
            return ScorerResult(
                scores=np.zeros(accumulator.num_docs, dtype=np.float32),
                metadata={"enabled": False}
            )
        
        concept_hub_z_threshold = config.get("concept_hub_z_threshold", 4.0)
        use_contrastive_delta = config.get("use_contrastive_delta", True)
        use_bucket_concentration = config.get("use_bucket_concentration", True)
        
        N = accumulator.num_docs
        concept_hub_rates = accumulator.compute_concept_hub_rates()
        
        # Need at least 2 concepts for meaningful concept-aware detection
        if len(concept_hub_rates) < 2:
            return ScorerResult(
                scores=np.zeros(N, dtype=np.float32),
                metadata={"enabled": True, "num_concepts": len(concept_hub_rates)}
            )
        
        # Initialize score components
        final_scores = np.zeros(N, dtype=np.float32)
        max_concept_hub_z = np.zeros(N, dtype=np.float32)
        top_concept_ids = np.full(N, -1, dtype=np.int32)
        
        # Compute z-scores within each concept
        concept_z_scores = {}
        for concept_id, rates in concept_hub_rates.items():
            active_mask = rates > 0
            if active_mask.sum() > 2:
                z, _, _ = robust_zscore(rates[active_mask])
                # Cap z-scores to prevent numerical instability
                z = np.clip(z, -50, 50)
                full_z = np.zeros(N, dtype=np.float32)
                full_z[active_mask] = z
                concept_z_scores[concept_id] = full_z
        
        # Find max concept z-score for each doc
        for concept_id, z_scores in concept_z_scores.items():
            mask = z_scores > max_concept_hub_z
            max_concept_hub_z[mask] = z_scores[mask]
            top_concept_ids[mask] = concept_id
        
        # Boost for documents with high concept-specific hubness
        concept_boost = np.clip(max_concept_hub_z - concept_hub_z_threshold, 0, 20)
        final_scores += 0.3 * concept_boost
        
        # Contrastive delta: hub_rate(d, c) much higher than others
        if use_contrastive_delta:
            contrastive_scores = self._compute_contrastive_delta(concept_hub_rates, N)
            if contrastive_scores.max() > 0:
                norm_contrastive = contrastive_scores / (contrastive_scores.max() + 1e-8)
                final_scores += 2.0 * norm_contrastive
        
        # Bucket concentration (Gini coefficient)
        if use_bucket_concentration:
            concentration_scores = self._compute_bucket_concentration(concept_hub_rates, N)
            if concentration_scores.max() > 0:
                norm_concentration = concentration_scores / (concentration_scores.max() + 1e-8)
                final_scores += 1.5 * norm_concentration
        
        metadata = {
            "enabled": True,
            "num_concepts": len(concept_hub_rates),
            "max_concept_hub_z": max_concept_hub_z.tolist(),
            "top_concept_ids": top_concept_ids.tolist(),
        }
        
        return ScorerResult(scores=final_scores, metadata=metadata)
    
    def _compute_contrastive_delta(
        self,
        concept_hub_rates: Dict[int, np.ndarray],
        num_docs: int,
    ) -> np.ndarray:
        """Compute contrastive bucket deltas.
        
        delta(d, c) = hub_rate(d, c) - median(hub_rate(d, c') for c' != c)
        
        High delta means document is specifically targeting one concept.
        """
        deltas = np.zeros(num_docs, dtype=np.float32)
        
        concept_ids = list(concept_hub_rates.keys())
        if len(concept_ids) < 2:
            return deltas
        
        # Stack all rates: shape (num_concepts, num_docs)
        all_rates = np.stack([concept_hub_rates[cid] for cid in concept_ids])
        
        for doc_id in range(num_docs):
            rates = all_rates[:, doc_id]
            if rates.max() == 0:
                continue
            
            # For each concept, compute delta from median of others
            max_delta = 0.0
            for i, rate in enumerate(rates):
                if rate > 0:
                    other_rates = np.concatenate([rates[:i], rates[i+1:]])
                    delta = rate - np.median(other_rates)
                    max_delta = max(max_delta, delta)
            
            deltas[doc_id] = max_delta
        
        return deltas
    
    def _compute_bucket_concentration(
        self,
        concept_hub_rates: Dict[int, np.ndarray],
        num_docs: int,
    ) -> np.ndarray:
        """Compute bucket concentration using Gini coefficient.
        
        High concentration = one bucket dominates = suspicious for targeted attacks.
        """
        concentration = np.zeros(num_docs, dtype=np.float32)
        
        concept_ids = list(concept_hub_rates.keys())
        if len(concept_ids) < 2:
            return concentration
        
        # Stack all rates
        all_rates = np.stack([concept_hub_rates[cid] for cid in concept_ids])
        
        for doc_id in range(num_docs):
            rates = all_rates[:, doc_id]
            total = rates.sum()
            if total <= 0:
                continue
            
            # Normalize to probabilities
            p = rates / total
            
            # Gini coefficient
            sorted_p = np.sort(p)
            n = len(sorted_p)
            cumsum = np.cumsum(sorted_p)
            if cumsum[-1] > 0:
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                concentration[doc_id] = gini
        
        return concentration

