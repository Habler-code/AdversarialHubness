#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Adversarial hub planting strategies for RAG benchmarks.

Implements different strategies for creating adversarial hubs:
1. Geometric Hub (Center of Mass)
2. Multi-Centroid Hub
3. Gradient-Based Hub
4. Stealth Hub (Low Similarity)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.cluster import MiniBatchKMeans


class HubStrategy:
    """Base class for hub planting strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create adversarial hub embeddings.
        
        Args:
            embeddings: Document embeddings (N, D)
            num_hubs: Number of hubs to create
            **kwargs: Strategy-specific parameters
            
        Returns:
            Tuple of (hub_embeddings, metadata)
        """
        raise NotImplementedError


class GeometricHubStrategy(HubStrategy):
    """
    Strategy A: Geometric Hub (Center of Mass)
    
    Creates a hub embedding as the weighted average of multiple diverse
    document embeddings. This hub will be geometrically close to many queries.
    
    Detection difficulty: Easy to Medium
    """
    
    def __init__(self):
        super().__init__(
            name="geometric_hub",
            description="Hub as weighted average of diverse documents"
        )
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        num_samples: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create geometric hubs.
        
        Args:
            embeddings: Document embeddings (N, D)
            num_hubs: Number of hubs to create
            num_samples: Number of documents to average
            
        Returns:
            Tuple of (hub_embeddings, metadata)
        """
        N, D = embeddings.shape
        hub_embeddings = []
        
        for i in range(num_hubs):
            # Sample diverse documents
            np.random.seed(42 + i)
            sample_indices = np.random.choice(N, num_samples, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # Create hub as weighted average
            # Give higher weight to documents that are far apart (more diverse)
            hub = np.mean(sample_embeddings, axis=0)
            
            # Normalize
            hub = hub / np.linalg.norm(hub)
            hub_embeddings.append(hub)
        
        hub_embeddings = np.array(hub_embeddings, dtype=np.float32)
        
        metadata = {
            "strategy": self.name,
            "num_samples": num_samples,
            "description": self.description,
        }
        
        return hub_embeddings, metadata


class MultiCentroidHubStrategy(HubStrategy):
    """
    Strategy B: Multi-Centroid Hub
    
    Creates multiple hub variants that target different semantic clusters.
    Each variant is optimized for a specific cluster of queries.
    
    Detection difficulty: Medium
    """
    
    def __init__(self):
        super().__init__(
            name="multi_centroid_hub",
            description="Multiple hubs targeting different semantic clusters"
        )
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        num_clusters: int = 10,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create multi-centroid hubs.
        
        Args:
            embeddings: Document embeddings (N, D)
            num_hubs: Number of hubs to create
            num_clusters: Number of clusters to target
            
        Returns:
            Tuple of (hub_embeddings, metadata)
        """
        N, D = embeddings.shape
        
        # Cluster embeddings
        print(f"Clustering {N} embeddings into {num_clusters} clusters...")
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_
        
        hub_embeddings = []
        
        for i in range(num_hubs):
            # Sample subset of centroids
            np.random.seed(42 + i)
            num_target_clusters = max(3, num_clusters // 3)
            target_cluster_indices = np.random.choice(num_clusters, num_target_clusters, replace=False)
            target_centroids = centroids[target_cluster_indices]
            
            # Create hub as average of target centroids
            hub = np.mean(target_centroids, axis=0)
            
            # Normalize
            hub = hub / np.linalg.norm(hub)
            hub_embeddings.append(hub)
        
        hub_embeddings = np.array(hub_embeddings, dtype=np.float32)
        
        metadata = {
            "strategy": self.name,
            "num_clusters": num_clusters,
            "target_clusters_per_hub": num_target_clusters,
            "description": self.description,
        }
        
        return hub_embeddings, metadata


class GradientBasedHubStrategy(HubStrategy):
    """
    Strategy C: Gradient-Based Adversarial Hub
    
    Uses gradient descent to optimize hub embedding to maximize retrieval
    probability across diverse queries.
    
    Detection difficulty: Hard
    """
    
    def __init__(self):
        super().__init__(
            name="gradient_based_hub",
            description="Hub optimized via gradient descent for maximum retrieval"
        )
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create gradient-based hubs.
        
        Args:
            embeddings: Document embeddings (N, D)
            num_hubs: Number of hubs to create
            num_iterations: Number of gradient descent iterations
            learning_rate: Learning rate
            
        Returns:
            Tuple of (hub_embeddings, metadata)
        """
        N, D = embeddings.shape
        hub_embeddings = []
        
        for i in range(num_hubs):
            # Initialize hub randomly
            np.random.seed(42 + i)
            hub = np.random.randn(D).astype(np.float32)
            hub = hub / np.linalg.norm(hub)
            
            # Sample query documents
            num_query_samples = min(1000, N)
            query_indices = np.random.choice(N, num_query_samples, replace=False)
            query_embeddings = embeddings[query_indices]
            
            # Gradient descent to maximize similarity
            for iteration in range(num_iterations):
                # Compute similarities
                similarities = np.dot(query_embeddings, hub)
                
                # Gradient: maximize mean similarity
                gradient = np.mean(query_embeddings, axis=0)
                
                # Update hub
                hub = hub + learning_rate * gradient
                
                # Normalize
                hub = hub / np.linalg.norm(hub)
            
            hub_embeddings.append(hub)
        
        hub_embeddings = np.array(hub_embeddings, dtype=np.float32)
        
        metadata = {
            "strategy": self.name,
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "description": self.description,
        }
        
        return hub_embeddings, metadata


class StealthHubStrategy(HubStrategy):
    """
    Strategy D: Stealth Hub (Low Similarity)
    
    Creates hub that appears at lower ranks but consistently across many queries.
    Harder to detect with traditional methods because it doesn't have very high
    similarity scores.
    
    Detection difficulty: Very Hard
    """
    
    def __init__(self):
        super().__init__(
            name="stealth_hub",
            description="Hub with moderate similarity but high consistency"
        )
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        target_similarity: float = 0.3,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create stealth hubs.
        
        Args:
            embeddings: Document embeddings (N, D)
            num_hubs: Number of hubs to create
            target_similarity: Target similarity (lower = stealthier)
            
        Returns:
            Tuple of (hub_embeddings, metadata)
        """
        N, D = embeddings.shape
        hub_embeddings = []
        
        for i in range(num_hubs):
            # Start with geometric hub
            np.random.seed(42 + i)
            sample_indices = np.random.choice(N, 100, replace=False)
            sample_embeddings = embeddings[sample_indices]
            hub = np.mean(sample_embeddings, axis=0)
            
            # Add noise to reduce similarity
            noise = np.random.randn(D).astype(np.float32)
            noise = noise / np.linalg.norm(noise)
            
            # Mix hub with noise to achieve target similarity
            # target_similarity ≈ cosine(hub, noisy_hub)
            # We want: alpha * hub + beta * noise, normalized
            # where cosine(hub, result) ≈ target_similarity
            
            alpha = target_similarity
            beta = np.sqrt(1 - alpha**2)
            
            noisy_hub = alpha * hub + beta * noise
            noisy_hub = noisy_hub / np.linalg.norm(noisy_hub)
            
            hub_embeddings.append(noisy_hub)
        
        hub_embeddings = np.array(hub_embeddings, dtype=np.float32)
        
        metadata = {
            "strategy": self.name,
            "target_similarity": target_similarity,
            "description": self.description,
        }
        
        return hub_embeddings, metadata


# Registry of available strategies
STRATEGIES = {
    "geometric": GeometricHubStrategy(),
    "multi_centroid": MultiCentroidHubStrategy(),
    "gradient": GradientBasedHubStrategy(),
    "stealth": StealthHubStrategy(),
}


def get_strategy(name: str) -> HubStrategy:
    """Get hub strategy by name."""
    if name == "all":
        return list(STRATEGIES.values())
    
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name]

