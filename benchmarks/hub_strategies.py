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


class LexicalHubStrategy(HubStrategy):
    """
    Strategy D: Lexical Hub (Keyword-Optimized)
    
    Creates hubs optimized for lexical/keyword search (BM25) by generating
    documents that contain common keywords from queries. These hubs will rank
    highly in BM25-based retrieval but may not be semantically similar.
    
    Detection difficulty: Medium (for lexical search)
    """
    
    def __init__(self):
        super().__init__(
            name="lexical_hub",
            description="Hub optimized for keyword/lexical search (BM25)"
        )
    
    def create_hub(
        self,
        embeddings: np.ndarray,
        num_hubs: int = 1,
        query_texts: List[str] = None,
        doc_texts: List[str] = None,
        top_k_keywords: int = 50,
        min_keyword_freq: int = 3,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create lexical hubs optimized for BM25 retrieval.
        
        Args:
            embeddings: Document embeddings (N, D) - used for embedding the hub text
            num_hubs: Number of hubs to create
            query_texts: List of query texts to analyze for common keywords
            doc_texts: List of document texts (optional, for context)
            top_k_keywords: Number of top keywords to include in hub
            min_keyword_freq: Minimum frequency for a keyword to be considered
            
        Returns:
            Tuple of (hub_embeddings, hub_texts, metadata)
        """
        if query_texts is None or len(query_texts) == 0:
            raise ValueError("query_texts required for lexical hub strategy")
        
        from collections import Counter
        import re
        
        # Extract keywords from queries
        all_keywords = []
        for query in query_texts:
            # Simple tokenization (split on whitespace, lowercase)
            words = re.findall(r'\b\w+\b', query.lower())
            all_keywords.extend(words)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Filter common stopwords and get top keywords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'some', 'any', 'no', 'not', 'only', 'just', 'more', 'most', 'very',
            'too', 'so', 'than', 'then', 'now', 'here', 'there', 'up', 'down',
            'out', 'off', 'over', 'under', 'again', 'further', 'once', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below'
        }
        
        # Filter stopwords and get top keywords
        filtered_keywords = {
            word: count for word, count in keyword_counts.items()
            if word not in stopwords and len(word) > 2 and count >= min_keyword_freq
        }
        
        top_keywords = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)[:top_k_keywords]
        
        if not top_keywords:
            # Fallback: use most common words even if they're stopwords
            top_keywords = keyword_counts.most_common(top_k_keywords)
        
        print(f"Top keywords for lexical hubs: {[w for w, _ in top_keywords[:10]]}")
        
        # Create hub texts by combining top keywords with context
        hub_texts = []
        hub_embeddings_list = []
        
        # Try to import sentence-transformers for embedding generation
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            has_model = True
        except ImportError:
            print("Warning: sentence-transformers not available. Using random embeddings.")
            has_model = False
        
        for i in range(num_hubs):
            # AGGRESSIVE Strategy: Create super-hubs that match MOST queries
            # Include MANY query text snippets + EXTREME keyword repetition
            
            # Sample MANY query texts to maximize match probability
            # More queries = more potential matches = higher hubness score
            np.random.seed(42 + i)
            num_query_samples = min(60, len(query_texts))  # Use up to 60 queries per hub (balanced)
            sampled_query_indices = np.random.choice(len(query_texts), num_query_samples, replace=False)
            sampled_queries = [query_texts[idx] for idx in sampled_query_indices]
            
            hub_sentences = []
            
            # Method 1: Include MANY actual query text snippets (direct matches)
            # Include first 120 words of each query for better matching
            # Repeat 3 times to maximize term frequency while keeping hubs natural-looking
            for q in sampled_queries:
                q_words = q.split()[:120]  # Take first 120 words
                hub_sentences.append(" ".join(q_words))
                # Repeat each query snippet 3 times for higher term frequency
                hub_sentences.append(" ".join(q_words))
                hub_sentences.append(" ".join(q_words))
            
            # Method 2: Extract ALL keywords from sampled queries and repeat EXTENSIVELY
            all_query_keywords = []
            for q in sampled_queries:
                words = re.findall(r'\b\w+\b', q.lower())
                all_query_keywords.extend([w for w in words if len(w) > 2])
            
            # Combine with top keywords from all queries
            combined_keywords = [w for w, _ in top_keywords[:40]] + all_query_keywords
            keyword_counts_local = Counter(combined_keywords)
            top_local_keywords = sorted(keyword_counts_local.items(), key=lambda x: x[1], reverse=True)[:50]
            keywords_to_use = [w for w, _ in top_local_keywords]
            
            # Method 3: EXTREME keyword repetition (150 times per keyword for high BM25 scores)
            # BM25 heavily rewards term frequency, so maximize it aggressively
            # This makes hubs appear in MANY query results, increasing hubness score significantly
            for keyword in keywords_to_use[:28]:  # Use top 28 keywords (more keywords = more matches)
                for _ in range(150):  # Repeat each keyword 150 times for high BM25 scores
                    hub_sentences.append(f"This comprehensive document extensively covers {keyword} technology and {keyword} systems.")
                    hub_sentences.append(f"The {keyword} system and {keyword} methods are extensively discussed and {keyword} applications.")
                    hub_sentences.append(f"Key aspects include {keyword} applications and {keyword} implementations and {keyword} solutions.")
                    hub_sentences.append(f"This guide extensively discusses {keyword} approaches and {keyword} solutions and {keyword} technologies.")
                    hub_sentences.append(f"Important topics cover {keyword} systems and {keyword} technologies and {keyword} methods.")
            
            # Method 4: Create ULTRA-dense keyword paragraph (maximum term frequency)
            # Repeat each keyword 150 times in a dense block for high BM25 scores
            dense_keywords = []
            for keyword in keywords_to_use[:22]:
                dense_keywords.extend([keyword] * 150)  # Repeat each keyword 150 times
            hub_sentences.append(" ".join(dense_keywords))
            hub_sentences.append(" ".join(dense_keywords))  # Repeat the dense block
            
            # Method 5: Create multi-keyword phrases that match common query patterns
            # Increase repetition to ensure hubs match MANY query combinations
            for j in range(0, min(35, len(keywords_to_use)), 2):  # More keyword pairs
                if j + 1 < len(keywords_to_use):
                    kw1, kw2 = keywords_to_use[j], keywords_to_use[j+1]
                    for _ in range(35):  # Repeat phrases 35 times (balanced)
                        hub_sentences.append(f"This comprehensive resource extensively covers {kw1} and {kw2} technologies.")
                        hub_sentences.append(f"Important topics include {kw1} systems and {kw2} applications.")
                        hub_sentences.append(f"Key aspects discuss {kw1} methods and {kw2} implementations.")
                        hub_sentences.append(f"Detailed coverage of {kw1} approaches and {kw2} solutions.")
            
            # Method 6: Add triple-keyword combinations for even more matches
            for j in range(0, min(24, len(keywords_to_use)), 3):
                if j + 2 < len(keywords_to_use):
                    kw1, kw2, kw3 = keywords_to_use[j], keywords_to_use[j+1], keywords_to_use[j+2]
                    for _ in range(25):  # Repeat triple combinations
                        hub_sentences.append(f"This document covers {kw1} {kw2} and {kw3} technologies.")
                        hub_sentences.append(f"Important topics include {kw1} {kw2} {kw3} systems.")
                        hub_sentences.append(f"Key aspects discuss {kw1} {kw2} {kw3} methods.")
            
            hub_text = " ".join(hub_sentences)
            
            # Ensure long length (1200-1500 words) for high BM25 scores
            # Longer documents with high keyword density = higher BM25 scores = more top-k appearances
            words = hub_text.split()
            if len(words) < 1200:
                # Repeat to reach target length
                repetitions = (1200 // len(words)) + 1
                hub_text = " ".join([hub_text] * repetitions)
            
            # Limit maximum length but keep it long
            words = hub_text.split()
            if len(words) > 1500:
                hub_text = " ".join(words[:1500])
            
            hub_texts.append(hub_text)
            
            # Generate embedding for hub text
            if has_model:
                hub_embedding = model.encode(hub_text, show_progress_bar=False)
                hub_embedding = hub_embedding / np.linalg.norm(hub_embedding)  # Normalize
            else:
                # Fallback: use average of random document embeddings
                np.random.seed(42 + i)
                sample_indices = np.random.choice(len(embeddings), 10, replace=False)
                hub_embedding = np.mean(embeddings[sample_indices], axis=0)
                hub_embedding = hub_embedding / np.linalg.norm(hub_embedding)
            
            hub_embeddings_list.append(hub_embedding.astype(np.float32))
        
        hub_embeddings = np.array(hub_embeddings_list, dtype=np.float32)
        
        metadata = {
            "strategy": self.name,
            "top_keywords": [w for w, _ in top_keywords[:20]],
            "num_keywords_used": top_k_keywords,
            "hub_texts": hub_texts,
            "description": self.description,
        }
        
        return hub_embeddings, metadata


# Registry of available strategies
STRATEGIES = {
    "geometric": GeometricHubStrategy(),
    "multi_centroid": MultiCentroidHubStrategy(),
    "gradient": GradientBasedHubStrategy(),
    "lexical": LexicalHubStrategy(),
}


def get_strategy(name: str) -> HubStrategy:
    """Get hub strategy by name."""
    if name == "all":
        return list(STRATEGIES.values())
    
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    
    return STRATEGIES[name]

