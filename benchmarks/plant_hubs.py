#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Plant adversarial hubs into RAG benchmark dataset.

This script:
1. Loads benchmark dataset
2. Creates adversarial hubs using specified strategy
3. Inserts hubs into embeddings
4. Updates metadata with ground truth labels
5. Saves modified dataset
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from hub_strategies import get_strategy, STRATEGIES


def plant_hubs(
    dataset_dir: Path,
    strategy_name: str,
    hub_rate: float = 0.01,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Plant adversarial hubs into benchmark dataset.
    
    Args:
        dataset_dir: Directory containing benchmark dataset
        strategy_name: Hub planting strategy ("geometric", "multi_centroid", "gradient", "stealth", or "all")
        hub_rate: Fraction of documents that should be hubs (e.g., 0.01 = 1%)
        output_dir: Output directory (defaults to dataset_dir)
        
    Returns:
        Statistics about planted hubs
    """
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = dataset_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Planting hubs using strategy: {strategy_name}")
    print(f"Hub rate: {hub_rate * 100:.2f}%")
    
    # Load dataset
    print("\n1. Loading dataset...")
    embeddings = np.load(dataset_dir / "embeddings.npy")
    
    with open(dataset_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    with open(dataset_dir / "dataset_info.json", "r") as f:
        dataset_info = json.load(f)
    
    N, D = embeddings.shape
    print(f"Loaded {N} embeddings of dimension {D}")
    
    # Calculate number of hubs to plant
    num_hubs = max(1, int(N * hub_rate))
    print(f"Planting {num_hubs} adversarial hubs")
    
    # Get strategy/strategies
    if strategy_name == "all":
        strategies = list(STRATEGIES.values())
        hubs_per_strategy = max(1, num_hubs // len(strategies))
        print(f"Using all {len(strategies)} strategies, {hubs_per_strategy} hubs each")
    else:
        strategies = [get_strategy(strategy_name)]
        hubs_per_strategy = num_hubs
    
    # Create hubs
    print("\n2. Creating adversarial hubs...")
    all_hub_embeddings = []
    all_hub_metadata = []
    
    for strategy in strategies:
        print(f"\n   Strategy: {strategy.name}")
        print(f"   {strategy.description}")
        
        hub_embeddings, hub_meta = strategy.create_hub(
            embeddings,
            num_hubs=hubs_per_strategy
        )
        
        all_hub_embeddings.append(hub_embeddings)
        
        # Add strategy-specific metadata for each hub
        for i in range(len(hub_embeddings)):
            hub_metadata_entry = {
                "hub_id": f"hub_{len(all_hub_metadata):04d}",
                "strategy": strategy.name,
                "strategy_description": strategy.description,
                **hub_meta,
            }
            all_hub_metadata.append(hub_metadata_entry)
        
        print(f"   Created {len(hub_embeddings)} hubs")
    
    # Combine all hubs
    all_hub_embeddings = np.vstack(all_hub_embeddings)
    total_hubs = len(all_hub_embeddings)
    print(f"\nTotal hubs created: {total_hubs}")
    
    # Insert hubs into embeddings
    print("\n3. Inserting hubs into dataset...")
    
    # Randomly select positions to replace with hubs
    np.random.seed(42)
    hub_positions = np.random.choice(N, total_hubs, replace=False)
    hub_positions = sorted(hub_positions)
    
    # Create new embeddings array with hubs
    modified_embeddings = embeddings.copy()
    
    for i, pos in enumerate(hub_positions):
        modified_embeddings[pos] = all_hub_embeddings[i]
    
    # Update metadata
    print("\n4. Updating metadata...")
    modified_metadata = metadata.copy()
    
    for i, pos in enumerate(hub_positions):
        # Mark as adversarial
        modified_metadata[pos]["is_adversarial"] = True
        modified_metadata[pos]["hub_id"] = all_hub_metadata[i]["hub_id"]
        modified_metadata[pos]["hub_strategy"] = all_hub_metadata[i]["strategy"]
        modified_metadata[pos]["hub_strategy_description"] = all_hub_metadata[i]["strategy_description"]
        
        # Add synthetic text to indicate it's a hub
        modified_metadata[pos]["original_text"] = modified_metadata[pos]["text"]
        modified_metadata[pos]["text"] = f"[ADVERSARIAL HUB {all_hub_metadata[i]['hub_id']}] {modified_metadata[pos]['text'][:100]}..."
        modified_metadata[pos]["text_hash"] = hashlib.md5(modified_metadata[pos]["text"].encode()).hexdigest()
    
    # Save modified dataset
    print("\n5. Saving modified dataset...")
    
    # Save embeddings
    np.save(output_dir / "embeddings.npy", modified_embeddings)
    print(f"Saved embeddings: {output_dir / 'embeddings.npy'}")
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(modified_metadata, f, indent=2)
    print(f"Saved metadata: {output_dir / 'metadata.json'}")
    
    # Save ground truth
    ground_truth = {
        "num_total": N,
        "num_adversarial": total_hubs,
        "hub_rate": total_hubs / N,
        "hub_positions": hub_positions.tolist(),
        "hub_ids": [meta["hub_id"] for meta in all_hub_metadata],
        "strategies": [meta["strategy"] for meta in all_hub_metadata],
        "hub_metadata": all_hub_metadata,
    }
    
    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth: {output_dir / 'ground_truth.json'}")
    
    # Update dataset info
    dataset_info["has_adversarial_hubs"] = True
    dataset_info["num_adversarial_hubs"] = total_hubs
    dataset_info["hub_rate"] = total_hubs / N
    dataset_info["hub_strategies"] = list(set(meta["strategy"] for meta in all_hub_metadata))
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset info: {output_dir / 'dataset_info.json'}")
    
    print(f"\n✅ Hubs planted successfully!")
    print(f"   Total chunks: {N}")
    print(f"   Adversarial hubs: {total_hubs} ({total_hubs / N * 100:.2f}%)")
    print(f"   Strategies: {', '.join(set(meta['strategy'] for meta in all_hub_metadata))}")
    print(f"   Output: {output_dir}")
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Plant adversarial hubs in RAG benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark dataset directory",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Hub planting strategy",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.01,
        help="Hub rate (fraction of documents that should be hubs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (defaults to dataset directory)",
    )
    
    args = parser.parse_args()
    
    plant_hubs(
        dataset_dir=Path(args.dataset),
        strategy_name=args.strategy,
        hub_rate=args.rate,
        output_dir=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()

