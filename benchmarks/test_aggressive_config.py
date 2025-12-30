#!/usr/bin/env python3
"""Test aggressive configuration for better detection of hard hubs."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.run_benchmark import run_benchmark


def main():
    print("="*60)
    print("Testing Aggressive Configuration")
    print("="*60)
    print("\nChanges from default:")
    print("  - k: 20 → 50 (deeper search)")
    print("  - Stability detector: OFF → ON")
    print("  - hub_z threshold: 4.0 → 3.5")
    print("  - percentile: 5% → 8%")
    print("  - Scoring weights adjusted for stability")
    print("="*60)
    
    dataset_dir = Path("benchmarks/data/test/")
    config_path = Path("benchmarks/configs/aggressive.yaml")
    output_dir = Path("benchmarks/results/aggressive/")
    
    results = run_benchmark(
        dataset_dir=dataset_dir,
        config_path=config_path,
        output_dir=output_dir,
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON: Default vs Aggressive")
    print("="*60)
    
    print("\nDefault Config Results:")
    print("  gradient_based_hub: 16.7% recall")
    print("  stealth_hub: 8.3% recall")
    
    print("\nAggressive Config Results:")
    metrics_by_strategy = results["metrics_by_strategy"]
    
    for strategy in ["gradient_based_hub", "stealth_hub"]:
        if strategy in metrics_by_strategy:
            recall_high = metrics_by_strategy[strategy]["high"]["recall"]
            recall_all = metrics_by_strategy[strategy]["all"]["recall"]
            print(f"  {strategy}: {recall_high*100:.1f}% recall (HIGH), {recall_all*100:.1f}% recall (ALL)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

