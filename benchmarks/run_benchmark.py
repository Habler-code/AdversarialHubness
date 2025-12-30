#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Run RAG adversarial hubness benchmark.

This script:
1. Loads benchmark dataset with planted hubs
2. Runs HubScan to detect adversarial hubs
3. Compares detected hubs to ground truth
4. Calculates metrics (precision, recall, F1)
5. Saves results
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))

from hubscan import Config, Scanner
from hubscan.sdk import get_suspicious_documents, Verdict


def run_benchmark(
    dataset_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run benchmark.
    
    Args:
        dataset_dir: Dataset directory with planted hubs
        config_path: HubScan configuration file
        output_dir: Output directory for results
        
    Returns:
        Benchmark results
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running benchmark on dataset: {dataset_dir}")
    
    # Load ground truth
    print("\n1. Loading ground truth...")
    with open(dataset_dir / "ground_truth.json", "r") as f:
        ground_truth = json.load(f)
    
    num_total = ground_truth["num_total"]
    num_adversarial = ground_truth["num_adversarial"]
    hub_positions = set(ground_truth["hub_positions"])
    
    print(f"Total chunks: {num_total}")
    print(f"Adversarial hubs: {num_adversarial} ({num_adversarial / num_total * 100:.2f}%)")
    
    # Load HubScan config
    print("\n2. Loading HubScan configuration...")
    config = Config.from_yaml(str(config_path))
    
    # Override input paths to use benchmark dataset
    config.input.mode = "embeddings_only"
    config.input.embeddings_path = str(dataset_dir / "embeddings.npy")
    config.input.metadata_path = str(dataset_dir / "metadata.json")
    config.output.out_dir = str(output_dir)
    
    print(f"Config: {config_path}")
    
    # Run HubScan
    print("\n3. Running HubScan...")
    start_time = time.time()
    
    scanner = Scanner(config)
    scanner.load_data()
    results = scanner.scan()
    
    runtime = time.time() - start_time
    print(f"Scan completed in {runtime:.2f} seconds")
    
    # Extract results
    json_report = results["json_report"]
    verdicts = results["verdicts"]
    
    # Get detected hubs
    detected_high = set()
    detected_medium = set()
    detected_all = set()
    
    for doc_idx, verdict in verdicts.items():
        if verdict == Verdict.HIGH:
            detected_high.add(doc_idx)
            detected_all.add(doc_idx)
        elif verdict == Verdict.MEDIUM:
            detected_medium.add(doc_idx)
            detected_all.add(doc_idx)
    
    print(f"\nDetected:")
    print(f"  HIGH: {len(detected_high)}")
    print(f"  MEDIUM: {len(detected_medium)}")
    print(f"  Total: {len(detected_all)}")
    
    # Calculate metrics
    print("\n4. Calculating metrics...")
    
    def calculate_metrics(detected: set, ground_truth_set: set, total: int) -> Dict[str, float]:
        """Calculate precision, recall, F1, FPR."""
        tp = len(detected & ground_truth_set)  # True positives
        fp = len(detected - ground_truth_set)  # False positives
        fn = len(ground_truth_set - detected)  # False negatives
        tn = total - tp - fp - fn  # True negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    metrics_high = calculate_metrics(detected_high, hub_positions, num_total)
    metrics_all = calculate_metrics(detected_all, hub_positions, num_total)
    
    print("\nMetrics (HIGH only):")
    print(f"  Precision: {metrics_high['precision']:.4f}")
    print(f"  Recall: {metrics_high['recall']:.4f}")
    print(f"  F1: {metrics_high['f1']:.4f}")
    print(f"  FPR: {metrics_high['fpr']:.6f}")
    print(f"  TP: {metrics_high['tp']}, FP: {metrics_high['fp']}, FN: {metrics_high['fn']}")
    
    print("\nMetrics (HIGH + MEDIUM):")
    print(f"  Precision: {metrics_all['precision']:.4f}")
    print(f"  Recall: {metrics_all['recall']:.4f}")
    print(f"  F1: {metrics_all['f1']:.4f}")
    print(f"  FPR: {metrics_all['fpr']:.6f}")
    print(f"  TP: {metrics_all['tp']}, FP: {metrics_all['fp']}, FN: {metrics_all['fn']}")
    
    # Analyze by strategy
    print("\n5. Analyzing by strategy...")
    
    strategy_metrics = {}
    for strategy_name in set(ground_truth["strategies"]):
        strategy_positions = set(
            pos for pos, strat in zip(ground_truth["hub_positions"], ground_truth["strategies"])
            if strat == strategy_name
        )
        
        strategy_metrics[strategy_name] = {
            "high": calculate_metrics(detected_high, strategy_positions, num_total),
            "all": calculate_metrics(detected_all, strategy_positions, num_total),
            "num_hubs": len(strategy_positions),
        }
        
        print(f"\n{strategy_name}:")
        print(f"  Hubs: {len(strategy_positions)}")
        print(f"  Recall (HIGH): {strategy_metrics[strategy_name]['high']['recall']:.4f}")
        print(f"  Recall (ALL): {strategy_metrics[strategy_name]['all']['recall']:.4f}")
    
    # Save results
    print("\n6. Saving results...")
    
    benchmark_results = {
        "dataset": str(dataset_dir),
        "config": str(config_path),
        "runtime": runtime,
        "ground_truth": ground_truth,
        "detection": {
            "high": list(detected_high),
            "medium": list(detected_medium),
            "all": list(detected_all),
        },
        "metrics": {
            "high_only": metrics_high,
            "high_and_medium": metrics_all,
        },
        "metrics_by_strategy": strategy_metrics,
        "hubscan_report": json_report,
    }
    
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    print("\n✅ Benchmark completed!")
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Run RAG adversarial hubness benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark dataset with planted hubs",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to HubScan configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset_dir=Path(args.dataset),
        config_path=Path(args.config),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()

