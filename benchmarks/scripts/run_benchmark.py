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
from typing import Dict, Any, List, Optional

import sys
# Add project root to path for hubscan import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hubscan import Config, Scanner, get_suspicious_documents, Verdict


def run_benchmark(
    dataset_dir: Path,
    config_path: Path,
    output_dir: Path,
    query_embeddings_path: Optional[Path] = None,
    ranking_methods: Optional[List[str]] = None,
    enable_concept_aware: bool = False,
    enable_modality_aware: bool = False,
) -> Dict[str, Any]:
    """
    Run benchmark.
    
    Args:
        dataset_dir: Dataset directory with planted hubs
        config_path: HubScan configuration file
        output_dir: Output directory for results
        ranking_methods: Optional list of ranking methods to compare (default: use config)
        
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
    all_hub_positions = ground_truth["hub_positions"]
    all_strategies = ground_truth["strategies"]
    
    print(f"Total chunks: {num_total}")
    print(f"Total adversarial hubs: {num_adversarial} ({num_adversarial / num_total * 100:.2f}%)")
    print(f"Hub strategies: {set(all_strategies)}")
    
    # Map ranking methods to their optimized hub strategies
    ranking_to_strategies = {
        "vector": ["geometric_hub", "gradient_based_hub", "multi_centroid_hub"],
        "hybrid": ["geometric_hub", "gradient_based_hub", "multi_centroid_hub", "lexical_hub"],  # Hybrid can detect both
        "lexical": ["lexical_hub"],
        # Concept and cross-modal strategies can be tested with any method
        "concept_specific": ["concept_specific_hub"],
        "cross_modal": ["cross_modal_hub"],
        # Reranking can be applied to any method, so we don't need separate hub types
    }
    
    # Check for concept-specific and cross-modal hubs
    has_concept_hubs = "concept_specific_hub" in all_strategies
    has_cross_modal_hubs = "cross_modal_hub" in all_strategies
    
    if has_concept_hubs:
        print(f"Dataset contains concept-specific hubs")
        # Add concept hubs to all ranking methods
        for method in ["vector", "hybrid", "lexical"]:
            ranking_to_strategies[method].append("concept_specific_hub")
    
    if has_cross_modal_hubs:
        print(f"Dataset contains cross-modal hubs")
        # Add cross-modal hubs to all ranking methods
        for method in ["vector", "hybrid", "lexical"]:
            ranking_to_strategies[method].append("cross_modal_hub")
    
    # Load HubScan config
    print("\n2. Loading HubScan configuration...")
    config = Config.from_yaml(str(config_path))
    
    # Override input paths to use benchmark dataset
    config.input.mode = "embeddings_only"
    config.input.embeddings_path = str(dataset_dir / "embeddings.npy")
    config.input.metadata_path = str(dataset_dir / "metadata.json")
    config.output.out_dir = str(output_dir)

    # If query embeddings are provided (or exist in dataset), use real query sampling.
    # This enables paper-style evaluation where queries != documents (e.g., text->image retrieval).
    if query_embeddings_path is None:
        candidate = dataset_dir / "query_embeddings.npy"
        if candidate.exists():
            query_embeddings_path = candidate

    if query_embeddings_path is not None:
        config.scan.query_sampling = "real_queries"
        config.scan.query_embeddings_path = str(query_embeddings_path)
        print(f"Using real query embeddings: {query_embeddings_path}")
    
    # Enable concept/modality-aware detection if requested
    if enable_concept_aware or has_concept_hubs:
        print("Enabling concept-aware detection")
        config.detectors.concept_aware.enabled = True
    
    if enable_modality_aware or has_cross_modal_hubs:
        print("Enabling modality-aware detection")
        config.detectors.modality_aware.enabled = True
    
    # Check if query texts exist for lexical/hybrid search
    query_texts_path = dataset_dir / "query_texts.json"
    has_query_texts = query_texts_path.exists()
    
    if has_query_texts:
        config.scan.query_texts_path = str(query_texts_path)
        print(f"Found query texts: {query_texts_path}")
    else:
        print(f"Warning: query_texts.json not found. Lexical/hybrid search will not be available.")
    
    print(f"Config: {config_path}")
    
    # Determine ranking methods to test
    if ranking_methods is None:
        ranking_methods = [config.scan.ranking.method]
    
    # Check if rank-bm25 is available for lexical/hybrid
    try:
        import rank_bm25  # type: ignore
        has_bm25 = True
    except ImportError:
        has_bm25 = False
        print("Warning: rank-bm25 not installed. Lexical/hybrid search will be skipped.")
        print("  Install with: pip install rank-bm25")
    
    # Filter out methods that require query texts if not available
    if not has_query_texts:
        ranking_methods = [m for m in ranking_methods if m not in ["lexical", "hybrid"]]
        if not ranking_methods:
            print("Error: No valid ranking methods available. Need query_texts.json for lexical/hybrid.")
            return None
    
    # Filter out methods that require BM25 if not available
    if not has_bm25:
        ranking_methods = [m for m in ranking_methods if m not in ["lexical", "hybrid"]]
        if not ranking_methods:
            print("Error: No valid ranking methods available. Need rank-bm25 for lexical/hybrid.")
            return None
    
    print(f"Ranking methods to test: {', '.join(ranking_methods)}")
    
    # Run HubScan for each ranking method
    all_results = {}
    all_runtimes = {}
    method_ground_truths = {}  # Store filtered ground truth per method
    
    for ranking_method in ranking_methods:
        print(f"\n3. Running HubScan with {ranking_method} ranking...")
        
        # Filter ground truth to only include hubs optimized for this ranking method
        # Handle reranking suffix by using base method strategies
        base_method = ranking_method.replace("+rerank", "")
        method_strategies = ranking_to_strategies.get(ranking_method, ranking_to_strategies.get(base_method, []))
        method_hub_positions = set()
        method_hub_indices = []
        method_strategy_list = []
        
        for i, (pos, strategy) in enumerate(zip(all_hub_positions, all_strategies)):
            if strategy in method_strategies:
                method_hub_positions.add(pos)
                method_hub_indices.append(i)
                method_strategy_list.append(strategy)
        
        method_ground_truth = {
            "num_total": num_total,
            "num_adversarial": len(method_hub_positions),
            "hub_positions": list(method_hub_positions),
            "strategies": method_strategy_list,
            "hub_ids": [ground_truth["hub_ids"][i] for i in method_hub_indices],
        }
        # Store ground truth with original ranking_method name (will be updated later if reranking is enabled)
        method_ground_truths[ranking_method] = method_ground_truth
        
        print(f"  Optimized hubs for {ranking_method}: {len(method_hub_positions)} ({', '.join(set(method_strategy_list))})")
        
        # Parse ranking method and reranking flag
        # Format: "vector", "hybrid", "lexical", or "vector+rerank", "hybrid+rerank", etc.
        if "+rerank" in ranking_method:
            base_method = ranking_method.replace("+rerank", "")
            config.scan.ranking.method = base_method
            config.scan.ranking.rerank = True
        else:
            config.scan.ranking.method = ranking_method
            config.scan.ranking.rerank = False
        
        # Set query texts path if needed (use base method, not the +rerank variant)
        base_ranking_method = config.scan.ranking.method
        if base_ranking_method in ["lexical", "hybrid"]:
            if not has_query_texts:
                print(f"  Skipping {ranking_method} - query texts not available")
                continue
            config.scan.query_texts_path = str(query_texts_path)
        else:
            # Clear query texts path for vector methods
            config.scan.query_texts_path = None
        
        # Update method name for output directory (include rerank suffix if enabled)
        output_method_name = ranking_method  # Keep original for output dir
        if config.scan.ranking.rerank and "+rerank" not in output_method_name:
            output_method_name = f"{base_ranking_method}+rerank"
        
        method_output_dir = output_dir / output_method_name if len(ranking_methods) > 1 else output_dir
        config.output.out_dir = str(method_output_dir)
        
        start_time = time.time()
        
        try:
            scanner = Scanner(config)
            scanner.load_data()
            results = scanner.scan()
        except Exception as e:
            print(f"  Error running {ranking_method}: {e}")
            continue
        
        runtime = time.time() - start_time
        all_results[output_method_name] = results
        all_runtimes[output_method_name] = runtime
        # Also store ground truth with output method name
        method_ground_truths[output_method_name] = method_ground_truth
        
        print(f"Scan completed in {runtime:.2f} seconds")
    
    # Calculate metrics for each method separately using its filtered ground truth
    print("\n4. Calculating metrics per ranking method...")
    
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
    
    # Calculate metrics for each method
    all_method_metrics = {}
    successful_methods = list(all_results.keys())  # Use actual output method names
    
    for method_name in successful_methods:
        method_results = all_results[method_name]
        method_verdicts = method_results["verdicts"]
        method_gt = method_ground_truths.get(method_name, method_ground_truths.get(method_name.replace("+rerank", ""), {}))
        method_hub_positions = set(method_gt["hub_positions"])
        
        # Get detected hubs for this method
        method_detected_high = set()
        method_detected_medium = set()
        method_detected_all = set()
        
        for doc_idx, verdict in method_verdicts.items():
            if verdict == Verdict.HIGH:
                method_detected_high.add(doc_idx)
                method_detected_all.add(doc_idx)
            elif verdict == Verdict.MEDIUM:
                method_detected_medium.add(doc_idx)
                method_detected_all.add(doc_idx)
        
        method_metrics_high = calculate_metrics(method_detected_high, method_hub_positions, num_total)
        method_metrics_all = calculate_metrics(method_detected_all, method_hub_positions, num_total)
        
        all_method_metrics[method_name] = {
            "high": method_metrics_high,
            "all": method_metrics_all,
            "detected_high": method_detected_high,
            "detected_all": method_detected_all,
        }
        
        print(f"\n{method_name.upper()} (testing on {len(method_hub_positions)} optimized hubs):")
        print(f"  Detected HIGH: {len(method_detected_high)}")
        print(f"  Detected ALL: {len(method_detected_all)}")
        print(f"  Recall (HIGH): {method_metrics_high['recall']:.4f}")
        print(f"  Precision (HIGH): {method_metrics_high['precision']:.4f}")
    
    # Use first method's results for overall display (or aggregate)
    first_method = successful_methods[0] if successful_methods else ranking_methods[0]
    if len(successful_methods) == 1:
        results = all_results[first_method]
        runtime = all_runtimes[first_method]
        metrics_high = all_method_metrics[first_method]["high"]
        metrics_all = all_method_metrics[first_method]["all"]
        detected_high = all_method_metrics[first_method]["detected_high"]
        detected_all = all_method_metrics[first_method]["detected_all"]
        hub_positions = set(method_ground_truths.get(first_method, method_ground_truths.get(first_method.replace("+rerank", ""), {}))["hub_positions"])
    else:
        # For comparison, aggregate across methods
        results = all_results[first_method]
        runtime = sum(all_runtimes.values())
        # Use first method's metrics for overall display
        metrics_high = all_method_metrics[first_method]["high"]
        metrics_all = all_method_metrics[first_method]["all"]
        detected_high = all_method_metrics[first_method]["detected_high"]
        detected_all = all_method_metrics[first_method]["detected_all"]
        hub_positions = set(method_ground_truths.get(first_method, method_ground_truths.get(first_method.replace("+rerank", ""), {}))["hub_positions"])
    
    # Extract results
    json_report = results["json_report"]
    verdicts = results["verdicts"]
    
    print(f"\nOverall Detected (using {first_method} method):")
    print(f"  HIGH: {len(detected_high)}")
    print(f"  MEDIUM: {len(detected_all - detected_high)}")
    print(f"  Total: {len(detected_all)}")
    
    # Calculate metrics
    print("\n5. Overall Metrics...")
    
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
    
    # Compare ranking methods if multiple tested
    ranking_comparison = None
    # successful_methods already defined above as list(all_results.keys())
    if len(successful_methods) > 1:
        print("\n6. Ranking Method Comparison (each tested on its optimized hubs):")
        ranking_comparison = {}
        print(f"{'Method':<15} {'Hubs':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Runtime':<12}")
        print("-" * 75)
        for method in successful_methods:
            method_gt = method_ground_truths.get(method, method_ground_truths.get(method.replace("+rerank", ""), {}))
            method_metrics = all_method_metrics[method]
            method_metrics_high = method_metrics["high"]
            
            ranking_comparison[method] = {
                "runtime": all_runtimes[method],
                "num_hubs": len(method_gt["hub_positions"]),
                "hub_strategies": list(set(method_gt["strategies"])),
                "metrics_high": method_metrics_high,
                "metrics_all": method_metrics["all"],
                "detection_metrics": all_results[method].get("detection_metrics"),
            }
            
            strategies_str = ",".join(set(method_gt["strategies"]))[:20]
            print(f"{method:<15} {len(method_gt['hub_positions']):<8} {method_metrics_high['precision']:<12.4f} {method_metrics_high['recall']:<12.4f} {method_metrics_high['f1']:<12.4f} {all_runtimes[method]:<12.2f}")
            print(f"  Strategies: {strategies_str}")
    
    # Analyze by strategy (for the method being displayed)
    print("\n7. Analyzing by strategy...")
    
    strategy_metrics = {}
    current_gt = method_ground_truths.get(first_method, method_ground_truths.get(first_method.replace("+rerank", ""), {}))
    
    for strategy_name in set(current_gt["strategies"]):
        strategy_positions = set(
            pos for pos, strat in zip(current_gt["hub_positions"], current_gt["strategies"])
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
    print("\n7. Saving results...")
    
    # Compare ranking methods if multiple tested
    ranking_comparison = None
    if len(ranking_methods) > 1:
        ranking_comparison = {}
        for method in ranking_methods:
            method_results = all_results[method]
            method_verdicts = method_results["verdicts"]
            method_detected_high = {idx for idx, v in method_verdicts.items() if v == Verdict.HIGH}
            method_detected_all = {idx for idx, v in method_verdicts.items() if v in [Verdict.HIGH, Verdict.MEDIUM]}
            
            ranking_comparison[method] = {
                "runtime": all_runtimes[method],
                "metrics_high": calculate_metrics(method_detected_high, hub_positions, num_total),
                "metrics_all": calculate_metrics(method_detected_all, hub_positions, num_total),
                "detection_metrics": method_results.get("detection_metrics"),
            }
    
    benchmark_results = {
        "dataset": str(dataset_dir),
        "config": str(config_path),
        "ranking_methods": ranking_methods,
        "runtime": runtime,
        "runtimes_by_method": all_runtimes,
        "ground_truth": ground_truth,  # Full ground truth
        "method_ground_truths": {  # Filtered ground truth per method
            method: gt for method, gt in method_ground_truths.items()
        },
        "detection": {
            "high": list(detected_high),
            "medium": list(detected_all - detected_high),
            "all": list(detected_all),
        },
        "metrics": {
            "high_only": metrics_high,
            "high_and_medium": metrics_all,
        },
        "metrics_by_method": {  # Metrics per method with its optimized hubs
            method: {
                "high": all_method_metrics[method]["high"],
                "all": all_method_metrics[method]["all"],
                "num_hubs": len(method_ground_truths[method]["hub_positions"]),
                "hub_strategies": list(set(method_ground_truths[method]["strategies"])),
            }
            for method in successful_methods
        },
        "metrics_by_strategy": strategy_metrics,
        "ranking_comparison": ranking_comparison,
        "hubscan_report": json_report,
        "detection_metrics": results.get("detection_metrics"),
    }
    
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Results saved: {results_path}")
    
    print("\nBenchmark completed!")
    
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
    parser.add_argument(
        "--query-embeddings",
        type=str,
        default=None,
        help="Optional path to query embeddings (.npy). If omitted, will use <dataset>/query_embeddings.npy if present.",
    )
    parser.add_argument(
        "--ranking-methods",
        nargs="+",
        choices=["vector", "hybrid", "lexical", "vector+rerank", "hybrid+rerank"],
        help="Ranking methods to compare (default: use config)",
    )
    parser.add_argument(
        "--enable-concept-aware",
        action="store_true",
        help="Enable concept-aware hub detection",
    )
    parser.add_argument(
        "--enable-modality-aware",
        action="store_true",
        help="Enable modality-aware hub detection",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset_dir=Path(args.dataset),
        config_path=Path(args.config),
        output_dir=Path(args.output),
        query_embeddings_path=Path(args.query_embeddings) if args.query_embeddings else None,
        ranking_methods=args.ranking_methods,
        enable_concept_aware=args.enable_concept_aware,
        enable_modality_aware=args.enable_modality_aware,
    )


if __name__ == "__main__":
    main()

