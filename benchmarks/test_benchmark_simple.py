#!/usr/bin/env python3
"""
Simple test of benchmark using existing toy data.
Tests the full workflow without requiring Wikipedia/sentence-transformers.
"""

import sys
import numpy as np
import json
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.hub_strategies import get_strategy
from benchmarks.plant_hubs import plant_hubs
from benchmarks.run_benchmark import run_benchmark


def create_test_dataset():
    """Create a simple test dataset from scratch."""
    print("Creating test dataset...")
    
    # Create synthetic embeddings
    np.random.seed(42)
    num_docs = 1000
    dim = 128
    
    # Generate random embeddings
    embeddings = np.random.randn(num_docs, dim).astype(np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Create metadata
    metadata = []
    for i in range(num_docs):
        metadata.append({
            "chunk_id": f"chunk_{i:06d}",
            "text": f"This is document {i} with some synthetic content for testing.",
            "text_hash": f"hash_{i}",
            "is_adversarial": False,
        })
    
    # Save dataset
    output_dir = Path("benchmarks/data/test/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "embeddings.npy", embeddings)
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    dataset_info = {
        "size": "test",
        "num_chunks": num_docs,
        "embedding_dim": dim,
        "synthetic": True,
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Created test dataset: {output_dir}")
    print(f"   Documents: {num_docs}")
    print(f"   Dimension: {dim}")
    
    return output_dir


def main():
    print("="*60)
    print("BENCHMARK TEST - Full Workflow")
    print("="*60)
    
    # Step 1: Create test dataset
    print("\n" + "="*60)
    print("STEP 1: Create Test Dataset")
    print("="*60)
    dataset_dir = create_test_dataset()
    
    # Step 2: Plant hubs
    print("\n" + "="*60)
    print("STEP 2: Plant Adversarial Hubs")
    print("="*60)
    
    ground_truth = plant_hubs(
        dataset_dir=dataset_dir,
        strategy_name="all",
        hub_rate=0.05,  # 5% hubs (50 hubs in 1000 docs)
        output_dir=dataset_dir,
    )
    
    print(f"\n✅ Planted {ground_truth['num_adversarial']} hubs")
    
    # Step 3: Run benchmark
    print("\n" + "="*60)
    print("STEP 3: Run Benchmark (HubScan Detection)")
    print("="*60)
    
    config_path = Path("benchmarks/configs/default.yaml")
    output_dir = Path("benchmarks/results/test/")
    
    results = run_benchmark(
        dataset_dir=dataset_dir,
        config_path=config_path,
        output_dir=output_dir,
    )
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    metrics_high = results["metrics"]["high_only"]
    metrics_all = results["metrics"]["high_and_medium"]
    
    print("\n📊 Detection Performance (HIGH verdicts only):")
    print(f"   Precision: {metrics_high['precision']:.4f}")
    print(f"   Recall:    {metrics_high['recall']:.4f}")
    print(f"   F1 Score:  {metrics_high['f1']:.4f}")
    print(f"   FPR:       {metrics_high['fpr']:.6f}")
    print(f"   TP={metrics_high['tp']}, FP={metrics_high['fp']}, FN={metrics_high['fn']}, TN={metrics_high['tn']}")
    
    print("\n📊 Detection Performance (HIGH + MEDIUM verdicts):")
    print(f"   Precision: {metrics_all['precision']:.4f}")
    print(f"   Recall:    {metrics_all['recall']:.4f}")
    print(f"   F1 Score:  {metrics_all['f1']:.4f}")
    print(f"   FPR:       {metrics_all['fpr']:.6f}")
    print(f"   TP={metrics_all['tp']}, FP={metrics_all['fp']}, FN={metrics_all['fn']}, TN={metrics_all['tn']}")
    
    print("\n📊 Performance by Hub Strategy:")
    for strategy_name, strategy_metrics in results["metrics_by_strategy"].items():
        recall_high = strategy_metrics["high"]["recall"]
        recall_all = strategy_metrics["all"]["recall"]
        num_hubs = strategy_metrics["num_hubs"]
        print(f"   {strategy_name:20s}: {num_hubs} hubs, Recall(HIGH)={recall_high:.3f}, Recall(ALL)={recall_all:.3f}")
    
    # Verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check if detection worked
    if metrics_high['recall'] > 0.5:
        print("✅ PASS: Detection is working (recall > 0.5)")
    else:
        print(f"❌ FAIL: Detection recall too low ({metrics_high['recall']:.3f})")
        return False
    
    if metrics_high['precision'] > 0.7:
        print("✅ PASS: Precision is good (> 0.7)")
    else:
        print(f"⚠️  WARNING: Precision could be better ({metrics_high['precision']:.3f})")
    
    if metrics_high['fpr'] < 0.01:
        print("✅ PASS: False positive rate is low (< 1%)")
    else:
        print(f"⚠️  WARNING: FPR is high ({metrics_high['fpr']:.3f})")
    
    print("\n" + "="*60)
    print("✅ BENCHMARK TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Dataset saved to: {dataset_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

