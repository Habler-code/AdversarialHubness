#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Pretty-print HubScan benchmark results.

This script reads the JSON produced by:
  benchmarks/scripts/run_benchmark.py

and prints a compact, human-friendly summary (great for "Evaluation" sections).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100:6.2f}%"


def _f(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _print_kv(k: str, v: Any) -> None:
    print(f"{k:<22} {v}")


def _print_table_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def pretty_print(results_path: Path) -> None:
    with open(results_path, "r") as f:
        results = json.load(f)

    _print_table_header("Benchmark summary")
    _print_kv("dataset", _get(results, "dataset"))
    _print_kv("config", _get(results, "config"))
    _print_kv("ranking_methods", ", ".join(_get(results, "ranking_methods") or []))
    _print_kv("runtime_sec", _f(_get(results, "runtime"), digits=2))

    gt_total = _get(results, "ground_truth.num_total")
    gt_hubs = _get(results, "ground_truth.num_adversarial")
    _print_kv("num_docs", gt_total)
    _print_kv("num_hubs", gt_hubs)
    if isinstance(gt_total, int) and isinstance(gt_hubs, int) and gt_total > 0:
        _print_kv("hub_rate", _pct(gt_hubs / gt_total))

    _print_table_header("Overall metrics (reported by runner)")
    m_high = _get(results, "metrics.high_only") or {}
    m_all = _get(results, "metrics.high_and_medium") or {}

    print(f"{'Scope':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}   TP  FP  FN")
    print("-" * 72)
    print(
        f"{'HIGH only':<18} "
        f"{_pct(m_high.get('precision')):>10} {_pct(m_high.get('recall')):>10} {_f(m_high.get('f1')):>10} {_pct(m_high.get('fpr')):>10}"
        f"   {m_high.get('tp','-'):>2} {m_high.get('fp','-'):>2} {m_high.get('fn','-'):>2}"
    )
    print(
        f"{'HIGH+MEDIUM':<18} "
        f"{_pct(m_all.get('precision')):>10} {_pct(m_all.get('recall')):>10} {_f(m_all.get('f1')):>10} {_pct(m_all.get('fpr')):>10}"
        f"   {m_all.get('tp','-'):>2} {m_all.get('fp','-'):>2} {m_all.get('fn','-'):>2}"
    )

    # Per-ranking-method (if present)
    mbm = _get(results, "metrics_by_method")
    if isinstance(mbm, dict) and mbm:
        _print_table_header("Per ranking method (each evaluated on its optimized hub set)")
        print(f"{'Method':<18} {'Hubs':>6} {'Prec(H)':>10} {'Rec(H)':>10} {'F1(H)':>10}")
        print("-" * 60)
        for method, md in sorted(mbm.items()):
            high = (md or {}).get("high") or {}
            num_hubs = (md or {}).get("num_hubs")
            print(
                f"{method:<18} {str(num_hubs):>6} "
                f"{_pct(high.get('precision')):>10} {_pct(high.get('recall')):>10} {_f(high.get('f1')):>10}"
            )

    # Per-strategy breakdown (if present)
    mbs = _get(results, "metrics_by_strategy")
    if isinstance(mbs, dict) and mbs:
        _print_table_header("Per hub strategy (recall against each strategy’s hubs)")
        print(f"{'Strategy':<22} {'Hubs':>6} {'Rec(HIGH)':>12} {'Rec(ALL)':>12}")
        print("-" * 56)
        for strat, sd in sorted(mbs.items()):
            num_hubs = (sd or {}).get("num_hubs")
            high = (sd or {}).get("high") or {}
            allm = (sd or {}).get("all") or {}
            print(
                f"{strat:<22} {str(num_hubs):>6} "
                f"{_pct(high.get('recall')):>12} {_pct(allm.get('recall')):>12}"
            )

    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Pretty-print HubScan benchmark results JSON")
    p.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to benchmark_results.json (from run_benchmark.py)",
    )
    args = p.parse_args()
    pretty_print(Path(args.results))


if __name__ == "__main__":
    main()

