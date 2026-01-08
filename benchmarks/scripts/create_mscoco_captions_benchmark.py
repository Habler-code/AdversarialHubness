#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Create a HubScan benchmark dataset from MSCOCO captions (paper-compatible dataset source).

Why this exists:
- The adv_hub repo (paper: arXiv:2412.14113) uses standard benchmark datasets like MSCOCO,
  but it does not ship precomputed embeddings inside the repo.
- HubScan's benchmark runner expects a local folder with:
    - embeddings.npy
    - metadata.json
    - (optional) query_texts.json for lexical/hybrid ranking
    - dataset_info.json
  and then you can plant hubs with `plant_hubs.py` and evaluate with `run_benchmark.py`.

This script:
- Reads COCO "captions" annotations (e.g. captions_val2017.json)
- Treats each caption as a "document"
- Builds dense text embeddings using TF-IDF + TruncatedSVD (no GPU / model downloads required)
- Writes HubScan-compatible benchmark artifacts

NOTE: These embeddings are not the same as the paper's multimodal embeddings; they are
      a lightweight way to benchmark HubScan on a real dataset source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


def _load_coco_captions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    # Each annotation has: id (caption_id), image_id, caption
    out = []
    for a in anns:
        cap = a.get("caption")
        if not isinstance(cap, str):
            continue
        cap = cap.strip()
        if not cap:
            continue
        out.append(
            {
                "caption_id": int(a["id"]),
                "image_id": int(a["image_id"]),
                "caption": cap,
            }
        )
    return out


def _sample_queries(
    captions: List[str],
    max_queries: int,
    seed: int,
) -> List[str]:
    if max_queries <= 0:
        return []
    rng = np.random.default_rng(seed)
    if len(captions) <= max_queries:
        return captions
    idx = rng.choice(len(captions), size=max_queries, replace=False)
    return [captions[i] for i in idx]


def create_mscoco_captions_benchmark(
    coco_captions_path: Path,
    output_dir: Path,
    max_captions: int | None = None,
    embedding_dim: int = 384,
    max_features: int = 200_000,
    min_df: int = 2,
    max_df: float = 0.95,
    query_sample_size: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    if not HAS_SKLEARN:
        raise RuntimeError(
            "scikit-learn is required for TF-IDF+SVD embeddings. "
            "Install with: pip install scikit-learn"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    anns = _load_coco_captions(coco_captions_path)
    if not anns:
        raise ValueError(f"No captions found in {coco_captions_path}")

    if max_captions is not None and max_captions > 0:
        anns = anns[:max_captions]

    texts = [a["caption"] for a in anns]

    # Build a dense embedding space from text with TF-IDF -> SVD
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)

    # SVD target dim must be < min(n_samples, n_features)
    max_possible = min(X.shape[0] - 1, X.shape[1] - 1)
    if max_possible < 2:
        raise ValueError(
            f"Not enough data for SVD (samples={X.shape[0]}, features={X.shape[1]})."
        )
    k = min(int(embedding_dim), int(max_possible))

    svd = TruncatedSVD(n_components=k, random_state=seed)
    dense = svd.fit_transform(X).astype(np.float32)

    # Normalize for cosine similarity
    dense = normalize(dense, norm="l2", axis=1, copy=False).astype(np.float32)

    # Metadata in HubScan benchmark format
    metadata: List[Dict[str, Any]] = []
    for i, a in enumerate(anns):
        text = a["caption"]
        metadata.append(
            {
                "doc_id": f"mscoco_caption_{i:07d}",
                "caption_id": a["caption_id"],
                "image_id": a["image_id"],
                "text": text,
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "is_adversarial": False,
                "modality": "text",
                # No ground-truth concept labels in captions-only annotations; keep stable field.
                "concept": "mscoco",
            }
        )

    query_texts = _sample_queries(texts, max_queries=query_sample_size, seed=seed)

    dataset_info: Dict[str, Any] = {
        "source": "mscoco_captions",
        "coco_captions_path": str(coco_captions_path),
        "num_docs": len(metadata),
        "embedding_dim": int(dense.shape[1]),
        "embedding_method": "tfidf+svd",
        "tfidf": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, 2],
            "stop_words": "english",
        },
        "svd_explained_variance_ratio_sum": float(
            np.sum(svd.explained_variance_ratio_)
        ),
        "modality": "text",
        "has_concepts": True,
        "concept_note": "Captions-only dataset; concept is a constant label unless you enrich metadata.",
        "has_adversarial_hubs": False,
        "num_adversarial_hubs": 0,
    }

    # Write outputs
    np.save(output_dir / "embeddings.npy", dense)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "query_texts.json", "w") as f:
        json.dump(query_texts, f, indent=2)
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    return dataset_info


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create HubScan benchmark from MSCOCO captions annotations"
    )
    p.add_argument(
        "--coco-captions",
        type=str,
        required=True,
        help="Path to COCO captions JSON (e.g. captions_val2017.json)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for HubScan benchmark dataset",
    )
    p.add_argument(
        "--max-captions",
        type=int,
        default=None,
        help="Optional cap on number of captions (useful for a quick run)",
    )
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Target embedding dimension (reduced automatically if needed)",
    )
    p.add_argument(
        "--query-sample-size",
        type=int,
        default=10_000,
        help="How many captions to save as query_texts.json (for lexical/hybrid ranking)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = p.parse_args()
    info = create_mscoco_captions_benchmark(
        coco_captions_path=Path(args.coco_captions),
        output_dir=Path(args.output),
        max_captions=args.max_captions,
        embedding_dim=args.embedding_dim,
        query_sample_size=args.query_sample_size,
        seed=args.seed,
    )
    print("Created dataset:")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

