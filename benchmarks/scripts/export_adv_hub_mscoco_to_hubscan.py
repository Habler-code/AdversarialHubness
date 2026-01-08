#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Export an adv_hub MSCOCO "poisoned" run into HubScan benchmark format.

Goal
----
The adv_hub paper (arXiv:2412.14113) produces *adversarial hubs* (poisoned images/audio)
for multimodal retrieval (e.g., text->image on MSCOCO using ImageBind/OpenCLIP).

HubScan can then scan the *document index* (image embeddings) using *real query embeddings*
(text embeddings) and (optionally) compute detection metrics if `is_adversarial` exists in
metadata.

This script builds a HubScan dataset directory containing:
  - embeddings.npy           (document embeddings: clean MSCOCO val images + poisoned adv images)
  - query_embeddings.npy     (query embeddings: MSCOCO val captions embedded by the same model)
  - metadata.json            (one entry per document, includes is_adversarial + modality)
  - ground_truth.json        (hub positions for evaluation)
  - dataset_info.json

Assumptions / expectations
--------------------------
This script is designed to be run in an environment where the paper pipeline already runs:
- torch, torchvision, PIL, etc.
- the model code used by adv_hub (ImageBind/OpenCLIP) is available
- COCO val2017 images + captions annotations are present
- you have already run adv_hub to generate `x_advs.npy` for MSCOCO (poisoned images tensor)

Important: adv_hub's ImageBind weights path is hard-coded relative to CWD in their fork.
This exporter will `chdir` into adv_hub_root before loading the model so relative paths
match the paper repo expectations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


def _load_coco_val_annotations(captions_json: Path) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    """
    Returns:
      - image_id -> file_name (from the COCO 'images' section)
      - annotations list with fields: id, image_id, caption
    """
    with open(captions_json, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    anns = data.get("annotations", [])

    image_id_to_file: Dict[int, str] = {}
    for img in images:
        try:
            image_id_to_file[int(img["id"])] = str(img["file_name"])
        except Exception:
            continue

    cleaned_anns: List[Dict[str, Any]] = []
    for a in anns:
        cap = a.get("caption")
        if not isinstance(cap, str):
            continue
        cap = cap.strip()
        if not cap:
            continue
        cleaned_anns.append(
            {
                "caption_id": int(a["id"]),
                "image_id": int(a["image_id"]),
                "caption": cap,
            }
        )

    return image_id_to_file, cleaned_anns


def export_adv_hub_mscoco(
    adv_hub_root: Path,
    coco_val_images_dir: Path,
    coco_val_captions_json: Path,
    adv_x_advs_npy: Path,
    output_dir: Path,
    model_flag: str = "imagebind",
    device: str = "cuda:0",
    image_batch_size: int = 64,
    text_batch_size: int = 512,
) -> Dict[str, Any]:
    """
    Build HubScan-format dataset from MSCOCO val2017 + adv_hub poisoned images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Switch into adv_hub root so their relative paths (e.g. ImageBind weights) resolve.
    os.chdir(adv_hub_root)

    import torch  # noqa: WPS433

    # Reuse the paper repo's model + preprocessing logic for exact compatibility.
    # (This requires adv_hub to be importable after chdir.)
    from models import load_model  # type: ignore
    from dataset_utils import imagenet_loader  # type: ignore

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(model_flag, device=str(dev))

    # ---- Documents: clean MSCOCO val images (image embeddings)
    image_id_to_file, anns = _load_coco_val_annotations(coco_val_captions_json)
    unique_image_ids = sorted(image_id_to_file.keys())

    clean_doc_meta: List[Dict[str, Any]] = []
    clean_embeddings: List[np.ndarray] = []

    def _embed_image_batch(paths: List[Path]) -> np.ndarray:
        tensors = [imagenet_loader(str(p), model, device=str(dev)) for p in paths]
        batch = torch.cat(tensors, dim=0)  # (B, C, H, W)
        with torch.no_grad():
            emb = model.forward(batch.to(dev), "vision", normalize=False)
        return emb.detach().cpu().numpy().astype(np.float32)

    # Batch clean images for embeddings
    for i in range(0, len(unique_image_ids), image_batch_size):
        batch_ids = unique_image_ids[i : i + image_batch_size]
        batch_paths: List[Path] = []
        for image_id in batch_ids:
            fn = image_id_to_file[image_id]
            batch_paths.append(coco_val_images_dir / fn)

        embs = _embed_image_batch(batch_paths)
        clean_embeddings.append(embs)

        for image_id in batch_ids:
            fn = image_id_to_file[image_id]
            clean_doc_meta.append(
                {
                    "doc_id": f"mscoco_val2017_{image_id}",
                    "image_id": int(image_id),
                    "file_name": fn,
                    "path": str((coco_val_images_dir / fn).resolve()),
                    "modality": "image",
                    "concept": "mscoco",
                    "is_adversarial": False,
                }
            )

    clean_embeddings_np = np.vstack(clean_embeddings).astype(np.float32, copy=False)

    # ---- Poisoned docs: adv_hub adversarial images tensor (x_advs.npy)
    x_advs = np.load(adv_x_advs_npy)
    # Expected shape: (N_adv, C, H, W) matching the model input space used in adv_hub.
    x_advs_t = torch.tensor(x_advs, dtype=torch.float32, device=dev)

    adv_embeddings: List[np.ndarray] = []
    adv_doc_meta: List[Dict[str, Any]] = []
    for j in range(0, x_advs_t.shape[0], image_batch_size):
        batch = x_advs_t[j : j + image_batch_size]
        with torch.no_grad():
            emb = model.forward(batch, "vision", normalize=False)
        adv_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

    adv_embeddings_np = np.vstack(adv_embeddings).astype(np.float32, copy=False)
    for k in range(adv_embeddings_np.shape[0]):
        adv_doc_meta.append(
            {
                "doc_id": f"adv_hub_poison_{k:05d}",
                "source": "adv_hub",
                "adv_x_advs_path": str(Path(adv_x_advs_npy).resolve()),
                "modality": "image",
                "concept": "mscoco",
                "is_adversarial": True,
                "hub_id": f"hub_{k:05d}",
                "hub_strategy": "adv_hub_poison",
            }
        )

    # ---- Queries: MSCOCO val captions embedded as text queries
    captions = [a["caption"] for a in anns]
    query_meta: List[Dict[str, Any]] = []
    query_embeddings: List[np.ndarray] = []

    for i in range(0, len(captions), text_batch_size):
        batch_caps = captions[i : i + text_batch_size]
        with torch.no_grad():
            emb = model.forward(batch_caps, "text", normalize=False)
        query_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

    query_embeddings_np = np.vstack(query_embeddings).astype(np.float32, copy=False)
    for a in anns:
        cap = a["caption"]
        query_meta.append(
            {
                "caption_id": int(a["caption_id"]),
                "image_id": int(a["image_id"]),
                "text": cap,
                "text_hash": hashlib.md5(cap.encode()).hexdigest(),
                "modality": "text",
                "concept": "mscoco",
            }
        )

    # Normalize embeddings for cosine search (HubScan will normalize docs if metric=cosine,
    # but query embeddings are loaded via query_embeddings_path and are *not* auto-normalized).
    clean_embeddings_np = _l2_normalize(clean_embeddings_np)
    adv_embeddings_np = _l2_normalize(adv_embeddings_np)
    query_embeddings_np = _l2_normalize(query_embeddings_np)

    # Combine docs: clean first, then poisoned appended
    doc_embeddings = np.vstack([clean_embeddings_np, adv_embeddings_np]).astype(np.float32, copy=False)
    metadata = clean_doc_meta + adv_doc_meta

    # Ground truth: poisoned docs are the last N_adv entries
    num_clean = len(clean_doc_meta)
    num_adv = len(adv_doc_meta)
    hub_positions = list(range(num_clean, num_clean + num_adv))

    ground_truth = {
        "num_total": int(len(metadata)),
        "num_adversarial": int(num_adv),
        "hub_rate": float(num_adv / max(1, len(metadata))),
        "hub_positions": hub_positions,
        "hub_ids": [m["hub_id"] for m in adv_doc_meta],
        "strategies": [m["hub_strategy"] for m in adv_doc_meta],
        "note": "Poisoned documents are adv_hub adversarial images appended after clean MSCOCO val images.",
    }

    dataset_info = {
        "source": "adv_hub_mscoco",
        "model_flag": model_flag,
        "doc_embedding_dim": int(doc_embeddings.shape[1]),
        "query_embedding_dim": int(query_embeddings_np.shape[1]),
        "num_docs_clean": int(num_clean),
        "num_docs_poison": int(num_adv),
        "num_queries": int(query_embeddings_np.shape[0]),
        "coco_val_images_dir": str(Path(coco_val_images_dir).resolve()),
        "coco_val_captions_json": str(Path(coco_val_captions_json).resolve()),
        "adv_x_advs_npy": str(Path(adv_x_advs_npy).resolve()),
        "normalized": True,
        "modality": "image_docs_text_queries",
        "has_adversarial_hubs": True,
    }

    # Write HubScan benchmark artifacts
    np.save(output_dir / "embeddings.npy", doc_embeddings)
    np.save(output_dir / "query_embeddings.npy", query_embeddings_np)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "query_metadata.json", "w") as f:
        json.dump(query_meta, f, indent=2)
    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    return dataset_info


def main() -> None:
    p = argparse.ArgumentParser(description="Export adv_hub MSCOCO poisoned run to HubScan format")
    p.add_argument("--adv-hub-root", type=str, required=True, help="Path to adv_hub repo root")
    p.add_argument(
        "--coco-val-images-dir",
        type=str,
        required=True,
        help="Path to COCO val2017 images directory (val2017/)",
    )
    p.add_argument(
        "--coco-val-captions-json",
        type=str,
        required=True,
        help="Path to COCO captions val json (captions_val2017.json or *_modified.json)",
    )
    p.add_argument(
        "--adv-x-advs",
        type=str,
        required=True,
        help="Path to adv_hub output x_advs.npy (poisoned images tensor)",
    )
    p.add_argument("--output", type=str, required=True, help="Output dataset directory (HubScan format)")
    p.add_argument(
        "--model-flag",
        type=str,
        default="imagebind",
        choices=["imagebind", "openclip", "openclip_rn50", "openclip_vit_b32"],
        help="Model flag (must match paper run)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (e.g. cuda:0 or cpu)",
    )
    p.add_argument("--image-batch-size", type=int, default=64, help="Batch size for image embeddings")
    p.add_argument("--text-batch-size", type=int, default=512, help="Batch size for text embeddings")

    args = p.parse_args()
    info = export_adv_hub_mscoco(
        adv_hub_root=Path(args.adv_hub_root),
        coco_val_images_dir=Path(args.coco_val_images_dir),
        coco_val_captions_json=Path(args.coco_val_captions_json),
        adv_x_advs_npy=Path(args.adv_x_advs),
        output_dir=Path(args.output),
        model_flag=args.model_flag,
        device=args.device,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

