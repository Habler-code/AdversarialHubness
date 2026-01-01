#!/usr/bin/env python3
# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Create multimodal RAG benchmark with separate embedding spaces.

This benchmark uses separate embedding models for each modality:
- ResNet50 for images (preserves visual semantics)
- Sentence-transformers for text (preserves textual semantics)

Using separate embedding spaces preserves semantic margins,
improving hubness detection accuracy for multimodal systems.
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings

# Check for required dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not installed. Install with: pip install datasets")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")

try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not installed. Install with: pip install scikit-learn")

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not installed. Install with: pip install torch torchvision")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")


def load_multimodal_dataset(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a multimodal dataset from HuggingFace."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    if max_samples is None:
        max_samples = 500
    
    print("Loading multimodal dataset from HuggingFace...")
    
    dataset = None
    dataset_name = None
    
    # Try food101 (good for testing - clear categories, reasonable size)
    try:
        print("Trying food101...")
        # Load full dataset first, then sample evenly across classes
        full_dataset = load_dataset("food101", split="train", trust_remote_code=True)
        # Shuffle to get diverse samples (dataset is ordered by class)
        full_dataset = full_dataset.shuffle(seed=42)
        # Take first max_samples after shuffle
        dataset = full_dataset.select(range(min(max_samples, len(full_dataset))))
        dataset_name = "food101"
        print(f"Loaded {dataset_name}: {len(dataset)} samples (shuffled)")
    except Exception as e:
        print(f"Could not load food101: {e}")
    
    # Fallback to cifar10
    if dataset is None:
        try:
            print("Trying cifar10...")
            dataset = load_dataset("uoft-cs/cifar10", split=f"train[:{max_samples}]", trust_remote_code=True)
            dataset_name = "cifar10"
            print(f"Loaded {dataset_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"Could not load cifar10: {e}")
    
    if dataset is None:
        print("ERROR: Could not load any dataset")
        return []
    
    samples = []
    for i, item in enumerate(tqdm(dataset, desc="Processing samples")):
        if i >= max_samples:
            break
        
        image = item.get("image") or item.get("img")
        if image is None:
            continue
        
        # Get label/category
        label = item.get("label", i)
        if isinstance(label, int):
            if hasattr(dataset, 'features') and 'label' in dataset.features:
                try:
                    label_name = dataset.features['label'].int2str(label)
                except Exception:
                    label_name = f"class_{label}"
            else:
                label_name = f"class_{label}"
        else:
            label_name = str(label)
        
        samples.append({
            "id": f"{dataset_name}_{i:05d}",
            "image": image,
            "label": label,
            "label_name": label_name,
            "caption": f"An image of {label_name.replace('_', ' ')}",
        })
    
    print(f"Processed {len(samples)} samples from {dataset_name}")
    return samples


def create_text_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Create text embeddings using sentence-transformers."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers required")
    
    print(f"Loading text model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Creating text embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings.astype(np.float32)


def create_image_embeddings_resnet(
    images: List[Any],
) -> np.ndarray:
    """
    Create image embeddings using ResNet50 (pretrained on ImageNet).
    
    This is a simple, widely-available option that:
    - Does NOT create a shared image-text space
    - Preserves visual hierarchy
    - Has clear semantic margins between classes
    """
    if not HAS_TORCH:
        raise ImportError("torch required. Install with: pip install torch torchvision")
    
    from torchvision import models
    
    print("Loading ResNet50 for image embeddings...")
    
    # Load pretrained ResNet50, remove final classification layer
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final FC layer to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Image preprocessing
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    embeddings = []
    print("Creating image embeddings...")
    
    with torch.no_grad():
        for img in tqdm(images, desc="Embedding images"):
            if isinstance(img, str):
                img = Image.open(img)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Preprocess and embed
            img_tensor = preprocess(img).unsqueeze(0)
            emb = model(img_tensor).squeeze().numpy()
            embeddings.append(emb)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    print(f"Created {len(embeddings)} image embeddings, dim={embeddings.shape[1]}")
    return embeddings


def create_multimodal_benchmark(
    output_dir: Path,
    max_samples: Optional[int] = None,
    text_model: str = "all-MiniLM-L6-v2",
    num_concepts: int = 10,
) -> Dict[str, Any]:
    """
    Create multimodal benchmark with SEPARATE embedding spaces.
    
    Architecture:
    - Images: ResNet50 embeddings (2048-dim)
    - Text: Sentence-transformers (384-dim)
    - Each modality is stored and queried SEPARATELY
    
    This is the correct approach for secure multimodal RAG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Creating Multimodal Benchmark (SEPARATE SPACES)")
    print("=" * 60)
    print("\nNOTE: This benchmark uses separate embedding spaces:")
    print("  - Images: ResNet50 (2048-dim)")
    print("  - Text: Sentence-transformers (384-dim)")
    print()
    
    # Check dependencies
    if not HAS_DATASETS:
        print("ERROR: datasets library not installed")
        return None
    if not HAS_SENTENCE_TRANSFORMERS:
        print("ERROR: sentence-transformers not installed")
        return None
    if not HAS_TORCH:
        print("ERROR: torch not installed")
        return None
    
    # Load dataset
    samples = load_multimodal_dataset(max_samples=max_samples)
    if not samples:
        print("ERROR: No samples loaded")
        return None
    
    # Create image embeddings (ResNet50)
    images = [s["image"] for s in samples]
    image_embeddings = create_image_embeddings_resnet(images)
    
    # Create text embeddings (sentence-transformers)
    texts = [s["caption"] for s in samples]
    text_embeddings = create_text_embeddings(texts, model_name=text_model)
    
    # Create metadata for each document
    # We'll have BOTH image and text documents in the index
    metadata = []
    
    # Image documents
    for i, sample in enumerate(samples):
        metadata.append({
            "doc_id": f"{sample['id']}_img",
            "type": "image",
            "modality": "image",
            "label": sample["label_name"],
            "caption": sample["caption"],
            "text": f"[IMAGE] {sample['caption']}",
            "text_hash": hashlib.md5(f"img_{sample['id']}".encode()).hexdigest(),
            "source_id": sample["id"],
            "is_adversarial": False,
            "embedding_idx": i,  # Index into image_embeddings
        })
    
    # Text documents
    for i, sample in enumerate(samples):
        metadata.append({
            "doc_id": f"{sample['id']}_txt",
            "type": "text",
            "modality": "text",
            "label": sample["label_name"],
            "caption": sample["caption"],
            "text": sample["caption"],
            "text_hash": hashlib.md5(sample["caption"].encode()).hexdigest(),
            "source_id": sample["id"],
            "is_adversarial": False,
            "embedding_idx": i,  # Index into text_embeddings
        })
    
    # Gold standard: Create SEPARATE indexes for each modality
    # This enables parallel retrieval + late fusion architecture
    
    # Build separate FAISS indexes
    if HAS_FAISS:
        # Text index (using inner product for cosine similarity with normalized vectors)
        text_faiss_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        text_faiss_index.add(text_embeddings.astype(np.float32))
        
        # Image index (using inner product for cosine similarity with normalized vectors)
        image_faiss_index = faiss.IndexFlatIP(image_embeddings.shape[1])
        image_faiss_index.add(image_embeddings.astype(np.float32))
        
        # Save separate indexes
        faiss.write_index(text_faiss_index, str(output_dir / "text_index.index"))
        faiss.write_index(image_faiss_index, str(output_dir / "image_index.index"))
        print(f"Saved separate indexes: text={text_embeddings.shape}, image={image_embeddings.shape}")
    else:
        print("Warning: FAISS not available, skipping separate index creation")
    
    # For backward compatibility, also create combined embeddings (padded)
    # This allows existing single-index workflows to still work
    max_dim = max(image_embeddings.shape[1], text_embeddings.shape[1])
    
    # Pad embeddings to same dimension
    image_padded = np.zeros((len(image_embeddings), max_dim), dtype=np.float32)
    image_padded[:, :image_embeddings.shape[1]] = image_embeddings
    
    text_padded = np.zeros((len(text_embeddings), max_dim), dtype=np.float32)
    text_padded[:, :text_embeddings.shape[1]] = text_embeddings
    
    # Concatenate: images first, then texts
    all_embeddings = np.vstack([image_padded, text_padded])
    
    # Re-normalize after padding
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / (norms + 1e-8)
    
    # Assign concepts based on labels
    unique_labels = list(set(s["label_name"] for s in samples))
    label_to_concept = {label: i for i, label in enumerate(unique_labels)}
    
    for meta in metadata:
        label = meta["label"]
        meta["concept"] = label
        meta["concept_id"] = label_to_concept.get(label, 0)
    
    # Create query texts
    query_texts = [meta["text"] for meta in metadata]
    
    # Save everything
    print("\nSaving dataset...")
    
    np.save(output_dir / "embeddings.npy", all_embeddings)
    print(f"Saved embeddings: {all_embeddings.shape}")
    
    # Also save separate embeddings for advanced use
    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "text_embeddings.npy", text_embeddings)
    print(f"Saved separate embeddings: image={image_embeddings.shape}, text={text_embeddings.shape}")
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(output_dir / "query_texts.json", "w") as f:
        json.dump(query_texts, f, indent=2)
    
    # Statistics
    modality_counts = {"image": len(samples), "text": len(samples)}
    concept_counts = {}
    for meta in metadata:
        c = meta.get("concept", "unknown")
        concept_counts[c] = concept_counts.get(c, 0) + 1
    
    dataset_info = {
        "name": "multimodal_separate_spaces",
        "description": "Multimodal benchmark with separate embedding spaces per modality (gold standard architecture)",
        "note": "Images use ResNet50 (2048d), text uses sentence-transformers (384d). Supports parallel retrieval + late fusion.",
        "num_samples": len(samples),
        "num_documents": len(metadata),
        "embedding_dim": all_embeddings.shape[1],
        "image_embedding_dim": image_embeddings.shape[1],
        "text_embedding_dim": text_embeddings.shape[1],
        "image_model": "ResNet50",
        "text_model": text_model,
        "has_concepts": True,
        "num_concepts": len(unique_labels),
        "concept_counts": concept_counts,
        "has_modalities": True,
        "modalities": ["image", "text"],
        "modality_counts": modality_counts,
        "gold_standard": True,
        "separate_indexes": True,
        "text_index_path": "text_index.index",
        "image_index_path": "image_index.index",
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Multimodal Benchmark Created Successfully!")
    print("=" * 60)
    print(f"   Location: {output_dir}")
    print(f"   Samples: {len(samples)}")
    print(f"   Documents: {len(metadata)}")
    print(f"   Image embeddings: {image_embeddings.shape}")
    print(f"   Text embeddings: {text_embeddings.shape}")
    print(f"   Combined (padded, for backward compat): {all_embeddings.shape}")
    print(f"   Concepts: {len(unique_labels)}")
    print()
    print("Gold Standard Architecture:")
    print("  - Separate indexes: text_index.index, image_index.index")
    print("  - Supports parallel retrieval + late fusion")
    print("  - Combined embeddings.npy provided for backward compatibility")
    
    return dataset_info


def main():
    parser = argparse.ArgumentParser(
        description="Create multimodal benchmark with separate embedding spaces"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/data/multimodal/",
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Maximum number of samples (default: 300)",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Text embedding model",
    )
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=10,
        help="Number of concept clusters",
    )
    
    args = parser.parse_args()
    
    create_multimodal_benchmark(
        output_dir=Path(args.output),
        max_samples=args.max_samples,
        text_model=args.text_model,
        num_concepts=args.num_concepts,
    )


if __name__ == "__main__":
    main()
