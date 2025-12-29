# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate toy test data for HubScan."""

import numpy as np
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_docs = 1000
embedding_dim = 128
num_hubs = 5  # Number of adversarial hubs to plant

# Generate normal document embeddings
print(f"Generating {num_docs} document embeddings...")
doc_embeddings = np.random.randn(num_docs, embedding_dim).astype(np.float32)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

# Plant adversarial hubs: create vectors that are close to many query vectors
print(f"Planting {num_hubs} adversarial hubs...")
hub_indices = np.random.choice(num_docs, num_hubs, replace=False)

# Create hub vectors that are close to cluster centroids
for hub_idx in hub_indices:
    # Make this vector close to multiple random document clusters
    cluster_centers = np.random.randn(10, embedding_dim).astype(np.float32)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    # Hub is average of cluster centers (normalized)
    hub_vector = np.mean(cluster_centers, axis=0)
    hub_vector = hub_vector / np.linalg.norm(hub_vector)
    doc_embeddings[hub_idx] = hub_vector

# Generate metadata
print("Generating metadata...")
metadata = {
    "doc_id": [f"doc_{i:04d}" for i in range(num_docs)],
    "source": [f"source_{i % 10}" for i in range(num_docs)],
    "text": [f"Document {i} content. This is sample text for testing hubness detection." for i in range(num_docs)],
    "text_hash": [hash(f"Document {i} content. This is sample text for testing hubness detection.") % (10**10) for i in range(num_docs)],
}

# Mark hubs in metadata
for hub_idx in hub_indices:
    metadata["text"][hub_idx] = f"[ADVERSARIAL HUB] Document {hub_idx} - This is a planted adversarial hub."

# Save embeddings
embeddings_path = Path("examples/toy_embeddings.npy")
embeddings_path.parent.mkdir(parents=True, exist_ok=True)
np.save(embeddings_path, doc_embeddings)
print(f"Saved embeddings to {embeddings_path}")

# Save metadata
metadata_path = Path("examples/toy_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {metadata_path}")

print(f"\nPlanted adversarial hubs at indices: {hub_indices.tolist()}")
print("These should be detected by the scanner!")

