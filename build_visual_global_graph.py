import os
import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans

# === Config ===
DATASET_DIR = 'datasets/Amazon_grocery_2018'
EMBEDDINGS_PATH = os.path.join(DATASET_DIR, 'clip_image_embeddings.pkl')
OUTPUT_GRAPH_PATH = os.path.join(DATASET_DIR, 'image_global_graph.pkl')
K = 300  # Number of clusters (concept nodes)

# === Load CLIP embeddings ===
with open(EMBEDDINGS_PATH, 'rb') as f:
    clip_embeddings = pickle.load(f)

asin_list = list(clip_embeddings.keys())
asin_to_id = {asin: idx for idx, asin in enumerate(asin_list)}
id_to_asin = {idx: asin for asin, idx in asin_to_id.items()}
item_embeddings = np.stack([clip_embeddings[asin].numpy() for asin in asin_list])
item_tensor = torch.tensor(item_embeddings, dtype=torch.float32)

# === KMeans clustering ===
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
labels = kmeans.fit_predict(item_embeddings)
centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

# === Build edge index ===
item_indices = []
concept_indices = []
N = len(asin_list)

for item_id, cluster_id in enumerate(labels):
    item_indices.append(item_id)
    concept_indices.append(N + cluster_id)

# Bidirectional edges (item <-> concept)
edge_index = torch.tensor(
    [item_indices + concept_indices, concept_indices + item_indices],
    dtype=torch.long
)

# === Combine features ===
x = torch.cat([item_tensor, centroids], dim=0)  # Shape: [N+K, 512]

# === Save the graph ===
graph = {
    'edge_index': edge_index,
    'x': x,
    'item_id_to_cluster': {i: int(c) for i, c in enumerate(labels)},
    'asin_to_id': asin_to_id,
    'id_to_asin': id_to_asin,
    'concept_centroids': centroids,
}
with open(OUTPUT_GRAPH_PATH, 'wb') as f:
    pickle.dump(graph, f)

print(f"✅ Saved image-based global graph to: {OUTPUT_GRAPH_PATH}")
print(f"Total items: {N}, Total concept nodes: {K}, Total nodes: {x.shape[0]}")
