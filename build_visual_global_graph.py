import os
import pickle
import torch
from sklearn.cluster import KMeans

# === Configurations ===
DATASET_DIR = 'datasets/Amazon_grocery_2018'
EMBEDDING_PATH = os.path.join(DATASET_DIR, 'clip_image_embeddings.pkl')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'visual_global_graph')
NUM_CLUSTERS = 300

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load CLIP image embeddings ===
print("📥 Loading CLIP image embeddings...")
with open(EMBEDDING_PATH, 'rb') as f:
    asin_to_embedding = pickle.load(f)

asins = list(asin_to_embedding.keys())
embedding_matrix = torch.stack([asin_to_embedding[asin] for asin in asins]).numpy()

# === Perform KMeans clustering to create concept nodes ===
print(f"🔍 Clustering {len(asins)} items into {NUM_CLUSTERS} concept nodes...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(embedding_matrix)

# === Build graph mappings ===
item_to_concept = {asin: int(label) for asin, label in zip(asins, cluster_labels)}
concept_to_items = {}
for asin, concept_id in item_to_concept.items():
    concept_to_items.setdefault(concept_id, []).append(asin)

concept_vectors = torch.tensor(kmeans.cluster_centers_)  # shape: [NUM_CLUSTERS, 512]

# === Save graph components ===
with open(os.path.join(OUTPUT_DIR, 'item_to_concept.pkl'), 'wb') as f:
    pickle.dump(item_to_concept, f)

with open(os.path.join(OUTPUT_DIR, 'concept_vectors.pkl'), 'wb') as f:
    pickle.dump(concept_vectors, f)

with open(os.path.join(OUTPUT_DIR, 'visual_graph.pkl'), 'wb') as f:
    pickle.dump({
        'item_to_concept': item_to_concept,
        'concept_to_items': concept_to_items
    }, f)

print("✅ Visual global graph saved.")
print(f"📦 Output directory: {OUTPUT_DIR}")
print(f"🔗 Total items: {len(asins)}")
print(f"🧠 Concept nodes: {NUM_CLUSTERS}")
print(f"📈 Average items per concept: {len(asins)/NUM_CLUSTERS:.2f}")
