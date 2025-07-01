import os
import pickle
import random
import json

# ========== CONFIG ==========
DATASET_DIR = 'datasets/Amazon_grocery_2018'
GRAPH_PATH = os.path.join(DATASET_DIR, 'image_global_graph.pkl')
MAPPING_PATH = os.path.join(DATASET_DIR, 'mapping', 'id_asin_url.pkl')
METADATA_PATH = os.path.join(DATASET_DIR, 'meta_Grocery_and_Gourmet_Food.json')  # ← or actual meta file

N_CONCEPTS_TO_SAMPLE = 5
N_ITEMS_PER_CONCEPT = 10

# ========== LOAD ==========
print("📦 Loading graph and mappings...")
with open(GRAPH_PATH, 'rb') as f:
    graph = pickle.load(f)

with open(MAPPING_PATH, 'rb') as f:
    id_asin_url = pickle.load(f)

asin_to_info = {x['asin']: x for x in id_asin_url}
item_id_to_asin = graph['id_to_asin']
item_id_to_cluster = graph['item_id_to_cluster']

# Optional: load metadata
asin_to_title_cat = {}
if os.path.exists(METADATA_PATH):
    print("🔍 Loading metadata for title/category...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                asin = obj.get('asin')
                title = obj.get('title', '[NO TITLE]')
                category = obj.get('category', ['[NO CATEGORY]'])
                if isinstance(category, list):
                    category = category[-1]
                asin_to_title_cat[asin] = (title, category)
            except:
                continue

# ========== RANDOM CONCEPT NODE INSPECTION ==========
K = graph['concept_centroids'].shape[0]
sampled_clusters = random.sample(range(K), N_CONCEPTS_TO_SAMPLE)
print(f"🎯 Sampled concept nodes: {sampled_clusters}")

results = []

for cid in sampled_clusters:
    matched_items = [iid for iid, cluster in item_id_to_cluster.items() if cluster == cid]
    matched_items = matched_items[:N_ITEMS_PER_CONCEPT]

    items_info = []
    for iid in matched_items:
        asin = item_id_to_asin[iid]
        url = asin_to_info.get(asin, {}).get("url", "")
        title, category = asin_to_title_cat.get(asin, ('[NO TITLE]', '[NO CATEGORY]'))

        items_info.append({
            'item_id': iid,
            'asin': asin,
            'title': title,
            'category': category,
            'image_url': url
        })

    results.append({
        'concept_id': cid,
        'num_items': len(matched_items),
        'items': items_info
    })

# ========== SAVE TO FILE ==========
os.makedirs("inspect_output", exist_ok=True)
OUTPUT_PATH = os.path.join("inspect_output", "cluster_case_study_no_text.json")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved visual-only cluster case study to: {OUTPUT_PATH}")
