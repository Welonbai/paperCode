import os
import json
import pickle
from tqdm import tqdm

# === Configurations ===
DATASET_DIR = 'datasets/Amazon_grocery_2018'
META_PATH = os.path.join(DATASET_DIR, 'meta_Grocery_and_Gourmet_Food.json')
ITEMS_WITH_IMAGE_PATH = os.path.join(DATASET_DIR, 'items_with_image.pkl')
OUTPUT_PATH = os.path.join(DATASET_DIR, 'asin_to_imageurl.pkl')

# === Load ASINs with image ===
print(f"Loading ASINs from: {ITEMS_WITH_IMAGE_PATH}")
with open(ITEMS_WITH_IMAGE_PATH, 'rb') as f:
    valid_asins = set(pickle.load(f))

print(f"Total valid ASINs: {len(valid_asins)}")

# === Parse metadata and collect ASIN → first image URL mapping ===
asin_to_imageurl = {}
print(f"Processing metadata from: {META_PATH}")

with open(META_PATH, 'rb') as f:
    for line in tqdm(f, desc="Extracting image URLs"):
        try:
            item = json.loads(line)
            asin = item.get('asin')
            images = item.get('imageURLHighRes')
            if asin in valid_asins and isinstance(images, list) and len(images) > 0:
                asin_to_imageurl[asin] = images[0]  # Only take the first image
        except Exception:
            continue

print(f"✅ Collected {len(asin_to_imageurl)} ASIN-to-URL mappings.")

# === Save output ===
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(asin_to_imageurl, f)

print(f"✅ Saved asin_to_imageurl.pkl to: {OUTPUT_PATH}")

# === Show a preview ===
print("\n🔍 First 5 ASIN → image URL pairs:")
for i, (asin, url) in enumerate(asin_to_imageurl.items()):
    print(f"{asin}: {url}")
    if i >= 4:
        break
