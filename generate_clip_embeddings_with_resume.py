import os
import pickle
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import clip
import requests
from io import BytesIO
import time

# === Configurations ===
DATASET_DIR = 'datasets/Amazon_grocery_2018'
ASIN2URL_PATH = os.path.join(DATASET_DIR, 'asin_to_imageurl.pkl')
FINAL_OUTPUT_PATH = os.path.join(DATASET_DIR, 'clip_image_embeddings.pkl')
PARTIAL_SAVE_PATH = os.path.join(DATASET_DIR, 'clip_image_embeddings_partial.pkl')
FAILED_LOG_PATH = os.path.join(DATASET_DIR, 'failed_items_log.pkl')
SAVE_INTERVAL = 1000  # Save every 1000 items
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load CLIP Model ===
print("🚀 Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# === Load ASIN → URL mapping ===
print(f"📂 Loading ASIN to URL mapping from: {ASIN2URL_PATH}")
with open(ASIN2URL_PATH, 'rb') as f:
    asin_to_url = pickle.load(f)

# === Load previous progress if exists ===
clip_embeddings = {}
failed_items = []
if os.path.exists(PARTIAL_SAVE_PATH):
    print(f"🔄 Resuming from partial save: {PARTIAL_SAVE_PATH}")
    with open(PARTIAL_SAVE_PATH, 'rb') as f:
        clip_embeddings = pickle.load(f)

    print(f"✅ Already processed: {len(clip_embeddings)} items")

# === Processing ===
start_time = time.time()
processed_count = 0

for asin, url in tqdm(asin_to_url.items(), desc="🧠 Generating CLIP embeddings"):
    if asin in clip_embeddings:
        continue  # Skip already processed

    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(image_input).cpu().squeeze(0)
        clip_embeddings[asin] = image_features
        processed_count += 1

        # Periodic saving
        if processed_count % SAVE_INTERVAL == 0:
            with open(PARTIAL_SAVE_PATH, 'wb') as f:
                pickle.dump(clip_embeddings, f)
            print(f"💾 Saved partial at {processed_count} items")

    except Exception as e:
        failed_items.append((asin, str(e)))

# === Final Save ===
with open(FINAL_OUTPUT_PATH, 'wb') as f:
    pickle.dump(clip_embeddings, f)

# Save failed log
with open(FAILED_LOG_PATH, 'wb') as f:
    pickle.dump(failed_items, f)

end_time = time.time()

# === Summary ===
print(f"\n✅ Final save complete: {len(clip_embeddings)} embeddings")
print(f"⚠️ Total failed items: {len(failed_items)} — see {FAILED_LOG_PATH}")
if failed_items:
    print("🧨 Example failure:", failed_items[0])
print(f"🕒 Total time: {(end_time - start_time)/60:.2f} minutes")
