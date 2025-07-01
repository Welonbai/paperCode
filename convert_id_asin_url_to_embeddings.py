import os
import pickle
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# === Configuration ===
MAPPING_PKL_PATH = "datasets/Amazon_grocery_2018/mapping/id_asin_url.pkl"
OUTPUT_PKL_PATH = "datasets/Amazon_grocery_2018/clip_image_embeddings.pkl"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load ID-ASIN-URL mapping ===
with open(MAPPING_PKL_PATH, "rb") as f:
    id_asin_url = pickle.load(f)

# === Load CLIP model ===
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# === Preprocessing transform ===
transform = Compose([
    Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    CenterCrop(IMAGE_SIZE),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

# === Generate CLIP embeddings ===
clip_embeddings = {}
for entry in tqdm(id_asin_url, desc="Generating CLIP embeddings"):
    asin = entry['asin']
    url = entry['url']

    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input).squeeze().cpu()
        
        clip_embeddings[asin] = image_features
    except Exception as e:
        print(f"⚠️ Failed to process {asin}: {e}")

# === Save the final embeddings ===
with open(OUTPUT_PKL_PATH, "wb") as f:
    pickle.dump(clip_embeddings, f)

print(f"✅ Saved CLIP embeddings for {len(clip_embeddings)} items to: {OUTPUT_PKL_PATH}")
