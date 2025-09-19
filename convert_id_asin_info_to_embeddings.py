#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-shot converter for side-information embeddings (image / category / title), NO CACHE.

- Reads mapping from:
    datasets/Amazon_cellPhone_2018/mapping/id_asin_info.pkl
- Writes outputs under:
    datasets/Amazon_cellPhone_2018/embeddings/
  Files:
    image_embeddings.pkl, category_embeddings.pkl, title_embeddings.pkl
    image_matrix.pt/.npy, category_matrix.pt/.npy, title_matrix.pt/.npy

Notes
-----
- Image embeddings: CLIP ViT-B/32 → 512-D (L2-normalized).
- Category/Title: SentenceTransformer all-MiniLM-L6-v2 → 384-D (L2-normalized).
- Category processing: drop the first token if it equals "Cell Phones & Accessories",
  then join the rest with " > " (handles variable length).
- Padded matrices are shaped [max_id+1, D] with row 0 = zeros (padding).
"""

import os, io, ast, pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import requests
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# =========================
# PATHS & SETTINGS
# =========================
MAPPING_PATH = "datasets/Amazon_cellPhone_2018/mapping/id_asin_info.pkl"
OUTPUT_ROOT = "datasets/Amazon_cellPhone_2018"
OUT_DIR = os.path.join(OUTPUT_ROOT, "embeddings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"

DROP_ROOT_CATEGORY = "Cell Phones & Accessories"  # drop if it's the first element
L2_NORMALIZE = True  # L2 normalize each embedding row


# ================
# Helper functions
# ================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_pkl(o, p):
    ensure_dir(os.path.dirname(p))
    with open(p, "wb") as f:
        pickle.dump(o, f)

def load_mapping(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        data = pickle.load(open(path, "rb"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    for c in ["id", "asin", "url", "title", "category"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["id"] = df["id"].astype(int)
    return df.sort_values("id")

def parse_category(cell):
    """Return a flat list of tokens, handling strings and nested lists."""
    if isinstance(cell, str):
        try:
            obj = ast.literal_eval(cell)
        except Exception:
            # fallback: split heuristics
            parts = [p.strip() for p in cell.replace(">", ",").split(",") if p.strip()]
            obj = parts
    else:
        obj = cell
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], list):
        obj = obj[0]
    if not isinstance(obj, list):
        obj = [str(obj)]
    return [str(x) for x in obj]

def category_to_text(tokens, drop_root=DROP_ROOT_CATEGORY):
    """Drop the first token if it equals drop_root, then join the rest with ' > '."""
    if len(tokens) > 0 and tokens[0] == drop_root:
        tokens = tokens[1:]
    if len(tokens) == 0:
        return ""  # fallback
    return " > ".join(tokens)

def l2n(arr: np.ndarray) -> np.ndarray:
    if not L2_NORMALIZE:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def padded_matrix(ids, emb):
    """Build [max_id+1, D] with row 0 = zeros; rows at ids filled with emb."""
    D = emb.shape[1]; M = max(ids)
    mat = np.zeros((M + 1, D), dtype=emb.dtype)
    mat[np.array(ids, int), :] = emb
    mat[0, :] = 0
    return mat


# =========================
# Image embedding (CLIP)
# =========================
def build_clip(device=DEVICE, model_name=CLIP_MODEL_NAME):
    import clip  # lazy import
    model, _ = clip.load(model_name, device=device)
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess

def fetch_image_no_cache(url: str, timeout=10.0) -> Image.Image:
    """Download image and return PIL.Image without saving to disk."""
    headers = {"User-Agent": "Mozilla/5.0 (embedding-fetcher)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def encode_images(df: pd.DataFrame) -> tuple[list[int], np.ndarray]:
    clip_model, clip_pre = build_clip()
    clip_model.eval()

    ids, vecs = [], []
    # infer dim for fallback; ViT-B/32 → 512
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
        default_dim = clip_model.encode_image(dummy).shape[-1]

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Image"):
        _id, url = int(row.id), str(row.url)
        try:
            img = fetch_image_no_cache(url)
            with torch.no_grad():
                t = clip_pre(img).unsqueeze(0).to(DEVICE)
                v = clip_model.encode_image(t).squeeze(0).float().cpu().numpy()
        except Exception as e:
            print(f"⚠️ image failed for id={_id}: {e} → using zeros")
            v = np.zeros((default_dim,), dtype=np.float32)
        ids.append(_id)
        vecs.append(v)

    emb = l2n(np.stack(vecs, axis=0)).astype(np.float32)
    return ids, emb


# ===================================
# Text embeddings (category & title)
# ===================================
def build_text_model(device=DEVICE, model_name=TEXT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)

def encode_texts(texts: list[str], model) -> np.ndarray:
    with torch.no_grad():
        arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False)
    return l2n(arr.astype(np.float32))


# =====
# Run
# =====
def main():
    ensure_dir(OUT_DIR)
    df = load_mapping(MAPPING_PATH)

    # ---------- IMAGE ----------
    img_ids, img_emb = encode_images(df)
    save_pkl(
        {
            "ids": img_ids,
            "embeddings": img_emb,
            "id_to_index": {i: k for k, i in enumerate(img_ids)},
            "meta": {"modality": "image", "backend": "CLIP", "model": CLIP_MODEL_NAME, "shape": list(img_emb.shape)},
        },
        os.path.join(OUT_DIR, "image_embeddings.pkl"),
    )
    mat = padded_matrix(img_ids, img_emb)
    torch.save(torch.from_numpy(mat), os.path.join(OUT_DIR, "image_matrix.pt"))
    np.save(os.path.join(OUT_DIR, "image_matrix.npy"), mat)

    # ---------- TEXT MODEL (shared) ----------
    txt_model = build_text_model()

    # ---------- CATEGORY ----------
    cat_ids = df["id"].tolist()
    cat_texts = [category_to_text(parse_category(c)) for c in df["category"].tolist()]
    cat_emb = encode_texts(cat_texts, txt_model)
    save_pkl(
        {
            "ids": cat_ids,
            "embeddings": cat_emb,
            "id_to_index": {i: k for k, i in enumerate(cat_ids)},
            "meta": {
                "modality": "category",
                "backend": "SentenceTransformer",
                "model": TEXT_MODEL_NAME,
                "shape": list(cat_emb.shape),
                "rule": f"drop first token if '{DROP_ROOT_CATEGORY}'; join rest with ' > '",
            },
        },
        os.path.join(OUT_DIR, "category_embeddings.pkl"),
    )
    mat = padded_matrix(cat_ids, cat_emb)
    torch.save(torch.from_numpy(mat), os.path.join(OUT_DIR, "category_matrix.pt"))
    np.save(os.path.join(OUT_DIR, "category_matrix.npy"), mat)

    # ---------- TITLE ----------
    ttl_ids = df["id"].tolist()
    titles = [str(t).strip() if t is not None else "" for t in df["title"].tolist()]
    ttl_emb = encode_texts(titles, txt_model)
    save_pkl(
        {
            "ids": ttl_ids,
            "embeddings": ttl_emb,
            "id_to_index": {i: k for k, i in enumerate(ttl_ids)},
            "meta": {"modality": "title", "backend": "SentenceTransformer", "model": TEXT_MODEL_NAME, "shape": list(ttl_emb.shape)},
        },
        os.path.join(OUT_DIR, "title_embeddings.pkl"),
    )
    mat = padded_matrix(ttl_ids, ttl_emb)
    torch.save(torch.from_numpy(mat), os.path.join(OUT_DIR, "title_matrix.pt"))
    np.save(os.path.join(OUT_DIR, "title_matrix.npy"), mat)

    print("✅ Done. Wrote per-modality embeddings + padded matrices to:", OUT_DIR)


if __name__ == "__main__":
    main()
