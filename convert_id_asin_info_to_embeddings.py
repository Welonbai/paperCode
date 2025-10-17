#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate embeddings for image/title/category items and optional category-node artifacts.
Outputs are written to datasets/<dataset>/embeddings/ by default.
"""

import argparse
import ast
import io
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

DEFAULT_IMAGE_MODEL = "ViT-B/32"
DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"
FALLBACK_TEXT_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_mapping(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        data = pickle.load(open(path, "rb"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    expected = {"id", "asin", "url", "title", "category"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Mapping file missing columns: {missing}")
    df["id"] = df["id"].astype(int)
    return df.sort_values("id")


def parse_category(cell):
    if isinstance(cell, str):
        try:
            obj = ast.literal_eval(cell)
        except Exception:
            obj = [p.strip() for p in cell.replace(">", ",").split(",") if p.strip()]
    else:
        obj = cell
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], list):
        obj = obj[0]
    if not isinstance(obj, list):
        obj = [str(obj)]
    return [str(x) for x in obj]


def category_to_text(tokens: List[str], drop_root: str) -> str:
    if tokens and tokens[0] == drop_root:
        tokens = tokens[1:]
    return " > ".join(tokens) if tokens else ""


def extract_category_tokens(raw, drop_root: str) -> List[str]:
    tokens = parse_category(raw)
    if tokens and tokens[0] == drop_root:
        tokens = tokens[1:]
    seen = set()
    result = []
    for tok in tokens:
        tok = str(tok).strip()
        if not tok or tok in seen:
            continue
        seen.add(tok)
        result.append(tok)
    return result


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def padded_matrix(ids: List[int], emb: np.ndarray) -> np.ndarray:
    mat = np.zeros((max(ids) + 1, emb.shape[1]), dtype=emb.dtype)
    mat[np.asarray(ids, dtype=int), :] = emb
    mat[0] = 0.0
    return mat


def build_clip(model_name: str):
    import clip
    model, _ = clip.load(model_name, device=DEVICE)
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, preprocess


def fetch_image(url: str, timeout: float = 10.0) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (embedding-fetcher)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def encode_images(df: pd.DataFrame, model_name: str, l2: bool) -> tuple[List[int], np.ndarray]:
    clip_model, preprocess = build_clip(model_name)
    clip_model.eval()
    ids, vecs = [], []
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
        default_dim = clip_model.encode_image(dummy).shape[-1]
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="image"):
        item_id, url = int(row.id), str(row.url)
        try:
            img = fetch_image(url)
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            vec = clip_model.encode_image(tensor).squeeze(0).detach().float().cpu().numpy()
        except Exception as exc:
            print(f"[warn] image embedding failed for item {item_id}: {exc}. Using zeros.")
            vec = np.zeros(default_dim, dtype=np.float32)
        ids.append(item_id)
        vecs.append(vec)
    emb = np.stack(vecs, axis=0).astype(np.float32)
    return ids, l2_normalize(emb) if l2 else emb


def load_text_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(model_name, device=DEVICE)
    except Exception as exc:
        print("[warn] sentence_transformers could not be loaded:", exc)
        print("[warn] Falling back to zero text embeddings. Install `sentence-transformers` for real features.")
        return None


def encode_texts(texts: List[str], model, l2: bool, fallback_dim: int = FALLBACK_TEXT_DIM) -> np.ndarray:
    if model is None:
        arr = np.zeros((len(texts), fallback_dim), dtype=np.float32)
        return arr
    with torch.no_grad():
        arr = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False)
    arr = arr.astype(np.float32)
    return l2_normalize(arr) if l2 else arr


def build_category_nodes(df: pd.DataFrame, drop_root: str, text_model,
                         max_item_id: int, l2: bool, out_dir: Path) -> None:
    category_vocab: Dict[str, int] = {}
    category_freq: Dict[int, int] = {}
    item_category_links = [[] for _ in range(max_item_id + 1)]
    for item_id, raw in zip(df["id"], df["category"]):
        cat_tokens = extract_category_tokens(raw, drop_root)
        mapped = []
        for tok in cat_tokens:
            if tok not in category_vocab:
                category_vocab[tok] = len(category_vocab) + 1
            cid = category_vocab[tok]
            category_freq[cid] = category_freq.get(cid, 0) + 1
            mapped.append(cid)
        item_category_links[int(item_id)] = mapped

    if not category_vocab:
        print("[warn] no category nodes generated")
        return

    ordered = sorted(category_vocab.items(), key=lambda kv: kv[1])
    texts = [""] * (len(category_vocab) + 1)
    for text, cid in ordered:
        texts[cid] = text
    node_emb = encode_texts(texts[1:], text_model, l2=l2)
    node_mat = np.zeros((len(category_vocab) + 1, node_emb.shape[1]), dtype=np.float32)
    node_mat[1:, :] = node_emb
    torch.save(torch.from_numpy(node_mat), out_dir / "category_nodes_matrix.pt")
    np.save(out_dir / "category_nodes_matrix.npy", node_mat)
    save_pickle(
        [{"id": cid, "text": text, "freq": category_freq.get(cid, 0)} for text, cid in ordered],
        out_dir / "category_nodes.pkl",
    )
    save_pickle({"item_to_category": item_category_links}, out_dir / "item_category_links.pkl")


def main():
    parser = argparse.ArgumentParser(description="Generate modality embeddings and optional category-node assets.")
    parser.add_argument("--dataset", required=True, help="Dataset name (under datasets/).")
    parser.add_argument("--mapping", default="", help="Optional path to id_asin_info.{pkl,csv}.")
    parser.add_argument("--output-root", default="", help="Override output root (default datasets/<dataset>).")
    parser.add_argument("--drop-root-category", default="", help="Root category label to drop (e.g. 'Cell Phones & Accessories').")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL, help="CLIP model for image embeddings.")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL, help="SentenceTransformer for text embeddings.")
    parser.add_argument("--no-image", action="store_true", help="Skip image embeddings.")
    parser.add_argument("--no-title", action="store_true", help="Skip title embeddings.")
    parser.add_argument("--no-category", action="store_true", help="Skip category embeddings.")
    parser.add_argument("--category-nodes", action="store_true", help="Generate category-node embeddings and item->category links.")
    parser.add_argument("--no-l2", dest="l2_norm", action="store_false", help="Disable L2 normalization (enabled by default).")
    parser.set_defaults(l2_norm=True)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else Path("datasets") / args.dataset
    embed_dir = output_root / "embeddings"
    ensure_dir(embed_dir)

    mapping_path = args.mapping or str(output_root / "mapping" / "id_asin_info.pkl")
    df = load_mapping(mapping_path)
    max_item_id = int(df["id"].max())
    drop_root = args.drop_root_category or ""

    text_model = None

    if not args.no_image:
        ids, emb = encode_images(df, args.image_model, l2=args.l2_norm)
        save_pickle(
            {
                "ids": ids,
                "embeddings": emb,
                "id_to_index": {i: k for k, i in enumerate(ids)},
                "meta": {"modality": "image", "backend": "CLIP", "model": args.image_model, "shape": list(emb.shape)},
            },
            embed_dir / "image_embeddings.pkl",
        )
        mat = padded_matrix(ids, emb)
        torch.save(torch.from_numpy(mat), embed_dir / "image_matrix.pt")
        np.save(embed_dir / "image_matrix.npy", mat)

    if (not args.no_title) or (not args.no_category) or args.category_nodes:
        text_model = load_text_model(args.text_model)

    if not args.no_category:
        raw_categories = df["category"].tolist()
        if text_model is None:
            text_model = load_text_model(args.text_model)
        cat_texts = [category_to_text(parse_category(cat), drop_root) for cat in raw_categories]
        cat_emb = encode_texts(cat_texts, text_model, l2=args.l2_norm)
        ids = df["id"].tolist()
        save_pickle(
            {
                "ids": ids,
                "embeddings": cat_emb,
                "id_to_index": {i: k for k, i in enumerate(ids)},
                "meta": {
                    "modality": "category",
                    "backend": "SentenceTransformer",
                    "model": args.text_model,
                    "shape": list(cat_emb.shape),
                    "rule": f"drop root '{drop_root}' if present",
                },
            },
            embed_dir / "category_embeddings.pkl",
        )
        mat = padded_matrix(ids, cat_emb)
        torch.save(torch.from_numpy(mat), embed_dir / "category_matrix.pt")
        np.save(embed_dir / "category_matrix.npy", mat)

    if args.category_nodes:
        if text_model is None:
            text_model = load_text_model(args.text_model)
        build_category_nodes(df, drop_root, text_model, max_item_id, args.l2_norm, embed_dir)

    if not args.no_title:
        if text_model is None:
            text_model = load_text_model(args.text_model)
        titles = [str(t).strip() if pd.notnull(t) else "" for t in df["title"].tolist()]
        title_emb = encode_texts(titles, text_model, l2=args.l2_norm)
        ids = df["id"].tolist()
        save_pickle(
            {
                "ids": ids,
                "embeddings": title_emb,
                "id_to_index": {i: k for k, i in enumerate(ids)},
                "meta": {"modality": "title", "backend": "SentenceTransformer", "model": args.text_model, "shape": list(title_emb.shape)},
            },
            embed_dir / "title_embeddings.pkl",
        )
        mat = padded_matrix(ids, title_emb)
        torch.save(torch.from_numpy(mat), embed_dir / "title_matrix.pt")
        np.save(embed_dir / "title_matrix.npy", mat)

    print("[done] embeddings written to", embed_dir)


if __name__ == "__main__":
    main()
