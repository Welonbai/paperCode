#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an item–item "category_shared" graph from category token co-membership.

This version adds robust coverage safeguards:
  * Embedding-KNN augmentation for singleton/skinny categories during pre-topk.
  * Embedding-KNN backfill when candidate lists are empty.
  * Tunable diversity/asymmetry knobs.

Inputs (produced by convert_id_asin_info_to_embeddings.py):
  - embeddings/category_nodes.pkl              (list of {"id","text","freq"})
  - embeddings/item_category_links.pkl         ({"item_to_category": List[List[int]]})
  - embeddings/category_matrix.npy             (token embeddings)

Output:
  datasets/<dataset>/global_graph/global_graph_category_shared.pkl
"""

import argparse
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# ------------------------------
# Utilities
# ------------------------------
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def jaccard_distance(tokens_a: List[int], tokens_b: List[int],
                     emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if set_a or set_b:
        union = set_a | set_b
        if not union:
            return 1.0
        inter = set_a & set_b
        return 1.0 - (len(inter) / len(union))
    # fallback to cosine distance using embeddings
    dot = float(np.dot(emb_a, emb_b))
    dot = max(min(dot, 1.0), -1.0)
    return 1.0 - max(dot, 0.0)


def token_distance_for_items(
    idx_a: int,
    idx_b: int,
    tokens_filtered: List[List[int]],
    tokens_raw: List[List[int]],
    emb_filtered: np.ndarray,
    emb_raw: np.ndarray,
) -> float:
    toks_a = tokens_filtered[idx_a] if tokens_filtered[idx_a] else tokens_raw[idx_a]
    toks_b = tokens_filtered[idx_b] if tokens_filtered[idx_b] else tokens_raw[idx_b]
    emb_a = emb_filtered[idx_a] if tokens_filtered[idx_a] else emb_raw[idx_a]
    emb_b = emb_filtered[idx_b] if tokens_filtered[idx_b] else emb_raw[idx_b]
    return jaccard_distance(toks_a, toks_b, emb_a, emb_b)


def greedy_diverse_selection(
    node: int,
    nbr_dict: Dict[int, float],
    limit: int,
    tokens_filtered: List[List[int]],
    tokens_raw: List[List[int]],
    emb_filtered: np.ndarray,
    emb_raw: np.ndarray,
    diversity_threshold: float,
) -> Dict[int, float]:
    if not nbr_dict or limit <= 0:
        return {}
    ordered = sorted(nbr_dict.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[int] = []
    weights: Dict[int, float] = {}
    if ordered:
        first_j, first_w = ordered[0]
        selected.append(first_j)
        weights[first_j] = first_w
    while len(selected) < min(limit, len(ordered)):
        best = None
        best_key = (-1.0, -1.0)
        for j, w in ordered:
            if j in weights:
                continue
            # farthest-from-set selection
            min_dist = min(
                token_distance_for_items(
                    j, k, tokens_filtered, tokens_raw, emb_filtered, emb_raw
                ) for k in selected
            ) if selected else 1.0
            if min_dist < diversity_threshold:
                continue
            key = (min_dist, w)
            if key > best_key:
                best_key = key
                best = (j, w)
        if best is None:
            break
        selected.append(best[0])
        weights[best[0]] = best[1]
    return weights


def cosine_topk(vec: np.ndarray, mat: np.ndarray, topk: int, ignore: Optional[List[int]] = None) -> List[Tuple[int, float]]:
    sims = mat @ vec
    sims[:1] = -1.0  # ignore padding row 0
    if ignore:
        for i in ignore:
            if 0 <= i < sims.shape[0]:
                sims[i] = -1.0
    candidate_cap = min(len(sims) - 1, max(8, topk * 4))
    if candidate_cap <= 0:
        return []
    idx = np.argpartition(-sims, range(candidate_cap))[:candidate_cap]
    pairs = [(int(j), float(sims[j])) for j in idx if sims[j] > 0]
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs[:topk]


# ------------------------------
# Argparse
# ------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Construct item–item category_shared graph (Weighted Jaccard + IDF) with robust coverage."
    )
    ap.add_argument("--dataset", required=True, help="Dataset name under datasets/")
    ap.add_argument("--pre-topk", type=int, default=64,
                    help="Candidate neighbors per item before mutual filtering.")
    ap.add_argument("--topk", type=int, default=20,
                    help="Target neighbors per item after mutual/diversity selection.")
    ap.add_argument("--stop-freq-ratio", type=float, default=0.30,
                    help="Drop tokens with df/N above this ratio (generic labels).")
    ap.add_argument("--idf-floor", type=float, default=0.06,
                    help="Lower bound for IDF to prevent vanishing weights.")
    ap.add_argument("--use-leaf-only", action="store_true",
                    help="Use only the rarest token per item (acts like leaf).")
    ap.add_argument("--max-candidates-per-token", type=int, default=8000,
                    help="Cap candidate list length per token to control runtime.")
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="Drop edges with final (normalized) weight below this.")
    ap.add_argument("--k-min", type=int, default=16,
                    help="Minimum category neighbors to guarantee per item (coverage floor).")
    ap.add_argument("--output", default=None, help="Optional explicit output path (.pkl).")

    # Popularity/singleton regularizers (kept as no-ops if you don't use them)
    ap.add_argument("--wj-pop-penalty-alpha", type=float, default=0.0,
                    help="(Optional) Downweight WJ by (deg_i*deg_j)^alpha during selection (0 disables).")
    ap.add_argument("--singleton-cap", type=float, default=1.0,
                    help="Fraction of candidates allowed to be from singleton buckets (1.0 disables).")
    ap.add_argument("--singleton-freq-thresh", type=float, default=1.0,
                    help="Token frequency/N below which a token is considered singleton-ish.")

    # New control knobs
    ap.add_argument("--diversity-threshold", type=float, default=0.05,
                    help="Minimum token-distance required between selected neighbors.")
    ap.add_argument("--asym-top", type=int, default=12,
                    help="How many of pre-top candidates per node to consider for asymmetric additions.")
    ap.add_argument("--asym-tau", type=float, default=0.02,
                    help="Base threshold for asymmetric additions.")

    # Embedding augmentation
    ap.add_argument("--embed-augment-frac", type=float, default=0.50,
                    help="If pre-token candidates < frac*pre_topk, augment with embedding-KNN.")
    ap.add_argument("--embed-augment-topk", type=int, default=32,
                    help="How many embedding neighbors to inject when augmenting.")
    ap.add_argument("--embed-backfill-topk", type=int, default=64,
                    help="How many embedding neighbors to use if backfill has no candidates.")
    return ap.parse_args()


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    embed_base = Path("datasets") / args.dataset / "embeddings"
    graph_base = Path("datasets") / args.dataset / "global_graph"
    nodes = load_pickle(embed_base / "category_nodes.pkl")  # [{"id","text","freq"}, ...]
    links = load_pickle(embed_base / "item_category_links.pkl")["item_to_category"]  # List[List[int]]
    feature_path = embed_base / "category_matrix.npy"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Category embedding matrix missing at {feature_path}. "
            "Run convert_id_asin_info_to_embeddings.py with category enabled."
        )
    features = np.load(feature_path).astype(np.float32)
    features_norm = normalize_rows(features.copy())

    num_items = len(links) - 1  # row 0 is padding
    if num_items <= 0:
        raise RuntimeError("item_category_links has no items (did you run the converter?).")
    if not nodes:
        raise RuntimeError("category_nodes.pkl empty (did you run with --category-nodes?).")

    num_tokens = max(int(e["id"]) for e in nodes)
    # Token frequencies -> IDF
    df = np.zeros(num_tokens + 1, dtype=np.int64)
    for e in nodes:
        df[int(e["id"])] = max(1, int(e.get("freq", 1)))

    idf = np.log((num_items + 1) / (df + 1))
    idf = np.maximum(idf, args.idf_floor)

    # Per-item token sets
    item_tokens_raw: List[List[int]] = [None] * (num_items + 1)  # type: ignore
    for i in range(1, num_items + 1):
        toks = links[i] or []
        item_tokens_raw[i] = sorted(set(int(t) for t in toks))

    # Stop very generic tokens
    stop = (df / max(1, num_items)) > args.stop_freq_ratio

    # Filtered tokens (+ ensure at least 1–2 rare tokens kept)
    item_tokens: List[List[int]] = [None] * (num_items + 1)  # type: ignore
    for i in range(1, num_items + 1):
        raw = item_tokens_raw[i]
        if not raw:
            item_tokens[i] = []
            continue
        filtered = [t for t in raw if 1 <= t <= num_tokens and not stop[t]]
        if not filtered and raw:
            filtered = sorted(raw, key=lambda t: idf[t], reverse=True)[:2]
        elif len(filtered) == 1 and len(raw) > 1 and not args.use_leaf_only:
            extras = sorted([t for t in raw if t not in filtered], key=lambda t: idf[t], reverse=True)
            if extras:
                filtered = filtered + extras[:1]
        if args.use_leaf_only and filtered:
            filtered = sorted(filtered, key=lambda t: idf[t], reverse=True)[:1]
        item_tokens[i] = sorted(filtered)

    # Inverted index for filtered tokens
    token_to_items: Dict[int, List[int]] = {}
    for t in range(1, num_tokens + 1):
        if not stop[t]:
            token_to_items[t] = []
    for i in range(1, num_items + 1):
        for t in item_tokens[i]:
            token_to_items.setdefault(t, []).append(i)

    # Per-item embedding summaries
    embed_dim = features_norm.shape[1]
    item_embeddings_filtered = np.zeros((num_items + 1, embed_dim), dtype=np.float32)
    item_embeddings_raw = np.zeros((num_items + 1, embed_dim), dtype=np.float32)
    for i in range(1, num_items + 1):
        filt = [t for t in item_tokens[i] if 0 <= t < features_norm.shape[0]]
        raw = [t for t in item_tokens_raw[i] if 0 <= t < features_norm.shape[0]]
        if filt:
            item_embeddings_filtered[i] = features_norm[filt].mean(axis=0)
        if raw:
            item_embeddings_raw[i] = features_norm[raw].mean(axis=0)
    item_embeddings_filtered = normalize_rows(item_embeddings_filtered)
    item_embeddings_raw = normalize_rows(item_embeddings_raw)

    # Precompute per-item IDF sums
    sum_idf_per_item = np.zeros(num_items + 1, dtype=np.float32)
    for i in range(1, num_items + 1):
        sum_idf_per_item[i] = float(sum(idf[t] for t in item_tokens[i]))

    # Build pre-topk candidates with WJ+IDF; augment with embedding-KNN if too skinny
    neighbors: Dict[int, Dict[int, float]] = {i: {} for i in range(1, num_items + 1)}
    pre_candidates_sorted: Dict[int, List[Tuple[int, float]]] = {}
    min_needed = int(args.embed_augment_frac * max(1, args.pre_topk))

    for i in range(1, num_items + 1):
        toks_i = item_tokens[i]
        cand: Dict[int, float] = {}
        inter: Dict[int, float] = {}
        if toks_i:
            for t in toks_i:
                wt = float(idf[t])
                items = token_to_items.get(t, [])
                if len(items) > args.max_candidates_per_token:
                    items = items[: args.max_candidates_per_token]
                for j in items:
                    if j == i:
                        continue
                    if j not in cand:
                        cand[j] = 0.0
                        inter[j] = 0.0
                    inter[j] += wt

            sum_i = float(sum_idf_per_item[i])
            if sum_i > 0:
                for j in list(cand.keys()):
                    sum_j = float(sum_idf_per_item[j])
                    denom = sum_i + sum_j - inter[j]
                    if denom > 0:
                        w = inter[j] / denom
                        # Optional popularity penalty
                        if args.wj_pop_penalty_alpha > 0:
                            # simple degree proxy: number of posting lists touched by j
                            deg_proxy = max(1.0, len(item_tokens[j]))
                            w = w / (deg_proxy ** args.wj_pop_penalty_alpha)
                        if w > 0:
                            cand[j] = w
                        else:
                            cand.pop(j, None)
                    else:
                        cand.pop(j, None)

        # If we didn't get enough candidates (singleton/skinny), augment with embedding-KNN
        if len(cand) < min_needed:
            vec = item_embeddings_raw[i] if np.any(item_embeddings_filtered[i]) == 0 else item_embeddings_filtered[i]
            embed_pairs = cosine_topk(vec, item_embeddings_filtered, args.embed_augment_topk, ignore=[i])
            for j, s in embed_pairs:
                if j == i:
                    continue
                cand[j] = max(cand.get(j, 0.0), float(max(1e-8, s)))

        if cand:
            top = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[: args.pre_topk]
            neighbors[i] = {j: w for j, w in top}
            pre_candidates_sorted[i] = top
        else:
            neighbors[i] = {}
            pre_candidates_sorted[i] = []

    pre_edge_count = sum(len(nbrs) for nbrs in neighbors.values())

    # Mutual-K
    mutual_neighbors: Dict[int, Dict[int, float]] = {i: {} for i in range(1, num_items + 1)}
    pair_weights: Dict[Tuple[int, int], float] = {}
    for i in range(1, num_items + 1):
        for j, w_ij in neighbors[i].items():
            if j < 1 or j > num_items:
                continue
            w_ji = neighbors.get(j, {}).get(i)
            if w_ji is None:
                continue
            u, v = (i, j) if i < j else (j, i)
            if (u, v) in pair_weights:
                continue
            w = 0.5 * (w_ij + w_ji)
            pair_weights[(u, v)] = w
            mutual_neighbors[i][j] = w
            mutual_neighbors[j][i] = w
    mutual_edge_count = len(pair_weights)

    # Diversity-aware selection
    diverse_neighbors: Dict[int, Dict[int, float]] = {i: {} for i in range(1, num_items + 1)}
    for i in range(1, num_items + 1):
        diverse_neighbors[i] = greedy_diverse_selection(
            i,
            mutual_neighbors[i],
            args.topk,
            item_tokens,
            item_tokens_raw,
            item_embeddings_filtered,
            item_embeddings_raw,
            args.diversity_threshold,
        )

    # Prepare ordered mutual for backfill
    mutual_candidates_sorted: Dict[int, List[Tuple[int, float]]] = {
        i: (sorted(mutual_neighbors[i].items(), key=lambda kv: kv[1], reverse=True) if mutual_neighbors[i] else [])
        for i in range(1, num_items + 1)
    }

    # Final undirected set: union of diversified mutual selections
    final_pairs: Dict[Tuple[int, int], float] = {}
    for i in range(1, num_items + 1):
        for j, w in diverse_neighbors[i].items():
            if j < 1 or j > num_items:
                continue
            pair = (i, j) if i < j else (j, i)
            base_w = pair_weights.get(pair, w)
            if base_w <= 0:
                continue
            if pair not in final_pairs or base_w > final_pairs[pair]:
                final_pairs[pair] = base_w
    diverse_edge_count = len(final_pairs)

    # Build adjacency and deg counts
    final_adj: Dict[int, set] = {i: set() for i in range(1, num_items + 1)}
    deg_counts = np.zeros(num_items + 1, dtype=np.int32)
    for u, v in final_pairs.keys():
        final_adj[u].add(v)
        final_adj[v].add(u)
        deg_counts[u] += 1
        deg_counts[v] += 1

    # Asymmetric additions (use topT of pre-top candidates)
    def diversity_ok(node: int, candidate: int, threshold: float) -> bool:
        neigh = final_adj.get(node, set())
        if not neigh:
            return True
        distances = [
            token_distance_for_items(
                candidate, other, item_tokens, item_tokens_raw, item_embeddings_filtered, item_embeddings_raw
            ) for other in neigh
        ]
        return (not distances) or (min(distances) >= threshold)

    asymmetric_added = 0
    top_T = max(1, min(args.asym_top, args.pre_topk))
    tau_floor = float(args.asym_tau)
    for i in range(1, num_items + 1):
        top_candidates = pre_candidates_sorted.get(i, [])[:top_T]
        if not top_candidates:
            continue
        mutual_vals = list(mutual_neighbors[i].values())
        tau_base = float(np.median(mutual_vals)) if mutual_vals else (
            float(np.median([score for _, score in top_candidates])) if top_candidates else 0.0
        )
        tau = max(tau_floor, tau_base)
        extras: List[Tuple[float, float, int, Tuple[int, int]]] = []
        for j, score in top_candidates:
            pair = (i, j) if i < j else (j, i)
            if pair in final_pairs:
                continue
            base_w = pair_weights.get(pair, score)
            if base_w < tau:
                continue
            penalized = base_w / math.sqrt(deg_counts[j] + 1)
            extras.append((penalized, base_w, j, pair))
        extras.sort(key=lambda x: x[0], reverse=True)
        for penalized, base_w, j, pair in extras:
            if penalized <= 0:
                continue
            if pair in final_pairs:
                continue
            if not diversity_ok(i, j, args.diversity_threshold) or not diversity_ok(j, i, args.diversity_threshold):
                continue
            final_pairs[pair] = base_w
            final_adj[i].add(j)
            final_adj[j].add(i)
            deg_counts[i] += 1
            deg_counts[j] += 1
            asymmetric_added += 1

    # Backfill to k-min (with embedding fallback when no candidates)
    backfill_edges = 0
    for i in range(1, num_items + 1):
        while deg_counts[i] < args.k_min:
            # gather candidates from mutual+pre
            candidate_scores: Dict[int, float] = {}
            for j, score in mutual_candidates_sorted.get(i, []):
                candidate_scores[j] = max(candidate_scores.get(j, 0.0), float(score))
            for j, score in pre_candidates_sorted.get(i, []):
                candidate_scores[j] = max(candidate_scores.get(j, 0.0), float(score))

            # if none, synthesize from embeddings
            if not candidate_scores:
                vec = item_embeddings_filtered[i] if np.any(item_embeddings_filtered[i]) else item_embeddings_raw[i]
                if not np.any(vec):
                    break
                for j, s in cosine_topk(vec, item_embeddings_filtered, args.embed_backfill_topk, ignore=[i]):
                    candidate_scores[j] = max(candidate_scores.get(j, 0.0), float(max(1e-8, s)))

            if not candidate_scores:
                break

            dynamic_deg95 = float(np.percentile(deg_counts[1:], 95)) if deg_counts[1:].size else 0.0
            best_choice = None
            best_penalized = -1.0

            for j, base_candidate_score in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True):
                if j == i:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in final_pairs:
                    continue
                if dynamic_deg95 > 0 and deg_counts[j] >= dynamic_deg95:
                    continue

                base_weight = final_pairs.get(pair)
                if base_weight is None:
                    w_ij = neighbors[i].get(j, 0.0)
                    w_ji = neighbors.get(j, {}).get(i, 0.0)
                    if w_ij > 0 and w_ji > 0:
                        base_weight = 0.5 * (w_ij + w_ji)
                    else:
                        base_weight = max(w_ij, w_ji, base_candidate_score)

                if base_weight is None or base_weight <= 0:
                    continue

                if not diversity_ok(i, j, args.diversity_threshold) or not diversity_ok(j, i, args.diversity_threshold):
                    continue

                penalized = base_weight / math.sqrt(deg_counts[j] + 1)
                if penalized > best_penalized:
                    best_penalized = penalized
                    best_choice = (j, base_weight, pair)

            if best_choice is None:
                break

            j, base_weight, pair = best_choice
            final_pairs[pair] = base_weight
            final_adj[i].add(j)
            final_adj[j].add(i)
            deg_counts[i] += 1
            deg_counts[j] += 1
            backfill_edges += 1

    # Degree-normalize weights
    deg_float = deg_counts.astype(np.float32)
    deg_float[deg_float == 0] = 1.0
    normalized_pairs: Dict[Tuple[int, int], float] = {}
    for (u, v), base_w in final_pairs.items():
        norm_w = base_w / math.sqrt(deg_float[u] * deg_float[v])
        if args.min_weight > 0 and norm_w < args.min_weight:
            continue
        normalized_pairs[(u, v)] = norm_w

    # Convert to directed COO (no per-source rescale by default)
    src: List[int] = []
    dst: List[int] = []
    wts: List[float] = []
    for (u, v), w in normalized_pairs.items():
        src.append(u); dst.append(v); wts.append(w)
        src.append(v); dst.append(u); wts.append(w)

    if src:
        edge_index = np.stack([np.array(src, np.int64), np.array(dst, np.int64)], axis=0)
        edge_weight = np.array(wts, np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    # Stats for logging
    undirected_edge_count = len(normalized_pairs)
    deg_undirected_arr = np.zeros(num_items + 1, dtype=np.int32)
    for (u, v) in normalized_pairs.keys():
        deg_undirected_arr[u] += 1
        deg_undirected_arr[v] += 1
    deg_undirected = deg_undirected_arr[1:]
    zero_degree = int(np.sum(deg_undirected == 0)) if deg_undirected.size else 0
    deg_min = int(deg_undirected.min()) if deg_undirected.size else 0
    deg_median = float(np.median(deg_undirected)) if deg_undirected.size else 0.0
    deg_p95 = float(np.percentile(deg_undirected, 95)) if deg_undirected.size else 0.0
    deg_p99 = float(np.percentile(deg_undirected, 99)) if deg_undirected.size else 0.0
    deg_max = int(deg_undirected.max()) if deg_undirected.size else 0

    total_degree = float(deg_undirected.sum())
    top_share = 0.0
    if total_degree > 0 and num_items > 0:
        top_count = max(1, int(math.ceil(0.01 * num_items)))
        top_share = float(np.sort(deg_undirected)[::-1][:top_count].sum()) / total_degree

    weight_vals = np.array(list(normalized_pairs.values()), dtype=np.float32) if normalized_pairs else np.zeros((0,), np.float32)
    weight_min = float(weight_vals.min()) if weight_vals.size else 0.0
    weight_median = float(np.median(weight_vals)) if weight_vals.size else 0.0
    weight_p95 = float(np.percentile(weight_vals, 95)) if weight_vals.size else 0.0
    weight_p99 = float(np.percentile(weight_vals, 99)) if weight_vals.size else 0.0
    weight_max = float(weight_vals.max()) if weight_vals.size else 0.0

    # Neighbor similarity sample (optional telemetry)
    final_neighbors_map: Dict[int, List[int]] = {i: [] for i in range(1, num_items + 1)}
    for (u, v) in normalized_pairs.keys():
        final_neighbors_map[u].append(v)
        final_neighbors_map[v].append(u)
    neighbor_similarity_samples: List[float] = []
    max_pairs_per_node = 200
    for i in range(1, num_items + 1):
        nbrs = final_neighbors_map[i]
        if len(nbrs) < 2:
            continue
        pairs_added = 0
        for ia in range(len(nbrs)):
            if pairs_added >= max_pairs_per_node:
                break
            for ib in range(ia + 1, len(nbrs)):
                a = nbrs[ia]; b = nbrs[ib]
                sim = 1.0 - jaccard_distance(
                    item_tokens[a],
                    item_tokens[b],
                    item_embeddings_filtered[a],
                    item_embeddings_filtered[b],
                )
                neighbor_similarity_samples.append(sim)
                pairs_added += 1
                if pairs_added >= max_pairs_per_node:
                    break
    median_neighbor_similarity = float(np.median(neighbor_similarity_samples)) if neighbor_similarity_samples else float("nan")

    out = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "meta": {
            "dataset": args.dataset,
            "modality": "category_shared",
            "scoring": "weighted_jaccard_idf",
            "pre_topk": int(args.pre_topk),
            "topk_per_item": int(args.topk),
            "coverage_floor": int(args.k_min),
            "degree_normalized": True,
            "stop_freq_ratio": float(args.stop_freq_ratio),
            "idf_floor": float(args.idf_floor),
            "use_leaf_only": bool(args.use_leaf_only),
            "max_candidates_per_token": int(args.max_candidates_per_token),
            "min_weight": float(args.min_weight),
            "mutual_pairs": int(mutual_edge_count),
            "diversity_pairs": int(diverse_edge_count),
            "asymmetric_pairs": int(asymmetric_added),
            "backfilled_pairs": int(undirected_edge_count),
            "fallback_items": 0,  # we now augment proactively instead of late fallback
            "per_source_rescale": False,
            "wj_pop_penalty_alpha": float(args.wj_pop_penalty_alpha),
            "singleton_cap": float(args.singleton_cap),
            "singleton_freq_thresh": float(args.singleton_freq_thresh),
            "diversity_threshold": float(args.diversity_threshold),
            "asym_top": int(args.asym_top),
            "asym_tau": float(args.asym_tau),
            "embed_augment_frac": float(args.embed_augment_frac),
            "embed_augment_topk": int(args.embed_augment_topk),
            "embed_backfill_topk": int(args.embed_backfill_topk),
            "num_nodes_including_padding": int(num_items + 1),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    out_path = (Path(args.output) if args.output
                else Path("datasets") / args.dataset / "global_graph" / "global_graph_category_shared.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    deg_directed = np.bincount(edge_index[0], minlength=num_items + 1)[1:]
    print(f"Saved: {out_path}")
    print(
        "Edges (pipeline pre->mutual->diverse->final): "
        f"{pre_edge_count} -> {mutual_edge_count} -> {diverse_edge_count} -> {undirected_edge_count} "
        f"(undirected) | directed: {edge_index.shape[1]} | strong-asym added: {asymmetric_added} | backfill added: {undirected_edge_count - diverse_edge_count}"
    )
    print(
        "Degree stats (undirected) min/median/p95/p99/max: "
        f"{deg_min}/{deg_median:.1f}/{deg_p95:.1f}/{deg_p99:.1f}/{deg_max} | "
        f"zero-degree items: {zero_degree} | top1% edge-share: {top_share * 100:.1f}%"
    )
    print(
        "Edge weight stats (degree-norm) min/median/p95/p99/max: "
        f"{weight_min:.4f}/{weight_median:.4f}/{weight_p95:.4f}/{weight_p99:.4f}/{weight_max:.4f}"
    )
    if not math.isnan(median_neighbor_similarity):
        print(f"Median neighbor-pair similarity sample: {median_neighbor_similarity:.3f}")
    else:
        print("Median neighbor-pair similarity sample: n/a")
    print(
        f"Directed avg degree: {deg_directed.mean():.2f}"
    )


if __name__ == "__main__":
    main()
