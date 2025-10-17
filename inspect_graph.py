#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect global graph pickle files (per-modality or unified).

Examples
--------
Inspect image KNN graph:
    python inspect_graph.py --dataset Amazon_cellPhone_2018 --modality image

Inspect category_shared graph:
    python inspect_graph.py --dataset Amazon_cellPhone_2018 --modality category_shared

Inspect unified graph explicitly:
    python inspect_graph.py --dataset Amazon_cellPhone_2018 --modality unified

Or provide an explicit path:
    python inspect_graph.py --path datasets/Amazon_cellPhone_2018/global_graph/global_graph_title_knn.pkl
"""

import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEG_THRESHOLDS = (10, 20, 40, 80, 160, 320)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("min", "median", "p95", "p99", "max", "mean")}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def degree_thresholds(deg: np.ndarray) -> Dict[int, int]:
    return {thr: int((deg >= thr).sum()) for thr in DEG_THRESHOLDS}


def infer_graph_path(dataset: str, modality: str) -> Optional[Path]:
    base = Path("datasets") / dataset / "global_graph"
    candidates = []
    if modality == "unified":
        candidates.append(base / "global_graph_unified.pkl")
    else:
        candidates.extend([
            base / f"global_graph_{modality}_knn.pkl",
            base / f"global_graph_{modality}.pkl",
            base / f"global_graph_{modality}_shared.pkl",
        ])
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def aggregate_undirected(edge_index: np.ndarray, edge_weight: np.ndarray) -> Tuple[Dict[Tuple[int, int], float], List[Tuple[int, int]], int, float]:
    pair_values: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    directed_seen: Dict[Tuple[int, int], set] = defaultdict(set)
    for col in range(edge_index.shape[1]):
        u = int(edge_index[0, col])
        v = int(edge_index[1, col])
        if u == v or u == 0 or v == 0:
            continue
        w = float(edge_weight[col]) if edge_weight.size > 0 else 1.0
        key = (u, v) if u < v else (v, u)
        pair_values[key].append(w)
        directed_seen[key].add((u, v))

    pair_avg: Dict[Tuple[int, int], float] = {}
    asym_pairs = 0
    max_gap = 0.0
    for key, values in pair_values.items():
        if len(values) >= 2:
            gap = float(max(values) - min(values))
            if gap > 1e-6:
                asym_pairs += 1
                max_gap = max(max_gap, gap)
        pair_avg[key] = float(sum(values) / len(values))

    missing_reverse = [key for key, dirs in directed_seen.items() if len(dirs) < 2]
    return pair_avg, missing_reverse, asym_pairs, max_gap


def build_adjacency(pairs: Dict[Tuple[int, int], float], num_nodes: int) -> Dict[int, List[Tuple[int, float]]]:
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for (u, v), w in pairs.items():
        if u <= 0 or v <= 0 or u >= num_nodes or v >= num_nodes:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
    for lst in adj.values():
        lst.sort(key=lambda kv: kv[1], reverse=True)
    return adj


def percentile_summary(values: np.ndarray, name: str) -> None:
    if values.size == 0:
        print(f"{name}: no data")
        return
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    quant = np.percentile(values, percentiles)
    pretty = ", ".join(f"p{p}={q:.2f}" for p, q in zip(percentiles, quant))
    print(f"{name}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f} | {pretty}")


def summarize_per_modality(edge_index: np.ndarray,
                           edge_weight: np.ndarray,
                           num_nodes: int,
                           label: str) -> None:
    deg = np.bincount(edge_index[0], minlength=num_nodes)[1:]
    w_stats = stats(edge_weight)
    d_stats = stats(deg)
    thresh = degree_thresholds(deg)
    print(f"[{label}] edges={edge_index.shape[1]}  weight_stats={w_stats}")
    print(f"         degree_stats={d_stats}")
    print(f"         degree>=thresholds {DEG_THRESHOLDS}: {thresh}")


def infer_dataset_from_path(path: Path) -> Optional[str]:
    parts = path.resolve().parts
    for idx, part in enumerate(parts):
        if part == "datasets" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def format_tokens(tokens: Sequence[int], token_map: Dict[int, str], limit: int = 6) -> str:
    if not tokens:
        return "[]"
    names = [token_map.get(t, f"#{t}") for t in tokens[:limit]]
    if len(tokens) > limit:
        names.append(f"...(+{len(tokens) - limit})")
    return "[" + ", ".join(names) + "]"


def weighted_jaccard(ids_a: Sequence[int], ids_b: Sequence[int], idf: np.ndarray) -> float:
    set_a = set(ids_a)
    set_b = set(ids_b)
    if not set_a and not set_b:
        return 0.0
    inter = set_a & set_b
    sum_a = sum(idf[t] for t in set_a)
    sum_b = sum(idf[t] for t in set_b)
    inter_sum = sum(idf[t] for t in inter)
    denom = sum_a + sum_b - inter_sum
    if denom <= 0:
        return 0.0
    return inter_sum / denom


def describe_high_degree_nodes(adj: Dict[int, List[Tuple[int, float]]],
                               nodes: Sequence[int],
                               top_neighbors: int) -> None:
    for node in nodes:
        neighbors = adj.get(node, [])
        if not neighbors:
            print(f"  - item {node}: degree=0 (no neighbors)")
            continue
        weights = np.array([w for _, w in neighbors], dtype=np.float32)
        print(
            f"  - item {node}: degree={len(neighbors)} | weight min/median/max="
            f"{weights.min():.4f}/{np.median(weights):.4f}/{weights.max():.4f}"
        )
        for nbr, w in neighbors[:top_neighbors]:
            print(f"      -> neighbor {nbr} (w={w:.4f})")


def analyze_category_shared(dataset: Optional[str],
                            meta: Dict,
                            adj: Dict[int, List[Tuple[int, float]]],
                            highlight_nodes: Sequence[int],
                            top_neighbors: int) -> None:
    if not dataset:
        print("[warn] Dataset name unavailable; skipping category_shared token analysis.")
        return

    emb_dir = Path("datasets") / dataset / "embeddings"
    nodes_path = emb_dir / "category_nodes.pkl"
    links_path = emb_dir / "item_category_links.pkl"

    if not nodes_path.exists() or not links_path.exists():
        print(f"[warn] Missing category embeddings for dataset '{dataset}'. Expected {nodes_path} and {links_path}.")
        return

    try:
        nodes = load_pickle(nodes_path)
        links = load_pickle(links_path)["item_to_category"]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Could not load category metadata: {exc}")
        return

    token_text = {int(entry["id"]): str(entry.get("text", entry["id"])) for entry in nodes}
    num_items = len(links) - 1
    if num_items <= 0:
        print("[warn] item_category_links.pkl contains no items.")
        return

    stop_ratio = float(meta.get("stop_freq_ratio", 0.2))
    idf_floor = float(meta.get("idf_floor", 0.1))
    use_leaf_only = bool(meta.get("use_leaf_only", False))

    num_tokens = max(int(entry["id"]) for entry in nodes)
    df = np.zeros(num_tokens + 1, dtype=np.int64)
    for entry in nodes:
        df[int(entry["id"])] = max(1, int(entry.get("freq", 1)))

    idf = np.log((num_items + 1) / (df + 1))
    idf = np.maximum(idf, idf_floor)
    stop_mask = (df / max(1, num_items)) > stop_ratio

    item_tokens_raw: List[List[int]] = [None] * (num_items + 1)  # type: ignore[assignment]
    item_tokens_filtered: List[List[int]] = [None] * (num_items + 1)  # type: ignore[assignment]
    for idx in range(1, num_items + 1):
        raw = sorted(set(int(t) for t in links[idx] or []))
        item_tokens_raw[idx] = raw
        filtered = [t for t in raw if 0 <= t < stop_mask.shape[0] and not stop_mask[t]]
        if use_leaf_only and filtered:
            best = max(filtered, key=lambda t: idf[t])
            filtered = [best]
        item_tokens_filtered[idx] = filtered

    raw_counts = np.array([len(item_tokens_raw[idx]) for idx in range(1, num_items + 1)], dtype=np.int32)
    filtered_counts = np.array([len(item_tokens_filtered[idx]) for idx in range(1, num_items + 1)], dtype=np.int32)
    dropped_counts = raw_counts - filtered_counts

    zero_raw = int((raw_counts == 0).sum())
    zero_filtered = int((filtered_counts == 0).sum())
    fallback_meta = int(meta.get("fallback_items", 0))

    print("\n[category_shared] Token coverage overview")
    print(f"  Items with raw tokens: {num_items - zero_raw}/{num_items}  ({100.0 * (num_items - zero_raw) / max(1, num_items):.1f}%)")
    print(f"  Items surviving stoplist: {num_items - zero_filtered}/{num_items}  ({100.0 * (num_items - zero_filtered) / max(1, num_items):.1f}%)")
    print(f"  Items requiring fallback (meta): {fallback_meta}")
    print(f"  Items with filtered tokens == 0 (derived): {zero_filtered}")
    percentile_summary(raw_counts.astype(np.float32), "  Raw token count per item")
    percentile_summary(filtered_counts.astype(np.float32), "  Filtered token count per item")
    percentile_summary(dropped_counts.astype(np.float32), "  Tokens dropped per item (stoplist/leaf)")

    print("\n[category_shared] Highlighted nodes (token-level view)")
    for node in highlight_nodes:
        if node <= 0 or node > num_items:
            print(f"  - item {node}: outside category metadata range (num_items={num_items})")
            continue
        neighbors = adj.get(node, [])
        filtered = item_tokens_filtered[node]
        raw = item_tokens_raw[node]
        fallback = (len(filtered) == 0 and len(raw) > 0)
        print(
            f"  - item {node}: degree={len(neighbors)} | raw_tokens={len(raw)} | filtered_tokens={len(filtered)} | fallback={fallback}"
        )
        print(f"      raw:      {format_tokens(raw, token_text)}")
        print(f"      filtered: {format_tokens(filtered, token_text)}")
        for nbr, weight in neighbors[:top_neighbors]:
            if nbr <= 0 or nbr > num_items:
                print(f"      -> neighbor {nbr} (w={weight:.4f}) [out of range]")
                continue
            nbr_filtered = item_tokens_filtered[nbr]
            nbr_raw = item_tokens_raw[nbr]
            nbr_fallback = (len(nbr_filtered) == 0 and len(nbr_raw) > 0)
            tokens_self = filtered if filtered else raw
            tokens_peer = nbr_filtered if nbr_filtered else nbr_raw
            if tokens_self and tokens_peer:
                shared = sorted(set(tokens_self) & set(tokens_peer))
                wj = weighted_jaccard(filtered, nbr_filtered, idf)
            else:
                shared = []
                wj = 0.0
            shared_txt = format_tokens(shared, token_text)
            print(
                f"      -> neighbor {nbr} "
                f"(w={weight:.4f}, deg={len(adj.get(nbr, []))}, fallback={nbr_fallback}) "
                f"shared_tokens={shared_txt} wj_filtered={wj:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Inspect global graph pickle.")
    parser.add_argument("--dataset", help="Dataset name under datasets/")
    parser.add_argument("--modality", default="", help="Modality tag (image/title/category_shared/unified/...).")
    parser.add_argument("--path", help="Explicit path to graph pickle.")
    parser.add_argument(
        "--top-nodes",
        type=int,
        default=5,
        help="Number of highest-degree nodes to highlight in detailed output.",
    )
    parser.add_argument(
        "--top-neighbors",
        type=int,
        default=5,
        help="Number of neighbors to display for highlighted nodes.",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="Comma/space separated list of item ids to inspect (in addition to automatic selections).",
    )
    parser.add_argument(
        "--disable-category-detail",
        action="store_true",
        help="Skip category-specific deep analysis even if metadata indicates category_shared.",
    )
    args = parser.parse_args()

    graph_path: Optional[Path] = None
    modality_label = args.modality

    if args.path:
        graph_path = Path(args.path)
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        if not modality_label:
            modality_label = graph_path.stem.replace("global_graph_", "")
    elif args.dataset and args.modality:
        graph_path = infer_graph_path(args.dataset, args.modality)
        if graph_path is None:
            raise FileNotFoundError(f"Could not find graph for dataset={args.dataset}, modality={args.modality}")
    else:
        raise ValueError("Provide either --path or both --dataset and --modality.")

    data = load_pickle(graph_path)
    meta = data.get("meta", {})
    edge_index = np.asarray(data.get("edge_index"))
    edge_weight_raw = data.get("edge_weight")
    if edge_weight_raw is None:
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    else:
        edge_weight = np.asarray(edge_weight_raw)
    edge_type = np.asarray(data.get("edge_type")) if "edge_type" in data else None

    print(f"Graph path: {graph_path}")
    print(f"Metadata: {meta}")

    focus_ids: List[int] = []
    if args.focus:
        for token in args.focus.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            try:
                focus_ids.append(int(token))
            except ValueError:
                print(f"[warn] Could not parse focus id '{token}'. Skipping.")

    num_nodes_total = int(meta.get("num_nodes_including_padding", 0))
    if num_nodes_total <= 0:
        num_nodes_total = int(edge_index.max()) + 1 if edge_index.size else 0
    num_real_nodes = max(0, num_nodes_total - 1)

    if edge_type is not None and edge_type.size > 0:
        print("Detected unified/multi-relation graph.")
        relation_map = meta.get("relation_map", {})
        unique_rels = sorted(set(int(r) for r in edge_type.tolist()))
        for rel in unique_rels:
            mask = edge_type == rel
            label = next((name for name, idx in relation_map.items() if idx == rel), f"rel{rel}")
            summarize_per_modality(edge_index[:, mask], edge_weight[mask], num_nodes_total, label)
    else:
        label = meta.get("modality", modality_label or "graph")
        summarize_per_modality(edge_index, edge_weight, num_nodes_total, label)

    pairs, missing_reverse, asym_pairs, max_gap = aggregate_undirected(edge_index, edge_weight)
    adj = build_adjacency(pairs, num_nodes_total)

    deg_directed = np.bincount(edge_index[0], minlength=num_nodes_total)[1:]
    deg_undirected = np.array([len(adj[idx]) for idx in range(1, num_nodes_total)], dtype=np.int32)
    weight_pairs = np.array(list(pairs.values()), dtype=np.float32)

    print("\n[general] Directed/undirected consistency")
    print(f"  Directed edges: {edge_index.shape[1]}")
    print(f"  Unique undirected pairs: {len(pairs)}")
    print(f"  Missing reverse edges: {len(missing_reverse)}")
    print(f"  Asymmetric weight pairs (>1e-6 diff): {asym_pairs} | max gap={max_gap:.6f}")
    percentile_summary(deg_directed.astype(np.float32), "  Directed degree per item")
    percentile_summary(deg_undirected.astype(np.float32), "  Undirected degree per item")
    percentile_summary(weight_pairs, "  Undirected edge weight distribution")

    top_k = max(0, args.top_nodes)
    top_by_degree = sorted(
        range(1, num_real_nodes + 1),
        key=lambda nid: (len(adj.get(nid, [])), adj.get(nid, [(0, 0.0)])[0][1] if adj.get(nid) else 0.0),
        reverse=True,
    )[:top_k]

    highlight_nodes: List[int] = []
    highlight_nodes.extend(top_by_degree)
    for nid in focus_ids:
        if nid > 0 and nid not in highlight_nodes:
            highlight_nodes.append(nid)
    highlight_nodes = [nid for nid in highlight_nodes if nid > 0]

    if highlight_nodes:
        print(f"\n[top-degree nodes] showing up to {args.top_neighbors} neighbors each")
        describe_high_degree_nodes(adj, highlight_nodes, args.top_neighbors)

    dataset_name = args.dataset or infer_dataset_from_path(graph_path)
    is_category = False
    tag = (modality_label or "").lower()
    modality_meta = str(meta.get("modality", "")).lower()
    if "category_shared" in {tag, modality_meta}:
        is_category = True
    elif "category_shared" in graph_path.name:
        is_category = True

    if not args.disable_category_detail and is_category:
        analyze_category_shared(dataset_name, meta, adj, highlight_nodes, args.top_neighbors)


if __name__ == "__main__":
    main()
