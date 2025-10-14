#!/usr/bin/env python3
"""
Richer diagnostics for unified multi-relation global graphs.

Example:
    python inspect_unified_global_graph.py --dataset Amazon_cellPhone_2018
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEG_THRESHOLDS = (50, 100, 200, 500, 1000)


def load_pickle(path: Path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def degree_threshold_counts(values: np.ndarray, thresholds: Sequence[int]) -> Dict[int, int]:
    return {thr: int((values >= thr).sum()) for thr in thresholds}


def build_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "top1pct_edge_share": float("nan"),
            "threshold_counts": {},
        }

    sorted_vals = np.sort(values)
    total = float(sorted_vals.sum())
    take = max(int(len(sorted_vals) * 0.01), 1)
    top_share = float(sorted_vals[-take:].sum()) / total if total > 0 else float("nan")

    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "top1pct_edge_share": top_share,
        "threshold_counts": degree_threshold_counts(values, DEG_THRESHOLDS),
    }


def describe_degrees(edge_index: np.ndarray, edge_type: np.ndarray, num_nodes: int) -> Dict[int, Dict[str, float]]:
    deg_total = np.bincount(edge_index[0], minlength=num_nodes)
    stats = {-1: build_stats(deg_total[1:])}

    for rel in np.unique(edge_type):
        mask = edge_type == rel
        deg_rel = np.bincount(edge_index[0, mask], minlength=num_nodes)
        stats[int(rel)] = build_stats(deg_rel[1:])
    return stats


def top_nodes(edge_index: np.ndarray, num_nodes: int, topk: int = 10) -> List[Tuple[int, int]]:
    deg_total = np.bincount(edge_index[0], minlength=num_nodes)
    pairs = [(idx, int(deg_total[idx])) for idx in range(1, num_nodes)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topk]


def top_nodes_per_relation(edge_index: np.ndarray, edge_type: np.ndarray, num_nodes: int, rel: int, topk: int = 5) -> List[Tuple[int, int]]:
    mask = edge_type == rel
    if not np.any(mask):
        return []
    deg_rel = np.bincount(edge_index[0, mask], minlength=num_nodes)
    pairs = [(idx, int(deg_rel[idx])) for idx in range(1, num_nodes)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topk]


def modality_mix(edge_index: np.ndarray, edge_type: np.ndarray, num_nodes: int, relation_ids: Sequence[int], relation_names: Sequence[str]) -> Dict[str, object]:
    counts = np.zeros((num_nodes, len(relation_ids)), dtype=np.int64)
    for idx, rel in enumerate(relation_ids):
        mask = edge_type == rel
        if np.any(mask):
            np.add.at(counts[:, idx], edge_index[0, mask], 1)

    counts = counts[1:]  # drop padding
    totals = counts.sum(axis=1, keepdims=True)
    nonzero_mask = totals.squeeze() > 0
    shares = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, totals, out=shares, where=totals > 0)

    summary = {
        "mean_share": {name: float(shares[nonzero_mask, idx].mean()) for idx, name in enumerate(relation_names)},
        "median_share": {name: float(np.median(shares[nonzero_mask, idx])) for idx, name in enumerate(relation_names)},
        "zero_degree_items": int((totals.squeeze() == 0).sum()),
        "dominant_examples": {},
    }

    for idx, name in enumerate(relation_names):
        dominant = np.where(shares[:, idx] >= 0.9)[0] + 1  # convert back to item ids
        summary["dominant_examples"][name] = dominant[:10].tolist()

    return summary


def relation_overlap(edge_index: np.ndarray, edge_type: np.ndarray) -> Dict[str, object]:
    pair_relations: Dict[Tuple[int, int], set] = {}
    for src, dst, rel in zip(edge_index[0], edge_index[1], edge_type):
        if src == 0 or dst == 0 or src == dst:
            continue
        key = (int(src), int(dst)) if src < dst else (int(dst), int(src))
        pair_relations.setdefault(key, set()).add(int(rel))

    size_counter = Counter(len(rels) for rels in pair_relations.values())
    combo_counter = Counter(tuple(sorted(rels)) for rels in pair_relations.values() if len(rels) > 1)
    return {
        "overlap_size_counts": dict(sorted(size_counter.items())),
        "relation_combos": {"+".join(map(str, combo)): count for combo, count in combo_counter.items()},
    }


def build_adj_lists(edge_index: np.ndarray, edge_type: np.ndarray, num_nodes: int, relation_ids: Sequence[int]) -> Dict[int, List[set]]:
    adj = {rel: [set() for _ in range(num_nodes)] for rel in relation_ids}
    for src, dst, rel in zip(edge_index[0], edge_index[1], edge_type):
        if src == 0 or dst == 0:
            continue
        rel_list = adj[int(rel)]
        rel_list[int(src)].add(int(dst))
        rel_list[int(dst)].add(int(src))
    return adj


def modality_jaccard(adj_lists: Dict[int, List[set]], relation_ids: Sequence[int], relation_names: Sequence[str], sample_nodes: int = 1000) -> Dict[str, float]:
    if len(relation_ids) < 2:
        return {}
    rng = np.random.default_rng(42)
    num_nodes = len(next(iter(adj_lists.values())))
    candidates = np.arange(1, num_nodes)
    if candidates.size == 0:
        return {}
    sampled = rng.choice(candidates, size=min(sample_nodes, candidates.size), replace=False)

    results = {}
    for i, rel_a in enumerate(relation_ids):
        for j in range(i + 1, len(relation_ids)):
            rel_b = relation_ids[j]
            key = f"{relation_names[i]}<->{relation_names[j]}"
            vals = []
            list_a = adj_lists[rel_a]
            list_b = adj_lists[rel_b]
            for node in sampled:
                neigh_a = list_a[node]
                neigh_b = list_b[node]
                if not neigh_a and not neigh_b:
                    continue
                union = len(neigh_a | neigh_b)
                intersection = len(neigh_a & neigh_b)
                if union > 0:
                    vals.append(intersection / union)
            if vals:
                results[key] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "samples": len(vals),
                }
    return results


def main():
    parser = argparse.ArgumentParser(description="Inspect unified global graph pickle.")
    parser.add_argument("--dataset", required=True, help="Dataset folder under datasets/")
    parser.add_argument("--path", default="", help="Optional explicit path to unified graph pickle.")
    parser.add_argument("--topk", type=int, default=10, help="Show top-k nodes by total degree.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    default_path = base / "datasets" / args.dataset / "global_graph" / "global_graph_unified.pkl"
    graph_path = Path(args.path) if args.path else default_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Unified graph not found: {graph_path}")

    graph = load_pickle(graph_path)
    edge_index = np.asarray(graph["edge_index"])
    edge_type = np.asarray(graph["edge_type"])
    edge_weight = np.asarray(graph["edge_weight"])
    meta = graph.get("meta", {})
    modalities = meta.get("modalities", [])
    relation_map = meta.get("relation_map", {})
    if modalities and relation_map:
        relation_ids = [relation_map[name] for name in modalities]
    elif relation_map:
        modalities = [name for name, _ in sorted(relation_map.items(), key=lambda kv: kv[1])]
        relation_ids = [relation_map[name] for name in modalities]
    else:
        relation_ids = sorted(np.unique(edge_type).tolist())
        modalities = [f"rel{rel}" for rel in relation_ids]

    num_nodes = int(meta.get("num_nodes_including_padding", edge_index.max() + 1))

    undirected_pairs = set()
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src <= dst:
            undirected_pairs.add((int(src), int(dst)))
        else:
            undirected_pairs.add((int(dst), int(src)))

    degree_stats = describe_degrees(edge_index, edge_type, num_nodes)
    top_total = top_nodes(edge_index, num_nodes, args.topk)
    rel_top = {name: top_nodes_per_relation(edge_index, edge_type, num_nodes, rel, topk=min(5, args.topk))
               for name, rel in zip(modalities, relation_ids)}
    mix_summary = modality_mix(edge_index, edge_type, num_nodes, relation_ids, modalities)
    overlap_summary = relation_overlap(edge_index, edge_type)
    adj_lists = build_adj_lists(edge_index, edge_type, num_nodes, relation_ids)
    jaccard_stats = modality_jaccard(adj_lists, relation_ids, modalities)

    summary = {
        "path": str(graph_path),
        "modalities": modalities,
        "relation_map": relation_map,
        "num_nodes_including_padding": num_nodes,
        "directed_edges": int(edge_index.shape[1]),
        "unique_undirected_pairs": len(undirected_pairs),
        "edge_weight": build_stats(edge_weight),
        "degree_stats": degree_stats,
        "top_nodes": top_total,
        "top_nodes_per_relation": rel_top,
        "modality_mix": mix_summary,
        "relation_overlap": overlap_summary,
        "modality_jaccard": jaccard_stats,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Graph path: {summary['path']}")
    print(f"Modalities: {summary['modalities']}")
    print(f"Relation map: {summary['relation_map']}")
    print(f"Nodes (incl. padding): {summary['num_nodes_including_padding']}")
    print(f"Directed edges: {summary['directed_edges']}")
    print(f"Unique undirected pairs: {summary['unique_undirected_pairs']}")
    ew = summary["edge_weight"]
    print("Edge weight stats:", {k: v for k, v in ew.items() if k != "threshold_counts"})
    print("Edge weight thresholds counts:", ew.get("threshold_counts", {}))

    print("Degree stats (total = key -1):")
    for rel, stats in summary["degree_stats"].items():
        label = "total" if rel == -1 else f"relation {rel}"
        thresh = stats.get("threshold_counts", {})
        share = stats.get("top1pct_edge_share", float("nan"))
        core_stats = {k: v for k, v in stats.items() if k not in {"threshold_counts", "top1pct_edge_share"}}
        print(f"  {label}: {core_stats}")
        if thresh:
            print(f"    counts >= thresholds {DEG_THRESHOLDS}: {thresh}")
        if not np.isnan(share):
            print(f"    top 1% nodes edge-share: {share*100:.2f}%")

    print(f"Top {args.topk} nodes by total degree:")
    for idx, deg in summary["top_nodes"]:
        print(f"  item {idx}: degree={deg}")

    for name, nodes in summary["top_nodes_per_relation"].items():
        print(f"  Relation '{name}' top nodes:")
        for node_id, deg in nodes:
            print(f"    item {node_id}: relation-degree={deg}")

    print("Modality mix summary:")
    mix = summary["modality_mix"]
    print("  mean share:", mix["mean_share"])
    print("  median share:", mix["median_share"])
    print("  zero-degree items:", mix["zero_degree_items"])
    for name, examples in mix["dominant_examples"].items():
        print(f"  nodes with >=90% edges from '{name}' (up to 10): {examples}")

    print("Relation overlap summary:")
    print("  overlap size counts:", summary["relation_overlap"]["overlap_size_counts"])
    combos = summary["relation_overlap"]["relation_combos"]
    if combos:
        print("  relation combo counts:")
        for combo, count in combos.items():
            print(f"    {combo}: {count}")

    if summary["modality_jaccard"]:
        print("Modality Jaccard (based on sampled nodes):")
        for pair, stats in summary["modality_jaccard"].items():
            print(f"  {pair}: {stats}")


if __name__ == "__main__":
    main()
