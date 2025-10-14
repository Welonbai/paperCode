#!/usr/bin/env python3
"""
Inspect a single modality KNN graph produced by build_knn_global_graph.py.

Example:
    python inspect_modality_knn.py --dataset Amazon_cellPhone_2018 --modality image
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

DEG_THRESHOLDS = (20, 40, 80, 160, 320)


def load_pickle(path: Path):
    import pickle
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


def threshold_counts(values: np.ndarray) -> Dict[int, int]:
    return {thr: int((values >= thr).sum()) for thr in DEG_THRESHOLDS}


def main():
    parser = argparse.ArgumentParser(description="Inspect a per-modality KNN graph pickle.")
    parser.add_argument("--dataset", required=True, help="Dataset folder under datasets/")
    parser.add_argument("--modality", required=True, help="Modality name (image/title/category)")
    parser.add_argument("--path", default="", help="Optional explicit path to pickle.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    parser.add_argument("--topk", type=int, default=10, help="Top-K items by degree to display.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    default_path = base / "datasets" / args.dataset / "global_graph" / f"global_graph_{args.modality}_knn.pkl"
    graph_path = Path(args.path) if args.path else default_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Modality graph not found: {graph_path}")

    graph = load_pickle(graph_path)
    edge_index = np.asarray(graph["edge_index"])
    edge_weight = np.asarray(graph["edge_weight"])
    meta = graph.get("meta", {})
    num_nodes = int(meta.get("num_nodes_including_padding", edge_index.max() + 1))

    deg_total = np.bincount(edge_index[0], minlength=num_nodes)[1:]
    weight_stats = stats(edge_weight)
    degree_stats = stats(deg_total)
    degree_threshold = threshold_counts(deg_total)

    top_pairs = sorted(((idx + 1, int(deg_total[idx])) for idx in range(deg_total.size)),
                       key=lambda x: x[1], reverse=True)[:args.topk]

    summary = {
        "path": str(graph_path),
        "modality": args.modality,
        "num_nodes_including_padding": num_nodes,
        "directed_edges": int(edge_index.shape[1]),
        "edge_weight_stats": weight_stats,
        "degree_stats": degree_stats,
        "degree_threshold_counts": degree_threshold,
        "top_degree_nodes": top_pairs,
        "meta": meta,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Graph path: {summary['path']}")
    print(f"Modality: {summary['modality']}")
    print(f"Nodes (incl. padding): {summary['num_nodes_including_padding']}")
    print(f"Directed edges: {summary['directed_edges']}")
    print("Edge weight stats:", summary["edge_weight_stats"])
    print("Degree stats:", summary["degree_stats"])
    print(f"Degree threshold counts >= {DEG_THRESHOLDS}: {summary['degree_threshold_counts']}")
    print(f"Top {args.topk} nodes by degree:")
    for node_id, deg in summary["top_degree_nodes"]:
        print(f"  item {node_id}: degree={deg}")


if __name__ == "__main__":
    main()
