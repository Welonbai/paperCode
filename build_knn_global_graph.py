#!/usr/bin/env python3
'''Build a cosine KNN global graph from precomputed item embeddings.

The script is dataset-agnostic: it expects padded embedding matrices saved by
`datasets/<dataset>/embeddings`.
'''

import argparse
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_FINAL_K = {
    "image": 40,
    "title": 40,
    "category": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Construct refined cosine KNN graphs or build a unified multi-relation graph.')
    parser.add_argument('--mode', choices=('single', 'unified'), default='single',
                        help='single: build one modality graph; unified: merge existing modality graphs.')
    parser.add_argument('--dataset', required=True,
                        help='Dataset folder name under datasets/, e.g. Amazon_cellPhone_2018')
    parser.add_argument('--modality', default='image',
                        help='(single mode) Embedding modality prefix (matches <modality>_matrix.npy).')
    parser.add_argument('--k-raw', type=int, default=60,
                        help='Raw K used for initial cosine search (before filtering).')
    parser.add_argument('--final-k', type=int, default=None,
                        help='Final neighbour budget per node after refinement. Defaults depend on modality.')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Batch size for similarity computation.')
    parser.add_argument('--output', default=None,
                        help='Optional explicit output path (.pkl). Defaults to datasets/<dataset>/global_graph/global_graph_<modality>_knn.pkl')
    parser.add_argument('--device', default=None,
                        help='Torch device to use (default: cuda if available else cpu).')
    parser.add_argument('--disable-mutual', action='store_true',
                        help='Skip mutual-K filtering (keeps one-sided edges).')
    parser.add_argument('--snn-alpha', type=float, default=0.5,
                        help='Blend weight between cosine and shared-neighbour metric (0..1).')
    parser.add_argument('--snn-metric', choices=('jaccard', 'snn'), default='jaccard',
                        help='Shared-neighbour metric to blend with cosine.')
    parser.add_argument('--disable-local-scale', action='store_true',
                        help='Disable local scaling based on adaptive kernels.')
    parser.add_argument('--local-k', type=int, default=10,
                        help='Neighbour index used for local scaling bandwidth.')
    parser.add_argument('--local-beta', type=float, default=0.5,
                        help='Blend weight between cosine/SNN score and local scaling score.')
    parser.add_argument('--hub-penalty-exp', type=float, default=0.5,
                        help='Exponent used to down-weight neighbours with large raw degrees (only applied to category by default).')
    parser.add_argument('--disable-degree-norm', action='store_true',
                        help='Skip final degree normalisation of edge weights.')
    parser.add_argument('--disable-weight-rescale', action='store_true',
                        help='Skip min-max rescaling of final weights to [0,1].')
    parser.add_argument('--min-weight', type=float, default=0.0,
                        help='Drop edges whose refined weight falls below this threshold.')
    parser.add_argument('--cat-min-degree', type=int, default=10,
                        help='(single mode, category only) Ensure each node has at least this many category neighbours (after backfill).')
    parser.add_argument('--modalities', default='image,title,category',
                        help='(unified mode) Comma-separated modality list to merge.')
    parser.add_argument('--cap-out-per-rel', type=int, default=None,
                        help='(unified mode) Maximum undirected degree per relation.')
    parser.add_argument('--cap-total', type=int, default=None,
                        help='(unified mode) Maximum undirected degree overall.')
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embedding_matrix(root: Path, dataset: str, modality: str) -> np.ndarray:
    emb_path = root / 'datasets' / dataset / 'embeddings' / f'{modality}_matrix.npy'
    if not emb_path.exists():
        raise FileNotFoundError(f'Embedding matrix not found: {emb_path}')
    matrix = np.load(emb_path)
    if matrix.ndim != 2:
        raise ValueError(f'Expected 2D embedding matrix, got shape {matrix.shape}')
    return matrix.astype(np.float32)


def ensure_output_path(root: Path, dataset: str, modality: str, explicit: Optional[str]) -> Path:
    if explicit:
        out_path = Path(explicit)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    out_dir = root / 'datasets' / dataset / 'global_graph'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'global_graph_{modality}_knn.pkl'


def compute_topk_cosine(matrix: np.ndarray, topk: int, chunk_size: int, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Return (src_ids, dst_ids, weights) for directed KNN edges.'''
    if matrix.shape[0] <= 1:
        raise ValueError('Embedding matrix should include at least one non-padding row.')

    feats = torch.from_numpy(matrix).to(device=device)
    feats = feats.float()
    feats[0].zero_()

    real = feats[1:]
    real = F.normalize(real, p=2, dim=1)

    num_nodes = real.size(0)
    all_src: list[np.ndarray] = []
    all_dst: list[np.ndarray] = []
    all_w: list[np.ndarray] = []

    total_chunks = math.ceil(num_nodes / chunk_size)
    reference = real

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_nodes)
        block = real[start:end]
        sims = block @ reference.t()

        if end - start == 0:
            continue
        row_idx = torch.arange(end - start, device=device)
        self_cols = torch.arange(start, end, device=device)
        sims[row_idx, self_cols] = -float('inf')

        k = min(topk, num_nodes - 1)
        if k <= 0:
            raise ValueError('topk must be >= 1 when there is more than one node.')
        vals, idx = torch.topk(sims, k=k, dim=1)

        src_ids = torch.arange(start, end, device=device).unsqueeze(1).expand_as(idx)

        all_src.append(src_ids.cpu().numpy() + 1)
        all_dst.append(idx.cpu().numpy() + 1)
        all_w.append(vals.cpu().numpy())

    src = np.concatenate(all_src, axis=0).reshape(-1)
    dst = np.concatenate(all_dst, axis=0).reshape(-1)
    weight = np.concatenate(all_w, axis=0).reshape(-1)

    return src.astype(np.int64), dst.astype(np.int64), weight.astype(np.float32)


def compute_sigma(distance_lists: Dict[int, List[float]], num_nodes: int, local_k: int) -> np.ndarray:
    all_distances = [d for lst in distance_lists.values() for d in lst]
    default = float(np.median(all_distances)) if all_distances else 1.0
    sigma = np.full(num_nodes, default, dtype=np.float32)
    for node, dists in distance_lists.items():
        if not dists:
            continue
        sorted_d = sorted(dists)
        idx = min(local_k - 1, len(sorted_d) - 1)
        sigma[node] = max(sorted_d[idx], 1e-4)
    sigma[0] = 1.0
    return sigma


def shared_neighbor_score(set_a: set, set_b: set, metric: str) -> float:
    if metric == 'snn':
        denom = max(1, min(len(set_a), len(set_b)))
        return len(set_a & set_b) / denom
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def apply_pipeline(
    num_nodes: int,
    modality: str,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    raw_neighbors: Dict[int, set],
    raw_distances: Dict[int, List[float]],
    args: argparse.Namespace,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, Dict[int, float]]]:
    mutual_required = not args.disable_mutual
    if modality == "category":
        mutual_required = False
    directed = {(int(s), int(d)): float(w) for s, d, w in zip(src, dst, weight)}
    edge_pairs: Dict[Tuple[int, int], Dict[str, float]] = {}
    candidate_scores: Dict[int, Dict[int, float]] = defaultdict(dict)

    for (s, d), w_sd in directed.items():
        if s == d or s == 0 or d == 0:
            continue
        has_reverse = (d, s) in directed
        if mutual_required and not has_reverse:
            cos_sim = w_sd
        elif has_reverse:
            cos_sim = max(w_sd, directed[(d, s)])
        else:
            cos_sim = w_sd

        snn_score = shared_neighbor_score(raw_neighbors[s], raw_neighbors[d], args.snn_metric)
        blended = (1.0 - args.snn_alpha) * cos_sim + args.snn_alpha * snn_score

        sigma = None
        if not args.disable_local_scale:
            # compute on demand later; placeholder
            sigma = True  # mark to compute later

        # store candidate score (will adjust later if local scaling needed)
        candidate_scores[s][d] = max(candidate_scores[s].get(d, 0.0), blended)
        candidate_scores[d][s] = max(candidate_scores[d].get(s, 0.0), blended)

    sigma_values = compute_sigma(raw_distances, num_nodes, args.local_k) if not args.disable_local_scale else None

    for (s, d), _ in directed.items():
        if s == d or s == 0 or d == 0:
            continue
        if args.disable_local_scale:
            final_score = candidate_scores[s][d]
        else:
            cos_sim = max(directed[(s, d)], directed.get((d, s), directed[(s, d)]))
            dist = max(0.0, 1.0 - cos_sim)
            loc = math.exp(- (dist ** 2) / (sigma_values[s] * sigma_values[d] + 1e-8))
            snn_blended = candidate_scores[s][d]
            final_score = args.local_beta * snn_blended + (1.0 - args.local_beta) * loc
            candidate_scores[s][d] = final_score
            candidate_scores[d][s] = final_score

    for (s, d), score in directed.items():
        if s == d or s == 0 or d == 0:
            continue
        if mutual_required and (d, s) not in directed:
            continue
        u, v = (s, d) if s < d else (d, s)

        blended = candidate_scores[s][d]

        if modality == "category" and args.hub_penalty_exp > 0:
            deg_u = max(1, len(raw_neighbors[u]))
            deg_v = max(1, len(raw_neighbors[v]))
            penalty = (deg_u ** (-args.hub_penalty_exp)) * (deg_v ** (-args.hub_penalty_exp))
            blended *= penalty

        existing = edge_pairs.get((u, v))
        if existing is None or blended > existing["weight"]:
            edge_pairs[(u, v)] = {"weight": blended}

    # Drop edges below threshold
    edge_pairs = {k: v["weight"] for k, v in edge_pairs.items() if v["weight"] >= args.min_weight}
    return edge_pairs, candidate_scores


def select_topk_per_node(
    num_nodes: int,
    edge_pairs: Dict[Tuple[int, int], float],
    final_k: int,
) -> Dict[Tuple[int, int], float]:
    if final_k is None or final_k <= 0:
        return dict(edge_pairs)
    degree = [0] * num_nodes
    selected: Dict[Tuple[int, int], float] = {}
    for (u, v), w in sorted(edge_pairs.items(), key=lambda item: item[1], reverse=True):
        if degree[u] >= final_k or degree[v] >= final_k:
            continue
        selected[(u, v)] = w
        degree[u] += 1
        degree[v] += 1
    return selected


def degree_normalize(edge_pairs: Dict[Tuple[int, int], float], num_nodes: int) -> Dict[Tuple[int, int], float]:
    deg = np.zeros(num_nodes, dtype=np.int64)
    for (u, v) in edge_pairs.keys():
        deg[u] += 1
        deg[v] += 1
    normalized = {}
    for (u, v), w in edge_pairs.items():
        norm = math.sqrt(max(1, deg[u]) * max(1, deg[v]))
        normalized[(u, v)] = w / norm if norm > 0 else w
    return normalized


def rescale_weights(edge_pairs: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not edge_pairs:
        return edge_pairs
    values = np.array(list(edge_pairs.values()), dtype=np.float32)
    w_min = float(values.min())
    w_max = float(values.max())
    if w_max - w_min < 1e-9:
        return {k: 1.0 for k in edge_pairs.keys()}
    return {k: (w - w_min) / (w_max - w_min) for k, w in edge_pairs.items()}


def ensure_category_min_degree(edge_pairs: Dict[Tuple[int, int], float],
                               candidate_scores: Dict[int, Dict[int, float]],
                               num_nodes: int,
                               min_degree: int,
                               final_k: int) -> Dict[Tuple[int, int], float]:
    if min_degree <= 0:
        return edge_pairs
    current = defaultdict(int)
    for (u, v) in edge_pairs.keys():
        current[u] += 1
        current[v] += 1

    updated = dict(edge_pairs)

    for node in range(1, num_nodes):
        needed = min_degree - current.get(node, 0)
        if needed <= 0:
            continue
        candidates = candidate_scores.get(node, {})
        if not candidates:
            continue
        sorted_candidates = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        for neighbor, score in sorted_candidates:
            if neighbor == node:
                continue
            if current.get(node, 0) >= final_k:
                break
            if current.get(neighbor, 0) >= final_k:
                continue
            key = (node, neighbor) if node < neighbor else (neighbor, node)
            if key in updated:
                continue
            updated[key] = score + 1.0  # boost to keep during selection
            current[node] = current.get(node, 0) + 1
            current[neighbor] = current.get(neighbor, 0) + 1
            needed -= 1
            if needed <= 0:
                break
    return updated


def _apply_caps(edges: List[Tuple[int, int, int, float]], num_nodes_including_padding: int,
                cap_per_rel: Optional[int], cap_total: Optional[int]) -> Tuple[List[Tuple[int, int, int, float]], Dict[int, int], int]:
    if cap_per_rel is None and cap_total is None:
        return edges, {}, 0

    grouping: Dict[int, List[Tuple[int, int, int, float]]] = defaultdict(list)
    for edge in edges:
        grouping[edge[2]].append(edge)

    per_rel_dropped: Dict[int, int] = {}
    kept_edges: List[Tuple[int, int, int, float]] = []

    def greedy_cap(edge_list: List[Tuple[int, int, int, float]], cap: Optional[int]) -> Tuple[List[Tuple[int, int, int, float]], int]:
        if cap is None:
            return edge_list, 0
        deg = np.zeros(num_nodes_including_padding, dtype=np.int64)
        kept: List[Tuple[int, int, int, float]] = []
        dropped = 0
        for edge in sorted(edge_list, key=lambda x: x[3], reverse=True):
            u, v = edge[0], edge[1]
            if deg[u] >= cap or deg[v] >= cap:
                dropped += 1
                continue
            kept.append(edge)
            deg[u] += 1
            deg[v] += 1
        return kept, dropped

    for rel, elist in grouping.items():
        kept_rel, dropped = greedy_cap(elist, cap_per_rel)
        per_rel_dropped[rel] = dropped
        kept_edges.extend(kept_rel)

    if cap_total is None:
        return kept_edges, per_rel_dropped, 0

    deg_total = np.zeros(num_nodes_including_padding, dtype=np.int64)
    final_edges: List[Tuple[int, int, int, float]] = []
    dropped_total = 0
    for edge in sorted(kept_edges, key=lambda x: x[3], reverse=True):
        u, v = edge[0], edge[1]
        if deg_total[u] >= cap_total or deg_total[v] >= cap_total:
            dropped_total += 1
            continue
        final_edges.append(edge)
        deg_total[u] += 1
        deg_total[v] += 1

    return final_edges, per_rel_dropped, dropped_total


def _log_unified_stats(modalities: List[str], relation_map: Dict[str, int], edges: List[Tuple[int, int, int, float]]) -> None:
    counts = defaultdict(int)
    weights = defaultdict(list)
    for u, v, rel, w in edges:
        counts[rel] += 1
        weights[rel].append(w)
    print("Unified graph stats before capping:")
    for name in modalities:
        rel = relation_map.get(name)
        if rel is None:
            continue
        w_list = weights.get(rel, [])
        if not w_list:
            print(f"  {name}: 0 edges")
            continue
        arr = np.array(w_list, dtype=np.float32)
        print(f"  {name}: edges={counts[rel]}, weight[min/median/p95/max]={arr.min():.4f}/{np.median(arr):.4f}/{np.percentile(arr,95):.4f}/{arr.max():.4f}")


def build_unified_graph(
    root: Path,
    dataset: str,
    modalities: Sequence[str],
    cap_per_rel: Optional[int],
    cap_total: Optional[int],
    output_path: Path,
) -> None:
    relation_map = {mod: idx for idx, mod in enumerate(modalities)}
    combined: Dict[Tuple[int, int, int], float] = {}
    num_nodes_with_pad: Optional[int] = None
    embedding_dims: Dict[str, int] = {}

    for mod in modalities:
        base_dir = root / 'datasets' / dataset / 'global_graph'
        candidates = [
            base_dir / f'global_graph_{mod}_knn.pkl',
            base_dir / f'global_graph_{mod}.pkl',
        ]
        # Special-case shared naming if mod already includes suffix
        if not mod.endswith("_knn"):
            candidates.append(base_dir / f'global_graph_{mod}_shared.pkl')
        graph_path = None
        for cand in candidates:
            if cand.exists():
                graph_path = cand
                break
        if graph_path is None:
            raise FileNotFoundError(f"Modality graph not found for '{mod}'. Looked in: {candidates}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Modality graph not found for '{mod}': {graph_path}")
        graph = load_pickle(graph_path)
        edge_index = np.asarray(graph['edge_index'])
        edge_weight = np.asarray(graph['edge_weight'])
        meta = graph.get('meta', {})
        nodes = meta.get('num_nodes_including_padding', edge_index.max() + 1)
        if num_nodes_with_pad is None:
            num_nodes_with_pad = int(nodes)
        elif nodes != num_nodes_with_pad:
            raise ValueError(f"Modality '{mod}' has {nodes} nodes, expected {num_nodes_with_pad}")
        embedding_dims[mod] = meta.get('embedding_dim', 0)

        rel_id = relation_map[mod]
        for s, d, w in zip(edge_index[0], edge_index[1], edge_weight):
            if s == 0 or d == 0:
                continue
            u, v = (int(s), int(d)) if s < d else (int(d), int(s))
            key = (u, v, rel_id)
            if key not in combined or w > combined[key]:
                combined[key] = float(w)

    if num_nodes_with_pad is None:
        raise RuntimeError("No modality graphs loaded.")

    undirected_edges = [(u, v, rel, w) for (u, v, rel), w in combined.items()]
    _log_unified_stats(list(modalities), relation_map, undirected_edges)

    capped_edges, per_rel_dropped, dropped_total = _apply_caps(
        undirected_edges,
        num_nodes_with_pad,
        cap_per_rel,
        cap_total,
    )
    for rel, dropped in per_rel_dropped.items():
        if dropped > 0:
            print(f"[cap] relation {rel}: dropped {dropped} edges.")
    if dropped_total > 0:
        print(f"[cap] total-degree cap dropped {dropped_total} edges.")

    src = []
    dst = []
    weights = []
    edge_type = []
    for u, v, rel, w in capped_edges:
        src.extend([u, v])
        dst.extend([v, u])
        edge_type.extend([rel, rel])
        weights.extend([w, w])

    edge_index = np.stack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)], axis=0)
    edge_weight = np.array(weights, dtype=np.float32)
    edge_type = np.array(edge_type, dtype=np.int64)

    graph = {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'edge_type': edge_type,
        'meta': {
            'dataset': dataset,
            'modalities': list(modalities),
            'relation_map': relation_map,
            'num_nodes_including_padding': int(num_nodes_with_pad),
            'embedding_dims': embedding_dims,
            'caps': {
                'cap_per_rel': cap_per_rel,
                'cap_total': cap_total,
            },
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

    print(f"Saved unified graph to {output_path}")


def build_graph(root: Path, dataset: str, modality: str, args: argparse.Namespace, device: str, output_path: Path) -> None:
    print(f'Loading embedding matrix for dataset={dataset}, modality={modality} ...')
    matrix = load_embedding_matrix(root, dataset, modality)
    num_nodes_including_padding, emb_dim = matrix.shape
    print(f'Embedding matrix shape: {matrix.shape} (padding row included)')

    print(f'Computing cosine KNN (k_raw={args.k_raw}, chunk_size={args.chunk_size}, device={device}) ...')
    src, dst, weight = compute_topk_cosine(matrix, topk=args.k_raw, chunk_size=args.chunk_size, device=device)

    # Build raw neighbour sets and distance lists (1 - cosine)
    raw_neighbors: Dict[int, set] = defaultdict(set)
    raw_distances: Dict[int, List[float]] = defaultdict(list)
    for s, d, w in zip(src, dst, weight):
        raw_neighbors[int(s)].add(int(d))
        raw_distances[int(s)].append(float(1.0 - w))

    edge_pairs, candidate_scores = apply_pipeline(
        num_nodes_including_padding,
        modality,
        src,
        dst,
        weight,
        raw_neighbors,
        raw_distances,
        args,
    )

    final_k = args.final_k if args.final_k is not None else DEFAULT_FINAL_K.get(modality, 40)

    if modality == "category" and args.cat_min_degree > 0:
        edge_pairs = ensure_category_min_degree(
            edge_pairs,
            candidate_scores,
            num_nodes_including_padding,
            min_degree=args.cat_min_degree,
            final_k=final_k,
        )

    selected_pairs = select_topk_per_node(num_nodes_including_padding, edge_pairs, final_k)

    if not args.disable_degree_norm:
        selected_pairs = degree_normalize(selected_pairs, num_nodes_including_padding)

    if not args.disable_weight_rescale:
        selected_pairs = rescale_weights(selected_pairs)

    if not selected_pairs:
        raise RuntimeError("No edges remain after refinement. Check configuration.")

    src_sym: List[int] = []
    dst_sym: List[int] = []
    weight_sym: List[float] = []
    for (u, v), w in selected_pairs.items():
        src_sym.extend([u, v])
        dst_sym.extend([v, u])
        weight_sym.extend([w, w])

    edge_index = np.stack([np.array(src_sym, dtype=np.int64), np.array(dst_sym, dtype=np.int64)], axis=0)
    edge_weight = np.array(weight_sym, dtype=np.float32)

    deg = np.bincount(edge_index[0], minlength=num_nodes_including_padding)
    real_deg = deg[1:]

    graph = {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'x': matrix,
        'meta': {
            'dataset': dataset,
            'modality': modality,
            'k_raw': int(args.k_raw),
            'final_k': int(final_k),
            'mutual': not args.disable_mutual,
            'snn_alpha': float(args.snn_alpha),
            'snn_metric': args.snn_metric,
            'local_scaling': not args.disable_local_scale,
            'local_k': int(args.local_k),
            'local_beta': float(args.local_beta),
            'hub_penalty_exp': float(args.hub_penalty_exp) if modality == "category" else 0.0,
            'degree_norm': not args.disable_degree_norm,
            'weight_rescale': not args.disable_weight_rescale,
            'min_weight': float(args.min_weight),
            'num_nodes_including_padding': int(num_nodes_including_padding),
            'embedding_dim': int(emb_dim),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

    print(f'\nSaved global graph to: {output_path}')
    print(f'Edge count (directed): {edge_index.shape[1]}')
    print(f'Min/median/max degree (excluding padding): {real_deg.min()} / {np.median(real_deg):.1f} / {real_deg.max()}')
    print(f'Average degree: {real_deg.mean():.2f}')


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == 'single':
        output_path = ensure_output_path(root, args.dataset, args.modality, args.output)
        build_graph(root, args.dataset, args.modality, args, device, output_path)
    else:
        modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]
        if not modalities:
            raise ValueError("No modalities provided for unified mode.")
        output_path = Path(args.output) if args.output else root / 'datasets' / args.dataset / 'global_graph' / 'global_graph_unified.pkl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_unified_graph(
            root=root,
            dataset=args.dataset,
            modalities=modalities,
            cap_per_rel=args.cap_out_per_rel,
            cap_total=args.cap_total,
            output_path=output_path,
        )


if __name__ == '__main__':
    main()
