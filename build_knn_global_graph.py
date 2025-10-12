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
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Construct cosine KNN global graph')
    parser.add_argument('--dataset', required=True,
                        help='Dataset folder name under datasets/, e.g. Amazon_cellPhone_2018')
    parser.add_argument('--mode', choices=('single', 'unified'), default='single',
                        help='single: build one modality-specific graph; unified: merge multiple modalities into one multi-relation graph.')
    parser.add_argument('--modality', default='image',
                        help='Embedding modality prefix (matches <modality>_matrix.npy). Used in single mode.')
    parser.add_argument('--modalities', default='image,title,category',
                        help='Comma-separated list of modalities to include when --mode unified.')
    parser.add_argument('--topk', type=int, default=30,
                        help='Number of nearest neighbours (per node, directed).')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Batch size for similarity computation.')
    parser.add_argument('--output', default=None,
                        help='Optional explicit output path (.pkl). Defaults depend on mode.')
    parser.add_argument('--device', default=None,
                        help='Torch device to use (default: cuda if available else cpu).')
    parser.add_argument('--cap-out-per-rel', type=int, default=None,
                        help='(Unified mode) Maximum undirected degree per node per relation. High-weight edges kept first.')
    parser.add_argument('--cap-in-per-rel', type=int, default=None,
                        help='(Unified mode) Alias of cap-out (graphs are symmetrised); use the tighter bound if both provided.')
    parser.add_argument('--cap-total', type=int, default=None,
                        help='(Unified mode) Maximum undirected degree per node across all relations (after per-relation caps).')
    return parser.parse_args()


def load_embedding_matrix(root: Path, dataset: str, modality: str) -> np.ndarray:
    emb_path = root / 'datasets' / dataset / 'embeddings' / f'{modality}_matrix.npy'
    if not emb_path.exists():
        raise FileNotFoundError(f'Embedding matrix not found: {emb_path}')
    matrix = np.load(emb_path)
    if matrix.ndim != 2:
        raise ValueError(f'Expected 2D embedding matrix, got shape {matrix.shape}')
    return matrix.astype(np.float32)


def ensure_output_path(root: Path, dataset: str, mode: str, modality: str, explicit: str | None) -> Path:
    if explicit:
        out_path = Path(explicit)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    out_dir = root / 'datasets' / dataset / 'global_graph'
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == 'single':
        return out_dir / f'global_graph_{modality}_knn.pkl'
    return out_dir / 'global_graph_unified.pkl'


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


def build_single_graph(root: Path, dataset: str, modality: str, topk: int, chunk_size: int, device: str, output_path: Path) -> None:
    print(f'Loading embedding matrix for dataset={dataset}, modality={modality} ...')
    matrix = load_embedding_matrix(root, dataset, modality)
    num_nodes_including_padding, emb_dim = matrix.shape
    print(f'Embedding matrix shape: {matrix.shape} (padding row included)')

    print(f'Computing cosine KNN (topk={topk}, chunk_size={chunk_size}, device={device}) ...')
    src, dst, weight = compute_topk_cosine(matrix, topk=topk, chunk_size=chunk_size, device=device)

    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    weight_sym = np.concatenate([weight, weight])

    edge_index = np.stack([src_sym, dst_sym], axis=0)

    deg = np.bincount(src_sym, minlength=num_nodes_including_padding)
    real_deg = deg[1:]

    graph = {
        'edge_index': edge_index,
        'edge_weight': weight_sym,
        'x': matrix,
        'meta': {
            'dataset': dataset,
            'modality': modality,
            'topk': int(topk),
            'chunk_size': int(chunk_size),
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
    print(f'Average out-degree: {real_deg.mean():.2f}')


def _greedy_cap(edges: Sequence[Tuple[int, int, int, float]],
                num_nodes_including_padding: int,
                cap: int) -> Tuple[List[Tuple[int, int, int, float]], int]:
    if cap is None:
        return list(edges), 0
    if cap <= 0:
        return [], len(edges)
    deg = np.zeros(num_nodes_including_padding, dtype=np.int64)
    kept: List[Tuple[int, int, int, float]] = []
    dropped = 0
    for edge in sorted(edges, key=lambda x: x[3], reverse=True):
        i, j, rel, w = edge
        if deg[i] >= cap or deg[j] >= cap:
            dropped += 1
            continue
        kept.append(edge)
        deg[i] += 1
        deg[j] += 1
    return kept, dropped


def _apply_caps(edges: List[Tuple[int, int, int, float]],
                num_nodes_including_padding: int,
                cap_out: int | None,
                cap_in: int | None,
                cap_total: int | None) -> Tuple[List[Tuple[int, int, int, float]],
                                                Dict[int, int], int]:
    relation_groups: Dict[int, List[Tuple[int, int, int, float]]] = defaultdict(list)
    for edge in edges:
        relation_groups[edge[2]].append(edge)

    cap_rel = None
    if cap_out is not None and cap_in is not None:
        cap_rel = min(cap_out, cap_in)
    elif cap_out is not None:
        cap_rel = cap_out
    elif cap_in is not None:
        cap_rel = cap_in

    per_rel_dropped: Dict[int, int] = {}
    capped_edges: List[Tuple[int, int, int, float]] = []
    for rel, rel_edges in relation_groups.items():
        kept_rel, dropped = _greedy_cap(rel_edges, num_nodes_including_padding, cap_rel)
        per_rel_dropped[rel] = dropped
        capped_edges.extend(kept_rel)

    if cap_total is None:
        return capped_edges, per_rel_dropped, 0

    deg_total = np.zeros(num_nodes_including_padding, dtype=np.int64)
    kept_total: List[Tuple[int, int, int, float]] = []
    dropped_total = 0
    for edge in sorted(capped_edges, key=lambda x: x[3], reverse=True):
        i, j, rel, w = edge
        if deg_total[i] >= cap_total or deg_total[j] >= cap_total:
            dropped_total += 1
            continue
        kept_total.append(edge)
        deg_total[i] += 1
        deg_total[j] += 1

    return kept_total, per_rel_dropped, dropped_total


def _log_unified_stats(modality_order: List[str],
                       relation_map: Dict[str, int],
                       edges: List[Tuple[int, int, int, float]]) -> None:
    if not edges:
        print('[warn] Unified graph is empty after preprocessing.')
        return

    rel_counts: Dict[int, int] = defaultdict(int)
    rel_weights: Dict[int, List[float]] = defaultdict(list)
    rel_unique_pairs: Dict[int, set] = defaultdict(set)
    for i, j, rel, w in edges:
        rel_counts[rel] += 1
        rel_weights[rel].append(w)
        rel_unique_pairs[rel].add((i, j))

    print('Unified graph stats per relation:')
    for modality in modality_order:
        rel = relation_map[modality]
        count = rel_counts.get(rel, 0)
        weights = rel_weights.get(rel, [])
        unique_pairs = len(rel_unique_pairs.get(rel, set()))
        if not weights:
            print(f'  - {modality} (rel={rel}): 0 edges')
            continue
        arr = np.array(weights, dtype=np.float32)
        print(f'  - {modality} (rel={rel}): edges={count}'
              f', unique_pairs={unique_pairs}'
              f', weight[min/median/95p/max]={arr.min():.4f}/{np.median(arr):.4f}/{np.percentile(arr, 95):.4f}/{arr.max():.4f}')


def build_unified_graph(root: Path,
                        dataset: str,
                        modalities: Sequence[str],
                        topk: int,
                        chunk_size: int,
                        device: str,
                        output_path: Path,
                        cap_out_per_rel: int | None,
                        cap_in_per_rel: int | None,
                        cap_total: int | None) -> None:
    relation_map = {mod: idx for idx, mod in enumerate(modalities)}
    edges_dict: Dict[Tuple[int, int, int], float] = {}
    num_nodes_including_padding: int | None = None
    emb_dims: Dict[str, int] = {}

    for modality in modalities:
        print(f'Loading embedding matrix for modality={modality}')
        matrix = load_embedding_matrix(root, dataset, modality)
        if num_nodes_including_padding is None:
            num_nodes_including_padding = matrix.shape[0]
        elif matrix.shape[0] != num_nodes_including_padding:
            raise ValueError(f'Embedding matrix row count mismatch for modality={modality}: '
                             f'{matrix.shape[0]} vs expected {num_nodes_including_padding}')
        emb_dims[modality] = matrix.shape[1]

        print(f'Computing cosine KNN for modality={modality} (topk={topk})')
        src, dst, weight = compute_topk_cosine(matrix, topk=topk, chunk_size=chunk_size, device=device)
        rel_id = relation_map[modality]
        for s, d, w in zip(src, dst, weight):
            if s == d:
                continue
            i, j = (s, d) if s < d else (d, s)
            key = (i, j, rel_id)
            prev = edges_dict.get(key)
            if prev is None or w > prev:
                edges_dict[key] = float(w)

    if num_nodes_including_padding is None:
        raise RuntimeError('No modalities found to build unified graph.')

    undirected_edges = [(i, j, rel, weight) for (i, j, rel), weight in edges_dict.items()]
    undirected_edges.sort(key=lambda x: (x[2], x[0], x[1]))

    _log_unified_stats(list(modalities), relation_map, undirected_edges)

    capped_edges, per_rel_dropped, dropped_total = _apply_caps(
        undirected_edges,
        num_nodes_including_padding,
        cap_out_per_rel,
        cap_in_per_rel,
        cap_total,
    )

    if per_rel_dropped:
        for rel, dropped in per_rel_dropped.items():
            if dropped > 0:
                print(f'[cap] relation {rel}: dropped {dropped} edges due to per-relation cap.')
    if dropped_total > 0:
        print(f'[cap] total-degree cap dropped {dropped_total} edges.')

    if not capped_edges:
        raise RuntimeError('Unified graph is empty after applying caps.')

    edge_records = []
    for i, j, rel, weight in capped_edges:
        edge_records.append((i, j, rel, weight))
        edge_records.append((j, i, rel, weight))

    src = np.array([rec[0] for rec in edge_records], dtype=np.int64)
    dst = np.array([rec[1] for rec in edge_records], dtype=np.int64)
    edge_type = np.array([rec[2] for rec in edge_records], dtype=np.int64)
    edge_weight = np.array([rec[3] for rec in edge_records], dtype=np.float32)

    edge_index = np.stack([src, dst], axis=0)

    meta = {
        'dataset': dataset,
        'modalities': list(modalities),
        'relation_map': relation_map,
        'topk': int(topk),
        'chunk_size': int(chunk_size),
        'num_nodes_including_padding': int(num_nodes_including_padding),
        'embedding_dims': emb_dims,
        'caps': {
            'cap_out_per_rel': cap_out_per_rel,
            'cap_in_per_rel': cap_in_per_rel,
            'cap_total': cap_total,
        },
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    graph = {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'edge_weight': edge_weight,
        'meta': meta,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

    print(f'Saved unified global graph to: {output_path}')
    total_edges = edge_index.shape[1]
    print(f'total directed edges: {total_edges}')
    print(f'modalities: {modalities}')


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    modalities = [m.strip() for m in args.modalities.split(',') if m.strip()]

    if args.mode == 'single':
        output_path = ensure_output_path(root, args.dataset, args.mode, args.modality, args.output)
        build_single_graph(root, args.dataset, args.modality, args.topk, args.chunk_size, device, output_path)
    else:
        if not modalities:
            raise ValueError('No modalities provided for unified mode.')
        output_path = ensure_output_path(root, args.dataset, args.mode, args.modality, args.output)
        build_unified_graph(
            root=root,
            dataset=args.dataset,
            modalities=modalities,
            topk=args.topk,
            chunk_size=args.chunk_size,
            device=device,
            output_path=output_path,
            cap_out_per_rel=args.cap_out_per_rel,
            cap_in_per_rel=args.cap_in_per_rel,
            cap_total=args.cap_total,
        )


if __name__ == '__main__':
    main()
