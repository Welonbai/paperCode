#!/usr/bin/env python3
'''Build a cosine KNN global graph from precomputed item embeddings.

The script is dataset-agnostic: it expects padded embedding matrices saved by
`datasets/<dataset>/embeddings`.
'''

import argparse
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Construct cosine KNN global graph')
    parser.add_argument('--dataset', required=True,
                        help='Dataset folder name under datasets/, e.g. Amazon_cellPhone_2018')
    parser.add_argument('--modality', default='image',
                        help='Embedding modality prefix (matches <modality>_matrix.npy).')
    parser.add_argument('--topk', type=int, default=30,
                        help='Number of nearest neighbours (per node, directed).')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Batch size for similarity computation.')
    parser.add_argument('--output', default=None,
                        help='Optional explicit output path (.pkl). Defaults to datasets/<dataset>/global_graph/global_graph_<modality>_knn.pkl')
    parser.add_argument('--device', default=None,
                        help='Torch device to use (default: cuda if available else cpu).')
    return parser.parse_args()


def load_embedding_matrix(root: Path, dataset: str, modality: str) -> np.ndarray:
    emb_path = root / 'datasets' / dataset / 'embeddings' / f'{modality}_matrix.npy'
    if not emb_path.exists():
        raise FileNotFoundError(f'Embedding matrix not found: {emb_path}')
    matrix = np.load(emb_path)
    if matrix.ndim != 2:
        raise ValueError(f'Expected 2D embedding matrix, got shape {matrix.shape}')
    return matrix.astype(np.float32)


def ensure_output_path(root: Path, dataset: str, modality: str, explicit: str | None) -> Path:
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


def build_graph(root: Path, dataset: str, modality: str, topk: int, chunk_size: int, device: str, output_path: Path) -> None:
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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = ensure_output_path(root, args.dataset, args.modality, args.output)
    build_graph(root, args.dataset, args.modality, args.topk, args.chunk_size, device, output_path)


if __name__ == '__main__':
    main()