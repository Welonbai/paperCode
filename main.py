import time
import argparse
import pickle
import os
import numpy as np
from typing import Dict, Any, List
from model import *
from utils import *

def load_pickle_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_modality_embedding_matrix(dataset: str, modality: str) -> np.ndarray:
    emb_path = os.path.join('datasets', dataset, 'embeddings', f'{modality}_matrix.npy')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f'Embedding matrix not found for modality={modality}: {emb_path}')
    matrix = np.load(emb_path)
    if matrix.ndim != 2:
        raise ValueError(f'Embedding matrix for modality={modality} expected 2D, got shape {matrix.shape}')
    return matrix.astype(np.float32)

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Debug helpers
# -----------------------------
def debug_check_global_graph(graphs, num_node):
    print("===== DBG[global graph] =====")
    if not graphs:
        print("graphs=None (no global graphs loaded)")
        print("=============================")
        return
    def to_numpy(arr):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(arr, "detach"):
            arr = arr.detach()
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.asarray(arr)
    for idx, graph in enumerate(graphs):
        tag = graph.get("tag", f"graph{idx}")
        x = to_numpy(graph.get("features"))
        edge_index = to_numpy(graph.get("edge_index"))
        edge_weight = to_numpy(graph.get("edge_weight"))
        print(f"-- [{idx}] tag={tag}")
        if x is None or edge_index is None:
            print("   (missing features or edge_index)")
            continue
        print(f"   x.shape={x.shape} dtype={x.dtype}")
        zero_row_l2 = float(np.linalg.norm(x[0])) if x.shape[0] > 0 else float("nan")
        print(f"   x[0] L2 norm: {zero_row_l2:.6f}")
        rows_ok = (x.shape[0] == num_node + 1)
        print(f"   rows == num_node+1? {rows_ok}  ({x.shape[0]} vs {num_node+1})")
        if not np.isfinite(x).all():
            bad = x.size - int(np.isfinite(x).sum())
            print(f"   [warn] x contains {bad} non-finite values")
        print(f"   edge_index.shape={edge_index.shape}")
        if edge_index.size > 0:
            vmin, vmax = int(edge_index.min()), int(edge_index.max())
            print(f"   edge_index id range: [{vmin}, {vmax}] (expected within 0..{num_node})")
            if vmin < 0 or vmax > num_node:
                print("   [warn] edge_index contains ids outside [0, num_node]")
            deg = np.bincount(edge_index[0], minlength=num_node + 1)
            if deg.size > 1:
                d = deg[1:]
                print(f"   degree stats over real nodes: min={int(d.min())}, med={float(np.median(d)):.1f}, max={int(d.max())}")
        if edge_weight is not None and edge_weight.size > 0:
            print(f"   edge_weight stats: min={float(edge_weight.min()):.4f}, mean={float(edge_weight.mean()):.4f}, max={float(edge_weight.max()):.4f}")
    print("=============================")


def debug_check_unified_graph(unified_graph, num_node):
    if not unified_graph:
        print("[debug] unified graph payload empty.")
        return
    print("===== DBG[unified global graph] =====")
    edge_index = unified_graph.get("edge_index")
    edge_type = unified_graph.get("edge_type")
    edge_weight = unified_graph.get("edge_weight")
    meta = unified_graph.get("meta", {})
    relation_map = unified_graph.get("relation_map", {})
    modalities = unified_graph.get("modalities", [])
    arr_index = np.asarray(edge_index)
    arr_type = np.asarray(edge_type)
    arr_weight = np.asarray(edge_weight)
    print(f"edge_index.shape={arr_index.shape}")
    print(f"edge_type.shape={arr_type.shape}, edge_weight.shape={arr_weight.shape}")
    if arr_index.size > 0:
        vmin, vmax = int(arr_index.min()), int(arr_index.max())
        print(f"node id range in edges: [{vmin}, {vmax}] (expected ≤ {num_node})")
    rel_counts = {}
    for rel in np.unique(arr_type):
        rel_counts[int(rel)] = int((arr_type == rel).sum())
    print(f"relation edge counts: {rel_counts}")
    print(f"modalities: {modalities}")
    print(f"relation_map: {relation_map}")
    if arr_weight.size > 0:
        print(f"edge_weight stats min/median/max: {float(arr_weight.min()):.4f}/{float(np.median(arr_weight)):.4f}/{float(arr_weight.max()):.4f}")
    print("=============================")
def debug_check_dataset(train_data, test_data, num_node):
    print("===== DBG[data] =====")
    max_seq_len_train = max((len(s) for s in train_data.inputs if len(s) > 0), default=0)
    max_seq_len_test  = max((len(s) for s in test_data.inputs if len(s) > 0), default=0)
    print(f"max session len  train/test: {max_seq_len_train} / {max_seq_len_test} (pos_embedding size=200)")
    if max_seq_len_train > 200 or max_seq_len_test > 200:
        print("[warn] session length exceeds 200; position embedding is capped at 200")

    max_train_id = max([max(seq) for seq in train_data.inputs if len(seq) > 0])
    max_test_id  = max([max(seq) for seq in test_data.inputs if len(seq) > 0])
    print(f"max item id   train/test: {max_train_id} / {max_test_id}, num_node={num_node}")
    if max(max_train_id, max_test_id) >= num_node:
        print("[warn] some item ids exceed num_node (embedding will index OOB)")
    print("=====================")

def debug_peek_one_batch(opt, model, dataset, note="train"):
    print(f"===== DBG[one batch: {note}] =====")
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=model.batch_size,
                                         shuffle=True, pin_memory=True)
    batch = next(iter(loader))
    alias_inputs, local_adj, seq_items, seq_mask, targets, seq_item_ids = batch

    # basic stats
    print(f"seq_item_ids range: {int(seq_item_ids.min())}..{int(seq_item_ids.max())}")
    print(f"targets range:      {int(targets.min())}..{int(targets.max())}")
    print(f"targets==0 count:   {int((targets==0).sum())}  (should be 0 if 1-based)")

    uniq_adj = torch.unique(local_adj)
    if uniq_adj.numel() > 20:
        uniq_adj = uniq_adj[:20]
    print(f"unique adj values (truncated): {uniq_adj.tolist()}")

    # forward once
    model.eval()
    with torch.no_grad():
        t, scores = forward(model, batch)
        nan_ratio = float(torch.isnan(scores).float().mean().item()) * 100.0
        print(f"scores stats: nan%={nan_ratio:.4f}  min={scores.min().item():.4f}  "
              f"mean={scores.mean().item():.4f}  max={scores.max().item():.4f}")
        # quick top-1 on this batch
        top1 = scores.argmax(dim=1).cpu().numpy()
        acc1 = (top1 == (t.cpu().numpy() - 1)).mean()
        print(f"top1 accuracy on this batch: {acc1*100:.2f}%")
        if getattr(model, "global_mode", None) == 'unified':
            gate_mean = getattr(model.unified_encoder, "latest_gate_mean", None)
            if gate_mean is not None:
                modal_names = getattr(model.unified_encoder, "modalities", [])
                gate_vals = gate_mean.tolist()
                paired = ", ".join(f"{name}:{val:.3f}" for name, val in zip(modal_names, gate_vals))
                print(f"mean gate weights -> {paired}")
    print("==================================")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--global_graph_mods', type=str, default='image',
                    help='(legacy mode only) Modality tag for global graph filename, e.g. "image", "image+category". '
                         'Set empty string to disable loading.')
parser.add_argument('--disable_global_fusion', action='store_true',
                    help='Ignore global graph even if found (use local/session branch only).')
parser.add_argument('--global_graph_mode', choices=('legacy', 'unified'), default='legacy',
                    help='legacy: load per-modality branches (current behaviour); unified: load relation-aware unified graph.')
parser.add_argument('--unified_graph_path', type=str, default='',
                    help='Optional explicit path to unified graph pickle. Defaults to datasets/<dataset>/global_graph/global_graph_unified.pkl')
parser.add_argument('--node_embedding_fuse', choices=('avg', 'gate'), default='avg',
                    help='(unified mode) How to combine modality projections into the base node embedding: avg or gate.')
parser.add_argument('--debug_sanity', action='store_true',
                    help='Print extra diagnostics (targets, id ranges, NaNs, degree stats, zero-row check).')


opt = parser.parse_args()


def main():
    init_seed(2020)

    if opt.global_graph_mode != 'unified' and opt.node_embedding_fuse != 'avg':
        print("[warn] --node_embedding_fuse only applies in unified mode. Falling back to 'avg'.")
        opt.node_embedding_fuse = 'avg'

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'Amazon_grocery_2018':
        num_node = 11857
        opt.n_iter = 1
        opt.dropout_gcn = 0.2     # recommended values (you can adjust later)
        opt.dropout_local = 0.0
    elif opt.dataset == 'Amazon_cellPhone_2018':
        num_node = 22870
        opt.n_iter = 1
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")

    train_raw = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_raw = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.validation:
        train_raw, valid_raw = split_validation(train_raw, opt.valid_portion)
    else:
        valid_raw = None


    # ---------------- Load global graph (optional) ----------------
    global_graphs = []
    unified_graph_payload: Dict[str, Any] | None = None

    if opt.disable_global_fusion:
        print("[warn] --disable_global_fusion set. Global graph will NOT be used even if present.")
    elif opt.global_graph_mode == 'unified':
        default_unified_path = os.path.join("datasets", opt.dataset, "global_graph", "global_graph_unified.pkl")
        graph_path = opt.unified_graph_path or default_unified_path
        if not os.path.exists(graph_path):
            print(f"[warn] Unified global graph not found at {graph_path}. Proceeding without global fusion.")
        else:
            unified_raw = load_pickle_graph(graph_path)
            edge_index = unified_raw.get("edge_index")
            edge_type = unified_raw.get("edge_type")
            edge_weight = unified_raw.get("edge_weight")
            if edge_index is None or edge_type is None or edge_weight is None:
                print(f"[warn] Unified graph at {graph_path} is missing tensors. Skipping unified mode.")
            else:
                meta = unified_raw.get("meta", {})
                modalities: List[str] = meta.get("modalities") or []
                relation_map = meta.get("relation_map") or {}
                if not modalities and relation_map:
                    modalities = [name for name, _ in sorted(relation_map.items(), key=lambda kv: kv[1])]
                if not modalities:
                    raise ValueError("Unified graph metadata must include 'modalities' or 'relation_map'.")
                modality_features: Dict[str, np.ndarray] = {}
                for modality in modalities:
                    modality_features[modality] = load_modality_embedding_matrix(opt.dataset, modality)

                rows_with_pad = meta.get("num_nodes_including_padding")
                if rows_with_pad is None:
                    rows_with_pad = next(iter(modality_features.values())).shape[0]
                items_in_graph = rows_with_pad - 1
                if items_in_graph != num_node:
                    print(f"[warn] Adjusting num_node from {num_node} to {items_in_graph} based on unified graph.")
                    num_node = items_in_graph

                unified_graph_payload = {
                    "edge_index": edge_index,
                    "edge_type": edge_type,
                    "edge_weight": edge_weight,
                    "meta": meta,
                    "modalities": modalities,
                    "relation_map": relation_map,
                    "features": modality_features,
                    "path": graph_path,
                }
                print(f"[ok] Loaded unified global graph -> {graph_path}")
                print(f"   modalities: {modalities}")
                print(f"   edge_index shape: {np.asarray(edge_index).shape}, edge_type len: {len(edge_type)}, edge_weight len: {len(edge_weight)}")
    else:
        mod_tag_raw = getattr(opt, "global_graph_mods", "").strip()
        if mod_tag_raw:
            mod_tags = [seg.strip() for seg in mod_tag_raw.replace(",", "+").split("+") if seg.strip()]
            if not mod_tags:
                print("[warn] --global_graph_mods provided but no valid tags were parsed. Global graph disabled.")
            else:
                expected_items = None
                for tag in mod_tags:
                    gg_name = f"global_graph_{tag}.pkl"
                    graph_path = os.path.join("datasets", opt.dataset, "global_graph", gg_name)
                    if not os.path.exists(graph_path):
                        legacy = os.path.join("datasets", opt.dataset, "global_graph", f"global_graph_dec_{tag}.pkl")
                        if os.path.exists(legacy):
                            print(f"[warn] Requested global graph not found at {graph_path}. Falling back to legacy: {legacy}")
                            graph_path = legacy
                        else:
                            legacy = os.path.join("datasets", opt.dataset, "image_global_graph_dec.pkl")
                            if os.path.exists(legacy) and tag == mod_tags[0]:
                                print(f"[warn] Requested global graph not found at {graph_path}. Falling back to legacy: {legacy}")
                                graph_path = legacy
                    if not os.path.exists(graph_path):
                        print(f"[warn] No global graph file found for tag={tag}: {graph_path}. Skipping.")
                        continue
                    global_graph = load_pickle_graph(graph_path)
                    edge_index = global_graph.get("edge_index")
                    features = global_graph.get("x")
                    edge_weight = global_graph.get("edge_weight")
                    if edge_index is None or features is None:
                        print(f"[warn] Global graph at {graph_path} missing edge_index or features. Skipping.")
                        continue
                    meta = global_graph.get("meta", {})
                    rows_with_pad = meta.get("num_nodes_including_padding", getattr(features, "shape", [0])[0])
                    items_in_graph = rows_with_pad - 1
                    shape_x = getattr(features, "shape", None)
                    shape_e = getattr(edge_index, "shape", None)
                    print(f"[ok] Using global graph [{tag}] -> {graph_path}")
                    print(f"   x shape: {shape_x}, edge_index shape: {shape_e}")
                    if edge_weight is not None:
                        ew_len = edge_weight.shape[0] if hasattr(edge_weight, "shape") else len(edge_weight)
                        print(f"   edge_weight len: {ew_len}")
                    print(f"   rows (incl. pad): {rows_with_pad}  -> items: {items_in_graph}")
                    if expected_items is None:
                        expected_items = items_in_graph
                        if expected_items != num_node:
                            print(f"[warn] Setting num_node = {expected_items} (was {num_node}) to align with global graph.")
                            num_node = expected_items
                    elif items_in_graph != expected_items:
                        print(f"[warn] Global graph [{tag}] has item count {items_in_graph}, expected {expected_items}.")
                    global_graphs.append({
                        "tag": tag,
                        "edge_index": edge_index,
                        "edge_weight": edge_weight,
                        "features": features,
                        "meta": meta,
                        "path": graph_path,
                        "items": items_in_graph,
                    })
                if not global_graphs:
                    print("[warn] No global graphs were loaded. Continuing without global fusion.")
        else:
            print(
                "[info] --global_graph_mods not set. Global graph will NOT be used.\n"
                "   To enable, pass for example: --global_graph_mods image_knn  or  --global_graph_mods \"image_knn+category_knn\""
            )
    # adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    # num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_dataset = Data(train_raw)
    val_dataset = Data(valid_raw) if valid_raw is not None else None
    test_dataset = Data(test_raw)
    if opt.debug_sanity:
        debug_check_dataset(train_dataset, val_dataset if val_dataset is not None else test_dataset, num_node)
        if unified_graph_payload:
            debug_check_unified_graph(unified_graph_payload, num_node)
        else:
            debug_check_global_graph(global_graphs, num_node)


    # adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    # model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print("🔍 Checking max item ID in train and test:")
    max_train_id = max([max(seq) for seq in train_dataset.inputs if len(seq) > 0])
    max_test_id = max([max(seq) for seq in test_dataset.inputs if len(seq) > 0])
    print(f"Max item ID in train: {max_train_id}")
    print(f"Max item ID in test: {max_test_id}")
    print(f"num_node (embedding size): {num_node}")
    if max_test_id >= num_node:
        print("[warn] Test set has item ID exceeding num_node — this will cause an indexing error!")
    if unified_graph_payload:
        print("[info] CombineGraph will be initialized WITH unified global graph.")
    elif global_graphs:
        print(f"[info] CombineGraph will be initialized WITH {len(global_graphs)} global graph(s).")
    else:
        print("[info] CombineGraph will be initialized WITHOUT a global graph.")

    model = trans_to_cuda(CombineGraph(opt, num_node, global_graphs=global_graphs, unified_graph=unified_graph_payload))
    print(f"[CombineGraph] has_global_graph={getattr(model, 'has_global_graph', None)}")
    if opt.debug_sanity:
        debug_peek_one_batch(opt, model, train_dataset, note="train-before-train")



    print(opt)
    ks = [5, 10, 20]
    metric_keys = [f'Recall@{k}' for k in ks] + [f'MRR@{k}' for k in ks]
    start = time.time()
    eval_sets = []
    if val_dataset is not None:
        eval_sets.append(('val', val_dataset))
    eval_sets.append(('test', test_dataset))
    target_split = 'val' if val_dataset is not None else 'test'

    best_result = {key: float('-inf') for key in metric_keys}
    best_epoch = {key: 0 for key in metric_keys}
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        _train_loss_sum, eval_results = train_test(model, train_dataset, eval_sets)
        metrics_target = eval_results.get(target_split, {})
        flag = 0
        for key in metric_keys:
            value = metrics_target.get(key, float('-inf'))
            if value >= best_result.get(key, float('-inf')):
                best_result[key] = value
                best_epoch[key] = epoch
                flag = 1
        print('Current Result:')
        for split_name, metrics in eval_results.items():
            loss_val = metrics.get('Loss', float('nan'))
            header = f'  [{split_name}]'
            if not np.isnan(loss_val):
                header += f' Loss: {loss_val:.4f}'
            print(header)
            for k in ks:
                recall_k = metrics.get(f"Recall@{k}", float('nan'))
                mrr_k = metrics.get(f"MRR@{k}", float('nan'))
                print(f'    Recall@{k}: {recall_k:.4f}    MRR@{k}: {mrr_k:.4f}')

        print(f'Best Result (tracked on {target_split}):')
        for k in ks:
            best_recall = best_result.get(f'Recall@{k}', float('nan'))
            best_mrr = best_result.get(f'MRR@{k}', float('nan'))
            epoch_recall = best_epoch.get(f'Recall@{k}', 0)
            epoch_mrr = best_epoch.get(f'MRR@{k}', 0)
            print('\tRecall@{}:	{:.4f}	Epoch:	{}	MRR@{}:	{:.4f}	Epoch:	{}'.format(k, best_recall, epoch_recall, k, best_mrr, epoch_mrr))
        if opt.debug_sanity and getattr(model, "global_mode", None) == 'unified':
            gate_mean = getattr(model.unified_encoder, "latest_gate_mean", None)
            if gate_mean is not None:
                modal_names = getattr(model.unified_encoder, "modalities", [])
                values = gate_mean.tolist()
                formatted = ", ".join(f"{name}:{val:.3f}" for name, val in zip(modal_names, values))
                print(f"[DBG] mean gate weights after epoch {epoch}: {formatted}")
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
