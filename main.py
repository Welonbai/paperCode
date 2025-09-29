import time
import argparse
import pickle
import os
import numpy as np
from model import *
from utils import *

def load_image_global_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    return graph

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
                    help='Modality tag for global graph filename, e.g. "image", "image+category". '
                         'Set empty string to disable loading.')
parser.add_argument('--disable_global_fusion', action='store_true',
                    help='Ignore global graph even if found (use local/session branch only).')
parser.add_argument('--debug_sanity', action='store_true',
                    help='Print extra diagnostics (targets, id ranges, NaNs, degree stats, zero-row check).')


opt = parser.parse_args()


def main():
    init_seed(2020)

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

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))


    # ---------------- Load global graph (optional) ----------------
    global_graphs = []
    mod_tag_raw = getattr(opt, "global_graph_mods", "").strip()
    if mod_tag_raw and not opt.disable_global_fusion:
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
                global_graph = load_image_global_graph(graph_path)
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
    elif opt.disable_global_fusion:
        print("[warn] --disable_global_fusion set. Global graph will NOT be used even if present.")
    else:
        print(
            "[info] --global_graph_mods not set. Global graph will NOT be used.\n"
            "   To enable, pass for example: --global_graph_mods image_knn  or  --global_graph_mods \"image_knn+category_knn\""
        )
    # adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    # num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)
    if opt.debug_sanity:
        debug_check_dataset(train_data, test_data, num_node)
        debug_check_global_graph(global_graphs, num_node)


    # adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    # model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print("🔍 Checking max item ID in train and test:")
    max_train_id = max([max(seq) for seq in train_data.inputs if len(seq) > 0])
    max_test_id = max([max(seq) for seq in test_data.inputs if len(seq) > 0])
    print(f"Max item ID in train: {max_train_id}")
    print(f"Max item ID in test: {max_test_id}")
    print(f"num_node (embedding size): {num_node}")
    if max_test_id >= num_node:
        print("[warn] Test set has item ID exceeding num_node — this will cause an indexing error!")
    if global_graphs:
        print(f"[info] CombineGraph will be initialized WITH {len(global_graphs)} global graph(s).")
    else:
        print("[info] CombineGraph will be initialized WITHOUT a global graph.")

    model = trans_to_cuda(CombineGraph(opt, num_node, global_graphs=global_graphs))
    print(f"[CombineGraph] has_global_graph={getattr(model, 'has_global_graph', None)}")
    if opt.debug_sanity:
        debug_peek_one_batch(opt, model, train_data, note="train-before-train")



    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
