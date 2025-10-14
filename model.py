# model.py
import math
from typing import List, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from aggregator import LocalAggregator, GlobalAggregator

class GlobalGraphBranch(nn.Module):
    def __init__(self, features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, hidden_size: int, dropout: float, act=torch.relu, tag: str = ""):
        super().__init__()
        self.tag = tag
        self.register_buffer("features_raw", features)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        input_dim_raw = features.size(1)
        self.projector = nn.Linear(input_dim_raw, hidden_size)
        self.aggregator = GlobalAggregator(hidden_size, dropout=dropout, act=act)

    def forward(self) -> torch.Tensor:
        projected = self.projector(self.features_raw)
        return self.aggregator(projected, self.edge_index, self.edge_weight)


class GatedNodeEmbeddingFusion(nn.Module):
    def __init__(self, modalities: List[str], hidden_size: int, log_gate_stats: bool = False):
        super().__init__()
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.log_gate_stats = log_gate_stats

        self.layer_norms = nn.ModuleDict({
            modality: nn.LayerNorm(hidden_size)
            for modality in modalities
        })

        gate_hidden = max(hidden_size // 2, 16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * len(modalities), gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, len(modalities)),
        )

        self.latest_gate_mean: Optional[torch.Tensor] = None

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        normed_list = []
        for modality in self.modalities:
            proj = embeddings[modality]
            normed = self.layer_norms[modality](proj)
            normed_list.append(normed)

        stacked = torch.stack(normed_list, dim=1)                            # [N+1, M, d]
        fused_input = torch.cat(normed_list, dim=-1)                         # [N+1, M*d]

        logits = self.gate_mlp(fused_input)                                  # [N+1, M]
        weights = torch.softmax(logits, dim=-1)                              # [N+1, M]

        fused = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)            # [N+1, d]
        if fused.size(0) > 0:
            fused[0] = 0

        if self.log_gate_stats:
            self.latest_gate_mean = weights.mean(dim=0).detach().cpu()

        return fused


class UnifiedGlobalEncoder(nn.Module):
    def __init__(
        self,
        modalities: List[str],
        features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden_size: int,
        dropout: float,
        fuse_mode: str = 'avg',
        log_gate_stats: bool = False,
    ):
        super().__init__()
        if not modalities:
            raise ValueError("UnifiedGlobalEncoder requires at least one modality.")
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.fuse_mode = fuse_mode
        self.log_gate_stats = log_gate_stats

        self.projectors = nn.ModuleDict()
        for modality in modalities:
            feat = torch.as_tensor(features[modality], dtype=torch.float32)
            self.register_buffer(f"feat_{modality}", feat)
            self.projectors[modality] = nn.Linear(feat.size(1), hidden_size, bias=False)

        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        edge_type = torch.as_tensor(edge_type, dtype=torch.long)
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.register_buffer("edge_weight", edge_weight)

        unique_relations = torch.unique(edge_type).tolist()
        unique_relations = [int(r) for r in sorted(unique_relations)]
        self.relation_ids = unique_relations

        self.relation_linears = nn.ModuleDict({
            str(rel): nn.Linear(hidden_size, hidden_size, bias=False) for rel in unique_relations
        })

        for rel in unique_relations:
            rel_mask = (edge_type == rel).nonzero(as_tuple=False).view(-1)
            self.register_buffer(f"edge_ids_rel_{rel}", rel_mask)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gated_fusion: Optional[GatedNodeEmbeddingFusion] = None
        if self.fuse_mode == 'gate' and len(modalities) > 1:
            self.gated_fusion = GatedNodeEmbeddingFusion(modalities, hidden_size, log_gate_stats=log_gate_stats)
        self.latest_gate_mean: Optional[torch.Tensor] = None

    def _base_embedding(self) -> torch.Tensor:
        projected_map: Dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            feat = getattr(self, f"feat_{modality}")
            proj = self.projectors[modality](feat)
            projected_map[modality] = proj

        if self.gated_fusion is not None:
            h0 = self.gated_fusion(projected_map)
            if self.log_gate_stats:
                self.latest_gate_mean = self.gated_fusion.latest_gate_mean
        else:
            proj_stack = torch.stack([projected_map[mod] for mod in self.modalities], dim=0)
            h0 = proj_stack.mean(dim=0)
            h0 = h0.clone()
            if h0.size(0) > 0:
                h0[0] = 0

        return h0

    def forward(self) -> torch.Tensor:
        h0 = self._base_embedding()
        device = h0.device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device)

        aggregated = torch.zeros_like(h0)
        num_nodes = h0.size(0)
        for rel in self.relation_ids:
            edge_ids = getattr(self, f"edge_ids_rel_{rel}").to(device)
            if edge_ids.numel() == 0:
                continue
            src = edge_index[0, edge_ids]
            dst = edge_index[1, edge_ids]
            weights = edge_weight[edge_ids]
            neighbor_embed = self.relation_linears[str(rel)](h0[src])
            weighted_msgs = neighbor_embed * weights.unsqueeze(-1)

            agg_rel = torch.zeros_like(h0)
            agg_rel.index_add_(0, dst, weighted_msgs)

            denom = h0.new_zeros(num_nodes)
            denom.index_add_(0, dst, weights)
            denom = denom.clamp_min_(1e-8).unsqueeze(-1)
            mean_rel = agg_rel / denom
            aggregated = aggregated + mean_rel

        h1 = self.layer_norm(h0 + self.dropout(aggregated))
        if h1.size(0) > 0:
            h1[0] = 0
        return h1


class CombineGraph(nn.Module):
    """
    Session-based recommender with optional catalog-level (global) graphs.

    Inputs at construction:
      - num_node: number of items (1..num_node are valid item ids; 0 is padding)
      - global_graphs: optional iterable/dict describing catalog graphs. Each entry should
        provide keys `features`, `edge_index`, `edge_weight` (optional), and `tag` (optional).
      - edge_index / features / edge_weight: legacy single-graph arguments kept for backward compatibility.
    """

    def __init__(self, opt, num_node, global_graphs=None, edge_index=None, edge_weight=None, features=None, unified_graph=None):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node                      # number of real items (ids 1..num_node)
        self.hidden_size = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.global_mode = 'legacy'
        self.node_embedding_fuse = getattr(opt, 'node_embedding_fuse', 'avg')

        # ---------------------------
        # Local/session branch
        # ---------------------------
        self.local_agg = LocalAggregator(self.hidden_size, self.opt.alpha, dropout=0.0)

        # Item embeddings (1 extra row for padding id=0) and position embeddings
        self.item_embedding = nn.Embedding(num_node + 1, self.hidden_size)
        self.pos_embedding = nn.Embedding(200, self.hidden_size)

        # Attention / scoring parameters (same formulation as your original code)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()

        # ---------------------------
        # Global/catalog branch (optional)
        # ---------------------------
        self.unified_encoder: Optional[UnifiedGlobalEncoder] = None
        self.has_global_graph = False
        if global_graphs is None:
            if edge_index is not None and features is not None:
                global_graphs = [{
                    'tag': 'legacy',
                    'edge_index': edge_index,
                    'edge_weight': edge_weight,
                    'features': features,
                }]
            else:
                global_graphs = []
        elif isinstance(global_graphs, dict):
            global_graphs = [global_graphs]

        self.global_branches = nn.ModuleList()
        self.global_modalities: List[str] = []
        if unified_graph is not None:
            self.global_mode = 'unified'
            modalities = unified_graph.get('modalities', [])
            features_map = unified_graph.get('features', {})
            edge_index_u = unified_graph.get('edge_index')
            edge_type_u = unified_graph.get('edge_type')
            edge_weight_u = unified_graph.get('edge_weight')
            if edge_index_u is None or edge_type_u is None or edge_weight_u is None:
                raise ValueError("[CombineGraph] Unified graph payload missing tensors.")
            if not modalities:
                raise ValueError("[CombineGraph] Unified graph requires modality list.")
            self.unified_encoder = UnifiedGlobalEncoder(
                modalities=modalities,
                features={mod: features_map[mod] for mod in modalities},
                edge_index=edge_index_u,
                edge_type=edge_type_u,
                edge_weight=edge_weight_u,
                hidden_size=self.hidden_size,
                dropout=opt.dropout_global,
                fuse_mode=self.node_embedding_fuse,
                log_gate_stats=bool(getattr(opt, 'debug_sanity', False)),
            )
            self.has_global_graph = True
        else:
            for idx, graph in enumerate(global_graphs):
                tag = graph.get('tag', f'graph{idx}')
                features_data = graph.get('features')
                edge_index_data = graph.get('edge_index')
                edge_weight_data = graph.get('edge_weight')

                if features_data is None or edge_index_data is None:
                    print(f"[warn] [CombineGraph] Global graph '{tag}' missing features or edge_index. Skipping.")
                    continue

                features_tensor = torch.as_tensor(features_data, dtype=torch.float32)
                edge_index_tensor = torch.as_tensor(edge_index_data, dtype=torch.long)

                if edge_index_tensor.dim() != 2 or edge_index_tensor.size(0) != 2:
                    raise ValueError(f"[CombineGraph] edge_index for '{tag}' must have shape [2, E], got {tuple(edge_index_tensor.shape)}")
                if features_tensor.dim() != 2:
                    raise ValueError(f"[CombineGraph] features for '{tag}' must be 2D, got {tuple(features_tensor.shape)}")
                if features_tensor.size(0) != (num_node + 1):
                    print(f"[warn] [CombineGraph] features rows ({features_tensor.size(0)}) != num_node+1 ({num_node+1}) for '{tag}'.")

                if edge_weight_data is None:
                    edge_weight_tensor = torch.ones(edge_index_tensor.size(1), dtype=torch.float32)
                else:
                    edge_weight_tensor = torch.as_tensor(edge_weight_data, dtype=torch.float32).view(-1)
                    if edge_weight_tensor.size(0) != edge_index_tensor.size(1):
                        raise ValueError(
                            f"[CombineGraph] edge_weight length ({edge_weight_tensor.size(0)}) does not match edge_index columns ({edge_index_tensor.size(1)}) for '{tag}'."
                        )

                branch = GlobalGraphBranch(
                    features_tensor,
                    edge_index_tensor,
                    edge_weight_tensor,
                    self.hidden_size,
                    opt.dropout_global,
                    torch.relu,
                    tag=tag,
                )
                self.global_branches.append(branch)
                self.global_modalities.append(tag)

            self.has_global_graph = len(self.global_branches) > 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        # Init trainable parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Uniform init similar to the original code."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.dim() > 1:
                nn.init.uniform_(w, -stdv, stdv)
            else:
                nn.init.uniform_(w, -stdv, stdv)

    # -------------------------------------------------------------------------
    # Scoring head
    # -------------------------------------------------------------------------
    def compute_scores(self, seq_hidden_states: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute scores over the item catalog for each sequence in the batch.

        Args:
          seq_hidden_states: [batch_size, seq_len, hidden_size]
          seq_mask:          [batch_size, seq_len] (1 for real token, 0 for padding)

        Returns:
          scores:            [batch_size, num_node] (scores for items 1..num_node)
        """
        # Expand mask for broadcasting
        mask_expanded = seq_mask.float().unsqueeze(-1)                        # [batch_size, seq_len, 1]

        batch_size, seq_len, hidden_size = seq_hidden_states.shape

        # Position embeddings for the current sequence length
        pos_emb = self.pos_embedding.weight[:seq_len]                         # [seq_len, hidden_size]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)               # [batch_size, seq_len, hidden_size]

        # Session representation: mean of valid positions
        sum_hidden = torch.sum(seq_hidden_states * mask_expanded, dim=1, keepdim=True)      # [batch_size, 1, hidden_size]
        denom = torch.sum(mask_expanded, dim=1, keepdim=True).clamp_min(1e-9)               # avoid div-by-zero
        session_mean = sum_hidden / denom                                                   # [batch_size, 1, hidden_size]
        session_mean_tiled = session_mean.repeat(1, seq_len, 1)                             # [batch_size, seq_len, hidden_size]

        # Attention to select an informative position
        nh = torch.matmul(torch.cat([pos_emb, seq_hidden_states], dim=-1), self.w_1)        # [batch_size, seq_len, hidden_size]
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(session_mean_tiled))
        attn_weights = torch.matmul(nh, self.w_2)                                           # [batch_size, seq_len, 1]
        attn_weights = attn_weights * mask_expanded                                         # mask out paddings
        selected_rep = torch.sum(attn_weights * seq_hidden_states, dim=1)                   # [batch_size, hidden_size]

        # Catalog item table (skip padding row 0)
        item_table = self.item_embedding.weight[1:]                                         # [num_node, hidden_size]

        # Scores over all items (1..num_node)
        scores = torch.matmul(selected_rep, item_table.transpose(1, 0))                     # [batch_size, num_node]
        return scores

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self,
                inputs: torch.Tensor,        # [batch, n_nodes]   unique-node list for this session (a.k.a. seq_items)
                local_adj: torch.Tensor,     # [batch, n_nodes, n_nodes]  adjacency over 'inputs'
                mask_item: torch.Tensor,     # [batch, n_nodes]   1 for real nodes, 0 for padding (aligned with 'inputs')
                _unused_seq_ids: torch.Tensor  # kept for compatibility with existing Data loader signature
                ) -> torch.Tensor:
        """
        Returns:
          fused_hidden: [batch_size, n_nodes, hidden_size], aligned with 'inputs'
        """
        # ----- Local/session branch (must align with inputs/adj/mask) -----
        node_emb = self.item_embedding(inputs)                                   # [batch, n_nodes, hidden]
        local_hidden = self.local_agg(node_emb, local_adj, mask_item)            # [batch, n_nodes, hidden]
        local_hidden = F.dropout(local_hidden, self.dropout_local, training=self.training)

        # ----- Global/catalog branch (optional) -----
        fused_hidden = local_hidden
        if self.global_mode == 'unified' and self.unified_encoder is not None:
            global_node_emb = self.unified_encoder()                         # [N+1, hidden]
            branch_hidden = global_node_emb[inputs]                          # [batch, n_nodes, hidden]
            fused_hidden = fused_hidden + branch_hidden
        elif self.has_global_graph and len(self.global_branches) > 0:
            aggregated_global = None
            for branch in self.global_branches:
                global_node_emb = branch()                                   # [N+1, hidden]
                branch_hidden = global_node_emb[inputs]                      # [batch, n_nodes, hidden]
                if aggregated_global is None:
                    aggregated_global = branch_hidden
                else:
                    aggregated_global = aggregated_global + branch_hidden
            if aggregated_global is not None:
                aggregated_global = aggregated_global / len(self.global_branches)
                fused_hidden = fused_hidden + aggregated_global

        return fused_hidden


# ---------------------------
# Utilities (unchanged API)
# ---------------------------
def trans_to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def trans_to_cpu(x):
    return x.cpu() if torch.cuda.is_available() else x


def forward(model, batch):
    """
    Wrapper used by train/test:
      batch = (alias_inputs, local_adj, seq_items, seq_mask, targets, seq_item_ids)

    Note:
      - local_adj & seq_mask are built over seq_items (unique-node list).
      - We must call model with (seq_items, local_adj, seq_mask, seq_item_ids).
    """
    alias_inputs, local_adj, seq_items, seq_mask, targets, seq_item_ids = batch

    alias_inputs = trans_to_cuda(alias_inputs).long()   # [batch, n_seq_pos] indices mapping to original order
    seq_items    = trans_to_cuda(seq_items).long()      # [batch, n_nodes]   unique-node list
    local_adj    = trans_to_cuda(local_adj).float()     # [batch, n_nodes, n_nodes]
    seq_mask     = trans_to_cuda(seq_mask).long()       # [batch, n_nodes]
    seq_item_ids = trans_to_cuda(seq_item_ids).long()   # [batch, n_seq_pos] (kept for signature compatibility)

    # Fused hidden states from the model (ALIGNED WITH seq_items!)
    hidden_all_positions = model(seq_items, local_adj, seq_mask, seq_item_ids)  # [batch, n_nodes, hidden]

    # Reorder to original (alias) order per sequence
    def pick_sequence_hidden(idx):
        return hidden_all_positions[idx][alias_inputs[idx]]                      # [n_seq_pos, hidden]

    batch_indices = torch.arange(len(alias_inputs)).long()
    seq_hidden_reordered = torch.stack([pick_sequence_hidden(i) for i in batch_indices])  # [batch, n_seq_pos, hidden]

    # Compute catalog scores
    scores = model.compute_scores(seq_hidden_reordered, seq_mask)
    return targets, scores


def train_test(model, train_data, eval_datasets):
    import datetime
    from tqdm import tqdm
    import numpy as np

    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    # (your debug blocks remain unchanged)
    if getattr(model.opt, "debug_sanity", False):
        sanity_loader = torch.utils.data.DataLoader(
            train_data, num_workers=0, batch_size=model.batch_size,
            shuffle=True, pin_memory=False
        )
        _b = next(iter(sanity_loader))
        _, _, _, _, _targets, _ = _b
        print(f"[DBG] targets min/max: {int(_targets.min())}/{int(_targets.max())}, zeros={( _targets==0 ).sum().item()}")
        bad_lt1 = (_targets < 1).sum().item()
        bad_gtN = (_targets > model.num_node).sum().item() if hasattr(model, 'num_node') else 0
        print(f"[DBG] targets <1: {bad_lt1}, targets > num_node: {bad_gtN}")

        print("[DBG] doing 3 mini-steps on one batch to see if loss drops")
        micro_loader = torch.utils.data.DataLoader(
            train_data, num_workers=0, batch_size=model.batch_size,
            shuffle=True, pin_memory=False
        )
        micro_batch = next(iter(micro_loader))
        last_loss = None
        for i in range(3):
            model.optimizer.zero_grad()
            t, s = forward(model, micro_batch)
            t = trans_to_cuda(t).long()
            loss_micro = model.loss_function(s, t - 1)
            loss_micro.backward()
            emb_grad = getattr(model.item_embedding, "weight").grad
            if emb_grad is None:
                print("[DBG] item_embedding grad is None (unexpected)")
            else:
                print("[DBG] item_embedding grad stats:",
                      "nan%", float(torch.isnan(emb_grad).float().mean().item())*100,
                      "min", float(emb_grad.min().item()),
                      "mean", float(emb_grad.mean().item()),
                      "max", float(emb_grad.max().item()))
            model.optimizer.step()
            cur = float(loss_micro.detach().cpu().item())
            print(f"[DBG] step {i}: loss={cur:.4f}" + ("" if last_loss is None else f"  (Δ={cur-last_loss:+.4f})"))
            last_loss = cur

    for batch in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, batch)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)  # targets are 1-based; subtract 1 to match [0..num_node-1]
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.detach().cpu().item())

    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    ks = [5, 10, 20]
    max_k = max(ks)
    results = {}

    with torch.no_grad():
        for split_name, dataset in eval_datasets:
            loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=model.batch_size,
                                                 shuffle=False, pin_memory=True)
            hits_per_k = {k: [] for k in ks}
            mrr_per_k = {k: [] for k in ks}
            loss_sum = 0.0
            sample_count = 0
            for batch in loader:
                targets, scores = forward(model, batch)
                targets = trans_to_cuda(targets).long()
                loss = model.loss_function(scores, targets - 1)
                loss_sum += float(loss.detach().cpu().item()) * targets.size(0)
                sample_count += targets.size(0)

                topk_indices = scores.topk(max_k)[1].detach().cpu().numpy()  # [batch, max_k]
                targets_np = targets.detach().cpu().numpy() - 1  # convert to 0-based once
                for pred_row, tgt_idx in zip(topk_indices, targets_np):
                    for k in ks:
                        candidate_row = pred_row[:k]
                        hit = tgt_idx in candidate_row
                        hits_per_k[k].append(hit)
                        if hit:
                            rank = int(np.where(candidate_row == tgt_idx)[0][0]) + 1
                            mrr_per_k[k].append(1.0 / rank)
                        else:
                            mrr_per_k[k].append(0.0)

            metrics = {}
            if sample_count > 0:
                metrics['Loss'] = loss_sum / sample_count
            for k in ks:
                metrics[f'Recall@{k}'] = float(np.mean(hits_per_k[k]) * 100.0)
                metrics[f'MRR@{k}'] = float(np.mean(mrr_per_k[k]) * 100.0)
            results[split_name] = metrics

    return total_loss, results

