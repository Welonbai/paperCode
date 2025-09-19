# model.py
import math
import torch
from torch import nn
import torch.nn.functional as F
from aggregator import LocalAggregator, GlobalAggregator


class CombineGraph(nn.Module):
    """
    Session-based recommender with optional catalog-level (global) graph.

    Inputs at construction:
      - num_node: number of items (1..num_node are valid item ids; 0 is padding)
      - features: [num_node+1, D_raw] global side-info features (row 0 should be zeros)
      - edge_index: [2, E] global edges (ids aligned with padded item table)
      - edge_weight: [E] optional edge strengths (cosine similarity, etc.)

    Dynamic projector:
      - We infer D_raw from `features.shape[1]` and create nn.Linear(D_raw, hiddenSize).
      - Works for 384 (category), 512 (image), 896 (image+category), etc.
    """

    def __init__(self, opt, num_node, edge_index=None, edge_weight=None, features=None):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node                      # number of real items (ids 1..num_node)
        self.hidden_size = opt.hiddenSize
        self.dropout_local = opt.dropout_local

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        # ---------------------------
        # Global/catalog branch (optional)
        # ---------------------------
        self.has_global_graph = (edge_index is not None and features is not None)
        if self.has_global_graph:
            # Ensure tensors and keep them as buffers so they move with .to(device) and are saved with the model
            if not torch.is_tensor(features):
                features = torch.as_tensor(features, dtype=torch.float32)
            if not torch.is_tensor(edge_index):
                edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(f"[CombineGraph] edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
            else:
                if not torch.is_tensor(edge_weight):
                    edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32)
                else:
                    edge_weight = edge_weight.float()
                edge_weight = edge_weight.view(-1)
                if edge_weight.size(0) != edge_index.size(1):
                    raise ValueError(f"[CombineGraph] edge_weight length ({edge_weight.size(0)}) does not match edge_index columns ({edge_index.size(1)})")

            if features.dim() != 2:
                raise ValueError(f"[CombineGraph] features must be 2D, got {tuple(features.shape)}")

            if features.size(0) != (num_node + 1):
                # Not fatal, but very likely an id/padding misalignment.
                print(
                    f"⚠️ [CombineGraph] features rows ({features.size(0)}) "
                    f"!= num_node+1 ({num_node+1}). Check id remapping and padding row 0."
                )

            self.register_buffer("global_features_raw", features)     # [N+1, D_raw]
            self.register_buffer("global_edge_index", edge_index)     # [2, E]
            self.register_buffer("global_edge_weight", edge_weight)   # [E]

            # ---- Dynamic projector: D_raw -> hidden_size
            input_dim_raw = features.size(1)
            self.global_projector = nn.Linear(input_dim_raw, self.hidden_size)

            # Lightweight message-passing over the catalog graph
            self.global_gnn = GlobalAggregator(self.hidden_size, dropout=opt.dropout_global, act=torch.relu)
        else:
            self.global_features_raw = None
            self.global_edge_index = None
            self.global_edge_weight = None

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
        if self.has_global_graph:
            # 1) Project raw side-info features to hidden space
            global_features_proj = self.global_projector(self.global_features_raw)  # [N+1, hidden]

            # 2) Propagate over the catalog graph
            global_node_emb = self.global_gnn(
                global_features_proj,
                self.global_edge_index,
                self.global_edge_weight,
            )  # [N+1, hidden]

            # 3) Gather per-token global vectors by indexing with the session's unique nodes
            global_hidden = global_node_emb[inputs]                                # [batch, n_nodes, hidden]

            # 4) Fuse local + global (element-wise sum keeps shapes aligned)
            fused_hidden = local_hidden + global_hidden
        else:
            fused_hidden = local_hidden

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


def train_test(model, train_data, test_data):
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
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    hit_list, mrr_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            targets, scores = forward(model, batch)
            topk_indices = scores.topk(20)[1].detach().cpu().numpy()  # [batch, 20]
            targets_np = targets.detach().cpu().numpy()
            for pred_row, tgt in zip(topk_indices, targets_np):
                tgt_idx = tgt - 1  # convert to 0-based
                hit = (tgt_idx in pred_row)
                hit_list.append(hit)
                if hit:
                    rank = int((pred_row == tgt_idx).nonzero()[0][0]) + 1
                    mrr_list.append(1.0 / rank)
                else:
                    mrr_list.append(0.0)

    import numpy as np
    recall20 = np.mean(hit_list) * 100.0
    mrr20 = np.mean(mrr_list) * 100.0
    return [recall20, mrr20]
