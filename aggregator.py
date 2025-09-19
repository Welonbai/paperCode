from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super().__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        raise NotImplementedError


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        n_nodes = h.shape[1]

        a_input = (h.repeat(1, 1, n_nodes).view(batch_size, n_nodes * n_nodes, self.dim)
                   * h.repeat(1, n_nodes, 1)).view(batch_size, n_nodes, n_nodes, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, n_nodes, n_nodes)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, n_nodes, n_nodes)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, n_nodes, n_nodes)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, n_nodes, n_nodes)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super().__init__()
        self.dropout = dropout
        self.act_fn = act
        self.dim = dim

        self.linear_self = nn.Linear(dim, dim, bias=False)
        self.linear_neigh = nn.Linear(dim, dim, bias=False)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: Optional[torch.Tensor],
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            out = self.linear_self(node_features)
            out = self.act_fn(out)
            out = F.dropout(out, self.dropout, training=self.training)
            out[0] = 0
            return out

        device = node_features.device
        dtype = node_features.dtype
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=device, dtype=dtype)
        else:
            edge_weight = edge_weight.to(device=device, dtype=dtype)

        num_nodes = node_features.size(0)
        agg = torch.zeros_like(node_features)

        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.scatter_add_(0, row, edge_weight)
        norm = edge_weight / deg[row].clamp(min=1e-12)

        messages = node_features[col] * norm.unsqueeze(-1)
        agg.index_add_(0, row, messages)

        out = self.linear_self(node_features) + self.linear_neigh(agg)
        out = self.act_fn(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out[0] = 0
        return out