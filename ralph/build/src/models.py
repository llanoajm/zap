"""
P2 — GNN model definitions.

BaselineSurrogate: message-passing GNN operating on the sparse incidence
(no torch_geometric required).  Size-agnostic — works on any graph.

Architecture
------------
MPLayer: single message-passing step.
  msg_ij = MLP_msg([h_i || h_j || e_ij])
  h_i'   = MLP_upd([h_i || mean_j(msg_ij)])

MPGNNEncoder: stack of 3 MPLayers with hidden_dim=64, ReLU activations,
layer-norm between passes.

BaselineSurrogate (P2 supervised control):
  - Encoder → node embeddings  (n_bus × D)
  - LMP head: linear D→1 per bus
  - Dispatch head: gather embedding at gen_bus, cat [gen_pmax_norm, gen_cost_norm],
    MLP → scalar per generator, clamp to [0, 1] (fraction of pmax)
  - Returns (pred_lmp [n_bus], pred_pg [n_gen])

JEPAEncoderGNN (shared backbone for P3/P4):
  - Same MPGNNEncoder with the same interface; wraps it for standalone use.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MPLayer(nn.Module):
    """One message-passing step with edge features, mean aggregation."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, hidden_dim)
        self.upd_mlp = MLP(node_dim + hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,          # (n_bus, node_dim)
        edge_index: torch.Tensor, # (2, n_branch)  long
        edge_feats: torch.Tensor, # (n_branch, edge_dim)
    ) -> torch.Tensor:            # (n_bus, hidden_dim)
        src, dst = edge_index[0], edge_index[1]  # (n_branch,)
        h_src = h[src]   # (n_branch, node_dim)
        h_dst = h[dst]   # (n_branch, node_dim)

        # Messages for each edge (bidirectional — add both orientations)
        msg_fwd = self.msg_mlp(torch.cat([h_src, h_dst, edge_feats], dim=-1))
        msg_bwd = self.msg_mlp(torch.cat([h_dst, h_src, edge_feats], dim=-1))

        n_bus = h.size(0)
        agg = torch.zeros(n_bus, msg_fwd.size(-1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg_fwd)
        agg.index_add_(0, src, msg_bwd)

        # Degree (count both directions)
        deg = torch.zeros(n_bus, 1, device=h.device, dtype=h.dtype)
        ones = torch.ones(edge_index.size(1), 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, ones)
        deg.index_add_(0, src, ones)
        deg = deg.clamp(min=1.0)
        agg = agg / deg

        h_new = self.upd_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h_new)


class MPGNNEncoder(nn.Module):
    """3-layer message-passing encoder.  Output: node embeddings (n_bus × D)."""

    def __init__(self, node_in: int = 4, edge_in: int = 2, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_in, hidden_dim)
        self.layers = nn.ModuleList([
            MPLayer(hidden_dim, edge_in, hidden_dim) for _ in range(n_layers)
        ])

    def forward(
        self,
        node_feats: torch.Tensor,   # (n_bus, node_in)
        edge_index: torch.Tensor,   # (2, n_branch)
        edge_feats: torch.Tensor,   # (n_branch, edge_in)
    ) -> torch.Tensor:              # (n_bus, hidden_dim)
        h = F.relu(self.input_proj(node_feats))
        for layer in self.layers:
            h = F.relu(h + layer(h, edge_index, edge_feats))  # residual
        return h


# ---------------------------------------------------------------------------
# P2 Supervised baseline
# ---------------------------------------------------------------------------

class BaselineSurrogate(nn.Module):
    """
    Size-agnostic surrogate that predicts (LMP, dispatch) directly.

    Inputs
    ------
    node_feats  : (n_bus, 4)
    edge_index  : (2, n_branch)
    edge_feats  : (n_branch, 2)
    gen_bus     : (n_gen,) long — bus index of each generator
    gen_pmax    : (n_gen,) float — capacity [p.u.]
    gen_cost    : (n_gen,) float — normalized linear cost

    Outputs
    -------
    pred_lmp  : (n_bus,)  — predicted nodal prices
    pred_pg   : (n_gen,)  — predicted dispatch [p.u.]
    """

    def __init__(self, hidden_dim: int = 64, n_mp_layers: int = 3):
        super().__init__()
        self.encoder = MPGNNEncoder(
            node_in=4, edge_in=2, hidden_dim=hidden_dim, n_layers=n_mp_layers
        )
        D = hidden_dim
        # LMP head
        self.lmp_head = nn.Linear(D, 1)
        # Dispatch head: node embedding + gen features → dispatch fraction
        self.dispatch_head = MLP(D + 2, D // 2, 1, n_layers=2)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        gen_bus: torch.Tensor,
        gen_pmax: torch.Tensor,
        gen_cost: torch.Tensor,
    ):
        h = self.encoder(node_feats, edge_index, edge_feats)   # (n_bus, D)

        # LMP prediction
        pred_lmp = self.lmp_head(h).squeeze(-1)                # (n_bus,)

        # Dispatch prediction: fraction of pmax, clamp to [0,1]
        h_gen = h[gen_bus]                                     # (n_gen, D)
        gen_pmax_n = gen_pmax / (gen_pmax.max().clamp(min=1e-6))
        gen_cost_n = gen_cost / (gen_cost.max().clamp(min=1e-6))
        gen_extra = torch.stack([gen_pmax_n, gen_cost_n], dim=-1)  # (n_gen, 2)
        disp_frac = torch.sigmoid(
            self.dispatch_head(torch.cat([h_gen, gen_extra], dim=-1)).squeeze(-1)
        )                                                        # (n_gen,)
        pred_pg = disp_frac * gen_pmax                          # (n_gen,)

        return pred_lmp, pred_pg


# ---------------------------------------------------------------------------
# Shared encoder alias for P3/P4
# ---------------------------------------------------------------------------

JEPAEncoderGNN = MPGNNEncoder
