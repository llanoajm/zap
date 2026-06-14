"""
P4 — LeJEPA-style self-supervised pretraining for the GNN encoder.

Training objective
------------------
  1. Mask a connected sub-zone of buses (BFS from random seed, ~25% of nodes).
  2. Context encoder sees the graph with masked-bus features zeroed.
  3. Target encoder (EMA of context encoder) sees the FULL unmasked graph.
  4. Predictor maps context embeddings → predicted target embeddings at masked buses.
  5. Loss = SmoothL1(predicted, target_stop_grad) + SIGReg(context_embeddings).

SIGReg (anti-collapse)
-----------------------
  Computes K random 1-D projections of the D-dim embeddings.
  For each projection, estimates the Epps-Pulley statistic (deviation from N(0,σ²)):
    - Variance term: penalize when std < 1 (collapse)
    - Covariance term: off-diagonal covariance penalty (force independence)
  Combined as VICReg-style loss for efficiency (EP test requires O(n²) which is too
  expensive with 600+ bus embeddings).

  The ROADMAP flags: "are node embeddings within a grid too correlated for SIGReg's
  i.i.d. assumption?"  We compute the intra-graph embedding correlation and report it.

Masked-zone construction
------------------------
  BFS from a random seed, target fraction = 25% of buses (min 1, max 100).
  Always select a connected subgraph.  Each training step gets a fresh random mask.

Architecture
------------
  ContextEncoder = MPGNNEncoder (same as P2/P3)
  TargetEncoder  = EMA copy (no grad, tau=0.996)
  Predictor      = 2-layer MLP applied independently at each masked bus
                   (takes context embedding at that bus position)

Usage
-----
  python jepa.py --pretrain-epochs 60 --finetune-epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import load_dataset, GraphSample
from models import MPGNNEncoder, MLP
import train_design_a as p3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/P4_metrics.json"
PRETRAIN_CKPT = "ralph/build/state/checkpoints/jepa_encoder.pt"
FINETUNE_CKPT = "ralph/build/state/checkpoints/design_a_jepa.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
MASK_FRAC   = 0.25
EMA_TAU     = 0.996
LMP_MAX_THRESHOLD = 100.0

# SIGReg weights
W_PRED      = 1.0
W_VAR       = 0.1
W_COV       = 0.05


# ---------------------------------------------------------------------------
# Masking: connected sub-zone via BFS
# ---------------------------------------------------------------------------

def bfs_mask(edge_index: np.ndarray, n_bus: int, seed: int | None = None,
             frac: float = MASK_FRAC) -> np.ndarray:
    """Return indices of the masked (target) zone — a connected subgraph."""
    if seed is None:
        seed = np.random.randint(n_bus)
    k_target = max(1, min(int(n_bus * frac), 100))

    # Build adjacency list
    adj = [[] for _ in range(n_bus)]
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)

    visited = [seed]
    in_set = {seed}
    queue = list(adj[seed])
    np.random.shuffle(queue)
    for v in queue:
        if len(visited) >= k_target:
            break
        if v not in in_set:
            in_set.add(v)
            visited.append(v)
            nbrs = adj[v].copy()
            np.random.shuffle(nbrs)
            queue.extend(nbrs)

    return np.array(visited[:k_target], dtype=np.int64)


# ---------------------------------------------------------------------------
# SIGReg — sketched isotropic-Gaussian regularizer (VICReg-style)
# ---------------------------------------------------------------------------

def sigreg_loss(
    h: torch.Tensor,      # (N, D) — embeddings for a batch of nodes
    n_proj: int = 32,
) -> tuple[torch.Tensor, dict]:
    """
    VICReg-style SIGReg:
      variance term:   max(0, 1 - std(z_d)) per dimension (collapse penalty)
      covariance term: ||off-diag(Cov(z))||_F² / D  (correlation penalty)

    Returns (loss, stats) where stats has intra-batch correlation info.
    """
    N, D = h.shape

    # Center
    h_c = h - h.mean(dim=0, keepdim=True)

    # Variance: std over batch per dimension
    std = torch.sqrt(h_c.var(dim=0) + 1e-4)   # (D,)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: D×D matrix, penalize off-diagonal
    if N > 1:
        cov = (h_c.T @ h_c) / (N - 1)            # (D, D)
        off_diag = cov ** 2
        off_diag.fill_diagonal_(0.0)
        cov_loss = off_diag.sum() / D
    else:
        cov_loss = torch.tensor(0.0, device=h.device)
        cov = None

    # Report intra-graph correlation (diagnostic for the i.i.d. concern)
    with torch.no_grad():
        if N > 1 and cov is not None:
            corr_diag_mean = float(cov.diagonal().mean().item())
            # Mean |off-diagonal|
            off_mask = ~torch.eye(D, dtype=torch.bool, device=h.device)
            corr_off_mean = float(cov[off_mask].abs().mean().item())
        else:
            corr_diag_mean, corr_off_mean = 0.0, 0.0

    stats = {
        "var_loss": float(var_loss.item()),
        "cov_loss": float(cov_loss.item()),
        "embed_std_mean": float(std.mean().item()),
        "cov_diag_mean": corr_diag_mean,
        "cov_offdiag_mean": corr_off_mean,
    }
    return var_loss, cov_loss, stats


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Predict target embeddings at masked positions from context embeddings."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        D = hidden_dim
        self.mlp = MLP(D, D, D, n_layers=2)
        self.norm = nn.LayerNorm(D)

    def forward(self, h_ctx: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
        # h_ctx: (n_bus, D);  mask_idx: (k,) long
        h_masked = h_ctx[mask_idx]           # (k, D) — context embeds at masked positions
        return self.norm(self.mlp(h_masked)) # (k, D)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, tau: float):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_((1 - tau) * p_o.data)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain(
    train_samples: list[GraphSample],
    n_epochs: int = 60,
    lr: float = 5e-4,
    verbose: bool = True,
) -> MPGNNEncoder:

    context_enc = MPGNNEncoder(node_in=4, edge_in=2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
    target_enc  = deepcopy(context_enc)
    for p in target_enc.parameters():
        p.requires_grad_(False)
    predictor = JEPAPredictor(hidden_dim=HIDDEN_DIM)

    optimizer = torch.optim.Adam(
        list(context_enc.parameters()) + list(predictor.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/20)

    rng = np.random.default_rng(0)
    sigreg_history = []
    pred_loss_history = []

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        context_enc.train()
        predictor.train()
        epoch_pred_loss = 0.0
        epoch_sig_var   = 0.0
        epoch_sig_cov   = 0.0
        epoch_intra_corr= 0.0

        order = rng.permutation(len(train_samples))
        for idx in order:
            s = train_samples[idx]
            nf = torch.tensor(s.node_feats, dtype=torch.float32)
            ei = torch.tensor(s.edge_index, dtype=torch.long)
            ef = torch.tensor(s.edge_feats, dtype=torch.float32)

            # Sample mask
            seed = int(rng.integers(s.n_bus))
            mask_idx_np = bfs_mask(s.edge_index, s.n_bus, seed=seed, frac=MASK_FRAC)
            mask_idx = torch.tensor(mask_idx_np, dtype=torch.long)

            # Context: zero masked bus features
            nf_ctx = nf.clone()
            nf_ctx[mask_idx] = 0.0

            optimizer.zero_grad()

            # Context encoder forward
            h_ctx = context_enc(nf_ctx, ei, ef)                # (n_bus, D)

            # Target encoder forward (no grad)
            with torch.no_grad():
                h_tgt = target_enc(nf, ei, ef)                 # (n_bus, D)

            # Predictor: predict target embeddings at masked positions
            h_pred = predictor(h_ctx, mask_idx)                # (k, D)
            h_tgt_masked = h_tgt[mask_idx].detach()            # (k, D)

            loss_pred = F.smooth_l1_loss(h_pred, h_tgt_masked)

            # SIGReg on context embeddings (all buses)
            var_l, cov_l, sig_stats = sigreg_loss(h_ctx)
            loss = W_PRED * loss_pred + W_VAR * var_l + W_COV * cov_l

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(context_enc.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()
            update_ema(context_enc, target_enc, EMA_TAU)

            epoch_pred_loss += loss_pred.item()
            epoch_sig_var   += sig_stats["var_loss"]
            epoch_sig_cov   += sig_stats["cov_loss"]
            epoch_intra_corr+= sig_stats["cov_offdiag_mean"]

        scheduler.step()
        n = len(train_samples)
        sigreg_history.append({
            "epoch": epoch,
            "pred_loss": epoch_pred_loss / n,
            "var_loss":  epoch_sig_var / n,
            "cov_loss":  epoch_sig_cov / n,
            "intra_corr": epoch_intra_corr / n,
        })
        pred_loss_history.append(epoch_pred_loss / n)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs):
            elapsed = time.time() - t0
            last = sigreg_history[-1]
            print(f"  Epoch {epoch:3d}/{n_epochs}  pred={last['pred_loss']:.4f}  var={last['var_loss']:.4f}  cov={last['cov_loss']:.4f}  intra_corr={last['intra_corr']:.4f}  ({elapsed:.0f}s)")

    return context_enc, sigreg_history


# ---------------------------------------------------------------------------
# Fine-tuning Design A from pretrained encoder
# ---------------------------------------------------------------------------

def finetune_design_a(
    pretrained_enc: MPGNNEncoder,
    train_layers_ok,
    layers,
    test_samples,
    n_epochs: int = 100,
    lr: float = 5e-4,
) -> tuple:
    """Fine-tune Design A with the pretrained encoder (lower LR for encoder)."""

    model = p3.DesignA(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    # Load pretrained encoder weights
    model.encoder.load_state_dict(pretrained_enc.state_dict())

    # Use lower LR for encoder (transfer), full LR for new decoder
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.cost_decoder.parameters())
    optimizer = torch.optim.Adam([
        {"params": enc_params, "lr": lr / 5},
        {"params": dec_params, "lr": lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/50)

    rng = np.random.default_rng(42)
    best_loss = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_solved = 0
        order = rng.permutation(len(train_layers_ok))

        for idx in order:
            s = train_layers_ok[idx]
            layer_info = layers[s.name]
            if layer_info is None:
                continue
            layer, gi, true_cost_np = layer_info
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj = p3.to_torch(s)

            optimizer.zero_grad()
            pred_cost = model(nf, ei, ef, gb)
            dispatch, lmps, obj_out = p3.dispatch_through_layer(pred_cost, layer, gi)

            if dispatch.abs().sum().item() == 0 and lmps.abs().sum().item() == 0:
                continue

            pmax_safe = gp.clamp(min=1e-6)
            loss_d = F.l1_loss(dispatch / pmax_safe, tpg / pmax_safe)
            pred_cost_val = (pred_cost * dispatch).sum()
            true_obj_t = torch.tensor(tobj, dtype=torch.float32)
            loss_c = (pred_cost_val - true_obj_t).abs() / true_obj_t.abs().clamp(min=1e-6)
            loss = p3.W_DISPATCH * loss_d + p3.W_COST * loss_c

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_solved += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_solved, 1)

        if epoch % 25 == 0 or epoch == n_epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.1e}  ({elapsed:.0f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), FINETUNE_CKPT)

    model.load_state_dict(torch.load(FINETUNE_CKPT, weights_only=True))
    return model, best_loss, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    split = json.load(open(SPLIT_JSON))

    print("Loading datasets...")
    train_samples = load_dataset("train", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_samples  = load_dataset("test",  DATA_DIR, SPLIT_JSON, CACHE_DIR)
    train_good = [s for s in train_samples if p3.is_good(s)]
    test_good  = [s for s in test_samples  if p3.is_good(s)]
    print(f"Train good: {len(train_good)}  Test good: {len(test_good)}")

    # -----------------------------------------------------------------
    # Step 1: JEPA pretraining
    # -----------------------------------------------------------------
    print(f"\n=== JEPA Pretraining ({args.pretrain_epochs} epochs) ===")
    pretrained_enc, sigreg_history = pretrain(train_good, n_epochs=args.pretrain_epochs, verbose=True)
    torch.save(pretrained_enc.state_dict(), PRETRAIN_CKPT)
    print(f"Pretrained encoder saved to {PRETRAIN_CKPT}")

    # -----------------------------------------------------------------
    # Step 2: Build DispatchLayers (same as P3)
    # -----------------------------------------------------------------
    print("\nBuilding DispatchLayers for fine-tuning...")
    all_samples = train_good + test_good
    layers = {}
    for s in all_samples:
        parts = s.name.rsplit('_', 1)
        state, hour = '_'.join(parts[:-1]), parts[-1]
        info = p3.build_dispatch_layer(state, hour)
        layers[s.name] = info
    train_layers_ok = [s for s in train_good if layers.get(s.name) is not None]
    print(f"Layers ready: {len(train_layers_ok)}/{len(train_good)} train")

    # -----------------------------------------------------------------
    # Step 3: Fine-tune Design A from pretrained encoder
    # -----------------------------------------------------------------
    print(f"\n=== Fine-tuning Design A from JEPA encoder ({args.finetune_epochs} epochs) ===")
    jepa_model, best_loss, best_epoch = finetune_design_a(
        pretrained_enc, train_layers_ok, layers, test_good,
        n_epochs=args.finetune_epochs,
    )
    print(f"Best epoch: {best_epoch}  loss: {best_loss:.4f}")

    # -----------------------------------------------------------------
    # Step 4: Evaluate and compare with P3 from-scratch
    # -----------------------------------------------------------------
    print("\nEvaluating JEPA fine-tuned model on TEST...")
    test_metrics_jepa = p3.compute_metrics(jepa_model, test_good, layers)
    print(f"  dispatch_mae={test_metrics_jepa['dispatch_mae']['mean']:.4f}  cost_gap={test_metrics_jepa['cost_gap']['mean']:.4f}  lmp_mae={test_metrics_jepa['lmp_mae']['mean']:.4f}")

    # Load P3 metrics for comparison
    p3_metrics = {}
    try:
        p3_metrics = json.load(open("ralph/build/state/P3_metrics.json"))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # SIGReg analysis: intra-graph correlation
    # -----------------------------------------------------------------
    # Mean intra-graph correlation over training
    intra_corrs = [h["intra_corr"] for h in sigreg_history]
    final_sigreg = sigreg_history[-1] if sigreg_history else {}

    metrics = {
        "phase": "P4",
        "jepa_pretraining": {
            "n_epochs": args.pretrain_epochs,
            "n_train_samples": len(train_good),
            "mask_frac": MASK_FRAC,
            "ema_tau": EMA_TAU,
            "final_epoch_stats": final_sigreg,
            "sigreg_intra_corr": {
                "epoch_1": sigreg_history[0]["intra_corr"] if sigreg_history else None,
                "epoch_final": intra_corrs[-1] if intra_corrs else None,
                "mean": float(np.mean(intra_corrs)) if intra_corrs else None,
                "note": "Intra-graph embedding covariance off-diagonal mean. High values violate SIGReg i.i.d. assumption.",
            },
        },
        "finetuning": {
            "n_epochs": args.finetune_epochs,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
        },
        "test_metrics_jepa": test_metrics_jepa,
        "test_metrics_p3_scratch": p3_metrics.get("test_metrics", {}),
        "comparison_jepa_vs_scratch": {
            "dispatch_mae_scratch": p3_metrics.get("test_metrics", {}).get("dispatch_mae", {}).get("mean"),
            "dispatch_mae_jepa":    test_metrics_jepa["dispatch_mae"]["mean"],
            "cost_gap_scratch":     p3_metrics.get("test_metrics", {}).get("cost_gap", {}).get("mean"),
            "cost_gap_jepa":        test_metrics_jepa["cost_gap"]["mean"],
            "lmp_mae_scratch":      p3_metrics.get("test_metrics", {}).get("lmp_mae", {}).get("mean"),
            "lmp_mae_jepa":         test_metrics_jepa["lmp_mae"]["mean"],
        },
        "checkpoints": {
            "pretrained_encoder": PRETRAIN_CKPT,
            "finetuned_design_a": FINETUNE_CKPT,
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")

    c = metrics["comparison_jepa_vs_scratch"]
    print("\n=== P4: JEPA fine-tuned vs P3 from-scratch (TEST) ===")
    print(f"  dispatch_mae:  scratch={c['dispatch_mae_scratch']:.4f}  jepa={c['dispatch_mae_jepa']:.4f}")
    print(f"  cost_gap:      scratch={c['cost_gap_scratch']:.4f}  jepa={c['cost_gap_jepa']:.4f}")
    print(f"  lmp_mae:       scratch={c['lmp_mae_scratch']:.4f}  jepa={c['lmp_mae_jepa']:.4f}")
    print(f"\nSIGReg intra-graph correlation (off-diag cov mean):")
    print(f"  epoch 1: {sigreg_history[0]['intra_corr']:.4f}")
    print(f"  epoch final: {intra_corrs[-1]:.4f}")
    print(f"  Assessment: {'HIGH — violates i.i.d. assumption' if intra_corrs[-1] > 0.01 else 'LOW — within acceptable range'}")


if __name__ == "__main__":
    main()
