"""
Improvement experiment v13: residual-capacity interaction feature in decoder.

Root cause of connecticut_16h LMP error (all iter-3 experiments):
  - gen 30: dc_frac=0.0, pmax=4.6, true cost=6.39 (CHEAP but not dispatched by DC ref)
  - Peakers: dc_frac=0.1, pmax=0.2-4.1, true cost=7.0-7.8 (EXPENSIVE, partially dispatched)
  - Model sees dc_frac=0.0 → predicts gen 30 expensive (from training signal)
  - Model sees dc_frac=0.1 → predicts peakers cheap

Key insight: the ambiguity in dc_frac=0.0 is broken by UNCOMMITTED CAPACITY.
  - "Not dispatched because expensive": dc_frac=0.0 AND pmax is small relative to system
  - "Not dispatched because not needed": dc_frac=0.0 AND pmax is LARGE relative to system

New decoder feature:
  dc_res_cap_norm = (1 - dc_frac) × pmax / max_pmax_in_system
    gen 30  (dc_frac=0.0, pmax=4.6, max_pmax=4.6): dc_res_cap_norm = 1.0
    peakers (dc_frac=0.1, pmax=0.2-4.1):            dc_res_cap_norm = 0.04-0.80

This interaction feature explicitly encodes "large generator with lots of uncommitted
capacity" → model can learn this predicts a cheap generator not yet needed.

Architecture: same as v9/v11 (node_in=5, edge_in=3) but decoder gets 3 scalars:
  [h_gen || dc_frac || dc_res_cap_norm]
  MLP(D + 2, D // 2, 1, n_layers=2)

W_AUX=1.5 (intermediate: v9=0.5, v11=2.0). W_RANK=0.2. 150 epochs.

Metrics: state/improve_v13_metrics.json
Checkpoint: state/checkpoints/design_a_v13.pt
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import load_dataset
from models import MPGNNEncoder, MLP
from train_design_a import dispatch_through_layer, is_good, build_dispatch_layer

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/improve_v13_metrics.json"
CKPT_PATH   = "ralph/build/state/checkpoints/design_a_v13.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
LR          = 5e-4
LR_MIN      = 1e-5
N_EPOCHS    = 150
W_DISPATCH  = 1.0
W_COST      = 0.1
W_AUX_COST  = 1.5   # between v9 (0.5) and v11 (2.0)
W_RANK      = 0.2
RANK_MARGIN = 0.05


class DesignAv13(nn.Module):
    """
    GNN encoder + decoder conditioned on [dc_frac || dc_res_cap_norm].

    dc_res_cap_norm = (1 - dc_frac) * pmax / max_pmax_system
    Encodes uncommitted capacity — distinguishes large cheap generators
    (dc_frac=0, large pmax → res_cap ≈ 1) from expensive generators
    (dc_frac=0, small pmax → res_cap ≈ 0.1) and peakers
    (dc_frac=0.1, any pmax → res_cap = 0.9*(pmax/max_pmax)).
    """
    def __init__(self, hidden_dim=64, n_mp_layers=3):
        super().__init__()
        D = hidden_dim
        self.encoder = MPGNNEncoder(node_in=5, edge_in=3, hidden_dim=D, n_layers=n_mp_layers)
        self.cost_decoder = MLP(D + 2, D // 2, 1, n_layers=2)  # +2: dc_frac + dc_res_cap

    def forward(self, node_feats, edge_index, edge_feats, gen_bus,
                gen_dc_frac, dc_res_cap_norm):
        h = self.encoder(node_feats, edge_index, edge_feats)
        h_gen = h[gen_bus]
        feats = torch.stack([gen_dc_frac, dc_res_cap_norm], dim=-1)  # (n_gen, 2)
        raw = self.cost_decoder(torch.cat([h_gen, feats], dim=-1)).squeeze(-1)
        return F.softplus(raw)


def pairwise_rank_loss(pred_cost, true_cost, margin=0.05):
    n = pred_cost.shape[0]
    if n <= 1:
        return torch.tensor(0.0)
    tc = true_cost.detach()
    mask = tc.unsqueeze(1) < tc.unsqueeze(0)
    pc_row = pred_cost.unsqueeze(1).expand(n, n)
    pc_col = pred_cost.unsqueeze(0).expand(n, n)
    return (F.relu(pc_row - pc_col + margin) * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def compute_dc_res_cap(dc_frac, pmax):
    """Compute (1 - dc_frac) * pmax / max_pmax, returning torch tensor."""
    res = (1.0 - dc_frac) * pmax
    max_p = pmax.max().clamp(min=1e-6)
    return res / max_p


def to_torch(s):
    dc_frac = torch.tensor(s.gen_dc_frac, dtype=torch.float32)
    pmax    = torch.tensor(s.gen_pmax,    dtype=torch.float32)
    dc_res_cap = compute_dc_res_cap(dc_frac, pmax)
    return (
        torch.tensor(s.node_feats, dtype=torch.float32),
        torch.tensor(s.edge_index, dtype=torch.long),
        torch.tensor(s.edge_feats, dtype=torch.float32),
        torch.tensor(s.gen_bus,    dtype=torch.long),
        torch.tensor(s.gen_pmax,   dtype=torch.float32),
        torch.tensor(s.gen_cost,   dtype=torch.float32),
        torch.tensor(s.target_pg,  dtype=torch.float32),
        torch.tensor(s.target_lmp, dtype=torch.float32),
        s.target_obj,
        dc_frac,
        dc_res_cap,
    )


def compute_metrics(model, samples, layers):
    model.eval()
    disp_maes, disp_frac_maes, cost_gaps, lmp_maes, pb_errs, decoder_devs = [], [], [], [], [], []
    per_sample = {}

    with torch.no_grad():
        for s in samples:
            if s.name not in layers or layers[s.name] is None:
                continue
            layer, gi, true_cost_np = layers[s.name]
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj, dc_frac, dc_res_cap = to_torch(s)

            pred_cost = model(nf, ei, ef, gb, dc_frac, dc_res_cap)
            dispatch, lmps, _ = dispatch_through_layer(pred_cost, layer, gi)
            pred_pg = dispatch.numpy()
            pred_lmp = lmps.numpy()

            if np.all(pred_pg == 0) and np.all(pred_lmp == 0):
                continue

            disp_mae = float(np.mean(np.abs(pred_pg - s.target_pg)))
            pmax_safe = np.where(s.gen_pmax > 1e-6, s.gen_pmax, 1.0)
            disp_frac_mae = float(np.mean(np.abs(pred_pg / pmax_safe - s.target_pg / pmax_safe)))
            true_cost_eval = float(np.dot(s.gen_cost, pred_pg))
            cost_gap = abs(true_cost_eval - tobj) / max(abs(tobj), 1e-6)
            lmp_mae = float(np.mean(np.abs(pred_lmp - s.target_lmp)))
            total_gen = float(np.sum(s.target_pg))
            pb_err = abs(float(np.sum(pred_pg)) - total_gen) / total_gen if total_gen > 1e-6 else 0.0

            tc = true_cost_np
            pred_c = pred_cost.numpy()
            dev = float(np.mean(np.abs(pred_c - tc) / np.where(np.abs(tc) > 1e-6, np.abs(tc), 1.0)))

            disp_maes.append(disp_mae); disp_frac_maes.append(disp_frac_mae)
            cost_gaps.append(cost_gap); lmp_maes.append(lmp_mae)
            pb_errs.append(pb_err); decoder_devs.append(dev)
            per_sample[s.name] = {"cost_gap": cost_gap, "lmp_mae": lmp_mae, "power_balance_err": pb_err}

    def _s(arr):
        if not arr:
            return {"mean": None, "median": None}
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

    return {
        "n_evaluated": len(disp_maes),
        "dispatch_mae":      _s(disp_maes),
        "dispatch_frac_mae": _s(disp_frac_maes),
        "cost_gap":          _s(cost_gaps),
        "lmp_mae":           _s(lmp_maes),
        "power_balance_err": _s(pb_errs),
        "decoder_deviation": _s(decoder_devs),
        "per_sample":        per_sample,
    }


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_samples = load_dataset("train", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_samples  = load_dataset("test",  DATA_DIR, SPLIT_JSON, CACHE_DIR)
    train_good = [s for s in train_samples if is_good(s)]
    test_good  = [s for s in test_samples  if is_good(s)]
    train_bad  = [s.name for s in train_samples if not is_good(s)]
    print(f"Train good: {len(train_good)}/{len(train_samples)}  filtered: {train_bad}")
    print(f"Test  good: {len(test_good)}/{len(test_samples)}")

    # Check interaction feature correlation for first few training samples
    from scipy.stats import spearmanr
    for s in train_good[:3]:
        if s.gen_dc_frac is not None:
            dc_frac = np.asarray(s.gen_dc_frac)
            pmax    = np.asarray(s.gen_pmax)
            max_p   = pmax.max() if pmax.max() > 1e-6 else 1.0
            res_cap = (1.0 - dc_frac) * pmax / max_p
            corr_dc,  _ = spearmanr(dc_frac, s.gen_cost)
            corr_res, _ = spearmanr(res_cap, s.gen_cost)
            print(f"  {s.name}: Spearman(dc_frac, cost)={corr_dc:.3f}  "
                  f"Spearman(res_cap, cost)={corr_res:.3f}")

    print("\nBuilding DispatchLayers...")
    layers = {}
    for s in train_good + test_good:
        parts = s.name.rsplit('_', 1)
        state, hour = '_'.join(parts[:-1]), parts[-1]
        info = build_dispatch_layer(state, hour)
        layers[s.name] = info
        if info is None:
            print(f"  {s.name}: FAILED")

    train_ok = [s for s in train_good if layers.get(s.name) is not None]
    print(f"DispatchLayers ready: {len(train_ok)}/{len(train_good)}")

    model = DesignAv13(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)
    rng = np.random.default_rng(42)
    best_loss = float("inf"); best_epoch = 0

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0; n_solved = 0
        for idx in rng.permutation(len(train_ok)):
            s = train_ok[idx]
            layer_info = layers[s.name]
            if layer_info is None:
                continue
            layer, gi, _ = layer_info
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj, dc_frac, dc_res_cap = to_torch(s)

            optimizer.zero_grad()
            pred_cost = model(nf, ei, ef, gb, dc_frac, dc_res_cap)
            dispatch, lmps, obj_out = dispatch_through_layer(pred_cost, layer, gi)

            if dispatch.abs().sum().item() == 0 and lmps.abs().sum().item() == 0:
                continue

            pmax_safe = gp.clamp(min=1e-6)
            loss_d = F.l1_loss(dispatch / pmax_safe, tpg / pmax_safe)
            pred_cost_val = (pred_cost * dispatch).sum()
            true_obj_t = torch.tensor(tobj, dtype=torch.float32)
            loss_c = (pred_cost_val - true_obj_t).abs() / true_obj_t.abs().clamp(min=1e-6)
            loss_aux = F.mse_loss(pred_cost, gc)
            loss_rank = pairwise_rank_loss(pred_cost, gc, margin=RANK_MARGIN)

            loss = W_DISPATCH * loss_d + W_COST * loss_c + W_AUX_COST * loss_aux + W_RANK * loss_rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_solved += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_solved, 1)
        if epoch % 30 == 0 or epoch == N_EPOCHS:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  n={n_solved}  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  ({time.time()-t0:.0f}s)")
        if avg_loss < best_loss:
            best_loss = avg_loss; best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)

    print(f"\nDone in {time.time()-t0:.1f}s. Best epoch: {best_epoch} loss: {best_loss:.4f}")
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))

    print("\nEvaluating TRAIN...")
    train_m = compute_metrics(model, train_ok, layers)
    print(f"  cost_gap={train_m['cost_gap']['mean']:.4f}  lmp_mae={train_m['lmp_mae']['mean']:.4f}")

    print("Evaluating TEST...")
    test_m = compute_metrics(model, test_good, layers)
    orig_8 = [s for s in test_good if 'massachusetts' not in s.name]
    m8 = compute_metrics(model, orig_8, layers)
    print(f"  All 10: cost_gap={test_m['cost_gap']['mean']:.4f}  lmp_mae={test_m['lmp_mae']['mean']:.4f}")
    print(f"  Orig 8: cost_gap={m8['cost_gap']['mean']:.4f}  lmp_mae={m8['lmp_mae']['mean']:.4f}")

    # Per-sample for connecticut_16h analysis
    ct16 = m8['per_sample'].get('connecticut_16h', {})
    print(f"\n  connecticut_16h: cost_gap={ct16.get('cost_gap','?'):.3f}  "
          f"lmp_mae={ct16.get('lmp_mae','?'):.3f}")

    metrics = {
        "experiment": "improve_v13_res_cap",
        "changes": [
            "decoder input: h_gen + dc_frac + dc_res_cap_norm",
            "dc_res_cap_norm = (1 - dc_frac) * pmax / max_pmax_system",
            "W_AUX=1.5, W_RANK=0.2, 150 epochs",
            "node_in=5, edge_in=3 (same as v9/v11)",
            "Hypothesis: distinguishes 'large cheap gen not dispatched' from 'expensive gen'",
        ],
        "hyperparams": {
            "W_AUX_COST": W_AUX_COST, "W_RANK": W_RANK,
            "N_EPOCHS": N_EPOCHS, "HIDDEN_DIM": HIDDEN_DIM, "N_LAYERS": N_LAYERS,
        },
        "training": {"n_epochs": N_EPOCHS, "best_epoch": best_epoch, "best_loss": best_loss},
        "quality_filter": {"train_n_good": len(train_good), "train_filtered": train_bad,
                           "test_n_good": len(test_good)},
        "train_metrics": {k: v for k, v in train_m.items() if k != "per_sample"},
        "test_metrics_all10": {k: v for k, v in test_m.items() if k != "per_sample"},
        "test_metrics_orig8":  {k: v for k, v in m8.items()   if k != "per_sample"},
        "per_sample_orig8":    m8.get("per_sample", {}),
        "checkpoint": CKPT_PATH,
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")

    print("\n=== Summary (original 8 test) ===")
    print(f"  v9  (W_AUX=0.5): cost_gap=0.0818  lmp_mae_mean=0.8227  lmp_mae_med=0.3319")
    print(f"  v11 (W_AUX=2.0): cost_gap=0.0247  lmp_mae_mean=0.9262  lmp_mae_med=0.3503")
    print(f"  v13 (W_AUX=1.5, +res_cap): cost_gap={m8['cost_gap']['mean']:.4f}  "
          f"lmp_mae_mean={m8['lmp_mae']['mean']:.4f}  lmp_mae_med={m8['lmp_mae']['median']:.4f}")


if __name__ == "__main__":
    train()
