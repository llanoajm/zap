"""
P2 — Train the supervised GNN surrogate baseline.

Trains BaselineSurrogate to predict (LMP, dispatch) directly from graph
features, without any solver.  This is the "GridSFM-style" control.

Quality filter
--------------
Exclude samples where max(|target_lmp|) > 100 (sign of failed cost normalization;
e.g. vermont_04h inaccurate solve, massachusetts/new_mexico/utah cost extremes).
These are noted in metrics.

Training
--------
Loss = w_d * MAE(pred_pg / pmax, true_pg / pmax)
     + w_l * MAE(pred_lmp / lmp_scale, true_lmp / lmp_scale)
  where lmp_scale = per-sample mean |true_lmp|  (prevents large-LMP domination)

Adam, lr=1e-3, cosine-decay to 1e-4, 150 epochs, batch=1 (one graph per step).

Evaluation (TEST states only)
-----------------------------
  dispatch_mae      — mean |pred_pg - true_pg| [p.u.]
  dispatch_frac_mae — mean |pred_pg/pmax - true_pg/pmax|
  cost_gap          — |sum(cost*pred_pg) - true_obj| / |true_obj|
  lmp_mae           — mean |pred_lmp - true_lmp| [$/p.u.]
  power_balance_err — |sum(pred_pg) - total_load| / total_load
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import load_dataset, GraphSample
from models import BaselineSurrogate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/P2_metrics.json"
CKPT_PATH   = "ralph/build/state/checkpoints/baseline.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
LR          = 1e-3
LR_MIN      = 1e-4
N_EPOCHS    = 150
W_DISPATCH  = 1.0
W_LMP       = 0.2
LMP_MAX_THRESHOLD = 100.0   # filter samples with extreme LMPs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_torch(sample: GraphSample, device: str = "cpu"):
    return (
        torch.tensor(sample.node_feats, dtype=torch.float32, device=device),
        torch.tensor(sample.edge_index, dtype=torch.long, device=device),
        torch.tensor(sample.edge_feats, dtype=torch.float32, device=device),
        torch.tensor(sample.gen_bus, dtype=torch.long, device=device),
        torch.tensor(sample.gen_pmax, dtype=torch.float32, device=device),
        torch.tensor(sample.gen_cost, dtype=torch.float32, device=device),
        torch.tensor(sample.target_pg, dtype=torch.float32, device=device),
        torch.tensor(sample.target_lmp, dtype=torch.float32, device=device),
        sample.target_obj,
    )


def is_good_sample(sample: GraphSample) -> bool:
    return float(np.max(np.abs(sample.target_lmp))) < LMP_MAX_THRESHOLD


def compute_metrics(model: nn.Module, samples: list[GraphSample], device: str = "cpu"):
    model.eval()
    disp_maes, disp_frac_maes, cost_gaps, lmp_maes, pb_errs = [], [], [], [], []
    with torch.no_grad():
        for s in samples:
            if not is_good_sample(s):
                continue
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj = to_torch(s, device)
            pred_lmp, pred_pg = model(nf, ei, ef, gb, gp, gc)
            pred_pg_np = pred_pg.cpu().numpy()
            pred_lmp_np = pred_lmp.cpu().numpy()
            true_pg = s.target_pg
            true_lmp = s.target_lmp

            disp_mae = float(np.mean(np.abs(pred_pg_np - true_pg)))
            pmax = s.gen_pmax
            pmax_safe = np.where(pmax > 1e-6, pmax, 1.0)
            disp_frac_mae = float(np.mean(np.abs(pred_pg_np / pmax_safe - true_pg / pmax_safe)))

            pred_cost = float(np.dot(s.gen_cost, pred_pg_np))
            cost_gap = abs(pred_cost - tobj) / max(abs(tobj), 1e-6)

            lmp_mae = float(np.mean(np.abs(pred_lmp_np - true_lmp)))

            # Power balance: total load from node features (load_frac × total_load)
            # We can estimate total load from the data: node_feats col0 sums to ≤1 fraction
            # but we don't have raw total load directly; use sum(true_pg) ≈ total_load
            total_load = float(np.sum(true_pg))  # at optimum, gen ≈ load for non-curtailed
            if total_load > 1e-6:
                pb_err = abs(float(np.sum(pred_pg_np)) - total_load) / total_load
            else:
                pb_err = float("nan")

            disp_maes.append(disp_mae)
            disp_frac_maes.append(disp_frac_mae)
            cost_gaps.append(cost_gap)
            lmp_maes.append(lmp_mae)
            if not np.isnan(pb_err):
                pb_errs.append(pb_err)

    def _stats(arr):
        if not arr:
            return {"mean": None, "median": None}
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

    return {
        "n_evaluated": len(disp_maes),
        "dispatch_mae":      _stats(disp_maes),
        "dispatch_frac_mae": _stats(disp_frac_maes),
        "cost_gap":          _stats(cost_gaps),
        "lmp_mae":           _stats(lmp_maes),
        "power_balance_err": _stats(pb_errs),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading datasets...")
    train_samples = load_dataset("train", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_samples  = load_dataset("test",  DATA_DIR, SPLIT_JSON, CACHE_DIR)

    train_good = [s for s in train_samples if is_good_sample(s)]
    test_good  = [s for s in test_samples  if is_good_sample(s)]
    train_bad  = [s.name for s in train_samples if not is_good_sample(s)]
    test_bad   = [s.name for s in test_samples  if not is_good_sample(s)]

    print(f"Train: {len(train_good)}/{len(train_samples)} good  (filtered: {train_bad})")
    print(f"Test:  {len(test_good)}/{len(test_samples)} good   (filtered: {test_bad})")

    model = BaselineSurrogate(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    rng = np.random.default_rng(42)
    train_losses = []
    best_val_loss = float("inf")
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        order = rng.permutation(len(train_good))
        for idx in order:
            s = train_good[idx]
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj = to_torch(s)
            optimizer.zero_grad()
            pred_lmp, pred_pg = model(nf, ei, ef, gb, gp, gc)

            # Dispatch loss: fraction of pmax
            pmax_safe = gp.clamp(min=1e-6)
            loss_d = nn.functional.l1_loss(pred_pg / pmax_safe, tpg / pmax_safe)

            # LMP loss: normalize by per-sample mean |true_lmp|
            lmp_scale = tlmp.abs().mean().clamp(min=1e-6)
            loss_l = nn.functional.l1_loss(pred_lmp / lmp_scale, tlmp / lmp_scale)

            loss = W_DISPATCH * loss_d + W_LMP * loss_l
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_good)
        train_losses.append(avg_loss)

        if epoch % 25 == 0 or epoch == N_EPOCHS:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)")

        # Save best on (first eval) simple train loss proxy
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)

    elapsed_total = time.time() - t0
    print(f"\nTraining done in {elapsed_total:.1f}s.  Best epoch: {best_epoch}  loss: {best_val_loss:.4f}")

    # Load best checkpoint
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))

    # Evaluate
    print("\nEvaluating on TRAIN (good only)...")
    train_metrics = compute_metrics(model, train_good)
    print(f"  dispatch_mae={train_metrics['dispatch_mae']['mean']:.4f}  cost_gap={train_metrics['cost_gap']['mean']:.4f}  lmp_mae={train_metrics['lmp_mae']['mean']:.4f}")

    print("Evaluating on TEST (good only)...")
    test_metrics = compute_metrics(model, test_good)
    print(f"  dispatch_mae={test_metrics['dispatch_mae']['mean']:.4f}  cost_gap={test_metrics['cost_gap']['mean']:.4f}  lmp_mae={test_metrics['lmp_mae']['mean']:.4f}")

    metrics = {
        "phase": "P2",
        "model": {"hidden_dim": HIDDEN_DIM, "n_mp_layers": N_LAYERS, "n_params": n_params},
        "training": {
            "n_epochs": N_EPOCHS, "best_epoch": best_epoch,
            "best_train_loss": best_val_loss,
            "elapsed_s": elapsed_total,
        },
        "quality_filter": {
            "lmp_threshold": LMP_MAX_THRESHOLD,
            "train_filtered_out": train_bad,
            "test_filtered_out": test_bad,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint": CKPT_PATH,
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")
    print(f"\n=== P2 SUMMARY (TEST) ===")
    print(f"  n_evaluated:     {test_metrics['n_evaluated']}")
    print(f"  dispatch_mae:    {test_metrics['dispatch_mae']['mean']:.4f}  (median {test_metrics['dispatch_mae']['median']:.4f})")
    print(f"  disp_frac_mae:   {test_metrics['dispatch_frac_mae']['mean']:.4f}")
    print(f"  cost_gap:        {test_metrics['cost_gap']['mean']:.4f}  (median {test_metrics['cost_gap']['median']:.4f})")
    print(f"  lmp_mae:         {test_metrics['lmp_mae']['mean']:.4f}")
    print(f"  power_bal_err:   {test_metrics['power_balance_err']['mean']:.4f}")


if __name__ == "__main__":
    train()
