"""
Improvement experiment: cost-mapping fix + auxiliary gen-cost supervised loss.

Two changes vs P3 baseline:
  1. gridsfm_zap.py now imputes missing gen costs from the per-case median of
     real costs instead of defaulting to 1.0. This recovers vermont, new_mexico
     (train) and massachusetts (test) from the quality filter.
  2. Added auxiliary supervised loss: MSE(pred_cost_normalized, true_gen_cost).
     This directly trains the decoder toward the correct cost ordering before
     the OPF pathway provides gradient signal.

Metrics written to: state/improve_cost_fix_metrics.json
Checkpoint: state/checkpoints/design_a_v2.pt
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
import torch.nn.functional as F

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cvxpy as cp
import zap
from zap import DispatchLayer
from gridsfm_zap import load_case

from dataset import load_dataset, GraphSample, _device_index
from models import MPGNNEncoder, MLP

# Reuse DispatchFunc from train_design_a
from train_design_a import DispatchFunc, DesignA, dispatch_through_layer, to_torch, is_good, build_dispatch_layer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/improve_cost_fix_metrics.json"
CKPT_PATH   = "ralph/build/state/checkpoints/design_a_v2.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
LR          = 5e-4
LR_MIN      = 1e-5
N_EPOCHS    = 120          # slightly more epochs to utilize extra data
W_DISPATCH  = 1.0
W_COST      = 0.1
W_AUX_COST  = 0.5         # auxiliary supervised loss weight on true gen costs
LMP_MAX_THRESHOLD = 100.0
REGULARIZE  = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(model: nn.Module, samples: list, layers: dict):
    model.eval()
    disp_maes, disp_frac_maes, cost_gaps, lmp_maes, pb_errs = [], [], [], [], []
    decoder_devs = []

    with torch.no_grad():
        for s in samples:
            key = s.name
            if key not in layers:
                continue
            layer_info = layers[key]
            if layer_info is None:
                continue
            layer, gi, true_cost_np = layer_info
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj = to_torch(s)

            pred_cost = model(nf, ei, ef, gb)
            dispatch, lmps, obj_tensor = dispatch_through_layer(pred_cost, layer, gi)

            pred_pg_np = dispatch.cpu().numpy()
            pred_lmp_np = lmps.cpu().numpy()
            pred_cost_np = pred_cost.cpu().numpy()
            true_pg = s.target_pg
            true_lmp = s.target_lmp

            if np.all(pred_pg_np == 0) and np.all(pred_lmp_np == 0):
                continue

            disp_mae = float(np.mean(np.abs(pred_pg_np - true_pg)))
            pmax = s.gen_pmax
            pmax_safe = np.where(pmax > 1e-6, pmax, 1.0)
            disp_frac_mae = float(np.mean(np.abs(pred_pg_np / pmax_safe - true_pg / pmax_safe)))

            true_cost_eval = float(np.dot(s.gen_cost, pred_pg_np))
            cost_gap = abs(true_cost_eval - tobj) / max(abs(tobj), 1e-6)

            lmp_mae = float(np.mean(np.abs(pred_lmp_np - true_lmp)))

            total_gen = float(np.sum(true_pg))
            if total_gen > 1e-6:
                pb_err = abs(float(np.sum(pred_pg_np)) - total_gen) / total_gen
                pb_errs.append(pb_err)

            tc = true_cost_np
            tc_safe = np.where(np.abs(tc) > 1e-6, np.abs(tc), 1.0)
            dev = float(np.mean(np.abs(pred_cost_np - tc) / tc_safe))
            decoder_devs.append(dev)

            disp_maes.append(disp_mae)
            disp_frac_maes.append(disp_frac_mae)
            cost_gaps.append(cost_gap)
            lmp_maes.append(lmp_mae)

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
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading datasets (with rebuilt cache for fixed states)...")
    from dataset import build_cache
    build_cache(DATA_DIR, SPLIT_JSON, CACHE_DIR, verbose=True)

    train_samples = load_dataset("train", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_samples  = load_dataset("test",  DATA_DIR, SPLIT_JSON, CACHE_DIR)

    train_good = [s for s in train_samples if is_good(s)]
    test_good  = [s for s in test_samples  if is_good(s)]
    train_bad  = [s.name for s in train_samples if not is_good(s)]
    test_bad   = [s.name for s in test_samples  if not is_good(s)]
    print(f"Train good: {len(train_good)}/{len(train_samples)}  (filtered: {train_bad})")
    print(f"Test  good: {len(test_good)}/{len(test_samples)}   (filtered: {test_bad})")

    print("\nBuilding DispatchLayers...")
    all_samples = train_good + test_good
    layers = {}
    for s in all_samples:
        parts = s.name.rsplit('_', 1)
        state, hour = '_'.join(parts[:-1]), parts[-1]
        info = build_dispatch_layer(state, hour)
        layers[s.name] = info
        if info is None:
            print(f"  {s.name}: FAILED")

    train_layers_ok = [s for s in train_good if layers.get(s.name) is not None]
    print(f"DispatchLayers ready: {len(train_layers_ok)}/{len(train_good)} train")

    model = DesignA(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    rng = np.random.default_rng(42)
    best_loss = float("inf")
    best_epoch = 0
    train_losses = []

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
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
            nf, ei, ef, gb, gp, gc, tpg, tlmp, tobj = to_torch(s)

            optimizer.zero_grad()
            pred_cost = model(nf, ei, ef, gb)
            dispatch, lmps, obj_out = dispatch_through_layer(pred_cost, layer, gi)

            if dispatch.abs().sum().item() == 0 and lmps.abs().sum().item() == 0:
                continue

            # Dispatch MAE loss (normalized by pmax)
            pmax_safe = gp.clamp(min=1e-6)
            loss_d = F.l1_loss(dispatch / pmax_safe, tpg / pmax_safe)

            # Cost alignment loss
            pred_cost_val = (pred_cost * dispatch).sum()
            true_obj_t = torch.tensor(tobj, dtype=torch.float32)
            loss_c = (pred_cost_val - true_obj_t).abs() / true_obj_t.abs().clamp(min=1e-6)

            # Auxiliary supervised loss: directly supervise gen cost predictions
            # gc = true normalized gen costs (from fixed gridsfm_zap.py)
            loss_aux = F.mse_loss(pred_cost, gc)

            loss = W_DISPATCH * loss_d + W_COST * loss_c + W_AUX_COST * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_solved += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_solved, 1)
        train_losses.append(avg_loss)

        if epoch % 20 == 0 or epoch == N_EPOCHS:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  solved={n_solved}/{len(train_layers_ok)}  lr={scheduler.get_last_lr()[0]:.1e}  ({elapsed:.0f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)

    elapsed_total = time.time() - t0
    print(f"\nTraining done in {elapsed_total:.1f}s.  Best epoch: {best_epoch}  loss: {best_loss:.4f}")

    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))

    print("\nEvaluating on TRAIN (good + layer-ok)...")
    train_metrics = compute_metrics(model, train_layers_ok, layers)
    print(f"  dispatch_mae={train_metrics['dispatch_mae']['mean']:.4f}  "
          f"cost_gap={train_metrics['cost_gap']['mean']:.4f}  "
          f"lmp_mae={train_metrics['lmp_mae']['mean']:.4f}")

    print("Evaluating on TEST...")
    test_metrics = compute_metrics(model, test_good, layers)
    print(f"  dispatch_mae={test_metrics['dispatch_mae']['mean']:.4f}  "
          f"cost_gap={test_metrics['cost_gap']['mean']:.4f}  "
          f"lmp_mae={test_metrics['lmp_mae']['mean']:.4f}")

    # Load P3 baseline for comparison
    p3 = {}
    try:
        p3 = json.load(open("ralph/build/state/P3_metrics.json"))
    except Exception:
        pass

    metrics = {
        "experiment": "improve_cost_fix",
        "changes": [
            "gridsfm_zap.py: impute missing gen costs from per-case median of real costs",
            "train: add auxiliary supervised loss on gen costs (W_AUX=0.5)",
            "train: 120 epochs (vs 100 baseline)",
        ],
        "model": {"hidden_dim": HIDDEN_DIM, "n_mp_layers": N_LAYERS, "n_params": n_params},
        "training": {
            "n_epochs": N_EPOCHS, "best_epoch": best_epoch,
            "best_train_loss": best_loss, "elapsed_s": elapsed_total,
            "w_dispatch": W_DISPATCH, "w_cost": W_COST, "w_aux_cost": W_AUX_COST,
        },
        "quality_filter": {
            "lmp_threshold": LMP_MAX_THRESHOLD,
            "train_n_good": len(train_good),
            "train_filtered_out": train_bad,
            "test_n_good": len(test_good),
            "test_filtered_out": test_bad,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint": CKPT_PATH,
        "comparison_vs_P3": {
            "test_dispatch_mae_P3":  p3.get("test_metrics", {}).get("dispatch_mae", {}).get("mean"),
            "test_dispatch_mae_v2":  test_metrics["dispatch_mae"]["mean"],
            "test_cost_gap_P3":      p3.get("test_metrics", {}).get("cost_gap", {}).get("mean"),
            "test_cost_gap_v2":      test_metrics["cost_gap"]["mean"],
            "test_lmp_mae_P3":       p3.get("test_metrics", {}).get("lmp_mae", {}).get("mean"),
            "test_lmp_mae_v2":       test_metrics["lmp_mae"]["mean"],
            "test_n_samples_P3":     p3.get("test_metrics", {}).get("n_evaluated"),
            "test_n_samples_v2":     test_metrics["n_evaluated"],
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")

    comp = metrics["comparison_vs_P3"]
    print("\n=== v2 vs P3 COMPARISON (TEST) ===")
    print(f"  n_test_samples: P3={comp['test_n_samples_P3']}  v2={comp['test_n_samples_v2']}")
    print(f"  dispatch_mae:   P3={comp['test_dispatch_mae_P3']:.4f}  v2={comp['test_dispatch_mae_v2']:.4f}")
    print(f"  cost_gap:       P3={comp['test_cost_gap_P3']:.4f}  v2={comp['test_cost_gap_v2']:.4f}")
    print(f"  lmp_mae:        P3={comp['test_lmp_mae_P3']:.4f}  v2={comp['test_lmp_mae_v2']:.4f}")


if __name__ == "__main__":
    train()
