"""
Improvement experiment v4: tune aux cost loss weight for balance.

v2 (W_AUX=0.5) gave best cost_gap (0.473) but worst LMP_MAE (1.664).
v3 (W_AUX=0.1 + gen feats) gave best LMP_MAE (1.291) but worst cost_gap (0.866).

v4: W_AUX=0.15 (between v2 and v3) without gen features (which hurt generalization).
Also runs longer (200 epochs) since training loss had not converged.

Hypothesis: a moderate aux weight should balance cost-gap and LMP improvements.

Metrics written to: state/improve_v4_metrics.json
Checkpoint: state/checkpoints/design_a_v4.pt
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

from dataset import load_dataset
from train_design_a import (DesignA, dispatch_through_layer, to_torch,
                             is_good, build_dispatch_layer, compute_metrics)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/improve_v4_metrics.json"
CKPT_PATH   = "ralph/build/state/checkpoints/design_a_v4.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
LR          = 5e-4
LR_MIN      = 1e-5
N_EPOCHS    = 200
W_DISPATCH  = 1.0
W_COST      = 0.1
W_AUX_COST  = 0.15
LMP_MAX_THRESHOLD = 100.0
REGULARIZE  = 1e-6


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading datasets...")
    train_samples = load_dataset("train", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_samples  = load_dataset("test",  DATA_DIR, SPLIT_JSON, CACHE_DIR)

    train_good = [s for s in train_samples if is_good(s)]
    test_good  = [s for s in test_samples  if is_good(s)]
    train_bad  = [s.name for s in train_samples if not is_good(s)]
    print(f"Train good: {len(train_good)}/{len(train_samples)}  (filtered: {train_bad})")
    print(f"Test  good: {len(test_good)}/{len(test_samples)}   (filtered: [])")

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

            pmax_safe = gp.clamp(min=1e-6)
            loss_d = F.l1_loss(dispatch / pmax_safe, tpg / pmax_safe)

            pred_cost_val = (pred_cost * dispatch).sum()
            true_obj_t = torch.tensor(tobj, dtype=torch.float32)
            loss_c = (pred_cost_val - true_obj_t).abs() / true_obj_t.abs().clamp(min=1e-6)

            loss_aux = F.mse_loss(pred_cost, gc)

            loss = W_DISPATCH * loss_d + W_COST * loss_c + W_AUX_COST * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_solved += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_solved, 1)

        if epoch % 40 == 0 or epoch == N_EPOCHS:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  n={n_solved}  lr={scheduler.get_last_lr()[0]:.1e}  ({elapsed:.0f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_PATH)

    elapsed_total = time.time() - t0
    print(f"\nTraining done in {elapsed_total:.1f}s.  Best epoch: {best_epoch}")

    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))

    print("\nEvaluating TRAIN...")
    train_metrics = compute_metrics(model, train_layers_ok, layers)
    print(f"  cost_gap={train_metrics['cost_gap']['mean']:.4f}  lmp_mae={train_metrics['lmp_mae']['mean']:.4f}")

    print("Evaluating TEST...")
    test_metrics = compute_metrics(model, test_good, layers)
    orig_8 = [s for s in test_good if 'massachusetts' not in s.name]
    m_orig8 = compute_metrics(model, orig_8, layers)
    print(f"  All 10: cost_gap={test_metrics['cost_gap']['mean']:.4f}  lmp_mae={test_metrics['lmp_mae']['mean']:.4f}")
    print(f"  Orig 8: cost_gap={m_orig8['cost_gap']['mean']:.4f}  lmp_mae={m_orig8['lmp_mae']['mean']:.4f}")

    metrics = {
        "experiment": "improve_v4",
        "changes": [
            "aux cost loss weight: 0.15 (between v2=0.5 and v3=0.1)",
            "200 epochs (vs 120/150)",
            "no gen features (v3 showed they hurt generalization)",
        ],
        "training": {
            "n_epochs": N_EPOCHS, "best_epoch": best_epoch,
            "best_train_loss": best_loss, "elapsed_s": elapsed_total,
        },
        "quality_filter": {
            "train_n_good": len(train_good), "train_filtered_out": train_bad,
            "test_n_good": len(test_good),
        },
        "train_metrics": train_metrics,
        "test_metrics_all10": test_metrics,
        "test_metrics_orig8": m_orig8,
        "checkpoint": CKPT_PATH,
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")
    print("\n=== Summary (original 8 test) ===")
    print(f"  P3 baseline: cost_gap=0.634  lmp_mae=1.323")
    print(f"  v2 (W_AUX=0.5): cost_gap=0.473  lmp_mae=1.664")
    print(f"  v4 (W_AUX=0.15): cost_gap={m_orig8['cost_gap']['mean']:.4f}  lmp_mae={m_orig8['lmp_mae']['mean']:.4f}")


if __name__ == "__main__":
    train()
