"""
P3 — Design A: GNN encoder + differentiable zap OPF head.

Architecture
------------
  GNN encoder → per-gen cost decoder → DispatchLayer (exact DC-OPF)
                                              ↓ implicit-function backward
  Loss: MAE(layer_dispatch/pmax , true_dispatch/pmax)
       + λ * MAE(layer_cost, true_obj) / |true_obj|

The DispatchLayer's backward uses zap's implicit-function theorem (KKT
sensitivity); the loss gradient flows all the way back to the encoder.

Compared to P2 (surrogate-without-solver), Design A:
  - Gets exact power-balance for free (solver enforces it)
  - Gets exact dual LMPs (from KKT multipliers)
  - Trains to minimize dispatch error via physics-constrained path

"Is it just a warm-start?" test
---------------------------------
  After training, measure decoder deviation:
    mean|pred_cost - true_cost| / mean|true_cost|
  If deviation ≈ 0, the decoder collapsed to passing through true params
  (trivial solution).  We report this alongside performance metrics.
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
CKPT_DIR    = "ralph/build/state/checkpoints"
METRICS_OUT = "ralph/build/state/P3_metrics.json"
CKPT_PATH   = "ralph/build/state/checkpoints/design_a.pt"

HIDDEN_DIM  = 64
N_LAYERS    = 3
LR          = 5e-4
LR_MIN      = 1e-5
N_EPOCHS    = 100
W_DISPATCH  = 1.0
W_COST      = 0.1
LMP_MAX_THRESHOLD = 100.0
REGULARIZE  = 1e-6   # implicit-function regularizer


# ---------------------------------------------------------------------------
# Custom autograd: wraps DispatchLayer forward/backward in PyTorch
# ---------------------------------------------------------------------------

class DispatchFunc(torch.autograd.Function):
    """
    Differentiable forward pass through a zap DispatchLayer.

    forward:  (pred_cost [n_gen]) → dispatch [n_gen]
    backward: implicit-function theorem via layer.backward()
    """

    @staticmethod
    def forward(ctx, pred_cost: torch.Tensor, layer: DispatchLayer, gi: int, regularize: float):
        cost_np = pred_cost.detach().cpu().numpy().reshape(-1, 1)
        try:
            y = layer(gen_cost=cost_np)
            dispatch_np = np.asarray(y.power[gi][0]).ravel()
            lmps_np = np.asarray(y.prices).ravel()
            obj = float(y.problem.value)
        except Exception:
            # Solver failed — return zero gradient signal
            n_gen = pred_cost.shape[0]
            n_bus = layer._network.num_nodes if hasattr(layer, '_network') else 1
            ctx.failed = True
            ctx.save_for_backward(pred_cost)
            ctx.extra = (None, None, None, None, None, None)
            dispatch = torch.zeros(n_gen, dtype=pred_cost.dtype)
            return dispatch, torch.zeros(1, dtype=pred_cost.dtype), torch.zeros(1, dtype=pred_cost.dtype)

        ctx.failed = False
        ctx.save_for_backward(pred_cost)
        ctx.extra = (layer, gi, cost_np, y, regularize, None)

        return (
            torch.tensor(dispatch_np, dtype=pred_cost.dtype),
            torch.tensor(lmps_np, dtype=pred_cost.dtype),
            torch.tensor([obj], dtype=pred_cost.dtype),
        )

    @staticmethod
    def backward(ctx, grad_dispatch, grad_lmps, grad_obj):
        if ctx.failed:
            pred_cost, = ctx.saved_tensors
            return torch.zeros_like(pred_cost), None, None, None

        pred_cost, = ctx.saved_tensors
        layer, gi, cost_np, y, regularize, _ = ctx.extra

        # Build dJ: gradient of loss w.r.t. dispatch
        dJ = y.package(np.zeros_like(y.vectorize()))
        grad_d_np = grad_dispatch.detach().cpu().numpy().reshape(dJ.power[gi][0].shape)
        dJ.power[gi][0] = grad_d_np

        try:
            grads = layer.backward(y, dJ, gen_cost=cost_np, regularize=regularize)
            cost_grad = torch.tensor(
                np.asarray(grads['gen_cost']).ravel(), dtype=pred_cost.dtype
            )
        except Exception:
            cost_grad = torch.zeros_like(pred_cost)

        return cost_grad, None, None, None


def dispatch_through_layer(pred_cost: torch.Tensor, layer: DispatchLayer, gi: int, regularize: float = REGULARIZE):
    return DispatchFunc.apply(pred_cost, layer, gi, regularize)


# ---------------------------------------------------------------------------
# Design A model
# ---------------------------------------------------------------------------

class DesignA(nn.Module):
    """
    GNN encoder + per-generator cost decoder.
    Produces pred_gen_cost (n_gen,) which is fed into DispatchLayer.
    """

    def __init__(self, hidden_dim: int = 64, n_mp_layers: int = 3):
        super().__init__()
        D = hidden_dim
        self.encoder = MPGNNEncoder(node_in=4, edge_in=2, hidden_dim=D, n_layers=n_mp_layers)
        # Cost decoder: gen embedding → positive cost prediction
        self.cost_decoder = MLP(D, D // 2, 1, n_layers=2)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        gen_bus: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(node_feats, edge_index, edge_feats)   # (n_bus, D)
        h_gen = h[gen_bus]                                      # (n_gen, D)
        raw = self.cost_decoder(h_gen).squeeze(-1)              # (n_gen,)
        return F.softplus(raw)                                  # positive costs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_torch(s: GraphSample):
    return (
        torch.tensor(s.node_feats, dtype=torch.float32),
        torch.tensor(s.edge_index, dtype=torch.long),
        torch.tensor(s.edge_feats, dtype=torch.float32),
        torch.tensor(s.gen_bus, dtype=torch.long),
        torch.tensor(s.gen_pmax, dtype=torch.float32),
        torch.tensor(s.gen_cost, dtype=torch.float32),
        torch.tensor(s.target_pg, dtype=torch.float32),
        torch.tensor(s.target_lmp, dtype=torch.float32),
        s.target_obj,
    )


def is_good(s: GraphSample) -> bool:
    return float(np.max(np.abs(s.target_lmp))) < LMP_MAX_THRESHOLD


def build_dispatch_layer(state: str, hour: str) -> tuple | None:
    """Load case and build DispatchLayer for gen_cost as parameter."""
    model_path = os.path.join(DATA_DIR, hour, f"{state}_model.json")
    dc_path    = os.path.join(DATA_DIR, hour, f"{state}_dc_results.json")
    if not os.path.exists(model_path):
        return None
    try:
        case = load_case(model_path, dc_path if os.path.exists(dc_path) else None)
        gi = _device_index(case.devices, zap.Generator)
        if gi is None:
            return None
        parameter_names = {'gen_cost': (gi, 'linear_cost')}
        layer = DispatchLayer(
            case.net, case.devices,
            parameter_names=parameter_names,
            time_horizon=1, solver=cp.CLARABEL, add_ground=False,
        )
        true_cost = np.asarray(case.devices[gi].linear_cost).ravel().copy()
        return layer, gi, true_cost
    except Exception as e:
        print(f"  [layer build failed for {state}_{hour}]: {e}")
        return None


def compute_metrics(model: nn.Module, samples: list[GraphSample], layers: dict):
    model.eval()
    disp_maes, disp_frac_maes, cost_gaps, lmp_maes, pb_errs = [], [], [], [], []
    decoder_devs = []  # |pred_cost - true_cost| / |true_cost| — warm-start check

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
                continue  # failed solve

            disp_mae = float(np.mean(np.abs(pred_pg_np - true_pg)))
            pmax = s.gen_pmax
            pmax_safe = np.where(pmax > 1e-6, pmax, 1.0)
            disp_frac_mae = float(np.mean(np.abs(pred_pg_np / pmax_safe - true_pg / pmax_safe)))

            # Cost gap: evaluate predicted dispatch at TRUE gen costs (apples-to-apples vs P2)
            true_cost_eval = float(np.dot(s.gen_cost, pred_pg_np))
            cost_gap = abs(true_cost_eval - tobj) / max(abs(tobj), 1e-6)

            lmp_mae = float(np.mean(np.abs(pred_lmp_np - true_lmp)))

            total_gen = float(np.sum(true_pg))
            if total_gen > 1e-6:
                pb_err = abs(float(np.sum(pred_pg_np)) - total_gen) / total_gen
                pb_errs.append(pb_err)

            # Decoder deviation (warm-start test)
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
    split = json.load(open(SPLIT_JSON))

    print("Loading datasets...")
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
        status = "ok" if info is not None else "FAILED"
        if info is None:
            print(f"  {s.name}: {status}")

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

            # Skip failed solves
            if dispatch.abs().sum().item() == 0 and lmps.abs().sum().item() == 0:
                continue

            # Dispatch MAE loss (normalized by pmax)
            pmax_safe = gp.clamp(min=1e-6)
            loss_d = F.l1_loss(dispatch / pmax_safe, tpg / pmax_safe)

            # Cost alignment loss
            pred_cost_val = (pred_cost * dispatch).sum()
            true_obj_t = torch.tensor(tobj, dtype=torch.float32)
            loss_c = (pred_cost_val - true_obj_t).abs() / true_obj_t.abs().clamp(min=1e-6)

            loss = W_DISPATCH * loss_d + W_COST * loss_c
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
    print(f"  dispatch_mae={train_metrics['dispatch_mae']['mean']:.4f}  cost_gap={train_metrics['cost_gap']['mean']:.4f}  lmp_mae={train_metrics['lmp_mae']['mean']:.4f}  decoder_dev={train_metrics['decoder_deviation']['mean']:.4f}")

    print("Evaluating on TEST...")
    test_metrics = compute_metrics(model, test_good, layers)
    print(f"  dispatch_mae={test_metrics['dispatch_mae']['mean']:.4f}  cost_gap={test_metrics['cost_gap']['mean']:.4f}  lmp_mae={test_metrics['lmp_mae']['mean']:.4f}  decoder_dev={test_metrics['decoder_deviation']['mean']:.4f}")

    # Load P2 metrics for comparison
    p2 = {}
    try:
        p2 = json.load(open("ralph/build/state/P2_metrics.json"))
    except Exception:
        pass

    metrics = {
        "phase": "P3",
        "model": {"hidden_dim": HIDDEN_DIM, "n_mp_layers": N_LAYERS, "n_params": n_params},
        "training": {
            "n_epochs": N_EPOCHS, "best_epoch": best_epoch,
            "best_train_loss": best_loss, "elapsed_s": elapsed_total,
        },
        "quality_filter": {
            "lmp_threshold": LMP_MAX_THRESHOLD,
            "train_filtered_out": train_bad,
            "test_filtered_out": test_bad,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "checkpoint": CKPT_PATH,
        "comparison_vs_P2": {
            "test_dispatch_mae_P2":  p2.get("test_metrics", {}).get("dispatch_mae", {}).get("mean"),
            "test_dispatch_mae_P3":  test_metrics["dispatch_mae"]["mean"],
            "test_cost_gap_P2":      p2.get("test_metrics", {}).get("cost_gap", {}).get("mean"),
            "test_cost_gap_P3":      test_metrics["cost_gap"]["mean"],
            "test_lmp_mae_P2":       p2.get("test_metrics", {}).get("lmp_mae", {}).get("mean"),
            "test_lmp_mae_P3":       test_metrics["lmp_mae"]["mean"],
            "test_power_balance_P2": p2.get("test_metrics", {}).get("power_balance_err", {}).get("mean"),
            "test_power_balance_P3": test_metrics["power_balance_err"]["mean"],
        },
        "warm_start_check": {
            "decoder_deviation_mean": test_metrics["decoder_deviation"]["mean"],
            "note": "If decoder_deviation ≈ 0, model collapses to pass-through of true params (trivial).",
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")

    comp = metrics["comparison_vs_P2"]
    print("\n=== P3 vs P2 COMPARISON (TEST) ===")
    print(f"  dispatch_mae:   P2={comp['test_dispatch_mae_P2']:.4f}  P3={comp['test_dispatch_mae_P3']:.4f}")
    print(f"  cost_gap:       P2={comp['test_cost_gap_P2']:.4f}  P3={comp['test_cost_gap_P3']:.4f}")
    print(f"  lmp_mae:        P2={comp['test_lmp_mae_P2']:.4f}  P3={comp['test_lmp_mae_P3']:.4f}")
    print(f"  power_balance:  P2={comp['test_power_balance_P2']:.4f}  P3={comp['test_power_balance_P3']:.4f}")
    print(f"  decoder_dev:    {metrics['warm_start_check']['decoder_deviation_mean']:.4f}  (>0 means not trivial pass-through)")


if __name__ == "__main__":
    train()
