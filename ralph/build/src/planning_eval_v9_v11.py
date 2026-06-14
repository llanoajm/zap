"""
Planning gradient evaluation for iter-3 best models (v9, v11) on delaware.

Adapts planning_improved.py for DesignAv9 architecture (node_in=5, edge_in=3,
decoder conditioned on gen_dc_frac).

Evaluates:
  - v9  (best LMP,      design_a_v9.pt)
  - v11 (best cost_gap, design_a_v11.pt)
on delaware (2 periods: 04h, 16h) — same setup as P6 / improve_planning_metrics.json.

Also evaluates on 2 test states (oregon_04h, kansas_04h) for cross-topology check.

Writes: state/improve_planning_v11_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zap
from zap import DispatchLayer
from gridsfm_zap import load_case
from dataset import _device_index, _load_sample
from models import MPGNNEncoder, MLP

METRICS_OUT = "ralph/build/state/improve_planning_v11_metrics.json"
INVEST_COST = 0.1
DATA_DIR = "ralph/build/data/raw"
CACHE_DIR = "ralph/build/state/cache"


# ---------------------------------------------------------------------------
# DesignAv9 — same architecture used by both v9 and v11 checkpoints
# ---------------------------------------------------------------------------

class DesignAv9(nn.Module):
    def __init__(self, hidden_dim=64, n_mp_layers=3):
        super().__init__()
        D = hidden_dim
        self.encoder = MPGNNEncoder(node_in=5, edge_in=3, hidden_dim=D, n_layers=n_mp_layers)
        self.cost_decoder = MLP(D + 1, D // 2, 1, n_layers=2)

    def forward(self, node_feats, edge_index, edge_feats, gen_bus, gen_dc_frac):
        h = self.encoder(node_feats, edge_index, edge_feats)
        h_gen = h[gen_bus]
        raw = self.cost_decoder(torch.cat([h_gen, gen_dc_frac.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return F.softplus(raw)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def total_cost_fn(case_list, gi_list, scale, gen_costs_list=None):
    total = 0.0
    for i, (case, gi) in enumerate(zip(case_list, gi_list)):
        devs = deepcopy(case.devices)
        orig_nc = np.asarray(devs[gi].nominal_capacity).ravel()
        devs[gi].nominal_capacity = (orig_nc * scale).reshape(-1, 1)
        if gen_costs_list is not None:
            devs[gi].linear_cost = gen_costs_list[i].reshape(-1, 1)
        try:
            out = case.net.dispatch(devs, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
            total += out.problem.value
        except Exception:
            total += 1e6
    total += float(INVEST_COST * np.sum(scale))
    return total


def finite_diff_grad(fn, scale, delta=5e-3):
    n = len(scale)
    grad = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = delta
        grad[i] = (fn(scale + e) - fn(scale - e)) / (2 * delta)
    return grad


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a / na, b / nb))


def predict_costs(model, sample):
    nf = torch.tensor(sample.node_feats, dtype=torch.float32)
    ei = torch.tensor(sample.edge_index, dtype=torch.long)
    ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
    gb = torch.tensor(sample.gen_bus, dtype=torch.long)
    dc_frac = torch.tensor(sample.gen_dc_frac, dtype=torch.float32)
    with torch.no_grad():
        return model(nf, ei, ef, gb, dc_frac).numpy()


def evaluate_state(state, hours, model_tag, model, exact_grad_cache=None):
    case_list, gi_list, samples = [], [], []
    for hour in hours:
        model_path = os.path.join(DATA_DIR, hour, f"{state}_model.json")
        dc_path = os.path.join(DATA_DIR, hour, f"{state}_dc_results.json")
        case = load_case(model_path, dc_path)
        gi = _device_index(case.devices, zap.Generator)
        case_list.append(case)
        gi_list.append(gi)
        s = _load_sample(CACHE_DIR, state, hour)
        samples.append(s)

    if any(s is None for s in samples):
        print(f"  [{state}] Missing cached samples — skipping")
        return None

    n_gen = len(np.asarray(case_list[0].devices[gi_list[0]].terminal).ravel())
    scale = np.ones(n_gen)

    # Exact gradient (or reuse cached)
    if exact_grad_cache is not None and state in exact_grad_cache:
        exact_grad = exact_grad_cache[state]
    else:
        exact_grad = finite_diff_grad(lambda s: total_cost_fn(case_list, gi_list, s), scale)
        if exact_grad_cache is not None:
            exact_grad_cache[state] = exact_grad

    # Design A gradient (using predicted costs)
    pred_costs_list = [predict_costs(model, s) for s in samples]
    da_grad = finite_diff_grad(
        lambda s: total_cost_fn(case_list, gi_list, s, gen_costs_list=pred_costs_list),
        scale,
    )

    cos = cosine_sim(exact_grad, da_grad)
    rel_err = float(np.linalg.norm(exact_grad - da_grad) / (np.linalg.norm(exact_grad) + 1e-9))
    mag_ratio = float(np.linalg.norm(da_grad) / (np.linalg.norm(exact_grad) + 1e-9))

    # Dispatch quality
    lmp_maes, cost_gaps = [], []
    for case, gi, sample in zip(case_list, gi_list, samples):
        pred_cost = predict_costs(model, sample)
        try:
            layer = DispatchLayer(case.net, case.devices,
                parameter_names={'gen_cost': (gi, 'linear_cost')},
                time_horizon=1, solver=cp.CLARABEL, add_ground=False)
            out = layer(gen_cost=pred_cost.reshape(-1, 1))
            pred_pg = np.asarray(out.power[gi][0]).ravel()
            pred_lmp = np.asarray(out.prices).ravel()
            lmp_maes.append(float(np.mean(np.abs(pred_lmp - sample.target_lmp))))
            cost_gaps.append(abs(float(np.dot(sample.gen_cost, pred_pg)) - sample.target_obj)
                             / max(abs(sample.target_obj), 1e-6))
        except Exception:
            pass

    return {
        "state": state,
        "model": model_tag,
        "n_gen": n_gen,
        "cosine_similarity": cos,
        "relative_error": rel_err,
        "magnitude_ratio": mag_ratio,
        "mean_lmp_mae": float(np.mean(lmp_maes)) if lmp_maes else None,
        "mean_cost_gap": float(np.mean(cost_gaps)) if cost_gaps else None,
    }


def main():
    CHECKPOINTS = {
        "v9_best_lmp":      "ralph/build/state/checkpoints/design_a_v9.pt",
        "v11_best_costgap": "ralph/build/state/checkpoints/design_a_v11.pt",
    }

    # States and periods to evaluate
    # Training state: delaware (2 periods) — matches P6/improve_planning_metrics
    # Test states: single period each (04h) to limit solve time
    EVAL_CONFIGS = [
        ("delaware",   ["04h", "16h"]),
        ("oregon",     ["04h"]),
        ("kansas",     ["04h"]),
    ]

    print("=== Planning gradient evaluation: v9 vs v11 ===")
    print("States:", [s for s, _ in EVAL_CONFIGS])

    exact_grad_cache = {}
    all_results = {}

    for tag, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- Model: {tag} ({ckpt_path}) ---")
        model = DesignAv9(hidden_dim=64, n_mp_layers=3)
        try:
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()
        except Exception as e:
            print(f"  Failed to load {tag}: {e}")
            continue

        model_results = []
        for state, hours in EVAL_CONFIGS:
            print(f"  Evaluating {state} ({hours})...")
            r = evaluate_state(state, hours, tag, model, exact_grad_cache)
            if r is not None:
                model_results.append(r)
                print(f"    cos_sim={r['cosine_similarity']:.4f}  rel_err={r['relative_error']:.4f}  "
                      f"cost_gap={r['mean_cost_gap']:.4f}  lmp_mae={r['mean_lmp_mae']:.4f}")
        all_results[tag] = model_results

    # Summary table
    print("\n=== Summary ===")
    print(f"{'State':12s} | {'Model':20s} | cos_sim | rel_err | cost_gap | lmp_mae")
    print("-" * 80)
    for tag, results in all_results.items():
        for r in results:
            print(f"{r['state']:12s} | {tag:20s} | {r['cosine_similarity']:.4f}  | "
                  f"{r['relative_error']:.4f}  | {r['mean_cost_gap']:.4f}   | {r['mean_lmp_mae']:.4f}")

    print("\nReference (from improve_planning_metrics.json):")
    print(f"  P3_baseline:  delaware cos_sim=0.834, rel_err=11.2×")
    print(f"  v2_cost_fix:  delaware cos_sim=0.977, rel_err=2.5×")
    print(f"  Target:       cos_sim > 0.95")

    metrics = {
        "reference": {
            "P3_baseline_delaware_cos": 0.834,
            "v2_delaware_cos": 0.977,
            "target_cos": 0.95,
        },
        "results": all_results,
        "exact_grads": {k: v.tolist() for k, v in exact_grad_cache.items()},
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
