"""
Compare planning gradient quality: P3 baseline vs v2 (cost-mapping fix + aux loss).

Uses delaware (33 buses, 9 generators, 2 periods) as in planning_demo.py.
Writes to state/improve_planning_metrics.json.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import cvxpy as cp

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zap
from zap import DispatchLayer
from gridsfm_zap import load_case, solve_with_zap
from dataset import _device_index, _load_sample
import train_design_a as p3

METRICS_OUT = "ralph/build/state/improve_planning_metrics.json"
INVEST_COST = 0.1
DATA_DIR = "ralph/build/data/raw"
CACHE_DIR = "ralph/build/state/cache"


def total_cost_fn(case_list, gi_list, scale, gen_costs_list=None):
    """True OPF total cost across periods with scaled capacity."""
    total = 0.0
    for i, (case, gi) in enumerate(zip(case_list, gi_list)):
        devs = deepcopy(case.devices)
        orig_nc = np.asarray(devs[gi].nominal_capacity).ravel()
        devs[gi].nominal_capacity = (orig_nc * scale).reshape(-1, 1)
        # Optionally override gen costs (for Design A evaluation)
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


def compute_grad_exact(case_list, gi_list, scale):
    fn = lambda s: total_cost_fn(case_list, gi_list, s)
    return finite_diff_grad(fn, scale)


def compute_grad_design_a(case_list, gi_list, scale, model, samples, ckpt_name=""):
    """Gradient through Design A dispatch (using predicted gen costs)."""
    pred_costs_list = []
    for case, sample in zip(case_list, samples):
        nf = torch.tensor(sample.node_feats, dtype=torch.float32)
        ei = torch.tensor(sample.edge_index, dtype=torch.long)
        ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
        gb = torch.tensor(sample.gen_bus, dtype=torch.long)
        with torch.no_grad():
            pred_cost = model(nf, ei, ef, gb).numpy()
        pred_costs_list.append(pred_cost)

    fn = lambda s: total_cost_fn(case_list, gi_list, s, gen_costs_list=pred_costs_list)
    return finite_diff_grad(fn, scale)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a / na, b / nb))


def main():
    state = "delaware"
    hours = ["04h", "16h"]

    # Load cases
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

    n_gen = len(np.asarray(case_list[0].devices[gi_list[0]].terminal).ravel())
    scale = np.ones(n_gen)
    print(f"Grid: {state}  n_gen={n_gen}  n_periods={len(hours)}")

    # Exact gradient
    print("\nComputing exact gradient...")
    exact_grad = compute_grad_exact(case_list, gi_list, scale)
    print(f"Exact grad: {exact_grad.round(4).tolist()}")

    results = {"exact_gradient": exact_grad.tolist()}

    # Compare P3 and v2 models
    for ckpt_name, ckpt_path in [
        ("P3_baseline", "ralph/build/state/checkpoints/design_a.pt"),
        ("v2_cost_fix", "ralph/build/state/checkpoints/design_a_v2.pt"),
    ]:
        model = p3.DesignA(hidden_dim=64, n_mp_layers=3)
        try:
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()
        except Exception as e:
            print(f"  Failed to load {ckpt_name}: {e}")
            continue

        print(f"\nComputing gradient for {ckpt_name}...")
        da_grad = compute_grad_design_a(case_list, gi_list, scale, model, samples, ckpt_name)
        cos = cosine_sim(exact_grad, da_grad)
        rel_err = float(np.linalg.norm(exact_grad - da_grad) / (np.linalg.norm(exact_grad) + 1e-9))
        mag_ratio = float(np.linalg.norm(da_grad) / (np.linalg.norm(exact_grad) + 1e-9))

        print(f"  {ckpt_name} grad: {da_grad.round(4).tolist()}")
        print(f"  Cosine similarity vs exact: {cos:.4f}")
        print(f"  Relative error:             {rel_err:.4f}")
        print(f"  Magnitude ratio:            {mag_ratio:.4f}")

        # Also measure dispatch quality for this model
        lmp_maes, cost_gaps = [], []
        for case, gi, sample in zip(case_list, gi_list, samples):
            if sample is None:
                continue
            nf = torch.tensor(sample.node_feats, dtype=torch.float32)
            ei = torch.tensor(sample.edge_index, dtype=torch.long)
            ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
            gb = torch.tensor(sample.gen_bus, dtype=torch.long)
            with torch.no_grad():
                pred_cost = model(nf, ei, ef, gb).numpy()
            try:
                layer = DispatchLayer(case.net, case.devices,
                    parameter_names={'gen_cost': (gi, 'linear_cost')},
                    time_horizon=1, solver=cp.CLARABEL, add_ground=False)
                out = layer(gen_cost=pred_cost.reshape(-1, 1))
                pred_pg = np.asarray(out.power[gi][0]).ravel()
                pred_lmp = np.asarray(out.prices).ravel()
                lmp_mae = float(np.mean(np.abs(pred_lmp - sample.target_lmp)))
                cost_gap = abs(float(np.dot(sample.gen_cost, pred_pg)) - sample.target_obj) / max(abs(sample.target_obj), 1e-6)
                lmp_maes.append(lmp_mae)
                cost_gaps.append(cost_gap)
            except Exception:
                pass

        results[ckpt_name] = {
            "gradient": da_grad.tolist(),
            "cosine_similarity": cos,
            "relative_error": rel_err,
            "magnitude_ratio": mag_ratio,
            "mean_lmp_mae": float(np.mean(lmp_maes)) if lmp_maes else None,
            "mean_cost_gap": float(np.mean(cost_gaps)) if cost_gaps else None,
        }

    # Print summary
    print("\n=== Planning gradient summary (delaware, 2 periods) ===")
    print(f"{'Model':20s} | cos_sim | rel_err | mag_ratio | lmp_mae | cost_gap")
    print("-" * 75)
    for name in ["P3_baseline", "v2_cost_fix"]:
        if name in results:
            r = results[name]
            print(f"{name:20s} | {r['cosine_similarity']:.4f}  | {r['relative_error']:.4f}  | {r['magnitude_ratio']:.4f}    | {r['mean_lmp_mae']:.4f}  | {r['mean_cost_gap']:.4f}")
    print()
    print("P6 baseline: cos_sim=0.76, rel_err=9.9×")

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(results, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
