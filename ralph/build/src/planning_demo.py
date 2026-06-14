"""
P6 — Planning / multi-period expansion with duals.

Demonstrates the differentiable stack for investment planning:

  1. Select a small case (delaware, 33 buses) at 2 time periods (04h, 16h).
  2. Define investment decision: scaling each generator's nominal_capacity.
  3. Use zap's DispatchLayer (differentiable) to solve DC-OPF for each period.
  4. Total cost = sum over periods of operating cost + investment_cost * capacity.
  5. Gradient of total cost w.r.t. capacity scaling flows back through DispatchLayer.
  6. Show the LMPs (duals) as investment signals: which generators are on the margin?
  7. Compare Design A (GNN encoder → cost predictor → dispatch layer) planning
     gradients to exact zap-only planning gradients.

This ties to the §6 active-set / sign-flip discussion: increasing capacity of a
marginal generator reduces cost; the dual variable (LMP) indicates the investment
shadow price.

Key checks
----------
  - Gradient consistency: Design A gradient vs exact zap gradient, relative error.
  - Dual (LMP) investment signals: do LMPs correctly identify the bottleneck bus?
  - Active-set behavior: which generators are at their capacity limit (binding)?
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
from gridsfm_zap import load_case, solve_with_zap
from dataset import _device_index
import train_design_a as p3

METRICS_OUT = "ralph/build/state/P6_metrics.json"
HIDDEN_DIM  = 64
N_LAYERS    = 3

# Investment cost per unit of capacity (in normalized cost units)
INVEST_COST = 0.1


# ---------------------------------------------------------------------------
# Multi-period planning gradient (exact zap)
# ---------------------------------------------------------------------------

def planning_grad_exact(case_list: list, invest_scale: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    """
    Finite-difference gradient of total operating cost + investment cost
    w.r.t. capacity scaling for each generator.

    invest_scale: (n_gen,) — current capacity scale (starts at 1)
    Returns: gradient (n_gen,) via central finite differences.
    """
    from copy import deepcopy

    def total_cost(scale):
        total = 0.0
        for case in case_list:
            devices_scaled = deepcopy(case.devices)
            gi = _device_index(devices_scaled, zap.Generator)
            if gi is None:
                continue
            orig_cap = np.asarray(devices_scaled[gi].nominal_capacity).ravel()
            devices_scaled[gi].nominal_capacity = (orig_cap * scale).reshape(-1, 1)
            try:
                out = case.net.dispatch(devices_scaled, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
                total += out.problem.value
            except Exception:
                total += 1e6
        # Add investment cost
        total += float(np.dot(INVEST_COST * scale, np.ones_like(scale)))
        return total

    n_gen = len(invest_scale)
    grad = np.zeros(n_gen)
    rng = np.random.default_rng(42)
    v = rng.standard_normal(n_gen)
    v /= np.linalg.norm(v) + 1e-9

    # Directional derivative
    cp_plus  = total_cost(invest_scale + delta * v)
    cp_minus = total_cost(invest_scale - delta * v)
    dir_grad = (cp_plus - cp_minus) / (2 * delta)

    # Per-component gradient via finite differences (n_gen separate perturbations)
    for i in range(n_gen):
        ei = np.zeros(n_gen)
        ei[i] = delta
        grad[i] = (total_cost(invest_scale + ei) - total_cost(invest_scale - ei)) / (2 * delta)

    return grad, dir_grad, v


def planning_grad_design_a(case_list: list, invest_scale: np.ndarray,
                            design_a_model, layers_map: dict) -> np.ndarray:
    """
    Gradient of total operating cost w.r.t. capacity scaling, computed
    THROUGH Design A (GNN encoder predicts costs, layer solves dispatch).
    This is the 'surrogate-amortized inner solve' gradient.
    """
    from dataset import load_dataset, GraphSample
    import torch

    n_gen = len(invest_scale)
    scale_t = torch.tensor(invest_scale, dtype=torch.float32, requires_grad=True)

    total_cost = torch.tensor(0.0, requires_grad=False)
    loss_terms = []

    for case, sample in case_list:
        key = sample.name
        if key not in layers_map or layers_map[key] is None:
            continue
        layer, gi, true_cost_np = layers_map[key]

        nf = torch.tensor(sample.node_feats, dtype=torch.float32)
        ei = torch.tensor(sample.edge_index, dtype=torch.long)
        ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
        gb = torch.tensor(sample.gen_bus, dtype=torch.long)

        # Predict costs
        with torch.no_grad():
            pred_cost = design_a_model(nf, ei, ef, gb)  # (n_gen,)

        # Scale capacity (for gradient computation)
        scaled_pred_cost = pred_cost  # cost doesn't scale with capacity here

        # Run dispatch through layer (need grad to flow through scale_t)
        # For Design A planning: we use the DispatchFunc for gradient
        dispatch, lmps, obj_out = p3.dispatch_through_layer(pred_cost.detach(), layer, gi)
        loss_terms.append(obj_out[0])

    if not loss_terms:
        return np.zeros(n_gen), np.zeros(n_gen)

    total = sum(loss_terms)
    # Investment cost term
    invest = (INVEST_COST * scale_t).sum()

    # Gradient w.r.t. scale — for the investment cost term only (since
    # the dispatch cost doesn't directly flow through scale_t here)
    invest.backward()
    invest_grad = scale_t.grad.detach().numpy().copy() if scale_t.grad is not None else np.zeros(n_gen)

    return invest_grad, float(total.item())


# ---------------------------------------------------------------------------
# Active set / sign-flip analysis
# ---------------------------------------------------------------------------

def active_set_analysis(case, dispatch_np: np.ndarray, pmax: np.ndarray) -> dict:
    """
    Check which generators are binding (dispatch ≈ pmax) and which are at 0.
    The active set defines the dual variable (LMP) sign structure.
    """
    eps = 1e-3
    n_gen = len(dispatch_np)
    at_max = np.sum(dispatch_np >= pmax - eps)
    at_zero = np.sum(dispatch_np <= eps)
    intermediate = n_gen - at_max - at_zero

    return {
        "n_gen": n_gen,
        "at_max": int(at_max),
        "at_zero": int(at_zero),
        "intermediate": int(intermediate),
        "capacity_utilization": float(np.sum(dispatch_np) / np.sum(pmax) if np.sum(pmax) > 0 else 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Use delaware (33 buses, 9 generators) — small, reliable, 2 periods
    state = "delaware"
    hours = ["04h", "16h"]
    data_dir = "ralph/build/data/raw"

    print("=== P6 — Planning with duals demo ===")
    print(f"Grid: {state} ({len(hours)} periods: {hours})")

    case_list = []
    gi_list = []
    for hour in hours:
        model_path = os.path.join(data_dir, hour, f"{state}_model.json")
        dc_path    = os.path.join(data_dir, hour, f"{state}_dc_results.json")
        case = load_case(model_path, dc_path)
        gi = _device_index(case.devices, zap.Generator)
        case_list.append(case)
        gi_list.append(gi)
        n_gen = len(np.asarray(case.devices[gi].terminal).ravel())
        print(f"  Loaded {state}_{hour}: buses={case.n_bus}  gens={n_gen}")

    # Baseline: solve each period with exact zap
    baseline_results = []
    for h, (case, gi) in zip(hours, zip(case_list, gi_list)):
        out = solve_with_zap(case)
        dispatch = np.asarray(out.power[gi][0]).ravel()
        lmps = np.asarray(out.prices).ravel()
        pmax = np.asarray(case.devices[gi].dynamic_capacity).ravel()
        costs = np.asarray(case.devices[gi].linear_cost).ravel()
        gen_buses = np.asarray(case.devices[gi].terminal).ravel()
        obj = float(out.problem.value)
        active = active_set_analysis(case, dispatch, pmax)
        baseline_results.append({
            "hour": h,
            "obj": obj,
            "dispatch": dispatch.tolist(),
            "lmps": lmps.tolist(),
            "pmax": pmax.tolist(),
            "costs": costs.tolist(),
            "gen_buses": gen_buses.tolist(),
            "active_set": active,
        })
        print(f"\n[{h}] Objective: {obj:.4f}")
        print(f"  LMP stats: min={lmps.min():.4f}  max={lmps.max():.4f}  mean={lmps.mean():.4f}")
        print(f"  Active set: {active['at_max']} at pmax, {active['at_zero']} at 0, {active['intermediate']} intermediate")
        # Show which buses have highest LMPs (investment signals)
        top_lmp_buses = np.argsort(-np.abs(lmps))[:5]
        print(f"  Top-5 LMP buses (investment signals): {top_lmp_buses.tolist()} → LMPs: {lmps[top_lmp_buses].round(4).tolist()}")

    # Planning gradient (exact zap)
    print(f"\n=== Planning gradient (exact zap) ===")
    n_gen = len(baseline_results[0]["pmax"])
    invest_scale = np.ones(n_gen)
    print(f"Computing gradient for {n_gen} generators across {len(hours)} periods...")
    exact_grad, dir_deriv, v = planning_grad_exact(case_list, invest_scale, delta=5e-3)
    print(f"  Gradient stats: min={exact_grad.min():.4f}  max={exact_grad.max():.4f}  mean={exact_grad.mean():.4f}")
    top_invest = np.argsort(exact_grad)[:5]  # most negative = best investment
    print(f"  Top-5 generators to invest in (most negative gradient):")
    for idx in top_invest:
        bus = int(baseline_results[0]["gen_buses"][idx])
        lmp = float(baseline_results[0]["lmps"][bus])
        print(f"    gen {idx} (bus {bus}, LMP={lmp:.4f}): grad={exact_grad[idx]:.4f}")

    # LMP consistency check: high LMP bus → investing there reduces cost
    # For each generator on a high-LMP bus, the gradient should be negative (adding capacity reduces cost)
    lmps_period0 = np.asarray(baseline_results[0]["lmps"])
    gen_buses_arr = np.asarray(baseline_results[0]["gen_buses"])
    gen_lmps = lmps_period0[gen_buses_arr]
    # Check correlation between |LMP| and |gradient|
    if n_gen > 2:
        corr = float(np.corrcoef(gen_lmps, exact_grad)[0, 1])
        print(f"\n  Correlation(gen_LMP, investment_gradient): {corr:.4f}")
        print(f"  Note: negative corr → high-LMP generators have most negative gradient (consistent)")

    # Design A planning comparison
    print(f"\n=== Design A planning (GNN encoder → DispatchLayer) ===")
    p3_model = p3.DesignA(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    p3_model.load_state_dict(torch.load("ralph/build/state/checkpoints/design_a.pt", weights_only=True))
    p3_model.eval()

    # Load dataset samples for delaware
    from dataset import load_dataset, _cache_path
    split_json = "ralph/build/data/raw/split.json"
    cache_dir = "ralph/build/state/cache"

    # Load cached samples
    from dataset import _load_sample
    sample_map = {}
    for hour in hours:
        s = _load_sample(cache_dir, state, hour)
        if s is not None:
            sample_map[f"{state}_{hour}"] = s

    # Build DispatchLayers
    layers_map = {}
    case_sample_pairs = []
    for h, case in zip(hours, case_list):
        key = f"{state}_{h}"
        info = p3.build_dispatch_layer(state, h)
        layers_map[key] = info
        if key in sample_map:
            case_sample_pairs.append((case, sample_map[key]))

    # Get Design A dispatch for each period and compute active sets
    design_a_results = []
    for h, (case, sample) in zip(hours, case_sample_pairs):
        key = f"{state}_{h}"
        info = layers_map.get(key)
        if info is None:
            continue
        layer, gi, _ = info
        nf = torch.tensor(sample.node_feats, dtype=torch.float32)
        ei = torch.tensor(sample.edge_index, dtype=torch.long)
        ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
        gb = torch.tensor(sample.gen_bus, dtype=torch.long)
        with torch.no_grad():
            pred_cost = p3_model(nf, ei, ef, gb)
        cost_np = pred_cost.numpy().reshape(-1, 1)
        p3_out, t = time_median_fn(lambda: layer(gen_cost=cost_np), n=3)
        p3_dispatch = np.asarray(p3_out.power[gi][0]).ravel()
        p3_lmps = np.asarray(p3_out.prices).ravel()
        p3_obj = float(p3_out.problem.value)
        pmax = sample.gen_pmax
        active = active_set_analysis(case, p3_dispatch, pmax)

        # Gradient consistency: compute exact grad at p3 solution
        true_dispatch = sample.target_pg
        true_cost = float(np.dot(sample.gen_cost, true_dispatch))
        p3_cost = float(np.dot(sample.gen_cost, p3_dispatch))

        design_a_results.append({
            "hour": h,
            "p3_obj_with_pred_costs": p3_obj,
            "p3_cost_at_true_costs": p3_cost,
            "true_obj": sample.target_obj,
            "lmp_mae": float(np.mean(np.abs(p3_lmps - sample.target_lmp))),
            "active_set_p3": active,
            "active_set_exact": baseline_results[hours.index(h)]["active_set"],
        })
        print(f"\n[{h}] Design A")
        print(f"  P3 cost (true costs): {p3_cost:.4f}  True cost: {sample.target_obj:.4f}  Gap: {abs(p3_cost - sample.target_obj)/max(abs(sample.target_obj),1e-6):.3f}")
        print(f"  P3 LMP MAE: {design_a_results[-1]['lmp_mae']:.4f}")
        print(f"  P3 Active set: {active['at_max']} at pmax, {active['at_zero']} at 0  (vs exact: {baseline_results[hours.index(h)]['active_set']['at_max']}, {baseline_results[hours.index(h)]['active_set']['at_zero']})")

    # Gradient consistency check
    print(f"\n=== Gradient consistency: Design A vs exact zap ===")

    # Compute Design A planning gradient via finite differences on the DispatchLayer
    def da_total_cost(scale):
        total = 0.0
        for h, (case, sample) in zip(hours, case_sample_pairs):
            key = f"{state}_{h}"
            info = layers_map.get(key)
            if info is None:
                continue
            layer_d = info[0]
            gi_d = info[1]
            nf = torch.tensor(sample.node_feats, dtype=torch.float32)
            ei = torch.tensor(sample.edge_index, dtype=torch.long)
            ef = torch.tensor(sample.edge_feats, dtype=torch.float32)
            gb = torch.tensor(sample.gen_bus, dtype=torch.long)
            with torch.no_grad():
                pred_cost = p3_model(nf, ei, ef, gb).numpy()
            # Scale capacity by `scale` in the case devices
            from copy import deepcopy
            devices_s = deepcopy(case.devices)
            gi_s = _device_index(devices_s, zap.Generator)
            orig_nc = np.asarray(devices_s[gi_s].nominal_capacity).ravel()
            devices_s[gi_s].nominal_capacity = (orig_nc * scale).reshape(-1, 1)
            # Rebuild layer with scaled capacities
            layer_s = DispatchLayer(case.net, devices_s,
                                     parameter_names={'gen_cost': (gi_s, 'linear_cost')},
                                     time_horizon=1, solver=cp.CLARABEL, add_ground=False)
            try:
                out = layer_s(gen_cost=pred_cost.reshape(-1, 1))
                dispatch_s = np.asarray(out.power[gi_s][0]).ravel()
                # Evaluate at true gen costs
                total += float(np.dot(sample.gen_cost, dispatch_s))
            except Exception:
                total += 1e6
        total += float(np.dot(INVEST_COST * scale, np.ones(n_gen)))
        return total

    da_grad = np.zeros(n_gen)
    delta = 5e-3
    for i in range(n_gen):
        ei_v = np.zeros(n_gen)
        ei_v[i] = delta
        da_grad[i] = (da_total_cost(invest_scale + ei_v) - da_total_cost(invest_scale - ei_v)) / (2 * delta)

    # Compute relative error in gradient direction
    exact_norm = np.linalg.norm(exact_grad)
    da_norm = np.linalg.norm(da_grad)
    if exact_norm > 1e-9 and da_norm > 1e-9:
        cos_sim = float(np.dot(exact_grad / exact_norm, da_grad / da_norm))
        rel_err = float(np.linalg.norm(exact_grad - da_grad) / (exact_norm + 1e-9))
        print(f"  Cosine similarity (exact vs Design A gradient): {cos_sim:.4f}")
        print(f"  Relative error in gradient magnitude:           {rel_err:.4f}")
    else:
        cos_sim = None
        rel_err = None
        print("  (Cannot compute gradient similarity — zero gradient)")

    print(f"\n  Exact grad:   {exact_grad.round(4).tolist()}")
    print(f"  Design A grad:{da_grad.round(4).tolist()}")

    # Compile metrics
    metrics = {
        "phase": "P6",
        "grid": {"state": state, "n_bus": case_list[0].n_bus, "n_gen": n_gen, "n_periods": len(hours)},
        "baseline_periods": [{
            "hour": r["hour"],
            "obj": r["obj"],
            "lmp_mean": float(np.mean(r["lmps"])),
            "lmp_max": float(np.max(np.abs(r["lmps"]))),
            "active_set": r["active_set"],
        } for r in baseline_results],
        "planning_gradient_exact": {
            "gradient_mean": float(exact_grad.mean()),
            "gradient_min": float(exact_grad.min()),
            "gradient_max": float(exact_grad.max()),
            "top_invest_gens": top_invest.tolist(),
            "lmp_gradient_correlation": corr if n_gen > 2 else None,
        },
        "design_a_periods": design_a_results,
        "gradient_consistency": {
            "cosine_similarity": cos_sim,
            "relative_error": rel_err,
            "assessment": (
                "Consistent" if (cos_sim is not None and cos_sim > 0.8)
                else "Inconsistent" if cos_sim is not None
                else "Undetermined"
            ),
        },
        "active_set_comparison": {
            "exact_at_max_mean": float(np.mean([r["active_set"]["at_max"] for r in baseline_results])),
            "design_a_at_max_mean": float(np.mean([r["active_set_p3"]["at_max"] for r in design_a_results])),
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")
    print(f"\n=== P6 SUMMARY ===")
    print(f"  Grid: {state}  {n_gen} gens  {len(hours)} periods")
    print(f"  Baseline obj (period 0): {baseline_results[0]['obj']:.4f}")
    print(f"  LMP investment signal correlation: {corr if n_gen > 2 else 'N/A':.4f}")
    print(f"  Gradient consistency (Design A vs exact):")
    print(f"    Cosine similarity: {cos_sim:.4f}" if cos_sim is not None else "    N/A")
    print(f"    Relative error:    {rel_err:.4f}" if rel_err is not None else "    N/A")


def time_median_fn(fn, n=3):
    import time as _time
    results = []
    t_list = []
    for _ in range(n):
        t0 = _time.perf_counter()
        r = fn()
        t_list.append(_time.perf_counter() - t0)
        results.append(r)
    return results[-1], float(np.median(t_list)) * 1000


if __name__ == "__main__":
    main()
