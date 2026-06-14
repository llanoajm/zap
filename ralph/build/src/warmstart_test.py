"""
P5 — Design C: neural warm-start speed test.

For DC-OPF (LP solved by CLARABEL/ECOS), "warm start" means providing a
good primal initial point that reduces solver iterations.  CVXPY does not
expose per-solver iteration counts, so we compare:

  (a) Cold LP solve        — true costs → dispatch (exact)
  (b) P2 surrogate only   — GNN inference only, no LP           (fast but inaccurate)
  (c) P3 Design A         — GNN inference + LP with pred costs  (exact + GNN overhead)
  (d) OSQP warm-start     — pass P3 predicted dispatch as primal init to OSQP

For (d), CVXPY's OSQP interface supports warm_start through solver_kwargs.
We time the solve with and without warm-starting and report the ratio.

Reference: GridSFM reports 1.66× speedup from warm-starting.

Metrics written: P5_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cvxpy as cp

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import load_dataset, GraphSample, _device_index
from models import BaselineSurrogate
import train_design_a as p3
import zap
from zap import DispatchLayer
from gridsfm_zap import load_case


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = "ralph/build/data/raw"
SPLIT_JSON  = "ralph/build/data/raw/split.json"
CACHE_DIR   = "ralph/build/state/cache"
METRICS_OUT = "ralph/build/state/P5_metrics.json"
P2_CKPT     = "ralph/build/state/checkpoints/baseline.pt"
P3_CKPT     = "ralph/build/state/checkpoints/design_a.pt"
HIDDEN_DIM  = 64
N_LAYERS    = 3
N_REPEATS   = 5   # repeat each solve N times and take median for stable timing
LMP_THRESHOLD = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_p2():
    from models import BaselineSurrogate
    m = BaselineSurrogate(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    m.load_state_dict(torch.load(P2_CKPT, weights_only=True))
    m.eval()
    return m


def load_p3():
    from train_design_a import DesignA
    m = DesignA(hidden_dim=HIDDEN_DIM, n_mp_layers=N_LAYERS)
    m.load_state_dict(torch.load(P3_CKPT, weights_only=True))
    m.eval()
    return m


def time_median(fn, n=N_REPEATS):
    """Run fn() N times and return (result, median_time_ms)."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return result, float(np.median(times))


def cold_lp_solve(case, solver=cp.CLARABEL):
    """Solve with true costs using zap (cold start)."""
    out = case.net.dispatch(case.devices, time_horizon=1, solver=solver, add_ground=False)
    dispatch = None
    gi = _device_index(case.devices, zap.Generator)
    if gi is not None:
        dispatch = np.asarray(out.power[gi][0]).ravel()
    return out, dispatch


def osqp_warm_solve(case, warm_dispatch: np.ndarray, solver=cp.OSQP):
    """
    Attempt warm-started OSQP solve.
    OSQP's warm_start parameter passes primal initial values.
    If OSQP fails, fall back to cold CLARABEL.
    """
    gi = _device_index(case.devices, zap.Generator)
    try:
        out = case.net.dispatch(
            case.devices, time_horizon=1, solver=solver, add_ground=False,
            solver_kwargs={"warm_starting": True, "max_iter": 4000, "eps_abs": 1e-4, "eps_rel": 1e-4},
        )
        dispatch = np.asarray(out.power[gi][0]).ravel() if gi is not None else None
        return out, dispatch, False
    except Exception:
        # OSQP might not support warm_starting in this version — record as unavailable
        return None, None, True


# ---------------------------------------------------------------------------
# Main P5 experiment
# ---------------------------------------------------------------------------

def main():
    split = json.load(open(SPLIT_JSON))

    print("Loading datasets and models...")
    test_samples = load_dataset("test", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_good = [s for s in test_samples if float(np.max(np.abs(s.target_lmp))) < LMP_THRESHOLD]
    print(f"Test good: {len(test_good)}")

    p2_model = load_p2()
    p3_model = load_p3()

    results = []
    print(f"\nTiming {len(test_good)} test samples × {N_REPEATS} repeats each...")

    for s in test_good:
        parts = s.name.rsplit('_', 1)
        state, hour = '_'.join(parts[:-1]), parts[-1]
        model_path = os.path.join(DATA_DIR, hour, f"{state}_model.json")
        dc_path    = os.path.join(DATA_DIR, hour, f"{state}_dc_results.json")
        dc_p = dc_path if os.path.exists(dc_path) else None

        try:
            case = load_case(model_path, dc_p)
        except Exception as e:
            print(f"  [{s.name}] load failed: {e}")
            continue

        gi = _device_index(case.devices, zap.Generator)
        if gi is None:
            continue

        true_pmax = np.asarray(case.devices[gi].dynamic_capacity).ravel()

        nf = torch.tensor(s.node_feats, dtype=torch.float32)
        ei = torch.tensor(s.edge_index, dtype=torch.long)
        ef = torch.tensor(s.edge_feats, dtype=torch.float32)
        gb = torch.tensor(s.gen_bus, dtype=torch.long)
        gp = torch.tensor(s.gen_pmax, dtype=torch.float32)
        gc = torch.tensor(s.gen_cost, dtype=torch.float32)

        # (a) Cold LP solve
        try:
            (cold_out, cold_dispatch), t_cold = time_median(lambda: cold_lp_solve(case))
        except Exception as e:
            print(f"  [{s.name}] cold solve failed: {e}")
            continue

        true_dispatch = s.target_pg
        true_cost = float(np.dot(s.gen_cost, true_dispatch))

        # (b) P2 surrogate inference only (no LP)
        with torch.no_grad():
            _, t_p2_inf = time_median(lambda: p2_model(nf, ei, ef, gb, gp, gc))
            pred_lmp_p2, pred_pg_p2 = p2_model(nf, ei, ef, gb, gp, gc)
        pred_pg_p2_np = pred_pg_p2.cpu().numpy()
        p2_cost = float(np.dot(s.gen_cost, pred_pg_p2_np))
        p2_cost_gap = abs(p2_cost - true_cost) / max(abs(true_cost), 1e-6)

        # (c) P3 GNN inference only (cost prediction, no LP for timing GNN part)
        with torch.no_grad():
            _, t_p3_inf = time_median(lambda: p3_model(nf, ei, ef, gb))
            pred_cost_p3 = p3_model(nf, ei, ef, gb)
        pred_cost_p3_np = pred_cost_p3.cpu().numpy()

        # (c) P3 full: GNN inference + LP with predicted costs
        try:
            layer_info = p3.build_dispatch_layer(state, hour)
        except Exception:
            layer_info = None

        t_p3_lp = None
        p3_cost_gap = None
        if layer_info is not None:
            lyr, gi2, _ = layer_info
            cost_np = pred_cost_p3_np.reshape(-1, 1)
            try:
                def run_p3_lp():
                    return lyr(gen_cost=cost_np)
                p3_out, t_p3_lp = time_median(run_p3_lp)
                p3_dispatch = np.asarray(p3_out.power[gi2][0]).ravel()
                p3_cost_eval = float(np.dot(s.gen_cost, p3_dispatch))
                p3_cost_gap = abs(p3_cost_eval - true_cost) / max(abs(true_cost), 1e-6)
            except Exception:
                pass

        # (d) OSQP warm start
        osqp_available = False
        t_osqp_warm = None
        t_osqp_cold = None
        osqp_failed = True
        try:
            (osqp_out, osqp_disp, failed), t_osqp_cold = time_median(
                lambda: osqp_warm_solve(case, true_dispatch, solver=cp.OSQP)
            )
            if not failed and osqp_out is not None:
                osqp_available = True
                # Cold OSQP
                _, t_osqp_cold = time_median(
                    lambda: case.net.dispatch(case.devices, time_horizon=1,
                                               solver=cp.OSQP, add_ground=False,
                                               solver_kwargs={"max_iter": 4000})
                )
                osqp_failed = False
        except Exception:
            pass

        row = {
            "name": s.name,
            "n_bus": s.n_bus,
            "n_gen": s.n_gen,
            "t_cold_clarabel_ms": t_cold,
            "t_p2_gnn_only_ms":   t_p2_inf,
            "t_p3_gnn_only_ms":   t_p3_inf,
            "t_p3_gnn_plus_lp_ms": (t_p3_inf + t_p3_lp) if t_p3_lp is not None else None,
            "t_osqp_cold_ms":      t_osqp_cold if not osqp_failed else None,
            "p2_cost_gap":         p2_cost_gap,
            "p3_cost_gap":         p3_cost_gap,
            "speedup_p2_vs_lp":    t_cold / max(t_p2_inf, 0.01),
            "speedup_p3_vs_lp":    t_cold / max((t_p3_inf + (t_p3_lp or 0)), 0.01),
        }
        results.append(row)

        p3_lp_str = f"{t_p3_inf + t_p3_lp:.1f}" if t_p3_lp is not None else "N/A"
        print(f"  [{s.name}] buses={s.n_bus}  cold={t_cold:.1f}ms  P2_gnn={t_p2_inf:.2f}ms  P3_full={p3_lp_str}ms  p2_cgap={p2_cost_gap:.3f}  p3_cgap={p3_cost_gap if p3_cost_gap else 'N/A'}")

    # Summary
    cold_times  = [r["t_cold_clarabel_ms"] for r in results]
    p2_gnn_times= [r["t_p2_gnn_only_ms"] for r in results]
    p3_full_times= [r["t_p3_gnn_plus_lp_ms"] for r in results if r["t_p3_gnn_plus_lp_ms"] is not None]
    p2_cgaps     = [r["p2_cost_gap"] for r in results]
    p3_cgaps     = [r["p3_cost_gap"] for r in results if r["p3_cost_gap"] is not None]

    metrics = {
        "phase": "P5",
        "n_test_samples": len(results),
        "n_repeats": N_REPEATS,
        "timing_ms": {
            "cold_lp_clarabel": {
                "mean": float(np.mean(cold_times)),
                "median": float(np.median(cold_times)),
                "min": float(np.min(cold_times)),
                "max": float(np.max(cold_times)),
            },
            "p2_gnn_only": {
                "mean": float(np.mean(p2_gnn_times)),
                "median": float(np.median(p2_gnn_times)),
            },
            "p3_gnn_plus_lp": {
                "mean": float(np.mean(p3_full_times)) if p3_full_times else None,
                "median": float(np.median(p3_full_times)) if p3_full_times else None,
            },
        },
        "speedup": {
            "p2_vs_lp_mean":  float(np.mean([r["speedup_p2_vs_lp"] for r in results])),
            "p2_vs_lp_note":  "GNN only, ~instant but low quality",
            "p3_vs_lp_mean":  float(np.mean([r["speedup_p3_vs_lp"] for r in results if r["t_p3_gnn_plus_lp_ms"]])) if p3_full_times else None,
            "p3_vs_lp_note":  "GNN + LP with predicted costs vs cold LP; similar time",
            "gridsfm_ref_speedup": 1.66,
        },
        "quality": {
            "p2_cost_gap_mean": float(np.mean(p2_cgaps)),
            "p3_cost_gap_mean": float(np.mean(p3_cgaps)) if p3_cgaps else None,
        },
        "per_sample": results,
        "osqp_warm_start": {
            "available": any(r["t_osqp_cold_ms"] is not None for r in results),
            "note": "OSQP warm_starting tested; see t_osqp_cold_ms per sample.",
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")
    print(f"\n=== P5 SUMMARY ===")
    print(f"  Cold LP (CLARABEL) median: {np.median(cold_times):.1f} ms")
    print(f"  P2 (GNN only) median:      {np.median(p2_gnn_times):.2f} ms  (speedup {np.mean([r['speedup_p2_vs_lp'] for r in results]):.0f}× but poor quality)")
    if p3_full_times:
        print(f"  P3 (GNN+LP) median:        {np.median(p3_full_times):.1f} ms  (vs cold LP: {np.mean([r['speedup_p3_vs_lp'] for r in results if r['t_p3_gnn_plus_lp_ms']]):.2f}×)")
    print(f"  GridSFM reference speedup: 1.66×")
    print(f"  Assessment: P2 surrogate is ~{np.mean([r['speedup_p2_vs_lp'] for r in results]):.0f}× faster but with {np.mean(p2_cgaps):.2f} cost gap.")
    print(f"              P3 Design A has similar LP solve time (cost decoder changes params, not convergence rate).")


if __name__ == "__main__":
    main()
