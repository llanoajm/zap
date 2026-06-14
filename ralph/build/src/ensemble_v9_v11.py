"""
Ensemble evaluation: average predicted costs from v9 (best LMP) + v11 (best cost_gap).

Hypothesis: averaging cost predictions from two complementary models may reduce
individual model biases and improve combined metrics.

- v9  (best LMP,      W_AUX=0.5): cost_gap=0.082, LMP_mean=0.823
- v11 (best cost_gap, W_AUX=2.0): cost_gap=0.025, LMP_mean=0.926
- Ensemble expected between these on each metric.

Writes: state/ensemble_v9_v11_metrics.json
"""
from __future__ import annotations
import json, os, sys
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
METRICS_OUT = "ralph/build/state/ensemble_v9_v11_metrics.json"


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


def to_torch(s):
    return (
        torch.tensor(s.node_feats, dtype=torch.float32),
        torch.tensor(s.edge_index, dtype=torch.long),
        torch.tensor(s.edge_feats, dtype=torch.float32),
        torch.tensor(s.gen_bus,    dtype=torch.long),
        torch.tensor(s.gen_dc_frac, dtype=torch.float32),
    )


def compute_metrics(model_v9, model_v11, alpha, samples, layers):
    """alpha=1.0: pure v9; alpha=0.0: pure v11; alpha=0.5: equal ensemble."""
    disp_maes, disp_frac_maes, cost_gaps, lmp_maes, pb_errs = [], [], [], [], []
    per_sample = {}

    with torch.no_grad():
        for s in samples:
            if s.name not in layers or layers[s.name] is None:
                continue
            layer, gi, _ = layers[s.name]
            nf, ei, ef, gb, dc_frac = to_torch(s)

            cost_v9  = model_v9(nf, ei, ef, gb, dc_frac)
            cost_v11 = model_v11(nf, ei, ef, gb, dc_frac)
            pred_cost = alpha * cost_v9 + (1 - alpha) * cost_v11

            dispatch, lmps, _ = dispatch_through_layer(pred_cost, layer, gi)
            pred_pg = dispatch.numpy(); pred_lmp = lmps.numpy()
            if np.all(pred_pg == 0) and np.all(pred_lmp == 0):
                continue

            disp_mae = float(np.mean(np.abs(pred_pg - s.target_pg)))
            pmax_safe = np.where(s.gen_pmax > 1e-6, s.gen_pmax, 1.0)
            disp_frac_mae = float(np.mean(np.abs(pred_pg / pmax_safe - s.target_pg / pmax_safe)))
            cost_gap = abs(float(np.dot(s.gen_cost, pred_pg)) - s.target_obj) / max(abs(s.target_obj), 1e-6)
            lmp_mae = float(np.mean(np.abs(pred_lmp - s.target_lmp)))
            total_gen = float(np.sum(s.target_pg))
            pb_err = abs(float(np.sum(pred_pg)) - total_gen) / total_gen if total_gen > 1e-6 else 0.0

            disp_maes.append(disp_mae); disp_frac_maes.append(disp_frac_mae)
            cost_gaps.append(cost_gap); lmp_maes.append(lmp_mae); pb_errs.append(pb_err)
            per_sample[s.name] = {"cost_gap": cost_gap, "lmp_mae": lmp_mae}

    def _s(arr):
        if not arr: return {"mean": None, "median": None}
        return {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

    return {
        "n_evaluated": len(disp_maes),
        "dispatch_mae":      _s(disp_maes),
        "dispatch_frac_mae": _s(disp_frac_maes),
        "cost_gap":          _s(cost_gaps),
        "lmp_mae":           _s(lmp_maes),
        "power_balance_err": _s(pb_errs),
        "per_sample":        per_sample,
    }


def main():
    test_samples = load_dataset("test", DATA_DIR, SPLIT_JSON, CACHE_DIR)
    test_good    = [s for s in test_samples if is_good(s)]

    print("Building DispatchLayers...")
    layers = {}
    for s in test_good:
        parts = s.name.rsplit('_', 1)
        state, hour = '_'.join(parts[:-1]), parts[-1]
        layers[s.name] = build_dispatch_layer(state, hour)
    print(f"  {sum(v is not None for v in layers.values())}/{len(layers)} ready")

    model_v9  = DesignAv9(hidden_dim=64, n_mp_layers=3)
    model_v11 = DesignAv9(hidden_dim=64, n_mp_layers=3)
    model_v9.load_state_dict(torch.load("ralph/build/state/checkpoints/design_a_v9.pt",  weights_only=True))
    model_v11.load_state_dict(torch.load("ralph/build/state/checkpoints/design_a_v11.pt", weights_only=True))
    model_v9.eval(); model_v11.eval()

    orig_8 = [s for s in test_good if 'massachusetts' not in s.name]

    print("\n=== Ensemble sweep over alpha (v9 weight) ===")
    print(f"{'alpha':>6} | cost_gap_mean | cost_gap_med | lmp_mean | lmp_med | notes")
    print("-" * 70)

    best_by_lmp = None; best_by_cost = None
    all_results = []
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = compute_metrics(model_v9, model_v11, alpha, orig_8, layers)
        cg = m['cost_gap']
        lm = m['lmp_mae']
        ct16 = m['per_sample'].get('connecticut_16h', {})
        note = f"ct16_lmp={ct16.get('lmp_mae','?'):.2f}" if ct16 else ""
        print(f"{alpha:6.1f} | {cg['mean']:.4f}       | {cg['median']:.4f}      | "
              f"{lm['mean']:.4f}   | {lm['median']:.4f}  | {note}")
        all_results.append({"alpha": alpha, "cost_gap": cg, "lmp_mae": lm,
                             "per_sample": m["per_sample"]})
        if best_by_lmp is None or lm['mean'] < best_by_lmp['lmp_mae']['mean']:
            best_by_lmp = all_results[-1]
        if best_by_cost is None or cg['mean'] < best_by_cost['cost_gap']['mean']:
            best_by_cost = all_results[-1]

    print(f"\nBest by LMP mean:  alpha={best_by_lmp['alpha']}  "
          f"cost_gap={best_by_lmp['cost_gap']['mean']:.4f}  "
          f"lmp_mean={best_by_lmp['lmp_mae']['mean']:.4f}")
    print(f"Best by cost_gap:  alpha={best_by_cost['alpha']}  "
          f"cost_gap={best_by_cost['cost_gap']['mean']:.4f}  "
          f"lmp_mean={best_by_cost['lmp_mae']['mean']:.4f}")

    metrics = {
        "experiment": "ensemble_v9_v11",
        "description": "Sweep over alpha (v9 weight) in pred_cost = alpha*v9 + (1-alpha)*v11",
        "models": {"v9": "design_a_v9.pt (W_AUX=0.5, best LMP)",
                   "v11": "design_a_v11.pt (W_AUX=2.0, best cost_gap)"},
        "results": all_results,
        "best_by_lmp_mean": best_by_lmp,
        "best_by_cost_gap": best_by_cost,
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    json.dump(metrics, open(METRICS_OUT, "w"), indent=2)
    print(f"\nMetrics written to {METRICS_OUT}")


if __name__ == "__main__":
    main()
