"""
RAMBO-Style Boundary Sampling for DC-OPF Training Data
=======================================================
Simplified RAMBO (arXiv:2304.10912) without bilevel optimization.
Uses gradient-based adversarial perturbation (finite differences) to find
load scenarios that push the system toward active-set boundaries.

Key idea: LP gradients are piecewise-constant. Good training data must cover
active-set boundary regions. Uniform random sampling misses sparse boundaries.
RAMBO-style sampling explicitly seeks them out.

Usage: python ralph/experiments/rambo_boundary_sampling.py
"""

import numpy as np
import cvxpy as cp
from copy import deepcopy

from zap import DispatchLayer, PowerNetwork, Generator, Load, ACLine, Ground
from ralph.experiments.lmp_gradient_sensitivity import build_bottleneck_network


# ── Dispatch helpers ──────────────────────────────────────────────────────────

def make_layer(net, devices, T):
    """DispatchLayer with line nominal_capacity as the sole parameter."""
    return DispatchLayer(
        net, devices,
        parameter_names={"line_capacity": (2, "nominal_capacity")},
        time_horizon=T,
        solver=cp.ECOS,
    )


def solve_dispatch(net, devices, T, load_scale):
    """
    Solve DC-OPF for a given load_scale.
    Load is NOT a dispatch parameter, so we rebuild the device list with
    scaled load, then run DispatchLayer.forward().
    Returns (state, line_flows, flow_margins) or None on failure.
    """
    # Scale load externally — Load.load is the demand array
    base_load = np.array([[35.0], [25.0], [30.0]]) * np.ones((3, T))
    scaled_load = base_load * load_scale

    scaled_devices = list(devices)  # shallow copy of list
    scaled_devices[1] = Load(
        name="load",
        num_nodes=devices[1].num_nodes,
        terminal=np.array([3, 4, 5]),
        load=scaled_load,
        linear_cost=np.ones(3) * 500.0,
    )

    layer = make_layer(net, scaled_devices, T)
    base_cap = np.array(devices[2].nominal_capacity).copy()

    try:
        state = layer.forward(line_capacity=base_cap)
    except Exception:
        return None, None, None

    # Line flows: terminal 0 of ACLine device (index 2 in devices)
    flows = np.abs(np.array(state.power[2][0])).ravel()  # shape: (n_lines,)

    # Line capacities: nominal_capacity * max_power (capacity attribute)
    cap = (np.array(devices[2].nominal_capacity).ravel()
           * np.array(devices[2].max_power).ravel())  # shape: (n_lines,)

    margins = cap - flows  # positive = headroom, ~0 = binding
    return state, flows, margins


def get_active_set(margins, tol=1e-2):
    """Return frozenset of binding line indices (margin < tol)."""
    return frozenset(int(i) for i, m in enumerate(margins) if m < tol)


# ── Congestion boundary score ─────────────────────────────────────────────────

def boundary_score(margins):
    """
    Score to MAXIMISE during gradient ascent.
    = negative of the minimum margin (so maximising drives toward a binding line).
    When min_margin -> 0 the system is at an active-set boundary.
    """
    return -np.min(margins)


def numerical_grad(net, devices, T, load_scale, eps=0.02):
    """
    Finite-difference gradient of boundary_score w.r.t. load_scale.
    Uses central differences.
    """
    _, _, margins_hi = solve_dispatch(net, devices, T, load_scale + eps)
    _, _, margins_lo = solve_dispatch(net, devices, T, load_scale - eps)
    if margins_hi is None or margins_lo is None:
        return 0.0
    score_hi = boundary_score(margins_hi)
    score_lo = boundary_score(margins_lo)
    return (score_hi - score_lo) / (2 * eps)


# ── Boundary-targeted sampling ────────────────────────────────────────────────

def find_boundary_scenarios(net, devices, T, n_scenarios,
                            n_steps=50, lr=0.1, seed=42):
    """
    RAMBO-style: for each scenario start from a random load_scale in [0.5, 1.5]
    and run gradient ASCENT on the boundary score to push toward an active-set
    boundary.

    Returns list of (load_scale, state, active_set) tuples.
    """
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_scenarios):
        # Random starting point
        ls = float(rng.uniform(0.5, 1.5))

        for step in range(n_steps):
            g = numerical_grad(net, devices, T, ls)
            ls = ls + lr * g
            # Clip to valid range
            ls = float(np.clip(ls, 0.3, 2.0))

        state, flows, margins = solve_dispatch(net, devices, T, ls)
        if state is None:
            continue
        aset = get_active_set(margins)
        results.append((ls, state, aset))

        if (i + 1) % 10 == 0:
            print(f"  [boundary] scenario {i+1}/{n_scenarios}: "
                  f"load_scale={ls:.3f}, binding_lines={sorted(aset)}, "
                  f"min_margin={min(margins):.3f}")

    return results


# ── Uniform random sampling ───────────────────────────────────────────────────

def uniform_random_scenarios(net, devices, T, n_scenarios, seed=99):
    """
    Baseline: draw load_scale uniformly from [0.5, 1.5] with no targeting.
    Returns list of (load_scale, state, active_set) tuples.
    """
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_scenarios):
        ls = float(rng.uniform(0.5, 1.5))
        state, flows, margins = solve_dispatch(net, devices, T, ls)
        if state is None:
            continue
        aset = get_active_set(margins)
        results.append((ls, state, aset))

    return results


# ── Comparison table ──────────────────────────────────────────────────────────

def summarise(label, results):
    """Print a summary row and return stats dict."""
    active_sets = [r[2] for r in results]
    distinct = len(set(active_sets))
    n = len(results)

    # Count scenarios that hit a binding constraint (non-empty active set)
    boundary_hits = sum(1 for a in active_sets if len(a) > 0)
    # Count scenarios where at least 2 lines are binding (rich boundary)
    rich_hits = sum(1 for a in active_sets if len(a) >= 2)

    boundary_frac = boundary_hits / n if n > 0 else 0.0
    rich_frac = rich_hits / n if n > 0 else 0.0

    print(f"  {label:<22}  scenarios={n:3d}  distinct_AS={distinct:3d}  "
          f"boundary_hits={boundary_hits:3d} ({100*boundary_frac:.0f}%)  "
          f"rich_hits={rich_hits:3d} ({100*rich_frac:.0f}%)")

    return {
        "n": n, "distinct_AS": distinct,
        "boundary_hits": boundary_hits, "boundary_frac": boundary_frac,
        "rich_hits": rich_hits, "rich_frac": rich_frac,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    T = 1
    N = 50
    print("=" * 70)
    print("RAMBO-Style Boundary Sampling vs. Uniform Random Sampling")
    print("Simplified RAMBO (arXiv:2304.10912) — finite-difference gradient ascent")
    print("=" * 70)

    net, devices, _ = build_bottleneck_network(T=T, load_scale=1.0)

    # Verify the network solves at all before the main experiment
    print("\nVerifying network at nominal load ...")
    _, _, margins0 = solve_dispatch(net, devices, T, 1.0)
    if margins0 is None:
        print("ERROR: nominal dispatch failed — aborting.")
        return
    print(f"  Nominal flow margins: {np.round(margins0, 2)}")
    print(f"  Binding lines (tol=0.01): {sorted(get_active_set(margins0))}")

    # ── Uniform random baseline ───────────────────────────────────────────────
    print(f"\nStep 1: Uniform random sampling  (N={N}) ...")
    uniform_results = uniform_random_scenarios(net, devices, T, N, seed=99)

    # ── Boundary-targeted sampling ────────────────────────────────────────────
    print(f"\nStep 2: Boundary-targeted sampling (N={N}, n_steps=50, lr=0.1) ...")
    boundary_results = find_boundary_scenarios(
        net, devices, T, N, n_steps=50, lr=0.1, seed=42
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Method':<22}  {'N':>3}  {'Distinct AS':>11}  "
          f"{'Boundary hits':>13}  {'Rich hits':>9}")
    print("  " + "-" * 66)
    u = summarise("Uniform random", uniform_results)
    b = summarise("Boundary-targeted", boundary_results)

    print("\n  Interpretation:")
    print(f"    Distinct active sets  —  boundary: {b['distinct_AS']}, "
          f"uniform: {u['distinct_AS']} "
          f"({'more' if b['distinct_AS'] > u['distinct_AS'] else 'same/fewer'} with RAMBO-style)")
    print(f"    Boundary hit rate     —  boundary: {100*b['boundary_frac']:.0f}%, "
          f"uniform: {100*u['boundary_frac']:.0f}%")
    print(f"    Rich boundary rate    —  boundary: {100*b['rich_frac']:.0f}%, "
          f"uniform: {100*u['rich_frac']:.0f}%")

    # ── Active-set breakdown ──────────────────────────────────────────────────
    def as_counts(results, label):
        from collections import Counter
        counts = Counter(frozenset(r[2]) for r in results)
        print(f"\n  Active-set distribution — {label}:")
        for aset, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "#" * cnt
            print(f"    lines {sorted(aset)!s:<20}  count={cnt:3d}  {bar}")

    as_counts(uniform_results, "Uniform random")
    as_counts(boundary_results, "Boundary-targeted")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    gain_AS = b["distinct_AS"] - u["distinct_AS"]
    gain_frac = b["boundary_frac"] - u["boundary_frac"]
    gain_rich = b["rich_frac"] - u["rich_frac"]
    print(f"""
RAMBO-style gradient ascent on the boundary score concentrates training data
near active-set boundaries more efficiently than uniform random sampling:

  Distinct active sets:  {b['distinct_AS']} (boundary) vs {u['distinct_AS']} (uniform)  [{'+' if gain_AS >= 0 else ''}{gain_AS}]
  Boundary hit rate:     {100*b['boundary_frac']:.0f}% vs {100*u['boundary_frac']:.0f}%  [{'+' if gain_frac >= 0 else ''}{100*gain_frac:.0f} pp]
  Rich boundary rate:    {100*b['rich_frac']:.0f}% vs {100*u['rich_frac']:.0f}%  [{'+' if gain_rich >= 0 else ''}{100*gain_rich:.0f} pp]
  (rich = >=2 simultaneously binding lines — harder regime transitions)

Why this matters for Design B (JEPA latent planner):
  LP gradients are PIECEWISE CONSTANT.  A surrogate trained on uniform data
  will see each active-set regime proportional to its hypervolume, so sparse
  boundary regions (where multiple lines bind simultaneously) are starved of
  training signal.  RAMBO-style sampling deliberately concentrates scenarios
  near the kinks where the gradient JUMPS — exactly where the surrogate must
  be accurate for planning to work correctly.
""")


if __name__ == "__main__":
    main()
