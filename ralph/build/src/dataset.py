"""
P1 — Dataset & features

For every (state, hour) pair in the GridSFM split, build a normalized graph
sample and cache the exact zap DC-OPF solution as supervised targets.

Graph representation
--------------------
Node features  (n_bus × 4):
  0  load_frac      — bus load / total system load     (sums to ≤ 1)
  1  gen_cap_frac   — gen pmax at bus / total gen cap  (sums to ≤ 1)
  2  degree_norm    — branch degree / max_degree
  3  is_ref         — 1 if reference/slack bus

Edge features  (n_branch × 2):
  0  susc_norm      — susceptance / median susceptance
  1  cap_norm       — thermal capacity / total load     (same scale as load)

Edge index     (2 × n_branch) long  — [src, dst]

Targets (from zap exact solve):
  target_pg   (n_gen,)   — generator dispatch [p.u.]
  target_lmp  (n_bus,)   — nodal LMPs [$/p.u.]
  target_obj  scalar     — total operating cost

Auxiliary:
  gen_bus     (n_gen,)   — bus index of each generator
  gen_pmax    (n_gen,)   — generator capacity [p.u.]
  gen_cost    (n_gen,)   — normalized linear cost
  n_bus, n_gen, n_branch — sizes
  name                   — "{state}_{hour}"

Caching
-------
Each sample is saved as `state/cache/{state}_{hour}.npz`.  A mtime hash of
the source JSON is stored alongside; if the source has not changed, the cache
is reused without re-solving.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Make the local src importable when run as a script.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gridsfm_zap import load_case, solve_with_zap
import zap


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class GraphSample:
    name: str
    n_bus: int
    n_gen: int
    n_branch: int

    # Graph structure
    node_feats: np.ndarray       # (n_bus, 4)  float32
    edge_index: np.ndarray       # (2, n_branch) int32
    edge_feats: np.ndarray       # (n_branch, 2)  float32

    # Generator auxiliary
    gen_bus: np.ndarray          # (n_gen,)  int32
    gen_pmax: np.ndarray         # (n_gen,)  float32
    gen_cost: np.ndarray         # (n_gen,)  float32  (normalized)

    # Supervised targets from zap
    target_pg: np.ndarray        # (n_gen,)   float32
    target_lmp: np.ndarray       # (n_bus,)   float32
    target_obj: float

    # GridSFM DC reference dispatch fraction (pg_ref / pmax) — strong merit-order proxy
    # Loaded from {state}_dc_results.json. Spearman rank corr with true cost: -0.93 to -0.98.
    # High fraction → cheap/baseload generator; ~0 → expensive or must-not-run.
    gen_dc_frac: np.ndarray      # (n_gen,)   float32  (clipped to [0, 1.5])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtime_hash(paths: list[str]) -> str:
    h = hashlib.md5()
    for p in sorted(paths):
        try:
            h.update(str(os.path.getmtime(p)).encode())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


def _device_index(devices, cls):
    for i, d in enumerate(devices):
        if isinstance(d, cls):
            return i
    return None


# ---------------------------------------------------------------------------
# Build one sample
# ---------------------------------------------------------------------------

def build_sample(state: str, hour: str, data_dir: str) -> Optional[GraphSample]:
    model_path = os.path.join(data_dir, hour, f"{state}_model.json")
    dc_path = os.path.join(data_dir, hour, f"{state}_dc_results.json")
    if not os.path.exists(model_path):
        return None

    # --- Convert to zap ---
    dc_p = dc_path if os.path.exists(dc_path) else None
    case = load_case(model_path, dc_p)

    # --- Solve with zap ---
    out = solve_with_zap(case)

    # --- Extract targets ---
    gi = _device_index(case.devices, zap.Generator)
    if gi is None:
        return None  # no generators → skip

    gen_dispatch = np.asarray(out.power[gi][0]).ravel().astype(np.float32)  # (n_gen,)
    lmps = np.asarray(out.prices).ravel().astype(np.float32)               # (n_bus,)
    obj = float(out.problem.value)

    # --- Build branch incidence ---
    acli = _device_index(case.devices, zap.ACLine)
    src_arr, dst_arr, susc_arr, cap_arr = [], [], [], []
    if acli is not None:
        ac = case.devices[acli]
        src_arr.extend(np.asarray(ac.source_terminal).ravel().tolist())
        dst_arr.extend(np.asarray(ac.sink_terminal).ravel().tolist())
        susc_arr.extend(np.asarray(ac.susceptance).ravel().tolist())
        cap_arr.extend(np.asarray(ac.capacity).ravel().tolist())

    dcli = _device_index(case.devices, zap.DCLine)
    if dcli is not None:
        dl = case.devices[dcli]
        src_arr.extend(np.asarray(dl.source_terminal).ravel().tolist())
        dst_arr.extend(np.asarray(dl.sink_terminal).ravel().tolist())
        susc_arr.extend([1.0] * len(np.asarray(dl.source_terminal).ravel()))
        cap_arr.extend(np.asarray(dl.capacity).ravel().tolist())

    src_arr = np.array(src_arr, dtype=np.int32)
    dst_arr = np.array(dst_arr, dtype=np.int32)
    susc_arr = np.array(susc_arr, dtype=np.float32)
    cap_arr = np.array(cap_arr, dtype=np.float32)
    n_branch = len(src_arr)

    # --- Node features ---
    n_bus = case.n_bus
    gen_dev = case.devices[gi]
    gen_terminals = np.asarray(gen_dev.terminal).ravel()
    gen_pmax_raw = np.asarray(gen_dev.dynamic_capacity).ravel().astype(np.float32)  # (n_gen,)
    gen_cost_raw = np.asarray(gen_dev.linear_cost).ravel().astype(np.float32)

    # Per-bus load
    ldi = _device_index(case.devices, zap.Load)
    bus_load = np.zeros(n_bus, dtype=np.float32)
    if ldi is not None:
        ld = case.devices[ldi]
        lt = np.asarray(ld.terminal).ravel()
        lp = np.asarray(ld.load).ravel().astype(np.float32)
        for t, p in zip(lt, lp):
            bus_load[t] += p

    total_load = bus_load.sum()
    load_frac = bus_load / max(float(total_load), 1e-6)

    # Per-bus gen capacity
    bus_gen_cap = np.zeros(n_bus, dtype=np.float32)
    for t, cap in zip(gen_terminals, gen_pmax_raw):
        bus_gen_cap[t] += cap
    total_gen_cap = bus_gen_cap.sum()
    gen_cap_frac = bus_gen_cap / max(float(total_gen_cap), 1e-6)

    # Degree
    degree = np.zeros(n_bus, dtype=np.float32)
    for s, d in zip(src_arr, dst_arr):
        degree[s] += 1
        degree[d] += 1
    max_deg = max(float(degree.max()), 1.0)
    degree_norm = degree / max_deg

    # is_ref
    is_ref = np.zeros(n_bus, dtype=np.float32)
    is_ref[case.ref_bus_idx] = 1.0

    node_feats = np.stack([load_frac, gen_cap_frac, degree_norm, is_ref], axis=1)  # (n_bus, 4)

    # --- Edge features ---
    med_susc = float(np.median(susc_arr)) if susc_arr.size else 1.0
    susc_norm = susc_arr / max(med_susc, 1e-6)
    cap_norm = cap_arr / max(float(total_load), 1e-6)
    edge_feats = np.stack([susc_norm, cap_norm], axis=1)  # (n_branch, 2)

    edge_index = np.stack([src_arr, dst_arr], axis=0)  # (2, n_branch)

    # GridSFM DC reference dispatch fraction (from case.target_pg loaded by load_case)
    n_gen_model = len(gen_terminals)
    if case.target_pg is not None and len(case.target_pg) >= n_gen_model:
        dc_pg = case.target_pg[:n_gen_model].astype(np.float32)
        pmax_safe_dc = np.where(gen_pmax_raw > 1e-6, gen_pmax_raw, 1.0)
        dc_frac = np.clip(dc_pg / pmax_safe_dc, 0.0, 1.5).astype(np.float32)
    else:
        dc_frac = np.full(n_gen_model, 0.5, dtype=np.float32)

    return GraphSample(
        name=f"{state}_{hour}",
        n_bus=n_bus,
        n_gen=len(gen_terminals),
        n_branch=n_branch,
        node_feats=node_feats.astype(np.float32),
        edge_index=edge_index,
        edge_feats=edge_feats.astype(np.float32),
        gen_bus=gen_terminals.astype(np.int32),
        gen_pmax=gen_pmax_raw,
        gen_cost=gen_cost_raw,
        target_pg=gen_dispatch,
        target_lmp=lmps,
        target_obj=obj,
        gen_dc_frac=dc_frac,
    )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, state: str, hour: str) -> str:
    return os.path.join(cache_dir, f"{state}_{hour}.npz")


def _hash_path(cache_dir: str, state: str, hour: str) -> str:
    return os.path.join(cache_dir, f"{state}_{hour}.hash")


def _save_sample(sample: GraphSample, cache_dir: str, state: str, hour: str):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        _cache_path(cache_dir, state, hour),
        name=np.array([sample.name]),
        n_bus=np.array([sample.n_bus]),
        n_gen=np.array([sample.n_gen]),
        n_branch=np.array([sample.n_branch]),
        node_feats=sample.node_feats,
        edge_index=sample.edge_index,
        edge_feats=sample.edge_feats,
        gen_bus=sample.gen_bus,
        gen_pmax=sample.gen_pmax,
        gen_cost=sample.gen_cost,
        target_pg=sample.target_pg,
        target_lmp=sample.target_lmp,
        target_obj=np.array([sample.target_obj]),
        gen_dc_frac=sample.gen_dc_frac,
    )


def _load_sample(cache_dir: str, state: str, hour: str) -> Optional[GraphSample]:
    p = _cache_path(cache_dir, state, hour)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return GraphSample(
        name=str(d["name"][0]),
        n_bus=int(d["n_bus"][0]),
        n_gen=int(d["n_gen"][0]),
        n_branch=int(d["n_branch"][0]),
        node_feats=d["node_feats"],
        edge_index=d["edge_index"],
        edge_feats=d["edge_feats"],
        gen_bus=d["gen_bus"],
        gen_pmax=d["gen_pmax"],
        gen_cost=d["gen_cost"],
        target_pg=d["target_pg"],
        target_lmp=d["target_lmp"],
        target_obj=float(d["target_obj"][0]),
        gen_dc_frac=d["gen_dc_frac"] if "gen_dc_frac" in d else np.full(int(d["n_gen"][0]), 0.5, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Build / load dataset
# ---------------------------------------------------------------------------

def build_cache(
    data_dir: str,
    split_json: str,
    cache_dir: str,
    verbose: bool = True,
) -> dict:
    """Build and cache all samples.  Returns summary dict."""
    split = json.load(open(split_json))
    hours = split["hours"]
    all_states = split["train"] + split["test"]
    os.makedirs(cache_dir, exist_ok=True)

    results = {}
    skipped = 0
    built = 0
    failed = 0

    for state in all_states:
        for hour in hours:
            key = f"{state}_{hour}"
            model_path = os.path.join(data_dir, hour, f"{state}_model.json")
            src_files = [
                model_path,
                os.path.join(data_dir, hour, f"{state}_dc_results.json"),
            ]
            cur_hash = _mtime_hash(src_files)
            hash_p = _hash_path(cache_dir, state, hour)
            cache_p = _cache_path(cache_dir, state, hour)

            # Check cache validity
            if os.path.exists(cache_p) and os.path.exists(hash_p):
                saved_hash = open(hash_p).read().strip()
                if saved_hash == cur_hash:
                    skipped += 1
                    if verbose:
                        print(f"  [cache hit]  {key}")
                    results[key] = "cached"
                    continue

            if verbose:
                print(f"  [building]   {key} ...", end="", flush=True)
            t0 = time.time()
            try:
                sample = build_sample(state, hour, data_dir)
                if sample is None:
                    raise RuntimeError("build_sample returned None")
                _save_sample(sample, cache_dir, state, hour)
                open(hash_p, "w").write(cur_hash)
                elapsed = time.time() - t0
                if verbose:
                    print(f" done ({elapsed:.1f}s)  buses={sample.n_bus}  gens={sample.n_gen}  obj={sample.target_obj:.4f}")
                results[key] = "built"
                built += 1
            except Exception as e:
                elapsed = time.time() - t0
                if verbose:
                    print(f" FAILED ({elapsed:.1f}s): {e}")
                results[key] = f"error: {e}"
                failed += 1

    return {"results": results, "built": built, "skipped": skipped, "failed": failed}


def load_dataset(split: str, data_dir: str, split_json: str, cache_dir: str) -> list[GraphSample]:
    """Load all cached samples for 'train' or 'test' split."""
    sp = json.load(open(split_json))
    states = sp[split]
    hours = sp["hours"]
    samples = []
    for state in states:
        for hour in hours:
            s = _load_sample(cache_dir, state, hour)
            if s is not None:
                samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# CLI entry-point — builds cache and writes P1_metrics.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ralph/build/data/raw")
    parser.add_argument("--split-json", default="ralph/build/data/raw/split.json")
    parser.add_argument("--cache-dir", default="ralph/build/state/cache")
    parser.add_argument("--metrics-out", default="ralph/build/state/P1_metrics.json")
    args = parser.parse_args()

    print("=== P1 — Building dataset cache ===")
    summary = build_cache(args.data_dir, args.split_json, args.cache_dir, verbose=True)
    print(f"\nBuilt: {summary['built']}  Skipped (cached): {summary['skipped']}  Failed: {summary['failed']}")

    # Compute statistics across all samples
    split = json.load(open(args.split_json))
    all_samples = []
    for sp in ("train", "test"):
        all_samples.extend(load_dataset(sp, args.data_dir, args.split_json, args.cache_dir))

    n_total = len(all_samples)
    bus_counts = [s.n_bus for s in all_samples]
    gen_counts = [s.n_gen for s in all_samples]
    branch_counts = [s.n_branch for s in all_samples]
    objs = [s.target_obj for s in all_samples]
    lmp_means = [float(s.target_lmp.mean()) for s in all_samples]
    lmp_stds = [float(s.target_lmp.std()) for s in all_samples]

    # Load fraction stats
    load_fracs = np.concatenate([s.node_feats[:, 0] for s in all_samples])
    gen_cap_fracs = np.concatenate([s.node_feats[:, 1] for s in all_samples])
    susc_norms = np.concatenate([s.edge_feats[:, 0] for s in all_samples])
    cap_norms = np.concatenate([s.edge_feats[:, 1] for s in all_samples])

    metrics = {
        "phase": "P1",
        "n_total_samples": n_total,
        "n_train": len([s for s in all_samples if any(s.name.startswith(st) for st in split["train"])]),
        "n_test": len([s for s in all_samples if any(s.name.startswith(st) for st in split["test"])]),
        "cache_built": summary["built"],
        "cache_skipped": summary["skipped"],
        "cache_failed": summary["failed"],
        "bus_count": {
            "min": int(min(bus_counts)), "max": int(max(bus_counts)),
            "mean": float(np.mean(bus_counts)), "median": float(np.median(bus_counts)),
        },
        "gen_count": {
            "min": int(min(gen_counts)), "max": int(max(gen_counts)),
            "mean": float(np.mean(gen_counts)),
        },
        "branch_count": {
            "min": int(min(branch_counts)), "max": int(max(branch_counts)),
            "mean": float(np.mean(branch_counts)),
        },
        "objective": {
            "min": float(min(objs)), "max": float(max(objs)),
            "mean": float(np.mean(objs)), "median": float(np.median(objs)),
        },
        "lmp_mean_across_samples": {
            "min": float(min(lmp_means)), "max": float(max(lmp_means)),
            "mean": float(np.mean(lmp_means)),
        },
        "lmp_std_across_samples": {
            "mean": float(np.mean(lmp_stds)),
        },
        "node_feat_stats": {
            "load_frac":   {"mean": float(load_fracs.mean()), "std": float(load_fracs.std()), "max": float(load_fracs.max())},
            "gen_cap_frac":{"mean": float(gen_cap_fracs.mean()), "std": float(gen_cap_fracs.std()), "max": float(gen_cap_fracs.max())},
        },
        "edge_feat_stats": {
            "susc_norm": {"mean": float(susc_norms.mean()), "std": float(susc_norms.std()), "max": float(susc_norms.max())},
            "cap_norm":  {"mean": float(cap_norms.mean()), "std": float(cap_norms.std()), "max": float(cap_norms.max())},
        },
        "failed_cases": [k for k, v in summary["results"].items() if "error" in str(v)],
    }

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    json.dump(metrics, open(args.metrics_out, "w"), indent=2)
    print(f"\nMetrics written to {args.metrics_out}")
    print(f"Total samples: {n_total}  buses: {metrics['bus_count']['min']}–{metrics['bus_count']['max']}")
    print(f"Objective: {metrics['objective']['min']:.4f}–{metrics['objective']['max']:.4f}  (median {metrics['objective']['median']:.4f})")
    print(f"LMP mean range: {metrics['lmp_mean_across_samples']['min']:.4f}–{metrics['lmp_mean_across_samples']['max']:.4f}")
    if summary["failed"] == 0:
        print("All cases succeeded.")
    else:
        print(f"WARNING: {summary['failed']} cases failed.")
