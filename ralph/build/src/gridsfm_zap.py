"""
GridSFM (MATPOWER-format JSON) -> zap PowerNetwork converter.

This is the bridge that lets us solve Microsoft's GridSFM US power-grid models
with zap's differentiable DC-OPF and extract EXACT dual variables (LMPs), which
the GridSFM dataset itself does not provide.

Data schema (per state, per hour):
  {region}_model.json       MATPOWER-like: bus, gen, branch, load, dcline, shunt, baseMVA
  {region}_dc_results.json  DC-OPF primal solution: gen.pg, bus.va, branch flows, objective
  {region}_ac_results.json  AC-OPF primal solution (analogous)

We convert the *model* into zap devices and solve with zap. The DC results give a
reference objective/dispatch for validation.

Units: GridSFM is per-unit on baseMVA (per_unit=True). We keep everything in p.u.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import zap
from zap import PowerNetwork, Generator, Load, ACLine, DCLine, Ground


@dataclass
class ConvertedCase:
    name: str
    net: PowerNetwork
    devices: list
    n_bus: int
    bus_id_to_idx: dict
    ref_bus_idx: int
    dc_objective: float | None  # reference objective from GridSFM DC results
    # supervised targets (per-bus angle, per-gen dispatch), indexed in zap order
    target_va: np.ndarray | None
    target_pg: np.ndarray | None
    gen_bus_idx: np.ndarray


def _f(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def load_case(model_path: str, dc_results_path: str | None = None) -> ConvertedCase:
    model = json.load(open(model_path))
    name = model.get("name", "unknown")

    buses = model["bus"]
    # Stable, contiguous bus indexing.
    bus_ids = sorted(buses.keys(), key=lambda k: int(buses[k]["bus_i"]))
    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    n_bus = len(bus_ids)

    # Reference / slack bus: MATPOWER bus_type 3 == ref. Fall back to bus 0.
    ref_bus_idx = 0
    for bid in bus_ids:
        if int(buses[bid].get("bus_type", 1)) == 3:
            ref_bus_idx = bus_id_to_idx[bid]
            break

    devices = []

    # ---- Generators -------------------------------------------------------
    gens = model.get("gen", {})
    gen_terminals, gen_pmax, gen_pmin, gen_lin_cost = [], [], [], []
    gen_keys = [k for k in gens if int(gens[k].get("gen_status", 1)) == 1]
    for k in gen_keys:
        g = gens[k]
        b = str(g["gen_bus"])
        if b not in bus_id_to_idx:
            continue
        gen_terminals.append(bus_id_to_idx[b])
        gen_pmax.append(_f(g.get("pmax"), 0.0))
        gen_pmin.append(_f(g.get("pmin"), 0.0))
        # MATPOWER gencost model 2 polynomial: cost = [c_{n-1}, ..., c1, c0].
        # Linear coefficient is the second-to-last entry; convert $/p.u. roughly.
        cost = g.get("cost", [0.0, 0.0, 0.0])
        lin = cost[-2] if isinstance(cost, list) and len(cost) >= 2 else 1.0
        gen_lin_cost.append(_f(lin, 1.0))
    gen_terminals = np.array(gen_terminals, dtype=int)
    gen_pmax = np.array(gen_pmax)
    gen_lin_cost = np.array(gen_lin_cost)
    # Normalize costs to O(1)-O(100) to keep the LP well-scaled.
    if gen_lin_cost.size and gen_lin_cost.max() > 0:
        gen_lin_cost = gen_lin_cost / max(1.0, np.median(gen_lin_cost[gen_lin_cost > 0]))

    if gen_terminals.size:
        generator = Generator(
            num_nodes=n_bus,
            name="gen",
            terminal=gen_terminals,
            dynamic_capacity=gen_pmax.reshape(-1, 1),
            linear_cost=gen_lin_cost.reshape(-1, 1),
            nominal_capacity=np.ones(gen_terminals.size),
        )
        devices.append(generator)

    # ---- Loads ------------------------------------------------------------
    loads = model.get("load", {})
    load_terminals, load_pd = [], []
    for k, ld in loads.items():
        if int(ld.get("status", 1)) != 1:
            continue
        b = str(ld["load_bus"])
        if b not in bus_id_to_idx:
            continue
        load_terminals.append(bus_id_to_idx[b])
        load_pd.append(_f(ld.get("pd"), 0.0))
    load_terminals = np.array(load_terminals, dtype=int)
    load_pd = np.array(load_pd)
    if load_terminals.size:
        # High curtailment penalty (value of lost load) so demand is served when feasible.
        voll = 1e3 * max(1.0, float(gen_lin_cost.max()) if gen_lin_cost.size else 1.0)
        load = Load(
            num_nodes=n_bus,
            name="load",
            terminal=load_terminals,
            load=load_pd.reshape(-1, 1),
            linear_cost=np.full(load_terminals.size, voll),
        )
        devices.append(load)

    # ---- AC branches (DC power flow) -------------------------------------
    branches = model.get("branch", {})
    src, dst, susc, cap = [], [], [], []
    for k, br in branches.items():
        if int(br.get("br_status", 1)) != 1:
            continue
        fb, tb = str(br["f_bus"]), str(br["t_bus"])
        if fb not in bus_id_to_idx or tb not in bus_id_to_idx:
            continue
        x = _f(br.get("br_x"), 0.0)
        if abs(x) < 1e-9:
            continue
        src.append(bus_id_to_idx[fb])
        dst.append(bus_id_to_idx[tb])
        susc.append(1.0 / x)  # DC: susceptance b = 1/x
        rate = _f(br.get("rate_a"), 0.0)
        cap.append(rate if rate > 0 else 1e3)  # 0 rate => effectively unlimited
    if src:
        acline = ACLine(
            num_nodes=n_bus,
            name="acline",
            source_terminal=np.array(src, dtype=int),
            sink_terminal=np.array(dst, dtype=int),
            susceptance=np.array(susc),
            capacity=np.array(cap),
            nominal_capacity=np.ones(len(src)),
        )
        devices.append(acline)

    # ---- DC lines (controllable injections) ------------------------------
    dclines = model.get("dcline", {})
    dsrc, ddst, dcap = [], [], []
    for k, dl in dclines.items():
        if int(dl.get("br_status", 1)) != 1:
            continue
        fb, tb = str(dl.get("f_bus")), str(dl.get("t_bus"))
        if fb not in bus_id_to_idx or tb not in bus_id_to_idx:
            continue
        dsrc.append(bus_id_to_idx[fb])
        ddst.append(bus_id_to_idx[tb])
        pmax = _f(dl.get("pmaxf") or dl.get("pmax"), 0.0)
        dcap.append(pmax if pmax > 0 else 1e2)
    if dsrc:
        dcline = DCLine(
            num_nodes=n_bus,
            name="dcline",
            source_terminal=np.array(dsrc, dtype=int),
            sink_terminal=np.array(ddst, dtype=int),
            capacity=np.array(dcap),
            nominal_capacity=np.ones(len(dsrc)),
        )
        devices.append(dcline)

    # ---- Ground (sets reference bus angle = 0) ---------------------------
    devices.append(Ground(num_nodes=n_bus, terminal=np.array([ref_bus_idx])))

    net = PowerNetwork(n_bus)

    # ---- Reference solution (for validation) -----------------------------
    dc_obj, target_va, target_pg = None, None, None
    if dc_results_path is not None:
        try:
            dc = json.load(open(dc_results_path))
            dc_obj = _f(dc.get("objective"), None)
            sol = dc.get("solution", {})
            sbus = sol.get("bus", {})
            target_va = np.array(
                [_f(sbus.get(bid, {}).get("va"), 0.0) for bid in bus_ids]
            )
            sgen = sol.get("gen", {})
            target_pg = np.array([_f(sgen.get(k, {}).get("pg"), 0.0) for k in gen_keys])
        except Exception:
            pass

    return ConvertedCase(
        name=name,
        net=net,
        devices=devices,
        n_bus=n_bus,
        bus_id_to_idx=bus_id_to_idx,
        ref_bus_idx=ref_bus_idx,
        dc_objective=dc_obj,
        target_va=target_va,
        target_pg=target_pg,
        gen_bus_idx=gen_terminals,
    )


def solve_with_zap(case: ConvertedCase, solver=None):
    """Solve the converted case with zap's DC-OPF, with solver fallbacks."""
    import cvxpy as cp

    candidates = [solver] if solver is not None else [cp.CLARABEL, cp.ECOS, cp.SCS]
    last_err = None
    for s in candidates:
        try:
            return case.net.dispatch(
                case.devices, time_horizon=1, solver=s, add_ground=False,
            )
        except Exception as e:  # noqa: BLE001 - try next solver
            last_err = e
    raise RuntimeError(f"all solvers failed for {case.name}: {last_err}")


if __name__ == "__main__":
    import sys

    model_p = sys.argv[1] if len(sys.argv) > 1 else "ralph/build/data/raw/04h/delaware_model.json"
    dc_p = model_p.replace("_model.json", "_dc_results.json")
    case = load_case(model_p, dc_p)
    print(f"Case: {case.name}  buses={case.n_bus}  devices={[type(d).__name__ for d in case.devices]}")
    print(f"GridSFM DC objective (reference): {case.dc_objective}")
    out = solve_with_zap(case)
    cost = float(np.sum(out.power[0][0])) if out.power else float("nan")
    lmps = np.asarray(out.prices).ravel()
    print(f"zap solved. LMP stats: min={lmps.min():.4f} max={lmps.max():.4f} mean={lmps.mean():.4f}")
    print(f"distinct LMP levels (rounded 3dp): {len(np.unique(np.round(lmps,3)))}")
