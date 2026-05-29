"""PyPSA LP roundtrip reference (roadmap item 1.1).

Build one small, deterministic radial DC network, solve it two ways — zap's
conic dispatch (``CLARABEL``) and PyPSA's LP optimiser (HiGHS) — and report the
per-node LMP gap and per-line flow gap between them. This is the self-contained
validation reference the §8.4 accuracy benchmark (item 2.3) later builds on.

Why both representations are built from one shared spec rather than round-tripping
through ``zap.importers.pypsa``: that importer mutates read-only Copy-on-Write
arrays in place (``scale_costs`` does ``capital_cost /= ...``), which raises under
pandas >= 3.0 where CoW is mandatory and ``.values`` is read-only — and we may not
patch zap core from here. Building the zap devices and the PyPSA components from
the same parameter spec gives the two solvers an *identical* LP, so any remaining
gap is pure solver-vs-solver numerical noise — exactly the quantity this reference
exists to bound.

Sign / scale alignment (asserted empirically by the tests):

- zap dispatch ``prices`` are nodal power-balance duals laid out ``(node, hour)``;
  PyPSA ``buses_t.marginal_price`` is ``(snapshot, bus)``. With PyPSA snapshot
  weightings set to 1.0 the two are the same $/MWh nodal price, compared
  node-by-node in bus-insertion order.
- zap's line ``power[0]`` is the injection into the network at the *source*
  terminal — the negative of the flow leaving ``bus0`` — while PyPSA
  ``lines_t.p0`` is that outbound flow. The comparable zap flow is ``-power[0]``.

A radial (path) topology is chosen deliberately: on a radial network line flows
are fixed by power balance alone, independent of susceptance, so the DC-OPF flow
solution is unique and the comparison is not confounded by zap-vs-PyPSA
susceptance-scaling differences.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa
from attrs import define

from zap.devices import ACLine, Generator, Load
from zap.network import PowerNetwork

from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand, fidelity_band

EXPERIMENT_ID = "1.1-pypsa-roundtrip"
DATASET = "reference-3bus-radial"

# Acceptance tolerances, reused from zap's own PyPSA roundtrip test
# (``zap/tests/test_pypsa_dispatch.py``: PRICE_TOLERANCE / POWER_TOLERANCE).
LMP_GAP_TOL = 1e-2  # $/MWh
FLOW_GAP_TOL = 1e-3  # MW

_BUS_NAMES = ("bus0", "bus1", "bus2")
_LINE_NAMES = ("l01", "l12")


@define(kw_only=True)
class ReferenceSpec:
    """Parameters of the bundled 3-bus radial reference network.

    ``bus0`` hosts a cheap generator, ``bus2`` an expensive generator and the
    only load; ``bus1`` is a pass-through. The ``bus1``-``bus2`` line is the
    binding corridor — its capacity is set below peak load so the cheap unit
    cannot fully serve the peak, forcing the expensive unit on and separating
    nodal prices for part of the horizon (and leaving the rest uncongested).
    """

    load_profile: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0, 120.0, 50.0)
    cheap_cost: float = 10.0
    expensive_cost: float = 80.0
    gen_pnom: float = 300.0
    reactance: float = 0.1
    free_line_snom: float = 400.0
    congested_line_snom: float = 70.0

    @property
    def hours(self) -> int:
        return len(self.load_profile)


def build_zap_problem(spec: ReferenceSpec) -> tuple[PowerNetwork, list]:
    """Construct the zap network + device list for ``spec``."""
    hours = spec.hours
    load = np.asarray(spec.load_profile, dtype=float).reshape(1, hours)

    net = PowerNetwork(len(_BUS_NAMES))
    generators = Generator(
        name=np.array(["cheap", "expensive"]),
        num_nodes=net.num_nodes,
        terminal=np.array([0, 2]),
        dynamic_capacity=np.ones((2, hours)),
        nominal_capacity=np.full(2, spec.gen_pnom),
        linear_cost=np.array([spec.cheap_cost, spec.expensive_cost]),
        capital_cost=np.ones(2),
        emission_rates=np.array([0.4, 0.9]),
    )
    loads = Load(
        name=np.array(["load"]),
        num_nodes=net.num_nodes,
        terminal=np.array([2]),
        load=load,
        linear_cost=np.array([10_000.0]),  # value of lost load (never binds here)
    )
    lines = ACLine(
        name=np.array(list(_LINE_NAMES)),
        num_nodes=net.num_nodes,
        source_terminal=np.array([0, 1]),
        sink_terminal=np.array([1, 2]),
        susceptance=np.array([1.0 / spec.reactance, 1.0 / spec.reactance]),
        capacity=np.ones(2),
        nominal_capacity=np.array([spec.free_line_snom, spec.congested_line_snom]),
        linear_cost=np.zeros(2),
        capital_cost=np.ones(2),
    )
    return net, [generators, loads, lines]


def build_pypsa_problem(spec: ReferenceSpec) -> tuple[pypsa.Network, pd.DatetimeIndex]:
    """Construct the equivalent PyPSA network for ``spec``."""
    snapshots = pd.date_range("2025-01-01", periods=spec.hours, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    for bus in _BUS_NAMES:
        n.add("Bus", bus)
    n.add("Carrier", "ac", co2_emissions=0.0)
    n.add("Generator", "cheap", bus="bus0", p_nom=spec.gen_pnom,
          marginal_cost=spec.cheap_cost, carrier="ac")
    n.add("Generator", "expensive", bus="bus2", p_nom=spec.gen_pnom,
          marginal_cost=spec.expensive_cost, carrier="ac")
    n.add("Load", "load", bus="bus2", p_set=np.asarray(spec.load_profile, dtype=float))
    n.add("Line", "l01", bus0="bus0", bus1="bus1",
          s_nom=spec.free_line_snom, x=spec.reactance)
    n.add("Line", "l12", bus0="bus1", bus1="bus2",
          s_nom=spec.congested_line_snom, x=spec.reactance)
    return n, snapshots


@define(kw_only=True)
class RefComparison:
    """The aligned zap-vs-PyPSA solve and its gap statistics.

    Every array and band here is computed from two real solves; nothing is a
    hand-written constant. ``zap_flow`` / ``pypsa_flow`` are already sign-aligned
    to the same source->sink convention.
    """

    zap_lmp: np.ndarray  # (node, hour)
    pypsa_lmp: np.ndarray  # (node, hour)
    zap_flow: np.ndarray  # (line, hour), source->sink positive
    pypsa_flow: np.ndarray  # (line, hour), source->sink positive
    lmp_band: FidelityBand
    flow_band: FidelityBand
    zap_objective: float
    pypsa_objective: float

    @property
    def max_lmp_gap(self) -> float:
        return self.lmp_band.max_abs_gap

    @property
    def max_flow_gap(self) -> float:
        return self.flow_band.max_abs_gap

    @property
    def objective_rel_gap(self) -> float:
        denom = abs(self.pypsa_objective)
        if denom < 1e-9:
            return abs(self.zap_objective - self.pypsa_objective)
        return abs(self.zap_objective - self.pypsa_objective) / denom

    def to_bench_result(self) -> BenchResult:
        """Emit a :class:`BenchResult`: headline = the max nodal LMP gap.

        The LMP gap is the fidelity band; the flow gap and objective gap ride
        along in ``sensitivities`` so a reader sees all three validation numbers.
        """
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.max_lmp_gap,
            units="$/MWh",
            fidelity_band=self.lmp_band,
            assumptions={
                "topology": "3-bus radial path (bus0-bus1-bus2)",
                "zap_solver": "CLARABEL",
                "pypsa_solver": "highs",
                "n_nodes": int(self.zap_lmp.shape[0]),
                "n_lines": int(self.zap_flow.shape[0]),
                "hours": int(self.zap_lmp.shape[1]),
                "lmp_gap_tol": LMP_GAP_TOL,
                "flow_gap_tol": FLOW_GAP_TOL,
            },
            sensitivities={
                "max_flow_gap_mw": self.max_flow_gap,
                "mean_flow_gap_mw": self.flow_band.mean_abs_gap,
                "p90_flow_gap_mw": self.flow_band.p90_abs_gap,
                "objective_rel_gap": self.objective_rel_gap,
                "zap_objective": self.zap_objective,
                "pypsa_objective": self.pypsa_objective,
            },
        )


def _solve_zap(net: PowerNetwork, devices: list) -> tuple[np.ndarray, np.ndarray, float]:
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    line_idx = next(i for i, d in enumerate(devices) if isinstance(d, ACLine))
    lmp = np.asarray(out.prices, dtype=float)
    # zap source-terminal power is the negative of the bus0->bus1 outbound flow.
    flow = -np.asarray(out.power[line_idx][0], dtype=float)
    return lmp, flow, float(out.problem.value)


def _solve_pypsa(n: pypsa.Network) -> tuple[np.ndarray, np.ndarray, float]:
    n = n.copy()
    # Unit snapshot weightings so PyPSA's objective and marginal prices are the
    # unweighted per-snapshot quantities zap computes.
    n.snapshot_weightings.loc[:, :] = 1.0
    n.optimize(solver_name="highs")
    lmp = n.buses_t.marginal_price[list(_BUS_NAMES)].to_numpy(dtype=float).T
    flow = n.lines_t.p0[list(_LINE_NAMES)].to_numpy(dtype=float).T
    return lmp, flow, float(n.objective)


def run_reference(spec: ReferenceSpec | None = None) -> RefComparison:
    """Solve the bundled reference net in zap and PyPSA and align the results."""
    spec = spec or ReferenceSpec()

    zap_net, zap_devices = build_zap_problem(spec)
    pypsa_net, _ = build_pypsa_problem(spec)

    zap_lmp, zap_flow, zap_obj = _solve_zap(zap_net, zap_devices)
    pypsa_lmp, pypsa_flow, pypsa_obj = _solve_pypsa(pypsa_net)

    lmp_band = fidelity_band(zap_lmp, pypsa_lmp, reference="pypsa-dc",
                             metric="lmp", units="$/MWh")
    flow_band = fidelity_band(zap_flow, pypsa_flow, reference="pypsa-dc",
                              metric="flow", units="MW")

    return RefComparison(
        zap_lmp=zap_lmp,
        pypsa_lmp=pypsa_lmp,
        zap_flow=zap_flow,
        pypsa_flow=pypsa_flow,
        lmp_band=lmp_band,
        flow_band=flow_band,
        zap_objective=zap_obj,
        pypsa_objective=pypsa_obj,
    )


def run(report_path=None) -> BenchResult:
    """Run the reference comparison and emit (optionally write) a ``BenchResult``."""
    result = run_reference().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    comp = run_reference()
    print(f"max LMP gap : {comp.max_lmp_gap:.3e} $/MWh (tol {LMP_GAP_TOL})")
    print(f"max flow gap: {comp.max_flow_gap:.3e} MW    (tol {FLOW_GAP_TOL})")
    print(f"objective   : zap={comp.zap_objective:.4f} pypsa={comp.pypsa_objective:.4f}"
          f" (rel gap {comp.objective_rel_gap:.3e})")
