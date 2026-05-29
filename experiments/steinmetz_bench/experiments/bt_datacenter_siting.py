"""Data-center siting backtest (roadmap item 3.1, Steinmetz §7.1-A).

Where you plug a large new load into the grid decides what it pays for power. This
backtest ranks candidate nodes for siting a data center by the *distribution* of the
nodal price it would face — its LMP duration curve — plus how often its demand would
be curtailed, then quantifies the realized effective ``$/MWh`` saved by choosing the
best node over a naive default.

The synthetic world (loop-runnable, the path tested here) is a star: a hub bus holds
a large cheap generator; each candidate bus has a local load, an expensive backstop
generator, and a tie line back to the hub. One candidate (``cheap_node``) is given a
fat tie line so it imports cheap hub power even with the data center attached — its
LMP sits at the hub's marginal cost — while the other candidates have thin lines that
saturate once the data center lands, forcing their pricey local backstop on the
margin (and, for a deliberately capacity-starved node, occasional curtailment at the
value of lost load). The cheap node is therefore the correct siting choice, and the
test asserts the ranking recovers it.

Every number is computed from real zap DC-OPF solves: for each scenario (seeded
fuel-cost noise + load scaling) and each candidate placement the data center is
attached at that node, the network is solved, and the node's realized LMP and served
fraction are read straight off the dispatch. The headline ``$/MWh`` delta is the
consumption-relevant price gap between the default and recommended nodes, paired per
scenario-hour, with a bootstrap CI over those pairs. Nothing is hand-written.

A ``--real`` path is reserved for staged ISO data (a human drops node price/load
history into ``data/<name>/``); it blocks via :class:`DataNotStagedError` rather than
downloading, matching the rest of the suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa
from attrs import define, field

from zap.devices import ACLine, Generator, Load
from zap.network import PowerNetwork

from experiments.steinmetz_bench.datasets.registry import DATA_ROOT, DataNotStagedError
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import (
    CIResult,
    DurationCurve,
    FidelityBand,
    bootstrap_ci,
    duration_curve,
    fidelity_band,
)

EXPERIMENT_ID = "3.1-datacenter-siting"
DATASET = "synthetic-siting-star"

# A node counts as curtailed in an hour when its served data-center load falls more
# than this many MW short of the requested flat draw.
CURTAIL_TOL = 1e-3

# Per-tie-line susceptance of the star network (shared by the zap build and its
# PyPSA twin, where the equivalent line reactance is ``1 / _TIE_SUSCEPTANCE``).
_TIE_SUSCEPTANCE = 10.0


@define(kw_only=True)
class SitingConfig:
    """Knobs of the synthetic siting world.

    ``cheap_node`` gets the fat tie line (``line_caps`` large there) so it imports
    hub power at the hub's marginal cost; the capacity-starved node has a small
    backstop so it curtails under high load. ``default_node`` is the naive baseline
    the savings are measured against. All randomness is seeded.
    """

    n_candidates: int = 4
    hours: int = 24
    n_scenarios: int = 8
    dc_mw: float = 80.0
    cheap_node: int = 2
    default_node: int = 0
    hub_cost: float = 10.0
    backstop_costs: tuple[float, ...] = (60.0, 80.0, 100.0, 70.0)
    line_caps: tuple[float, ...] = (90.0, 90.0, 600.0, 90.0)
    backstop_caps: tuple[float, ...] = (300.0, 300.0, 300.0, 45.0)
    hub_cap: float = 4000.0
    baseline_peak: float = 48.0
    baseline_amp: float = 14.0
    voll: float = 1000.0
    cost_noise: float = 0.06
    load_scale_lo: float = 0.85
    load_scale_hi: float = 1.45
    seed: int = 0

    def __attrs_post_init__(self):
        k = self.n_candidates
        for name in ("backstop_costs", "line_caps", "backstop_caps"):
            if len(getattr(self, name)) != k:
                raise ValueError(f"{name} must have {k} entries (one per candidate)")
        for node in (self.cheap_node, self.default_node):
            if not 0 <= node < k:
                raise ValueError(f"node {node} out of range for {k} candidates")


@define(kw_only=True)
class Scenario:
    """One seeded operating scenario: per-(gen, hour) costs and per-(node, hour) load."""

    gen_cost: np.ndarray  # (n_candidates + 1, hours); row 0 is the hub
    load_profile: np.ndarray  # (n_candidates, hours)


@define(kw_only=True)
class NodeSiting:
    """Realized price + curtailment outcome of siting the data center at one node.

    ``lmp`` and ``served`` are ``(scenario, hour)`` arrays read off real solves;
    every summary below is derived from them, not asserted.
    """

    node: int
    lmp: np.ndarray  # (scenario, hour), $/MWh the data center faces
    served: np.ndarray  # (scenario, hour), MW actually delivered to the data center
    requested_mw: float

    @property
    def effective_price(self) -> float:
        """Mean nodal price across all scenario-hours — the duration-curve level."""
        return float(self.lmp.mean())

    @property
    def curtailment_frequency(self) -> float:
        """Fraction of scenario-hours the data center's draw was curtailed."""
        shortfall = self.requested_mw - self.served
        return float(np.mean(shortfall > CURTAIL_TOL))

    @property
    def duration_curve(self) -> DurationCurve:
        return duration_curve(self.lmp.ravel())

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.lmp, p))


@define(kw_only=True)
class SitingResult:
    """Ranked candidates plus the realized $/MWh delta of the siting decision."""

    config: SitingConfig
    nodes: dict[int, NodeSiting]
    recommended_node: int
    default_node: int
    delta_samples: np.ndarray  # (scenario * hour,), default LMP - recommended LMP
    ci: CIResult
    fidelity: FidelityBand  # DC-vs-PyPSA LMP gap on the headline placements
    source: str = field(default="synthetic")

    @property
    def headline_delta(self) -> float:
        """Effective $/MWh saved by the recommended node vs the default node."""
        rec = self.nodes[self.recommended_node].effective_price
        dflt = self.nodes[self.default_node].effective_price
        return dflt - rec

    def ranking(self) -> list[int]:
        """Candidate nodes ordered best (cheapest, least-curtailed) first."""
        return sorted(
            self.nodes,
            key=lambda n: (
                self.nodes[n].effective_price,
                self.nodes[n].curtailment_frequency,
                n,
            ),
        )

    def to_bench_result(self) -> BenchResult:
        rec = self.nodes[self.recommended_node]
        dflt = self.nodes[self.default_node]
        per_node = {
            str(n): {
                "effective_price": s.effective_price,
                "p50": s.percentile(50.0),
                "p90": s.percentile(90.0),
                "p95": s.percentile(95.0),
                "max": float(s.lmp.max()),
                "curtailment_frequency": s.curtailment_frequency,
            }
            for n, s in self.nodes.items()
        }
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.headline_delta,
            units="$/MWh",
            ci=self.ci,
            fidelity_band=self.fidelity,
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "pypsa_solver": "highs",
                "topology": "star: cheap hub generator + radial tie lines to candidates",
                "fidelity_band": (
                    "DC-vs-PyPSA nodal-LMP gap on the nominal scenario at the default "
                    "and recommended placements (both curtailment-free, so the two LPs "
                    "are identical and the band is the solver-vs-solver floor)"
                ),
                "dc_mw": self.config.dc_mw,
                "n_candidates": self.config.n_candidates,
                "n_scenarios": self.config.n_scenarios,
                "hours": self.config.hours,
                "default_node": self.default_node,
                "ranking_key": (
                    "ascending effective $/MWh (mean of the nodal LMP duration "
                    "curve), tie-broken by curtailment frequency"
                ),
                "headline": (
                    "effective $/MWh at the default node minus the recommended node, "
                    "paired per scenario-hour"
                ),
                "synthetic_note": (
                    "fuel costs carry seeded per-hour noise and loads a per-scenario "
                    "scale; a human stages real node price/load history and re-runs "
                    "with --real"
                ),
            },
            sensitivities={
                "recommended_node": self.recommended_node,
                "ranking": self.ranking(),
                "recommended_effective_price": rec.effective_price,
                "default_effective_price": dflt.effective_price,
                "recommended_curtailment_frequency": rec.curtailment_frequency,
                "default_curtailment_frequency": dflt.curtailment_frequency,
                "per_node": per_node,
            },
        )


def _baseline_shape(config: SitingConfig) -> np.ndarray:
    """A daily-shaped per-hour baseline load (before per-scenario scaling)."""
    t = np.arange(config.hours)
    shape = config.baseline_peak + config.baseline_amp * np.sin(2 * np.pi * (t - 6) / 24.0)
    return np.clip(shape, 1.0, None)


def _make_scenarios(config: SitingConfig) -> list[Scenario]:
    """Draw the seeded operating scenarios shared across all candidate placements."""
    rng = np.random.default_rng(config.seed)
    base = _baseline_shape(config)
    n_gen = config.n_candidates + 1
    costs = np.concatenate([[config.hub_cost], np.asarray(config.backstop_costs, float)])

    scenarios = []
    for _ in range(config.n_scenarios):
        noise = rng.normal(0.0, config.cost_noise, size=(n_gen, config.hours))
        gen_cost = np.clip(costs[:, None] * (1.0 + noise), 1e-3, None)
        scale = rng.uniform(config.load_scale_lo, config.load_scale_hi)
        load_profile = np.tile(base * scale, (config.n_candidates, 1))
        scenarios.append(Scenario(gen_cost=gen_cost, load_profile=load_profile))
    return scenarios


def _build_devices(config: SitingConfig, scenario: Scenario, dc_node: int):
    """Assemble the star network with the data center attached at ``dc_node``.

    Returns ``(network, devices, dc_index)`` where ``dc_index`` is the position of
    the data-center load in ``devices`` (so its served power can be read back).
    """
    k = config.n_candidates
    hub = k
    n_nodes = k + 1
    hours = config.hours

    net = PowerNetwork(n_nodes)
    gen_caps = np.concatenate([[config.hub_cap], np.asarray(config.backstop_caps, float)])
    generators = Generator(
        name=np.array(["hub"] + [f"backstop{i}" for i in range(k)]),
        num_nodes=n_nodes,
        terminal=np.concatenate([[hub], np.arange(k)]),
        dynamic_capacity=gen_caps[:, None] * np.ones((1, hours)),
        linear_cost=scenario.gen_cost,
        nominal_capacity=np.ones(k + 1),
        capital_cost=np.ones(k + 1),
        emission_rates=np.full(k + 1, 0.4),
    )
    loads = Load(
        name=np.array([f"load{i}" for i in range(k)]),
        num_nodes=n_nodes,
        terminal=np.arange(k),
        load=scenario.load_profile,
        linear_cost=np.full(k, config.voll),
    )
    lines = ACLine(
        name=np.array([f"tie{i}" for i in range(k)]),
        num_nodes=n_nodes,
        source_terminal=np.full(k, hub),
        sink_terminal=np.arange(k),
        susceptance=np.full(k, _TIE_SUSCEPTANCE),
        capacity=np.ones(k),
        nominal_capacity=np.asarray(config.line_caps, float),
        linear_cost=0.01 * np.ones(k),
        capital_cost=np.ones(k),
    )
    datacenter = Load(
        name=np.array(["datacenter"]),
        num_nodes=n_nodes,
        terminal=np.array([dc_node]),
        load=np.full((1, hours), config.dc_mw),
        linear_cost=np.array([config.voll]),
    )
    devices = [generators, loads, lines, datacenter]
    return net, devices, len(devices) - 1


def _solve_placement(config: SitingConfig, scenario: Scenario, dc_node: int):
    """Solve one scenario with the data center at ``dc_node``; return (lmp, served)."""
    net, devices, dc_index = _build_devices(config, scenario, dc_node)
    out = net.dispatch(devices, time_horizon=config.hours, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    lmp = np.asarray(out.prices, dtype=float)[dc_node]
    served = -np.asarray(out.power[dc_index][0], dtype=float).ravel()
    return lmp, served


def _nominal_scenario(config: SitingConfig) -> Scenario:
    """The no-noise, mid-load-scale scenario used for the PyPSA fidelity check.

    Stripping the seeded cost noise makes generator marginal costs constant over
    the horizon, so the zap and PyPSA LPs are term-for-term identical and any LMP
    gap is pure solver-vs-solver numerical noise.
    """
    base = _baseline_shape(config)
    n_gen = config.n_candidates + 1
    costs = np.concatenate([[config.hub_cost], np.asarray(config.backstop_costs, float)])
    gen_cost = costs[:, None] * np.ones((1, config.hours))
    mid_scale = 0.5 * (config.load_scale_lo + config.load_scale_hi)
    load_profile = np.tile(base * mid_scale, (config.n_candidates, 1))
    assert gen_cost.shape == (n_gen, config.hours)
    return Scenario(gen_cost=gen_cost, load_profile=load_profile)


def _pypsa_bus_names(config: SitingConfig) -> list[str]:
    """Bus order matching zap node indices: candidate ``i`` -> node ``i``, hub last."""
    return [f"cand{i}" for i in range(config.n_candidates)] + ["hub"]


def build_pypsa_placement(
    config: SitingConfig, scenario: Scenario, dc_node: int
) -> pypsa.Network:
    """Equivalent PyPSA star network with the data center attached at ``dc_node``.

    Generator marginal costs are read from ``scenario`` (constant over the horizon
    for the nominal scenario). Each candidate bus carries a load-shedding generator
    priced at the value of lost load, mirroring zap's curtailable :class:`Load`
    (whose ``linear_cost`` is the VOLL penalty for unserved demand), so the two
    formulations remain equivalent even if a placement curtails.
    """
    k = config.n_candidates
    bus_names = _pypsa_bus_names(config)
    snapshots = pd.date_range("2025-01-01", periods=config.hours, freq="h")
    pn = pypsa.Network()
    pn.set_snapshots(snapshots)
    for bus in bus_names:
        pn.add("Bus", bus)
    pn.add("Carrier", "ac", co2_emissions=0.0)

    pn.add("Generator", "hub", bus="hub", p_nom=config.hub_cap,
           marginal_cost=float(scenario.gen_cost[0, 0]), carrier="ac")
    shed_pnom = config.baseline_peak * config.load_scale_hi + config.dc_mw
    for i in range(k):
        pn.add("Generator", f"backstop{i}", bus=f"cand{i}",
               p_nom=float(config.backstop_caps[i]),
               marginal_cost=float(scenario.gen_cost[i + 1, 0]), carrier="ac")
        pn.add("Generator", f"shed{i}", bus=f"cand{i}", p_nom=float(shed_pnom),
               marginal_cost=config.voll, carrier="ac")
        pn.add("Load", f"load{i}", bus=f"cand{i}", p_set=scenario.load_profile[i])
        pn.add("Line", f"tie{i}", bus0="hub", bus1=f"cand{i}",
               s_nom=float(config.line_caps[i]), x=1.0 / _TIE_SUSCEPTANCE)
    pn.add("Load", "datacenter", bus=f"cand{dc_node}",
           p_set=np.full(config.hours, config.dc_mw))
    return pn


def run_pypsa_fidelity(config: SitingConfig, placements) -> FidelityBand:
    """DC-vs-PyPSA nodal-LMP band over the given placements on the nominal scenario.

    For each placement the star network is solved in zap (CLARABEL) and in PyPSA
    (HiGHS) and their full nodal-price vectors are compared. ``placements`` are the
    default and recommended nodes — the ones the headline $/MWh delta is read from.
    """
    nominal = _nominal_scenario(config)
    bus_names = _pypsa_bus_names(config)
    zap_prices: list[np.ndarray] = []
    pypsa_prices: list[np.ndarray] = []
    for node in placements:
        net, devices, _ = _build_devices(config, nominal, node)
        out = net.dispatch(devices, time_horizon=config.hours, solver=cp.CLARABEL)
        if out.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
        zap_prices.append(np.asarray(out.prices, dtype=float))

        pn = build_pypsa_placement(config, nominal, node)
        pn.snapshot_weightings.loc[:, :] = 1.0
        pn.optimize(solver_name="highs")
        pl = pn.buses_t.marginal_price[bus_names].to_numpy(dtype=float).T
        pypsa_prices.append(pl)

    zap = np.concatenate([a.ravel() for a in zap_prices])
    pypsa = np.concatenate([a.ravel() for a in pypsa_prices])
    return fidelity_band(zap, pypsa, reference="pypsa-dc", metric="lmp", units="$/MWh")


def run_siting(config: Optional[SitingConfig] = None) -> SitingResult:
    """Rank candidate nodes and quantify the siting $/MWh delta on synthetic data."""
    config = config or SitingConfig()
    scenarios = _make_scenarios(config)
    hours = config.hours
    s = config.n_scenarios

    nodes: dict[int, NodeSiting] = {}
    for node in range(config.n_candidates):
        lmp = np.empty((s, hours))
        served = np.empty((s, hours))
        for j, scenario in enumerate(scenarios):
            lmp[j], served[j] = _solve_placement(config, scenario, node)
        nodes[node] = NodeSiting(
            node=node, lmp=lmp, served=served, requested_mw=config.dc_mw
        )

    recommended = sorted(
        nodes,
        key=lambda n: (nodes[n].effective_price, nodes[n].curtailment_frequency, n),
    )[0]
    default = config.default_node
    delta_samples = (nodes[default].lmp - nodes[recommended].lmp).ravel()
    ci = bootstrap_ci(delta_samples, statistic=np.mean, confidence=0.90, seed=0)
    fidelity = run_pypsa_fidelity(config, sorted({default, recommended}))
    return SitingResult(
        config=config,
        nodes=nodes,
        recommended_node=recommended,
        default_node=default,
        delta_samples=delta_samples,
        ci=ci,
        fidelity=fidelity,
        source="synthetic",
    )


def load_staged_siting(name: str) -> None:
    """Human ``--real`` entry point: read staged node price/load history.

    Raises :class:`DataNotStagedError` when ``data/<name>/`` is empty (the loop path),
    never downloading. A human stages real ISO node history and wires the loader.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged siting data for {name!r}: expected node price/load history "
            f"under {cache_dir}. A human must stage real ISO data there (see "
            f"data/README.md); the benchmark loop never downloads."
        )
    raise NotImplementedError(
        "real staged-data siting is wired up by a human; the synthetic path is the "
        "loop-runnable one."
    )


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic siting backtest and emit (optionally write) a ``BenchResult``."""
    result = run_siting().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steinmetz data-center siting backtest (§7.1-A)")
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on the synthetic siting star (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="rank against staged ISO node history in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    if args.real:
        load_staged_siting(args.real)
    res = run_siting()
    print(f"recommended node : {res.recommended_node} (default {res.default_node})")
    print(f"ranking          : {res.ranking()}")
    print(f"$/MWh delta      : {res.headline_delta:.2f} "
          f"[{res.ci.lo:.2f}, {res.ci.hi:.2f}] (90% CI)")
    print(f"{'node':>4}{'eff $/MWh':>12}{'p90':>10}{'curtail freq':>14}")
    for n in res.ranking():
        s = res.nodes[n]
        print(f"{n:>4}{s.effective_price:>12.2f}{s.percentile(90):>10.2f}"
              f"{s.curtailment_frequency:>14.3f}")
