"""Vertically-integrated utility backtest (roadmap item 3.3, Steinmetz §7.2).

A vertically-integrated utility owns its fleet and its wires, so the dollars it leaves
on the table are the gap between the *least-cost* dispatch it could have run and the
*actual* dispatch it did run, plus the capacity it should build next. This backtest
quantifies both on synthetic-but-real-solve data.

**SCED vs. actual.** The synthetic world is a three-zone radial system (a path
``zone0-zone1-zone2``): zone 0 hosts a cheap generator, zone 1 a mid-cost unit, zone 2
an expensive one, each large enough to serve its own local load. The ``zone1-zone2``
corridor is deliberately thermal-limited, so coordinated security-constrained economic
dispatch (SCED) — zap's least-cost DC-OPF — ships cheap zone-0 power outward up to the
line limit and only then leans on the pricey local units. The "actual" dispatch models
a utility that does **not** coordinate economy transfers across zones: its inter-zonal
corridors are throttled to ``actual_line_factor`` of their true rating (default 0.0 —
fully islanded self-scheduling), so every zone self-supplies from its own (often
expensive) fleet. Because the islanded dispatch is a *feasible point of the coordinated
problem* (zero/throttled flows are always allowed) but not its optimum, the actual cost
can only be greater than or equal to the SCED cost — the difference is avoided fuel.
Both numbers are the objective value of a real zap solve; nothing is hand-written.

**5-year expansion ranking.** Given the binding corridor, what should the utility build?
For each candidate project (expand a line's rating, or add cheap-generator capacity) we
re-solve the SCED with that one capacity bumped and read the annual fuel saving straight
off the change in objective, then discount it over a five-year horizon and net out the
project's capital cost. Projects are ranked by net present value; the binding
``zone1-zone2`` corridor — the one SCED is paying to work around — comes out on top. The
headline ``NPV-delta`` is the best project's NPV.

**Fidelity.** The synthetic SCED is validated against PyPSA's LP optimiser on the same
radial spec (built from one shared parameter set so the two solvers see an identical LP);
the per-node LMP gap is the :class:`~...metrics.FidelityBand` attached to the result. A
radial topology is used deliberately so line flows are fixed by power balance alone and
the comparison is not confounded by zap-vs-PyPSA susceptance-scaling differences.

Every number is computed from an actual zap solve. The ``--real`` path is reserved for a
human who stages a real fleet/load/topology into ``data/<name>/``; it blocks via
:class:`DataNotStagedError` rather than downloading, matching the rest of the suite.
"""

from __future__ import annotations

import argparse
import copy
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
    FidelityBand,
    bootstrap_ci,
    fidelity_band,
)

EXPERIMENT_ID = "3.3-utility-sced"
DATASET = "synthetic-utility-3zone"

# Acceptance tolerance for the zap-vs-PyPSA LMP roundtrip, reused from item 1.1.
LMP_GAP_TOL = 1e-2  # $/MWh

_BUS_NAMES = ("zone0", "zone1", "zone2")
_LINE_NAMES = ("z01", "z12")


@define(kw_only=True)
class UtilityConfig:
    """Knobs of the synthetic three-zone utility world (one representative day).

    Zone 0 is the cheap exporter; zone 2 the expensive importer; the ``zone1-zone2``
    line (``line_caps[1]``) is the binding corridor. ``actual_line_factor`` throttles the
    inter-zonal lines in the "actual" (uncoordinated) dispatch — 0.0 means fully islanded
    self-scheduling. All randomness is seeded.
    """

    hours: int = 24
    n_scenarios: int = 16

    # Per-zone fleet: one generator per zone, ordered cheap -> expensive outward.
    gen_costs: tuple[float, ...] = (10.0, 35.0, 70.0)
    gen_caps: tuple[float, ...] = (200.0, 150.0, 150.0)

    # Per-zone baseline load (before per-scenario scaling), MW.
    base_loads: tuple[float, ...] = (50.0, 80.0, 90.0)
    load_amp: float = 0.25  # daily-shape amplitude as a fraction of the base load
    voll: float = 10_000.0  # value of lost load (never binds; loads are always served)

    # Radial corridors zone0-zone1-zone2 and their thermal ratings (MW).
    line_caps: tuple[float, ...] = (120.0, 70.0)
    reactance: float = 0.1

    # The "actual" uncoordinated dispatch throttles inter-zonal lines to this fraction
    # of their true rating (0.0 = islanded). Strictly < 1 so actual cost >= SCED cost.
    actual_line_factor: float = 0.0

    # 5-year expansion economics.
    expansion_delta_mw: float = 40.0
    expansion_years: int = 5
    discount_rate: float = 0.07
    days_per_year: int = 365
    # Annualized capital cost ($) of each candidate project's +delta_mw bump. Chosen so
    # the binding-corridor project clears NPV>0; the break-evens themselves are computed.
    line_capital_cost: float = 250_000.0
    gen_capital_cost: float = 900_000.0

    # Per-scenario seeded perturbations.
    cost_noise: float = 0.06
    load_scale_lo: float = 0.85
    load_scale_hi: float = 1.30
    seed: int = 0

    def __attrs_post_init__(self):
        n = len(self.gen_costs)
        if len(self.gen_caps) != n or len(self.base_loads) != n:
            raise ValueError("gen_costs, gen_caps and base_loads must share a length")
        if len(self.line_caps) != n - 1:
            raise ValueError("line_caps must have one entry per radial corridor (n_zones - 1)")
        if not 0.0 <= self.actual_line_factor < 1.0:
            raise ValueError("actual_line_factor must be in [0, 1)")
        # Each zone must be able to self-supply its peak so the islanded dispatch is
        # feasible (otherwise the 'actual' cost would include load shed, not just fuel).
        peak = max(self.base_loads) * (1.0 + self.load_amp) * self.load_scale_hi
        if any(cap < peak for cap in self.gen_caps):
            raise ValueError("every zone generator must cover its own peak load (islanding)")

    @property
    def n_zones(self) -> int:
        return len(self.gen_costs)


@define(kw_only=True)
class Scenario:
    """One seeded operating day: per-(gen, hour) linear cost and per-(zone, hour) load."""

    gen_cost: np.ndarray  # (n_zones, hours)
    load: np.ndarray  # (n_zones, hours)


def _daily_shape(config: UtilityConfig) -> np.ndarray:
    """Per-hour multiplier with a midday peak, mean 1.0."""
    t = np.arange(config.hours)
    return 1.0 + config.load_amp * np.sin(2 * np.pi * (t - 9) / 24.0)


def _nominal_scenario(config: UtilityConfig) -> Scenario:
    """The noise-free, unit-scaled representative day used for expansion + PyPSA checks."""
    shape = _daily_shape(config)
    gen_cost = np.asarray(config.gen_costs, float)[:, None] * np.ones((1, config.hours))
    load = np.asarray(config.base_loads, float)[:, None] * shape[None, :]
    return Scenario(gen_cost=gen_cost, load=load)


def _make_scenarios(config: UtilityConfig) -> list[Scenario]:
    """Draw the seeded scenarios for the SCED-vs-actual avoided-fuel distribution."""
    rng = np.random.default_rng(config.seed)
    shape = _daily_shape(config)
    costs = np.asarray(config.gen_costs, float)
    base = np.asarray(config.base_loads, float)

    scenarios = []
    for _ in range(config.n_scenarios):
        noise = rng.normal(0.0, config.cost_noise, size=(config.n_zones, config.hours))
        gen_cost = np.clip(costs[:, None] * (1.0 + noise), 1e-3, None)
        scale = rng.uniform(config.load_scale_lo, config.load_scale_hi)
        load = base[:, None] * shape[None, :] * scale
        scenarios.append(Scenario(gen_cost=gen_cost, load=load))
    return scenarios


def build_zap_devices(config: UtilityConfig, scenario: Scenario) -> tuple[PowerNetwork, list]:
    """Construct the zap network + device list for one scenario (true line ratings)."""
    n = config.n_zones
    hours = config.hours
    net = PowerNetwork(n)

    generators = Generator(
        name=np.array([f"gen{i}" for i in range(n)]),
        num_nodes=n,
        terminal=np.arange(n),
        dynamic_capacity=np.ones((n, hours)),
        nominal_capacity=np.asarray(config.gen_caps, float),
        linear_cost=scenario.gen_cost,
        capital_cost=np.ones(n),
        emission_rates=np.full(n, 0.4),
    )
    loads = Load(
        name=np.array([f"load{i}" for i in range(n)]),
        num_nodes=n,
        terminal=np.arange(n),
        load=scenario.load,
        linear_cost=np.full(n, config.voll),
    )
    n_lines = n - 1
    lines = ACLine(
        name=np.array(list(_LINE_NAMES[:n_lines])),
        num_nodes=n,
        source_terminal=np.arange(n_lines),
        sink_terminal=np.arange(1, n),
        susceptance=np.full(n_lines, 1.0 / config.reactance),
        capacity=np.ones(n_lines),
        nominal_capacity=np.asarray(config.line_caps, float),
        linear_cost=np.zeros(n_lines),
        capital_cost=np.ones(n_lines),
    )
    return net, [generators, loads, lines]


def _dispatch_cost(net: PowerNetwork, devices: list) -> float:
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    return float(out.problem.value)


def _dispatch_cost_and_lmp(net: PowerNetwork, devices: list) -> tuple[float, np.ndarray]:
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    return float(out.problem.value), np.asarray(out.prices, dtype=float)


def _islanded_devices(config: UtilityConfig, devices: list) -> list:
    """Copy ``devices`` with inter-zonal lines throttled to ``actual_line_factor``."""
    throttled = copy.deepcopy(devices)
    line = next(d for d in throttled if isinstance(d, ACLine))
    line.nominal_capacity = line.nominal_capacity * config.actual_line_factor
    return throttled


@define(kw_only=True)
class DispatchGap:
    """Per-scenario SCED vs. actual cost and the annualized avoided-fuel distribution."""

    config: UtilityConfig
    sced_daily: np.ndarray  # (n_scenarios,) $/day, coordinated least-cost
    actual_daily: np.ndarray  # (n_scenarios,) $/day, uncoordinated/islanded
    avoided_per_yr: np.ndarray  # (n_scenarios,) $/yr, (actual - sced) * days_per_year
    ci: CIResult

    @property
    def headline_avoided_per_yr(self) -> float:
        """Mean annual avoided fuel from coordinating dispatch (SCED vs. actual)."""
        return float(self.avoided_per_yr.mean())


def run_dispatch_gap(config: UtilityConfig) -> DispatchGap:
    """Solve SCED vs. islanded actual for every seeded scenario; annualize the gap."""
    scenarios = _make_scenarios(config)
    sced = np.empty(len(scenarios))
    actual = np.empty(len(scenarios))
    for j, scenario in enumerate(scenarios):
        net, devices = build_zap_devices(config, scenario)
        sced[j] = _dispatch_cost(net, devices)
        actual[j] = _dispatch_cost(net, _islanded_devices(config, devices))

    avoided = (actual - sced) * config.days_per_year
    ci = bootstrap_ci(avoided, statistic=np.mean, confidence=0.90, seed=0)
    return DispatchGap(
        config=config, sced_daily=sced, actual_daily=actual, avoided_per_yr=avoided, ci=ci
    )


@define(kw_only=True)
class ExpansionProject:
    """One candidate capacity addition and the economics read off real re-solves.

    ``avoided_per_yr`` is the annual fuel saving from the bump (computed from the change
    in SCED objective); ``npv`` discounts it over the horizon and nets the capital cost.
    """

    name: str
    kind: str  # "line" or "generator"
    delta_mw: float
    avoided_per_yr: float
    capital_cost: float
    npv: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "delta_mw": self.delta_mw,
            "avoided_per_yr": self.avoided_per_yr,
            "capital_cost": self.capital_cost,
            "npv": self.npv,
        }


def _npv(annual: float, capital: float, years: int, rate: float) -> float:
    pv = sum(annual / (1.0 + rate) ** t for t in range(1, years + 1))
    return pv - capital


def _bump(devices: list, kind: str, index: int, delta: float) -> list:
    """Copy ``devices`` with one capacity entry increased by ``delta`` MW."""
    bumped = copy.deepcopy(devices)
    cls = ACLine if kind == "line" else Generator
    dev = next(d for d in bumped if isinstance(d, cls))
    cap = np.asarray(dev.nominal_capacity, dtype=float).copy()
    cap[index] += delta
    dev.nominal_capacity = cap
    return bumped


def run_expansion_ranking(config: UtilityConfig) -> list[ExpansionProject]:
    """Rank candidate expansions by 5-year NPV of avoided fuel on the nominal day."""
    scenario = _nominal_scenario(config)
    net, devices = build_zap_devices(config, scenario)
    base_daily = _dispatch_cost(net, devices)
    delta = config.expansion_delta_mw

    candidates = [
        ("expand-z01", "line", 0, config.line_capital_cost),
        ("expand-z12", "line", 1, config.line_capital_cost),
        ("expand-gen0", "generator", 0, config.gen_capital_cost),
    ]
    projects = []
    for name, kind, index, capital in candidates:
        bumped_daily = _dispatch_cost(net, _bump(devices, kind, index, delta))
        avoided_per_yr = (base_daily - bumped_daily) * config.days_per_year
        npv = _npv(avoided_per_yr, capital, config.expansion_years, config.discount_rate)
        projects.append(
            ExpansionProject(
                name=name, kind=kind, delta_mw=delta, avoided_per_yr=avoided_per_yr,
                capital_cost=capital, npv=npv,
            )
        )
    projects.sort(key=lambda p: p.npv, reverse=True)
    return projects


def build_pypsa(config: UtilityConfig, scenario: Scenario) -> pypsa.Network:
    """Equivalent PyPSA network for the nominal scenario (true line ratings)."""
    n = config.n_zones
    snapshots = pd.date_range("2025-01-01", periods=config.hours, freq="h")
    net = pypsa.Network()
    net.set_snapshots(snapshots)
    for bus in _BUS_NAMES[:n]:
        net.add("Bus", bus)
    net.add("Carrier", "ac", co2_emissions=0.0)
    for i in range(n):
        net.add("Generator", f"gen{i}", bus=_BUS_NAMES[i], p_nom=config.gen_caps[i],
                marginal_cost=config.gen_costs[i], carrier="ac")
        net.add("Load", f"load{i}", bus=_BUS_NAMES[i], p_set=scenario.load[i])
    for k in range(n - 1):
        net.add("Line", _LINE_NAMES[k], bus0=_BUS_NAMES[k], bus1=_BUS_NAMES[k + 1],
                s_nom=config.line_caps[k], x=config.reactance)
    return net


def run_pypsa_fidelity(config: UtilityConfig) -> tuple[FidelityBand, float, float]:
    """SCED LMP validation against PyPSA on the nominal day; returns (band, zap, pypsa)."""
    scenario = _nominal_scenario(config)
    net, devices = build_zap_devices(config, scenario)
    zap_cost, zap_lmp = _dispatch_cost_and_lmp(net, devices)

    pn = build_pypsa(config, scenario)
    pn.snapshot_weightings.loc[:, :] = 1.0
    pn.optimize(solver_name="highs")
    pypsa_lmp = pn.buses_t.marginal_price[list(_BUS_NAMES[: config.n_zones])].to_numpy(float).T
    pypsa_cost = float(pn.objective)

    band = fidelity_band(zap_lmp, pypsa_lmp, reference="pypsa-dc", metric="lmp", units="$/MWh")
    return band, zap_cost, pypsa_cost


@define(kw_only=True)
class UtilityResult:
    """The combined backtest: dispatch gap + expansion ranking + PyPSA fidelity."""

    config: UtilityConfig
    gap: DispatchGap
    projects: list  # list[ExpansionProject], NPV-descending
    fidelity: FidelityBand
    zap_sced_cost: float
    pypsa_sced_cost: float
    source: str = field(default="synthetic")

    @property
    def best_project(self) -> ExpansionProject:
        return self.projects[0]

    @property
    def npv_delta(self) -> float:
        """NPV of the top-ranked expansion project (the build-it-next headline)."""
        return self.best_project.npv

    @property
    def pypsa_cost_rel_gap(self) -> float:
        denom = abs(self.pypsa_sced_cost)
        if denom < 1e-9:
            return abs(self.zap_sced_cost - self.pypsa_sced_cost)
        return abs(self.zap_sced_cost - self.pypsa_sced_cost) / denom

    def to_bench_result(self) -> BenchResult:
        gap = self.gap
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=gap.headline_avoided_per_yr,
            units="$/yr",
            ci=gap.ci,
            fidelity_band=self.fidelity,
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "pypsa_solver": "highs",
                "topology": "3-zone radial path (zone0-zone1-zone2)",
                "fleet": "one generator per zone, cheap->expensive outward",
                "binding_corridor": "zone1-zone2 (z12), thermal-limited below peak transfer",
                "actual_dispatch": (
                    "uncoordinated self-scheduling: inter-zonal lines throttled to "
                    f"{self.config.actual_line_factor:.0%} of rating (islanded), a feasible "
                    "but non-optimal point of the SCED problem"
                ),
                "headline": (
                    "mean annual avoided fuel = (actual islanded cost - SCED cost) * "
                    "days_per_year, paired per seeded scenario"
                ),
                "expansion_ranking": (
                    "candidates ranked by 5-year NPV of avoided fuel from a +"
                    f"{self.config.expansion_delta_mw:.0f} MW bump, discount "
                    f"{self.config.discount_rate:.0%}"
                ),
                "n_scenarios": self.config.n_scenarios,
                "hours": self.config.hours,
                "days_per_year": self.config.days_per_year,
                "lmp_gap_tol": LMP_GAP_TOL,
                "synthetic_note": (
                    "generator fuel costs carry seeded per-hour noise and loads a "
                    "per-scenario scale; a human stages a real fleet/load/topology and "
                    "re-runs with --real"
                ),
            },
            sensitivities={
                "npv_delta_best_project_usd": self.npv_delta,
                "best_project": self.best_project.name,
                "expansion_ranking": [p.to_dict() for p in self.projects],
                "sced_daily_cost": gap.sced_daily.tolist(),
                "actual_daily_cost": gap.actual_daily.tolist(),
                "avoided_per_yr": gap.avoided_per_yr.tolist(),
                "mean_sced_daily": float(gap.sced_daily.mean()),
                "mean_actual_daily": float(gap.actual_daily.mean()),
                "zap_sced_cost": self.zap_sced_cost,
                "pypsa_sced_cost": self.pypsa_sced_cost,
                "pypsa_cost_rel_gap": self.pypsa_cost_rel_gap,
            },
        )


def run_utility(config: Optional[UtilityConfig] = None) -> UtilityResult:
    """Run the dispatch gap, expansion ranking, and PyPSA fidelity on synthetic data."""
    config = config or UtilityConfig()
    gap = run_dispatch_gap(config)
    projects = run_expansion_ranking(config)
    fidelity, zap_cost, pypsa_cost = run_pypsa_fidelity(config)
    return UtilityResult(
        config=config, gap=gap, projects=projects, fidelity=fidelity,
        zap_sced_cost=zap_cost, pypsa_sced_cost=pypsa_cost,
    )


def load_staged_utility(name: str) -> None:
    """Human ``--real`` entry point: read a staged fleet/load/topology.

    Raises :class:`DataNotStagedError` when ``data/<name>/`` is empty (the loop path),
    never downloading. A human stages a real utility model and wires the loader.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged utility data for {name!r}: expected a fleet/load/topology under "
            f"{cache_dir}. A human must stage real data there (see data/README.md); the "
            f"benchmark loop never downloads."
        )
    raise NotImplementedError(
        "real staged-data utility backtest is wired up by a human; the synthetic path is "
        "the loop-runnable one."
    )


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic utility backtest and emit (optionally write) a ``BenchResult``."""
    result = run_utility().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steinmetz vertically-integrated utility backtest (§7.2)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on the synthetic 3-zone utility (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="run against a staged fleet/load/topology in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    if args.real:
        load_staged_utility(args.real)

    res = run_utility()
    g = res.gap
    print(f"mean SCED cost   : {g.sced_daily.mean():,.0f} $/day")
    print(f"mean actual cost : {g.actual_daily.mean():,.0f} $/day (islanded)")
    print(f"avoided fuel     : {g.headline_avoided_per_yr:,.0f} $/yr "
          f"[{g.ci.lo:,.0f}, {g.ci.hi:,.0f}] (90% CI)")
    print(f"PyPSA LMP gap    : {res.fidelity.max_abs_gap:.3e} $/MWh (tol {LMP_GAP_TOL})")
    print(f"\n{'project':<14}{'avoided $/yr':>16}{'NPV $':>16}")
    for p in res.projects:
        print(f"{p.name:<14}{p.avoided_per_yr:>16,.0f}{p.npv:>16,.0f}")
    print(f"\nbuild next: {res.best_project.name} (NPV-delta {res.npv_delta:,.0f} $)")
