"""Transmission-plan audit backtest (roadmap item 3.4, Steinmetz §7.3).

A transmission planner has to decide *which corridors to expand* before knowing
exactly how the year plays out. zap gives that decision a single ex-ante signal: the
marginal value of corridor capacity, ``-d(system cost)/d(line capacity)``, read straight
off the dispatch adjoint. This backtest audits that signal the way a regulator would —
by checking it against what *actually* congested in realized operation.

**The synthetic world.** A cheap generation hub (node 0) feeds ``K`` load zones over
``K`` radial corridors (spokes). Each zone also owns an expensive local backup unit large
enough to self-supply, so nothing is ever shed — congestion only shifts energy from the
cheap hub to a pricey backup. The spokes are deliberately sized so the first few are
comfortably over-provisioned (they carry the whole zone load) while the rest are
under-provisioned and bind, and the zones' backup costs increase down the line. The most
under-provisioned, most-expensive-backup corridor is the *known bottleneck*: relieving one
MW there displaces the dearest backup, so it has the highest marginal value by
construction. A radial topology is used on purpose — each spoke's flow is fixed by its
zone's power balance, so capacity is the only thing that binds and the marginal-value
signal is a clean thermal-relief reading, not a loop-flow artifact.

**Ex-ante ranking.** On the noise-free *forecast* day we solve the DC-OPF once and take
zap's adjoint gradient of system cost w.r.t. every spoke's ``nominal_capacity``. The
negated gradient is each corridor's marginal value; ranking corridors by it is the
planner's ex-ante recommendation. Every gradient is cross-checked against a central finite
difference (re-solving at ``cap ± eps``); the adjoint-vs-FD agreement is the
:class:`~...metrics.FidelityBand` attached to the result, so the ranking signal is
certified, not asserted.

**Realized congestion.** Independently — different code path, different data — we draw
seeded *realized* operating days (backup-cost noise + per-day load scaling) and, from each
solved dispatch, compute every corridor's **congestion rent** the way an ISO publishes it:
``|LMP_sink - LMP_source| * flow`` summed over the day, using only nodal prices and line
flows (no gradients). Averaged over the realized days this is the realized-congestion
vector the ex-ante ranking is audited against.

**The audit.** We report (a) that the known bottleneck is ranked #1 ex-ante, (b) the
Spearman rank-correlation between the ex-ante marginal-value vector and the realized
congestion-rent vector, (c) the R² of their linear fit (the headline ``BenchResult``
number), and (d) the count of "missed" corridors — high ex-ante value but low realized
congestion — which a good signal drives to zero. Forecast-vs-realized noise keeps the
correlation below a perfect 1.0, which is the whole point of auditing.

Every number is computed from an actual zap solve. The ``--real`` path is reserved for a
human who stages a real RTEP/MTEP vintage + approved-project list into ``data/<name>/``; it
blocks via :class:`DataNotStagedError` rather than downloading, matching the rest of suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
from attrs import define, field

from zap.devices import ACLine, Generator, Load
from zap.network import PowerNetwork

from experiments.steinmetz_bench.datasets.registry import DATA_ROOT, DataNotStagedError
from experiments.steinmetz_bench.experiments.grad_check import check_parameter
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import CIResult, FidelityBand, fidelity_band

EXPERIMENT_ID = "3.4-transmission-audit"
DATASET = "synthetic-radial-corridors"

# Acceptance: the ex-ante marginal-value ranking must agree this strongly (Spearman) with
# the realized congestion-rent ranking. Comfortably exceeded on the synthetic world; the
# forecast-vs-realized noise is what keeps it under 1.0.
RANK_CORR_TOL = 0.6
# A corridor counts as "high value" / "highly congested" once it clears this fraction of
# the respective vector's maximum; used for the missed-corridor audit.
HIGH_VALUE_FRAC = 0.5


@define(kw_only=True)
class AuditConfig:
    """Knobs of the synthetic hub-and-spokes transmission world (one representative day).

    Node 0 is the cheap hub; nodes ``1..K`` are load zones reached by one radial spoke
    each. ``spoke_caps`` is ascending-to-descending on purpose: the early zones are
    over-provisioned (uncongested) and the late ones bind, while ``backup_costs`` rises so
    the last, most-throttled corridor displaces the dearest backup and tops the ranking.
    All randomness is seeded.
    """

    hours: int = 24
    n_scenarios: int = 24

    # Cheap hub generator (node 0). A small quadratic term makes the hub price move
    # smoothly with total import so the marginal-value curve is differentiable.
    hub_cost: float = 10.0
    hub_quadratic_cost: float = 0.005
    hub_cap: float = 600.0

    # Per-zone expensive backup units (nodes 1..K), ascending cost. Each must cover its
    # zone's peak load so the islanded fallback is feasible (no load shed, only fuel cost).
    backup_costs: tuple[float, ...] = (40.0, 55.0, 70.0, 90.0, 120.0, 160.0)
    backup_cap: float = 120.0

    # Per-zone base load (MW), before the daily shape and per-scenario scaling.
    base_load: float = 80.0
    load_amp: float = 0.20  # daily-shape amplitude as a fraction of the base load
    voll: float = 10_000.0  # value of lost load (never binds; loads are always served)

    # Radial spoke ratings (MW), one per zone. Caps above base_load stay uncongested;
    # caps below it bind. Descending here so congestion severity grows down the line.
    spoke_caps: tuple[float, ...] = (130.0, 110.0, 70.0, 60.0, 50.0, 40.0)
    reactance: float = 0.1

    # Finite-difference step for the adjoint cross-check (MW); small enough to stay inside
    # the binding regime of every congested spoke.
    fd_eps: float = 1e-2

    # Per-scenario seeded perturbations (the forecast-vs-realized gap).
    cost_noise: float = 0.08
    load_scale_lo: float = 0.85
    load_scale_hi: float = 1.25
    seed: int = 0

    def __attrs_post_init__(self):
        k = len(self.backup_costs)
        if len(self.spoke_caps) != k:
            raise ValueError("backup_costs and spoke_caps must share a length (one per zone)")
        if k < 3:
            raise ValueError("need at least three corridors for a meaningful ranking")
        peak = self.base_load * (1.0 + self.load_amp) * self.load_scale_hi
        if self.backup_cap < peak:
            raise ValueError("each zone backup must cover its own peak load (feasible islanding)")
        if not any(c < self.base_load for c in self.spoke_caps):
            raise ValueError("at least one spoke must be undersized so a corridor binds")

    @property
    def n_zones(self) -> int:
        return len(self.backup_costs)

    @property
    def n_nodes(self) -> int:
        return self.n_zones + 1

    @property
    def bottleneck_zone(self) -> int:
        """Zone (0-based) of the known bottleneck: the highest marginal-value corridor.

        By construction this is the most-throttled, most-expensive-backup spoke — the one
        with the smallest ``spoke_cap`` among those whose backup is dearest. We pick it as
        the corridor maximizing ``backup_cost`` over the *binding* (cap < base_load) set.
        """
        binding = [i for i, c in enumerate(self.spoke_caps) if c < self.base_load]
        return max(binding, key=lambda i: self.backup_costs[i])

    @property
    def corridor_names(self) -> list[str]:
        return [f"hub-z{i}" for i in range(self.n_zones)]


@define(kw_only=True)
class Scenario:
    """One seeded operating day: per-(gen, hour) linear cost and per-(zone, hour) load."""

    gen_cost: np.ndarray  # (n_zones + 1, hours): row 0 = hub, rows 1.. = backups
    load: np.ndarray  # (n_zones, hours)


def _daily_shape(config: AuditConfig) -> np.ndarray:
    """Per-hour multiplier with a midday peak, mean ~1.0."""
    t = np.arange(config.hours)
    return 1.0 + config.load_amp * np.sin(2 * np.pi * (t - 9) / 24.0)


def _nominal_scenario(config: AuditConfig) -> Scenario:
    """The noise-free forecast day used for the ex-ante marginal-value ranking."""
    shape = _daily_shape(config)
    costs = np.concatenate([[config.hub_cost], np.asarray(config.backup_costs, float)])
    gen_cost = costs[:, None] * np.ones((1, config.hours))
    load = np.full(config.n_zones, config.base_load)[:, None] * shape[None, :]
    return Scenario(gen_cost=gen_cost, load=load)


def _make_scenarios(config: AuditConfig) -> list[Scenario]:
    """Draw the seeded realized days for the congestion-rent distribution."""
    rng = np.random.default_rng(config.seed)
    shape = _daily_shape(config)
    backups = np.asarray(config.backup_costs, float)

    scenarios = []
    for _ in range(config.n_scenarios):
        noise = rng.normal(0.0, config.cost_noise, size=(config.n_zones, config.hours))
        backup_cost = np.clip(backups[:, None] * (1.0 + noise), 1e-3, None)
        gen_cost = np.concatenate(
            [config.hub_cost * np.ones((1, config.hours)), backup_cost], axis=0
        )
        scale = rng.uniform(config.load_scale_lo, config.load_scale_hi)
        load = np.full(config.n_zones, config.base_load)[:, None] * shape[None, :] * scale
        scenarios.append(Scenario(gen_cost=gen_cost, load=load))
    return scenarios


def build_zap_devices(config: AuditConfig, scenario: Scenario) -> tuple[PowerNetwork, list]:
    """Construct the zap network + device list (hub gen, zone backups, loads, spokes)."""
    k = config.n_zones
    n = config.n_nodes
    hours = config.hours
    net = PowerNetwork(n)

    n_gen = k + 1  # hub at node 0, one backup per zone at nodes 1..k
    gen_caps = np.concatenate([[config.hub_cap], np.full(k, config.backup_cap)])
    quad = np.zeros((n_gen, 1))
    quad[0, 0] = config.hub_quadratic_cost  # smooth hub price; backups stay pure-linear
    generators = Generator(
        name=np.array(["hub"] + [f"backup{i}" for i in range(k)]),
        num_nodes=n,
        terminal=np.arange(n_gen),
        dynamic_capacity=np.ones((n_gen, hours)),
        nominal_capacity=gen_caps,
        linear_cost=scenario.gen_cost,
        quadratic_cost=quad,
        capital_cost=np.ones(n_gen),
        emission_rates=np.full(n_gen, 0.4),
    )
    loads = Load(
        name=np.array([f"load{i}" for i in range(k)]),
        num_nodes=n,
        terminal=np.arange(1, n),
        load=scenario.load,
        linear_cost=np.full(k, config.voll),
    )
    lines = ACLine(
        name=np.array(config.corridor_names),
        num_nodes=n,
        source_terminal=np.zeros(k, dtype=int),
        sink_terminal=np.arange(1, n),
        susceptance=np.full(k, 1.0 / config.reactance),
        capacity=np.ones(k),
        nominal_capacity=np.asarray(config.spoke_caps, float),
        linear_cost=np.zeros(k),
        capital_cost=np.ones(k),
    )
    return net, [generators, loads, lines]


def _line_index(devices: list) -> int:
    return next(i for i, d in enumerate(devices) if isinstance(d, ACLine))


def _dispatch(net: PowerNetwork, devices: list):
    horizon = max(d.time_horizon for d in devices)
    out = net.dispatch(devices, time_horizon=horizon, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    return out


@define(kw_only=True)
class ExAnteRanking:
    """Ex-ante corridor marginal value from zap's adjoint, with the FD cross-check."""

    config: AuditConfig
    marginal_value: np.ndarray  # (K,) $/MW-day, -d(system cost)/d(spoke capacity)
    fd_marginal_value: np.ndarray  # (K,) finite-difference, NaN where not evaluated
    active_mask: np.ndarray  # (K,) which corridors bind (gradient is meaningful)
    max_rel_err_fd: float

    @property
    def ranking(self) -> list[int]:
        """Corridor indices, highest marginal value first."""
        return list(np.argsort(self.marginal_value)[::-1])

    @property
    def top_corridor(self) -> int:
        return self.ranking[0]


def run_ex_ante(config: AuditConfig) -> ExAnteRanking:
    """Rank corridors by adjoint marginal value of capacity on the forecast day."""
    scenario = _nominal_scenario(config)
    net, devices = build_zap_devices(config, scenario)
    li = _line_index(devices)

    check = check_parameter(
        net, devices, li, "nominal_capacity", "line", solver=cp.CLARABEL, do_fd=True
    )
    # Adjoint is d(cost)/d(cap); marginal value is the cost *reduction* per MW added.
    marginal_value = -np.asarray(check.adjoint, dtype=float).ravel()
    fd_mv = -np.asarray(check.finite_difference, dtype=float).ravel()
    active = np.asarray(check.active_mask, dtype=bool).ravel()

    evaluated = np.isfinite(fd_mv)
    if evaluated.any():
        rel = np.abs(marginal_value[evaluated] - fd_mv[evaluated]) / np.maximum(
            np.abs(fd_mv[evaluated]), 1e-9
        )
        max_rel_err_fd = float(rel.max())
    else:
        max_rel_err_fd = float("nan")

    return ExAnteRanking(
        config=config,
        marginal_value=marginal_value,
        fd_marginal_value=fd_mv,
        active_mask=active,
        max_rel_err_fd=max_rel_err_fd,
    )


def _congestion_rent(config: AuditConfig, out, devices: list) -> np.ndarray:
    """Per-corridor congestion rent ``sum_t |LMP_sink - LMP_source| * flow`` for one day.

    Uses only the solved nodal prices and line flows — the observable an ISO publishes,
    independent of any gradient. Returns a ``(K,)`` vector in $/day.
    """
    li = _line_index(devices)
    line = devices[li]
    prices = np.asarray(out.prices, dtype=float)  # (n_nodes, hours)
    flow = np.asarray(out.power[li][1], dtype=float)  # sink-terminal power, (K, hours)

    source = np.asarray(line.source_terminal, dtype=int).ravel()
    sink = np.asarray(line.sink_terminal, dtype=int).ravel()
    spread = prices[sink] - prices[source]  # (K, hours)
    return np.sum(np.abs(spread * flow), axis=1)


@define(kw_only=True)
class RealizedCongestion:
    """Per-corridor realized congestion rent over the seeded realized days."""

    config: AuditConfig
    rent_per_scenario: np.ndarray  # (n_scenarios, K) $/day
    mean_rent: np.ndarray  # (K,) $/day, averaged over realized days


def run_realized(config: AuditConfig) -> RealizedCongestion:
    """Solve every realized day and accumulate per-corridor congestion rent."""
    scenarios = _make_scenarios(config)
    rent = np.empty((len(scenarios), config.n_zones))
    for j, scenario in enumerate(scenarios):
        net, devices = build_zap_devices(config, scenario)
        out = _dispatch(net, devices)
        rent[j] = _congestion_rent(config, out, devices)
    return RealizedCongestion(
        config=config, rent_per_scenario=rent, mean_rent=rent.mean(axis=0)
    )


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1-based), matching ``scipy.stats.rankdata`` without the dep."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # average of 1-based positions i+1..j+1
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank-correlation = Pearson correlation of the average-tie ranks."""
    return _pearson(_rankdata(a), _rankdata(b))


def r_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Coefficient of determination of the least-squares line ``b ~ a`` (= Pearson²)."""
    r = _pearson(a, b)
    return float(r * r)


@define(kw_only=True)
class AuditResult:
    """The full audit: ex-ante ranking vs realized congestion + the agreement metrics."""

    config: AuditConfig
    ex_ante: ExAnteRanking
    realized: RealizedCongestion
    rank_corr: float
    r2: float
    ci: CIResult
    missed_corridor_count: int
    source: str = field(default="synthetic")

    @property
    def top_corridor(self) -> int:
        return self.ex_ante.top_corridor

    @property
    def bottleneck_identified(self) -> bool:
        return self.top_corridor == self.config.bottleneck_zone

    def fidelity(self) -> FidelityBand:
        """Adjoint-vs-finite-difference agreement on the marginal-value gradient."""
        mv = self.ex_ante.marginal_value
        fd = self.ex_ante.fd_marginal_value
        evaluated = np.isfinite(fd)
        return fidelity_band(
            mv[evaluated], fd[evaluated],
            reference="finite-difference", metric="corridor-marginal-value",
            units="$/MW-day",
        )

    def to_bench_result(self) -> BenchResult:
        cfg = self.config
        ex = self.ex_ante
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.r2,
            units="R2",
            ci=self.ci,
            fidelity_band=self.fidelity(),
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "topology": (
                    f"radial hub-and-spokes: 1 cheap hub + {cfg.n_zones} load zones, "
                    "one corridor each"
                ),
                "ex_ante_signal": (
                    "corridor marginal value = -d(system cost)/d(spoke nominal_capacity) "
                    "from zap's dispatch adjoint on the noise-free forecast day"
                ),
                "realized_signal": (
                    "congestion rent = sum_t |LMP_sink - LMP_source| * flow per corridor, "
                    "averaged over seeded realized days (nodal prices + flows only, no "
                    "gradients)"
                ),
                "bottleneck_corridor": cfg.corridor_names[cfg.bottleneck_zone],
                "bottleneck_definition": (
                    "most-throttled spoke carrying the dearest backup; relieving it "
                    "displaces the most expensive energy, so it has the highest "
                    "marginal value by construction"
                ),
                "headline": (
                    "R^2 of the linear fit between the ex-ante marginal-value vector and "
                    "the realized congestion-rent vector across corridors"
                ),
                "rank_corr_tol": RANK_CORR_TOL,
                "high_value_frac": HIGH_VALUE_FRAC,
                "n_scenarios": cfg.n_scenarios,
                "hours": cfg.hours,
                "fd_eps_mw": cfg.fd_eps,
                "synthetic_note": (
                    "backup fuel costs carry seeded per-hour noise and loads a per-day "
                    "scale (the forecast-vs-realized gap); a human stages a real "
                    "RTEP/MTEP vintage + approved-project list and re-runs with --real"
                ),
            },
            sensitivities={
                "rank_correlation_spearman": self.rank_corr,
                "r_squared": self.r2,
                "missed_corridor_count": int(self.missed_corridor_count),
                "bottleneck_zone": int(cfg.bottleneck_zone),
                "top_corridor": int(self.top_corridor),
                "bottleneck_identified": bool(self.bottleneck_identified),
                "ex_ante_ranking": [cfg.corridor_names[i] for i in ex.ranking],
                "corridor_names": cfg.corridor_names,
                "marginal_value_per_mw_day": ex.marginal_value.tolist(),
                "fd_marginal_value_per_mw_day": [
                    float(v) if np.isfinite(v) else None for v in ex.fd_marginal_value
                ],
                "active_mask": ex.active_mask.astype(int).tolist(),
                "gradient_max_rel_err_fd": ex.max_rel_err_fd,
                "realized_congestion_rent_per_day": self.realized.mean_rent.tolist(),
            },
        )


def _missed_corridor_count(marginal_value: np.ndarray, realized_rent: np.ndarray) -> int:
    """Corridors flagged high-value ex-ante but not highly congested in realization."""
    mv = np.asarray(marginal_value, float).ravel()
    rent = np.asarray(realized_rent, float).ravel()
    mv_max = mv.max()
    rent_max = rent.max()
    if mv_max <= 0.0:
        return 0
    high_value = mv >= HIGH_VALUE_FRAC * mv_max
    high_rent = rent >= HIGH_VALUE_FRAC * rent_max if rent_max > 0.0 else np.zeros_like(rent, bool)
    return int(np.sum(high_value & ~high_rent))


def _bootstrap_r2_ci(
    marginal_value: np.ndarray,
    rent_per_scenario: np.ndarray,
    confidence: float = 0.90,
    n_boot: int = 2000,
    seed: int = 0,
) -> CIResult:
    """Percentile CI on the headline R^2 by resampling realized days.

    Each resample re-averages the per-day congestion rent and recomputes R^2 against the
    fixed ex-ante marginal-value vector, so the interval reflects realized-day sampling.
    """
    rent = np.asarray(rent_per_scenario, float)
    n = rent.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = r_squared(marginal_value, rent[idx].mean(axis=0))
    alpha = (1.0 - confidence) / 2.0
    return CIResult(
        lo=float(np.percentile(boot, 100.0 * alpha)),
        mid=float(np.percentile(boot, 50.0)),
        hi=float(np.percentile(boot, 100.0 * (1.0 - alpha))),
        confidence=confidence,
    )


def run_audit(config: Optional[AuditConfig] = None) -> AuditResult:
    """Run the ex-ante ranking + realized congestion and compute the audit metrics."""
    config = config or AuditConfig()
    ex_ante = run_ex_ante(config)
    realized = run_realized(config)

    mv = ex_ante.marginal_value
    rent = realized.mean_rent
    rank_corr = spearman(mv, rent)
    r2 = r_squared(mv, rent)
    ci = _bootstrap_r2_ci(mv, realized.rent_per_scenario)
    missed = _missed_corridor_count(mv, rent)

    return AuditResult(
        config=config, ex_ante=ex_ante, realized=realized,
        rank_corr=rank_corr, r2=r2, ci=ci, missed_corridor_count=missed,
    )


def load_staged_audit(name: str) -> None:
    """Human ``--real`` entry point: read a staged RTEP/MTEP vintage + project list.

    Raises :class:`DataNotStagedError` when ``data/<name>/`` is empty (the loop path),
    never downloading. A human stages a real transmission plan and wires the loader.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged transmission-plan data for {name!r}: expected an RTEP/MTEP vintage "
            f"+ approved-project list under {cache_dir}. A human must stage real data there "
            f"(see data/README.md); the benchmark loop never downloads."
        )
    raise NotImplementedError(
        "real staged-data transmission audit is wired up by a human; the synthetic path is "
        "the loop-runnable one."
    )


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic transmission-audit backtest and emit a ``BenchResult``."""
    result = run_audit().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steinmetz transmission-plan audit backtest (§7.3)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on the synthetic radial-corridor world (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="audit a staged RTEP/MTEP vintage in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    if args.real:
        load_staged_audit(args.real)

    res = run_audit()
    cfg = res.config
    print(f"corridors            : {cfg.n_zones} (radial hub-and-spokes)")
    print(f"known bottleneck     : {cfg.corridor_names[cfg.bottleneck_zone]}")
    print(f"top ex-ante corridor : {cfg.corridor_names[res.top_corridor]} "
          f"({'MATCH' if res.bottleneck_identified else 'MISS'})")
    print(f"rank correlation     : {res.rank_corr:.3f} (tol {RANK_CORR_TOL})")
    print(f"R^2                  : {res.r2:.3f} "
          f"[{res.ci.lo:.3f}, {res.ci.hi:.3f}] (90% CI)")
    print(f"missed corridors     : {res.missed_corridor_count}")
    print(f"adjoint-vs-FD max err: {res.ex_ante.max_rel_err_fd:.2e}")
    print(f"\n{'corridor':<10}{'marg.value $/MW-d':>20}{'realized rent $/d':>20}")
    for i in range(cfg.n_zones):
        print(f"{cfg.corridor_names[i]:<10}{res.ex_ante.marginal_value[i]:>20,.1f}"
              f"{res.realized.mean_rent[i]:>20,.1f}")
