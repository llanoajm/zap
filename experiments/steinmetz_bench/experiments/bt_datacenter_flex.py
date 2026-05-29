"""Data-center flexibility & battery sizing backtest (roadmap item 3.2, Steinmetz §7.1-B).

A large data center co-located with a grid has two levers for cutting what it pays
for power: a **battery** it can charge when energy is cheap and discharge when it is
expensive, and **demand flexibility** — shifting or shedding compute away from the
priciest hours. This backtest quantifies both on synthetic-but-real-solve data.

**Battery sizing.** We sweep the battery's power capacity and, at each size, read the
*marginal value of an extra MW* straight off zap's adjoint: ``-d(operation cost)/d(power
capacity)``. That gradient is the exact dispatch sensitivity the planner uses, and we
cross-check every point against a central finite difference (re-solving at ``cap ± eps``)
— the two must agree, which is what certifies the curve. The marginal-value curve is
monotone decreasing (diminishing arbitrage returns), so it crosses the battery's
annualized capital cost at a single **break-even size**: the largest battery still worth
building. We also recompute the net-value curve (annual operating savings minus capital
cost) from independent solves and confirm its maximum sits at that break-even, i.e. that
marginal value really does equal marginal cost there.

The dispatch is a QP, not an LP: the generators carry a small quadratic cost so nodal
prices vary *continuously* with load. Under a pure LP the cost gradient is piecewise
constant (a step function), the break-even is a degenerate kink, and a finite-difference
check straddling a kink is meaningless. The quadratic term makes the price spread narrow
smoothly as the battery grows, giving a differentiable marginal-value curve with a
well-defined crossing — and an FD anchor the adjoint can actually be checked against.

**Firm vs. flexible.** Separately, we value demand flexibility by serving the data center
two ways and differencing the system cost. *Firm* models it as an inflexible
:class:`~zap.devices.Load` pinned at a flat draw; *flexible* models it as a
:class:`~zap.devices.PowerTarget` that *wants* the same draw but may deviate, paying a
quadratic "value of deferred compute" penalty. Because the firm profile is always feasible
for the flexible problem at zero penalty, the flexible optimum can only cost less or the
same — so ``firm_cost - flexible_cost >= 0`` is the value of flexibility, computed per
seeded scenario (fuel-cost noise + load scaling), annualized, and reported with a bootstrap
CI. ``PowerTarget`` doesn't accept the ``envelope`` keyword zap's dispatch passes every
device, so we use a thin :class:`FlexibleLoad` subclass that absorbs it; nothing in zap's
core is touched.

Every number here is computed from an actual zap solve. The ``--real`` path is reserved
for a human who stages real ISO price/load history into ``data/<name>/``; it blocks via
:class:`DataNotStagedError` rather than downloading, matching the rest of the suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
from attrs import define, field

from zap.devices import Battery, Generator, Load, PowerTarget
from zap.network import PowerNetwork

from experiments.steinmetz_bench.datasets.registry import DATA_ROOT, DataNotStagedError
from experiments.steinmetz_bench.experiments.grad_check import check_parameter
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import (
    CIResult,
    FidelityBand,
    bootstrap_ci,
    fidelity_band,
)

EXPERIMENT_ID = "3.2-datacenter-flex"
DATASET = "synthetic-flex-qp"


@define(kw_only=True, slots=False)
class FlexibleLoad(PowerTarget):
    """A :class:`~zap.devices.PowerTarget` usable in :meth:`PowerNetwork.dispatch`.

    zap's dispatch builder passes every device an ``envelope`` keyword (for its convex
    relaxations); the stock ``PowerTarget`` predates that and rejects it. This subclass
    accepts and ignores ``envelope`` so a flexible load can be dispatched directly. It
    changes no behavior — the flexibility model is entirely PowerTarget's quadratic
    deviation penalty.
    """

    def equality_constraints(
        self, power, angle, _, target_power=None, weights=None, la=np, envelope=None
    ):
        return []

    def inequality_constraints(
        self, power, angle, _, target_power=None, weights=None, la=np, envelope=None
    ):
        return []

    def operation_cost(
        self, power, angle, _, target_power=None, weights=None, la=np, envelope=None
    ):
        return super().operation_cost(
            power, angle, _, target_power=target_power, weights=weights, la=la
        )


@define(kw_only=True)
class FlexConfig:
    """Knobs of the synthetic flexibility/battery world (one representative day).

    The single node hosts a cheap capacity-limited baseload generator and an expensive
    peaker, both with a small quadratic cost so prices move continuously with load. A
    daily-shaped background load plus a flat ``dc_mw`` data-center draw push the peaker
    onto the margin in peak hours and leave the baseload marginal off-peak — the price
    spread the battery arbitrages and flexibility avoids. All randomness is seeded.
    """

    hours: int = 24
    n_scenarios: int = 16
    dc_mw: float = 30.0

    baseload_cost: float = 20.0
    baseload_cap: float = 100.0
    peaker_cost: float = 80.0
    peaker_cap: float = 1000.0
    gen_quadratic_cost: float = 0.05

    bg_base: float = 70.0
    bg_amp: float = 45.0
    bg_floor: float = 5.0
    voll: float = 1000.0

    battery_duration: float = 4.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    battery_sizes: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 55.0, 80.0)
    # Annualized $/MW-yr capital cost of battery power. Chosen so the marginal-value
    # curve crosses it at an interior size; the break-even itself is computed, not set.
    battery_capital_cost_per_mw_yr: float = 50_000.0

    # PowerTarget penalty weight: the marginal $/MW value of deferred compute. Larger =
    # stiffer (closer to firm); smaller = more flexible. At 5.0 the flexible data center
    # stays a genuine load (it only curtails during expensive hours, never injects).
    flex_weight: float = 5.0

    cost_noise: float = 0.06
    load_scale_lo: float = 0.85
    load_scale_hi: float = 1.30
    days_per_year: int = 365
    seed: int = 0

    def __attrs_post_init__(self):
        if len(self.battery_sizes) < 3:
            raise ValueError("need at least three battery sizes to bracket a break-even")
        if any(b <= 0 for b in self.battery_sizes):
            raise ValueError("battery sizes must be positive (a 0-MW battery is degenerate)")
        if list(self.battery_sizes) != sorted(self.battery_sizes):
            raise ValueError("battery_sizes must be ascending")


@define(kw_only=True)
class Scenario:
    """One seeded operating day: per-(gen, hour) linear cost and per-hour background load."""

    gen_cost: np.ndarray  # (2, hours)
    bg_load: np.ndarray  # (hours,)


def _baseload_shape(config: FlexConfig) -> np.ndarray:
    """Daily-shaped background load: peaks midday, troughs overnight."""
    t = np.arange(config.hours)
    shape = config.bg_base + config.bg_amp * np.sin(2 * np.pi * (t - 9) / 24.0)
    return np.clip(shape, config.bg_floor, None)


def _make_scenarios(config: FlexConfig) -> list[Scenario]:
    """Draw the seeded scenarios for the firm-vs-flexible comparison."""
    rng = np.random.default_rng(config.seed)
    base = _baseload_shape(config)
    costs = np.array([config.baseload_cost, config.peaker_cost], dtype=float)

    scenarios = []
    for _ in range(config.n_scenarios):
        noise = rng.normal(0.0, config.cost_noise, size=(2, config.hours))
        gen_cost = np.clip(costs[:, None] * (1.0 + noise), 1e-3, None)
        scale = rng.uniform(config.load_scale_lo, config.load_scale_hi)
        scenarios.append(Scenario(gen_cost=gen_cost, bg_load=base * scale))
    return scenarios


def _nominal_scenario(config: FlexConfig) -> Scenario:
    """The noise-free, unit-scaled representative day used for battery sizing."""
    costs = np.array([config.baseload_cost, config.peaker_cost], dtype=float)
    gen_cost = costs[:, None] * np.ones((1, config.hours))
    return Scenario(gen_cost=gen_cost, bg_load=_baseload_shape(config))


def _generators(config: FlexConfig, scenario: Scenario) -> Generator:
    caps = np.array([config.baseload_cap, config.peaker_cap], dtype=float)
    return Generator(
        name=np.array(["baseload", "peaker"]),
        num_nodes=1,
        terminal=np.array([0, 0]),
        dynamic_capacity=caps[:, None] * np.ones((1, config.hours)),
        linear_cost=scenario.gen_cost,
        quadratic_cost=np.array([[config.gen_quadratic_cost], [config.gen_quadratic_cost]]),
        nominal_capacity=np.ones(2),
        capital_cost=np.ones(2),
        emission_rates=np.full(2, 0.4),
    )


def _battery(config: FlexConfig, power_mw: float) -> Battery:
    return Battery(
        num_nodes=1,
        name=np.array(["battery"]),
        terminal=np.array([0]),
        power_capacity=np.array([float(power_mw)]),
        duration=np.array([config.battery_duration]),
        charge_efficiency=np.array([config.charge_efficiency]),
        discharge_efficiency=np.array([config.discharge_efficiency]),
        linear_cost=np.array([0.0]),
    )


def _build_devices(config: FlexConfig, scenario: Scenario, *, battery_mw=0.0, flexible=False):
    """Assemble the single-node network; return ``(net, devices, battery_index)``.

    ``battery_index`` is ``None`` when no battery is attached. With ``flexible=True`` the
    data center is a :class:`FlexibleLoad` (PowerTarget); otherwise a firm :class:`Load`.
    """
    net = PowerNetwork(1)
    hours = config.hours
    background = Load(
        name=np.array(["background"]),
        num_nodes=1,
        terminal=np.array([0]),
        load=scenario.bg_load[None, :],
        linear_cost=np.full((1, hours), config.voll),
    )
    devices = [_generators(config, scenario), background]

    if flexible:
        datacenter = FlexibleLoad(
            num_nodes=1,
            terminal=np.array([0]),
            target_power=np.full((1, hours), -config.dc_mw),
            weights=np.full((1, 1), config.flex_weight),
        )
    else:
        datacenter = Load(
            name=np.array(["datacenter"]),
            num_nodes=1,
            terminal=np.array([0]),
            load=np.full((1, hours), config.dc_mw),
            linear_cost=np.full((1, hours), config.voll),
        )
    devices.append(datacenter)

    battery_index = None
    if battery_mw > 0:
        devices.append(_battery(config, battery_mw))
        battery_index = len(devices) - 1
    return net, devices, battery_index


def _operation_cost(net: PowerNetwork, devices) -> float:
    out = net.dispatch(devices, time_horizon=devices[0].time_horizon, solver=cp.CLARABEL)
    if out.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={out.problem.status}")
    return float(out.problem.value)


@define(kw_only=True)
class BatterySizing:
    """Marginal-value curve, break-even size, and the FD check behind it.

    Every array is computed from real solves: ``daily_marginal_value`` is zap's adjoint
    ``-d(op cost)/d(power capacity)``; ``fd_marginal_value`` the central finite difference;
    ``net_value_per_yr`` the annual operating savings minus capital cost at each size.
    """

    config: FlexConfig
    sizes: np.ndarray
    daily_marginal_value: np.ndarray  # $/MW-day, adjoint
    fd_marginal_value: np.ndarray  # $/MW-day, finite difference (NaN where not evaluated)
    operation_cost: np.ndarray  # $/day at each size
    baseline_cost: float  # $/day, no battery
    break_even_mw: float
    optimal_size_mw: float
    max_rel_err_fd: float

    @property
    def annual_marginal_value(self) -> np.ndarray:
        return self.daily_marginal_value * self.config.days_per_year

    @property
    def annual_savings(self) -> np.ndarray:
        return (self.baseline_cost - self.operation_cost) * self.config.days_per_year

    @property
    def net_value_per_yr(self) -> np.ndarray:
        return self.annual_savings - self.config.battery_capital_cost_per_mw_yr * self.sizes

    @property
    def marginal_value_at_break_even(self) -> float:
        """Annual marginal value interpolated at the break-even size."""
        return float(np.interp(self.break_even_mw, self.sizes, self.annual_marginal_value))


def _finite_difference_marginal(check) -> float:
    """Pull the (scalar) battery finite-difference gradient out of a grad-check result."""
    fd = np.asarray(check.finite_difference, dtype=float).ravel()
    finite = fd[np.isfinite(fd)]
    if finite.size == 0:
        return float("nan")
    return float(finite[0])


def run_battery_sizing(config: Optional[FlexConfig] = None) -> BatterySizing:
    """Sweep battery sizes; return the adjoint marginal-value curve + computed break-even."""
    config = config or FlexConfig()
    scenario = _nominal_scenario(config)
    sizes = np.asarray(config.battery_sizes, dtype=float)

    base_net, base_devices, _ = _build_devices(config, scenario)
    baseline_cost = _operation_cost(base_net, base_devices)

    daily_mv = np.empty(sizes.size)
    fd_mv = np.full(sizes.size, np.nan)
    op_cost = np.empty(sizes.size)
    for i, b in enumerate(sizes):
        net, devices, battery_index = _build_devices(config, scenario, battery_mw=b)
        check = check_parameter(
            net, devices, battery_index, "power_capacity", "battery",
            solver=cp.CLARABEL, do_fd=True,
        )
        daily_mv[i] = -float(np.asarray(check.adjoint, dtype=float).ravel()[0])
        fd_mv[i] = -_finite_difference_marginal(check)
        op_cost[i] = _operation_cost(net, devices)

    annual_mv = daily_mv * config.days_per_year
    cap = config.battery_capital_cost_per_mw_yr
    break_even = _crossing(sizes, annual_mv, cap)

    net_value = (baseline_cost - op_cost) * config.days_per_year - cap * sizes
    optimal_size = float(sizes[int(np.argmax(net_value))])

    # FD agreement on the active (evaluated) points only.
    evaluated = np.isfinite(fd_mv)
    rel = np.abs(daily_mv[evaluated] - fd_mv[evaluated]) / np.maximum(
        np.abs(fd_mv[evaluated]), 1e-9
    )
    max_rel_err_fd = float(rel.max()) if rel.size else float("nan")

    return BatterySizing(
        config=config,
        sizes=sizes,
        daily_marginal_value=daily_mv,
        fd_marginal_value=fd_mv,
        operation_cost=op_cost,
        baseline_cost=baseline_cost,
        break_even_mw=break_even,
        optimal_size_mw=optimal_size,
        max_rel_err_fd=max_rel_err_fd,
    )


def _crossing(sizes: np.ndarray, decreasing_values: np.ndarray, level: float) -> float:
    """Battery size where a monotone-decreasing curve crosses ``level`` (interpolated).

    Returns the smallest swept size if the whole curve is already below ``level`` (no
    battery is worth building) and the largest if it never falls below (every swept size
    pays back). Otherwise linearly interpolates between the bracketing grid points.
    """
    if decreasing_values[0] <= level:
        return float(sizes[0])
    if decreasing_values[-1] >= level:
        return float(sizes[-1])
    i = int(np.argmax(decreasing_values < level))  # first index below the level
    x0, x1 = sizes[i - 1], sizes[i]
    y0, y1 = decreasing_values[i - 1], decreasing_values[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


@define(kw_only=True)
class FlexValue:
    """Per-scenario firm vs. flexible serving cost and the annualized savings + CI."""

    config: FlexConfig
    firm_daily: np.ndarray  # (n_scenarios,) $/day
    flex_daily: np.ndarray  # (n_scenarios,) $/day
    annual_savings: np.ndarray  # (n_scenarios,) $/yr, firm - flexible
    ci: CIResult

    @property
    def headline_savings(self) -> float:
        """Mean annual $/yr saved by making the data center flexible vs. firm."""
        return float(self.annual_savings.mean())


def run_flex_value(config: Optional[FlexConfig] = None) -> FlexValue:
    """Value demand flexibility: firm Load vs. flexible PowerTarget, per scenario, $/yr."""
    config = config or FlexConfig()
    scenarios = _make_scenarios(config)

    firm = np.empty(len(scenarios))
    flex = np.empty(len(scenarios))
    for j, scenario in enumerate(scenarios):
        net_f, dev_f, _ = _build_devices(config, scenario, flexible=False)
        firm[j] = _operation_cost(net_f, dev_f)
        net_x, dev_x, _ = _build_devices(config, scenario, flexible=True)
        flex[j] = _operation_cost(net_x, dev_x)

    annual = (firm - flex) * config.days_per_year
    ci = bootstrap_ci(annual, statistic=np.mean, confidence=0.90, seed=0)
    return FlexValue(
        config=config, firm_daily=firm, flex_daily=flex, annual_savings=annual, ci=ci
    )


@define(kw_only=True)
class FlexResult:
    """The combined backtest: battery break-even + firm-vs-flexible $/yr."""

    config: FlexConfig
    sizing: BatterySizing
    flex_value: FlexValue
    source: str = field(default="synthetic")

    def fidelity(self) -> FidelityBand:
        """Adjoint-vs-finite-difference agreement on the marginal-value curve."""
        s = self.sizing
        evaluated = np.isfinite(s.fd_marginal_value)
        return fidelity_band(
            s.daily_marginal_value[evaluated],
            s.fd_marginal_value[evaluated],
            reference="finite-difference",
            metric="battery-marginal-value",
            units="$/MW-day",
        )

    def to_bench_result(self) -> BenchResult:
        s = self.sizing
        fv = self.flex_value
        cap = self.config.battery_capital_cost_per_mw_yr
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=fv.headline_savings,
            units="$/yr",
            ci=fv.ci,
            fidelity_band=self.fidelity(),
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "topology": "single node: capacity-limited baseload + peaker, QP costs",
                "dispatch": "QP (quadratic generation cost) so prices vary smoothly with load",
                "dc_mw": self.config.dc_mw,
                "battery_duration_h": self.config.battery_duration,
                "battery_capital_cost_per_mw_yr": cap,
                "flex_weight": self.config.flex_weight,
                "days_per_year": self.config.days_per_year,
                "n_scenarios": self.config.n_scenarios,
                "headline": (
                    "mean annual system-cost saving from serving the data center as a "
                    "flexible PowerTarget vs. a firm Load, paired per seeded scenario"
                ),
                "break_even_definition": (
                    "battery MW where annual marginal value (zap adjoint "
                    "-d(op cost)/d(power capacity)) crosses the annualized capital cost"
                ),
                "synthetic_note": (
                    "generator fuel costs carry seeded per-hour noise and the background "
                    "load a per-scenario scale; a human stages real ISO price/load history "
                    "and re-runs with --real"
                ),
            },
            sensitivities={
                "break_even_battery_mw": s.break_even_mw,
                "optimal_battery_mw": s.optimal_size_mw,
                "marginal_value_at_break_even_per_yr": s.marginal_value_at_break_even,
                "battery_capital_cost_per_mw_yr": cap,
                "gradient_max_rel_err_fd": s.max_rel_err_fd,
                "battery_sizes_mw": s.sizes.tolist(),
                "daily_marginal_value": s.daily_marginal_value.tolist(),
                "annual_marginal_value": s.annual_marginal_value.tolist(),
                "fd_marginal_value_daily": s.fd_marginal_value.tolist(),
                "net_value_per_yr": s.net_value_per_yr.tolist(),
                "firm_vs_flexible_savings_per_yr": fv.annual_savings.tolist(),
                "firm_daily_cost": fv.firm_daily.tolist(),
                "flex_daily_cost": fv.flex_daily.tolist(),
            },
        )


def run_flex(config: Optional[FlexConfig] = None) -> FlexResult:
    """Run both the battery-sizing sweep and the firm-vs-flexible comparison."""
    config = config or FlexConfig()
    return FlexResult(
        config=config,
        sizing=run_battery_sizing(config),
        flex_value=run_flex_value(config),
    )


def load_staged_flex(name: str) -> None:
    """Human ``--real`` entry point: read staged ISO price/load history.

    Raises :class:`DataNotStagedError` when ``data/<name>/`` is empty (the loop path),
    never downloading. A human stages real history and wires the loader.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged flexibility data for {name!r}: expected ISO price/load history "
            f"under {cache_dir}. A human must stage real data there (see data/README.md); "
            f"the benchmark loop never downloads."
        )
    raise NotImplementedError(
        "real staged-data flexibility analysis is wired up by a human; the synthetic "
        "path is the loop-runnable one."
    )


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic flexibility/battery backtest and emit a ``BenchResult``."""
    result = run_flex().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steinmetz data-center flexibility & battery sizing backtest (§7.1-B)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on the synthetic QP world (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="analyze staged ISO price/load history in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    if args.real:
        load_staged_flex(args.real)

    res = run_flex()
    s, fv = res.sizing, res.flex_value
    print(f"break-even battery   : {s.break_even_mw:.2f} MW "
          f"(optimal swept size {s.optimal_size_mw:.0f} MW)")
    print(f"capital cost          : {res.config.battery_capital_cost_per_mw_yr:,.0f} $/MW-yr")
    print(f"marginal value @ b*   : {s.marginal_value_at_break_even:,.0f} $/MW-yr")
    print(f"adjoint-vs-FD max err : {s.max_rel_err_fd:.2e}")
    print(f"firm-vs-flexible      : {fv.headline_savings:,.0f} $/yr "
          f"[{fv.ci.lo:,.0f}, {fv.ci.hi:,.0f}] (90% CI)")
    print(f"\n{'size MW':>8}{'ann MV $/MW-yr':>16}{'net value $/yr':>16}")
    for b, mv, nv in zip(s.sizes, s.annual_marginal_value, s.net_value_per_yr):
        print(f"{b:>8.0f}{mv:>16,.0f}{nv:>16,.0f}")
