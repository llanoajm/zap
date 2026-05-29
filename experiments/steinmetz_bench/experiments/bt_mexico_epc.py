"""Mexico EPC dual-regime corridor backtest (roadmap item 3.5, Steinmetz §7.4).

The Mexican grid is dispatched under two very different rulebooks. Historically CENACE ran
a roughly **merit-order** market: the cheapest energy clears first regardless of who owns
it. A recurring policy proposal (and, at times, operating rule) instead imposes a
**minimum CFE-share** mandate — the state utility's plants must supply at least ~54 % of
generation, even when private IPP/renewable energy is cheaper. The two regimes deliver the
same load but route it through the network completely differently, so the *transmission*
question — which corridors are worth expanding — has two different answers. This backtest
encodes both rulebooks in zap's dispatch and shows how the corridor ranking moves between
them. It doubles as the jurisdiction-rule-encoding capability demo: the CFE mandate is a
single global linear constraint bolted onto the standard DC-OPF.

**The synthetic world.** Two generation hubs feed ``K`` load zones over controllable DC
corridors. Node 0 is the cheap **private** hub (cheap IPP/renewable energy); node 1 is the
more expensive **CFE** hub. Each zone ``z`` is reached by exactly two corridors — one from
each hub (``P_z`` from private, ``C_z`` from CFE) — so the supply mix is a genuine routing
choice, not a fixed radial flow. Corridors are DC (controllable) lines on purpose: their
``nominal_capacity`` scales only the thermal limit, so the marginal value of a corridor is a
clean thermal-relief reading, not an impedance/loop-flow artifact. Both hubs carry a small
quadratic cost so the dispatch (and hence every price and gradient) is unique and smooth.
Per-corridor wheeling costs increase with distance so the corridor merit order — and the
binding pattern — is graded rather than tied. Capacities are sized so a single corridor
cannot serve its whole zone (both are used) and so the cheap private corridors bind under
merit order while the CFE corridors bind once the mandate forces CFE energy through them.

**Two regimes, same network.** We solve the DC-OPF twice. *Merit order* is the unconstrained
least-cost dispatch. *CFE mandate* adds one constraint to the very same problem —
``sum(CFE generation) >= 0.54 * sum(total generation)`` — built straight onto zap's
``model_dispatch_problem`` constraint list. Under merit order the cheap private hub wins and
CFE's share sits below the floor; the mandate is therefore binding, and its shadow price
(the constraint's dual) is the marginal $/MW cost of the policy.

**Corridor ranking.** In each regime we rank corridors by their marginal value of capacity,
``-d(system cost)/d(nominal_capacity)``. Under merit order this is read from zap's exact
adjoint and cross-checked against a central finite difference (the
:class:`~...metrics.FidelityBand`); under the CFE mandate — whose extra constraint the stock
adjoint does not see — it is the same finite difference applied to the constrained re-solve.
The merit-order adjoint-vs-FD agreement certifies the FD estimator, lending its credibility
to the CFE-regime numbers. The headline is the **ranking-agreement** between the two regimes
(Spearman rank-correlation of the two marginal-value vectors): a low or negative value is the
finding — the jurisdiction rule materially reorders which corridors are worth building. We
also emit each corridor's annualized congestion-relief value ($/MW-yr) in both regimes.

Every number is computed from an actual zap solve. The ``--real`` path is reserved for a
human who stages a CENACE PML history + a PRODESEN corridor list into ``data/<name>/``; it
blocks via :class:`DataNotStagedError` rather than downloading, matching the rest of suite.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
from attrs import define, field

from zap.devices import DCLine, Generator, Load
from zap.devices.ground import Ground
from zap.network import PowerNetwork, nested_evaluate
from zap.util import expand_params

from experiments.steinmetz_bench.datasets.registry import DATA_ROOT, DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_transmission_audit import spearman
from experiments.steinmetz_bench.experiments.grad_check import check_parameter
from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import CIResult, FidelityBand, fidelity_band

EXPERIMENT_ID = "3.5-mexico-epc"
DATASET = "synthetic-two-hub-cfe"

# CFE minimum-share mandate (Steinmetz §7.4): the state utility must supply at least this
# fraction of total generation. The synthetic costs put merit-order CFE share below it, so
# the mandate binds.
CFE_SHARE_FLOOR = 0.54
# Hours in a year, used to annualize the per-MW congestion-relief value of each corridor.
HOURS_PER_YEAR = 8760.0
# A corridor counts as "binding" (its marginal value is meaningful, not solver noise) once
# its marginal value clears this many $/MW-period.
BINDING_MV_TOL = 1e-2


@define(kw_only=True)
class MexicoConfig:
    """Knobs of the synthetic two-hub CFE world (one representative dispatch period).

    Node 0 is the cheap private hub; node 1 is the dearer CFE hub; nodes ``2..K+1`` are the
    load zones, each reached by a private corridor ``P_z`` and a CFE corridor ``C_z``.
    ``cap_private``/``cap_cfe`` are sized so both corridors of a zone are needed (neither
    can carry the whole zone) and so the private corridors bind under merit order while the
    CFE corridors bind once the mandate is enforced. Wheeling costs grow with distance so the
    corridor ranking is graded. All knobs are deterministic; there is no randomness.
    """

    hours: int = 1

    # Hub generators. Private (node 0) is cheaper; CFE (node 1) is dearer, so merit order
    # under-dispatches CFE and the share mandate binds. Small quadratic terms make the
    # dispatch, prices, and gradients unique and smooth.
    private_cost: float = 10.0
    cfe_cost: float = 22.0
    hub_quadratic_cost: float = 0.01
    hub_cap: float = 500.0

    # Per-zone load (MW) for the representative period.
    zone_loads: tuple[float, ...] = (70.0, 90.0, 110.0, 130.0)
    voll: float = 1.0e4  # value of lost load (never binds; loads are always served)

    # Per-zone corridor ratings (MW). cap_private[z] + cap_cfe[z] must exceed zone load z
    # (feasible by either-hub blend) and sum(cap_cfe) must clear the mandate (deliverable).
    cap_private: tuple[float, ...] = (45.0, 50.0, 55.0, 70.0)
    cap_cfe: tuple[float, ...] = (50.0, 55.0, 60.0, 65.0)

    # Per-corridor wheeling cost step ($/MWh). Private corridors get costlier with zone
    # index, CFE corridors the reverse, so each hub has a distinct corridor merit order.
    wheel_step: float = 0.5

    cfe_share_floor: float = CFE_SHARE_FLOOR
    # Finite-difference step for the marginal-value gradient (MW).
    fd_eps: float = 1.0e-2

    def __attrs_post_init__(self):
        k = len(self.zone_loads)
        if len(self.cap_private) != k or len(self.cap_cfe) != k:
            raise ValueError("zone_loads, cap_private and cap_cfe must share a length")
        if k < 3:
            raise ValueError("need at least three zones for a meaningful corridor ranking")
        loads = np.asarray(self.zone_loads, float)
        cp_ = np.asarray(self.cap_private, float)
        cc = np.asarray(self.cap_cfe, float)
        if np.any(cp_ + cc < loads):
            raise ValueError("each zone must be reachable: cap_private + cap_cfe >= load")
        if cc.sum() < self.cfe_share_floor * loads.sum():
            raise ValueError("sum(cap_cfe) must be able to deliver the CFE-share mandate")
        if self.cfe_cost <= self.private_cost:
            raise ValueError("CFE must be dearer than private for the mandate to bind")
        if not 0.0 < self.cfe_share_floor < 1.0:
            raise ValueError("cfe_share_floor must be a fraction in (0, 1)")

    @property
    def n_zones(self) -> int:
        return len(self.zone_loads)

    @property
    def n_nodes(self) -> int:
        return self.n_zones + 2

    @property
    def n_corridors(self) -> int:
        return 2 * self.n_zones

    @property
    def corridor_names(self) -> list[str]:
        priv = [f"priv-z{z}" for z in range(self.n_zones)]
        cfe = [f"cfe-z{z}" for z in range(self.n_zones)]
        return priv + cfe

    @property
    def is_cfe_corridor(self) -> np.ndarray:
        """Boolean mask over corridors: True for the CFE-hub corridors."""
        return np.array([False] * self.n_zones + [True] * self.n_zones)


def build_zap_devices(config: MexicoConfig) -> tuple[PowerNetwork, list]:
    """Construct the zap network + device list (two hub gens, zone loads, DC corridors)."""
    k = config.n_zones
    n = config.n_nodes
    hours = config.hours
    net = PowerNetwork(n)

    generators = Generator(
        name=np.array(["private", "cfe"]),
        num_nodes=n,
        terminal=np.array([0, 1]),
        dynamic_capacity=np.ones((2, hours)),
        nominal_capacity=np.full(2, config.hub_cap),
        linear_cost=np.array([[config.private_cost], [config.cfe_cost]]) * np.ones((1, hours)),
        quadratic_cost=np.full((2, 1), config.hub_quadratic_cost),
        capital_cost=np.ones(2),
        emission_rates=np.array([0.2, 0.5]),
    )
    loads = Load(
        name=np.array([f"zone{z}" for z in range(k)]),
        num_nodes=n,
        terminal=np.arange(2, n),
        load=np.asarray(config.zone_loads, float)[:, None] * np.ones((1, hours)),
        linear_cost=np.full(k, config.voll),
    )
    wheel_p = config.wheel_step * np.arange(1, k + 1, dtype=float)
    wheel_c = config.wheel_step * np.arange(k, 0, -1, dtype=float)
    lines = DCLine(
        num_nodes=n,
        name=np.array(config.corridor_names),
        source_terminal=np.concatenate([np.zeros(k), np.ones(k)]).astype(int),
        sink_terminal=np.concatenate([np.arange(2, n), np.arange(2, n)]).astype(int),
        capacity=np.ones(2 * k),
        nominal_capacity=np.concatenate([config.cap_private, config.cap_cfe]),
        linear_cost=np.concatenate([wheel_p, wheel_c]),
        capital_cost=np.ones(2 * k),
    )
    return net, [generators, loads, lines]


def _gen_index(devices: list) -> int:
    return next(i for i, d in enumerate(devices) if isinstance(d, Generator))


def _line_index(devices: list) -> int:
    return next(i for i, d in enumerate(devices) if isinstance(d, DCLine))


@define(kw_only=True)
class RegimeSolve:
    """One regime's solved dispatch: cost, nodal prices, generation, and the mandate dual."""

    cost: float
    prices: np.ndarray  # (n_nodes, hours)
    private_gen: float  # MWh over the horizon
    cfe_gen: float  # MWh over the horizon
    flow: np.ndarray  # (n_corridors, hours) sink-terminal power
    share_dual: float  # shadow price of the CFE-share mandate ($/MW); 0 in merit order

    @property
    def cfe_share(self) -> float:
        total = self.private_gen + self.cfe_gen
        return float(self.cfe_gen / total) if total > 0 else 0.0


def solve_regime(net: PowerNetwork, devices: list, *, cfe_share=None, solver=cp.CLARABEL):
    """Solve the DC-OPF, optionally enforcing the CFE minimum-share mandate.

    Reuses zap's :meth:`PowerNetwork.model_dispatch_problem` to assemble the standard
    least-cost DC-OPF, then — for the mandate regime — bolts one extra global constraint
    (``sum(CFE gen) >= cfe_share * sum(total gen)``) onto the constraint list before solving.
    The mandate's dual is recovered as its $/MW shadow price. Mirrors ``net.dispatch``'s
    ground handling so prices match the standard solve.
    """
    horizon = max(d.time_horizon for d in devices)
    devs = list(devices)
    params = expand_params(None, devs)
    devs = devs + [Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]
    params = params + [{}]

    costs, constraints, data = net.model_dispatch_problem(devs, horizon, parameters=params)
    gi = _gen_index(devs)
    li = _line_index(devs)
    gen_power = data["power"][gi][0]  # (n_gen, hours): row 0 = private, row 1 = CFE
    total_gen = cp.sum(gen_power)
    cfe_gen = cp.sum(gen_power[1, :])

    mandate = None
    if cfe_share is not None:
        mandate = cfe_share * total_gen - cfe_gen <= 0  # CFE share at least the floor
        constraints = constraints + [mandate]

    problem = cp.Problem(cp.Minimize(cp.sum(costs)), constraints)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"zap dispatch did not solve: status={problem.status}")

    power = nested_evaluate(data["power"])
    gen_values = power[gi][0]
    share_dual = float(mandate.dual_value) if mandate is not None else 0.0
    return RegimeSolve(
        cost=float(problem.value),
        prices=-data["power_balance"].dual_value,
        private_gen=float(gen_values[0].sum()),
        cfe_gen=float(gen_values[1].sum()),
        flow=np.asarray(power[li][1], dtype=float),
        share_dual=share_dual,
    )


def _marginal_value_fd(net, devices, *, cfe_share, eps) -> np.ndarray:
    """Corridor marginal value ``-d(system cost)/d(nominal_capacity)`` by central FD.

    Re-solves the (optionally mandate-constrained) dispatch at ``cap ± eps`` for each
    corridor. Works under both regimes — including the CFE mandate, whose extra constraint
    the stock adjoint cannot differentiate through.
    """
    li = _line_index(devices)
    shape = np.asarray(devices[li].nominal_capacity).shape
    base = np.asarray(devices[li].nominal_capacity, float).ravel()
    mv = np.empty(base.size)
    for j in range(base.size):
        def perturbed(sign):
            dev = copy.deepcopy(devices[li])
            v = np.asarray(dev.nominal_capacity, float).copy()
            v.ravel()[j] += sign * eps
            dev.nominal_capacity = v.reshape(shape)
            patched = list(devices)
            patched[li] = dev
            return solve_regime(net, patched, cfe_share=cfe_share).cost

        mv[j] = -(perturbed(+1) - perturbed(-1)) / (2.0 * eps)
    return mv


@define(kw_only=True)
class MeritRanking:
    """Merit-order corridor marginal value from zap's adjoint, with the FD cross-check."""

    marginal_value: np.ndarray  # (n_corridors,) -d(cost)/d(cap), the ranking signal
    adjoint: np.ndarray  # (n_corridors,) zap exact adjoint marginal value
    fd: np.ndarray  # (n_corridors,) finite difference, NaN where not evaluated
    active_mask: np.ndarray  # (n_corridors,) which corridors bind
    max_rel_err_fd: float

    def fidelity(self) -> FidelityBand:
        """Adjoint-vs-finite-difference agreement on the merit-order marginal value."""
        evaluated = np.isfinite(self.fd)
        return fidelity_band(
            self.adjoint[evaluated], self.fd[evaluated],
            reference="finite-difference", metric="corridor-marginal-value",
            units="$/MW-period",
        )


def run_merit_ranking(config: MexicoConfig) -> MeritRanking:
    """Rank corridors under merit order via zap's adjoint, cross-checked against FD."""
    net, devices = build_zap_devices(config)
    li = _line_index(devices)
    check = check_parameter(
        net, devices, li, "nominal_capacity", "line", solver=cp.CLARABEL, do_fd=True
    )
    adjoint = -np.asarray(check.adjoint, dtype=float).ravel()
    fd = -np.asarray(check.finite_difference, dtype=float).ravel()
    active = np.asarray(check.active_mask, dtype=bool).ravel()

    evaluated = np.isfinite(fd)
    if evaluated.any():
        rel = np.abs(adjoint[evaluated] - fd[evaluated]) / np.maximum(
            np.abs(fd[evaluated]), 1e-9
        )
        max_rel_err_fd = float(rel.max())
    else:
        max_rel_err_fd = float("nan")

    return MeritRanking(
        marginal_value=adjoint,
        adjoint=adjoint,
        fd=fd,
        active_mask=active,
        max_rel_err_fd=max_rel_err_fd,
    )


def _bootstrap_spearman_ci(
    merit_mv: np.ndarray,
    cfe_mv: np.ndarray,
    confidence: float = 0.90,
    n_boot: int = 2000,
    seed: int = 0,
) -> CIResult:
    """Percentile CI on the ranking-agreement Spearman by resampling corridors.

    Each resample draws corridor indices with replacement and recomputes the rank
    correlation between the two regimes' marginal-value vectors, so the interval reflects
    how much the agreement depends on the particular corridor set.
    """
    merit = np.asarray(merit_mv, float).ravel()
    cfe = np.asarray(cfe_mv, float).ravel()
    n = merit.size
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = spearman(merit[idx], cfe[idx])
    alpha = (1.0 - confidence) / 2.0
    return CIResult(
        lo=float(np.percentile(boot, 100.0 * alpha)),
        mid=float(np.percentile(boot, 50.0)),
        hi=float(np.percentile(boot, 100.0 * (1.0 - alpha))),
        confidence=confidence,
    )


@define(kw_only=True)
class MexicoResult:
    """The dual-regime audit: per-regime rankings, the mandate's effect, and agreement."""

    config: MexicoConfig
    merit: MeritRanking
    cfe_mv: np.ndarray  # (n_corridors,) CFE-regime marginal value (FD)
    merit_solve: RegimeSolve
    cfe_solve: RegimeSolve
    ranking_agreement: float  # Spearman between the two regimes' marginal-value vectors
    ci: CIResult
    source: str = field(default="synthetic")

    @property
    def merit_ranking(self) -> list[int]:
        return list(np.argsort(self.merit.marginal_value)[::-1])

    @property
    def cfe_ranking(self) -> list[int]:
        return list(np.argsort(self.cfe_mv)[::-1])

    @property
    def cost_increase(self) -> float:
        return self.cfe_solve.cost - self.merit_solve.cost

    @property
    def max_price_shift(self) -> float:
        return float(np.abs(self.cfe_solve.prices - self.merit_solve.prices).max())

    @property
    def cfe_generation_increase(self) -> float:
        return self.cfe_solve.cfe_gen - self.merit_solve.cfe_gen

    @property
    def mandate_binding(self) -> bool:
        """The mandate binds: merit share is below the floor and it lifts CFE to the floor."""
        cfg = self.config
        below = self.merit_solve.cfe_share < cfg.cfe_share_floor - 1e-6
        lifted = self.cfe_solve.cfe_share >= cfg.cfe_share_floor - 1e-4
        return bool(below and lifted)

    def _relief_per_year(self, mv: np.ndarray) -> list[float]:
        # mv is the value of 1 MW over the modeled `hours`-period horizon; scale up to a year.
        return (np.asarray(mv, float) * (HOURS_PER_YEAR / self.config.hours)).tolist()

    def to_bench_result(self) -> BenchResult:
        cfg = self.config
        merit = self.merit
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.ranking_agreement,
            units="spearman",
            ci=self.ci,
            fidelity_band=merit.fidelity(),
            assumptions={
                "source": self.source,
                "zap_solver": "CLARABEL",
                "topology": (
                    f"two hubs (cheap private @node0, dearer CFE @node1) feeding "
                    f"{cfg.n_zones} load zones over {cfg.n_corridors} controllable DC corridors "
                    "(one private + one CFE corridor per zone)"
                ),
                "regime_merit": "unconstrained least-cost DC-OPF (historical merit order)",
                "regime_cfe": (
                    f"same DC-OPF + mandate sum(CFE gen) >= {cfg.cfe_share_floor} * sum(total "
                    "gen), added directly to zap's model_dispatch_problem constraint list"
                ),
                "ranking_signal": (
                    "corridor marginal value = -d(system cost)/d(nominal_capacity); merit-order "
                    "value from zap's exact adjoint (FD-certified), CFE-regime value from the "
                    "same finite difference on the mandate-constrained re-solve"
                ),
                "headline": (
                    "ranking agreement = Spearman rank-correlation between the merit-order and "
                    "CFE-mandate corridor marginal-value vectors (low/negative => the rule "
                    "reorders which corridors are worth expanding)"
                ),
                "cfe_share_floor": cfg.cfe_share_floor,
                "hours_per_year": HOURS_PER_YEAR,
                "fd_eps_mw": cfg.fd_eps,
                "synthetic_note": (
                    "CFE is dearer than private, so merit order under-dispatches it and the "
                    "share mandate binds; a human stages a real CENACE PML history + PRODESEN "
                    "corridor list and re-runs with --real"
                ),
            },
            sensitivities={
                "ranking_agreement_spearman": self.ranking_agreement,
                "corridor_names": cfg.corridor_names,
                "is_cfe_corridor": cfg.is_cfe_corridor.astype(int).tolist(),
                "merit_marginal_value_per_mw": merit.marginal_value.tolist(),
                "cfe_marginal_value_per_mw": np.asarray(self.cfe_mv, float).tolist(),
                "merit_adjoint_per_mw": merit.adjoint.tolist(),
                "merit_fd_per_mw": [
                    float(v) if np.isfinite(v) else None for v in merit.fd
                ],
                "merit_active_mask": merit.active_mask.astype(int).tolist(),
                "gradient_max_rel_err_fd": merit.max_rel_err_fd,
                "merit_ranking": [cfg.corridor_names[i] for i in self.merit_ranking],
                "cfe_ranking": [cfg.corridor_names[i] for i in self.cfe_ranking],
                "congestion_relief_per_mw_year_merit": self._relief_per_year(
                    merit.marginal_value
                ),
                "congestion_relief_per_mw_year_cfe": self._relief_per_year(self.cfe_mv),
                "cfe_share_merit": self.merit_solve.cfe_share,
                "cfe_share_cfe": self.cfe_solve.cfe_share,
                "mandate_shadow_price_per_mw": self.cfe_solve.share_dual,
                "system_cost_merit": self.merit_solve.cost,
                "system_cost_cfe": self.cfe_solve.cost,
                "cfe_mandate_cost_increase": self.cost_increase,
                "max_nodal_price_shift": self.max_price_shift,
                "cfe_generation_increase_mwh": self.cfe_generation_increase,
                "mandate_binding": self.mandate_binding,
            },
        )


def run_mexico(config: Optional[MexicoConfig] = None) -> MexicoResult:
    """Run both dispatch regimes, rank corridors in each, and measure their agreement."""
    config = config or MexicoConfig()
    net, devices = build_zap_devices(config)

    merit = run_merit_ranking(config)
    cfe_mv = _marginal_value_fd(
        net, devices, cfe_share=config.cfe_share_floor, eps=config.fd_eps
    )
    merit_solve = solve_regime(net, devices, cfe_share=None)
    cfe_solve = solve_regime(net, devices, cfe_share=config.cfe_share_floor)

    agreement = spearman(merit.marginal_value, cfe_mv)
    ci = _bootstrap_spearman_ci(merit.marginal_value, cfe_mv)

    return MexicoResult(
        config=config,
        merit=merit,
        cfe_mv=cfe_mv,
        merit_solve=merit_solve,
        cfe_solve=cfe_solve,
        ranking_agreement=agreement,
        ci=ci,
    )


def load_staged_mexico(name: str) -> None:
    """Human ``--real`` entry point: read a staged CENACE PML + PRODESEN corridor list.

    Raises :class:`DataNotStagedError` when ``data/<name>/`` is empty (the loop path), never
    downloading. A human stages real CENACE market data and wires the loader.
    """
    cache_dir = DATA_ROOT / name
    if not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise DataNotStagedError(
            f"No staged Mexico EPC data for {name!r}: expected a CENACE PML history + a "
            f"PRODESEN corridor list under {cache_dir}. A human must stage real data there "
            f"(see data/README.md); the benchmark loop never downloads."
        )
    raise NotImplementedError(
        "real staged-data Mexico EPC audit is wired up by a human; the synthetic path is the "
        "loop-runnable one."
    )


def run(report_path: Optional[Path] = None) -> BenchResult:
    """Run the synthetic dual-regime Mexico EPC backtest and emit a ``BenchResult``."""
    result = run_mexico().to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steinmetz Mexico EPC dual-regime backtest (§7.4)")
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="run on the synthetic two-hub CFE world (default, loop-runnable)",
    )
    parser.add_argument(
        "--real", metavar="NAME", default=None,
        help="audit staged CENACE/PRODESEN data in data/NAME/ (human path)",
    )
    args = parser.parse_args()
    if args.real:
        load_staged_mexico(args.real)

    res = run_mexico()
    cfg = res.config
    print(f"zones / corridors    : {cfg.n_zones} / {cfg.n_corridors}")
    print(f"CFE share (merit)    : {res.merit_solve.cfe_share:.3f}")
    print(f"CFE share (mandate)  : {res.cfe_solve.cfe_share:.3f} (floor {cfg.cfe_share_floor})")
    print(f"mandate shadow price : {res.cfe_solve.share_dual:,.2f} $/MW")
    print(f"system cost increase : {res.cost_increase:,.2f}")
    print(f"max nodal price shift: {res.max_price_shift:,.3f}")
    print(f"ranking agreement    : {res.ranking_agreement:.3f} (Spearman) "
          f"[{res.ci.lo:.3f}, {res.ci.hi:.3f}]")
    print(f"adjoint-vs-FD max err: {res.merit.max_rel_err_fd:.2e}")
    print(f"\n{'corridor':<10}{'MV merit $/MW':>16}{'MV cfe $/MW':>16}")
    for i in range(cfg.n_corridors):
        print(f"{cfg.corridor_names[i]:<10}{res.merit.marginal_value[i]:>16,.3f}"
              f"{res.cfe_mv[i]:>16,.3f}")
    print(f"\nmerit ranking : {[cfg.corridor_names[i] for i in res.merit_ranking]}")
    print(f"cfe   ranking : {[cfg.corridor_names[i] for i in res.cfe_ranking]}")
