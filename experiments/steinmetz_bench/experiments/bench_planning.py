"""Planning benchmark (roadmap item 2.2, Steinmetz §8.4.2).

zap's headline planning claim is that its *gradient* expansion planner — projected
gradient descent on the dispatch layer's adjoint sensitivities — finds the same
least-cost capacity expansion that a global optimizer would, across multiple demand
scenarios and over generation, transmission, and storage levers at once. This
benchmark certifies that claim by pitting the gradient planner against an
**independent global optimum**.

The independent baseline is a single joint expansion LP assembled here in CVXPY
(:func:`build_expansion_lp`). Because every expansion lever in this benchmark scales
only a *bound* (generator output limit, DC-line flow limit, battery power/energy
limit) and never a coupling coefficient, the multi-scenario expansion problem is
jointly convex, so the LP's optimum is the true global optimum ``f*`` — a genuine
lower bound on what any feasible planner can achieve. The gradient planner produces
a feasible point, so its objective is ``>= f*`` up to solver noise; certifying
``planner_obj <= f* (1 + tol)`` therefore proves the planner converged to the global
optimum, not merely that it beat some coarse grid.

To keep the LP honest (an independent re-derivation, not a copy of zap's solver), a
companion test fixes the capacities and checks the LP's per-scenario operation cost
against zap's own :meth:`~zap.network.PowerNetwork.dispatch` — the same
baseline-faithfulness check the speed benchmark (item 2.1) uses.

Deliberately uses **DC** lines for the expandable corridor: an AC line's
``nominal_capacity`` scales both its thermal limit *and* its susceptance (the
bilinear effect documented in ``grad_check``), which would make the expansion
problem nonconvex and rob the LP of its global-optimum guarantee. The §8.4.2
network here is a synthetic two-node, multi-scenario expansion case; the large
multi-region expansion headline from the spec stays human-gated.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
from attrs import define

from zap.devices import Battery, DCLine, Generator, Ground, Load
from zap.layer import DispatchLayer
from zap.network import PowerNetwork
from zap.planning import (
    DispatchCostObjective,
    GradientDescent,
    InvestmentObjective,
    PlanningProblem,
    StochasticPlanningProblem,
)
from zap.planning.trackers import LOSS, PARAM, PROJ_GRAD_NORM, TIME

from experiments.steinmetz_bench.reports.result import BenchResult
from experiments.steinmetz_bench.scoring.metrics import FidelityBand, fidelity_band

EXPERIMENT_ID = "2.2-planning"
DATASET = "synthetic-2bus-multiscenario"

# Optimality tolerance (relative, on the joint planning objective). The LP is the
# global optimum f*; the planner is a feasible point, so it can only sit at or above
# f*. This bounds how far above f* the best gradient-descent iterate may land.
OPT_REL_TOL = 1e-2
# zap dispatch solver (conic). The LP baseline picks the best available LP solver.
ZAP_SOLVER = cp.CLARABEL
# Storage round-trip efficiency split symmetrically across charge/discharge.
_DUR = 4.0
_EFF = 0.95
_SOC = 0.5


@define(kw_only=True)
class PlanningConfig:
    """Knobs for the synthetic multi-scenario expansion case.

    Floors are both the planner's lower bounds and the investment baseline (zap
    charges ``capital_cost * (capacity - floor)``); uppers bound the LP and the
    projection step. The defaults give an interior optimum on at least one lever so
    the test is a real convergence check, not a trivial corner solution.
    """

    hours: int = 8
    n_scenarios: int = 2
    seed: int = 0
    # Generation: cheap expandable unit at node 0, expensive fixed backstop at node 1.
    gen_floor: float = 60.0
    gen_upper: float = 220.0
    gen_cheap_cost: float = 10.0
    gen_backstop_cost: float = 100.0
    gen_backstop_cap: float = 1000.0
    gen_capital: float = 180.0
    # DC corridor 0 -> 1.
    line_floor: float = 50.0
    line_upper: float = 260.0
    line_capital: float = 90.0
    # Battery at node 1.
    batt_floor: float = 1.0
    batt_upper: float = 130.0
    batt_capital: float = 40.0
    # Load (node 1) peak; scenarios scale and phase-shift this base.
    load_peak: float = 180.0
    voll: float = 1000.0
    snapshot_weight: float = 1.0
    # Small quadratic dispatch costs strictly convexify the problem: the optimal
    # value becomes a smooth strictly-convex function of the capacities, so projected
    # gradient descent converges geometrically to the unique optimum (a piecewise-
    # linear LP would instead leave it oscillating around a vertex). Kept small so
    # the economics stay merit-order; the QP baseline carries the same terms, so the
    # comparison is exact.
    gen_quadratic: float = 0.05
    line_quadratic: float = 0.02
    batt_quadratic: float = 0.02

    @property
    def floors(self) -> dict[str, float]:
        return {
            "gen_cap": self.gen_floor,
            "line_cap": self.line_floor,
            "battery_cap": self.batt_floor,
        }

    @property
    def uppers(self) -> dict[str, float]:
        return {
            "gen_cap": self.gen_upper,
            "line_cap": self.line_upper,
            "battery_cap": self.batt_upper,
        }

    @property
    def capital(self) -> dict[str, float]:
        return {
            "gen_cap": self.gen_capital,
            "line_cap": self.line_capital,
            "battery_cap": self.batt_capital,
        }


# Device positions in every scenario's device list (shared parameter indices).
PARAM_NAMES = {
    "gen_cap": (0, "nominal_capacity"),
    "line_cap": (3, "nominal_capacity"),
    "battery_cap": (4, "power_capacity"),
}


def _scenario_load(cfg: PlanningConfig, k: int) -> np.ndarray:
    """A deterministic per-scenario daily load shape at the single load node."""
    rng = np.random.default_rng(cfg.seed + 1000 * k)
    t = np.arange(cfg.hours)
    # Phase-shift the peak and scale the trough differently per scenario so the
    # optimal expansion mix genuinely differs across scenarios.
    phase = 5.0 + 2.0 * k
    scale = 1.0 - 0.18 * k
    base = 0.55 * cfg.load_peak + 0.45 * cfg.load_peak * np.sin(2 * np.pi * (t - phase) / cfg.hours)
    noise = rng.normal(0.0, 0.02 * cfg.load_peak, size=cfg.hours)
    return np.clip(scale * base + noise, 0.05 * cfg.load_peak, None)


def build_scenario(cfg: PlanningConfig, k: int) -> tuple[PowerNetwork, list, np.ndarray]:
    """Build scenario ``k``: a 2-node net with the shared expansion levers.

    Devices (fixed positions): ``[g_cheap, g_backstop, load, line, battery, ground]``.
    The capacities of ``g_cheap``, ``line`` and ``battery`` are the planner's three
    decision variables; every scenario starts them at the floor values so the
    investment baseline is identical across scenarios.
    """
    hours = cfg.hours
    net = PowerNetwork(2)
    load_profile = _scenario_load(cfg, k)

    g_cheap = Generator(
        name="g_cheap",
        num_nodes=2,
        terminal=np.array([0]),
        dynamic_capacity=np.ones((1, hours)),
        linear_cost=np.array([cfg.gen_cheap_cost]),
        quadratic_cost=np.array([cfg.gen_quadratic]),
        nominal_capacity=np.array([cfg.gen_floor]),
        capital_cost=np.array([cfg.gen_capital]),
    )
    g_backstop = Generator(
        name="g_backstop",
        num_nodes=2,
        terminal=np.array([1]),
        dynamic_capacity=np.ones((1, hours)),
        linear_cost=np.array([cfg.gen_backstop_cost]),
        quadratic_cost=np.array([cfg.gen_quadratic]),
        nominal_capacity=np.array([cfg.gen_backstop_cap]),
    )
    load = Load(
        name="load",
        num_nodes=2,
        terminal=np.array([1]),
        load=load_profile.reshape(1, hours),
        linear_cost=np.array([cfg.voll]),
    )
    line = DCLine(
        name="line",
        num_nodes=2,
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([1.0]),
        nominal_capacity=np.array([cfg.line_floor]),
        linear_cost=np.array([0.0]),
        quadratic_cost=np.array([cfg.line_quadratic]),
        capital_cost=np.array([cfg.line_capital]),
    )
    battery = Battery(
        name="battery",
        num_nodes=2,
        terminal=np.array([1]),
        power_capacity=np.array([cfg.batt_floor]),
        duration=np.array([_DUR]),
        charge_efficiency=np.array([_EFF]),
        discharge_efficiency=np.array([_EFF]),
        initial_soc=np.array([_SOC]),
        final_soc=np.array([_SOC]),
        linear_cost=np.array([0.0]),
        quadratic_cost=np.array([cfg.batt_quadratic]),
        capital_cost=np.array([cfg.batt_capital]),
    )
    ground = Ground(num_nodes=2, terminal=np.array([0]))
    devices = [g_cheap, g_backstop, load, line, battery, ground]
    return net, devices, load_profile


def make_scenarios(cfg: PlanningConfig) -> list[tuple[PowerNetwork, list, np.ndarray]]:
    return [build_scenario(cfg, k) for k in range(cfg.n_scenarios)]


# ----------------------------------------------------------------------------------
# Independent baseline: the joint multi-scenario expansion LP (the global optimum).
# ----------------------------------------------------------------------------------


def _cap_expr(name, cfg, fixed_caps, constraints):
    """A capacity term: a bounded CVXPY variable, or a constant when fixed."""
    if fixed_caps is not None:
        return float(fixed_caps[name])
    var = cp.Variable(nonneg=True, name=name)
    constraints += [var >= cfg.floors[name], var <= cfg.uppers[name]]
    return var


def _scenario_lp_terms(net, devices, load_profile, cfg, caps, constraints):
    """Operation-cost terms + node-balance constraints for one scenario.

    Mirrors zap's per-device operation cost and feasible set exactly so the LP's
    optimum is the same problem zap's dispatch solves, only with the capacities
    promoted to shared decision variables.
    """
    hours = cfg.hours
    inj = [[] for _ in range(net.num_nodes)]
    cost_terms: list = []

    g_cheap, g_backstop, load, line, battery, _ = devices

    # Cheap generator (expandable): 0 <= p <= availability * cap.
    p_cheap = cp.Variable((1, hours), name="p_cheap", nonneg=True)
    constraints.append(p_cheap <= caps["gen_cap"])  # availability == 1
    cost_terms.append(cfg.gen_cheap_cost * cp.sum(p_cheap)
                      + cfg.gen_quadratic * cp.sum_squares(p_cheap))
    inj[0].append(p_cheap[0, :])

    # Backstop generator (fixed cap).
    p_back = cp.Variable((1, hours), name="p_back", nonneg=True)
    constraints.append(p_back <= cfg.gen_backstop_cap)
    cost_terms.append(cfg.gen_backstop_cost * cp.sum(p_back)
                      + cfg.gen_quadratic * cp.sum_squares(p_back))
    inj[1].append(p_back[0, :])

    # Load: p in [-profile, 0]; curtailment cost VOLL*(p + profile).
    p_load = cp.Variable((1, hours), name="p_load")
    constraints += [p_load <= 0.0, p_load >= -load_profile.reshape(1, hours)]
    cost_terms.append(cfg.voll * cp.sum(p_load + load_profile.reshape(1, hours)))
    inj[1].append(p_load[0, :])

    # DC line (expandable): |flow| <= capacity * cap; sink gets +flow, source -flow.
    flow = cp.Variable((1, hours), name="flow")
    constraints += [flow <= caps["line_cap"], flow >= -caps["line_cap"]]
    cost_terms.append(cfg.line_quadratic * cp.sum_squares(flow))
    inj[1].append(flow[0, :])
    inj[0].append(-flow[0, :])

    # Battery (expandable power, fixed duration): zap StorageUnit model.
    charge = cp.Variable((1, hours), name="charge", nonneg=True)
    discharge = cp.Variable((1, hours), name="discharge", nonneg=True)
    energy = cp.Variable((1, hours + 1), name="energy", nonneg=True)
    ecap = caps["battery_cap"] * _DUR
    constraints += [
        charge <= caps["battery_cap"],
        discharge <= caps["battery_cap"],
        energy <= ecap,
        energy[:, 0:1] == _SOC * ecap,
        energy[:, hours : hours + 1] == _SOC * ecap,
    ]
    for t in range(hours):
        constraints.append(
            energy[:, t + 1] == energy[:, t] + _EFF * charge[:, t] - discharge[:, t] / _EFF
        )
    cost_terms.append(cfg.batt_quadratic * cp.sum_squares(discharge))
    inj[1].append((discharge - charge)[0, :])

    for node in range(net.num_nodes):
        if inj[node]:
            constraints.append(cp.sum(inj[node]) == 0.0)

    return cp.sum(cost_terms)


def build_expansion_lp(scenarios, cfg: PlanningConfig, fixed_caps=None):
    """Assemble the joint multi-scenario expansion LP.

    Returns ``(problem, caps, op_costs)`` where ``caps`` maps each lever to its
    CVXPY variable (or the fixed constant) and ``op_costs`` is the per-scenario
    operation-cost expression list (so a caller can read scenario costs back).
    """
    constraints: list = []
    caps = {name: _cap_expr(name, cfg, fixed_caps, constraints) for name in PARAM_NAMES}

    weight = 1.0 / len(scenarios)
    op_costs = [
        _scenario_lp_terms(net, devices, load, cfg, caps, constraints)
        for (net, devices, load) in scenarios
    ]
    operation = cfg.snapshot_weight * weight * cp.sum(op_costs)

    investment = cp.sum(
        [cfg.capital[name] * (caps[name] - cfg.floors[name]) for name in PARAM_NAMES]
    )

    objective = operation + investment
    return cp.Problem(cp.Minimize(objective), constraints), caps, op_costs


def _lp_solver() -> str:
    installed = set(cp.installed_solvers())
    for name in ("MOSEK", "HIGHS"):
        if name in installed:
            return name
    return "CLARABEL"


def solve_baseline(scenarios, cfg: PlanningConfig) -> tuple[float, dict, str]:
    """Solve the joint LP; return (objective, optimal caps, solver name)."""
    solver = _lp_solver()
    prob, caps, _ = build_expansion_lp(scenarios, cfg)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"baseline expansion LP failed: status={prob.status}")
    opt_caps = {name: float(np.asarray(v.value).ravel()[0]) for name, v in caps.items()}
    return float(prob.value), opt_caps, solver


# ----------------------------------------------------------------------------------
# The gradient planner under test.
# ----------------------------------------------------------------------------------


def build_planning_problem(scenarios, cfg: PlanningConfig):
    """Build the multi-scenario gradient planning problem (one subproblem/scenario)."""
    lower = {k: np.array([v]) for k, v in cfg.floors.items()}
    upper = {k: np.array([v]) for k, v in cfg.uppers.items()}

    subproblems = []
    for net, devices, _ in scenarios:
        layer = DispatchLayer(
            net, devices, parameter_names=PARAM_NAMES,
            time_horizon=cfg.hours, solver=ZAP_SOLVER, add_ground=False,
        )
        op_obj = DispatchCostObjective(net, devices)
        inv_obj = InvestmentObjective(devices, layer)
        subproblems.append(
            PlanningProblem(
                operation_objective=op_obj,
                investment_objective=inv_obj,
                layer=layer,
                lower_bounds=lower,
                upper_bounds=upper,
                snapshot_weight=cfg.snapshot_weight,
            )
        )

    weights = [1.0 / len(subproblems)] * len(subproblems)
    return StochasticPlanningProblem(subproblems, weights)


@define(kw_only=True)
class PlannerRun:
    """Outcome of the gradient planner: best iterate, trajectory, and timing."""

    best_obj: float
    best_caps: dict
    initial_obj: float
    n_iterations: int
    final_proj_grad_norm: float
    loss_history: list
    solve_seconds: float


# Diminishing-step schedule: (step_size, iterations) phases run back-to-back, each
# starting from the previous phase's final state. A fixed step descends fast but then
# oscillates around the optimum; shrinking it across phases damps that oscillation so
# the best iterate lands within OPT_REL_TOL of the global optimum. Each phase's clip
# matches its step so a single step can never jump more than ~step*clip in capacity.
DEFAULT_SCHEDULE: tuple[tuple[float, int], ...] = ((0.08, 50), (0.025, 50), (0.008, 50))


def run_planner(
    scenarios, cfg: PlanningConfig, schedule=DEFAULT_SCHEDULE, clip: float = 1e3,
) -> PlannerRun:
    """Run scheduled projected gradient descent; return its best (lowest-loss) iterate.

    Projected gradient descent on the (strictly convex, smooth) expansion objective
    converges geometrically within a phase but stalls in an oscillation whose
    amplitude scales with the step, so the schedule shrinks the step across phases and
    the *best* visited iterate — not the last — is the planner's reported solution.
    This is the standard convergence guarantee for the method and is what makes the
    optimality check against the global optimum meaningful.
    """
    problem = build_planning_problem(scenarios, cfg)
    state = {k: np.array([float(v)]) for k, v in cfg.floors.items()}

    losses: list[float] = []
    params: list[dict] = []
    last_proj_grad = float("nan")
    total_iters = 0

    t0 = time.perf_counter()
    for step_size, n_iter in schedule:
        state, history = problem.solve(
            num_iterations=n_iter,
            algorithm=GradientDescent(step_size=step_size, clip=clip),
            initial_state={k: np.array(v, dtype=float) for k, v in state.items()},
            trackers=[LOSS, PARAM, PROJ_GRAD_NORM, TIME],
            verbosity=0,
        )
        losses += [float(np.asarray(x).ravel()[0]) for x in history[LOSS]]
        params += list(history[PARAM])
        last_proj_grad = float(np.asarray(history[PROJ_GRAD_NORM][-1]).ravel()[0])
        total_iters += n_iter
    elapsed = time.perf_counter() - t0

    best_i = int(np.argmin(losses))
    best_state = params[best_i]
    best_caps = {k: float(np.asarray(v).ravel()[0]) for k, v in best_state.items()}

    return PlannerRun(
        best_obj=losses[best_i],
        best_caps=best_caps,
        initial_obj=losses[0],
        n_iterations=total_iters,
        final_proj_grad_norm=last_proj_grad,
        loss_history=losses,
        solve_seconds=elapsed,
    )


# ----------------------------------------------------------------------------------
# Report assembly.
# ----------------------------------------------------------------------------------


@define(kw_only=True)
class PlanningReport:
    """The planner run, the LP baseline, and the config that produced them."""

    cfg: PlanningConfig
    planner: PlannerRun
    baseline_obj: float
    baseline_caps: dict
    baseline_solver: str

    @property
    def optimality_gap(self) -> float:
        """Signed relative gap of the planner above the global optimum f*."""
        denom = max(abs(self.baseline_obj), 1.0)
        return (self.planner.best_obj - self.baseline_obj) / denom

    @property
    def savings_vs_floor(self) -> float:
        """Total cost reduction the planner achieved over the no-expansion start."""
        return self.planner.initial_obj - self.planner.best_obj

    def fidelity(self) -> FidelityBand:
        """Planner-vs-global-optimum objective agreement, as the result's band."""
        return fidelity_band(
            [self.planner.best_obj], [self.baseline_obj],
            reference="joint-expansion-lp", metric="planning-objective", units="$",
        )

    def to_bench_result(self) -> BenchResult:
        return BenchResult(
            experiment_id=EXPERIMENT_ID,
            dataset=DATASET,
            headline_number=self.planner.best_obj,
            units="$",
            fidelity_band=self.fidelity(),
            assumptions={
                "planner": "zap projected gradient descent (DispatchLayer adjoint)",
                "baseline": "independent joint multi-scenario expansion LP (global optimum f*)",
                "baseline_solver": self.baseline_solver,
                "zap_solver": "CLARABEL",
                "levers": ["generator", "dc_line", "battery"],
                "n_scenarios": self.cfg.n_scenarios,
                "hours": self.cfg.hours,
                "snapshot_weight": self.cfg.snapshot_weight,
                "opt_rel_tol": OPT_REL_TOL,
                "convexity_note": (
                    "DC line used so nominal_capacity scales only the flow limit; the "
                    "multi-scenario expansion is then jointly convex and the LP optimum "
                    "is the true global lower bound"
                ),
                "headline_gating": (
                    "synthetic two-node multi-scenario case; the large multi-region "
                    "expansion headline is human-gated"
                ),
            },
            sensitivities={
                "optimality_gap": self.optimality_gap,
                "savings_vs_no_expansion": self.savings_vs_floor,
                "initial_objective": self.planner.initial_obj,
                "planner_objective": self.planner.best_obj,
                "baseline_objective": self.baseline_obj,
                "planner_caps": self.planner.best_caps,
                "baseline_caps": self.baseline_caps,
                "n_iterations": self.planner.n_iterations,
                "final_proj_grad_norm": self.planner.final_proj_grad_norm,
                "planner_solve_seconds": self.planner.solve_seconds,
            },
        )


def run_planning_benchmark(
    cfg: PlanningConfig | None = None, schedule=DEFAULT_SCHEDULE,
) -> PlanningReport:
    """Run the gradient planner and the independent LP baseline on ``cfg``."""
    cfg = cfg or PlanningConfig()
    scenarios = make_scenarios(cfg)
    baseline_obj, baseline_caps, solver = solve_baseline(scenarios, cfg)
    planner = run_planner(scenarios, cfg, schedule=schedule)
    return PlanningReport(
        cfg=cfg, planner=planner,
        baseline_obj=baseline_obj, baseline_caps=baseline_caps, baseline_solver=solver,
    )


def run(report_path=None, cfg: PlanningConfig | None = None, schedule=DEFAULT_SCHEDULE) -> BenchResult:
    """Run the benchmark and emit (optionally write) a :class:`BenchResult`."""
    result = run_planning_benchmark(cfg=cfg, schedule=schedule).to_bench_result()
    if report_path is not None:
        result.write_markdown(report_path)
    return result


if __name__ == "__main__":
    report = run_planning_benchmark()
    print(f"baseline solver: {report.baseline_solver}")
    print(f"{'lever':<12} {'planner':>10} {'baseline':>10}")
    for name in PARAM_NAMES:
        print(f"{name:<12} {report.planner.best_caps[name]:>10.2f} "
              f"{report.baseline_caps[name]:>10.2f}")
    print(f"\ninitial obj : {report.planner.initial_obj:>14.2f}")
    print(f"planner obj : {report.planner.best_obj:>14.2f}")
    print(f"baseline obj: {report.baseline_obj:>14.2f}")
    print(f"opt gap     : {report.optimality_gap:>14.3e}  (tol {OPT_REL_TOL})")
    print(f"savings     : {report.savings_vs_floor:>14.2f}")
    print(f"planner time: {report.planner.solve_seconds:>14.3f} s")
