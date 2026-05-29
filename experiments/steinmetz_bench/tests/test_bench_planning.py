"""Tests for the planning benchmark (item 2.2, Steinmetz §8.4.2).

The benchmark pits zap's gradient expansion planner against an independent joint
multi-scenario expansion LP/QP whose optimum is the true global optimum ``f*`` (every
lever scales only a bound, so the expansion is jointly convex). These tests re-derive
the optimality gap straight from the two solves rather than trusting the report,
confirm the expansion actually helps (so the convergence check is non-trivial), and
cross-check the independent baseline against zap's own dispatch at fixed capacities so
the LP is a genuine re-derivation rather than a copy of the headline number.

A small, fast config (few hours, short diminishing-step schedule) keeps the per-item
pytest verify tractable while still converging well inside the tolerance.
"""

import cvxpy as cp
import numpy as np
import pytest

from experiments.steinmetz_bench.experiments.bench_planning import (
    OPT_REL_TOL,
    PARAM_NAMES,
    PlanningConfig,
    ZAP_SOLVER,
    build_expansion_lp,
    make_scenarios,
    run,
    run_planning_benchmark,
    solve_baseline,
)
from experiments.steinmetz_bench.reports import read_markdown

# Fast convergence: hours=4 with a three-phase shrinking step lands ~2.5e-3 above f*
# in well under a minute, leaving comfortable margin under OPT_REL_TOL (1e-2).
_TEST_CFG = PlanningConfig(hours=4)
_TEST_SCHEDULE = ((0.08, 30), (0.025, 30), (0.008, 20))


@pytest.fixture(scope="module")
def report():
    return run_planning_benchmark(cfg=_TEST_CFG, schedule=_TEST_SCHEDULE)


def test_planner_reaches_global_optimum(report):
    # The LP is the global optimum f*; the planner is a feasible point, so the only
    # way its objective can sit within tol of f* is genuine convergence.
    assert report.planner.best_obj <= report.baseline_obj * (1.0 + OPT_REL_TOL)

    # Re-derive the gap from the raw objectives rather than trusting the property.
    denom = max(abs(report.baseline_obj), 1.0)
    recomputed = (report.planner.best_obj - report.baseline_obj) / denom
    assert recomputed == report.optimality_gap
    assert report.optimality_gap < OPT_REL_TOL
    # A feasible planner cannot beat the global optimum by more than solver noise.
    assert report.optimality_gap > -OPT_REL_TOL


def test_expansion_is_beneficial(report):
    # The no-expansion start (all caps at floor) must cost strictly more than the
    # optimum, otherwise "converging to f*" would be a trivial corner solution and the
    # optimality check would prove nothing.
    assert report.baseline_obj < report.planner.initial_obj
    # The planner must have actually moved off the floor toward that optimum.
    assert report.planner.best_obj < report.planner.initial_obj
    assert report.savings_vs_floor > 0.0


def test_three_levers_multiscenario_and_converged(report):
    assert report.cfg.n_scenarios >= 2
    assert set(report.planner.best_caps) == set(PARAM_NAMES) == {
        "gen_cap", "line_cap", "battery_cap",
    }
    floors = report.cfg.floors
    uppers = report.cfg.uppers
    for name, cap in report.planner.best_caps.items():
        assert floors[name] - 1e-6 <= cap <= uppers[name] + 1e-6
    # At least one lever sits strictly interior (not pinned to a bound), so the
    # planner solved a real optimization rather than racing to a corner.
    interior = [
        name for name, cap in report.planner.best_caps.items()
        if cap > floors[name] + 1.0 and cap < uppers[name] - 1.0
    ]
    assert interior
    # Iteration budget is fixed and finite.
    assert report.planner.n_iterations == sum(n for _, n in _TEST_SCHEDULE)


def test_baseline_matches_zap_dispatch_at_fixed_caps():
    """Independent check: the LP's per-scenario operation cost equals zap's own
    dispatch at the same (floor) capacities — the baseline is re-derived, not copied."""
    cfg = _TEST_CFG
    scenarios = make_scenarios(cfg)

    # LP operation cost per scenario with capacities frozen at the floors.
    prob, _, op_costs = build_expansion_lp(scenarios, cfg, fixed_caps=cfg.floors)
    prob.solve(solver=cp.HIGHS if "HIGHS" in cp.installed_solvers() else cp.CLARABEL)
    assert prob.status in ("optimal", "optimal_inaccurate")

    for (net, devices, _), op_expr in zip(scenarios, op_costs):
        # zap dispatch at the same (floor) capacities: operation cost only.
        out = net.dispatch(devices, time_horizon=cfg.hours, solver=ZAP_SOLVER)
        zap_op = float(out.problem.value)
        lp_op = float(op_expr.value)
        rel = abs(zap_op - lp_op) / max(abs(zap_op), 1.0)
        assert rel < OPT_REL_TOL


def test_solve_baseline_is_feasible_and_interior():
    cfg = _TEST_CFG
    scenarios = make_scenarios(cfg)
    obj, caps, solver = solve_baseline(scenarios, cfg)
    assert np.isfinite(obj) and obj > 0.0
    assert solver in cp.installed_solvers() or solver == "CLARABEL"
    for name, cap in caps.items():
        assert cfg.floors[name] - 1e-6 <= cap <= cfg.uppers[name] + 1e-6


def test_emits_reparseable_bench_result(report, tmp_path):
    result = report.to_bench_result()
    assert result.experiment_id == "2.2-planning"
    assert result.units == "$"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "joint-expansion-lp"
    # Headline is the planner's converged objective.
    assert result.headline_number == report.planner.best_obj
    assert result.sensitivities["planner_objective"] == report.planner.best_obj
    assert result.sensitivities["baseline_objective"] == report.baseline_obj
    # Timing is recorded and real.
    assert result.sensitivities["planner_solve_seconds"] > 0.0

    md_path = tmp_path / "bench_planning.md"
    result.write_markdown(md_path)
    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()


def test_run_writes_report(tmp_path):
    md_path = tmp_path / "planning.md"
    result = run(report_path=md_path, cfg=_TEST_CFG, schedule=_TEST_SCHEDULE)
    assert md_path.exists()
    assert result.headline_number < result.sensitivities["initial_objective"]
