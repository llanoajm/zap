"""Tests for the CPU speed benchmark (item 2.1, Steinmetz §8.4.1).

The benchmark times zap's dispatch against an independent CVXPY LP baseline over
several network sizes and certifies they agree on the optimum. These tests
re-derive the objective gap straight from the two solves (not just read the
headline back), confirm the >=3-size timing table is well-formed and real
(positive wall-clock, computed objectives), and check the emitted BenchResult
round-trips.
"""

import numpy as np

from experiments.steinmetz_bench.experiments.bench_speed import (
    OBJECTIVE_GAP_TOL,
    build_baseline_lp,
    run,
    run_speed_benchmark,
)
from experiments.steinmetz_bench.datasets.registry import DatasetSpec, resolve
from experiments.steinmetz_bench.reports import read_markdown

# Small sizes / single repeat keep the per-item pytest verify fast.
_TEST_SIZES = ((4, 6), (8, 6), (12, 6))


def test_objective_parity_across_sizes():
    report = run_speed_benchmark(sizes=_TEST_SIZES, repeats=1)

    assert len(report.rows) >= 3
    for r in report.rows:
        # Re-derive the gap from the raw objectives rather than trusting the row.
        denom = max(abs(r.zap_objective), 1.0)
        recomputed = abs(r.zap_objective - r.baseline_objective) / denom
        assert recomputed == r.objective_gap
        assert r.objective_gap < OBJECTIVE_GAP_TOL
        # A dispatch with positive load + costs has positive cost on both paths.
        assert r.zap_objective > 0.0
        assert r.baseline_objective > 0.0

    assert report.max_objective_gap < OBJECTIVE_GAP_TOL


def test_timing_table_is_real():
    report = run_speed_benchmark(sizes=_TEST_SIZES, repeats=1)
    for r in report.rows:
        # Wall-clock must be a measured, positive number for both paths.
        assert r.zap_s > 0.0
        assert r.baseline_s > 0.0
        assert np.isfinite(r.speedup)


def test_baseline_matches_zap_on_one_solve():
    """Independently solve one size and confirm the LP objective equals zap's."""
    import cvxpy as cp

    from experiments.steinmetz_bench.experiments.bench_speed import ZAP_SOLVER

    ds = resolve(DatasetSpec(name="speed-check", kind="synthetic",
                             n_nodes=10, hours=8, congested=True, seed=3))
    horizon = ds.time_horizon

    out = ds.network.dispatch(ds.devices, time_horizon=horizon, solver=ZAP_SOLVER)
    prob, _ = build_baseline_lp(ds.network, ds.devices, horizon)
    prob.solve(solver=cp.HIGHS)

    rel = abs(float(out.problem.value) - float(prob.value)) / max(abs(float(out.problem.value)), 1.0)
    assert rel < OBJECTIVE_GAP_TOL


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "bench_speed.md"
    result = run(report_path=md_path, sizes=_TEST_SIZES, repeats=1)

    assert result.experiment_id == "2.1-speed-cpu"
    assert result.units == "relative"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "cvxpy-lp"
    assert result.fidelity_band.metric == "objective"
    # Headline is the worst objective gap and is below tolerance.
    assert result.headline_number < OBJECTIVE_GAP_TOL
    assert result.headline_number == result.sensitivities["max_objective_gap"]

    table = result.sensitivities["timing_table"]
    assert len(table) == len(_TEST_SIZES)
    for entry in table:
        assert {"zap_s", "baseline_s", "zap_objective", "baseline_objective"} <= entry.keys()

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()
