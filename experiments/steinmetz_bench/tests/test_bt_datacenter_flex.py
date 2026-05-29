"""Tests for the data-center flexibility & battery sizing backtest (item 3.2).

These re-derive every headline straight from the per-solve arrays rather than reading
summaries back: the break-even battery size is recomputed from the marginal-value curve
and shown to coincide with the net-value maximum (so marginal value really equals
marginal cost there); the adjoint marginal value is cross-checked against a finite
difference; the firm-vs-flexible saving is shown to be non-negative scenario-by-scenario
(the firm profile is always feasible for the flexible problem) with a strictly positive
bootstrap CI; the flexible load is confirmed to stay a genuine load (never injecting);
and the human --real path is confirmed to block via ``DataNotStagedError``.
"""

import cvxpy as cp
import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_datacenter_flex import (
    FlexConfig,
    _build_devices,
    _crossing,
    _nominal_scenario,
    load_staged_flex,
    run,
    run_battery_sizing,
    run_flex,
    run_flex_value,
)
from experiments.steinmetz_bench.reports import read_markdown

# The adjoint and the central finite difference of the (QP-smoothed) dispatch should
# agree tightly; observed worst case is ~3e-5.
FD_REL_TOL = 5e-3


def test_marginal_value_is_monotone_non_increasing():
    sizing = run_battery_sizing()
    diffs = np.diff(sizing.daily_marginal_value)
    assert np.all(diffs <= 1e-6)
    # And strictly positive (a battery always has some arbitrage value here).
    assert np.all(sizing.daily_marginal_value > 0.0)


def test_break_even_is_interior_and_rederivable():
    sizing = run_battery_sizing()
    sizes = sizing.sizes
    assert sizes[0] < sizing.break_even_mw < sizes[-1]
    # Recompute the crossing from the public arrays; must match the stored break-even.
    cap = sizing.config.battery_capital_cost_per_mw_yr
    rederived = _crossing(sizes, sizing.annual_marginal_value, cap)
    assert sizing.break_even_mw == pytest.approx(rederived)


def test_marginal_value_crosses_capital_cost():
    """Below break-even an extra MW pays back; above it does not."""
    sizing = run_battery_sizing()
    cap = sizing.config.battery_capital_cost_per_mw_yr
    sizes, mv = sizing.sizes, sizing.annual_marginal_value
    below = sizes < sizing.break_even_mw
    above = sizes > sizing.break_even_mw
    assert np.all(mv[below] >= cap)
    assert np.all(mv[above] <= cap)


def test_break_even_matches_net_value_maximum():
    """Marginal value == marginal cost at the size that maximizes net value."""
    sizing = run_battery_sizing()
    # The net-value-maximizing swept size sits within one grid step of the break-even.
    grid_step = float(np.max(np.diff(sizing.sizes)))
    assert abs(sizing.optimal_size_mw - sizing.break_even_mw) <= grid_step
    # Net value really is maximized at optimal_size_mw (recompute argmax from arrays).
    assert sizing.optimal_size_mw == pytest.approx(
        sizing.sizes[int(np.argmax(sizing.net_value_per_yr))]
    )


def test_gradient_agrees_with_finite_difference():
    sizing = run_battery_sizing()
    assert np.isfinite(sizing.max_rel_err_fd)
    assert sizing.max_rel_err_fd < FD_REL_TOL
    # Re-derive the agreement on every evaluated point, not just the stored summary.
    evaluated = np.isfinite(sizing.fd_marginal_value)
    assert evaluated.sum() >= 3
    rel = np.abs(
        sizing.daily_marginal_value[evaluated] - sizing.fd_marginal_value[evaluated]
    ) / np.abs(sizing.fd_marginal_value[evaluated])
    assert rel.max() < FD_REL_TOL


def test_firm_vs_flexible_savings_are_nonnegative_per_scenario():
    fv = run_flex_value()
    # Flexibility is an option the firm profile could always decline, so it never costs more.
    assert np.all(fv.firm_daily - fv.flex_daily >= -1e-6)
    assert fv.headline_savings > 0.0
    # Annualization and the headline are exactly the per-scenario arithmetic.
    expected_annual = (fv.firm_daily - fv.flex_daily) * fv.config.days_per_year
    np.testing.assert_allclose(fv.annual_savings, expected_annual)
    assert fv.headline_savings == pytest.approx(fv.annual_savings.mean())


def test_firm_vs_flexible_ci_is_positive():
    fv = run_flex_value()
    assert fv.ci.lo <= fv.ci.mid <= fv.ci.hi
    assert fv.ci.lo > 0.0


def test_flexible_load_never_injects():
    """The flexible data center only curtails during expensive hours; it stays a load."""
    config = FlexConfig()
    scenario = _nominal_scenario(config)
    net, devices, _ = _build_devices(config, scenario, flexible=True)
    out = net.dispatch(devices, time_horizon=config.hours, solver=cp.CLARABEL)
    dc_power = np.asarray(out.power[2][0], dtype=float).ravel()  # device 2 = data center
    assert np.all(dc_power <= 1e-6)  # power <= 0 means it withdraws (consumes)
    # It does flex: consumption is strictly below the firm target in at least one hour.
    assert np.any(-dc_power < config.dc_mw - 1e-3)


def test_is_deterministic():
    a = run_flex(FlexConfig(seed=3))
    b = run_flex(FlexConfig(seed=3))
    assert a.sizing.break_even_mw == pytest.approx(b.sizing.break_even_mw)
    np.testing.assert_array_equal(
        a.flex_value.annual_savings, b.flex_value.annual_savings
    )


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "flex.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "3.2-datacenter-flex"
    assert result.units == "$/yr"
    assert result.ci is not None and result.ci.lo > 0.0
    assert result.headline_number > 0.0
    # Battery sizing surfaces as sensitivities and the FD check as the fidelity band.
    assert result.sensitivities["break_even_battery_mw"] > 0.0
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "finite-difference"
    assert result.fidelity_band.max_abs_gap >= 0.0

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()


def test_real_path_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_staged_flex("definitely_not_staged_iso_xyz")
    assert "definitely_not_staged_iso_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()
