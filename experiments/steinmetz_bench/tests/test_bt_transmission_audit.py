"""Tests for the transmission-plan audit backtest (item 3.4).

The backtest ranks corridors ex-ante by zap's adjoint marginal value of capacity, audits
that ranking against an independently-computed realized congestion-rent vector, and emits
an R^2 ``BenchResult``. These tests confirm the known bottleneck ranks #1, the rank
correlation clears the acceptance threshold, the agreement metrics re-derive from the
stored vectors, the adjoint gradient agrees with a finite difference (so the ranking
signal is certified, not asserted), the congestion rent is a real solve-derived quantity,
the rank-correlation helpers are correct, and the human --real path blocks.
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_transmission_audit import (
    HIGH_VALUE_FRAC,
    RANK_CORR_TOL,
    AuditConfig,
    _congestion_rent,
    _dispatch,
    _line_index,
    _nominal_scenario,
    build_zap_devices,
    load_staged_audit,
    r_squared,
    run,
    run_audit,
    spearman,
)
from experiments.steinmetz_bench.reports import read_markdown


def test_known_bottleneck_ranks_first():
    """The most-throttled, dearest-backup corridor must top the ex-ante ranking."""
    res = run_audit()
    assert res.bottleneck_identified
    assert res.top_corridor == res.config.bottleneck_zone


def test_rank_correlation_exceeds_threshold():
    res = run_audit()
    assert res.rank_corr > RANK_CORR_TOL
    # Forecast-vs-realized noise should keep it strictly below a perfect 1.0 R^2.
    assert res.r2 < 1.0


def test_agreement_metrics_rederivable_from_vectors():
    """rank correlation and R^2 must recompute from the stored ex-ante/realized vectors."""
    res = run_audit()
    mv = res.ex_ante.marginal_value
    rent = res.realized.mean_rent
    assert res.rank_corr == pytest.approx(spearman(mv, rent))
    assert res.r2 == pytest.approx(r_squared(mv, rent))
    # R^2 is the Pearson coefficient squared, so it lives in [0, 1].
    assert 0.0 <= res.r2 <= 1.0


def test_adjoint_agrees_with_finite_difference():
    """zap's marginal-value gradient must match a central FD on every congested corridor."""
    res = run_audit()
    assert res.ex_ante.max_rel_err_fd < 1e-2
    band = res.fidelity()
    assert band.reference == "finite-difference"
    assert band.metric == "corridor-marginal-value"
    assert band.n >= 1
    assert band.max_abs_gap < 1.0  # $/MW-day, gradients are O(1e3)


def test_uncongested_corridors_carry_no_marginal_value():
    """An over-provisioned spoke saves nothing when expanded (zero marginal value)."""
    res = run_audit()
    mv = res.ex_ante.marginal_value
    active = res.ex_ante.active_mask
    # Inactive (uncongested) corridors have ~zero marginal value; active ones are positive.
    assert np.all(mv[active] > 0.0)
    assert np.all(np.abs(mv[~active]) < 1.0)


def test_congestion_rent_is_solve_derived_and_physical():
    """Recompute one day's rent straight from prices/flows; binding corridors are positive."""
    config = AuditConfig()
    scenario = _nominal_scenario(config)
    net, devices = build_zap_devices(config, scenario)
    out = _dispatch(net, devices)
    rent = _congestion_rent(config, out, devices)

    # Independent recomputation from the solved nodal prices and line flows.
    li = _line_index(devices)
    line = devices[li]
    prices = np.asarray(out.prices, float)
    flow = np.asarray(out.power[li][1], float)
    src = np.asarray(line.source_terminal, int)
    snk = np.asarray(line.sink_terminal, int)
    expected = np.sum(np.abs((prices[snk] - prices[src]) * flow), axis=1)
    np.testing.assert_allclose(rent, expected)

    # The binding bottleneck corridor must carry positive rent; the most over-provisioned
    # spoke (cap well above load) carries essentially none.
    assert rent[config.bottleneck_zone] > 0.0
    assert rent[0] == pytest.approx(0.0, abs=1.0)


def test_missed_corridor_count_zero_on_clean_synthetic():
    res = run_audit()
    assert res.missed_corridor_count == 0
    # Sanity: the audit definition is the one the result reports.
    mv = res.ex_ante.marginal_value
    rent = res.realized.mean_rent
    high_value = mv >= HIGH_VALUE_FRAC * mv.max()
    high_rent = rent >= HIGH_VALUE_FRAC * rent.max()
    assert int(np.sum(high_value & ~high_rent)) == 0


def test_spearman_and_r2_helpers():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    # Perfectly monotone (nonlinear) -> Spearman 1, R^2 < 1.
    b = np.array([1.0, 4.0, 9.0, 16.0])
    assert spearman(a, b) == pytest.approx(1.0)
    assert r_squared(a, b) < 1.0
    # Perfect line -> both 1.
    c = 2.0 * a + 5.0
    assert spearman(a, c) == pytest.approx(1.0)
    assert r_squared(a, c) == pytest.approx(1.0)
    # Reversed -> Spearman -1.
    assert spearman(a, a[::-1]) == pytest.approx(-1.0)
    # Constant vector -> defined as zero correlation (no variance).
    assert spearman(a, np.ones(4)) == pytest.approx(0.0)


def test_bench_result_headline_is_r2_and_reparses(tmp_path):
    md_path = tmp_path / "audit.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "3.4-transmission-audit"
    assert result.units == "R2"
    assert result.ci is not None
    assert result.fidelity_band is not None
    assert result.headline_number == pytest.approx(run_audit().r2)
    assert 0.0 <= result.headline_number <= 1.0
    assert result.sensitivities["missed_corridor_count"] == 0
    assert result.sensitivities["bottleneck_identified"] is True
    assert result.sensitivities["rank_correlation_spearman"] > RANK_CORR_TOL

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()


def test_bootstrap_ci_brackets_the_headline_r2():
    res = run_audit()
    ci = res.ci
    assert ci.lo <= ci.mid <= ci.hi
    assert ci.lo <= res.r2 <= ci.hi or ci.lo == pytest.approx(res.r2, abs=0.1)
    assert ci.lo > 0.0


def test_config_guards():
    with pytest.raises(ValueError):
        AuditConfig(backup_costs=(40.0, 80.0), spoke_caps=(50.0,))  # length mismatch
    with pytest.raises(ValueError):
        AuditConfig(spoke_caps=(200.0, 200.0, 200.0, 200.0, 200.0, 200.0))  # nothing binds
    with pytest.raises(ValueError):
        AuditConfig(backup_cap=10.0)  # backup cannot cover its own peak load


def test_is_deterministic():
    a = run_audit(AuditConfig(seed=5))
    b = run_audit(AuditConfig(seed=5))
    np.testing.assert_array_equal(a.realized.mean_rent, b.realized.mean_rent)
    np.testing.assert_array_equal(a.ex_ante.marginal_value, b.ex_ante.marginal_value)
    assert a.r2 == b.r2


def test_real_path_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_staged_audit("definitely_not_staged_audit_xyz")
    assert "definitely_not_staged_audit_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()
