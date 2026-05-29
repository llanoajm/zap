"""Tests for the Mexico EPC dual-regime corridor backtest (item 3.5).

The backtest dispatches one synthetic two-hub network under two rulebooks — historical
merit order and a CFE >= 54%-share mandate — ranks corridors by their adjoint/FD marginal
value of capacity in each, and emits a ranking-agreement ``BenchResult``. These tests
confirm both regimes produce rankings, the mandate is demonstrably binding (CFE share,
cost, prices and dispatch all shift measurably), the merit-order adjoint agrees with a
finite difference (so the ranking signal is certified, not asserted), the CFE-regime
marginal value is itself a real solve-derived quantity, the binding pattern flips between
regimes, the agreement metric re-derives from the stored vectors, and the human --real path
blocks.
"""

import copy

import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_mexico_epc import (
    BINDING_MV_TOL,
    CFE_SHARE_FLOOR,
    HOURS_PER_YEAR,
    MexicoConfig,
    _marginal_value_fd,
    build_zap_devices,
    load_staged_mexico,
    run,
    run_mexico,
    solve_regime,
)
from experiments.steinmetz_bench.experiments.bt_transmission_audit import spearman
from experiments.steinmetz_bench.reports import read_markdown


@pytest.fixture(scope="module")
def result():
    """Run the (deterministic) dual-regime backtest once for the whole module."""
    return run_mexico()


def test_both_regimes_produce_rankings(result):
    """Each regime must yield a full corridor ranking (a permutation of all corridors)."""
    n = result.config.n_corridors
    assert sorted(result.merit_ranking) == list(range(n))
    assert sorted(result.cfe_ranking) == list(range(n))
    # The jurisdiction rule must actually reorder the corridors — otherwise it is a no-op.
    assert result.merit_ranking != result.cfe_ranking


def test_cfe_mandate_is_binding(result):
    """The mandate must lift CFE from a below-floor merit share and visibly move the system."""
    cfg = result.config
    # Merit order under-dispatches CFE; the mandate raises it to the floor.
    assert result.merit_solve.cfe_share < cfg.cfe_share_floor - 1e-6
    assert result.cfe_solve.cfe_share >= cfg.cfe_share_floor - 1e-4
    assert result.mandate_binding
    # A binding mandate has a strictly positive shadow price (its $/MW marginal cost).
    assert result.cfe_solve.share_dual > 0.0
    # Merit order leaves the mandate constraint absent, so no shadow price there.
    assert result.merit_solve.share_dual == 0.0
    # Measurable shift across regimes: cost rises, prices move, CFE generation grows.
    assert result.cost_increase > 1.0
    assert result.max_price_shift > 0.1
    assert result.cfe_generation_increase > 1.0


def test_ranking_agreement_metric_emitted_and_rederivable(result):
    """The headline agreement must recompute from the stored marginal-value vectors."""
    agreement = spearman(result.merit.marginal_value, result.cfe_mv)
    assert result.ranking_agreement == pytest.approx(agreement)
    assert -1.0 <= result.ranking_agreement <= 1.0
    # The rule reorders corridor priorities, so agreement is far from a perfect 1.0.
    assert result.ranking_agreement < 0.5


def test_merit_adjoint_agrees_with_finite_difference(result):
    """zap's merit-order marginal value must match a central FD on every active corridor."""
    assert result.merit.max_rel_err_fd < 1e-2
    band = result.merit.fidelity()
    assert band.reference == "finite-difference"
    assert band.metric == "corridor-marginal-value"
    assert band.n >= 1
    assert band.max_abs_gap < 1.0  # $/MW-period; gradients are O(1-10)


def test_merit_marginal_value_is_solve_derived(result):
    """Recompute a merit corridor's marginal value by an independent FD re-solve."""
    cfg = result.config
    net, devices = build_zap_devices(cfg)
    fd = _marginal_value_fd(net, devices, cfe_share=None, eps=cfg.fd_eps)
    # The independent FD must reproduce zap's adjoint-derived ranking signal.
    np.testing.assert_allclose(fd, result.merit.marginal_value, atol=1e-2)
    # And at least one private corridor carries real, positive marginal value (it binds).
    assert fd.max() > BINDING_MV_TOL


def test_binding_pattern_flips_between_regimes(result):
    """Private corridors bind under merit; CFE corridors bind only once the mandate is on."""
    cfg = result.config
    is_cfe = cfg.is_cfe_corridor
    merit_mv = result.merit.marginal_value
    cfe_mv = result.cfe_mv

    # Under merit order the cheap private corridors bind and the CFE corridors do not.
    assert merit_mv[~is_cfe].max() > BINDING_MV_TOL
    assert np.all(np.abs(merit_mv[is_cfe]) < BINDING_MV_TOL)
    # The mandate forces energy through the CFE corridors, so at least one now binds.
    assert cfe_mv[is_cfe].max() > BINDING_MV_TOL


def test_solve_regime_enforces_share(result):
    """``solve_regime`` honors the mandate when asked and leaves it off otherwise."""
    cfg = result.config
    net, devices = build_zap_devices(cfg)
    free = solve_regime(net, devices, cfe_share=None)
    forced = solve_regime(net, devices, cfe_share=cfg.cfe_share_floor)
    assert free.cfe_share < cfg.cfe_share_floor - 1e-6
    assert forced.cfe_share >= cfg.cfe_share_floor - 1e-4
    # Enforcing a costlier dispatch cannot reduce total system cost.
    assert forced.cost >= free.cost - 1e-6


def test_bench_result_headline_and_reparses(result, tmp_path):
    md_path = tmp_path / "mexico.md"
    res = run(report_path=md_path)

    assert res.experiment_id == "3.5-mexico-epc"
    assert res.units == "spearman"
    assert res.ci is not None
    assert res.fidelity_band is not None
    assert res.headline_number == pytest.approx(result.ranking_agreement)
    assert -1.0 <= res.headline_number <= 1.0
    assert res.sensitivities["mandate_binding"] is True
    assert res.sensitivities["cfe_share_merit"] < CFE_SHARE_FLOOR

    n = result.config.n_corridors
    relief_merit = res.sensitivities["congestion_relief_per_mw_year_merit"]
    relief_cfe = res.sensitivities["congestion_relief_per_mw_year_cfe"]
    assert len(relief_merit) == n and len(relief_cfe) == n
    # The annualized relief is the per-period marginal value scaled to a full year.
    np.testing.assert_allclose(
        relief_merit,
        result.merit.marginal_value * (HOURS_PER_YEAR / result.config.hours),
    )

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == res.to_dict()


def test_bootstrap_ci_brackets_the_agreement(result):
    ci = result.ci
    assert ci.lo <= ci.mid <= ci.hi
    assert -1.0 <= ci.lo <= 1.0 and -1.0 <= ci.hi <= 1.0


def test_is_deterministic():
    a = run_mexico()
    b = run_mexico()
    np.testing.assert_array_equal(a.merit.marginal_value, b.merit.marginal_value)
    np.testing.assert_array_equal(a.cfe_mv, b.cfe_mv)
    assert a.ranking_agreement == b.ranking_agreement


def test_config_guards():
    with pytest.raises(ValueError):
        MexicoConfig(zone_loads=(70.0, 90.0), cap_private=(45.0,), cap_cfe=(50.0, 55.0))
    with pytest.raises(ValueError):
        # Zone 0 unreachable: cap_private + cap_cfe < load.
        MexicoConfig(
            zone_loads=(200.0, 90.0, 110.0, 130.0),
            cap_private=(45.0, 50.0, 55.0, 70.0),
            cap_cfe=(50.0, 55.0, 60.0, 65.0),
        )
    with pytest.raises(ValueError):
        MexicoConfig(private_cost=30.0, cfe_cost=10.0)  # CFE must be the dearer hub
    with pytest.raises(ValueError):
        # sum(cap_cfe) cannot deliver the mandated CFE share.
        MexicoConfig(cap_cfe=(5.0, 5.0, 5.0, 5.0))


def test_real_path_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_staged_mexico("definitely_not_staged_mexico_xyz")
    assert "definitely_not_staged_mexico_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()


def test_marginal_value_fd_handles_deepcopy_isolation(result):
    """FD must not mutate the caller's device list (deepcopy isolation)."""
    cfg = result.config
    net, devices = build_zap_devices(cfg)
    before = np.asarray(copy.deepcopy(devices[2].nominal_capacity), float).copy()
    _marginal_value_fd(net, devices, cfe_share=cfg.cfe_share_floor, eps=cfg.fd_eps)
    after = np.asarray(devices[2].nominal_capacity, float)
    np.testing.assert_array_equal(before, after)
