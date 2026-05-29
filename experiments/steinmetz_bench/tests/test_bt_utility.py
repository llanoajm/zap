"""Tests for the vertically-integrated utility backtest (item 3.3).

The backtest compares coordinated SCED least-cost dispatch to a deliberately
uncoordinated (islanded) "actual" dispatch, ranks 5-year expansion projects by NPV
of avoided fuel, and validates the SCED LMP against PyPSA. These tests re-derive the
headline avoided-fuel and NPV numbers from the per-solve arrays (not just read summaries
back), confirm SCED can only be cheaper than the actual dispatch, confirm the binding
corridor is the top-ranked expansion, check the PyPSA roundtrip gap is within tolerance,
and check the human --real path blocks via ``DataNotStagedError``.
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_utility import (
    LMP_GAP_TOL,
    UtilityConfig,
    _npv,
    load_staged_utility,
    run,
    run_utility,
)
from experiments.steinmetz_bench.reports import read_markdown


def test_sced_is_at_most_actual():
    """Least-cost SCED can never beat the islanded actual dispatch, per scenario."""
    res = run_utility()
    gap = res.gap
    assert np.all(gap.sced_daily <= gap.actual_daily + 1e-6)
    # The coordination is genuinely worth something on this congested fleet.
    assert gap.headline_avoided_per_yr > 0.0


def test_avoided_fuel_rederivable_from_costs():
    config = UtilityConfig()
    res = run_utility(config)
    gap = res.gap
    expected = (gap.actual_daily - gap.sced_daily) * config.days_per_year
    np.testing.assert_allclose(gap.avoided_per_yr, expected)
    assert gap.headline_avoided_per_yr == pytest.approx(float(expected.mean()))


def test_expansion_ranking_sorted_and_binding_corridor_wins():
    res = run_utility()
    projects = res.projects
    assert len(projects) >= 1
    npvs = [p.npv for p in projects]
    assert npvs == sorted(npvs, reverse=True)
    # The thermal-limited zone1-zone2 corridor is what SCED pays to work around, so
    # expanding it must deliver the most value.
    assert projects[0].name == "expand-z12"
    binding = next(p for p in projects if p.name == "expand-z12")
    assert binding.avoided_per_yr > 0.0


def test_npv_rederivable_from_avoided_fuel():
    config = UtilityConfig()
    res = run_utility(config)
    for p in res.projects:
        expected = _npv(
            p.avoided_per_yr, p.capital_cost, config.expansion_years, config.discount_rate
        )
        assert p.npv == pytest.approx(expected)
    assert res.npv_delta == pytest.approx(res.best_project.npv)


def test_pypsa_roundtrip_within_tol():
    res = run_utility()
    assert res.fidelity.max_abs_gap < LMP_GAP_TOL
    assert res.fidelity.reference == "pypsa-dc"
    assert res.fidelity.metric == "lmp"
    # Objective parity is a second, independent validation of the same solve.
    assert res.pypsa_cost_rel_gap < 1e-2


def test_bootstrap_ci_brackets_a_positive_avoided_fuel():
    res = run_utility()
    ci = res.gap.ci
    assert ci.lo <= ci.mid <= ci.hi
    assert ci.lo > 0.0


def test_islanding_feasibility_guard_rejects_undersized_fleet():
    with pytest.raises(ValueError):
        UtilityConfig(gen_caps=(200.0, 150.0, 50.0))  # zone 2 cannot self-supply its peak


def test_is_deterministic():
    a = run_utility(UtilityConfig(seed=3))
    b = run_utility(UtilityConfig(seed=3))
    np.testing.assert_array_equal(a.gap.avoided_per_yr, b.gap.avoided_per_yr)
    assert [p.name for p in a.projects] == [p.name for p in b.projects]


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "utility.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "3.3-utility-sced"
    assert result.units == "$/yr"
    assert result.ci is not None
    assert result.fidelity_band is not None
    assert result.headline_number > 0.0
    assert "npv_delta_best_project_usd" in result.sensitivities
    assert result.sensitivities["best_project"] == "expand-z12"
    assert len(result.sensitivities["expansion_ranking"]) >= 1

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()


def test_real_path_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_staged_utility("definitely_not_staged_utility_xyz")
    assert "definitely_not_staged_utility_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()
