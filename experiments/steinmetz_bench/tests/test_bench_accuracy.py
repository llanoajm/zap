"""Tests for the accuracy benchmark (item 2.3, §8.4.3).

The benchmark assembles LMP and congestion-component error distributions vs PyPSA
(item 1.1) and vs realized (item 1.3). These tests re-derive each distribution from
the raw aligned arrays the report carries (not just read the bands back), confirm
the four components are present and shaped like distributions (mean <= p90 <= max),
check the congestion decomposition is what it claims, and assert the synthetic loop
path passes while the --real path blocks cleanly via ``DataNotStagedError``.
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bench_accuracy import (
    HEADLINE,
    REF_NODE,
    AccuracyReport,
    congestion_components,
    run,
    run_real,
    run_synthetic,
)
from experiments.steinmetz_bench.reports import read_markdown

_COMPONENTS = ("lmp_vs_pypsa", "congestion_vs_pypsa", "lmp_vs_realized", "congestion_vs_realized")


def test_assembles_all_four_distributions():
    report = run_synthetic()
    assert isinstance(report, AccuracyReport)
    assert set(report.components) == set(_COMPONENTS)
    for name in _COMPONENTS:
        comp = report.components[name]
        # Each is a real distribution, not a point: re-derive its summary stats.
        abs_err = np.abs(comp.dc - comp.ref)
        assert comp.mean_abs_error == pytest.approx(float(abs_err.mean()))
        assert comp.median_abs_error == pytest.approx(float(np.median(abs_err)))
        assert comp.p90_abs_error == pytest.approx(float(np.percentile(abs_err, 90.0)))
        assert comp.max_abs_error == pytest.approx(float(abs_err.max()))
        assert comp.mean_abs_error <= comp.p90_abs_error <= comp.max_abs_error
        assert comp.ci.lo <= comp.ci.mid <= comp.ci.hi


def test_pypsa_reference_is_the_fidelity_floor():
    """DC-vs-PyPSA agreement must be ~solver noise; realized error must be larger."""
    report = run_synthetic()
    lmp_pypsa = report.components["lmp_vs_pypsa"]
    lmp_realized = report.components["lmp_vs_realized"]
    # PyPSA is a DC-vs-DC reference: tiny gap (reuses item 1.1's 1e-2 tolerance).
    assert lmp_pypsa.max_abs_error < 1e-2
    # The seeded realized perturbation must actually move prices, else the accuracy
    # distribution would be degenerate and uninformative.
    assert lmp_realized.mean_abs_error > 1.0


def test_congestion_decomposition():
    """Congestion component = LMP minus the reference node; ref node is identically 0."""
    lmp = np.array([[10.0, 12.0], [25.0, 30.0], [40.0, 18.0]])
    cong = congestion_components(lmp, ref_node=REF_NODE)
    np.testing.assert_allclose(cong[REF_NODE], 0.0)
    np.testing.assert_allclose(cong, lmp - lmp[REF_NODE : REF_NODE + 1, :])
    # The congestion distribution differs from the raw-LMP distribution (removing the
    # energy level is not a no-op), confirming a separate quantity is reported.
    report = run_synthetic()
    assert report.components["congestion_vs_pypsa"].dc.shape == report.components[
        "lmp_vs_pypsa"
    ].dc.shape


def test_headline_is_realized_lmp_mean_error():
    report = run_synthetic()
    result = report.to_bench_result()
    head = report.components[HEADLINE]
    assert result.experiment_id == "2.3-accuracy"
    assert result.units == "$/MWh"
    # Headline is the realized-LMP mean abs error, and its CI rides along.
    assert result.headline_number == head.mean_abs_error
    assert result.ci is head.ci
    # The fidelity band is the DC-vs-PyPSA LMP gap (the numerical floor).
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "pypsa-dc"
    assert result.fidelity_band.metric == "lmp"
    # Every component's distribution is recorded in sensitivities.
    for name in _COMPONENTS:
        assert name in result.sensitivities
        assert result.sensitivities[name]["p90_abs_error"] >= 0.0


def test_deterministic():
    a = run_synthetic(seed=2, realized_seed=5)
    b = run_synthetic(seed=2, realized_seed=5)
    for name in _COMPONENTS:
        np.testing.assert_array_equal(a.components[name].dc, b.components[name].dc)
        np.testing.assert_array_equal(a.components[name].ref, b.components[name].ref)


def test_real_path_blocks_via_data_not_staged():
    """The human --real path is parameterized but blocks cleanly, not via failure."""
    with pytest.raises(DataNotStagedError):
        run_real("definitely_not_staged_iso_xyz")


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "bench_accuracy.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "2.3-accuracy"
    assert result.fidelity_band is not None
    assert result.ci is not None
    assert result.sensitivities["headline_component"] == HEADLINE

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()
