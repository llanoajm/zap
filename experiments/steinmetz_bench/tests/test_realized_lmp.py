"""Tests for the realized-LMP comparator (item 1.3).

The comparator differences zap's modeled LMPs against a realized ``price_frame``
and reports a per-node/hour error distribution. These tests re-derive the
distribution from the raw aligned arrays (not just read the band back), confirm it
is non-degenerate (the seeded perturbation actually moved prices), check the
frame<->array alignment round-trips, and assert the missing-cache path blocks via
``DataNotStagedError`` instead of failing.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.realized_lmp import (
    align_frame_to_array,
    compare,
    load_realized_frame,
    price_frame_from_array,
    run,
    run_realized,
    run_synthetic,
)
from experiments.steinmetz_bench.reports import read_markdown


def test_synthetic_emits_error_distribution():
    comp = run_synthetic()

    # Re-derive the distribution straight from the aligned arrays.
    abs_err = np.abs(comp.zap_lmp - comp.realized_lmp)
    assert comp.mean_abs_error == pytest.approx(float(abs_err.mean()))
    assert comp.median_abs_error == pytest.approx(float(np.median(abs_err)))
    assert comp.p90_abs_error == pytest.approx(float(np.percentile(abs_err, 90.0)))
    assert comp.max_abs_error == pytest.approx(float(abs_err.max()))

    # mean <= p90 <= max is the basic shape of an absolute-error distribution.
    assert comp.mean_abs_error <= comp.p90_abs_error <= comp.max_abs_error
    # Bootstrap CI on the mean brackets a positive value.
    assert comp.ci.lo <= comp.ci.mid <= comp.ci.hi
    assert comp.ci.lo > 0.0


def test_synthetic_comparison_is_non_degenerate():
    """A zero-error comparison would be uninformative; the perturbation must bite."""
    comp = run_synthetic()
    assert comp.mean_abs_error > 1.0
    # The realized world genuinely differs from the modeled one at most nodes.
    per_node = np.abs(comp.error).mean(axis=1)
    assert np.count_nonzero(per_node > 1e-6) >= comp.zap_lmp.shape[0] - 1


def test_synthetic_is_deterministic():
    a = run_synthetic(seed=3, realized_seed=4)
    b = run_synthetic(seed=3, realized_seed=4)
    np.testing.assert_array_equal(a.error, b.error)


def test_price_frame_round_trips():
    lmp = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    frame = price_frame_from_array(lmp, idx)

    assert frame.shape == (3, 2)  # snapshot x node
    assert list(frame.columns) == [0, 1]
    np.testing.assert_array_equal(align_frame_to_array(frame, 2, idx), lmp)


def test_align_rejects_incomplete_frame():
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    frame = pd.DataFrame({0: [1.0, 2.0]}, index=idx)  # missing node 1
    with pytest.raises(ValueError):
        align_frame_to_array(frame, 2, idx)


def test_compare_zero_error_on_identical_arrays():
    lmp = np.array([[5.0, 6.0], [7.0, 8.0]])
    comp = compare(lmp, lmp.copy(), source="synthetic")
    assert comp.mean_abs_error == 0.0
    assert comp.max_abs_error == 0.0


def test_missing_cache_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_realized_frame("definitely_not_staged_iso_xyz")
    assert "definitely_not_staged_iso_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()


def test_run_realized_missing_cache_blocks():
    """The human --real entry point blocks cleanly, not with a test failure."""
    zap_lmp = np.zeros((3, 4))
    idx = pd.date_range("2025-01-01", periods=4, freq="h")
    with pytest.raises(DataNotStagedError):
        run_realized("definitely_not_staged_iso_xyz", zap_lmp, idx)


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "realized_lmp.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "1.3-realized-lmp"
    assert result.units == "$/MWh"
    assert result.ci is not None
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "realized-lmp"
    assert result.fidelity_band.metric == "lmp"
    # Headline is the mean abs error and matches the band it was taken from.
    assert result.headline_number == result.fidelity_band.mean_abs_gap
    assert "p90_abs_error" in result.sensitivities
    assert "median_abs_error" in result.sensitivities

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()
