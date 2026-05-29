"""Tests for the scoring harness: CI ordering, monotone duration curve, gaps."""

import numpy as np
import pytest

from experiments.steinmetz_bench.scoring import (
    bootstrap_ci,
    counterfactual_delta,
    duration_curve,
    fidelity_band,
)


def test_bootstrap_ci_is_ordered():
    rng = np.random.default_rng(0)
    samples = rng.normal(10.0, 2.0, size=500)
    ci = bootstrap_ci(samples, seed=1)
    assert ci.lo <= ci.mid <= ci.hi
    # The interval must bracket the true mean for a well-behaved sample.
    assert ci.lo < 10.0 < ci.hi
    lo, mid, hi = ci.as_tuple()
    assert (lo, mid, hi) == (ci.lo, ci.mid, ci.hi)


def test_bootstrap_ci_is_deterministic_given_seed():
    samples = np.linspace(0.0, 1.0, 50)
    a = bootstrap_ci(samples, seed=42)
    b = bootstrap_ci(samples, seed=42)
    assert a.as_tuple() == b.as_tuple()


def test_bootstrap_ci_widens_with_confidence():
    rng = np.random.default_rng(3)
    samples = rng.normal(0.0, 1.0, size=300)
    narrow = bootstrap_ci(samples, confidence=0.80, seed=5)
    wide = bootstrap_ci(samples, confidence=0.99, seed=5)
    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_duration_curve_is_monotone_non_increasing():
    rng = np.random.default_rng(7)
    values = rng.normal(50.0, 15.0, size=200)
    dc = duration_curve(values)

    diffs = np.diff(dc.value)
    assert np.all(diffs <= 1e-12)
    # Exceedance runs from 1/n up to exactly 1.0.
    assert dc.exceedance[-1] == pytest.approx(1.0)
    assert dc.exceedance[0] == pytest.approx(1.0 / values.size)
    # The curve is a permutation of the inputs (nothing dropped or invented).
    np.testing.assert_array_equal(np.sort(dc.value), np.sort(values))


def test_duration_curve_percentile_matches_numpy():
    values = np.arange(0.0, 100.0)
    dc = duration_curve(values)
    assert dc.percentile(90) == pytest.approx(np.percentile(values, 90))


def test_counterfactual_delta_scalar_and_array():
    assert counterfactual_delta(100.0, 70.0) == pytest.approx(30.0)
    out = counterfactual_delta([5.0, 2.0], [1.0, 4.0])
    np.testing.assert_allclose(out, np.array([4.0, -2.0]))


def test_fidelity_band_cross_checks_against_numpy():
    rng = np.random.default_rng(11)
    dc = rng.normal(0.0, 1.0, size=64)
    ref = dc + rng.normal(0.0, 0.05, size=64)
    band = fidelity_band(dc, ref, reference="pypsa-dc", metric="lmp", units="$/MWh")

    gap = np.abs(dc - ref)
    assert band.max_abs_gap == pytest.approx(gap.max())
    assert band.mean_abs_gap == pytest.approx(gap.mean())
    assert band.p90_abs_gap == pytest.approx(np.percentile(gap, 90.0))
    assert band.n == 64
    assert band.reference == "pypsa-dc"
    assert band.to_dict()["metric"] == "lmp"


def test_fidelity_band_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fidelity_band([1.0, 2.0], [1.0], reference="r", metric="m")
