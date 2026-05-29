"""Tests for the gradient-vs-exact-dual check (item 1.2, Steinmetz §8.4.4).

These tests certify that zap's adjoint sensitivities match the exact gradient implied
by LP duality (the envelope theorem) for line capacity, generator capacity, and battery
power. Every number is computed from real CLARABEL solves: the adjoint comes from zap's
KKT backward pass, the reference from the device Lagrangian evaluated with the solver's
duals, and a third anchor from re-solving under finite-difference perturbations. The
tests re-derive the agreement from the raw gradient arrays rather than trusting the
precomputed error scalars, and guard against the degenerate case where every gradient is
zero (which would make the relative-error test vacuous).
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.experiments.grad_check import (
    EXPERIMENT_ID,
    FD_REL_TOL,
    GRAD_REL_TOL,
    run,
    run_grad_check,
)
from experiments.steinmetz_bench.reports import read_markdown


@pytest.fixture(scope="module")
def report():
    return run_grad_check(do_fd=True)


def test_all_three_device_types_covered(report):
    types = {c.device_type for c in report.checks}
    assert {"line", "generator", "battery"} <= types


def test_adjoint_matches_exact_dual_for_every_device_type(report):
    per_type = report.per_device_type()
    # The headline acceptance: worst-case relative error under tolerance per device type.
    for device_type in ("line", "generator", "battery"):
        assert per_type[device_type] < GRAD_REL_TOL, (
            f"{device_type}: rel err {per_type[device_type]:.2e} >= {GRAD_REL_TOL}"
        )
    assert report.max_rel_err_dual < GRAD_REL_TOL


def test_relative_errors_recomputed_from_raw_arrays(report):
    """Re-derive the relative error straight from the adjoint/dual arrays."""
    for c in report.checks:
        active = c.active_mask
        assert active.any()  # a vacuous (all-inactive) check would prove nothing
        rel = np.abs(c.adjoint[active] - c.exact_dual[active]) / np.abs(c.exact_dual[active])
        assert float(rel.max()) == pytest.approx(c.max_rel_err_dual)
        assert rel.max() < GRAD_REL_TOL


def test_finite_difference_agrees(report):
    """The solver-independent FD re-derivation confirms neither path is self-consistent-only."""
    for c in report.checks:
        assert c.max_rel_err_fd < FD_REL_TOL


def test_gradients_are_nonzero(report):
    """At least one active gradient per device type is materially nonzero."""
    for c in report.checks:
        assert np.abs(c.exact_dual[c.active_mask]).max() > 1e-3


def test_inactive_constraints_have_negligible_gap(report):
    """Where the dual gradient is ~0 (slack constraints), the adjoint must be too."""
    for c in report.checks:
        scale = max(float(np.abs(c.exact_dual[c.active_mask]).max()), 1.0)
        assert c.max_abs_gap_inactive < 1e-3 * scale


def test_generator_satisfies_clean_minus_mu_identity(report):
    """For a generator the only parameter dependence is the capacity inequality, so the
    full envelope gradient equals the capacity-dual (``-mu``) term exactly."""
    gen_checks = [c for c in report.checks if c.device_type == "generator"]
    assert gen_checks
    for c in gen_checks:
        a = c.active_mask
        np.testing.assert_allclose(c.exact_dual[a], c.capacity_dual[a], rtol=1e-6, atol=1e-6)


def test_ac_line_gradient_exceeds_thermal_dual_term(report):
    """An AC line's capacity scales susceptance too, so its exact gradient is strictly
    richer than the thermal ``-mu`` term — at least one network must show the gap."""
    line_checks = [c for c in report.checks if c.device_type == "line"]
    assert line_checks
    gaps = [
        float(np.abs(c.exact_dual[c.active_mask] - c.capacity_dual[c.active_mask]).max())
        for c in line_checks
    ]
    assert max(gaps) > 1e-3


def test_emits_reparseable_bench_result(report, tmp_path):
    md_path = tmp_path / "grad_check.md"
    result = run(report_path=md_path, do_fd=False)

    assert result.experiment_id == EXPERIMENT_ID
    assert result.units == "relative"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "exact-dual"
    assert result.fidelity_band.metric == "cost-gradient"
    assert result.headline_number < GRAD_REL_TOL
    assert set(result.sensitivities["rel_err_by_device_type"]) >= {
        "line", "generator", "battery"
    }

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()
