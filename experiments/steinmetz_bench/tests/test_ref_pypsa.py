"""Tests for the PyPSA LP roundtrip reference (item 1.1).

The reference solves one bundled radial net in both zap (CLARABEL) and PyPSA
(HiGHS) and reports the nodal LMP gap and line flow gap. These tests re-derive
the gaps from the raw aligned arrays (not just read the band back), confirm the
acceptance tolerances, and check the comparison is non-degenerate — the
congested corridor must actually separate prices, otherwise a trivial
uniform-price solve would make the agreement meaningless.
"""

import numpy as np

from experiments.steinmetz_bench.experiments.ref_pypsa import (
    FLOW_GAP_TOL,
    LMP_GAP_TOL,
    ReferenceSpec,
    run,
    run_reference,
)
from experiments.steinmetz_bench.reports import read_markdown


def test_lmp_and_flow_gaps_within_tolerance():
    comp = run_reference()

    # Re-derive the headline gaps straight from the aligned solve arrays rather
    # than trusting the precomputed bands.
    recomputed_lmp_gap = float(np.abs(comp.zap_lmp - comp.pypsa_lmp).max())
    recomputed_flow_gap = float(np.abs(comp.zap_flow - comp.pypsa_flow).max())

    assert recomputed_lmp_gap == comp.max_lmp_gap
    assert recomputed_flow_gap == comp.max_flow_gap

    assert comp.max_lmp_gap < LMP_GAP_TOL
    assert comp.max_flow_gap < FLOW_GAP_TOL


def test_objectives_agree():
    comp = run_reference()
    assert comp.objective_rel_gap < 1e-2
    # A pure dispatch problem with positive load + costs has positive cost.
    assert comp.zap_objective > 0.0


def test_comparison_is_non_degenerate():
    """The reference must exercise congestion, else the gap is uninformative."""
    comp = run_reference()
    spec = ReferenceSpec()
    load = np.asarray(spec.load_profile)

    # Hours whose load exceeds the binding line capacity must see the price
    # separate between the cheap node (bus0) and the load node (bus2).
    congested_hours = np.where(load > spec.congested_line_snom)[0]
    assert congested_hours.size > 0
    nodal_spread = comp.zap_lmp.max(axis=0) - comp.zap_lmp.min(axis=0)
    assert np.all(nodal_spread[congested_hours] > 1.0)

    # The binding corridor (line 1) carries flow at its capacity during a peak.
    peak_hour = int(np.argmax(load))
    assert abs(comp.zap_flow[1, peak_hour]) > spec.congested_line_snom - 1.0


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "ref_pypsa.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "1.1-pypsa-roundtrip"
    assert result.units == "$/MWh"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "pypsa-dc"
    assert result.fidelity_band.metric == "lmp"
    # Headline is the max LMP gap and matches the band it was taken from.
    assert result.headline_number == result.fidelity_band.max_abs_gap
    assert "max_flow_gap_mw" in result.sensitivities

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()
