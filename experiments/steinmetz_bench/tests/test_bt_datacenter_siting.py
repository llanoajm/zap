"""Tests for the data-center siting backtest (item 3.1).

The backtest ranks candidate nodes by their LMP duration curve + curtailment
frequency and quantifies the $/MWh saved by the best node over a default. These
tests re-derive the ranking and the headline delta straight from the per-solve
arrays (not just read summaries back), confirm the deliberately cheap node is
recommended, confirm curtailment is a live signal (zero at the well-supplied node,
positive at the capacity-starved one), and check the human --real path blocks via
``DataNotStagedError`` rather than failing.
"""

import numpy as np
import pytest

from experiments.steinmetz_bench.datasets import DataNotStagedError
from experiments.steinmetz_bench.experiments.bt_datacenter_siting import (
    SitingConfig,
    load_staged_siting,
    run,
    run_siting,
)
from experiments.steinmetz_bench.reports import read_markdown


def test_recommends_the_cheap_node():
    config = SitingConfig()
    res = run_siting(config)
    # The deliberately cheap node (fat tie line to the hub) must win.
    assert res.recommended_node == config.cheap_node
    assert res.ranking()[0] == config.cheap_node
    # The cheap node's mean price is strictly below every other candidate.
    cheap_price = res.nodes[config.cheap_node].effective_price
    for node, siting in res.nodes.items():
        if node != config.cheap_node:
            assert cheap_price < siting.effective_price


def test_headline_delta_matches_effective_prices():
    res = run_siting()
    rec = res.nodes[res.recommended_node].effective_price
    dflt = res.nodes[res.default_node].effective_price
    assert res.headline_delta == pytest.approx(dflt - rec)
    assert res.headline_delta > 0.0
    # Paired-sample mean equals the difference of the two means.
    assert res.delta_samples.mean() == pytest.approx(dflt - rec)


def test_effective_price_rederivable_from_arrays():
    res = run_siting()
    for siting in res.nodes.values():
        assert siting.effective_price == pytest.approx(float(siting.lmp.mean()))


def test_bootstrap_ci_brackets_a_positive_delta():
    res = run_siting()
    assert res.ci.lo <= res.ci.mid <= res.ci.hi
    assert res.ci.lo > 0.0


def test_curtailment_is_a_live_signal():
    """Well-supplied cheap node never curtails; the starved node sometimes does."""
    config = SitingConfig()
    res = run_siting(config)
    assert res.nodes[config.cheap_node].curtailment_frequency == 0.0
    starved = config.n_candidates - 1  # last candidate has the smallest backstop
    assert res.nodes[starved].curtailment_frequency > 0.0
    # Curtailment is derived from the served array, not asserted.
    siting = res.nodes[starved]
    shortfall = siting.requested_mw - siting.served
    expected = float(np.mean(shortfall > 1e-3))
    assert siting.curtailment_frequency == pytest.approx(expected)


def test_duration_curve_is_monotone_non_increasing():
    res = run_siting()
    curve = res.nodes[res.recommended_node].duration_curve
    diffs = np.diff(curve.value)
    assert np.all(diffs <= 1e-9)


def test_is_deterministic():
    a = run_siting(SitingConfig(seed=5))
    b = run_siting(SitingConfig(seed=5))
    assert a.recommended_node == b.recommended_node
    np.testing.assert_array_equal(a.delta_samples, b.delta_samples)


def test_emits_reparseable_bench_result(tmp_path):
    md_path = tmp_path / "siting.md"
    result = run(report_path=md_path)

    assert result.experiment_id == "3.1-datacenter-siting"
    assert result.units == "$/MWh"
    assert result.ci is not None
    assert result.headline_number > 0.0
    assert result.sensitivities["recommended_node"] == SitingConfig().cheap_node
    assert "ranking" in result.sensitivities
    assert "per_node" in result.sensitivities

    assert md_path.exists()
    assert read_markdown(md_path).to_dict() == result.to_dict()


def test_real_path_blocks_via_data_not_staged():
    with pytest.raises(DataNotStagedError) as exc:
        load_staged_siting("definitely_not_staged_iso_xyz")
    assert "definitely_not_staged_iso_xyz" in str(exc.value)
    assert "stage" in str(exc.value).lower()
