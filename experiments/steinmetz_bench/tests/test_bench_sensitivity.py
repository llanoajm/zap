"""Tests for the sensitivity-correctness report (item 2.4, §8.4.4).

This report publishes the item-1.2 gradient-vs-dual check as a per-device-type table.
These tests confirm the table covers all three device types, that each published cell
is the same number 1.2 computed (not a fresh constant), that the per-device worst-case
relative error clears the §8.4.4 acceptance, and that the written markdown report
contains a real table and re-parses to an identical BenchResult.
"""

import pytest

from experiments.steinmetz_bench.experiments.bench_sensitivity import (
    EXPERIMENT_ID,
    PER_DEVICE_TOL,
    SensitivityTable,
    run,
    run_table,
)
from experiments.steinmetz_bench.reports import read_markdown


@pytest.fixture(scope="module")
def table():
    return run_table(do_fd=True)


def test_covers_all_three_device_types(table):
    assert {"line", "generator", "battery"} <= set(table.per_device_type())


def test_per_device_max_error_under_tolerance(table):
    """The §8.4.4 acceptance: every device type's worst-case rel error < 1e-3."""
    per_type = table.per_device_type()
    for device_type in ("line", "generator", "battery"):
        assert per_type[device_type] < PER_DEVICE_TOL, (
            f"{device_type}: {per_type[device_type]:.2e} >= {PER_DEVICE_TOL}"
        )
    assert table.headline_number < PER_DEVICE_TOL


def test_rows_trace_back_to_underlying_check(table):
    """Each table cell must equal the value the 1.2 report computed — no re-invention."""
    by_key = {(c.network, c.device_type): c for c in table.report.checks}
    assert table.rows
    for row in table.rows:
        c = by_key[(row.network, row.device_type)]
        assert row.max_rel_err_dual == c.max_rel_err_dual
        assert row.max_rel_err_fd == c.max_rel_err_fd
        assert row.n_active == c.n_active


def test_headline_is_worst_device_type(table):
    per_type = table.per_device_type()
    assert table.headline_number == max(per_type.values())


def test_from_report_is_consistent(table):
    rebuilt = SensitivityTable.from_report(table.report)
    assert rebuilt.headline_number == table.headline_number
    assert len(rebuilt.rows) == len(table.rows)


def test_emits_reparseable_bench_result_with_table(tmp_path):
    md_path = tmp_path / "bench_sensitivity.md"
    result = run(report_path=md_path, do_fd=False)

    assert result.experiment_id == EXPERIMENT_ID
    assert result.units == "relative"
    assert result.fidelity_band is not None
    assert result.fidelity_band.reference == "exact-dual"
    assert result.headline_number < PER_DEVICE_TOL
    assert set(result.sensitivities["rel_err_by_device_type"]) >= {
        "line", "generator", "battery"
    }
    assert result.sensitivities["table"]

    assert md_path.exists()
    text = md_path.read_text()
    # A genuine published markdown table, not just the JSON dump.
    assert "| network | device_type |" in text
    assert "| device_type | max_rel_err_dual | pass |" in text
    assert read_markdown(md_path).to_dict() == result.to_dict()
