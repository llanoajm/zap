"""Phase 4 item 4.2: the master report generator (``run_all``).

``run_all.generate(synthetic=True)`` runs every §8.4 capability benchmark and §7 dollar
backtest once and assembles ``reports/STEINMETZ_BENCH.md`` plus a machine-readable
``STEINMETZ_BENCH_results.json`` sidecar. These tests assert:

* the report and sidecar are written and contain all ten §8.4 + §7 headline experiment
  ids (the GPU id appears even when its result is cache-blocked);
* every headline rendered in the prose is the data-derived number from the sidecar
  ``BenchResult`` — re-derived here with the same formatter, not matched against a
  hand-written constant (anti-demoware);
* the CLI ``main(["--synthetic"])`` exits 0;
* ``--real`` against an empty ``data/`` blocks via ``DataNotStagedError`` rather than
  hanging, downloading, or fabricating numbers.

The full synthetic collection is expensive (it solves every experiment), so it runs once
behind a module-scoped fixture and the assertions share it.
"""

import json

import pytest

from experiments.steinmetz_bench import run_all
from experiments.steinmetz_bench.datasets.registry import DataNotStagedError
from experiments.steinmetz_bench.reports.result import BenchResult

EXPECTED_HEADLINE_IDS = {
    "2.1-speed-cpu",
    "2.2-planning",
    "2.3-accuracy",
    "2.4-sensitivity",
    "2.5-gpu-modal",
    "3.1-datacenter-siting",
    "3.2-datacenter-flex",
    "3.3-utility-sced",
    "3.4-transmission-audit",
    "3.5-mexico-epc",
}


@pytest.fixture(scope="module")
def generated(bench_results):
    """Write the master report from the shared collection; yield (results, md, sidecar).

    Reuses the session-wide ``bench_results`` (see ``conftest.py``) instead of re-running
    every experiment, so item 4.2's test adds only the cheap render step.
    """
    run_all.write_report(bench_results, synthetic=True)
    report_text = run_all.REPORT_MD.read_text()
    sidecar = json.loads(run_all.RESULTS_JSON.read_text())
    return bench_results, report_text, sidecar


def test_headline_ids_constant_matches_spec():
    """The module's HEADLINE_IDS are exactly the ten §8.4 + §7 experiments."""
    assert set(run_all.HEADLINE_IDS) == EXPECTED_HEADLINE_IDS


def test_report_and_sidecar_written(generated):
    _results, report_text, sidecar = generated
    assert run_all.REPORT_MD.is_file()
    assert run_all.RESULTS_JSON.is_file()
    assert report_text.strip()
    assert isinstance(sidecar, dict) and sidecar


def test_report_contains_every_headline_id(generated):
    _results, report_text, sidecar = generated
    for experiment_id in EXPECTED_HEADLINE_IDS:
        assert f"`{experiment_id}`" in report_text, (
            f"{experiment_id} missing from the master report"
        )
        assert experiment_id in sidecar, f"{experiment_id} missing from the JSON sidecar"


def test_synthetic_experiments_produced_results(generated):
    """Every headline except the cache-gated GPU benchmark yields a real result."""
    results, _report_text, _sidecar = generated
    for experiment_id in EXPECTED_HEADLINE_IDS:
        if experiment_id == "2.5-gpu-modal":
            continue  # legitimately None without a cached Modal dispatch
        assert isinstance(results[experiment_id], BenchResult)


def test_headlines_are_data_derived_not_handwritten(generated):
    """Each rendered headline equals the sidecar BenchResult's computed number.

    Anti-demoware: the report cannot carry a number that isn't in the JSON. We rebuild
    each non-blocked row's headline cell with the same formatter ``run_all`` used and
    require it to appear verbatim in the report.
    """
    _results, report_text, sidecar = generated
    checked = 0
    for experiment_id, payload in sidecar.items():
        if payload is None:
            continue
        result = BenchResult.from_dict(payload)
        assert result.experiment_id == experiment_id
        headline_cell = f"{run_all.fmt_number(result.headline_number)} {result.units}".strip()
        assert headline_cell in report_text, (
            f"headline for {experiment_id} ({headline_cell!r}) is not rendered from its "
            "BenchResult — possible hand-written number"
        )
        checked += 1
    # All synthetic headlines (everything but the possibly-blocked GPU row) were checked.
    assert checked >= len(EXPECTED_HEADLINE_IDS) - 1


def test_gpu_row_present_even_when_blocked(generated):
    """The GPU id is always in the report; if blocked it is labelled, not dropped."""
    results, report_text, _sidecar = generated
    assert "`2.5-gpu-modal`" in report_text
    if results["2.5-gpu-modal"] is None:
        assert "no cached Modal run" in report_text


def test_main_synthetic_exits_zero(bench_results, monkeypatch):
    """The CLI wires --synthetic through to a 0 exit and writes the report.

    Collection is stubbed with the shared results so the CLI test doesn't re-solve
    every experiment; only the routing + file writing is exercised here.
    """
    monkeypatch.setattr(run_all, "collect_results", lambda synthetic=True: bench_results)
    assert run_all.main(["--synthetic"]) == 0


def test_real_mode_blocks_without_staged_data():
    """``--real`` with an empty ``data/`` raises DataNotStagedError (clean block)."""
    with pytest.raises(DataNotStagedError):
        run_all.collect_results(synthetic=False)
