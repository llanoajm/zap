"""Phase 5 item 5.1: the finished whitepaper builder (``build_whitepaper``).

``build_whitepaper.build()`` renders ``STEINMETZ_WHITEPAPER.md`` from the JSON sidecar that
``run_all`` writes, and emits a ``grid_app_route/`` mount bundle. These tests enforce the
roadmap's strict anti-demoware acceptance:

* every benchmark/backtest headline in the data-derived results block is traceable to a real
  ``BenchResult`` id in the sidecar — each row is re-derived from the JSON with the builder's
  own formatter and required to appear verbatim;
* no numeric token in any result row is absent from that experiment's JSON record (a
  hand-written number cannot hide in the table);
* each cited value carries a fidelity band, and a bootstrap CI wherever the quantity is
  stochastic (deterministic accuracy/parity numbers are explicitly labelled, not given a
  fabricated CI);
* synthetic-vs-real provenance is labelled on every figure;
* the §5 architecture section references real symbols that actually exist in the codebase
  (grep-confirmed here), not aspirational ones;
* the ready-to-mount grid-app route bundle is emitted.

The sidecar is refreshed once from the shared, session-scoped ``bench_results`` fixture so the
whitepaper is built against the current code's real solves, not a stale committed file.
"""

import json
from pathlib import Path

import pytest

from experiments.steinmetz_bench import run_all
from experiments.steinmetz_bench.reports import build_whitepaper

ZAP_ROOT = Path(__file__).resolve().parents[3]
GRID_APP_MODAL = Path("/home/agent/grid-app/infra/modal/solver_app.py")

# Where each architecture symbol is literally grep-able, proving it is a real symbol.
SYMBOL_SOURCES = {
    "zap.network": ZAP_ROOT / "zap" / "layer.py",
    "PlanningProblem": ZAP_ROOT / "zap" / "planning" / "__init__.py",
    "PowerTarget": ZAP_ROOT / "zap" / "devices" / "power_target.py",
    "ADMMSolver": ZAP_ROOT / "zap" / "admm" / "basic_solver.py",
    "zap-opf-solver": GRID_APP_MODAL,
}


@pytest.fixture(scope="module")
def built(bench_results):
    """Refresh the sidecar from the shared results, build the whitepaper, return artifacts.

    Mirrors ``run_all``'s ``generated`` fixture: it reuses the session-wide solves instead of
    re-running every experiment, so this item adds only the cheap render step.
    """
    run_all.write_report(bench_results, synthetic=True)
    results = build_whitepaper.build()
    text = build_whitepaper.WHITEPAPER_MD.read_text()
    sidecar = json.loads(build_whitepaper.RESULTS_JSON.read_text())
    return results, text, sidecar


def _results_block(text: str) -> str:
    start = text.index(build_whitepaper.RESULTS_BLOCK_START)
    end = text.index(build_whitepaper.RESULTS_BLOCK_END)
    assert start < end, "results block markers out of order"
    return text[start:end]


def _data_rows(block: str) -> list[str]:
    """Table rows in the block that carry an experiment id (``| `<id>` | ...``)."""
    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("| `") and "`" in stripped[3:]:
            rows.append(stripped)
    return rows


def _row_id(row: str) -> str:
    return row.split("`")[1]


def test_whitepaper_and_bundle_written(built):
    _results, text, sidecar = built
    assert build_whitepaper.WHITEPAPER_MD.is_file()
    assert text.strip()
    assert isinstance(sidecar, dict) and sidecar
    assert (build_whitepaper.ROUTE_DIR / "page.tsx").is_file()
    assert (build_whitepaper.ROUTE_DIR / build_whitepaper.WHITEPAPER_MD.name).is_file()
    assert (build_whitepaper.ROUTE_DIR / "README.md").is_file()
    # The mounted copy is verbatim — the route can never drift from the source whitepaper.
    assert (build_whitepaper.ROUTE_DIR / build_whitepaper.WHITEPAPER_MD.name).read_text() == text


def test_every_headline_is_data_derived_from_json(built):
    """Each non-blocked experiment's row is re-derived from its JSON record and must appear.

    This is the core traceability guard: the builder's ``render_result_row`` is the single
    source for both the whitepaper and this assertion, so a headline that isn't in the
    sidecar cannot be rendered.
    """
    _results, text, sidecar = built
    block = _results_block(text)
    checked = 0
    for experiment_id, payload in sidecar.items():
        if payload is None:
            continue
        result = build_whitepaper.BenchResult.from_dict(payload)
        assert result.experiment_id == experiment_id
        row = build_whitepaper.render_result_row(result)
        assert row in block, (
            f"row for {experiment_id} is not rendered from its BenchResult — "
            "possible hand-written headline"
        )
        checked += 1
    # Every non-GPU headline (and the GPU one when its Modal cache is present) was checked.
    assert checked >= len(build_whitepaper.RESULT_META) - 1


def test_no_untraceable_number_in_results_block(built):
    """Every numeric token in every result row traces to that experiment's JSON record."""
    _results, text, sidecar = built
    block = _results_block(text)
    for row in _data_rows(block):
        experiment_id = _row_id(row)
        payload = sidecar.get(experiment_id)
        if payload is None:
            # Blocked row carries no numerals except the id itself.
            continue
        result = build_whitepaper.BenchResult.from_dict(payload)
        allowed = build_whitepaper.allowed_number_tokens(result)
        for token in build_whitepaper.number_tokens(row):
            assert token in allowed, (
                f"token {token!r} in the {experiment_id} row is not traceable to its "
                f"BenchResult (allowed: {sorted(allowed)})"
            )


def test_every_figure_carries_fidelity_and_ci_treatment(built):
    """Each cited value carries a fidelity band; stochastic ones carry a real CI.

    Per the anti-demoware rule, a deterministic accuracy/parity figure is *not* given an
    invented bootstrap CI — it is labelled ``n/a (deterministic)`` while still carrying its
    fidelity band. So the contract is: fidelity band always present, and a real CI present
    exactly when the JSON record has one.
    """
    _results, text, sidecar = built
    block = _results_block(text)
    rows_by_id = {_row_id(r): r for r in _data_rows(block)}
    for experiment_id, payload in sidecar.items():
        if payload is None:
            continue
        result = build_whitepaper.BenchResult.from_dict(payload)
        assert result.fidelity_band is not None, f"{experiment_id} lost its fidelity band"
        row = rows_by_id[experiment_id]
        assert build_whitepaper._fidelity_cell(result) in row
        if result.ci is not None:
            assert build_whitepaper._ci_cell(result) in row
        else:
            assert "n/a (deterministic)" in row


def test_every_figure_labels_provenance(built):
    """Every result row carries a synthetic-vs-real provenance label."""
    _results, text, sidecar = built
    block = _results_block(text)
    for row in _data_rows(block):
        experiment_id = _row_id(row)
        if sidecar.get(experiment_id) is None:
            assert "human-gated" in row
        else:
            assert "synthetic fixture" in row or "real (staged)" in row


def test_architecture_cites_real_symbols(built):
    """The §5 architecture section references symbols that actually exist in the code."""
    _results, text, _sidecar = built
    arch_marker = "## §5 Architecture"
    assert arch_marker in text
    assert set(build_whitepaper.ARCHITECTURE_SYMBOLS) == set(SYMBOL_SOURCES)
    for symbol, source in SYMBOL_SOURCES.items():
        # Grep-confirm the symbol is real before requiring the whitepaper to cite it.
        assert source.is_file(), f"source for {symbol} missing: {source}"
        assert symbol in source.read_text(), f"{symbol} not found in {source} — not a real symbol"
        assert symbol in text, f"architecture section does not cite real symbol {symbol}"


def test_all_ten_headline_ids_present(built):
    """All ten §7 + §8.4 headline experiments appear in the whitepaper."""
    _results, text, _sidecar = built
    expected = {
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
    for experiment_id in expected:
        assert f"`{experiment_id}`" in text, f"{experiment_id} missing from the whitepaper"


def test_cli_builds(bench_results, monkeypatch, capsys):
    """The CLI entrypoint builds without regenerating (sidecar already fresh)."""
    run_all.write_report(bench_results, synthetic=True)
    assert build_whitepaper.main([]) == 0
    out = capsys.readouterr().out
    assert "STEINMETZ_WHITEPAPER.md" in out
