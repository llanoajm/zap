"""Master report generator (Phase 4 item 4.2).

Assembles every Steinmetz §8.4 capability benchmark and §7 dollar backtest (plus the
Phase 1 validation references) into a single human-readable report,
``reports/STEINMETZ_BENCH.md``, alongside a machine-readable
``reports/STEINMETZ_BENCH_results.json`` sidecar that the Phase 5 whitepaper builder
reads back. Each row's headline, CI, and fidelity band are rendered straight from the
``BenchResult`` an experiment's real solve produced — never a hand-written constant.

Two modes:

* ``--synthetic`` (default, loop-runnable): runs each experiment's synthetic entrypoint
  on bundled / generated fixtures. This is what the autonomous loop verifies.
* ``--real`` (human): the publishable run against staged ISO/topology data under
  ``data/``. With an empty ``data/`` it raises
  :class:`~experiments.steinmetz_bench.datasets.registry.DataNotStagedError` — a clean
  block, never a download or a hang.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from experiments.steinmetz_bench.datasets.registry import DatasetSpec, resolve
from experiments.steinmetz_bench.experiments import (
    bench_accuracy,
    bench_gpu_modal,
    bench_planning,
    bench_sensitivity,
    bench_speed,
    bt_datacenter_flex,
    bt_datacenter_siting,
    bt_mexico_epc,
    bt_transmission_audit,
    bt_utility,
    grad_check,
    realized_lmp,
    ref_pypsa,
)
from experiments.steinmetz_bench.reports.result import BenchResult

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
PER_EXPERIMENT_DIR = REPORTS_DIR / "experiments"
REPORT_MD = REPORTS_DIR / "STEINMETZ_BENCH.md"
RESULTS_JSON = REPORTS_DIR / "STEINMETZ_BENCH_results.json"

# Real datasets a human stages before ``--real``; resolving any one against an empty
# ``data/`` raises DataNotStagedError, which is exactly the clean block we want.
REAL_DATASETS = ("ercot_west", "pjm")

# The experiment modules, grouped into the report's sections. Each module exposes an
# ``EXPERIMENT_ID`` and a synthetic, loop-runnable ``run(report_path=...)`` returning a
# ``BenchResult`` (only the cache-gated GPU benchmark returns ``None`` when no Modal run
# has been dispatched).
PHASE_1_MODULES = [ref_pypsa, grad_check, realized_lmp]
CAPABILITY_MODULES = [
    bench_speed,
    bench_planning,
    bench_accuracy,
    bench_sensitivity,
    bench_gpu_modal,
]
BACKTEST_MODULES = [
    bt_datacenter_siting,
    bt_datacenter_flex,
    bt_utility,
    bt_transmission_audit,
    bt_mexico_epc,
]
ALL_MODULES = PHASE_1_MODULES + CAPABILITY_MODULES + BACKTEST_MODULES

# The ten §8.4 + §7 headline experiments the master report must always carry (4 capability
# benchmarks + the GPU benchmark + 5 backtests). The GPU id appears even when its result is
# blocked, so the report's coverage never silently shrinks.
HEADLINE_IDS = tuple(m.EXPERIMENT_ID for m in CAPABILITY_MODULES + BACKTEST_MODULES)


def fmt_number(value: float) -> str:
    """Stable rendering of a headline / gap number, shared by the report and its test.

    Using one formatter for both guarantees the prose number is the data-derived one
    (anti-demoware): the test recomputes the same string from the JSON sidecar.
    """
    return f"{value:.6g}"


def _require_real_data_staged() -> None:
    """Raise ``DataNotStagedError`` unless real data has been staged under ``data/``.

    Delegates to the dataset registry so the staging contract lives in one place.
    """
    for name in REAL_DATASETS:
        resolve(DatasetSpec(name=name, kind="cache"))


def collect_results(synthetic: bool = True) -> dict[str, Optional[BenchResult]]:
    """Run every experiment once, writing per-experiment stubs, and return the results.

    The returned mapping is keyed by ``EXPERIMENT_ID`` and preserves section order. The
    GPU benchmark maps to ``None`` when no Modal run is cached (blocked, not broken).
    """
    if not synthetic:
        _require_real_data_staged()

    results: dict[str, Optional[BenchResult]] = {}
    for module in ALL_MODULES:
        experiment_id = module.EXPERIMENT_ID
        report_path = PER_EXPERIMENT_DIR / f"{experiment_id}.md"
        results[experiment_id] = module.run(report_path=report_path)
    return results


def _ci_cell(result: BenchResult) -> str:
    if result.ci is None:
        return "n/a"
    ci = result.ci
    return f"[{fmt_number(ci.lo)}, {fmt_number(ci.hi)}] ({ci.confidence:.0%})"


def _fidelity_cell(result: BenchResult) -> str:
    if result.fidelity_band is None:
        return "n/a"
    fb = result.fidelity_band
    units = f" {fb.units}" if fb.units else ""
    return f"{fb.reference}/{fb.metric}: max {fmt_number(fb.max_abs_gap)}{units} (n={fb.n})"


def _result_row(experiment_id: str, result: Optional[BenchResult]) -> str:
    if result is None:
        return (
            f"| `{experiment_id}` | _blocked_ | — | — | — | "
            "no cached Modal run (dispatch via `bench_gpu_modal --dispatch`) |"
        )
    headline = f"{fmt_number(result.headline_number)} {result.units}".strip()
    return (
        f"| `{experiment_id}` | {result.dataset} | {headline} | "
        f"{_ci_cell(result)} | {_fidelity_cell(result)} |"
    )


def _section(title: str, modules, results: dict[str, Optional[BenchResult]]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| Experiment | Dataset | Headline | CI | Fidelity band |")
    lines.append("| --- | --- | --- | --- | --- |")
    for module in modules:
        experiment_id = module.EXPERIMENT_ID
        lines.append(_result_row(experiment_id, results.get(experiment_id)))
    lines.append("")
    return lines


def render_report(results: dict[str, Optional[BenchResult]], synthetic: bool) -> str:
    """Render the master markdown report from collected results."""
    provenance = "synthetic fixtures" if synthetic else "staged real data (`--real`)"
    produced = sum(1 for r in results.values() if r is not None)
    lines = [
        "# Steinmetz Benchmark — Master Report",
        "",
        f"- **Provenance:** {provenance}",
        f"- **Experiments reporting a result:** {produced} / {len(results)}",
        "- **Source of every number:** the JSON sidecar "
        "`STEINMETZ_BENCH_results.json`; each headline below is rendered directly from a "
        "`BenchResult` produced by a real zap solve (no hand-written constants).",
        "",
    ]
    lines += _section("§8.4 capability benchmarks", CAPABILITY_MODULES, results)
    lines += _section("§7 dollar backtests", BACKTEST_MODULES, results)
    lines += _section("Phase 1 validation references", PHASE_1_MODULES, results)
    return "\n".join(lines)


def _results_to_json(results: dict[str, Optional[BenchResult]]) -> str:
    import json

    payload = {
        eid: (result.to_dict() if result is not None else None)
        for eid, result in results.items()
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def write_report(results: dict[str, Optional[BenchResult]], synthetic: bool) -> None:
    """Write the master markdown report + JSON sidecar from already-collected results.

    Cheap relative to :func:`collect_results` (no solves), so tests can collect once and
    render many times.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_report(results, synthetic) + "\n")
    RESULTS_JSON.write_text(_results_to_json(results) + "\n")


def generate(synthetic: bool = True) -> dict[str, Optional[BenchResult]]:
    """Collect every result and write the report + JSON sidecar; return the results."""
    results = collect_results(synthetic=synthetic)
    write_report(results, synthetic)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the Steinmetz master report.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--synthetic",
        action="store_true",
        help="run on synthetic/bundled fixtures (default; loop-runnable)",
    )
    group.add_argument(
        "--real",
        action="store_true",
        help="run on staged real data under data/ (human; blocks if data/ is empty)",
    )
    args = parser.parse_args(argv)
    synthetic = not args.real
    results = generate(synthetic=synthetic)
    produced = sum(1 for r in results.values() if r is not None)
    print(f"wrote {REPORT_MD} ({produced}/{len(results)} experiments reporting)")
    print(f"wrote {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
