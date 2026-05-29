"""Phase 5 item 5.1: assemble the finished Steinmetz whitepaper from real loop results.

``build_whitepaper.build()`` reads the machine-readable benchmark sidecar
(``STEINMETZ_BENCH_results.json``, written by :mod:`run_all`) and renders
``STEINMETZ_WHITEPAPER.md`` whose §7 dollar-backtest and §8.4 capability-benchmark
sections are filled in **exclusively from the ``BenchResult`` records** — every headline,
CI, and fidelity band is rendered through the same ``run_all.fmt_number`` formatter the
master report uses, never a hand-written constant. The §5 architecture section is grounded
in the real codebase: it cites the actual zap symbols (``zap.network`` dispatch + KKT
adjoint, ``PlanningProblem`` gradient loop, the ``PowerTarget`` flexible load, the
``ADMMSolver`` conic/GPU path), the ``zap-opf-solver`` Modal app (H100, ``solve_direct`` +
``modal.fastapi_endpoint``, image build, cold-start, the pandas copy-on-write +
``x=0 -> inf`` susceptance gotchas), the opencode harness, and the grid-app frontend.

It also emits a ready-to-mount bundle under ``reports/grid_app_route/`` (a markdown copy +
a minimal Next.js page scaffold + a human mount README) so a person can drop the whitepaper
into a gated grid-app route. Mounting is a human cross-repo step — this builder only
produces the artifact inside ``experiments/steinmetz_bench/``.

Anti-demoware: the only numbers inside the data-derived results block (delimited by the
``WHITEPAPER:RESULTS`` markers) are the ones tokenisable from each experiment's JSON
record. ``tests/test_build_whitepaper.py`` re-derives every row from the sidecar and scans
the block for any numeric token that is not traceable to a JSON ``BenchResult``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from experiments.steinmetz_bench import run_all
from experiments.steinmetz_bench.reports.result import BenchResult

REPORTS_DIR = Path(__file__).resolve().parent
RESULTS_JSON = REPORTS_DIR / "STEINMETZ_BENCH_results.json"
WHITEPAPER_MD = REPORTS_DIR / "STEINMETZ_WHITEPAPER.md"
ROUTE_DIR = REPORTS_DIR / "grid_app_route"

# Fences around the data-derived results region. Everything a number could be
# "hand-written" into lives between these markers; the test isolates the table rows here
# and proves every numeric token traces to the JSON sidecar.
RESULTS_BLOCK_START = "<!-- WHITEPAPER:RESULTS:START -->"
RESULTS_BLOCK_END = "<!-- WHITEPAPER:RESULTS:END -->"

# Real symbols the architecture section must cite; the test greps the actual codebase to
# confirm each exists, then asserts the whitepaper references it (no aspirational claims).
ARCHITECTURE_SYMBOLS = (
    "zap.network",
    "PlanningProblem",
    "PowerTarget",
    "ADMMSolver",
    "zap-opf-solver",
)

# Section grouping + a digit-free human title per experiment. Titles carry no numerals so
# the only numeric tokens in a result row come from that experiment's JSON record.
SECTION_BACKTESTS = "§7 dollar backtests"
SECTION_CAPABILITY = "§8.4 capability benchmarks"
SECTION_REFERENCES = "Phase 1 validation references"

RESULT_META: dict[str, tuple[str, str]] = {
    "1.1-pypsa-roundtrip": (SECTION_REFERENCES, "PyPSA LP roundtrip reference"),
    "1.2-grad-vs-dual": (SECTION_REFERENCES, "Adjoint gradient vs. exact dual identity"),
    "1.3-realized-lmp": (SECTION_REFERENCES, "Realized-LMP error comparator"),
    "2.1-speed-cpu": (SECTION_CAPABILITY, "Dispatch speed vs. a cvxpy LP baseline"),
    "2.2-planning": (SECTION_CAPABILITY, "Gradient expansion planner vs. the joint LP optimum"),
    "2.3-accuracy": (SECTION_CAPABILITY, "LMP / congestion error distribution"),
    "2.4-sensitivity": (SECTION_CAPABILITY, "Sensitivity correctness, adjoint vs. exact dual"),
    "2.5-gpu-modal": (SECTION_CAPABILITY, "GPU dispatch via the zap-opf-solver Modal app"),
    "3.1-datacenter-siting": (SECTION_BACKTESTS, "Data-center siting by LMP duration curve"),
    "3.2-datacenter-flex": (SECTION_BACKTESTS, "Data-center flexibility and battery sizing"),
    "3.3-utility-sced": (SECTION_BACKTESTS, "Vertically-integrated utility least-cost dispatch"),
    "3.4-transmission-audit": (SECTION_BACKTESTS, "Transmission-plan corridor audit"),
    "3.5-mexico-epc": (SECTION_BACKTESTS, "Mexico EPC dual-regime corridor ranking"),
}

SECTION_ORDER = (SECTION_BACKTESTS, SECTION_CAPABILITY, SECTION_REFERENCES)

# Token grammar shared by the builder and its test: a signed int/float with optional
# scientific exponent. The builder never emits a numeral outside this grammar in a row.
_NUMBER_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def number_tokens(text: str) -> list[str]:
    """All numeric tokens in ``text`` under the shared grammar (ints, floats, sci)."""
    return _NUMBER_TOKEN.findall(text)


def load_results() -> dict[str, Optional[BenchResult]]:
    """Read the JSON sidecar into ``{experiment_id: BenchResult | None}``.

    ``None`` survives for the cache-gated GPU benchmark when no Modal run was dispatched.
    """
    payload = json.loads(RESULTS_JSON.read_text())
    return {
        eid: (BenchResult.from_dict(rec) if rec is not None else None)
        for eid, rec in payload.items()
    }


def _headline_cell(result: BenchResult) -> str:
    return f"{run_all.fmt_number(result.headline_number)} {result.units}".strip()


def _ci_cell(result: BenchResult) -> str:
    if result.ci is None:
        # A bootstrap CI on a deterministic accuracy/parity number would be fabrication;
        # we mark it explicitly rather than invent one. The fidelity band still applies.
        return "n/a (deterministic)"
    ci = result.ci
    return f"[{run_all.fmt_number(ci.lo)}, {run_all.fmt_number(ci.hi)}] @ {ci.confidence:.0%}"


def _fidelity_cell(result: BenchResult) -> str:
    fb = result.fidelity_band
    if fb is None:
        return "n/a"
    units = f" {fb.units}" if fb.units else ""
    return f"{fb.reference}/{fb.metric}: max {run_all.fmt_number(fb.max_abs_gap)}{units} (n={fb.n})"


def allowed_number_tokens(result: BenchResult) -> set[str]:
    """Every numeric token that may legitimately appear in this experiment's row.

    The union of tokens drawn from the experiment's own JSON record: its id, dataset,
    units, headline, CI bounds + confidence, and fidelity-band figures. A token in the row
    that is absent from this set is, by construction, not traceable to the JSON — i.e. a
    hand-written number — and the test fails on it.
    """
    sources = [result.experiment_id, result.dataset, result.units, _headline_cell(result)]
    if result.ci is not None:
        sources.append(_ci_cell(result))
    if result.fidelity_band is not None:
        sources.append(_fidelity_cell(result))
    tokens: set[str] = set()
    for source in sources:
        tokens.update(number_tokens(source))
    return tokens


def render_result_row(result: BenchResult) -> str:
    """Render one data-derived markdown table row from a ``BenchResult``.

    Single source of truth for both the whitepaper and its test, so the prose number can
    never drift from the sidecar.
    """
    return (
        f"| `{result.experiment_id}` | {result.dataset} | {_headline_cell(result)} | "
        f"{_ci_cell(result)} | {_fidelity_cell(result)} | synthetic fixture |"
    )


def _blocked_row(experiment_id: str) -> str:
    return (
        f"| `{experiment_id}` | — | _blocked_ | — | — | "
        "human-gated (no cached Modal run) |"
    )


def _section_lines(section: str, results: dict[str, Optional[BenchResult]]) -> list[str]:
    lines = [
        f"### {section}",
        "",
        f"_Every figure below is rendered directly from a `BenchResult` in "
        f"`{RESULTS_JSON.name}`; provenance is labelled per row._",
        "",
        "| Experiment | Dataset | Headline | CI | Fidelity band | Provenance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for experiment_id, (sec, _title) in RESULT_META.items():
        if sec != section:
            continue
        result = results.get(experiment_id)
        lines.append(_blocked_row(experiment_id) if result is None else render_result_row(result))
    lines.append("")
    # A digit-free reading guide so a human knows what each headline means without any
    # number leaking outside the table (the token scan only inspects the rows above).
    lines.append("Reading guide:")
    for experiment_id, (sec, title) in RESULT_META.items():
        if sec != section:
            continue
        lines.append(f"- `{experiment_id}` — {title}.")
    lines.append("")
    return lines


def _architecture_lines() -> list[str]:
    return [
        "## §5 Architecture (grounded in the real codebase)",
        "",
        "This section cites only symbols that exist in the shipped code; nothing here is",
        "aspirational. The benchmark suite imports the `zap` library read-only.",
        "",
        "### Dispatch and sensitivities — `zap.network`",
        "",
        "The convex DC-OPF solve lives in `zap.network` (`PowerNetwork.dispatch`), which",
        "builds the CVXPY problem from the device set and returns nodal prices as the power-",
        "balance duals. Sensitivities come from the same module's KKT adjoint:",
        "`PowerNetwork.kkt` assembles the optimality system and `kkt_vjp_parameters` solves",
        "the adjoint linear system, so `d(system cost)/d(parameter)` is exact rather than",
        "finite-differenced. Item 1.2 / 2.4 validate this against the envelope-theorem dual",
        "identity for line, generator, and battery capacities.",
        "",
        "### Gradient-based planning — `PlanningProblem`",
        "",
        "Investment / expansion problems are expressed with `PlanningProblem` (exported from",
        "`zap.planning`). It composes the differentiable dispatch layer with an investment",
        "objective and runs projected-gradient descent over capacities — the loop item 2.2",
        "drives against the joint multi-scenario expansion LP optimum as a global lower bound.",
        "",
        "### Flexible demand — `PowerTarget`",
        "",
        "Flexible loads (the data-center backtests, §7.1) are modelled with the `PowerTarget`",
        "device (`zap.devices.power_target`). Backtest 3.2 serves the data center as a",
        "`PowerTarget` with a co-located battery and finds the break-even battery size where",
        "the adjoint marginal value crosses the annualized capital cost.",
        "",
        "### Conic / GPU path — `ADMMSolver` and the `zap-opf-solver` Modal app",
        "",
        "The GPU path uses the operator-splitting `ADMMSolver` (`zap.admm.basic_solver`). It",
        "is deployed as the `zap-opf-solver` Modal app (`grid-app/infra/modal/solver_app.py`):",
        "an H100-backed `modal.App` exposing `solve_direct` and a `modal.fastapi_endpoint`",
        "POST route. The image is built from a pinned `fastapi[standard]` + torch stack, so",
        "the first call pays a cold-start before the H100 is warm. Two real gotchas are baked",
        "in: pandas copy-on-write means the importer must not mutate shared frames in place,",
        "and a PyPSA line with reactance `x=0` makes the importer's `susceptance = 1/x` blow",
        "up to `inf` — the endpoint sanitises non-finite values to JSON `null`. Item 2.5",
        "dispatches this app once, caches the JSON to `data/gpu_runs/`, and the test reads only",
        "the cache (Modal is never called from the per-item verify).",
        "",
        "### Harness and frontend — opencode + grid-app",
        "",
        "These benchmarks are produced by an autonomous opencode harness (the `claude -p`",
        "Ralph loop driving `BENCH_ROADMAP.md`), and the published artifact is meant to be",
        "mounted at a gated route in the grid-app Next.js frontend. The mount itself is a",
        "human cross-repo step; this builder only emits the `grid_app_route/` bundle.",
        "",
    ]


def render_whitepaper(results: dict[str, Optional[BenchResult]]) -> str:
    produced = sum(1 for r in results.values() if r is not None)
    lines = [
        "# Steinmetz — Verification & Quantification-of-Value Whitepaper",
        "",
        "_Grounded rewrite of the product spec: every §7 / §8.4 figure is replaced with the",
        "actual number this benchmark loop computed from a real zap solve. Synthetic-fixture",
        "values are labelled as such; real-data values appear only where a human staged",
        "`data/` and re-ran with `--real`._",
        "",
        f"- **Provenance of every figure below:** synthetic fixtures "
        f"(`{RESULTS_JSON.name}`).",
        f"- **Experiments reporting a result:** {produced} / {len(results)}.",
        "- **Anti-demoware contract:** no headline in the results block is hand-written; each",
        "  is rendered from a `BenchResult` and carries a fidelity band (and a bootstrap CI",
        "  where the quantity is stochastic).",
        "",
    ]
    lines += _architecture_lines()
    lines += [
        "## §7 / §8.4 results (data-derived)",
        "",
        RESULTS_BLOCK_START,
    ]
    for section in SECTION_ORDER:
        lines += _section_lines(section, results)
    lines += [
        RESULTS_BLOCK_END,
        "",
        "## Real-data path",
        "",
        "Each figure above is a synthetic-fixture result. A human stages ISO LMP/load,",
        "topology, RTEP/MTEP, and CENACE data into `data/` and re-runs",
        "`python -m experiments.steinmetz_bench.run_all --real`; the same code paths then",
        "emit publishable dollar numbers, which this builder will re-render with a `real",
        "(staged)` provenance label in place of `synthetic fixture`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _page_scaffold(markdown_name: str) -> str:
    return f"""import {{ promises as fs }} from "fs"
import path from "path"

// Minimal scaffold for a GATED grid-app route. Drop this directory into
// grid-app `app/app/<route>/` (behind the existing auth gate) and replace the <pre>
// with the project's markdown renderer if one exists. The whitepaper markdown
// ({markdown_name}) is colocated so the route is self-contained.
export default async function SteinmetzWhitepaperPage() {{
  const file = path.join(process.cwd(), "app", "app", "whitepaper", "{markdown_name}")
  const markdown = await fs.readFile(file, "utf8")
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto w-full px-6 py-8">
        <pre className="whitespace-pre-wrap font-mono text-sm">{{markdown}}</pre>
      </div>
    </div>
  )
}}
"""


def _route_readme(markdown_name: str) -> str:
    return (
        "# grid-app route bundle (human mount step)\n"
        "\n"
        "This directory is a ready-to-mount bundle for the Steinmetz whitepaper. It is\n"
        "produced inside the zap repo by `reports/build_whitepaper.py`; mounting it into\n"
        "grid-app is a human cross-repo step (the zap loop's verify does not cover the\n"
        "grid-app TypeScript).\n"
        "\n"
        "Contents:\n"
        f"- `{markdown_name}` — a verbatim copy of the generated whitepaper.\n"
        "- `page.tsx` — a minimal Next.js server-component scaffold that renders the copy.\n"
        "\n"
        "To mount (by hand):\n"
        "1. Copy this directory to a gated route, e.g. grid-app `app/app/whitepaper/`.\n"
        "2. Swap the `<pre>` for grid-app's markdown renderer if it has one.\n"
        "3. Verify with grid-app's own `tsc --noEmit && npm run test:unit`.\n"
    )


def write_route_bundle() -> None:
    """Emit the ready-to-mount grid-app bundle (md copy + page scaffold + README)."""
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)
    markdown_name = WHITEPAPER_MD.name
    shutil.copyfile(WHITEPAPER_MD, ROUTE_DIR / markdown_name)
    (ROUTE_DIR / "page.tsx").write_text(_page_scaffold(markdown_name))
    (ROUTE_DIR / "README.md").write_text(_route_readme(markdown_name))


def build() -> dict[str, Optional[BenchResult]]:
    """Render the whitepaper + the grid-app route bundle from the JSON sidecar."""
    results = load_results()
    WHITEPAPER_MD.write_text(render_whitepaper(results))
    write_route_bundle()
    return results


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the Steinmetz whitepaper.")
    parser.add_argument(
        "--regenerate-results",
        action="store_true",
        help="re-run run_all --synthetic to refresh the JSON sidecar before building",
    )
    args = parser.parse_args(argv)
    if args.regenerate_results:
        run_all.generate(synthetic=True)
    build()
    print(f"wrote {WHITEPAPER_MD}")
    print(f"wrote {ROUTE_DIR}/ (page.tsx + markdown copy + README)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
