# Steinmetz Benchmark Roadmap (§7 backtests + §8.4 benchmarks)

Autonomous-loop roadmap to build and self-validate the verification / quantification-of-value
machinery from the Steinmetz spec. Every item below is **loop-runnable**: it builds code that
works on **synthetic / bundled fixtures** and produces a machine-checkable result. The same code
is parameterized to accept **real staged data** later — that real run is a human step (see
"Human prerequisites", which are deliberately NOT loop items).

## Design principle: synthetic-first, real-data-parameterized

The §7 backtests and the realized-LMP pieces need external *market* data that a headless
`claude -p` loop cannot reliably acquire. So the loop builds the **entire harness** and proves
every code path on synthetic fixtures with deterministic acceptance tests. A human then drops real
data into `data/` and re-runs the *same* scripts with `--real` to get the publishable dollar
numbers. Nothing in the loop makes a live market-data call or installs Julia.

**Modal/GPU is the exception — it is already available.** `grid-app/infra/modal/solver_app.py`
deploys `modal.App("zap-opf-solver")` (zap's ADMM solver on **H100**), `~/.modal.toml` is
authenticated, and a prior loop already validated CPU-vs-GPU parity (`infra/modal/PARITY_REPORT.md`).
So the GPU speed benchmark IS a loop item — but cost-guarded: it dispatches Modal **once**, caches
the result to `data/gpu_runs/*.json`, and its test reads only the cache. Modal is NEVER called from
the per-item verify command.

## Loop configuration (for loop.sh)

- **Project root:** `/home/agent/zap`
- **Branch:** `steinmetz-bench` (cut from `main`; `main` stays untouched, mirroring the redesign loop)
- **Python:** `/home/agent/zap/.venv/bin/python` (zap installed editable here)
- **Verify command (per item):** `cd /home/agent/zap && .venv/bin/python -m pytest experiments/steinmetz_bench/tests -q && .venv/bin/ruff check experiments/steinmetz_bench`
- **Rollback:** on verify failure, `git reset --hard` the item's changes (standard ralph).

## Guardrails (bake into loop.sh)

- **Only create/edit files under `experiments/steinmetz_bench/**`.** NEVER modify the zap library
  core (`zap/**`) or any other `experiments/**` directory. Benchmarks import zap read-only.
- **No live MARKET-data calls** in any loop item. Grid/price data comes from synthetic generators
  or the `data/` cache. If a real-data path finds the cache empty it must raise `DataNotStagedError`
  (clean block), never hang or retry a download.
- **Modal/GPU is allowed only inside the one dedicated GPU benchmark item (2.5).** It dispatches the
  existing `zap-opf-solver` app once and caches results to `data/gpu_runs/`. Modal is NEVER invoked
  from the per-item verify command (cost + cold-start). CPU path uses cvxpy (ECOS; Mosek only if a
  license is already present, else skipped with a recorded note).
- Keep the existing 86 zap tests green is NOT required to pass per-item (we don't touch core), but
  do not break collection: the verify command scopes pytest to `experiments/steinmetz_bench/tests`.

---

## Phase 0 — Scaffolding & shared harness

- [ ] **0.1 Package skeleton.** Create `experiments/steinmetz_bench/` package: `__init__.py`,
  subpkgs `datasets/`, `scoring/`, `experiments/`, `reports/`, `tests/`, and an empty `data/`
  with a README explaining what a human stages there. Add a one-line smoke test.
  - Acceptance: `pytest experiments/steinmetz_bench/tests -q` collects ≥1 test and the smoke test
    passes; `ruff check experiments/steinmetz_bench` is clean.
- [ ] **0.2 Dataset registry + loaders.** `datasets/registry.py` resolving a `DatasetSpec` to
  `(PowerNetwork, time_index, price_frame|None, load_frame|None)`. Two source kinds: (a) **synthetic**
  generators (params: `n_nodes`, `hours`, `congested: bool`, `seed`) that also wrap the existing
  toy/Garver importers; (b) **cache** loaders reading `data/<name>/` and raising `DataNotStagedError`
  when absent.
  - Acceptance: unit test loads a synthetic 5-node net and the Garver net, asserts expected device
    counts and shapes; the cache path raises `DataNotStagedError` with a clear message.
- [ ] **0.3 Scoring harness.** `scoring/metrics.py`: `counterfactual_delta()`, `bootstrap_ci()`
  (P50/P10 + CI tuple), `duration_curve()`, and `fidelity_band()` that records DC-vs-reference gaps.
  - Acceptance: unit tests on synthetic arrays return a sane CI tuple (lo ≤ mid ≤ hi) and a
    monotone-decreasing duration curve.
- [ ] **0.4 Result schema + report writer.** `reports/result.py`: a `BenchResult` dataclass
  (`experiment_id, dataset, headline_number, units, ci, fidelity_band, assumptions, sensitivities`)
  with JSON dump + a per-experiment markdown stub writer.
  - Acceptance: schema round-trips to/from JSON; a written report file exists and re-parses; test passes.

## Phase 1 — Self-contained validation references

- [ ] **1.1 PyPSA LP roundtrip reference.** `experiments/ref_pypsa.py`: solve a bundled network in
  PyPSA, align to zap's solve, return per-node LMP gap and per-line flow gap (reuse zap's existing
  roundtrip test tolerances).
  - Acceptance: on a bundled test net, max LMP gap < `1e-2` and max flow gap < `1e-3`; emits a `BenchResult`.
- [ ] **1.2 Gradient-vs-exact-dual check (§8.4.4 core).** `experiments/grad_check.py`: compute
  `∂(system cost)/∂(line capacity)` via zap's adjoint and compare to the line-limit dual `μ`
  (identity `∂f/∂cap = −μ`, paper Fig. 6). Repeat for generator capacity and battery power.
  - Acceptance: max relative gradient error < `1e-3` on Garver for all three device types; emits `BenchResult`.
- [ ] **1.3 Realized-LMP comparator.** `experiments/realized_lmp.py`: given a `price_frame`
  (synthetic fixture now, cached ISO later), compute per-node/per-hour zap-vs-realized LMP error
  distribution (mean/median/p90).
  - Acceptance: on a synthetic `price_frame` fixture, emits the error distribution; the missing-cache
    path blocks via `DataNotStagedError` rather than failing the test.

## Phase 2 — §8.4 capability benchmarks

- [ ] **2.1 Speed benchmark (§8.4.1, CPU).** `experiments/bench_speed.py`: wall-clock zap dispatch
  vs a cvxpy LP baseline (Mosek if licensed, else ECOS) across ≥3 network sizes; assert objective parity.
  - Acceptance: emits a timing table (per size: zap_s, baseline_s, objective_gap); objective_gap < `1e-2`
    on every size; report explicitly tags the WECC / 1000-contingency / Modal-H100 headline as human-gated.
- [ ] **2.2 Planning benchmark (§8.4.2).** `experiments/bench_planning.py`: gradient planner vs a
  baseline (brute grid or implicit-diff) on a small multi-scenario expansion (gen + line + battery caps).
  - Acceptance: planner final objective ≤ baseline objective + tol; converges within a fixed iteration
    budget; timing recorded in the `BenchResult`.
- [ ] **2.3 Accuracy benchmark (§8.4.3).** `experiments/bench_accuracy.py`: assemble LMP and
  congestion-component error vs PyPSA (1.1) and vs realized (1.3), reported as distributions not points.
  - Acceptance: emits a distribution `BenchResult` on fixtures; the realized-vs path is parameterized so
    `--real` will use staged data (human-gated) but `--synthetic` passes in the loop.
- [ ] **2.4 Sensitivity-correctness report (§8.4.4).** `experiments/bench_sensitivity.py`: wrap 1.2
  into a published per-device-type table.
  - Acceptance: table `BenchResult` with per-device max-error < `1e-3`; report file written.
- [ ] **2.5 GPU speed via Modal (§8.4.1 headline).** `experiments/bench_gpu_modal.py`: dispatch the
  EXISTING `zap-opf-solver` app (`grid-app/infra/modal/solver_app.py`, function
  `solve_direct.remote(...)` or `modal run ...::smoke`) on a modest network + one larger network;
  record GPU wall-clock and CPU-vs-GPU objective/LMP parity (build on `infra/modal/PARITY_REPORT.md`).
  Run Modal exactly once; cache the JSON result to `data/gpu_runs/<timestamp>.json`. The test reads
  ONLY the cached JSON — it must not call Modal. **Cost guard:** bounded network sizes only; the full
  WECC 1000-contingency headline stays an optional human run (see prerequisites).
  - Acceptance: if `modal` CLI + `~/.modal.toml` are present, a `data/gpu_runs/*.json` exists with
    `machine=="cuda"`, an `elapsed_s`, and CPU-vs-GPU objective gap < `1e-2`; emits a `BenchResult`.
    If Modal is unavailable, the item self-marks `[!]` (blocked, reason logged) rather than failing.

## Phase 3 — §7 dollar backtests (synthetic-validated)

- [ ] **3.1 Data-center siting (§7.1-A).** `experiments/bt_datacenter_siting.py`: rank candidate
  nodes by LMP duration curve + curtailment frequency over scenarios; pick the best-distribution node;
  compute realized effective $/MWh delta vs a default node.
  - Acceptance: on a synthetic net with one deliberately cheap node, the recommended node == that node;
    emits $/MWh delta with bootstrap CI.
- [ ] **3.2 Data-center flexibility & battery sizing (§7.1-B).** `experiments/bt_datacenter_flex.py`:
  model the load as a `PowerTarget` flexible device + co-located battery; compute
  `∂(savings)/∂(battery MW)` and find the break-even size; firm-vs-flexible $/yr.
  - Acceptance: a break-even battery size is found where marginal value ≈ marginal cost; firm-vs-flexible
    $/yr delta emitted with CI; gradient is finite-difference-checked.
- [ ] **3.3 Vertically-integrated utility (§7.2).** `experiments/bt_utility.py`: reconstruct a
  synthetic fleet + load, run SCED/PCM least-cost dispatch, compare to a deliberately-suboptimal
  "actual" dispatch; produce a 5-year expansion ranking.
  - Acceptance: modeled least-cost ≤ "actual" cost; expansion returns a ranked project list; PyPSA
    roundtrip gap < tol; avoided-fuel $/yr + NPV-delta emitted with fidelity band.
- [ ] **3.4 Transmission-plan audit (§7.3).** `experiments/bt_transmission_audit.py`: rank corridors
  ex-ante by `∂(system cost)/∂(line capacity)`; correlate with a (synthetic) realized-congestion vector;
  report R² + count of "missed" high-value corridors.
  - Acceptance: the known-congested corridor ranks #1; rank-correlation with the realized-congestion
    vector exceeds a threshold; emits R² `BenchResult`.
- [ ] **3.5 Mexico EPC, dual-regime (§7.4).** `experiments/bt_mexico_epc.py`: rank corridors under
  (a) historical merit-order dispatch and (b) a CFE-≥54%-share dispatch constraint; report ranking
  agreement + congestion-relief $/yr per corridor.
  - Acceptance: both regimes produce corridor rankings; the CFE constraint is demonstrably binding
    (prices/dispatch shift measurably between regimes); a ranking-agreement metric is emitted. (This item
    also serves as the jurisdiction-rule-encoding capability demo.)

## Phase 4 — Aggregation & publishable outputs

- [ ] **4.1 Fidelity band on every result.** Ensure each `BenchResult` from Phases 1–3 carries a
  non-null `fidelity_band` (DC-vs-PyPSA, and DC-vs-realized where applicable).
  - Acceptance: a test iterates all produced results and asserts `fidelity_band` is present and well-formed.
- [ ] **4.2 Master report generator.** `run_all.py` assembling the §8.4 benchmark tables + §7
  headlines into `reports/STEINMETZ_BENCH.md`, with CIs and fidelity bands. Supports `--synthetic`
  (default, loop-runnable) and `--real` (human, uses staged data).
  - Acceptance: `python -m experiments.steinmetz_bench.run_all --synthetic` exits 0 and writes a report
    containing all 10 experiments (4 capability benchmarks + GPU benchmark + 5 backtests) with tables;
    a test asserts the report contains each experiment id.

## Phase 5 — Finished whitepaper (grounded rewrite of the spec)

- [ ] **5.1 Assemble the Steinmetz whitepaper from real loop results.** `reports/build_whitepaper.py`
  produces `reports/STEINMETZ_WHITEPAPER.md` by taking the original product spec and:
  (a) **replacing the §7 backtest and §8.4 benchmark sections with the ACTUAL numbers** this loop
  produced — every headline gets its real value, bootstrap CI, and fidelity band pulled from the
  `BenchResult` JSON (synthetic-run values clearly labelled "synthetic fixture"; real values appear
  only where `data/` was staged); (b) **rewriting the §5 architecture section grounded in the real
  codebase** — cite the actual layers: zap's `network.py` dispatch + KKT adjoint, `planning/`
  gradient loop, the device set incl. `PowerTarget` flexible load, the ADMM/conic GPU path, the
  `zap-opf-solver` Modal app (H100, `solve_direct`/HTTP endpoint, image build, cold-start, the
  pandas-CoW + `x=0→inf susceptance` gotchas), the opencode harness, and the grid-app frontend —
  no aspirational claims that the code doesn't support. Also emit a ready-to-mount grid-app bundle
  under `reports/grid_app_route/` (a markdown copy + a minimal page-component scaffold) for a human
  to drop into a gated route under grid-app `app/app/`.
  - **Anti-demoware acceptance (strict):** a test parses `STEINMETZ_WHITEPAPER.md` and asserts every
    benchmark/backtest headline number is traceable to a real `BenchResult` id in the results JSON
    (no number appears in prose that isn't in the JSON); asserts each cited value carries a CI and a
    fidelity_band; asserts synthetic-vs-real provenance is labelled on every figure; and asserts the
    architecture section references real symbols that exist (grep `zap.network`, `PlanningProblem`,
    `PowerTarget`, `ADMMSolver`, `zap-opf-solver`). The build fails if any headline is hand-written
    rather than data-derived.

> **Cross-repo note (read before launch):** the actual mount of the whitepaper at a gated grid-app
> route is the one cross-repo step. This loop is rooted in `/home/agent/zap`; its `git reset --hard`
> rollback and Python verify do NOT cover grid-app TS. So item 5.1 only *produces* the artifact +
> route bundle inside zap. A human (or a tiny grid-app-rooted follow-on, where grid-app's own
> `tsc --noEmit` verify applies) copies `reports/grid_app_route/` into grid-app `app/app/<route>/`.

---

## Human prerequisites & gated runs (NOT loop items — do these by hand)

The loop will complete Phases 0–4 on synthetic data unattended. To turn that into the publishable
dollar numbers, a human must:

1. **Stage real data into `experiments/steinmetz_bench/data/`:**
   - ISO LMP / load / congestion snapshots via `gridstatus` (ERCOT West + PJM) for the backtest windows.
   - PyPSA-USA topology/fleet; RTS-GMLC; relevant `pglib-opf` cases.
   - A PJM RTEP (or MISO MTEP) vintage + its approved-project list (for §7.3).
   - CENACE PML history + a PRODESEN corridor list (for §7.4).
2. **Run the real backtests:** `python -m experiments.steinmetz_bench.run_all --real` once data is staged.
3. **Optional large-scale GPU headline:** the modest GPU run (item 2.5) happens in the loop. The full
   WECC-scale 1000-contingency SCED timing + 500-node expansion headline is left optional to avoid
   burning H100 credits unattended — run it by hand via the same `zap-opf-solver` app when desired.
4. **Mount the whitepaper at a gated grid-app route:** copy `reports/grid_app_route/` into grid-app
   `app/app/<route>/` (a gated, auth-required route), add a markdown renderer if grid-app lacks one,
   and verify with grid-app's own `tsc --noEmit && npm run test:unit`. (Cross-repo — see the note
   under item 5.1.)
5. **(Deferred, out of current scope)** Julia validation sandboxes (Sienna UC / PowerModels.jl AC) for a
   true UC/AC fidelity band — excluded per the "self-contained references only" decision; the harness
   leaves a typed hook (`scoring/fidelity.py: register_reference()`) so they can be added later.
