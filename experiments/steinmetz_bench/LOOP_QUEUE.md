# LOOP_QUEUE — Steinmetz benchmarks (§7 backtests + §8.4 benchmarks)

Derived from `BENCH_ROADMAP.md`. Legend: `[ ]` pending · `[x]` done · `[!]` blocked (needs human).
Each item's full spec + acceptance lives in `BENCH_ROADMAP.md`; read it first.

**Anti-demoware rule (applies to every item):** every number in a result or report must be COMPUTED
by code from an actual zap solve — never a hand-written constant. Tests must re-derive or
cross-check the number, not merely assert it is present. If you cannot produce a real number on
synthetic data, mark the item partial and say why — do not fabricate.

- [x] 0.1 Package skeleton (ROADMAP Phase 0) — create the benchmark package + smoke test.
  - context: `experiments/steinmetz_bench/` package with subpkgs datasets/ scoring/ experiments/ reports/ tests/ and an empty data/ (+ .gitignore for __pycache__/.pytest_cache).
  - acceptance:
    - `.venv/bin/python -m pytest experiments/steinmetz_bench/tests -q` collects ≥1 test and the smoke test passes.
    - `.venv/bin/ruff check experiments/steinmetz_bench` is clean.
    - `data/README.md` exists explaining what a human stages there.
- [ ] 0.2 Dataset registry + loaders (ROADMAP Phase 0) — resolve a DatasetSpec to a zap network + frames.
  - context: datasets/registry.py: synthetic generator (n_nodes, hours, congested, seed) wrapping toy/Garver importers + cache loaders reading data/<name>/.
  - acceptance:
    - test loads a synthetic 5-node net and the Garver net; asserts expected device counts/shapes.
    - missing-cache path raises `DataNotStagedError` with a clear message (test asserts the raise).
- [ ] 0.3 Scoring harness (ROADMAP Phase 0) — counterfactual delta, bootstrap CI, duration curve, fidelity band.
  - context: scoring/metrics.py with counterfactual_delta(), bootstrap_ci(), duration_curve(), fidelity_band().
  - acceptance:
    - unit test: bootstrap_ci returns lo<=mid<=hi on a synthetic array; duration_curve is monotone non-increasing.
- [ ] 0.4 Result schema + report writer (ROADMAP Phase 0) — BenchResult dataclass + JSON + md stub.
  - context: reports/result.py: BenchResult(experiment_id, dataset, headline_number, units, ci, fidelity_band, assumptions, sensitivities).
  - acceptance:
    - BenchResult round-trips to/from JSON (test); a written markdown stub file exists and re-parses.
- [ ] 1.1 PyPSA LP roundtrip reference (ROADMAP Phase 1) — zap-vs-PyPSA LMP/flow gap on a bundled net.
  - context: experiments/ref_pypsa.py solving a bundled net in PyPSA and aligning to zap's solve.
  - acceptance:
    - on a bundled net, max LMP gap < 1e-2 and max flow gap < 1e-3 (test asserts), emits a BenchResult.
- [ ] 1.2 Gradient-vs-exact-dual check (ROADMAP Phase 1 / §8.4.4) — adjoint grad == dual identity.
  - context: experiments/grad_check.py: d(cost)/d(line cap) via zap adjoint vs line-limit dual mu (and gen cap, battery power).
  - acceptance:
    - max relative gradient error < 1e-3 on Garver for all three device types (test asserts), emits BenchResult.
- [ ] 1.3 Realized-LMP comparator (ROADMAP Phase 1) — zap-vs-realized LMP error distribution.
  - context: experiments/realized_lmp.py computing per-node/hour error from a price_frame (synthetic fixture now).
  - acceptance:
    - on a synthetic price_frame fixture, emits an error distribution (mean/median/p90); missing-cache path blocks via DataNotStagedError, not test failure.
- [ ] 2.1 Speed benchmark CPU (ROADMAP §8.4.1) — zap vs cvxpy baseline across sizes.
  - context: experiments/bench_speed.py timing zap vs cvxpy LP (Mosek if licensed else CLARABEL/ECOS) over >=3 sizes.
  - acceptance:
    - emits a timing table (per size: zap_s, baseline_s, objective_gap); objective_gap < 1e-2 every size (test asserts).
- [ ] 2.2 Planning benchmark (ROADMAP §8.4.2) — gradient planner vs baseline on small expansion.
  - context: experiments/bench_planning.py: gradient planner vs brute/implicit-diff baseline, multi-scenario, gen+line+battery caps.
  - acceptance:
    - planner final objective <= baseline + tol (test asserts); converges within a fixed iteration budget; timing recorded in BenchResult.
- [ ] 2.3 Accuracy benchmark (ROADMAP §8.4.3) — LMP/congestion error distributions vs PyPSA and realized.
  - context: experiments/bench_accuracy.py assembling error vs ref_pypsa (1.1) and realized (1.3) as distributions.
  - acceptance:
    - emits a distribution BenchResult on fixtures; --real path parameterized (human) but --synthetic passes in the loop (test asserts synthetic path).
- [ ] 2.4 Sensitivity-correctness report (ROADMAP §8.4.4) — published per-device-type gradient-error table.
  - context: experiments/bench_sensitivity.py wrapping 1.2 into a table.
  - acceptance:
    - table BenchResult with per-device max-error < 1e-3 (test asserts); report file written.
- [ ] 2.5 GPU speed via Modal (ROADMAP §8.4.1 headline) — dispatch zap-opf-solver on H100, cache result.
  - context: experiments/bench_gpu_modal.py calling the EXISTING grid-app/infra/modal/solver_app.py (solve_direct.remote or `modal run ...::smoke`) on a modest + one larger net; run Modal ONCE, cache to data/gpu_runs/<ts>.json. The TEST reads only the cache, never calls Modal. Bounded sizes only (cost guard).
  - acceptance:
    - if `modal` CLI + ~/.modal.toml present: data/gpu_runs/*.json exists with machine=="cuda", an elapsed_s, CPU-vs-GPU objective gap < 1e-2; emits BenchResult.
    - if Modal unavailable: self-mark partial/blocked (reason logged) rather than failing the verify.
- [ ] 3.1 Data-center siting backtest (ROADMAP §7.1-A) — rank nodes by LMP duration curve + curtailment.
  - context: experiments/bt_datacenter_siting.py ranking candidate nodes over scenarios, $/MWh delta vs default node.
  - acceptance:
    - on a synthetic net with one deliberately cheap node, the recommended node == that node (test asserts); emits $/MWh delta with bootstrap CI.
- [ ] 3.2 Data-center flexibility & battery sizing (ROADMAP §7.1-B) — PowerTarget flex load + battery break-even.
  - context: experiments/bt_datacenter_flex.py: d(savings)/d(battery MW) break-even + firm-vs-flexible $/yr.
  - acceptance:
    - a break-even battery size is found where marginal value ~ marginal cost; firm-vs-flexible $/yr delta with CI; the gradient is finite-difference-checked (test asserts FD agreement).
- [ ] 3.3 Vertically-integrated utility backtest (ROADMAP §7.2) — SCED/PCM vs suboptimal actual + expansion ranking.
  - context: experiments/bt_utility.py: synthetic fleet+load, least-cost dispatch vs a deliberately-suboptimal "actual"; 5-yr expansion ranking.
  - acceptance:
    - modeled least-cost <= "actual" cost (test asserts); expansion returns a ranked list; PyPSA roundtrip gap < tol; avoided-fuel $/yr + NPV-delta with fidelity band emitted.
- [ ] 3.4 Transmission-plan audit backtest (ROADMAP §7.3) — corridor ranking vs realized congestion.
  - context: experiments/bt_transmission_audit.py: rank corridors by d(system cost)/d(line cap); correlate with synthetic realized-congestion; R2 + missed-corridor count.
  - acceptance:
    - the known-congested corridor ranks #1 (test asserts); rank-correlation with the realized-congestion vector exceeds a threshold; emits R2 BenchResult.
- [ ] 3.5 Mexico EPC dual-regime backtest (ROADMAP §7.4) — corridor ranking under merit-order vs CFE>=54%.
  - context: experiments/bt_mexico_epc.py: rank corridors under (a) merit-order and (b) CFE->=54%-share dispatch constraint; ranking agreement + congestion-relief $/yr.
  - acceptance:
    - both regimes produce rankings; the CFE constraint is demonstrably binding (prices/dispatch shift measurably between regimes — test asserts a measurable shift); a ranking-agreement metric is emitted.
- [ ] 4.1 Fidelity band on every result (ROADMAP Phase 4) — ensure all results carry a fidelity_band.
  - context: every BenchResult from Phases 1-3 carries a non-null fidelity_band (DC-vs-PyPSA, DC-vs-realized where applicable).
  - acceptance:
    - a test iterates all produced results and asserts fidelity_band present and well-formed.
- [ ] 4.2 Master report generator (ROADMAP Phase 4) — run_all.py assembling all results.
  - context: run_all.py building reports/STEINMETZ_BENCH.md with §8.4 tables + §7 headlines, CIs, fidelity bands; --synthetic (loop) / --real (human).
  - acceptance:
    - `python -m experiments.steinmetz_bench.run_all --synthetic` exits 0 and writes a report containing all 10 experiment ids (test asserts each id present).
- [ ] 5.1 Finished whitepaper (ROADMAP Phase 5) — grounded rewrite of the spec from real loop results.
  - context: reports/build_whitepaper.py producing reports/STEINMETZ_WHITEPAPER.md — §7/§8.4 sections replaced with ACTUAL BenchResult numbers (CI + fidelity band, synthetic-vs-real labelled); §5 architecture rewritten grounded in real code (zap.network dispatch+KKT adjoint, PlanningProblem, PowerTarget, ADMMSolver, the zap-opf-solver Modal app incl. its gotchas, opencode harness, grid-app). Also emit reports/grid_app_route/ bundle (md copy + minimal page scaffold) for a human to mount at a gated grid-app route. Do NOT edit grid-app itself.
  - acceptance:
    - a test parses STEINMETZ_WHITEPAPER.md and asserts every benchmark/backtest headline is traceable to a real BenchResult id in the results JSON (no prose number absent from JSON); each cited value carries a CI and fidelity_band; synthetic-vs-real provenance is labelled; the architecture section references real symbols (grep zap.network, PlanningProblem, PowerTarget, ADMMSolver, zap-opf-solver). Build fails if any headline is hand-written rather than data-derived.
