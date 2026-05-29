## Current item (from experiments/steinmetz_bench/LOOP_QUEUE.md line 83)
- [ ] 4.1 Fidelity band on every result (ROADMAP Phase 4) — ensure all results carry a fidelity_band.

## Attempt
2 of 5

## Context to load before working
- experiments/steinmetz_bench/BENCH_ROADMAP.md   (THE roadmap — full per-item spec, acceptance criteria, guardrails, the synthetic-first design principle; READ THIS FIRST and find your current item)
- AGENTS.md                                       (zap project guidance: ALWAYS run Python via ./.venv/bin/python; 100-char lines, ruff configured; attrs @define dataclasses; prefer cvxpy.CLARABEL/SCS when no Mosek; finite-difference test pattern lives in zap/tests/test_network.py)
- README.md                                       (zap scope: differentiable DC-OPF + gradients + gradient-based planning)
- /home/agent/grid-app/infra/modal/solver_app.py  (ONLY for item 2.5 — the EXISTING zap-opf-solver Modal app; call solve_direct.remote(network_nc, args, import_args))
- /home/agent/grid-app/infra/modal/README.md       (ONLY for item 2.5 — deploy/run recipe + caveats)
- /home/agent/grid-app/infra/modal/PARITY_REPORT.md (ONLY for item 2.5 — prior CPU-vs-GPU parity results to build on, not redo)
- /home/agent/.claude/projects/-home-agent/memory/MEMORY.md  (durable project principles)
- experiments/steinmetz_bench/LOOP_QUEUE.md            (the queue you're working from)
- recent tail of experiments/steinmetz_bench/LOOP_JOURNAL.md

## Protocol
1. Read the context above plus any acceptance criteria nested under the
   current item in experiments/steinmetz_bench/LOOP_QUEUE.md.
2. Implement the item against those acceptance criteria. Run `cd /home/agent/zap && .venv/bin/python -m pytest experiments/steinmetz_bench/tests -q && .venv/bin/ruff check experiments/steinmetz_bench`
   (and any other checks the criteria name) before concluding.
3. Commit your code changes with a descriptive conventional-commit message.
4. Overwrite experiments/steinmetz_bench/LOOP_HANDOFF.md to end with EXACTLY these fields, one per line:
   STATUS: done | partial
   SUMMARY: <1 sentence, will be embedded in the loop's tag commit>
   NEXT_STEPS: <only if partial; concrete handoff for the next agent>
   ACCEPTANCE: <which criteria pass, which don't>
   Do NOT commit experiments/steinmetz_bench/LOOP_HANDOFF.md — the loop owns the bookkeeping commit.

## Result
STATUS: done
SUMMARY: Item 4.1 — added a DC-vs-PyPSA nodal-LMP fidelity_band to the siting backtest (3.1, previously the only Phase 1-3 result with fidelity_band=None; band max ~0.01 $/MWh over 240 real zap-vs-PyPSA prices) and tests/test_fidelity_bands.py that runs every Phase 1-3 experiment and asserts each BenchResult carries a present, well-formed, JSON-roundtrip-safe fidelity_band.
ACCEPTANCE: All pass. The new test iterates all Phase 1-3 results (1.1-3.5; GPU 2.5 included via its cached run) and asserts fidelity_band is non-null and well-formed (non-empty reference/metric, n>0, finite non-negative gaps with max>=p90 and max>=mean); bands are computed from real solves, not constants. Full verify green: 133 passed in 413s, ruff clean.

## Constraints
- SCOPE: only create/edit files under experiments/steinmetz_bench/**. NEVER modify the zap library core (zap/**) or any other experiments/** dir. Benchmarks import zap read-only.
- PYTHON: run everything via /home/agent/zap/.venv/bin/python. Solver = cvxpy.CLARABEL or SCS (Mosek only if a license is already present). 100-char lines; ruff is configured; attrs @define dataclasses; snake_case modules.
- DATA: no live MARKET-data calls anywhere in loop work. All acceptance must pass on SYNTHETIC generators + bundled toy/Garver networks. Real-data code paths must raise DataNotStagedError when data/ is empty — never hang, retry, or download.
- MODAL/GPU: only in item 2.5. Dispatch the EXISTING grid-app/infra/modal/solver_app.py exactly once, cache the JSON to data/gpu_runs/, and read only the cache from tests. NEVER call Modal from the pytest verify command. If modal CLI / ~/.modal.toml is absent, write STATUS: partial with a reason — do not fail hard or fabricate GPU numbers.
- ANTI-DEMOWARE (critical): every result number must be COMPUTED by code from an actual zap solve — never a hand-written/expected constant. Tests must re-derive or cross-check the number (finite-difference, recompute, compare to the dual), not merely assert it exists. Do NOT copy the spec's "expected headline" numbers into any report. If you cannot produce a real number on synthetic data, mark the item partial and explain — fabrication is a hard failure.
- BOOKKEEPING: do not edit LOOP_QUEUE.md or loop.sh. You may overwrite LOOP_HANDOFF.md status fields only; the loop owns all bookkeeping commits.
- CROSS-REPO: do not edit the grid-app or opencode repos. Item 5.1 produces the whitepaper + grid_app_route/ bundle as ARTIFACTS inside experiments/steinmetz_bench/ only — mounting into grid-app is a human step.
- COMMITS: conventional-commit messages (e.g. feat(bench): ...). One queue item per iteration; keep changes minimal and additive.
VERIFIED: yes
