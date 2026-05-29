## Current item (from experiments/steinmetz_bench/LOOP_QUEUE.md line 42)
- [ ] 2.1 Speed benchmark CPU (ROADMAP §8.4.1) — zap vs cvxpy baseline across sizes.

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

## Constraints
- SCOPE: only create/edit files under experiments/steinmetz_bench/**. NEVER modify the zap library core (zap/**) or any other experiments/** dir. Benchmarks import zap read-only.
- PYTHON: run everything via /home/agent/zap/.venv/bin/python. Solver = cvxpy.CLARABEL or SCS (Mosek only if a license is already present). 100-char lines; ruff is configured; attrs @define dataclasses; snake_case modules.
- DATA: no live MARKET-data calls anywhere in loop work. All acceptance must pass on SYNTHETIC generators + bundled toy/Garver networks. Real-data code paths must raise DataNotStagedError when data/ is empty — never hang, retry, or download.
- MODAL/GPU: only in item 2.5. Dispatch the EXISTING grid-app/infra/modal/solver_app.py exactly once, cache the JSON to data/gpu_runs/, and read only the cache from tests. NEVER call Modal from the pytest verify command. If modal CLI / ~/.modal.toml is absent, write STATUS: partial with a reason — do not fail hard or fabricate GPU numbers.
- ANTI-DEMOWARE (critical): every result number must be COMPUTED by code from an actual zap solve — never a hand-written/expected constant. Tests must re-derive or cross-check the number (finite-difference, recompute, compare to the dual), not merely assert it exists. Do NOT copy the spec's "expected headline" numbers into any report. If you cannot produce a real number on synthetic data, mark the item partial and explain — fabrication is a hard failure.
- BOOKKEEPING: do not edit LOOP_QUEUE.md or loop.sh. You may overwrite LOOP_HANDOFF.md status fields only; the loop owns all bookkeeping commits.
- CROSS-REPO: do not edit the grid-app or opencode repos. Item 5.1 produces the whitepaper + grid_app_route/ bundle as ARTIFACTS inside experiments/steinmetz_bench/ only — mounting into grid-app is a human step.
- COMMITS: conventional-commit messages (e.g. feat(bench): ...). One queue item per iteration; keep changes minimal and additive.

STATUS: done
SUMMARY: bench_speed.py (2.1) times zap dispatch vs an independent CVXPY LP baseline across 3 synthetic sizes, certifying objective parity (max gap ~7e-9 < 1e-2) and emitting a timing-table BenchResult.
ACCEPTANCE: PASS — per-size timing table (zap_s, baseline_s, objective_gap) emitted; objective_gap < 1e-2 on every size (test recomputes from raw objectives); report tags WECC/1000-contingency/Modal-H100 headline as human-gated; full verify green (pytest 49 passed, ruff clean).
VERIFIED: yes
