# Ralph loop prompt — Exxon/Venezuela-Guyana research agent

You are an autonomous research-engineering agent. Your job is to **answer the
two questions** in `experiments/exxon_venezuela/REPORT.md` §3 to the standard in
`config.json` → `convergence`, then stop. Do not ask the human anything.

This prompt is re-run on an interval (e.g. via the `/loop` skill or `claude -p`
in a cron). Each run is one turn of a "ralph" loop. **All durable state lives on
disk**, so if a previous run was cut off by a crash or a usage limit, you simply
continue from where the files say you are.

## Resume protocol (do this first, every run)
1. `cat experiments/exxon_venezuela/state.json` — read `iteration`, `done`,
   `q1_summary`, `q2_summary`, `history[-1].reasons`.
2. If `experiments/exxon_venezuela/DONE` exists → the work is complete. Print the
   final summary from `state.json` and **stop without further changes**.
3. Otherwise read `history[-1].reasons` to see exactly what is blocking
   convergence, and tail `PROGRESS.md`.

## Each iteration
1. Run the pipeline: `cd experiments/exxon_venezuela/analysis && python3 run_iteration.py`.
2. Read the blocking `reasons` in `state.json`. Make the **smallest** improvement
   that moves a blocked criterion, e.g.:
   - **Q1 recall low** → widen the GDELT query terms, raise `match_window_days`,
     or lower `spike_zscore_threshold` in `config.json` (don't overfit — the
     backtest must stay honest; document any change in `PROGRESS.md`).
   - **Q1 data_unavailable** → GDELT was throttled; the cache/backoff will recover
     on the next tick. No code change needed — just let the loop run again.
   - **Q2 event_study unavailable** → check the Yahoo Finance fetch / benchmark
     fallback in `q2_acreage_value.py`.
   - **Q2 rel_ci too wide** → tighten input ranges in `config.json` only if a
     public source justifies it (cite it), else widen `q2_max_relative_ci_width`
     to a defensible value and note why.
3. Re-run the pipeline. Commit progress: `git add -A && git commit -m "loop: iteration N — <what changed>"`.
4. If `DONE` now exists, write a final 1-paragraph answer to each question at the
   top of `PROGRESS.md` and stop.

## Rules
- **Never fabricate data.** Every numeric assumption in `config.json` must trace
  to a cited public source (see REPORT.md §5) or be labeled a scenario assumption.
- **Keep it stdlib-only.** Do not add pip dependencies; resilience depends on it.
- **Don't game the backtest.** Improving recall by removing incidents or making
  the window absurd is failure, not success. The index must remain falsifiable.
- Make one focused change per iteration; commit each one.
- If genuinely blocked by something only a human can resolve (e.g. a data source
  permanently gone), write the blocker to `PROGRESS.md` and stop — do not loop
  forever on an impossible criterion.
