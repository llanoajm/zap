# Build-loop iteration — JEPA × zap OPF surrogate/planner

You are ONE iteration of an autonomous build loop running in the `zap` repo. Your job is to **build,
train, and iterate on real machine-learning architectures** toward the north star, on **real Microsoft
GridSFM data** solved with `zap` for exact ground-truth. This is engineering, not a literature review.

## Orient first (do this every time, it's cheap)

1. Read `ralph/build/ROADMAP.md` — the phases, ground rules, and the north star.
2. List `ralph/build/state/` — `*.done` sentinels and `*_metrics.json` tell you what's already finished.
   Read the most recent `state/PROGRESS.md` if it exists.
3. Read `ralph/build/src/gridsfm_zap.py` (the data→zap bridge) and any modules from earlier phases so
   you reuse them instead of rewriting.

## Then act

4. Pick the **single highest-value next step**: the first incomplete phase, OR a concrete improvement to
   a completed phase (better features, fixing a fidelity bug, a stronger architecture, an ablation).
   Prefer finishing a phase end-to-end over starting many.
5. **Write/modify code under `ralph/build/` only.** Keep models small (CPU, 15 GB RAM, 4 cores).
   Reuse the converter; cache expensive `zap` solves under `ralph/build/state/cache/`.
6. **Run it.** Actually execute training/eval. Capture real numbers. If a run would take too long, shrink
   it (fewer states, fewer epochs, smaller hidden dim) rather than faking results — but make it real.
7. **Record honestly.** Write `ralph/build/state/<phase>_metrics.json` with the numbers your run
   produced. Update `ralph/build/state/PROGRESS.md` with: what you did, what the numbers say, what's
   next, and any dead ends. If something underperforms or fails, say so.
8. When a phase is genuinely complete (code runs, metrics recorded), `touch ralph/build/state/<phase>.done`.
9. Commit: `git add ralph/build && git commit -m "build: <phase> — <one-line result>"`. Do NOT push
   (the outer loop pushes). Do NOT create PRs or forks. Do NOT touch the upstream/Microsoft repos.

## Engineering standards

- **Real metrics only.** Never write a number into a metrics file that code didn't produce this run.
- **Cross-topology is the point.** Always evaluate on held-out TEST states from `data/raw/split.json`,
  not just training states.
- **The comparison that matters:** supervised surrogate (P2, no solver) vs Design A (P3/P4, through the
  exact `zap` solver). Quantify: cost gap vs `zap`, LMP/dual accuracy, feasibility, generalization gap.
- **Be skeptical of your own model.** Run the "is Design A just a warm start?" check (does the decoder
  collapse to passing through true parameters?). Report active-set / sign-flip behavior where relevant.
- **Idempotent.** Re-running an iteration must not corrupt prior work. Guard expensive steps with cache
  checks. Append to `PROGRESS.md`, don't clobber it.
- If you hit an environment limit (memory, time, a missing package you can't pip install), record the
  limit in `PROGRESS.md`, scope the experiment down to something real, and proceed.

## Termination

When **all** phases P1–P7 are done and `ralph/build/RESULTS.md` answers the north star with measured,
cross-topology numbers backed by `state/*_metrics.json`, write `ralph/build/state/ALL_DONE` with a
one-paragraph honest conclusion. Only then. A reviewer should be able to reproduce every number.
