# Improvement-loop iteration — push the JEPA × zap OPF results

The 7 build phases are DONE (`ralph/build/RESULTS.md`, `state/P*_metrics.json`). A working,
honest baseline exists. Your job now is to **relentlessly improve the headline numbers** through
real experiments and architecture iteration — NOT to redo the phases or rewrite the report from
scratch. This continues an autonomous run toward the north star; the user is away.

## The north star, as concrete targets

On **held-out TEST states** (cross-topology), starting from iter-1 results, drive:

| Metric | iter-1 baseline | target |
|---|---|---|
| Cross-topology **cost gap** vs zap | 0.63 (63%) | **< 0.20** |
| **Power balance** error | ~0 (Design A) | keep ~0 |
| **LMP / dual** MAE | 1.32 | **halve it** |
| Planning-gradient **cosine** vs exact zap (P6) | 0.76 | **> 0.95** |
| Usable samples (un-filtered) | 30/36 train | **raise** (fix fidelity) |

You will not hit all of these; **make measurable progress on at least one per iteration** and
record it. Honesty over optimism — a well-measured negative result is valuable.

## Highest-value levers (the iter-1 results point here)

1. **Fix the cost-mapping fidelity in `src/gridsfm_zap.py`.** The quality filter drops cases
   (vermont/utah/new_mexico/massachusetts) because of bad gen-cost normalization and VOLL scaling,
   producing extreme/negative LMPs. Map MATPOWER gencost → zap `linear_cost` correctly (units,
   p.u., baseMVA), validate against each case's GridSFM `dc_results` objective. This is the root
   cause of the 63% cost gap and the i.i.d./LMP noise downstream.
2. **Better cost predictor.** iter-1 says "the bottleneck is cost parameter prediction quality."
   Try: supervise the decoder *directly* on true (normalized) gen costs as an auxiliary loss before/
   alongside the through-solver loss; richer node features (fuel_type, pmax/pmin, true cost coeffs,
   historical pg from the AC/DC solution as context); deeper/edge-aware message passing.
3. **Active-set fidelity (drives P6 gradients).** The planning-gradient magnitude error (9.9×) comes
   from putting too many generators at pmax. Add a loss/penalty that matches the binding set, or
   calibrate cost spread so the merit order matches.
4. **Warm-start that actually speeds up (P5).** CVXPY param-changes gave no speedup. Try zap's ADMM
   path (`zap/planning/problem_admm.py`) with a predicted primal/dual initialization and measure
   iteration reduction — the mechanism GridSFM's 1.66× uses.
5. **JEPA, done right (P4 hurt).** Either fix it (per-graph standardization so SIGReg's i.i.d.
   assumption holds; slower EMA; predictor capacity) or document conclusively *why* it doesn't help
   here. Both are acceptable outcomes if measured.

## Rules

- Read `ROADMAP.md`, `RESULTS.md`, `state/PROGRESS.md`, and `state/LEADERBOARD.md` (create if absent)
  first. Reuse existing modules in `src/`. Only touch files under `ralph/build/`.
- **Run real code, record real numbers.** Update the relevant `state/P*_metrics.json` (or add a new
  `state/improve_<topic>_metrics.json`) with this run's actual outputs. Never fabricate.
- Maintain `state/LEADERBOARD.md`: one row per experiment with the metric(s) moved, the change made,
  and whether it helped. Keep the current best clearly marked.
- Keep `RESULTS.md` updated as the single source of truth when a number materially changes.
- CPU-only, 15 GB RAM, 4 cores: keep runs to a few minutes; cache zap solves under `state/cache/`.
- Commit: `git add ralph/build && git commit -m "improve: <lever> — <result>"`. Do NOT push (outer
  loop pushes). No PRs, no forks, never touch upstream/Microsoft repos.

## Termination

This is a long improvement run. Do **not** write any global "done" sentinel — keep iterating; the
outer loop stops on its time budget. Each iteration should leave the repo with one more real,
recorded experiment and an updated leaderboard.
