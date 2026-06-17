# Exxon × Venezuela/Guyana — open-data research pipeline

A self-contained, **zero-dependency** (Python-stdlib-only) pipeline that answers
the two highest-leverage questions about ExxonMobil's constrained position in the
Venezuela/Guyana theater, validates them against ground truth, and runs inside a
**resilient loop** that resumes autonomously after any interruption.

Start with **[`REPORT.md`](REPORT.md)** for the situation analysis and the
framing of the two questions.

## The two questions
- **Q1 — Early warning.** Is Venezuela→Guyana escalation risk to Exxon's Stabroek
  operations rising or falling now, and does an open-data **Essequibo Tension
  Index** (GDELT) spike around known incidents? → `analysis/q1_tension_index.py`
- **Q2 — Risk pricing.** How many barrels/NPV are frozen by the dispute, what's
  the option value of de-risking, and did XOM mispriced the post-Maduro shift
  (event study vs. an oil/sector benchmark)? → `analysis/q2_acreage_value.py`

## Layout
```
REPORT.md                 # situation report + the two questions + sources
config.json               # all inputs/assumptions (cited) + convergence criteria
state.json                # loop state (atomic, resumable)
PROGRESS.md               # human-readable iteration log
DONE                      # sentinel; created when both questions converge
analysis/
  common.py               # stdlib HTTP w/ retry+backoff+cache, atomic state I/O
  q1_tension_index.py     # GDELT Essequibo Tension Index + incident backtest
  q2_acreage_value.py     # real-options valuation + Monte Carlo + event study
  run_iteration.py        # one loop turn: run Q1+Q2, update state, check convergence
artifacts/                # per-question JSON outputs + HTTP cache
loop/
  run_loop.sh             # resilient pure-Python driver (no model tokens)
  PROMPT.md               # agentic ralph-loop prompt (for /loop or `claude -p`)
```

## Run it
```bash
# one iteration
cd analysis && python3 run_iteration.py

# the resilient loop until converged (survives crashes & rate limits)
bash loop/run_loop.sh
INTERVAL=900 MAX_ITERS=20 bash loop/run_loop.sh
```

## Data sources (open, no key)
- **GDELT 2.0 DOC API** — geopolitical news volume + tone time series (Q1).
- **Yahoo Finance chart API** — XOM, Brent (`BZ=F`), `XLE` prices (Q2).

Network calls retry with exponential backoff and fall back to an on-disk cache,
so the pipeline always returns an answer even when a source is throttled
(GDELT limits to ~1 request / 5s) or briefly unreachable.

## Resilience / "doesn't stop until answered"
Two complementary layers:
1. **Data layer (`loop/run_loop.sh`)** — re-runs the pipeline until the `DONE`
   sentinel appears. Pure stdlib + HTTP, so it uses **no model tokens** and is
   immune to model usage limits. State is on disk → kill and re-run anytime.
2. **Agentic layer (`loop/PROMPT.md`)** — for improving the *method* itself, run
   the prompt on an interval via the `/loop` skill or `claude -p`. If a tick is
   cut off by a usage limit, the next tick reads `state.json` and continues; it
   stops as soon as `DONE` exists.

Convergence (in `config.json`): Q1 backtest recall ≥ threshold **and** has a
current reading; Q2 Monte-Carlo relative CI ≤ threshold **and** the event study
ran; after a minimum number of iterations.
