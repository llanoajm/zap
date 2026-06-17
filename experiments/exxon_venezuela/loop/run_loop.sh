#!/usr/bin/env bash
# Resilient "ralph" loop for the Exxon/Venezuela-Guyana research pipeline.
#
# Runs run_iteration.py over and over until the DONE sentinel appears (both
# questions answered to the standard in config.json). It is:
#   * resumable    - all state is on disk (state.json); kill it anytime, re-run,
#                    it picks up at the next iteration.
#   * fault-tolerant - a failed iteration triggers exponential backoff, not exit.
#   * usage-limit-proof - pure Python/stdlib + HTTP; it consumes no model tokens,
#                    so model usage limits never apply to the data work itself.
#
# Usage:
#   bash loop/run_loop.sh                 # loop until converged
#   MAX_ITERS=10 INTERVAL=900 bash loop/run_loop.sh
#
# Env knobs:
#   INTERVAL   seconds to wait between successful iterations (default 1800)
#   MAX_ITERS  hard cap on iterations (default 0 = unlimited)
#   MAX_BACKOFF max backoff seconds after failures (default 960 = 16 min)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

INTERVAL="${INTERVAL:-1800}"
MAX_ITERS="${MAX_ITERS:-0}"
MAX_BACKOFF="${MAX_BACKOFF:-960}"
LOG="$ROOT/loop/loop.log"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

if [[ -f "$ROOT/DONE" ]]; then
  log "DONE sentinel already present: $(cat "$ROOT/DONE"). Nothing to do."
  exit 0
fi

backoff=30
iters=0
log "Starting resilient loop (INTERVAL=${INTERVAL}s, MAX_ITERS=${MAX_ITERS:-unlimited})"

while [[ ! -f "$ROOT/DONE" ]]; do
  if [[ "$MAX_ITERS" -gt 0 && "$iters" -ge "$MAX_ITERS" ]]; then
    log "Reached MAX_ITERS=$MAX_ITERS without convergence; exiting (resume later by re-running)."
    exit 2
  fi
  iters=$((iters + 1))

  python3 "$ROOT/analysis/run_iteration.py"
  rc=$?

  if [[ -f "$ROOT/DONE" ]]; then
    log "CONVERGED: $(cat "$ROOT/DONE")"
    break
  fi

  if [[ "$rc" -eq 0 || "$rc" -eq 2 ]]; then
    # Clean run, just not converged yet -> normal cadence.
    backoff=30
    log "Iteration $iters done (rc=$rc), not converged. Sleeping ${INTERVAL}s."
    sleep "$INTERVAL"
  else
    # Error -> exponential backoff (capped), then retry.
    log "Iteration $iters errored (rc=$rc). Backing off ${backoff}s."
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    [[ "$backoff" -gt "$MAX_BACKOFF" ]] && backoff="$MAX_BACKOFF"
  fi
done

log "Loop finished. Final state:"
python3 - "$ROOT/state.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(json.dumps({k: s.get(k) for k in ("iteration","done","q1_summary","q2_summary")}, indent=2))
PY
exit 0
