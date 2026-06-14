#!/usr/bin/env bash
# Autonomous build loop: repeatedly run a headless Claude agent against
# BUILD_PROMPT.md to build/train/iterate the JEPA x zap OPF surrogate-planner.
# Runs until ALL_DONE sentinel, a wall-clock budget, or repeated hard failures.
#
# Usage: ./ralph/build/build_loop.sh [budget_hours] [max_iters]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/ralph/build"
PROMPT="$BUILD_DIR/BUILD_PROMPT.md"
DONE_FILE="$BUILD_DIR/state/ALL_DONE"
LOG_DIR="$BUILD_DIR/logs"
BRANCH="claude/leworld-lejpa-acpf-surrogate-aj3ew0"

BUDGET_HOURS="${1:-8}"
MAX_ITERS="${2:-60}"
BUDGET_SECS=$(awk "BEGIN{print int($BUDGET_HOURS*3600)}")
START=$(date +%s)

mkdir -p "$LOG_DIR" "$BUILD_DIR/state"
cd "$REPO_ROOT"

push_with_retry() {
    local delay=2
    for _ in 1 2 3 4 5; do
        if git push -u origin "$BRANCH" >>"$LOG_DIR/push.log" 2>&1; then return 0; fi
        sleep "$delay"; delay=$((delay * 2))
    done
    echo "WARN: push failed after retries" >>"$LOG_DIR/loop.log"; return 1
}

echo "=== build loop started $(date -u +%FT%TZ) | budget=${BUDGET_HOURS}h max_iters=$MAX_ITERS ===" >>"$LOG_DIR/loop.log"

fast_fails=0
for i in $(seq 1 "$MAX_ITERS"); do
    elapsed=$(( $(date +%s) - START ))
    if [[ -f "$DONE_FILE" ]]; then
        echo "ALL_DONE found before iter $i; stopping." >>"$LOG_DIR/loop.log"; break
    fi
    if [[ $elapsed -ge $BUDGET_SECS ]]; then
        echo "time budget reached (${elapsed}s >= ${BUDGET_SECS}s); stopping before iter $i." >>"$LOG_DIR/loop.log"; break
    fi

    echo "--- iter $i start $(date -u +%FT%TZ) elapsed=${elapsed}s ---" >>"$LOG_DIR/loop.log"
    iter_start=$(date +%s)
    IS_SANDBOX=1 claude -p "$(cat "$PROMPT")" \
        --dangerously-skip-permissions \
        >"$LOG_DIR/iter_${i}.log" 2>&1
    status=$?
    iter_secs=$(( $(date +%s) - iter_start ))
    echo "--- iter $i exit=$status duration=${iter_secs}s $(date -u +%FT%TZ) ---" >>"$LOG_DIR/loop.log"

    # Safety net: commit anything the agent left uncommitted, then push.
    git add ralph/build >/dev/null 2>&1
    git diff --cached --quiet || git commit -m "build: iter $i autosave" >>"$LOG_DIR/loop.log" 2>&1
    push_with_retry

    # Abort on repeated near-instant failures (CLI/auth broken, not real work).
    if [[ $status -ne 0 && $iter_secs -lt 30 ]]; then
        fast_fails=$((fast_fails + 1))
        if [[ $fast_fails -ge 3 ]]; then
            echo "ABORT: $fast_fails consecutive sub-30s failures." >>"$LOG_DIR/loop.log"; break
        fi
    else
        fast_fails=0
    fi
done

if [[ -f "$DONE_FILE" ]]; then
    echo "=== build loop finished: ALL_DONE ===" >>"$LOG_DIR/loop.log"
else
    echo "=== build loop finished: budget/max-iters/abort ===" >>"$LOG_DIR/loop.log"
fi
git add ralph/build >/dev/null 2>&1
git diff --cached --quiet || git commit -m "build: final autosave" >>"$LOG_DIR/loop.log" 2>&1
push_with_retry
