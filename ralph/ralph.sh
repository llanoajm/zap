#!/usr/bin/env bash
# Ralph loop: repeatedly run a headless Claude agent against PROMPT.md until it
# declares the research done (writes ralph/DONE) or MAX_ITERS is hit.
# Usage: ./ralph/ralph.sh [max_iters]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RALPH_DIR="$REPO_ROOT/ralph"
PROMPT="$RALPH_DIR/PROMPT.md"
DONE_FILE="$RALPH_DIR/DONE"
LOG_DIR="$RALPH_DIR/logs"
BRANCH="claude/leworld-lejpa-acpf-surrogate-aj3ew0"
MAX_ITERS="${1:-12}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

push_with_retry() {
    local delay=2
    for _ in 1 2 3 4 5; do
        if git push -u origin "$BRANCH" >>"$LOG_DIR/push.log" 2>&1; then
            return 0
        fi
        sleep "$delay"
        delay=$((delay * 2))
    done
    echo "WARN: push failed after retries" | tee -a "$LOG_DIR/push.log"
    return 1
}

echo "=== ralph loop started $(date -u +%FT%TZ) | max_iters=$MAX_ITERS ===" >>"$LOG_DIR/loop.log"

for i in $(seq 1 "$MAX_ITERS"); do
    if [[ -f "$DONE_FILE" ]]; then
        echo "DONE sentinel found before iter $i; stopping." >>"$LOG_DIR/loop.log"
        break
    fi

    echo "--- iter $i start $(date -u +%FT%TZ) ---" >>"$LOG_DIR/loop.log"
    claude -p "$(cat "$PROMPT")" \
        --dangerously-skip-permissions \
        >"$LOG_DIR/iter_${i}.log" 2>&1
    status=$?
    echo "--- iter $i exit=$status $(date -u +%FT%TZ) ---" >>"$LOG_DIR/loop.log"

    # Safety net: commit anything the agent left uncommitted, then push progress.
    git add ralph/ >/dev/null 2>&1
    git diff --cached --quiet || git commit -m "ralph: iter $i progress" >>"$LOG_DIR/loop.log" 2>&1
    push_with_retry

    # If the CLI itself is failing (auth/network), don't burn iterations.
    if [[ $status -ne 0 && ! -s "$LOG_DIR/iter_${i}.log" ]]; then
        echo "claude CLI produced no output and failed; aborting loop." >>"$LOG_DIR/loop.log"
        break
    fi
done

if [[ -f "$DONE_FILE" ]]; then
    echo "=== ralph loop finished: DONE ===" >>"$LOG_DIR/loop.log"
else
    echo "=== ralph loop finished: max iterations or abort ===" >>"$LOG_DIR/loop.log"
fi
git add ralph/ >/dev/null 2>&1
git diff --cached --quiet || git commit -m "ralph: final state" >>"$LOG_DIR/loop.log" 2>&1
push_with_retry
