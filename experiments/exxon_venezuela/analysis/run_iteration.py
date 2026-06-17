"""One iteration of the research loop: run Q1 + Q2, update state, check
convergence, and write the DONE sentinel when both questions are answered to
the standard set in config.json.

Idempotent and resumable: reads/writes state.json atomically, so it is safe to
re-run after any interruption (crash, usage limit, container reclaim).
"""

from __future__ import annotations

import os
import sys
import traceback

import common
import q1_tension_index
import q2_acreage_value

DONE_SENTINEL = os.path.join(common.ROOT, "DONE")
PROGRESS = os.path.join(common.ROOT, "PROGRESS.md")


def _check_convergence(cfg, q1, q2):
    c = cfg["convergence"]
    reasons = []
    q1_ok = (q1.get("status") == "ok"
             and (not c["q1_requires_current_reading"] or "current_eti" in q1)
             and q1.get("backtest", {}).get("recall", 0) >= c["q1_min_recall_on_incidents"])
    if not q1_ok:
        reasons.append(
            f"Q1 not converged (status={q1.get('status')}, "
            f"recall={q1.get('backtest', {}).get('recall')} "
            f"need>={c['q1_min_recall_on_incidents']})")
    q2_ok = (q2.get("monte_carlo", {}).get("relative_ci_width", 9e9) <= c["q2_max_relative_ci_width"]
             and (not c["q2_requires_event_study"] or q2.get("event_study", {}).get("status") == "ok"))
    if not q2_ok:
        reasons.append(
            f"Q2 not converged (rel_ci={q2.get('monte_carlo', {}).get('relative_ci_width')} "
            f"need<={c['q2_max_relative_ci_width']}, "
            f"event_study={q2.get('event_study', {}).get('status')})")
    return q1_ok and q2_ok, reasons


def _append_progress(state, q1, q2, converged, reasons):
    it = state["iteration"]
    lines = [
        f"\n## Iteration {it} - {common.now_iso()}",
        f"- Q1 status={q1.get('status')} | current_eti={q1.get('current_eti')} "
        f"({q1.get('regime')}, pct={q1.get('current_eti_percentile')}) | "
        f"backtest recall={q1.get('backtest', {}).get('recall')} "
        f"({q1.get('backtest', {}).get('matched')}/{q1.get('backtest', {}).get('incidents_evaluated')})",
        f"- Q2 EV=${q2.get('expected_value_bn_usd')}bn | "
        f"risk_discount=${q2.get('essequibo_risk_discount_bn_usd')}bn | "
        f"rel_ci={q2.get('monte_carlo', {}).get('relative_ci_width')} | "
        f"event_study={q2.get('event_study', {}).get('status')} "
        f"(CAR={q2.get('event_study', {}).get('car_pct')}%) | "
        f"mispricing={q2.get('mispricing', {}).get('verdict')}",
        f"- Converged: {converged}" + ("" if converged else f" | blocked: {'; '.join(reasons)}"),
    ]
    with open(PROGRESS, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cfg = common.load_config()
    state = common.load_json(common.STATE_PATH, {
        "iteration": 0, "created": common.now_iso(), "history": [], "done": False})

    state["iteration"] += 1
    state["last_run"] = common.now_iso()
    errors = []

    try:
        q1 = q1_tension_index.run(cfg, state)
    except Exception:  # noqa: BLE001
        q1 = {"status": "error"}
        errors.append("q1:\n" + traceback.format_exc())

    try:
        q2 = q2_acreage_value.run(cfg, state)
    except Exception:  # noqa: BLE001
        q2 = {"status": "error", "monte_carlo": {}, "event_study": {}, "mispricing": {}}
        errors.append("q2:\n" + traceback.format_exc())

    min_iter = cfg["convergence"]["min_iterations"]
    converged, reasons = _check_convergence(cfg, q1, q2)
    if state["iteration"] < min_iter:
        converged = False
        reasons.append(f"min_iterations not reached ({state['iteration']}/{min_iter})")

    state["q1_summary"] = {k: q1.get(k) for k in
                           ("status", "current_eti", "regime", "current_eti_percentile")}
    state["q1_summary"]["recall"] = q1.get("backtest", {}).get("recall")
    state["q2_summary"] = {
        "expected_value_bn_usd": q2.get("expected_value_bn_usd"),
        "essequibo_risk_discount_bn_usd": q2.get("essequibo_risk_discount_bn_usd"),
        "relative_ci_width": q2.get("monte_carlo", {}).get("relative_ci_width"),
        "event_study": q2.get("event_study", {}).get("status"),
        "mispricing": q2.get("mispricing", {}).get("verdict"),
    }
    state["history"].append({"iteration": state["iteration"], "ts": common.now_iso(),
                             "converged": converged, "reasons": reasons})
    state["history"] = state["history"][-50:]
    state["done"] = converged
    state["last_errors"] = errors[-3:]
    common.save_json(common.STATE_PATH, state)
    _append_progress(state, q1, q2, converged, reasons)

    if converged:
        with open(DONE_SENTINEL, "w") as f:
            f.write(f"converged at iteration {state['iteration']} {common.now_iso()}\n")
        print(f"[run_iteration] CONVERGED at iteration {state['iteration']}")
        return 0
    print(f"[run_iteration] iteration {state['iteration']} not converged: {'; '.join(reasons)}")
    return 2 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
