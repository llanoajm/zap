"""Q1 - Essequibo Tension Index (ETI): an open-data early-warning signal.

Question: Is Venezuela->Guyana escalation risk to Exxon's Stabroek operations
rising or falling right now, and does an open-data index spike *around* known
incidents (so it could plausibly flag them)?

Data: GDELT 2.0 DOC API timeline modes (free, no key, ~15-min cadence, global).
  - timelinevol : share of worldwide news coverage matching the query (intensity)
  - timelinetone: average sentiment tone of that coverage (negative => conflictual)

ETI(t) = z(volume_t) + z(-tone_t)   -- more coverage + more negative tone = hotter.

Validation: spikes (ETI > threshold) are matched against a labeled incident
calendar (config.json). Recall = fraction of in-window incidents with a nearby
spike. This is the falsifiable check the brief asks for.
"""

from __future__ import annotations

import statistics
import urllib.parse
from datetime import datetime, timedelta, timezone

import common

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _fetch_timeline(query: str, start: str, end: str, mode: str):
    qs = urllib.parse.urlencode({
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
    })
    url = f"{GDELT_DOC}?{qs}"
    # GDELT asks for <=1 request / 5s; throttle + cache for politeness & resilience.
    data, from_cache = common.http_get_json(
        url, cache_key=f"gdelt_{mode}_{query}_{start}_{end}",
        cache_ttl_sec=6 * 3600, throttle_sec=6.0, base_delay=6.0,
    )
    series = {}
    if not data or "timeline" not in data:
        return series, from_cache
    for block in data["timeline"]:
        for pt in block.get("data", []):
            raw = str(pt.get("date", ""))
            # GDELT timeline dates come as "20260601T000000Z" or "2026-06-01T..".
            datepart = raw.split("T")[0].replace("-", "")
            if len(datepart) >= 8 and datepart[:8].isdigit():
                day = f"{datepart[:4]}-{datepart[4:6]}-{datepart[6:8]}"
            else:
                continue
            try:
                series[day] = float(pt.get("value", 0.0))
            except (TypeError, ValueError):
                continue
    return series, from_cache


def _zscores(values):
    if len(values) < 3:
        return [0.0] * len(values)
    mu = statistics.mean(values)
    sd = statistics.pstdev(values) or 1.0
    return [(v - mu) / sd for v in values]


def run(config, state):
    cfg = config["q1_gdelt"]
    incidents = config["q1_known_incidents"]
    pair = cfg["primary_pair"]
    query = " ".join(pair)  # GDELT: space => both terms present

    today = datetime.now(timezone.utc).date()
    # Window covers labeled incidents within GDELT 2.0 coverage (>= 2017).
    in_window_incidents = [i for i in incidents
                           if datetime.strptime(i["date"], "%Y-%m-%d").date() >= datetime(2017, 1, 1).date()]
    earliest = min(datetime.strptime(i["date"], "%Y-%m-%d").date() for i in in_window_incidents)
    start_date = min(earliest - timedelta(days=30), today - timedelta(days=cfg["lookback_days"]))
    start = start_date.strftime("%Y%m%d000000")
    end = today.strftime("%Y%m%d000000")

    vol, vc = _fetch_timeline(query, start, end, "timelinevol")
    tone, tc = _fetch_timeline(query, start, end, "timelinetone")

    result = {
        "as_of": common.now_iso(),
        "query": query,
        "window": {"start": start_date.isoformat(), "end": today.isoformat()},
        "data_available": bool(vol),
        "from_cache": vc or tc,
    }

    if not vol:
        result["status"] = "data_unavailable"
        result["note"] = "GDELT returned no data (throttled/unreachable); loop will retry."
        common.save_json(common.ARTIFACT_DIR + "/q1_tension_index.json", result)
        return result

    days = sorted(vol.keys())
    vol_series = [vol[d] for d in days]
    tone_series = [tone.get(d, 0.0) for d in days]

    zv = _zscores(vol_series)
    zt = _zscores([-t for t in tone_series])  # negate: negative tone -> high stress
    eti = [a + b for a, b in zip(zv, zt)]
    eti_by_day = dict(zip(days, eti))

    thr = cfg["spike_zscore_threshold"]
    spikes = [d for d, v in eti_by_day.items() if v >= thr]

    # Backtest: did a spike land within match_window_days of each incident?
    win = cfg["match_window_days"]
    matched, detail = 0, []
    for inc in in_window_incidents:
        idate = datetime.strptime(inc["date"], "%Y-%m-%d").date()
        if idate > today:
            continue
        hit = None
        for s in spikes:
            sd = datetime.strptime(s, "%Y-%m-%d").date()
            if abs((sd - idate).days) <= win:
                hit = s
                break
        matched += 1 if hit else 0
        detail.append({"incident": inc["date"], "label": inc["label"],
                       "spike_detected": hit is not None, "nearest_spike": hit})
    n_eval = sum(1 for inc in in_window_incidents
                 if datetime.strptime(inc["date"], "%Y-%m-%d").date() <= today)
    recall = matched / n_eval if n_eval else 0.0

    # Current reading + regime.
    recent = eti[-14:] if len(eti) >= 14 else eti
    half = max(1, len(recent) // 2)
    slope = (statistics.mean(recent[half:]) - statistics.mean(recent[:half]))
    sorted_eti = sorted(eti)
    cur = eti[-1]
    pct = sum(1 for v in sorted_eti if v <= cur) / len(sorted_eti)

    result.update({
        "status": "ok",
        "n_days": len(days),
        "current_eti": round(cur, 3),
        "current_eti_percentile": round(pct, 3),
        "regime": "rising" if slope > 0.15 else ("falling" if slope < -0.15 else "flat"),
        "regime_slope_14d": round(slope, 3),
        "spike_threshold_z": thr,
        "n_spikes": len(spikes),
        "recent_spikes": spikes[-5:],
        "backtest": {"incidents_evaluated": n_eval, "matched": matched,
                     "recall": round(recall, 3), "match_window_days": win,
                     "detail": detail},
        "latest_days": {d: round(eti_by_day[d], 2) for d in days[-10:]},
    })
    common.save_json(common.ARTIFACT_DIR + "/q1_tension_index.json", result)
    return result


if __name__ == "__main__":
    cfg = common.load_config()
    st = common.load_json(common.STATE_PATH, {})
    r = run(cfg, st)
    print(common.__dict__["json"].dumps(r, indent=2) if False else r["status"])
    print("current_eti=", r.get("current_eti"), "regime=", r.get("regime"),
          "recall=", r.get("backtest", {}).get("recall"))
