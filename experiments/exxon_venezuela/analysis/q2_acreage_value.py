"""Q2 - Disputed-acreage value, risk discount, and XOM mispricing test.

Question: How many barrels / how much NPV are frozen by the Essequibo dispute,
what is the option value of de-risking it, and did XOM's market price reflect
the post-Maduro shift on 3 Jan 2026 (is the geopolitical discount mispriced)?

Three pieces, all from open data + cited assumptions (config.json):
  1. Real-options EV of the frozen Stabroek acreage net to Exxon (45% WI),
     probability-weighted over unlocked / frozen / conflict scenarios. Scenario
     probabilities are updated by the Q1 tension reading (couples the two models).
  2. Monte-Carlo over uncertain inputs -> distribution + confidence interval.
  3. Event study: XOM abnormal return vs an oil/sector benchmark around the
     3 Jan 2026 intervention -> what the market actually repriced. Compared to
     the model's implied de-risking value to flag cheap / fair / rich.
"""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone

import common

YF = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1d"


def _fetch_prices(symbol):
    url = YF.format(sym=symbol)
    data, from_cache = common.http_get_json(
        url, cache_key=f"yf_{symbol}", cache_ttl_sec=6 * 3600, base_delay=3.0)
    if not data:
        return {}, from_cache
    try:
        res = data["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]["close"]
        adj = res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", q)
        out = {}
        for t, c in zip(ts, adj if adj else q):
            if c is None:
                continue
            d = datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()
            out[d] = float(c)
        return out, from_cache
    except (KeyError, IndexError, TypeError):
        return {}, from_cache


def _log_returns(prices):
    days = sorted(prices.keys())
    rets = {}
    for i in range(1, len(days)):
        p0, p1 = prices[days[i - 1]], prices[days[i]]
        if p0 and p1:
            rets[days[i]] = math.log(p1 / p0)
    return rets


def _ols_beta(y, x):
    """Beta and alpha of y on x over common dates."""
    common_days = sorted(set(y) & set(x))
    ys = [y[d] for d in common_days]
    xs = [x[d] for d in common_days]
    if len(xs) < 10:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    var = sum((a - mx) ** 2 for a in xs)
    if var == 0:
        return None
    beta = cov / var
    alpha = my - beta * mx
    return alpha, beta, common_days


def _discount_factor(r, years):
    return 1.0 / ((1.0 + r) ** years)


def _scenario_probs(base, q1):
    """Tilt scenario probabilities by the Q1 tension reading."""
    base = {k: v for k, v in base.items() if not k.startswith("_")}
    p = dict(base)
    if q1 and q1.get("status") == "ok":
        pct = q1.get("current_eti_percentile", 0.5)
        shift = (pct - 0.5)
        if q1.get("regime") == "rising":
            shift += 0.10
        elif q1.get("regime") == "falling":
            shift -= 0.10
        p["unlocked"] = base["unlocked"] - 0.40 * shift
        p["frozen"] = base["frozen"] + 0.30 * shift
        p["conflict"] = base["conflict"] + 0.10 * shift
    p = {k: max(0.0, v) for k, v in p.items()}
    tot = sum(p.values()) or 1.0
    return {k: v / tot for k, v in p.items()}, (q1.get("regime") if q1 else None)


def _value_unlocked(v, disputed_frac, netback):
    """PV (USD bn) of the disputed acreage net to Exxon if unlocked."""
    boe_bn = v["stabroek_recoverable_bn_boe"] * v["exxon_working_interest"] * disputed_frac
    unit_margin = netback - v["development_cost_per_boe_usd"]  # USD/boe
    df = _discount_factor(v["discount_rate_annual"], v["years_to_first_oil_if_unlocked"])
    return boe_bn * unit_margin * df  # USD billion


def _tri(low, mode, high):
    return random.triangular(low, high, mode)


def run(config, state):
    v = config["q2_valuation"]
    es = config["q2_event_study"]
    q1 = common.load_json(common.ARTIFACT_DIR + "/q1_tension_index.json", {})

    probs, regime = _scenario_probs(v["scenario_probabilities"], q1)

    # ---- Point estimate -------------------------------------------------
    base_unlocked = _value_unlocked(v, v["disputed_acreage_fraction_of_resource"],
                                    v["netback_per_boe_usd"])
    scen_vals = {
        "unlocked": base_unlocked,
        "frozen": 0.0,
        "conflict": v["conflict_value_multiplier"] * base_unlocked,
    }
    ev_bn = sum(probs[k] * scen_vals[k] for k in probs)
    risk_discount_bn = base_unlocked - ev_bn  # haircut the dispute imposes
    per_share = ev_bn / v["shares_outstanding_bn"]
    unlocked_per_share = base_unlocked / v["shares_outstanding_bn"]

    # ---- Monte Carlo over uncertain inputs ------------------------------
    n = 20000
    samples = []
    for _ in range(n):
        df_frac = _tri(v["disputed_fraction_low"],
                       v["disputed_acreage_fraction_of_resource"],
                       v["disputed_fraction_high"])
        nb = _tri(v["netback_low"], v["netback_per_boe_usd"], v["netback_high"])
        # Dirichlet-style noise on probabilities.
        noisy = {k: max(1e-6, probs[k] * random.uniform(0.6, 1.4)) for k in probs}
        tot = sum(noisy.values())
        noisy = {k: val / tot for k, val in noisy.items()}
        u = _value_unlocked(v, df_frac, nb)
        sv = {"unlocked": u, "frozen": 0.0, "conflict": v["conflict_value_multiplier"] * u}
        samples.append(sum(noisy[k] * sv[k] for k in noisy))
    samples.sort()
    p5 = samples[int(0.05 * n)]
    p50 = samples[int(0.50 * n)]
    p95 = samples[int(0.95 * n)]
    rel_ci = (p95 - p5) / abs(p50) if p50 else float("inf")

    # ---- Event study around 3 Jan 2026 ---------------------------------
    event_study = {"status": "data_unavailable"}
    xom, _ = _fetch_prices(es["ticker"])
    bench, _ = _fetch_prices(es["oil_benchmark"])
    if len(xom) < 30 or len(bench) < 30:
        bench, _ = _fetch_prices(es["sector_benchmark"])  # fallback benchmark
        used_bench = es["sector_benchmark"]
    else:
        used_bench = es["oil_benchmark"]
    if len(xom) >= 30 and len(bench) >= 30:
        ry = _log_returns(xom)
        rx = _log_returns(bench)
        ev_date = datetime.strptime(es["event_date"], "%Y-%m-%d").date()
        est = {d: ry[d] for d in ry
               if (ev_date - datetime.strptime(d, "%Y-%m-%d").date()).days > es["event_window_days"]
               and 0 < (ev_date - datetime.strptime(d, "%Y-%m-%d").date()).days <= es["estimation_window_days"] + es["event_window_days"]}
        est_x = {d: rx[d] for d in est if d in rx}
        fit = _ols_beta(est, est_x)
        if fit:
            alpha, beta, _ = fit
            car = 0.0
            window_days = []
            for d in sorted(ry):
                dd = datetime.strptime(d, "%Y-%m-%d").date()
                if 0 <= (dd - ev_date).days <= es["event_window_days"] and d in rx:
                    expected = alpha + beta * rx[d]
                    car += ry[d] - expected
                    window_days.append(d)
            event_study = {
                "status": "ok",
                "benchmark": used_bench,
                "alpha": round(alpha, 5),
                "beta": round(beta, 3),
                "event_window_days": window_days,
                "cumulative_abnormal_return": round(car, 4),
                "car_pct": round(car * 100, 2),
            }

    # ---- Mispricing read -------------------------------------------------
    mispricing = {"status": "indeterminate"}
    if event_study.get("status") == "ok":
        # Implied $ of de-risking the market priced = CAR * XOM market cap.
        xom_last = xom[sorted(xom)[-1]] if xom else None
        mkt_cap_bn = (xom_last * v["shares_outstanding_bn"]) if xom_last else None
        market_repriced_bn = (event_study["cumulative_abnormal_return"] * mkt_cap_bn
                              if mkt_cap_bn else None)
        # Model says de-risking the frozen acreage is worth ~ risk_discount_bn.
        if market_repriced_bn is not None:
            ratio = market_repriced_bn / risk_discount_bn if risk_discount_bn else None
            if ratio is None:
                verdict = "indeterminate"
            elif ratio < 0.5:
                verdict = "UNDERPRICED (market repriced less than model de-risking value)"
            elif ratio > 1.5:
                verdict = "OVERPRICED (market repriced more than model de-risking value)"
            else:
                verdict = "FAIRLY PRICED (within 0.5x-1.5x of model)"
            mispricing = {
                "status": "ok",
                "xom_last": round(xom_last, 2) if xom_last else None,
                "implied_market_cap_bn": round(mkt_cap_bn, 1) if mkt_cap_bn else None,
                "market_repriced_bn_event_window": round(market_repriced_bn, 2),
                "model_risk_discount_bn": round(risk_discount_bn, 2),
                "market_to_model_ratio": round(ratio, 2) if ratio else None,
                "verdict": verdict,
            }

    result = {
        "as_of": common.now_iso(),
        "scenario_probabilities_used": {k: round(val, 3) for k, val in probs.items()},
        "q1_regime_input": regime,
        "exxon_net_disputed_boe_bn": round(
            v["stabroek_recoverable_bn_boe"] * v["exxon_working_interest"]
            * v["disputed_acreage_fraction_of_resource"], 3),
        "value_if_unlocked_bn_usd": round(base_unlocked, 2),
        "expected_value_bn_usd": round(ev_bn, 2),
        "essequibo_risk_discount_bn_usd": round(risk_discount_bn, 2),
        "ev_per_share_usd": round(per_share, 2),
        "unlocked_per_share_usd": round(unlocked_per_share, 2),
        "monte_carlo": {
            "n": n, "p5_bn": round(p5, 2), "median_bn": round(p50, 2),
            "p95_bn": round(p95, 2), "relative_ci_width": round(rel_ci, 3),
        },
        "event_study": event_study,
        "mispricing": mispricing,
    }
    common.save_json(common.ARTIFACT_DIR + "/q2_acreage_value.json", result)
    return result


if __name__ == "__main__":
    cfg = common.load_config()
    st = common.load_json(common.STATE_PATH, {})
    r = run(cfg, st)
    print("EV $bn=", r["expected_value_bn_usd"], "| risk_discount $bn=",
          r["essequibo_risk_discount_bn_usd"], "| rel_ci=",
          r["monte_carlo"]["relative_ci_width"])
    print("event_study=", r["event_study"].get("status"),
          "car_pct=", r["event_study"].get("car_pct"),
          "| mispricing=", r["mispricing"].get("verdict"))
