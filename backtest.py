#!/usr/bin/env python3
"""Backtest the scanner's scoring: run the REAL scanner functions on data
truncated to an as-of date 30-60 days ago, then measure what actually happened.

What it measures, per as-of date:
  1. Final-score buckets vs forward return to today (and vs SPY = alpha).
  2. Classification buckets (A/B/REJECT) vs forward return + win rate.
  3. Trade simulation for recommended longs (class A/A+, final>=75):
     entry trigger hit? then target (2R) vs stop — which hit first (daily H/L).
  4. Bearish score check: did high-bearish names underperform?

Honest caveats (also printed in the report):
  - Universe = today's dynamic leadership list -> SURVIVORSHIP BIAS inflates
    bullish results. Treat absolute numbers as optimistic; the *ordering*
    (do higher scores beat lower scores?) is the meaningful part.
  - No historical option chains: options-liquidity layer neutral (prior 50).
  - Earnings/sentiment/event-risk layers neutral (not replayable here).

Usage:  source tradier_creds.sh && python3 backtest.py [--offsets 21,31,42]
        (offsets are TRADING days back: 21~1mo, 31~6wk, 42~2mo calendar)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import requests

import vcp_tracker as V
import bearish as B

TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")
LOOKBACK_DAYS = 560          # calendar days of history to fetch (1y before oldest as-of)


def fetch_long_history(sym: str) -> pd.DataFrame | None:
    """Daily OHLCV going back LOOKBACK_DAYS via Tradier (^VIX -> VIX)."""
    s = "VIX" if sym == "^VIX" else sym
    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/history",
                         params={"symbol": s, "interval": "daily",
                                 "start": start, "end": today.isoformat()},
                         headers=hdr, timeout=20)
        if not r.ok:
            return None
        days = (r.json().get("history") or {}).get("day")
        if not days:
            return None
        if isinstance(days, dict):
            days = [days]
    except Exception:  # noqa: BLE001
        return None
    df = pd.DataFrame(days)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns={"open": "Open", "high": "High",
                                              "low": "Low", "close": "Close",
                                              "volume": "Volume"})
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")


def score_asof(t, full_df, bench_full, vix_full, asof_idx, sector_etfs):
    """Truncate to as-of and run the real scanner scoring. Returns (m, asof_date)."""
    df = full_df.iloc[:asof_idx]
    if len(df) < 210:
        return None, None
    asof = df.index[-1]
    bench = {k: s.loc[:asof] for k, s in bench_full.items()}
    vix = vix_full.loc[:asof] if vix_full is not None else None
    m = V.build_metrics(t, df, bench, "best")
    if m is None:
        return None, None
    regime = V.market_regime(bench, vix)
    m.market_regime = regime.score
    etf = bench.get(m.sector_etf)
    m.sector_score = (V.sector_strength(etf, bench.get("SPY"), bench.get("QQQ"))["score"]
                      if etf is not None else 50.0)
    m.rs_score = V.relative_strength_score(m)
    m.extension_flag = V.extension_flag(m)
    m.extended = bool(m.extension_flag)
    V.levels(m, 1.5)
    m.earn_score = 100.0          # neutral: earnings not replayable
    m.liq_score = -1.0            # neutral prior: no historical chains
    V.compute_final(m)
    m.classification = V.classify(m, allow_earnings=True)
    # bearish
    bm = B.bearish_market_score(bench, vix, [m])
    bs = B.bearish_sector_table(bench, sector_etfs)
    B.score_stock(m, bench, bm, bs)
    return m, asof


def simulate_long(full_df, asof_idx, entry, stop, target, trigger_window=10):
    """Did the breakout trigger, then target-vs-stop first (daily H/L walk)?"""
    fwd = full_df.iloc[asof_idx:]
    if fwd.empty or not (entry > 0 and stop > 0 and target > entry):
        return "invalid", 0.0
    trig = None
    for i in range(min(trigger_window, len(fwd))):
        if float(fwd["High"].iloc[i]) >= entry:
            trig = i
            break
    if trig is None:
        return "no-trigger", 0.0
    risk = entry - stop
    for i in range(trig, len(fwd)):
        lo, hi = float(fwd["Low"].iloc[i]), float(fwd["High"].iloc[i])
        if lo <= stop:                       # conservative: stop checked first
            return "stop", -1.0
        if hi >= target:
            return "target", 2.0
    last = float(fwd["Close"].iloc[-1])
    return "open", round((last - entry) / risk, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="21,31,42",
                    help="trading-day lookbacks (21~30cal, 31~45cal, 42~60cal)")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    if not TRADIER_TOKEN:
        print("TRADIER_TOKEN required (source tradier_creds.sh)", file=sys.stderr)
        return 1

    V.apply_universe()
    tickers = list(V.UNIVERSE)
    sector_etfs = list(V.SECTOR_ETFS)
    syms = tickers + ["SPY", "QQQ", "IWM"] + sector_etfs + ["^VIX"]

    print(f"  fetching {len(set(syms))} histories ({LOOKBACK_DAYS}d) ...", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, fetch_long_history(x)), set(syms)):
            if df is not None and len(df) > 250:
                hist[s] = df
    print(f"  got {len(hist)} histories", file=sys.stderr)

    bench_full = {k: hist[k]["Close"].astype(float)
                  for k in ["SPY", "QQQ", "IWM"] + sector_etfs if k in hist}
    vix_full = hist["^VIX"]["Close"].astype(float) if "^VIX" in hist else None
    spy_close = bench_full["SPY"]

    W = 96
    print("\n" + "=" * W)
    print("  SCANNER BACKTEST — scored as-of past dates with the REAL scanner code, "
          "checked against today")
    print("=" * W)
    print("  CAVEATS: survivorship-biased universe (today's leaders) -> absolute returns "
          "are optimistic;\n  no historical option chains (liq neutral); earnings/sentiment/"
          "event layers neutral.\n  The meaningful test is ORDERING: do higher scores beat "
          "lower scores?")

    for off in offsets:
        rows = []
        for t in tickers:
            df = hist.get(t)
            if df is None or len(df) <= off + 210:
                continue
            asof_idx = len(df) - off
            m, asof = score_asof(t, df, bench_full, vix_full, asof_idx, sector_etfs)
            if m is None:
                continue
            px_asof = float(df["Close"].iloc[asof_idx - 1])
            px_now = float(df["Close"].iloc[-1])
            fwd = (px_now / px_asof - 1) * 100
            # SPY benchmark over the same span
            spy_t = spy_close.loc[:asof]
            spy_asof = float(spy_t.iloc[-1])
            spy_now = float(spy_close.iloc[-1])
            spy_fwd = (spy_now / spy_asof - 1) * 100
            outcome, rmult = simulate_long(df, asof_idx, m.entry, m.stop, m.target)
            rows.append({"t": t, "final": m.final_score, "cls": m.classification,
                         "bear": m.bearish_final, "fwd": fwd, "alpha": fwd - spy_fwd,
                         "outcome": outcome, "r": rmult, "asof": str(asof.date())})
        if not rows:
            continue
        d = pd.DataFrame(rows)
        asof_date = d["asof"].mode()[0]
        n = len(d)
        corr = d["final"].rank().corr(d["fwd"].rank())   # Spearman via ranks (no scipy)
        print("\n" + "-" * W)
        print(f"  AS-OF {asof_date}  (~{off} trading days ago, n={n})   "
              f"SPY since then: {d['alpha'].iloc[0] and (d['fwd'].iloc[0]-d['alpha'].iloc[0]):+.1f}%"
              if n else "")
        print(f"  Spearman corr(final score -> fwd return): {corr:+.2f}")

        print(f"\n  {'SCORE BUCKET':<14}{'n':>4}{'avg fwd':>10}{'avg alpha':>11}{'%pos':>7}")
        for lo, hi, lab in [(90, 999, ">=90"), (80, 90, "80-89"), (70, 80, "70-79"),
                            (60, 70, "60-69"), (0, 60, "<60")]:
            b = d[(d["final"] >= lo) & (d["final"] < hi)]
            if len(b):
                print(f"  {lab:<14}{len(b):>4}{b['fwd'].mean():>9.1f}%{b['alpha'].mean():>10.1f}%"
                      f"{(b['fwd'] > 0).mean()*100:>6.0f}%")

        print(f"\n  {'CLASS':<14}{'n':>4}{'avg fwd':>10}{'avg alpha':>11}{'%pos':>7}")
        for cls in ["A+", "A", "B"]:
            b = d[d["cls"] == cls]
            if len(b):
                print(f"  {cls:<14}{len(b):>4}{b['fwd'].mean():>9.1f}%{b['alpha'].mean():>10.1f}%"
                      f"{(b['fwd'] > 0).mean()*100:>6.0f}%")
        b = d[d["cls"].str.startswith("REJECT")]
        if len(b):
            print(f"  {'REJECT':<14}{len(b):>4}{b['fwd'].mean():>9.1f}%{b['alpha'].mean():>10.1f}%"
                  f"{(b['fwd'] > 0).mean()*100:>6.0f}%")

        recs = d[(d["cls"].isin(["A+", "A"])) & (d["final"] >= 75)]
        if len(recs):
            tr = recs[recs["outcome"] != "no-trigger"]
            wins = (recs["outcome"] == "target").sum()
            stops = (recs["outcome"] == "stop").sum()
            openpos = recs[recs["outcome"] == "open"]
            print(f"\n  TRADE SIM (recommended longs: class A/A+, final>=75): {len(recs)} recs")
            print(f"    triggered: {len(tr)}/{len(recs)}  |  hit 2R target first: {wins}  |  "
                  f"stopped: {stops}  |  still open: {len(openpos)}"
                  + (f" (avg {openpos['r'].mean():+.1f}R)" if len(openpos) else ""))
            if wins + stops > 0:
                print(f"    decided-trade win rate: {wins/(wins+stops)*100:.0f}%   "
                      f"(2R wins -> breakeven needs >33%)")
            er = tr["r"].replace({-1.0: -1.0, 2.0: 2.0})
            if len(tr):
                print(f"    avg R across triggered trades (open at mark): {tr['r'].mean():+.2f}R")
            top = recs.nlargest(5, "final")
            print("    top-5 by score: " + ", ".join(
                f"{r.t}({r.final:.0f}:{r.outcome}{'' if r.outcome!='open' else f' {r.r:+.1f}R'},"
                f" fwd {r.fwd:+.0f}%)" for r in top.itertuples()))

        hb = d[d["bear"] >= 65]
        lb = d[d["bear"] < 50]
        if len(hb) >= 3:
            print(f"\n  BEARISH CHECK: high bearish-score (>=65, n={len(hb)}) avg fwd "
                  f"{hb['fwd'].mean():+.1f}% vs low (<50, n={len(lb)}) {lb['fwd'].mean():+.1f}%"
                  f"  -> {'✓ bearish names underperformed' if hb['fwd'].mean() < lb['fwd'].mean() else '✗ no edge this period'}")

    print("\n" + "=" * W)
    print("  Read: ORDERING (high beats low) and decided-trade win rate vs the 33% "
          "breakeven for 2R trades\n  are the signal. Absolute %s are inflated by "
          "survivorship. One bull-market window only.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
