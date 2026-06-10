#!/usr/bin/env python3
"""Research lab:
PART A — candidate NEW bullish patterns, backtested with the same trade rules
         as the scanner (entry trigger, 2*ATR stop, 2R target, 10d window).
PART B1 — bearish gates on the LEADERS universe (current behavior + relaxed +
         'exceptional stock breakdown' path) with simulated put trades.
PART B2 — bearish on S&P500 LAGGARDS (price<50DMA, RS<SPY): would puts have
         worked if the put side ran on a weakness universe instead of leaders?

Same caveats as other labs: overlapping windows, mostly-bull period,
leaders-universe survivorship (which BIASES AGAINST bearish results in B1).
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import backtest as BT
import backtest_lab as LAB
import vcp_tracker as V
import bearish as B

OFFSETS = [21, 28, 35, 42, 49, 56, 63]


# --------------------------------------------------------------------------
# PART A — candidate bullish pattern detectors (return entry or None)
# --------------------------------------------------------------------------
def det_gap_up_hold(df, atr):
    """Gap up >3% on 1.5x vol in last 7 bars; still above gap-day low; entry =
    post-gap high (gap-and-go continuation — inverse of our bearish gap-down)."""
    n = len(df)
    for i in range(max(1, n - 7), n):
        prev_c = float(df["Close"].iloc[i - 1])
        if prev_c <= 0:
            continue
        gap = float(df["Open"].iloc[i]) / prev_c - 1
        v20 = float(df["Volume"].iloc[max(0, i - 20):i].mean())
        if gap > 0.03 and v20 > 0 and float(df["Volume"].iloc[i]) > 1.5 * v20:
            if float(df["Close"].iloc[-1]) >= float(df["Low"].iloc[i]):
                return float(df["High"].iloc[i:].max()) * 1.001
    return None


def det_tight3(df, atr):
    """3 consecutive daily ranges < 0.6*ATR within 5% of 52w high (daily
    version of 3-weeks-tight: coiled spring near highs)."""
    last3 = df.iloc[-3:]
    rng = last3["High"] - last3["Low"]
    hi52 = float(df["Close"].tail(252).max())
    if atr > 0 and (rng < 0.6 * atr).all() and float(df["Close"].iloc[-1]) >= hi52 * 0.95:
        return float(last3["High"].max()) * 1.001
    return None


def det_new_high(df, atr):
    """Plain 52-week-high momentum: close within 1.5% below the 52w high;
    entry = the high (does simple new-high breakout add anything?)."""
    hi = float(df["High"].tail(252).max())
    c = float(df["Close"].iloc[-1])
    if hi * 0.985 <= c <= hi:
        return hi * 1.001
    return None


def det_cup_handle(df, atr):
    """Cup-with-handle (O'Neil classic, not in scanner): 12-35% cup over the
    last ~6mo, recovery to within 10% of rim, 1-8 bar handle drifting down on
    quiet volume; entry = handle high."""
    c = df["Close"]
    win = c.tail(120)
    rim = float(win.max())
    rim_pos = int(np.argmax(win.to_numpy()))
    after = win.iloc[rim_pos:]
    if len(after) < 25:
        return None
    trough = float(after.min())
    depth = (rim - trough) / rim if rim else 0
    if not (0.12 <= depth <= 0.35):
        return None
    cur = float(c.iloc[-1])
    if not (rim * 0.90 <= cur < rim):
        return None
    handle = c.tail(8)
    hh = float(handle.max())
    drift = (hh - cur) / hh if hh else 1
    if not (0.0 < drift <= 0.10):
        return None
    if float(df["Volume"].tail(8).mean()) > float(df["Volume"].tail(40).mean()):
        return None
    return hh * 1.001


CANDIDATES = [("GapUpHold", det_gap_up_hold), ("3DayTight", det_tight3),
              ("52wHighMomo", det_new_high), ("CupHandle", det_cup_handle)]


def simulate_put(full_df, asof_idx, entry, stop, target):
    """Put trade: in at as-of close; stop ABOVE (High hits first = loss),
    target BELOW (Low hits = win, 2R)."""
    fwd = full_df.iloc[asof_idx:]
    if fwd.empty or not (stop > entry > target > 0):
        return "invalid", 0.0
    risk = stop - entry
    for i in range(len(fwd)):
        if float(fwd["High"].iloc[i]) >= stop:
            return "stop", -1.0
        if float(fwd["Low"].iloc[i]) <= target:
            return "target", 2.0
    return "open", round((entry - float(fwd["Close"].iloc[-1])) / risk, 2)


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    V.apply_universe()
    tickers = list(V.UNIVERSE)
    sector_etfs = list(V.SECTOR_ETFS)
    syms = set(tickers + ["SPY", "QQQ", "IWM"] + sector_etfs + ["^VIX"])
    print(f"  fetching {len(syms)} leader histories ...", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), syms):
            if df is not None and len(df) > 250:
                hist[s] = df
    bench_full = {k: hist[k]["Close"].astype(float)
                  for k in ["SPY", "QQQ", "IWM"] + sector_etfs if k in hist}
    vix_full = hist["^VIX"]["Close"].astype(float) if "^VIX" in hist else None
    W = 100

    # ---------------- PART A ----------------
    print("\n" + "=" * W)
    print("  PART A — CANDIDATE NEW BULLISH PATTERNS (same trade rules: "
          "2*ATR stop, 2R target, 10d trigger)")
    print("=" * W)
    rowsA = {name: [] for name, _ in CANDIDATES}
    for off in OFFSETS:
        for t in tickers:
            df = hist.get(t)
            if df is None or len(df) <= off + 210:
                continue
            asof_idx = len(df) - off
            tdf = df.iloc[:asof_idx]
            c = tdf["Close"]
            if len(tdf) < 210:
                continue
            sma200 = float(c.rolling(200).mean().iloc[-1])
            if float(c.iloc[-1]) < sma200:          # uptrend gate, like the scanner
                continue
            atr = V._atr(tdf)
            for name, det in CANDIDATES:
                entry = det(tdf, atr)
                if entry is None:
                    continue
                stop = entry - 2.0 * atr
                target = entry + 2 * (entry - stop)
                out, rm = LAB.simulate(df, asof_idx, entry, stop, target)
                if out in ("target", "stop", "open"):
                    rowsA[name].append((out, rm))
    print(f"  {'pattern':<14}{'signals':>9}{'decided':>9}{'win%':>7}{'avgR':>8}"
          f"   vs scanner baseline 78% / +1.31R")
    for name, rows in rowsA.items():
        if not rows:
            print(f"  {name:<14}{'0':>9}")
            continue
        dec = [r for r in rows if r[0] in ("target", "stop")]
        wins = sum(1 for r in dec if r[0] == "target")
        avg = np.mean([r[1] for r in rows])
        wr = wins / len(dec) * 100 if dec else 0
        print(f"  {name:<14}{len(rows):>9}{len(dec):>9}{wr:>6.0f}%{avg:>+8.2f}")

    # ---------------- PART B1: bearish gates on LEADERS ----------------
    print("\n" + "=" * W)
    print("  PART B1 — BEARISH GATES ON THE LEADERS UNIVERSE "
          "(note: leaders bias AGAINST puts — floor, not ceiling)")
    print("=" * W)
    bear_rows = []
    for off in OFFSETS:
        for t in tickers:
            df = hist.get(t)
            if df is None or len(df) <= off + 210:
                continue
            asof_idx = len(df) - off
            m, asof = BT.score_asof(t, df, bench_full, vix_full, asof_idx, sector_etfs)
            if m is None:
                continue
            out, rm = simulate_put(df, asof_idx, m.put_entry, m.put_stop, m.put_target)
            bear_rows.append({"off": off, "t": t, "final": m.bearish_final,
                              "pat": m.bearish_pattern_score, "weak": m.bear_rel_weakness,
                              "mkt": m.bear_market_score, "below50": m.price < m.sma50,
                              "chase": m.do_not_chase_put, "out": out, "r": rm})
    bd = pd.DataFrame(bear_rows)
    print(f"  bearish_final distribution: max {bd['final'].max():.0f}, "
          f"p95 {bd['final'].quantile(.95):.0f}, p75 {bd['final'].quantile(.75):.0f}"
          f"   (current gate: >=75 -> {(bd['final']>=75).sum()} recs ever)")
    gates = [
        ("G0 current >=75", bd["final"] >= 75),
        ("G1 relaxed >=65", bd["final"] >= 65),
        ("G2 exceptional*", (bd["pat"] >= 65) & (bd["weak"] >= 70) & bd["below50"] & ~bd["chase"]),
    ]
    print(f"\n  {'gate':<20}{'recs':>6}{'decided':>9}{'win%':>7}{'avgR':>8}")
    for name, mask in gates:
        g = bd[mask & bd["out"].isin(["target", "stop", "open"])]
        dec = g[g["out"].isin(["target", "stop"])]
        wins = (dec["out"] == "target").sum()
        wr = wins / len(dec) * 100 if len(dec) else 0
        avg = g["r"].mean() if len(g) else 0
        print(f"  {name:<20}{len(g):>6}{len(dec):>9}{wr:>6.0f}%{avg:>+8.2f}")
    print("  *exceptional = bear-pattern>=65 & rel-weak>=70 & below 50DMA & not oversold-chase "
          "(any market) — the spec §11 path never implemented")

    # ---------------- PART B2: bearish on S&P500 LAGGARDS ----------------
    print("\n" + "=" * W)
    print("  PART B2 — SAME BEARISH LOGIC ON S&P500 LAGGARDS (the proper put pond)")
    print("=" * W)
    import universe as U
    cons = U.fetch_constituents()
    lag_syms = [s for s in cons if s not in set(tickers)][:600]
    print(f"  downloading {len(lag_syms)} S&P histories (chunked yahoo, 2y) ...", file=sys.stderr)
    import yfinance as yf, time as _time
    lhist = {}
    uniq = list(dict.fromkeys(lag_syms))
    for i in range(0, len(uniq), 40):
        part = uniq[i:i + 40]
        try:
            data = yf.download(part, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in part:
                    if t in data.columns.get_level_values(0):
                        sub = data[t].dropna(how="all")
                        if len(sub) > 300:
                            lhist[t] = sub
        except Exception:  # noqa: BLE001
            pass
        _time.sleep(0.8)
    print(f"  got {len(lhist)} laggard-candidate histories", file=sys.stderr)

    lrows = []
    for off in OFFSETS:
        for t, df in lhist.items():
            if len(df) <= off + 260:
                continue
            asof_idx = len(df) - off
            tdf = df.iloc[:asof_idx]
            c = tdf["Close"].astype(float)
            if len(tdf) < 260:
                continue
            px = float(c.iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            ret63 = px / float(c.iloc[-64]) - 1 if len(c) > 64 else 0
            spy = bench_full["SPY"].loc[:tdf.index[-1]]
            spy63 = float(spy.iloc[-1]) / float(spy.iloc[-64]) - 1 if len(spy) > 64 else 0
            if not (px < sma50 and (ret63 - spy63) < -0.05):
                continue                      # laggard filter: below 50DMA + lagging SPY
            V.UNIVERSE_META.setdefault(t, (t, cons.get(t, (t, "?"))[1]))
            V.TICKER_ETF.setdefault(t, U._gics_etf(cons.get(t, (t, "?"))[1]))
            try:
                m = V.build_metrics(t, tdf, bench_full, "best")
            except Exception:  # noqa: BLE001
                continue
            if m is None:
                continue
            bmkt = B.bearish_market_score(bench_full, vix_full, [m])
            bsec = B.bearish_sector_table(bench_full, sector_etfs)
            B.score_stock(m, bench_full, bmkt, bsec)
            out, rm = simulate_put(df, asof_idx, m.put_entry, m.put_stop, m.put_target)
            lrows.append({"off": off, "t": t, "final": m.bearish_final,
                          "pat": m.bearish_pattern_score, "weak": m.bear_rel_weakness,
                          "below50": True, "chase": m.do_not_chase_put,
                          "out": out, "r": rm})
    ld = pd.DataFrame(lrows)
    if len(ld):
        print(f"  laggard setups scored: {len(ld)} | bearish_final max {ld['final'].max():.0f}, "
              f"p95 {ld['final'].quantile(.95):.0f}")
        gates2 = [
            ("G0 current >=75", ld["final"] >= 75),
            ("G1 relaxed >=65", ld["final"] >= 65),
            ("G2 exceptional*", (ld["pat"] >= 65) & (ld["weak"] >= 70) & ~ld["chase"]),
            ("G3 G2 & final>=60", (ld["pat"] >= 65) & (ld["weak"] >= 70) & ~ld["chase"]
             & (ld["final"] >= 60)),
        ]
        print(f"\n  {'gate':<20}{'recs':>6}{'decided':>9}{'win%':>7}{'avgR':>8}"
              f"   (2R puts: breakeven 33%)")
        for name, mask in gates2:
            g = ld[mask & ld["out"].isin(["target", "stop", "open"])]
            dec = g[g["out"].isin(["target", "stop"])]
            wins = (dec["out"] == "target").sum()
            wr = wins / len(dec) * 100 if len(dec) else 0
            avg = g["r"].mean() if len(g) else 0
            print(f"  {name:<20}{len(g):>6}{len(dec):>9}{wr:>6.0f}%{avg:>+8.2f}")
    print("\n" + "=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
