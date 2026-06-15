#!/usr/bin/env python3
"""Validation backtest for the PROPOSED put engine (flow + confirmation + regime).

Question this answers before we build anything:
  Does layering a CONFIRMATION gate and REGIME scaling on top of raw UOA
  put-flow actually beat raw put-flow alone? And which TARGET (1.5R/2R/3R,
  5-day time-stop vs hold-to-10d) is best for shorts?

Signal source (validated edge): real per-contract PUT volume spikes mined the
same way as backtest_uoa.py (front-month ATM put, vol >= 3x its 10-day avg and
>= 500, 6-45 DTE, deduped). That is "Gate A — flow".

Gate B — confirmation, computed as-of the event date (no lookahead):
  price < 21EMA AND price < 50SMA  (rolling over)
  5-day return < SPY 5-day return  (relative weakness)
  RSI(14) in 30..55                (falling, not yet capitulated)

Regime as-of the event date = vcp_tracker.market_regime score (SPY/QQQ/IWM/VIX).

Trade sim (short the underlying as a put proxy): entry = event-day close;
stop = min(10-day swing high, entry + 1.5*ATR) ABOVE entry; target below at a
multiple of risk; walk daily High/Low (stop checked first); optional N-day
time-stop exits at the close. R = (entry - exit) / (stop - entry).

Baselines: raw flow (no gate) at the same target; and CASH (0R).

Caveats: ATM-strike approximation; unsigned volume (can't see buy/sell); one
~2.5-month window (Apr-Jun 2026, which DOES include the pullbacks); earnings
proximity not filtered here (live engine will).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as BT
import backtest_uoa as BU          # events_for(), EXPIRIES (reuses backtest_flow miner)
import bearish as B                # _rsi
import vcp_tracker as V            # market_regime
import uoa

FWD = 10                           # max sessions held


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def regime_asof(bench_full: dict, vix_full, d) -> float:
    bench = {k: s[s.index.date <= d] for k, s in bench_full.items()}
    vix = vix_full[vix_full.index.date <= d] if vix_full is not None else None
    try:
        return V.market_regime(bench, vix).score
    except Exception:  # noqa: BLE001
        return 50.0


def features_asof(und: pd.DataFrame, spy_close: pd.Series, d):
    """Confirmation features computed using data up to & including date d."""
    c = und["Close"].astype(float)
    sub = c[c.index.date <= d]
    if len(sub) < 60:
        return None
    price = float(sub.iloc[-1])
    ema21 = float(sub.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(sub.rolling(50).mean().iloc[-1])
    rsi = B._rsi(sub)
    ret5 = float(sub.iloc[-1] / sub.iloc[-6] - 1) * 100 if len(sub) > 6 else 0.0
    spy_sub = spy_close[spy_close.index.date <= d]
    spy_ret5 = float(spy_sub.iloc[-1] / spy_sub.iloc[-6] - 1) * 100 if len(spy_sub) > 6 else 0.0
    below = price < ema21 and price < sma50
    relweak = ret5 < spy_ret5
    not_capitulated = 30.0 <= rsi <= 55.0
    return {"price": price, "rsi": rsi, "ret5": ret5, "spy_ret5": spy_ret5,
            "confirmed": bool(below and relweak and not_capitulated),
            "below": below, "relweak": relweak, "rsi_ok": not_capitulated}


def sim_put(und: pd.DataFrame, d, tgt_mult: float, time_stop: int | None):
    c = und["Close"].astype(float)
    poss = np.where(c.index.date == d)[0]
    if not len(poss):
        return None
    pos = int(poss[0])
    if pos < 20 or pos + 1 >= len(und):
        return None
    entry = float(und["Close"].iloc[pos])
    atr = float(_atr(und).iloc[pos])
    swing_high = float(und["High"].iloc[max(0, pos - 9):pos + 1].max())
    stop = min(swing_high, entry + 1.5 * atr) if atr > 0 else swing_high
    if stop <= entry:
        stop = entry + 1.5 * max(atr, entry * 0.01)
    risk = stop - entry
    if risk <= 0:
        return None
    target = entry - tgt_mult * risk
    fwd = und.iloc[pos + 1:pos + 1 + FWD]
    for i in range(len(fwd)):
        hi, lo = float(fwd["High"].iloc[i]), float(fwd["Low"].iloc[i])
        if hi >= stop:
            return ("stop", -1.0)
        if lo <= target:
            return ("target", round((entry - target) / risk, 2))
        if time_stop is not None and (i + 1) >= time_stop:
            return ("time", round((entry - float(fwd["Close"].iloc[i])) / risk, 2))
    return ("open", round((entry - float(fwd["Close"].iloc[-1])) / risk, 2))


def stats(rs: list[float]):
    if not rs:
        return (0, 0.0, 0.0)
    a = np.array(rs)
    return (len(a), float(a.mean()), float((a > 0).mean() * 100))


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required (source tradier_creds.sh)", file=sys.stderr)
        return 1
    names = uoa.uoa_universe()
    print(f"  fetching {len(names)} underlyings + benchmarks ...", file=sys.stderr)
    syms = set(names + ["SPY", "QQQ", "IWM", "^VIX"])
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), syms):
            if df is not None and len(df) > 120:
                hist[s] = df
    bench_full = {k: hist[k]["Close"].astype(float) for k in ["SPY", "QQQ", "IWM"] if k in hist}
    vix_full = hist["^VIX"]["Close"].astype(float) if "^VIX" in hist else None
    spy_close = bench_full["SPY"]
    unds = {t: hist[t] for t in names if t in hist}

    print(f"  mining PUT-flow events across {len(unds)} names "
          f"(~{len(unds)*4} contract pulls) ...", file=sys.stderr)
    events = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, evs in ex.map(lambda x: (x, BU.events_for(x, unds[x])), list(unds)):
            for e in evs:
                if e["cp"] == "P":
                    events.append(e)
    print(f"  {len(events)} put-flow events found", file=sys.stderr)

    # annotate each event with confirmation + regime + the base 2R/5d outcome
    rows = []
    for e in events:
        und = unds[e["t"]]
        f = features_asof(und, spy_close, e["date"])
        if f is None:
            continue
        reg = regime_asof(bench_full, vix_full, e["date"])
        rows.append({**e, **f, "regime": reg})
    d = pd.DataFrame(rows)
    if d.empty:
        print("no usable events", file=sys.stderr)
        return 1

    W = 92
    print("\n" + "=" * W)
    print("  PUT-ENGINE VALIDATION — does confirmation + regime beat raw put-flow? "
          "which target?")
    print("=" * W)
    print(f"  {len(d)} put-flow events  |  confirmed by Gate B: {int(d['confirmed'].sum())} "
          f"|  CASH baseline = 0.00R")

    def run_group(mask, label, tgt=2.0, ts=5):
        sub = d[mask]
        rs = []
        for _, r in sub.iterrows():
            o = sim_put(unds[r["t"]], r["date"], tgt, ts)
            if o:
                rs.append(o[1])
        n, avg, pos = stats(rs)
        print(f"  {label:<40}{n:>5}{avg:>+9.2f}R{pos:>8.0f}%")
        return rs

    # ---- 1) does confirmation add edge? (fixed 2R / 5d) ----
    print(f"\n  [1] CONFIRMATION TEST (target 2R, 5-day time-stop)")
    print(f"  {'group':<40}{'n':>5}{'avgR':>10}{'pos%':>8}")
    run_group(d.index >= 0, "RAW FLOW (all put events)")
    run_group(d["confirmed"], "+ CONFIRMATION (Gate B)")
    run_group(~d["confirmed"], "  flow but NOT confirmed")

    # ---- 2) regime breakdown on the confirmed set ----
    print(f"\n  [2] CONFIRMED SET BY REGIME (target 2R, 5-day time-stop)")
    print(f"  {'regime bucket':<40}{'n':>5}{'avgR':>10}{'pos%':>8}")
    run_group(d["confirmed"] & (d["regime"] < 40), "bear  (<40)")
    run_group(d["confirmed"] & (d["regime"] >= 40) & (d["regime"] < 70), "chop  (40-70)")
    run_group(d["confirmed"] & (d["regime"] >= 70), "bull  (>=70)")

    # ---- 3) target sweep on the confirmed set ----
    print(f"\n  [3] TARGET SWEEP (confirmed set; let the data pick)")
    print(f"  {'variant':<40}{'n':>5}{'avgR':>10}{'pos%':>8}")
    for ts, tslab in [(5, "5d time-stop"), (None, "hold to 10d")]:
        for tgt in (1.5, 2.0, 3.0):
            run_group(d["confirmed"], f"{tgt:g}R target, {tslab}", tgt=tgt, ts=ts)

    print("\n" + "=" * W)
    print("  READ: [1] confirmation is worth keeping only if avgR/pos% beats RAW FLOW.")
    print("        [2] tells us how to SCALE by regime. [3] picks the target/time-stop.")
    print("        Anything that can't beat CASH (0R) does not ship.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
