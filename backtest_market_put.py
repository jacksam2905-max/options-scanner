#!/usr/bin/env python3
"""Backtest the ACUTE RISK-OFF -> index put rule on SPY/QQQ history.

Signal (computable daily, no lookahead), evaluated independently per index:
    VIX up >= 25% over 5 sessions
AND index below its 20-DMA
AND index 3-day return <= -2%
(dedup: only the first signal in any 5-session span)

Trade sim per signal: buy index put proxy = short at close, stop = close +
1.5*ATR(14) (above), target = 2R below, walk daily High/Low (stop checked
first). Also report raw forward 5d/10d returns vs the unconditional baseline.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import backtest as BT


def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run(idx_name: str, idx: pd.DataFrame, vix: pd.Series):
    c = idx["Close"].astype(float)
    sma20 = c.rolling(20).mean()
    a = atr(idx)
    vix = vix.reindex(c.index).ffill()
    sigs = []
    last_sig = -99
    for i in range(25, len(c) - 1):
        vix_chg = float(vix.iloc[i]) / float(vix.iloc[i - 5]) - 1 if float(vix.iloc[i - 5]) > 0 else 0
        ret3 = float(c.iloc[i]) / float(c.iloc[i - 3]) - 1
        if (vix_chg >= 0.25 and float(c.iloc[i]) < float(sma20.iloc[i])
                and ret3 <= -0.02 and i - last_sig >= 5):
            sigs.append(i)
            last_sig = i
    rows = []
    for i in sigs:
        entry = float(c.iloc[i])
        stop = entry + 1.5 * float(a.iloc[i])
        target = entry - 2 * (stop - entry)
        out, r = "open", 0.0
        for j in range(i + 1, len(c)):
            if float(idx["High"].iloc[j]) >= stop:
                out, r = "stop", -1.0
                break
            if float(idx["Low"].iloc[j]) <= target:
                out, r = "target", 2.0
                break
        if out == "open":
            r = (entry - float(c.iloc[-1])) / (stop - entry)
        f5 = float(c.iloc[min(i + 5, len(c) - 1)]) / entry - 1
        f10 = float(c.iloc[min(i + 10, len(c) - 1)]) / entry - 1
        rows.append({"date": str(c.index[i].date()), "out": out, "r": round(r, 2),
                     "f5": f5 * 100, "f10": f10 * 100})
    # baseline forward returns (all days)
    f5b = (c.shift(-5) / c - 1).dropna().mean() * 100
    f10b = (c.shift(-10) / c - 1).dropna().mean() * 100
    d = pd.DataFrame(rows)
    print(f"\n  {idx_name}: {len(d)} acute risk-off signals "
          f"(baseline fwd: 5d {f5b:+.2f}% / 10d {f10b:+.2f}%)")
    if not len(d):
        return
    dec = d[d["out"].isin(["target", "stop"])]
    wins = (dec["out"] == "target").sum()
    print(f"    put-trade sim: {wins}/{len(dec)} hit 2R target first "
          f"({wins/len(dec)*100:.0f}% win, breakeven 33%) | avg R {d['r'].mean():+.2f}")
    print(f"    avg fwd return after signal: 5d {d['f5'].mean():+.2f}% / 10d {d['f10'].mean():+.2f}%")
    print("    signals: " + ", ".join(f"{r.date}({r.out},{r.r:+.1f}R)" for r in d.itertuples()))


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    spy = BT.fetch_long_history("SPY")
    qqq = BT.fetch_long_history("QQQ")
    vixdf = BT.fetch_long_history("^VIX")
    vix = vixdf["Close"].astype(float)
    print("=" * 90)
    print("  ACUTE RISK-OFF -> INDEX PUT backtest "
          f"({str(spy.index[0].date())} .. {str(spy.index[-1].date())})")
    print("  signal: VIX +25%/5d AND below 20-DMA AND 3d return <= -2%  |  "
          "trade: short@close, stop +1.5ATR, 2R target")
    print("=" * 90)
    run("SPY", spy, vix)
    run("QQQ", qqq, vix)
    print("\n" + "=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
