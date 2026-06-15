#!/usr/bin/env python3
"""Backtest DEFENSIVE bear-market strategies over a multi-year window that
INCLUDES the 2022 bear market (the put-engine test couldn't — option history is
only ~180 days). Stock/ETF OHLCV goes back years, so we can test the approaches
that don't need option chains.

Strategies tested vs SPY buy-and-hold (return AND max drawdown both matter):

  A) 200DMA timing  — hold SPY when SPY>200DMA, else CASH (the classic
     capital-preservation rule). Does sitting out drawdowns help?
  B) Trend-confirmed INDEX short — short SPY/QQQ only when close<200DMA AND
     50DMA<200DMA (a confirmed downtrend, NOT a fear spike). Held as a position.
     This is the time-series-momentum short, the one with academic support.
  C) Defensive ROTATION — when SPY>200DMA hold SPY; else hold the strongest of
     {XLU,XLP,XLV,GLD,TLT} by 60-day momentum (or cash if none positive).
  D) Long+Short trend — SPY when bull, short SPY when confirmed downtrend, else
     cash. (A and B combined.)

All signals act NEXT day (no lookahead). Caveats: daily rebalance frictionless
(no costs/slippage); ETF set fixed; one historical path.
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import backtest as BT

LOOKBACK = 1825          # ~5 calendar years
DEFENSIVES = ["XLU", "XLP", "XLV", "GLD", "TLT"]


def fetch(sym: str) -> pd.DataFrame | None:
    s = "VIX" if sym == "^VIX" else sym
    today = dt.date.today()
    start = (today - dt.timedelta(days=LOOKBACK)).isoformat()
    hdr = {"Authorization": f"Bearer {BT.TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BT.TRADIER_BASE}/markets/history",
                         params={"symbol": s, "interval": "daily",
                                 "start": start, "end": today.isoformat()},
                         headers=hdr, timeout=30)
        days = (r.json().get("history") or {}).get("day") if r.ok else None
        if not days:
            return None
        if isinstance(days, dict):
            days = [days]
    except Exception:  # noqa: BLE001
        return None
    df = pd.DataFrame(days)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "volume": "Volume"})


def maxdd(equity: pd.Series) -> float:
    roll = equity.cummax()
    return float(((equity / roll) - 1).min() * 100)


def summarize(name: str, daily_ret: pd.Series, exposure: pd.Series | None = None):
    daily_ret = daily_ret.fillna(0.0)
    eq = (1 + daily_ret).cumprod()
    yrs = len(daily_ret) / 252.0
    total = float(eq.iloc[-1] - 1) * 100
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    dd = maxdd(eq)
    vol = float(daily_ret.std() * np.sqrt(252) * 100)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    exp = f"{float(exposure.mean())*100:>5.0f}%" if exposure is not None else "  -  "
    print(f"  {name:<30}{total:>9.0f}%{cagr:>8.1f}%{dd:>9.0f}%{vol:>8.1f}%{sharpe:>8.2f}{exp:>8}")


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    syms = ["SPY", "QQQ"] + DEFENSIVES
    print(f"  fetching {len(syms)} histories (~{LOOKBACK}d) ...", file=sys.stderr)
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, fetch(x)), syms):
            if df is not None and len(df) > 250:
                hist[s] = df
    if "SPY" not in hist:
        print("no SPY history", file=sys.stderr)
        return 1
    spy = hist["SPY"]["Close"].astype(float)
    qqq = hist["QQQ"]["Close"].astype(float) if "QQQ" in hist else spy
    # align all on SPY's index
    idx = spy.index
    rets = {s: hist[s]["Close"].astype(float).reindex(idx).ffill().pct_change()
            for s in hist}
    spy_ret = rets["SPY"]
    qqq_ret = rets["QQQ"] if "QQQ" in rets else spy_ret

    sma200 = spy.rolling(200).mean()
    sma50 = spy.rolling(50).mean()
    qsma200 = qqq.rolling(200).mean()
    qsma50 = qqq.rolling(50).mean()

    bull = (spy > sma200)
    down = (spy < sma200) & (sma50 < sma200)            # confirmed downtrend
    qdown = (qqq < qsma200) & (qsma50 < qsma200)

    # trim warmup (first 200 bars NaN)
    valid = sma200.notna()
    start_date = idx[valid][0]
    print(f"  evaluation window: {start_date.date()} .. {idx[-1].date()} "
          f"({int(valid.sum())} sessions, ~{valid.sum()/252:.1f}y) — includes 2022 bear",
          file=sys.stderr)

    def clip(s):
        return s[valid]

    W = 92
    print("\n" + "=" * W)
    print("  DEFENSIVE STRATEGY BACKTEST — return AND drawdown (the point of defense "
          "is smaller DD)")
    print("=" * W)
    print(f"  {'strategy':<30}{'total':>10}{'CAGR':>8}{'maxDD':>9}{'vol':>8}{'Sharpe':>8}{'expos':>8}")

    # Buy & hold baselines
    summarize("SPY buy & hold", clip(spy_ret), clip(pd.Series(1.0, index=idx)))
    summarize("QQQ buy & hold", clip(qqq_ret), clip(pd.Series(1.0, index=idx)))

    # A) 200DMA timing (cash when risk-off)
    a_exp = bull.shift(1).fillna(False).astype(float)
    summarize("A) SPY 200DMA timing (cash)", clip(spy_ret * a_exp), clip(a_exp))

    # B) trend-confirmed index short (held as position)
    b_exp = down.shift(1).fillna(False).astype(float)
    summarize("B) SPY confirmed short", clip(-spy_ret * b_exp), clip(b_exp))
    qb_exp = qdown.shift(1).fillna(False).astype(float)
    summarize("B) QQQ confirmed short", clip(-qqq_ret * qb_exp), clip(qb_exp))

    # C) defensive rotation
    mom = {s: hist[s]["Close"].astype(float).reindex(idx).ffill().pct_change(60)
           for s in DEFENSIVES if s in hist}
    if mom:
        mom_df = pd.DataFrame(mom)
        rot_ret = pd.Series(0.0, index=idx)
        nonempty = mom_df.dropna(how="all")             # rows with >=1 valid momentum
        best = pd.Series(index=idx, dtype=object)
        bestmom = pd.Series(index=idx, dtype=float)
        best.loc[nonempty.index] = nonempty.idxmax(axis=1)   # strongest defensive each day
        bestmom.loc[nonempty.index] = nonempty.max(axis=1)
        for i in range(1, len(idx)):
            d = idx[i]
            if bool(bull.iloc[i - 1]):                  # signal from prior day
                rot_ret.iloc[i] = spy_ret.iloc[i]
            else:
                pick = best.iloc[i - 1]
                if isinstance(pick, str) and bestmom.iloc[i - 1] > 0 and pick in rets:
                    rot_ret.iloc[i] = rets[pick].iloc[i]
                # else cash (0)
        in_pos = (bull.shift(1).fillna(False) | (bestmom.shift(1).fillna(-1) > 0)).astype(float)
        summarize("C) defensive rotation", clip(rot_ret), clip(in_pos))

    # D) long+short trend (SPY bull / short confirmed-down / else cash)
    d_ret = spy_ret * bull.shift(1).fillna(False).astype(float) \
        - spy_ret * down.shift(1).fillna(False).astype(float)
    d_exp = (bull | down).shift(1).fillna(False).astype(float)
    summarize("D) long+short trend (SPY)", clip(d_ret), clip(d_exp))

    print("\n" + "=" * W)
    print("  READ: a good DEFENSE keeps most of the upside CAGR while cutting maxDD "
          "vs buy&hold.\n  A short 'sleeve' (B) is only worth it if its standalone return "
          "is clearly POSITIVE.\n  If nothing beats 'A) cash timing' on drawdown-adjusted "
          "return, the answer is: time exposure, don't short.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
