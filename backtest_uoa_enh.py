#!/usr/bin/env python3
"""UOA ENHANCEMENT backtest — does sharpening the put-flow signal add edge?

The base UOA backtest (backtest_uoa.py) showed PUT-flow spikes carry ~+10pp
directional edge (P(-2% in 10d) vs baseline). This asks: do the LIVE scanner's
refinements actually improve that edge, or are they noise? It segments the same
mined PUT-flow events by:

  1) SPIKE SIZE      : 3-5x avg, 5-10x, 10x+  (does conviction scale?)
  2) PRE-MOVE        : underlying |chg| < 1% on the spike day (flow before move)
                       vs flow that fired AFTER the stock already moved
  3) TIME WINDOW     : hit-rate at 5d vs 10d (how fast does it pay?)
  4) MAGNITUDE (MFE) : avg best favourable close-to-close drop within 10d

Reuses the contract-volume miner + baseline from backtest_uoa.py. Same caveats:
ATM-strike approximation, unsigned volume, one ~2.5-month window.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as BT
import backtest_uoa as BU
import uoa

FWD5, FWD10, MOVE = 5, 10, 0.02


def fwd_stats(und: pd.DataFrame, d, want_down: bool):
    """Forward hit at 5d & 10d + best favourable excursion (close-based)."""
    c = und["Close"].astype(float)
    idx = c.index.date
    pos = next((i for i, dd in enumerate(idx) if dd == d), None)
    if pos is None or pos + 1 >= len(c):
        return None
    p0 = float(c.iloc[pos])
    seg = c.iloc[pos + 1: pos + 1 + FWD10] / p0 - 1
    if seg.empty:
        return None
    seg5 = seg.iloc[:FWD5]
    if want_down:
        hit5 = bool((seg5 <= -MOVE).any()); hit10 = bool((seg <= -MOVE).any())
        mfe = float(seg.min())                       # best down move (favourable for puts)
    else:
        hit5 = bool((seg5 >= MOVE).any()); hit10 = bool((seg >= MOVE).any())
        mfe = float(seg.max())
    return {"hit5": hit5, "hit10": hit10, "mfe": mfe}


def chg_on(und: pd.DataFrame, d) -> float | None:
    c = und["Close"].astype(float)
    idx = c.index.date
    pos = next((i for i, dd in enumerate(idx) if dd == d), None)
    if pos is None or pos < 1:
        return None
    return float(c.iloc[pos] / c.iloc[pos - 1] - 1) * 100


def seg_report(rows: list[dict], base5: float, base10: float):
    if not rows:
        print("    (no events)")
        return
    n = len(rows)
    h5 = np.mean([r["hit5"] for r in rows]) * 100
    h10 = np.mean([r["hit10"] for r in rows]) * 100
    mfe = np.mean([r["mfe"] for r in rows]) * 100
    print(f"    n={n:<4} hit5 {h5:>3.0f}% (edge {h5-base5:+.0f}pp) · "
          f"hit10 {h10:>3.0f}% (edge {h10-base10:+.0f}pp) · avg best-drop {mfe:+.1f}%")


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    names = uoa.uoa_universe()
    print(f"  fetching {len(names)} underlyings ...", file=sys.stderr)
    unds = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), names):
            if df is not None and len(df) > 120:
                unds[t] = df
    print(f"  mining PUT-flow events across {len(unds)} names ...", file=sys.stderr)
    events = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, evs in ex.map(lambda x: (x, BU.events_for(x, unds[x])), list(unds)):
            for e in evs:
                if e["cp"] == "P":
                    events.append(e)

    # baselines for ±2% down at 5d / 10d across all name-days
    dn5 = dn10 = nb = 0
    lo, hi = BU.EXPIRIES[0].replace(day=1), BU.EXPIRIES[-1]
    for und in unds.values():
        c = und["Close"].astype(float)
        for i in range(len(c) - 1):
            d = c.index[i].date()
            if not (lo <= d <= hi) or i + 1 + FWD10 > len(c):
                continue
            p0 = float(c.iloc[i]); seg = c.iloc[i + 1:i + 1 + FWD10] / p0 - 1
            dn5 += bool((seg.iloc[:FWD5] <= -MOVE).any()); dn10 += bool((seg <= -MOVE).any())
            nb += 1
    base5, base10 = dn5 / nb * 100, dn10 / nb * 100

    rows = []
    for e in events:
        und = unds[e["t"]]
        f = fwd_stats(und, e["date"], want_down=True)
        ch = chg_on(und, e["date"])
        if f is None or ch is None:
            continue
        rows.append({**e, **f, "chg": ch, "ratio": e["vol"] / max(e["base"], 1)})

    W = 88
    print("\n" + "=" * W)
    print("  UOA PUT-FLOW ENHANCEMENT — which refinements sharpen the +10pp edge?")
    print("=" * W)
    print(f"  baseline P(-2%): 5d {base5:.0f}% · 10d {base10:.0f}%   (n={nb:,} name-days)")

    print(f"\n  [ALL PUT-FLOW EVENTS]")
    seg_report(rows, base5, base10)

    print(f"\n  [BY SPIKE SIZE]")
    for lo_r, hi_r, lab in [(3, 5, "3-5x avg"), (5, 10, "5-10x avg"), (10, 1e9, "10x+ avg")]:
        print(f"  {lab}:")
        seg_report([r for r in rows if lo_r <= r["ratio"] < hi_r], base5, base10)

    print(f"\n  [PRE-MOVE vs REACTIVE]   (PRE-MOVE = stock |chg| < 1% on spike day)")
    print("  PRE-MOVE (flow before the move):")
    seg_report([r for r in rows if abs(r["chg"]) < 1.0], base5, base10)
    print("  REACTIVE (stock already moved >=1%):")
    seg_report([r for r in rows if abs(r["chg"]) >= 1.0], base5, base10)

    print(f"\n  [PRE-MOVE + BIG SPIKE (>=5x)]   (the live scanner's headline subset)")
    seg_report([r for r in rows if abs(r["chg"]) < 1.0 and r["ratio"] >= 5], base5, base10)

    print("\n" + "=" * W)
    print("  READ: a refinement is worth keeping only if its edge (vs baseline) and/or "
          "avg\n  best-drop is clearly BIGGER than 'ALL PUT-FLOW'. Otherwise it's noise/"
          "fewer-n.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
