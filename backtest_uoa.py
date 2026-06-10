#!/usr/bin/env python3
"""Backtest the STANDALONE UOA signal with real per-contract volume history.

Design (data-honest):
  - Universe: same as the live UOA scanner (leaders + laggards + active names).
  - For each monthly expiry (May 15, Jun 19 2026), take the ATM call and ATM
    put (strike at the window's median price; ±1-grid fallback) — front-month
    ATM is where volume actually exists, fixing the coverage failure of the
    earlier 40-DTE test.
  - UOA event: contract volume >= 3x its prior-10-day average AND >= 500,
    while the contract is 6-45 DTE (expiry week excluded — gamma noise).
    Events within 5 days of a prior one on the same name are deduped.
  - Outcome: did the UNDERLYING move >= +2% (call events) / <= -2% (put
    events) within 10 trading days? Plus mean 10-day forward return.
  - Baseline: the unconditional probability of the same move across all
    names/days in the window (so we measure EDGE, not bull-market drift).

Caveats: volume direction is unsigned (can't see buy-vs-sell historically);
ATM strike approximation; one ~2-month window.
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest as BT
import backtest_flow as BF
import uoa

EXPIRIES = [dt.date(2026, 5, 15), dt.date(2026, 6, 19)]
SPIKE_MULT, SPIKE_MIN = 3.0, 500
MOVE = 0.02          # ±2% directional move
FWD = 10             # within 10 trading days


def events_for(ticker: str, und: pd.DataFrame):
    """UOA spike events for one name across the monthly ATM contracts."""
    out = []
    closes = und["Close"].astype(float)
    for E in EXPIRIES:
        lo, hi = E - dt.timedelta(days=45), E - dt.timedelta(days=6)
        win = closes[(closes.index.date >= lo) & (closes.index.date <= hi)]
        if len(win) < 5:
            continue
        med = float(win.median())
        grid = BF.strike_grid(med)
        atm = round(med / grid) * grid
        for cp in ("C", "P"):
            hist = None
            for k in (0, 1, -1):
                K = atm + k * grid * (1 if cp == "P" else -1)
                hist = BF.contract_history(BF.occ(ticker, E, K, cp))
                if hist is not None and len(hist) >= 8:
                    break
            if hist is None or len(hist) < 8:
                continue
            v = hist["volume"].to_numpy()
            dates = hist["date"].tolist()
            for i in range(3, len(hist)):
                d = dates[i]
                if not (lo <= d <= hi):
                    continue
                prior = v[max(0, i - 10):i]
                base = float(prior.mean()) if len(prior) >= 3 else 0.0
                if base > 0 and v[i] >= SPIKE_MULT * base and v[i] >= SPIKE_MIN:
                    out.append({"t": ticker, "date": d, "cp": cp,
                                "vol": int(v[i]), "base": round(base, 1)})
    # dedupe: keep first event per (name, side) within 5 days
    out.sort(key=lambda e: e["date"])
    kept = []
    for e in out:
        if any(k["cp"] == e["cp"] and 0 <= (e["date"] - k["date"]).days < 5 for k in kept):
            continue
        kept.append(e)
    return kept


def outcome(und: pd.DataFrame, d: dt.date, want_up: bool):
    closes = und["Close"].astype(float)
    idx = closes.index.date
    pos = None
    for i, dd in enumerate(idx):
        if dd == d:
            pos = i
            break
    if pos is None or pos + 2 >= len(closes):
        return None
    p0 = float(closes.iloc[pos])
    fwd = closes.iloc[pos + 1: pos + 1 + FWD] / p0 - 1
    if want_up:
        return {"hit": bool((fwd >= MOVE).any()), "ret10": float(fwd.iloc[-1])}
    return {"hit": bool((fwd <= -MOVE).any()), "ret10": float(fwd.iloc[-1])}


def baseline(unds: dict):
    """Unconditional P(±2% within 10d) across all names/days in the window."""
    up = dn = n = 0
    rets = []
    lo, hi = EXPIRIES[0] - dt.timedelta(days=45), EXPIRIES[-1] - dt.timedelta(days=6)
    for und in unds.values():
        closes = und["Close"].astype(float)
        idx = closes.index.date
        for i, d in enumerate(idx):
            if not (lo <= d <= hi) or i + 1 + FWD > len(closes):
                continue
            p0 = float(closes.iloc[i])
            fwd = closes.iloc[i + 1: i + 1 + FWD] / p0 - 1
            up += bool((fwd >= MOVE).any())
            dn += bool((fwd <= -MOVE).any())
            rets.append(float(fwd.iloc[-1]))
            n += 1
    return up / n * 100, dn / n * 100, float(np.mean(rets)) * 100, n


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    names = uoa.uoa_universe()
    print(f"  fetching {len(names)} underlying histories ...", file=sys.stderr)
    unds = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), names):
            if df is not None and len(df) > 120:
                unds[t] = df
    print(f"  got {len(unds)}; mining contract histories for UOA events "
          f"(~{len(unds)*4} contract pulls) ...", file=sys.stderr)
    evs = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, e in ex.map(lambda x: (x, events_for(x, unds[x])), list(unds)):
            evs.extend(e)
    print(f"  {len(evs)} UOA spike events found", file=sys.stderr)

    rows = []
    for e in evs:
        und = unds.get(e["t"])
        o = outcome(und, e["date"], want_up=(e["cp"] == "C"))
        if o:
            rows.append({**e, **o})
    d = pd.DataFrame(rows)
    up_b, dn_b, ret_b, nb = baseline(unds)

    W = 96
    print("\n" + "=" * W)
    print(f"  UOA BACKTEST — front-month ATM contract volume spikes "
          f"(>= {SPIKE_MULT:g}x avg & >= {SPIKE_MIN}) -> forward UNDERLYING move")
    print("=" * W)
    print(f"  baseline (all {nb:,} name-days): P(+2% in {FWD}d) {up_b:.0f}% · "
          f"P(-2% in {FWD}d) {dn_b:.0f}% · mean 10d ret {ret_b:+.1f}%")
    for cp, lab, base_p in [("C", "CALL spikes (bullish flow)", up_b),
                            ("P", "PUT spikes (bearish flow)", dn_b)]:
        g = d[d["cp"] == cp]
        if not len(g):
            print(f"\n  {lab}: 0 events")
            continue
        hit = g["hit"].mean() * 100
        ret = g["ret10"].mean() * 100
        edge = hit - base_p
        print(f"\n  {lab}: n={len(g)}")
        print(f"    directional hit (±2% in {FWD}d): {hit:.0f}%  vs baseline {base_p:.0f}%  "
              f"-> edge {edge:+.0f}pp")
        print(f"    mean 10d underlying return: {ret:+.1f}%  (baseline {ret_b:+.1f}%)")
        big = g.nlargest(4, "vol")
        print("    biggest: " + ", ".join(
            f"{r.t} {r.date} {('+' if r.ret10>=0 else '')}{r.ret10*100:.1f}%/10d "
            f"(v{r.vol:,})" for r in big.itertuples()))
    print("\n" + "=" * W)
    print("  Verdict guide: CALL-spike hit% must beat the up-baseline (and PUT-spike the "
          "down-baseline)\n  by a real margin to claim edge — unsigned volume can be "
          "hedging or closing, not conviction.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
