#!/usr/bin/env python3
"""Backtest the FLOW idea with real per-contract option volume (Tradier serves
daily history for individual OCC option symbols).

For each historical recommendation (same rec set as the other labs):
  1. Reconstruct the contract the scanner would have picked: the monthly
     (3rd-Friday) expiry 30-45 DTE from the as-of date, strike ~4% ITM rounded
     to a standard grid (with neighbor fallbacks).
  2. Pull that contract's daily volume history.
  3. Flow signal: contract volume on the as-of day (or day before) >= 2x its
     prior-10-day average AND >= 50 contracts.
  4. Compare win rate / avg R: flow-confirmed vs not vs no-data.

Honest notes: strike reconstruction is approximate (real pick depended on live
IV); volume!=direction (we can't see buy-vs-sell or OI change historically);
same survivorship caveats as the other labs.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd
import requests

import backtest as BT
import vcp_tracker as V

OFFSETS = [21, 28, 35, 42, 49, 56, 63]


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    return d


def pick_expiry(asof: dt.date) -> dt.date:
    """Monthly expiry 30-45 DTE from as-of (closest to 40)."""
    cands = []
    for k in range(0, 4):
        mth = asof.month + k
        y, mth = asof.year + (mth - 1) // 12, (mth - 1) % 12 + 1
        e = third_friday(y, mth)
        dte = (e - asof).days
        if 25 <= dte <= 55:
            cands.append((abs(dte - 40), e))
    return min(cands)[1] if cands else third_friday(asof.year, asof.month % 12 + 1)


def strike_grid(price: float) -> float:
    if price < 25: return 0.5
    if price < 100: return 1.0
    if price < 250: return 2.5
    if price < 500: return 5.0
    if price < 1000: return 10.0
    return 25.0


def occ(ticker: str, expiry: dt.date, strike: float, cp: str = "C") -> str:
    return f"{ticker}{expiry.strftime('%y%m%d')}{cp}{int(round(strike*1000)):08d}"


def contract_history(sym: str) -> pd.DataFrame | None:
    hdr = {"Authorization": f"Bearer {BT.TRADIER_TOKEN}", "Accept": "application/json"}
    try:
        r = requests.get(f"{BT.TRADIER_BASE}/markets/history",
                         params={"symbol": sym, "interval": "daily",
                                 "start": (dt.date.today()-dt.timedelta(days=180)).isoformat(),
                                 "end": dt.date.today().isoformat()},
                         headers=hdr, timeout=15)
        days = (r.json().get("history") or {}).get("day") if r.ok else None
        if not days:
            return None
        if isinstance(days, dict):
            days = [days]
        df = pd.DataFrame(days)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df[["date", "volume"]]
    except Exception:  # noqa: BLE001
        return None


def flow_signal(ticker: str, price: float, asof: dt.date):
    """Was there unusual volume in the scanner's contract at signal time?"""
    expiry = pick_expiry(asof)
    grid = strike_grid(price)
    base = round(price * 0.96 / grid) * grid           # ~4% ITM, scanner-ish
    for k_off in (0, 1, -1, 2):                        # neighbor fallbacks
        K = base - k_off * grid
        if K <= 0:
            continue
        hist = contract_history(occ(ticker, expiry, K))
        if hist is None or len(hist) < 5:
            continue
        h = hist[hist["date"] <= asof]
        if len(h) < 2:
            return None                                # contract not trading yet
        # signal-time volume: max of the as-of day and the day before (flow
        # often front-runs the technical trigger by a session)
        sig_vol = float(h["volume"].iloc[-2:].max())
        prev = h["volume"].iloc[max(0, len(h)-12):-2]
        base_vol = float(prev.mean()) if len(prev) else 0.0
        spike = (base_vol > 0 and sig_vol >= 2 * base_vol and sig_vol >= 50) or \
                (base_vol == 0 and sig_vol >= 200)
        return {"spike": bool(spike), "sig_vol": sig_vol, "base_vol": round(base_vol, 1)}
    return None


def main():
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required", file=sys.stderr)
        return 1
    V.apply_universe()
    tickers = list(V.UNIVERSE)
    sector_etfs = list(V.SECTOR_ETFS)
    syms = set(tickers + ["SPY", "QQQ", "IWM"] + sector_etfs + ["^VIX"])
    print(f"  fetching {len(syms)} histories ...", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), syms):
            if df is not None and len(df) > 250:
                hist[s] = df
    bench_full = {k: hist[k]["Close"].astype(float)
                  for k in ["SPY", "QQQ", "IWM"] + sector_etfs if k in hist}
    vix_full = hist["^VIX"]["Close"].astype(float) if "^VIX" in hist else None

    recs = []
    for off in OFFSETS:
        for t in tickers:
            df = hist.get(t)
            if df is None or len(df) <= off + 210:
                continue
            asof_idx = len(df) - off
            m, asof = BT.score_asof(t, df, bench_full, vix_full, asof_idx, sector_etfs)
            if m is None or m.classification not in ("A+", "A") or m.final_score < 75:
                continue
            out, rm = BT.simulate_long(df, asof_idx, m.entry, m.stop, m.target)
            if out in ("no-trigger", "invalid"):
                continue
            recs.append({"t": t, "off": off, "asof": asof.date(), "price": m.price,
                         "out": out, "r": rm})
    print(f"  {len(recs)} triggered recommendations; pulling contract flow ...",
          file=sys.stderr)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for rec, f in ex.map(lambda r: (r, flow_signal(r["t"], r["price"], r["asof"])), recs):
            rec["flow"] = f

    d = pd.DataFrame(recs)
    have = d[d["flow"].notna()].copy()
    have["spike"] = have["flow"].apply(lambda f: f["spike"])
    W = 96
    print("\n" + "=" * W)
    print(f"  FLOW BACKTEST — real per-contract option volume at signal time "
          f"({len(have)}/{len(d)} recs had contract data)")
    print("=" * W)

    def stats(g, label):
        dec = g[g["out"].isin(["target", "stop"])]
        wins = (dec["out"] == "target").sum()
        wr = wins / len(dec) * 100 if len(dec) else 0
        print(f"  {label:<34}{len(g):>5}{len(dec):>9}{wr:>7.0f}%{g['r'].mean():>+8.2f}")

    print(f"  {'group':<34}{'recs':>5}{'decided':>9}{'win%':>7}{'avgR':>8}")
    stats(have, "ALL (with contract data)")
    stats(have[have["spike"]], "FLOW-CONFIRMED (vol >=2x avg)")
    stats(have[~have["spike"]], "no flow spike")
    nod = d[d["flow"].isna()]
    if len(nod):
        stats(nod, "no contract data (young/illiquid)")

    if have["spike"].sum() >= 5:
        spike_wr = (have[have["spike"] & have["out"].isin(["target","stop"])]["out"] == "target").mean()
        rest_wr = (have[~have["spike"] & have["out"].isin(["target","stop"])]["out"] == "target").mean()
        print(f"\n  VERDICT: flow-confirmed win rate {spike_wr*100:.0f}% vs "
              f"{rest_wr*100:.0f}% without — "
              + ("flow ADDS edge; worth gating/weighting."
                 if spike_wr > rest_wr + 0.05 else
                 "no meaningful edge in this window; keep flow as info only."))
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
