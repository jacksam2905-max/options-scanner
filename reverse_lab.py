#!/usr/bin/env python3
"""Reverse-engineer the Jun-10 move day.

PART 1  Who actually moved on Jun 10 (S&P500 + our universes)?
PART 2  Forensic audit: of the big losers, how many were flagged by signals we
        RECORDED the night before (Jun-9 UOA put-flow snapshot, scanner PUTs)?
PART 3  Signature: what did the big losers look like at the Jun-9 close
        (computable features only — no hindsight)?
PART 4  Validation: backtest the candidate continuation signals across the
        whole panel (~16 months) — do they predict next-1d/3d weakness
        beyond baseline, or was Jun 10 luck?
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

import universe as U

MOVE_DAY = dt.date(2026, 6, 10)
PRE_DAY = dt.date(2026, 6, 9)
LOSER_CUT = -4.0          # % move that counts as a "big loser"


def dl_panel(tickers, period="1y", chunk=40):
    out = {}
    uniq = list(dict.fromkeys(tickers))
    for i in range(0, len(uniq), chunk):
        part = uniq[i:i + chunk]
        try:
            data = yf.download(part, period=period, interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in part:
                    if t in data.columns.get_level_values(0):
                        sub = data[t].dropna(how="all")
                        if len(sub) > 220:
                            out[t] = sub
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.6)
    return out


def feats_asof(df, pre_idx):
    """Hindsight-free features at the PRE_DAY close."""
    d = df.iloc[:pre_idx + 1]
    c = d["Close"].astype(float)
    v = d["Volume"].astype(float)
    px = float(c.iloc[-1])
    s20 = float(c.rolling(20).mean().iloc[-1])
    s50 = float(c.rolling(50).mean().iloc[-1])
    v20 = float(v.rolling(20).mean().iloc[-1])
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    dn = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100.0 if dn == 0 else float(100 - 100 / (1 + up / dn))
    last10 = d.iloc[-10:]
    ch10 = last10["Close"].pct_change()
    dist10 = int(((last10["Close"] < last10["Close"].shift(1))
                  & (last10["Volume"] > last10["Volume"].shift(1))
                  & (ch10 < -0.01)).sum())
    return {
        "ret1": (px / float(c.iloc[-2]) - 1) * 100,
        "ret2": (px / float(c.iloc[-3]) - 1) * 100,
        "ret5": (px / float(c.iloc[-6]) - 1) * 100,
        "vs20": (px / s20 - 1) * 100,
        "vs50": (px / s50 - 1) * 100,
        "rsi": rsi,
        "vol_x": float(v.iloc[-1]) / v20 if v20 else 0,
        "dist10": dist10,
        "below20": px < s20,
        "weak_close": float(d["Close"].iloc[-1]) < float(d["Open"].iloc[-1]),
    }


def main():
    cons = U.fetch_constituents()
    try:
        import vcp_tracker as V
        V.apply_universe()
        extra = list(V.UNIVERSE) + list(V.BEAR_UNIVERSE)
    except Exception:  # noqa: BLE001
        extra = []
    import uoa as UOA
    names = list(dict.fromkeys(list(cons) + extra + UOA.ACTIVE_OPTION_NAMES))
    print(f"  downloading panel ({len(names)} names) ...", file=sys.stderr)
    hist = dl_panel(names)
    print(f"  got {len(hist)}", file=sys.stderr)

    # locate the move day / pre day indices per name
    movers = []
    for t, df in hist.items():
        dates = [d.date() for d in df.index]
        if MOVE_DAY not in dates or PRE_DAY not in dates:
            continue
        mi, pi = dates.index(MOVE_DAY), dates.index(PRE_DAY)
        if pi < 60:
            continue
        ret = (float(df["Close"].iloc[mi]) / float(df["Close"].iloc[pi]) - 1) * 100
        movers.append({"t": t, "mi": mi, "pi": pi, "ret": ret})
    mv = pd.DataFrame(movers)
    W = 100

    print("\n" + "=" * W)
    print(f"  PART 1 — WHAT MOVED ON {MOVE_DAY} (n={len(mv)})   "
          f"median {mv['ret'].median():+.2f}% | <-4%: {(mv['ret']<=-4).sum()} names | "
          f">+4%: {(mv['ret']>=4).sum()} names")
    print("=" * W)
    losers = mv.nsmallest(20, "ret")
    gainers = mv.nlargest(8, "ret")
    print("  top losers : " + ", ".join(f"{r.t}({r.ret:+.1f}%)" for r in losers.itertuples()))
    print("  top gainers: " + ", ".join(f"{r.t}({r.ret:+.1f}%)" for r in gainers.itertuples()))

    # ---- PART 2: forensic audit vs night-before recorded signals ----
    print("\n" + "=" * W)
    print("  PART 2 — DID OUR RECORDED NIGHT-BEFORE SIGNALS SEE IT?")
    print("=" * W)
    uoa9 = json.load(open("/tmp/uoa_jun9.json"))
    putflow = {r["ticker"] for r in uoa9["rows"]
               if r["direction"] == "PUT" or (any(s["type"] == "put" for s in r["standouts"])
                                              and r["pcr"] >= 1.0)}
    uoa_names = {r["ticker"] for r in uoa9["rows"]}
    big_losers = mv[mv["ret"] <= LOSER_CUT]
    bl_in_uoa = big_losers[big_losers["t"].isin(uoa_names)]
    hits = bl_in_uoa[bl_in_uoa["t"].isin(putflow)]
    print(f"  Jun-9 UOA put-flow list: {len(putflow)} names")
    print(f"  big losers (<= {LOSER_CUT}%) that were even scannable by UOA: {len(bl_in_uoa)}")
    print(f"  of those, flagged PUT-FLOW the night before: {len(hits)} "
          f"-> {', '.join(hits['t'])}" if len(hits) else "  none flagged")
    if len(bl_in_uoa):
        hr = len(hits) / len(bl_in_uoa) * 100
        # false-positive rate: putflow names that did NOT drop big
        pf = mv[mv["t"].isin(putflow)]
        print(f"  hit rate on scannable big losers: {hr:.0f}%")
        print(f"  all {len(pf)} put-flow names' Jun-10 move: avg {pf['ret'].mean():+.2f}% "
              f"(panel avg {mv['ret'].mean():+.2f}%), {(pf['ret']<0).mean()*100:.0f}% red")

    # ---- PART 3: the loser signature at the Jun-9 close ----
    print("\n" + "=" * W)
    print(f"  PART 3 — WHAT THE BIG LOSERS LOOKED LIKE AT THE {PRE_DAY} CLOSE")
    print("=" * W)
    rows = []
    for r in mv.itertuples():
        f = feats_asof(hist[r.t], r.pi)
        f["t"], f["ret"] = r.t, r.ret
        rows.append(f)
    fd = pd.DataFrame(rows)
    L = fd[fd["ret"] <= LOSER_CUT]
    R = fd[fd["ret"] > LOSER_CUT]
    print(f"  {'feature (as-of Jun 9)':<26}{'big losers':>12}{'rest':>10}")
    for k, lab in [("ret2", "2-day return %"), ("ret5", "5-day return %"),
                   ("vs20", "% vs 20DMA"), ("vs50", "% vs 50DMA"), ("rsi", "RSI(14)"),
                   ("vol_x", "volume x avg"), ("dist10", "distribution days /10")]:
        print(f"  {lab:<26}{L[k].mean():>12.2f}{R[k].mean():>10.2f}")
    print(f"  {'below 20DMA %':<26}{L['below20'].mean()*100:>11.0f}%{R['below20'].mean()*100:>9.0f}%")
    print(f"  {'weak close (C<O) %':<26}{L['weak_close'].mean()*100:>11.0f}%{R['weak_close'].mean()*100:>9.0f}%")

    # ---- PART 4: validate candidate continuation signals on the panel ----
    print("\n" + "=" * W)
    print("  PART 4 — DO THOSE SIGNATURES PREDICT WEAKNESS HISTORICALLY? "
          "(panel, ~16 months, fwd 1d/3d)")
    print("=" * W)
    sigs = {
        "S1 2d<=-4% & below20DMA": lambda f: f["ret2"] <= -4 and f["below20"],
        "S2 S1 & vol>=1.5x": lambda f: f["ret2"] <= -4 and f["below20"] and f["vol_x"] >= 1.5,
        "S3 dist>=3 & below20DMA": lambda f: f["dist10"] >= 3 and f["below20"],
        "S4 weak close & 1d<=-3%": lambda f: f["weak_close"] and f["ret1"] <= -3,
        "S5 S2 & RSI>30 (not washed)": lambda f: f["ret2"] <= -4 and f["below20"]
                                                  and f["vol_x"] >= 1.5 and f["rsi"] > 30,
    }
    stats = {k: [] for k in sigs}
    base = []
    sample = list(hist.items())
    for t, df in sample:
        c = df["Close"].astype(float)
        n = len(df)
        for i in range(60, n - 4):
            f = feats_asof(df, i)
            f1 = (float(c.iloc[i + 1]) / float(c.iloc[i]) - 1) * 100
            f3 = (float(c.iloc[min(i + 3, n - 1)]) / float(c.iloc[i]) - 1) * 100
            base.append((f1, f3))
            for k, fn in sigs.items():
                if fn(f):
                    stats[k].append((f1, f3))
    b = np.array(base)
    print(f"  baseline (n={len(b):,}): fwd1d {b[:,0].mean():+.3f}% · fwd3d {b[:,1].mean():+.3f}% · "
          f"%red next day {(b[:,0]<0).mean()*100:.0f}%")
    print(f"\n  {'signal':<30}{'n':>7}{'fwd1d':>8}{'fwd3d':>8}{'%red1d':>8}")
    for k, vals in stats.items():
        if not vals:
            continue
        a = np.array(vals)
        print(f"  {k:<30}{len(a):>7,}{a[:,0].mean():>+8.3f}{a[:,1].mean():>+8.3f}"
              f"{(a[:,0]<0).mean()*100:>7.0f}%")
    print("\n  (a useful PUT signal needs fwd returns clearly BELOW baseline and %red >55%)")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
