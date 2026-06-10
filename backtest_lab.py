#!/usr/bin/env python3
"""Backtest LAB — find which logic changes would actually raise the success rate.

Runs the real scanner as-of ~7 historical dates (21..63 trading days back),
collects every recommended long (class A/A+, final>=75) with its scan-time
features, simulates the trade (trigger -> 2R target vs stop), then:

  PART 1: winners-vs-losers feature comparison (what separates them?)
  PART 2: candidate rule variants, each tested across ALL windows with a
          consistency count (improved in how many windows?) to avoid
          overfitting one lucky period.

Variants tested:
  base : current logic (10d trigger window, tightest-of stop, 2R target)
  V1   : skip extended names (extension_flag set at scan time)
  V2   : only near-pivot setups (-1%..6% below pivot)
  V3   : volume confirmation on breakout day (vol > 1.2x avg20)
  V4   : pure ATR stop (entry - 2.0*ATR) instead of tightest-of
  V5   : take profit at 1.5R         V6: take profit at 3R
  V7   : shorter trigger window (5d)
  V8   : skip if bearish_final >= 60 (bull/bear conflict filter)
  V9   : market gate at trigger day (SPY close > its 20DMA that day)
  V10  : V2 + V3 combined

Same caveats as backtest.py: survivorship-biased universe; windows overlap.
Consistency across windows is the anti-overfit guard.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import backtest as BT
import vcp_tracker as V

OFFSETS = [21, 28, 35, 42, 49, 56, 63]


def simulate(full_df, asof_idx, entry, stop, target, trig_win=10,
             vol_confirm=False, spy=None, spy_sma20=None):
    """Walk daily bars: trigger then stop-vs-target (stop checked first)."""
    fwd = full_df.iloc[asof_idx:]
    if fwd.empty or not (entry > 0 and stop > 0 and target > entry and entry > stop):
        return "invalid", 0.0
    v20 = full_df["Volume"].rolling(20).mean()
    trig = None
    for i in range(min(trig_win, len(fwd))):
        if float(fwd["High"].iloc[i]) >= entry:
            if vol_confirm:
                vavg = v20.iloc[asof_idx + i]
                if not (vavg > 0 and float(fwd["Volume"].iloc[i]) > 1.2 * vavg):
                    continue
            if spy is not None:
                d = fwd.index[i]
                try:
                    if float(spy.asof(d)) <= float(spy_sma20.asof(d)):
                        continue
                except Exception:  # noqa: BLE001
                    pass
            trig = i
            break
    if trig is None:
        return "no-trigger", 0.0
    risk = entry - stop
    rwin = (target - entry) / risk
    for i in range(trig, len(fwd)):
        if float(fwd["Low"].iloc[i]) <= stop:
            return "stop", -1.0
        if float(fwd["High"].iloc[i]) >= target:
            return "target", round(rwin, 2)
    return "open", round((float(fwd["Close"].iloc[-1]) - entry) / risk, 2)


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
    spy = bench_full["SPY"]
    spy_sma20 = spy.rolling(20).mean()

    # ---- collect all recommended longs with features ----
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
            recs.append({
                "off": off, "t": t, "df": df, "asof_idx": asof_idx,
                "entry": m.entry, "stop": m.stop, "target": m.target,
                "atr": m.atr14, "final": m.final_score, "trend": m.trend,
                "pattern": m.combined, "rs": m.rs_score, "sector": m.sector_score,
                "regime": m.market_regime, "dist_piv": m.dist_to_pivot,
                "ext50": (m.price / m.sma50 - 1) * 100 if m.sma50 else 0,
                "from_high": m.pct_from_high, "atr_pct": m.atr14 / m.price * 100,
                "extflag": bool(m.extension_flag), "bear": m.bearish_final,
                "best": m.best_pattern,
            })
    print(f"  collected {len(recs)} recommended longs across {len(OFFSETS)} windows",
          file=sys.stderr)

    # ---- baseline outcomes + features ----
    for r in recs:
        out, rm = simulate(r["df"], r["asof_idx"], r["entry"], r["stop"], r["target"])
        r["out"], r["r"] = out, rm
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "df"} for r in recs])
    trig = d[d["out"] != "no-trigger"]
    dec = d[d["out"].isin(["target", "stop"])]

    W = 100
    print("\n" + "=" * W)
    print(f"  PART 1 — WINNERS vs LOSERS  ({len(dec)} decided trades from "
          f"{len(d)} recs, {len(OFFSETS)} windows)")
    print("=" * W)
    win, los = dec[dec["out"] == "target"], dec[dec["out"] == "stop"]
    print(f"  baseline: {len(win)} wins / {len(los)} stops  -> win rate "
          f"{len(win)/len(dec)*100:.0f}%  (2R breakeven 33%)  avg R (triggered) "
          f"{trig['r'].mean():+.2f}")
    print(f"\n  {'feature':<22}{'winners':>10}{'losers':>10}   read")
    rows = [
        ("final score", "final"), ("trend quality", "trend"), ("pattern score", "pattern"),
        ("rel strength", "rs"), ("sector score", "sector"), ("market regime", "regime"),
        ("dist to pivot %", "dist_piv"), ("% above 50DMA", "ext50"),
        ("% from 52w high", "from_high"), ("ATR % of price", "atr_pct"),
        ("bearish score", "bear"),
    ]
    for lab, c in rows:
        wm, lm = win[c].mean(), los[c].mean()
        gap = wm - lm
        flag = "  <-- separates" if abs(gap) > max(1.0, 0.15 * (abs(wm) + abs(lm)) / 2) else ""
        print(f"  {lab:<22}{wm:>10.1f}{lm:>10.1f}{flag}")
    extw = dec[dec["extflag"]]
    nxw = dec[~dec["extflag"]]
    if len(extw) and len(nxw):
        print(f"\n  extended-at-scan:   win rate {(extw['out']=='target').mean()*100:.0f}% "
              f"(n={len(extw)})   |   not-extended: {(nxw['out']=='target').mean()*100:.0f}% "
              f"(n={len(nxw)})")
    print("\n  by best pattern (decided n>=8):")
    for p, g in dec.groupby("best"):
        if len(g) >= 8:
            print(f"    {p:<16} win {((g['out']=='target').mean()*100):>4.0f}%  n={len(g)}")

    # ---- PART 2: variants ----
    def run_variant(name, filt=None, stop_fn=None, target_fn=None,
                    trig_win=10, vol_confirm=False, gate_spy=False):
        per_window = {}
        rs_all, ndec, nwin, ntrig, nrecs = [], 0, 0, 0, 0
        for r in recs:
            if filt and not filt(r):
                continue
            nrecs += 1
            entry = r["entry"]
            stop = stop_fn(r) if stop_fn else r["stop"]
            target = target_fn(r, entry, stop) if target_fn else entry + 2 * (entry - stop)
            out, rm = simulate(r["df"], r["asof_idx"], entry, stop, target,
                               trig_win=trig_win, vol_confirm=vol_confirm,
                               spy=spy if gate_spy else None,
                               spy_sma20=spy_sma20 if gate_spy else None)
            if out == "no-trigger" or out == "invalid":
                continue
            ntrig += 1
            rs_all.append(rm)
            per_window.setdefault(r["off"], []).append(rm)
            if out in ("target", "stop"):
                ndec += 1
                nwin += out == "target"
        if not ntrig:
            return None
        base_by_win = {off: np.mean([r2["r"] for r2 in recs
                                     if r2["off"] == off and r2["out"] != "no-trigger"])
                       for off in OFFSETS}
        better = sum(1 for off, vals in per_window.items()
                     if np.mean(vals) > base_by_win.get(off, -9) + 1e-9)
        return {"name": name, "recs": nrecs, "trig": ntrig, "dec": ndec,
                "win%": (nwin / ndec * 100) if ndec else 0,
                "avgR": float(np.mean(rs_all)), "windows+": f"{better}/{len(per_window)}"}

    variants = [
        run_variant("base (current)"),
        run_variant("V1 skip extended", filt=lambda r: not r["extflag"]),
        run_variant("V2 near-pivot only", filt=lambda r: -1 <= r["dist_piv"] <= 6),
        run_variant("V3 vol-confirm trigger", vol_confirm=True),
        run_variant("V4 pure 2*ATR stop", stop_fn=lambda r: r["entry"] - 2.0 * r["atr"]),
        run_variant("V5 1.5R target",
                    target_fn=lambda r, e, s: e + 1.5 * (e - s)),
        run_variant("V6 3R target",
                    target_fn=lambda r, e, s: e + 3.0 * (e - s)),
        run_variant("V7 5d trigger window", trig_win=5),
        run_variant("V8 skip bear>=60", filt=lambda r: r["bear"] < 60),
        run_variant("V9 SPY>20DMA gate", gate_spy=True),
        run_variant("V10 near-pivot + vol",
                    filt=lambda r: -1 <= r["dist_piv"] <= 6, vol_confirm=True),
        run_variant("V11 ATRstop + vol", stop_fn=lambda r: r["entry"] - 2.0 * r["atr"],
                    vol_confirm=True),
    ]
    print("\n" + "=" * W)
    print("  PART 2 — RULE VARIANTS (tested on the same trades; 'windows+' = "
          "windows where avg R beat baseline)")
    print("=" * W)
    print(f"  {'variant':<24}{'recs':>6}{'trig':>6}{'decided':>8}{'win%':>7}"
          f"{'avgR':>8}{'windows+':>10}")
    for v in variants:
        if v:
            print(f"  {v['name']:<24}{v['recs']:>6}{v['trig']:>6}{v['dec']:>8}"
                  f"{v['win%']:>6.0f}%{v['avgR']:>+8.2f}{v['windows+']:>10}")
    print("\n  avgR includes open positions at mark; win% is decided trades only.")
    print("  A variant is VALID if avgR improves AND windows+ is >= 5/7 "
          "(consistent, not one lucky window).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
