#!/usr/bin/env python3
"""Daily SWING self-assessment — "would the scanner's recommendations actually
have worked?", with REALISTIC portfolio constraints.

This is the swing-appropriate sibling of the intraday project's daily check.
Because swing trades take weeks to resolve, a same-day grade is meaningless — so
each run evaluates the COHORT of recommendations the scanner would have made
~COHORT_LOOKBACK trading days ago (resolved by now), keying off the data index so
weekends/holidays need no handling.

It reuses backtest.py (the only script that scores past dates with the REAL
scanner code and simulates the recommended trades), and adds the portfolio
reality backtest.py lacks: the same caps the live scanner enforces in main()
  - <= MAX_OPEN concurrent positions
  - <= MAX_PER_SECTOR per sector ETF
  - one per ticker
  - ranked by Final score
so it grades the trades the scanner would ACTUALLY have taken, not every signal.

It appends one line to backtest_journal.md and prints an analysis + PROPOSED
corrections. It NEVER changes scanner logic. Hard rule (printed every run): a
tweak that looks good on ONE day is usually overfitting — adopt only once it
holds across MULTIPLE journal days.

Usage:  source tradier_creds.sh && python3 daily_assess.py
        COHORT_LOOKBACK=30 python3 daily_assess.py    # tune the resolve horizon
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import backtest as BT
import vcp_tracker as V

ROOT = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(ROOT, "backtest_journal.md")
LOOKBACK = int(os.environ.get("COHORT_LOOKBACK", "25"))          # trading days back
MAX_OPEN = int(os.environ.get("ASSESS_MAX_OPEN", "5"))           # mirror scanner caps
MAX_PER_SECTOR = int(os.environ.get("ASSESS_MAX_PER_SECTOR", "2"))
RISK_PCT = float(os.environ.get("ASSESS_RISK_PCT", "1.5"))       # % equity risked / trade

_USAGE = (
    "**How & when to use this — it's a drift detector, NOT a trade signal.** Each row "
    "grades trades from ~5 weeks ago, so nothing here is time-sensitive; don't act on it "
    "like a signal.\n"
    "- **Review weekly / every ~10 rows**, not daily.\n"
    "- **Act on persistence, never one row.** A single bad row is noise (the sample is 0-5 trades).\n"
    "- **If an observation/proposal recurs across 5+ rows** (or Win%/Avg R trends down for "
    "weeks), turn it into ONE specific tweak, validate it with `backtest_lab.py`, and adopt "
    "only if it holds in >=5/7 windows. Never change logic on one day's evidence.\n"
    "- **Empty cohorts**: in a weak tape = correct (cash); in a strong tape = investigate the "
    "A/A+ gate.\n"
    "- Full per-run analysis + proposals are in `reports/assess_YYYY-MM-DD.log`.\n\n")
HEADER = (("# Backtest journal — daily swing self-assessment\n\n"
           "One line per run. Grades the cohort of A/A+ recommendations the scanner "
           "would have taken ~%d trading days earlier (portfolio-constrained: <=%d open, "
           "<=%d/sector), simulated trigger->3R vs ATR stop. Est net %% = sum(R) x %.1f%% "
           "risk/slot. Proposals live in the run logs, NOT here; adopt a change only "
           "after it holds across MULTIPLE rows.\n\n")
          % (LOOKBACK, MAX_OPEN, MAX_PER_SECTOR, RISK_PCT)
          + _USAGE
          + "| Run (local) | Cohort as-of | Sel | Trig | Win% | Avg R | Est net % | Notes |\n"
            "|---|---|---|---|---|---|---|---|\n")


def _prior_runs() -> int:
    try:
        with open(JOURNAL) as fh:
            return sum(1 for ln in fh if ln.startswith("| 20"))
    except FileNotFoundError:
        return 0


def _append_journal(line: str):
    new = not os.path.exists(JOURNAL)
    with open(JOURNAL, "a") as fh:
        if new:
            fh.write(HEADER)
        fh.write(line + "\n")


def main() -> int:
    if not BT.TRADIER_TOKEN:
        print("TRADIER_TOKEN required (source tradier_creds.sh or ../Candle/.env)",
              file=sys.stderr)
        return 1
    V.apply_universe()
    tickers = list(V.UNIVERSE)
    sector_etfs = list(V.SECTOR_ETFS)
    syms = set(tickers + ["SPY", "QQQ", "IWM"] + sector_etfs + ["^VIX"])
    print(f"  fetching {len(syms)} histories ...", file=sys.stderr)
    hist = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for s, df in ex.map(lambda x: (x, BT.fetch_long_history(x)), syms):
            if df is not None and len(df) > 250:
                hist[s] = df
    if "SPY" not in hist:
        print("no SPY history", file=sys.stderr)
        return 1
    bench_full = {k: hist[k]["Close"].astype(float)
                  for k in ["SPY", "QQQ", "IWM"] + sector_etfs if k in hist}
    vix_full = hist["^VIX"]["Close"].astype(float) if "^VIX" in hist else None
    spy = bench_full["SPY"]
    cohort_date = spy.index[len(spy) - LOOKBACK].date() if len(spy) > LOOKBACK else None

    # ---- score the cohort as-of LOOKBACK trading days ago ----
    cands = []
    for t in tickers:
        df = hist.get(t)
        if df is None or len(df) <= LOOKBACK + 210:
            continue
        asof_idx = len(df) - LOOKBACK
        m, asof = BT.score_asof(t, df, bench_full, vix_full, asof_idx, sector_etfs)
        if m is None or m.classification not in ("A+", "A") or m.final_score < 75:
            continue
        cands.append({"t": t, "df": df, "asof_idx": asof_idx, "final": m.final_score,
                      "sector": m.sector_etf, "entry": m.entry, "stop": m.stop,
                      "target": m.target, "cls": m.classification,
                      "weakvol": bool(getattr(m, "triggered_weak_vol", False))})

    # ---- portfolio selection mirroring the live scanner caps ----
    cands.sort(key=lambda c: -c["final"])
    selected, per_sec = [], {}
    for c in cands:
        if len(selected) >= MAX_OPEN:
            break
        if per_sec.get(c["sector"], 0) >= MAX_PER_SECTOR:
            continue
        selected.append(c)
        per_sec[c["sector"]] = per_sec.get(c["sector"], 0) + 1

    # ---- simulate the trades the scanner would have taken ----
    for c in selected:
        out, r = BT.simulate_long(c["df"], c["asof_idx"], c["entry"], c["stop"], c["target"])
        c["out"], c["r"] = out, r
    trig = [c for c in selected if c["out"] not in ("no-trigger", "invalid")]
    decided = [c for c in trig if c["out"] in ("target", "stop")]
    wins = [c for c in decided if c["out"] == "target"]
    winpct = len(wins) / len(decided) * 100 if decided else 0.0
    avg_r = float(np.mean([c["r"] for c in trig])) if trig else 0.0
    net_pct = sum(c["r"] for c in trig) * RISK_PCT          # equal-risk portfolio proxy
    notrig = [c for c in selected if c["out"] == "no-trigger"]

    # ---- observations (concise, factual) ----
    obs = []
    if not selected:
        obs.append("no A/A+ recs in cohort (regime/selectivity gated everything out)")
    else:
        obs.append(f"{len(trig)}/{len(selected)} triggered")
        if trig:
            best = max(trig, key=lambda c: c["r"])
            worst = min(trig, key=lambda c: c["r"])
            obs.append(f"best {best['t']} {best['r']:+.1f}R / worst {worst['t']} {worst['r']:+.1f}R")
        if notrig:
            obs.append(f"{len(notrig)} never triggered")
        wv_stops = [c for c in decided if c["out"] == "stop" and c["weakvol"]]
        if wv_stops:
            obs.append(f"{len(wv_stops)} stop(s) were weak-vol triggers")
    notes = "; ".join(obs)

    run_local = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = (f"| {run_local} | {cohort_date} | {len(selected)} | {len(trig)} | "
            f"{winpct:.0f}% | {avg_r:+.2f} | {net_pct:+.1f}% | {notes} |")
    _append_journal(line)
    run_n = _prior_runs()

    # ---- print analysis + PROPOSALS (never auto-applied) ----
    W = 84
    print("\n" + "=" * W)
    print(f"  DAILY SWING ASSESSMENT — cohort as-of {cohort_date} "
          f"(~{LOOKBACK} trading days ago)   [journal run #{run_n}]")
    print("=" * W)
    print(f"  recommendations in cohort: {len(cands)}  ->  portfolio-selected: "
          f"{len(selected)} (<= {MAX_OPEN} open, <= {MAX_PER_SECTOR}/sector)")
    if selected:
        print(f"  {'ticker':<8}{'cls':>4}{'final':>7}{'sector':>7}{'outcome':>11}{'R':>8}"
              f"{'  weak-vol':>10}")
        for c in sorted(selected, key=lambda x: -x["r"]):
            print(f"  {c['t']:<8}{c['cls']:>4}{c['final']:>7.0f}{c['sector']:>7}"
                  f"{c['out']:>11}{c['r']:>+8.2f}{('  yes' if c['weakvol'] else '  -'):>10}")
    print(f"\n  triggered {len(trig)}/{len(selected)} | decided win rate {winpct:.0f}% "
          f"(2R+ breakeven ~33%) | avg {avg_r:+.2f}R | est portfolio P&L {net_pct:+.1f}%")
    print(f"  journaled -> {os.path.relpath(JOURNAL, ROOT)}")

    print("\n  ANALYSIS + PROPOSED corrections (NOT applied):")
    proposals = []
    if decided and winpct < 40:
        proposals.append("decided win rate < 40% this cohort — possible selection/entry "
                         "weakness. Do NOT change yet; watch the journal.")
    if selected and len(notrig) >= max(2, 0.4 * len(selected)):
        proposals.append(f"{len(notrig)} of {len(selected)} entries never triggered — entries "
                         "may sit too far above the pivot; candidate: tighten the trigger "
                         "window or require closer-to-pivot. HOLD pending repeat.")
    wv_all = [c for c in decided if c["out"] == "stop" and c["weakvol"]]
    if wv_all:
        proposals.append(f"{len(wv_all)} stop(s) were weak-volume triggers — the volume-confirm "
                         "gate is the right lever; confirm it is capping these to B.")
    if not selected:
        proposals.append("empty cohort — if this persists across many bullish days, the A/A+ "
                         "gate may be too strict; if the tape was weak, this is correct (cash).")
    if not proposals:
        proposals.append("nothing actionable — cohort behaved within expectations.")
    for p in proposals:
        print(f"   - {p}")
    print(f"\n  RULE: a one-day result is usually a mirage. Adopt a correction only after it "
          f"holds\n  across MULTIPLE journal rows (run #{run_n}). This script never edits logic.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
