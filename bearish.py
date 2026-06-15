#!/usr/bin/env python3
"""Downside Radar — a transparent WEAKNESS ranking (NOT a trade engine).

  ⚠ The former 7-pattern bearish TRADE engine was REMOVED. It was backtested and
  had no edge, and every short approach we tested lost money:
    - bearish chart patterns                  -> no edge
    - acute risk-off index puts               -> SPY 33%/-0.06R, QQQ 25%/-0.25R
    - UOA put-flow + confirmation + regime    -> breakeven-to-negative (all targets)
    - trend-confirmed index short (2022 incl.)-> -6% / -1%, negative Sharpe
  The validated bear-market action is CASH (200DMA timing cut drawdown -23%->-11%),
  not shorting. See backtest_put_engine.py and backtest_defense.py.

This module now only computes a 0-100 *weakness score* per name + plain-English
flags, used for HEDGING / DE-RISKING context (which longs to trim) — never as a
put recommendation. It deliberately keeps the function names the scanner and the
backtest labs already import, so nothing downstream breaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Weakness band thresholds (informational only).
BANDS = [(75, "WEAK"), (55, "SOFT"), (35, "OK")]


def _clip(x):
    return float(np.clip(x, 0, 100))


def _ret(c: pd.Series, days: int) -> float:
    if c is None or len(c) <= days:
        return 0.0
    return float(c.iloc[-1] / c.iloc[-1 - days] - 1) * 100


def _sma(c: pd.Series, n: int) -> float:
    if c is None or len(c) < n:
        return float("nan")
    return float(c.rolling(n).mean().iloc[-1])


def _rsi(c: pd.Series, n: int = 14) -> float:
    if c is None or len(c) < n + 1:
        return 50.0
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean().iloc[-1]
    dn = (-d.clip(upper=0)).rolling(n).mean().iloc[-1]
    if dn == 0:
        return 100.0
    rs = up / dn
    return float(100 - 100 / (1 + rs))


# --------------------------------------------------------------------------
# Market / sector weakness context (used for the cash/de-risk read)
# --------------------------------------------------------------------------
def bearish_market_score(bench: dict, vix: pd.Series, metrics: list) -> float:
    spy, qqq, iwm = bench.get("SPY"), bench.get("QQQ"), bench.get("IWM")
    s = 0.0
    if spy is not None and spy.iloc[-1] < _sma(spy, 50): s += 15
    if qqq is not None and qqq.iloc[-1] < _sma(qqq, 50): s += 15
    if spy is not None and spy.iloc[-1] < _sma(spy, 20): s += 10
    if qqq is not None and qqq.iloc[-1] < _sma(qqq, 20): s += 10
    if qqq is not None and _sma(qqq, 20) < _sma(qqq, 50): s += 15
    if vix is not None and len(vix) > 6 and float(vix.iloc[-1]) > float(vix.iloc[-6]): s += 15
    if iwm is not None and iwm.iloc[-1] < _sma(iwm, 50): s += 10
    if metrics:
        below20 = sum(1 for m in metrics if m.price < _sma(m.df["Close"], 20)) / len(metrics)
        if below20 > 0.6: s += 10
    return _clip(s)


def bearish_sector_score(etf: pd.Series, spy: pd.Series, qqq: pd.Series) -> float:
    if etf is None:
        return 50.0
    s = 0.0
    price = float(etf.iloc[-1])
    if price < _sma(etf, 50): s += 20
    if price < _sma(etf, 20): s += 15
    if spy is not None and _ret(etf, 21) < _ret(spy, 21): s += 20
    if qqq is not None and _ret(etf, 21) < _ret(qqq, 21): s += 20
    if len(etf) > 20 and price < float(etf.tail(20).iloc[:10].min()): s += 15
    return _clip(s)


def bearish_sector_table(bench: dict, sector_etfs: list) -> dict:
    spy, qqq = bench.get("SPY"), bench.get("QQQ")
    return {e: bearish_sector_score(bench.get(e), spy, qqq) for e in sector_etfs}


def classify(weakness: float) -> str:
    for thresh, name in BANDS:
        if weakness >= thresh:
            return name
    return "FIRM"


# --------------------------------------------------------------------------
# Per-name weakness score (transparent, technical-only)
# --------------------------------------------------------------------------
def weakness_components(m, bench: dict):
    """0-100 weakness + plain-English flags. Higher = weaker / more de-risk."""
    s = 0.0
    flags = []
    if m.ema21 and m.price < m.ema21:
        s += 10; flags.append("<21EMA")
    if m.sma50 and m.price < m.sma50:
        s += 15; flags.append("<50DMA")
    if m.below_200:
        s += 15; flags.append("<200DMA")
    if m.sma50 and m.sma50_prev and m.sma50 < m.sma50_prev:
        s += 10; flags.append("50DMA falling")
    if m.rs_spy_1m < 0:
        s += 10; flags.append("RS 1m-")
    if m.rs_spy_3m < 0:
        s += 10; flags.append("RS 3m-")
    spy = bench.get("SPY")
    if spy is not None and _ret(m.df["Close"], 5) < _ret(spy, 5):
        s += 10; flags.append("lags SPY 5d")
    d = m.df.tail(20)
    ch = d["Close"].pct_change()
    if float(d["Volume"][ch < 0].sum()) > float(d["Volume"][ch > 0].sum()):
        s += 10; flags.append("distribution")
    if getattr(m, "lower_lows", False):
        s += 10; flags.append("lower lows")
    return _clip(s), flags


def score_stock(m, bench: dict, bear_market: float, sector_scores: dict):
    """Set the weakness fields on the metric. No put levels, no trade signal."""
    weakness, flags = weakness_components(m, bench)
    m.bearish_pattern_score = weakness
    m.bearish_final = weakness
    m.bearish_classification = classify(weakness)
    m.bear_detected = flags
    m.bear_best_pattern = flags[0] if flags else ""
    m.bear_market_score = bear_market
    m.bear_sector_score = sector_scores.get(m.sector_etf, 50.0)
    m.bear_rel_weakness = max(0.0, -(m.rs_spy_1m))    # magnitude of 1m underperformance


def compute_final(m):
    """Kept for API compatibility (labs call it). Weakness is already the final
    rank; nothing to re-blend now that the pattern engine is gone."""
    m.bearish_final = m.bearish_pattern_score
