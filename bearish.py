#!/usr/bin/env python3
"""Bearish PUT scanner — the downside half of the Master Opportunity Scanner.

Operates on the SAME metrics (M objects) the bullish scanner already built, so
it reuses all indicators. It is NOT an inversion of the bullish scanner: it uses
bearish-specific patterns (distribution, failed breakouts, breakdowns) because
down-moves are faster and more violent.

7 patterns -> bearish_pattern_score; combined with a bearish market regime,
bearish sector weakness, relative weakness, and put-option liquidity into a
final_bearish_score with A+/A/Watch/Ignore classification. Also computes put
levels (pattern-aware stops) and an oversold "do-not-chase" guard.

Standalone (no import of vcp_tracker) to avoid cycles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_WEIGHTS = {
    "failed_breakout": 0.25, "late_base": 0.20, "distribution": 0.20,
    "dma_breakdown": 0.15, "rel_weakness": 0.10, "bear_flag": 0.05, "gap_down": 0.05,
}
PATTERN_LABELS = {
    "failed_breakout": "Failed Breakout", "late_base": "Late-Stage Failed Base",
    "distribution": "Distribution Cluster", "dma_breakdown": "50DMA Breakdown",
    "rel_weakness": "Relative Weakness", "bear_flag": "Bear Flag",
    "gap_down": "Gap-Down Continuation",
}
FINAL_WEIGHTS = {"market": 0.30, "sector": 0.20, "pattern": 0.25, "relweak": 0.15, "liq": 0.10}
LIQ_PRIOR = 50.0


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
# 7 bearish pattern detectors  (each returns 0-100)
# --------------------------------------------------------------------------
def failed_breakout(m) -> float:
    df = m.df
    if len(df) < 60:
        return 0.0
    close, high, vol = df["Close"], df["High"], df["Volume"]
    res = float(high.iloc[-40:-10].max())          # prior resistance
    last10 = df.iloc[-10:]
    broke = bool((last10["High"] > res).any())
    below = m.price < res
    s = 0.0
    if broke:
        s += 20
    if broke and below:
        s += 25
    if float(vol.iloc[-1]) > m.avgvol20:
        s += 20
    if m.price < m.ema21 or m.price < m.sma50:
        s += 20
    if m.rs_spy_1m < 0 or m.rs_qqq_1m < 0:
        s += 15
    if broke and not below:                          # reclaimed -> reject
        s *= 0.3
    if below and float(vol.iloc[-1]) < m.avgvol20 * 0.8:  # low-vol failure
        s *= 0.6
    return _clip(s)


def late_base(m) -> float:
    close = m.df["Close"]
    if len(close) < 200:
        return 0.0
    adv = (float(close.tail(252).max()) / float(close.tail(252).min()) - 1) * 100
    s = 0.0
    if adv >= 50:
        s += 15
    if adv >= 50 and len(m.contractions) >= 2:
        s += 15
    if m.price < m.sma50:
        s += 25                                       # breakout failure / below support
    if m.price < m.sma50 and m.sma50 < m.sma50_prev:
        s += 25                                       # break below declining 50
    if float(m.df["Volume"].tail(5).mean()) > m.avgvol50:
        s += 20                                       # distribution
    if m.price > m.sma50 and m.sma50 > m.sma150:      # early-stage uptrend -> reject
        s *= 0.3
    return _clip(s)


def distribution(m) -> float:
    d = m.df.tail(20)
    ch = d["Close"].pct_change() * 100
    vol = d["Volume"]
    dd = int(((d["Close"] < d["Close"].shift(1)) &
              ((vol > vol.shift(1)) | (vol > m.avgvol20)) & (ch < -1)).sum())
    s = 0.0
    if dd >= 3:
        s += 25
    if dd >= 5:
        s += 15
    if float(vol[ch < 0].sum()) > float(vol[ch > 0].sum()):
        s += 20
    if (float(d["High"].iloc[-1]) < float(d["High"].iloc[:10].max())
            and float(d["Low"].iloc[-1]) < float(d["Low"].iloc[:10].min())):
        s += 20
    if m.rs_spy_1m < 0 or m.rs_qqq_1m < 0:
        s += 20
    if m.pct_from_high > -3 and m.rs_spy_3m > 0:      # still near highs, strong -> reject
        s *= 0.4
    return _clip(s)


def dma_breakdown(m) -> float:
    df = m.df
    if len(df) < 60:
        return 0.0
    below = m.price < m.sma50
    recent = df.tail(15)
    s = 0.0
    if below:
        s += 25
    if float(recent["Volume"].max()) > m.avgvol20 * 1.2:
        s += 20
    near = bool((abs(recent["High"] - m.sma50) / m.sma50 < 0.02).any())
    if near:
        s += 15
    if near and below:
        s += 25                                       # retest failed
    if m.rs_spy_1m < 0:
        s += 15
    if not below:
        s *= 0.2
    return _clip(s)


def rel_weakness(m, bench) -> float:
    spy, qqq = bench.get("SPY"), bench.get("QQQ")
    etf = bench.get(m.sector_etf)
    st = _ret(m.df["Close"], 5)
    s = 0.0
    if spy is not None and st < _ret(spy, 5):
        s += 15
    if m.rs_spy_1m < 0:
        s += 15
    if qqq is not None and st < _ret(qqq, 5):
        s += 15
    if m.rs_qqq_1m < 0:
        s += 15
    if (etf is not None and (st < _ret(etf, 5) or m.rs_sec_1m < 0)):
        s += 20
    if m.price < m.ema21 or m.price < m.sma50:
        s += 20
    return _clip(s)


def bear_flag(m) -> float:
    close, vol = m.df["Close"], m.df["Volume"]
    if len(close) < 20:
        return 0.0
    drop_seg = close.iloc[-15:-5]
    if len(drop_seg) < 2:
        return 0.0
    high0, low = float(drop_seg.iloc[0]), float(drop_seg.min())
    dropfall = (low / high0 - 1) * 100 if high0 else 0
    bounce = close.iloc[-5:]
    s = 0.0
    if dropfall <= -8:
        s += 20
    if float(vol.iloc[-5:].mean()) < m.avgvol20:
        s += 25                                       # weak low-volume bounce
    retr = (float(bounce.iloc[-1]) - low) / (high0 - low) * 100 if high0 > low else 0
    if 0 < retr < 50:
        s += 15
    if m.price <= float(bounce.min()) * 1.001:
        s += 25                                       # breaks flag support
    if m.rs_spy_1m < 0:
        s += 15
    return _clip(s)


def gap_down(m) -> float:
    df = m.df
    gd = None
    for i in range(len(df) - 1, max(len(df) - 11, 0), -1):
        if float(df["Open"].iloc[i]) < float(df["Close"].iloc[i - 1]) * 0.97:
            gd = i
            break
    if gd is None:
        return 0.0
    s = 20.0
    gd_high = float(df["High"].iloc[gd])
    if m.price < gd_high:
        s += 20
    if float(df["Volume"].iloc[gd]) > m.avgvol20:
        s += 20
    after = df.iloc[gd + 1:]
    if (len(after) >= 2 and float(after["High"].iloc[-1]) < gd_high
            and float(after["Low"].min()) < float(df["Low"].iloc[gd])):
        s += 20
    if m.rs_spy_1m < 0:
        s += 20
    if m.price > gd_high:                              # reversed back above -> reject
        s *= 0.2
    return _clip(s)


# --------------------------------------------------------------------------
# market / sector / final
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
    # (sector down-day volume needs ETF volume, not in bench -> omitted; max 90)
    return _clip(s)


def bearish_sector_table(bench: dict, sector_etfs: list) -> dict:
    spy, qqq = bench.get("SPY"), bench.get("QQQ")
    return {e: bearish_sector_score(bench.get(e), spy, qqq) for e in sector_etfs}


def classify(final: float) -> str:
    if final >= 85:
        return "A+"
    if final >= 75:
        return "A"
    if final >= 65:
        return "WATCH"
    return "IGNORE"


def do_not_chase(m) -> bool:
    below21 = (m.price - m.ema21) / m.ema21 * 100 if m.ema21 else 0
    below50 = (m.price - m.sma50) / m.sma50 * 100 if m.sma50 else 0
    rsi = _rsi(m.df["Close"])
    near_support = m.low_52w > 0 and m.price <= m.low_52w * 1.03
    return below21 < -10 or below50 < -15 or rsi < 25 or near_support


def put_levels(m, atr_mult: float = 1.5):
    """Entry (short on breakdown ~ current), pattern-aware stop ABOVE, 2R target."""
    entry = m.price
    swing_high = float(m.df["High"].tail(10).max())
    cands = [swing_high, m.sma50, entry + atr_mult * m.atr14, entry * 1.08]
    cands = [c for c in cands if c > entry]
    stop = min(cands) if cands else entry + atr_mult * m.atr14
    risk = stop - entry
    target = entry - 2 * risk if risk > 0 else entry * 0.9
    rr = (entry - target) / risk if risk > 0 else 0.0
    return round(entry, 2), round(stop, 2), round(target, 2), round(rr, 2)


# --------------------------------------------------------------------------
# orchestration: score one stock (sets m.bear_* fields, returns nothing)
# --------------------------------------------------------------------------
def score_stock(m, bench: dict, bear_market: float, sector_scores: dict):
    m.bear_failed_breakout = failed_breakout(m)
    m.bear_late_base = late_base(m)
    m.bear_distribution = distribution(m)
    m.bear_dma_breakdown = dma_breakdown(m)
    m.bear_rel_weakness = rel_weakness(m, bench)
    m.bear_flag = bear_flag(m)
    m.bear_gap_down = gap_down(m)

    scores = {"failed_breakout": m.bear_failed_breakout, "late_base": m.bear_late_base,
              "distribution": m.bear_distribution, "dma_breakdown": m.bear_dma_breakdown,
              "rel_weakness": m.bear_rel_weakness, "bear_flag": m.bear_flag,
              "gap_down": m.bear_gap_down}
    m.bearish_pattern_score = round(sum(PATTERN_WEIGHTS[k] * v for k, v in scores.items()), 1)
    m.bear_best_pattern = PATTERN_LABELS[max(scores, key=scores.get)]
    m.bear_detected = [PATTERN_LABELS[k] for k in PATTERN_WEIGHTS if scores[k] >= 60]

    m.bear_market_score = bear_market
    m.bear_sector_score = sector_scores.get(m.sector_etf, 50.0)
    m.do_not_chase_put = do_not_chase(m)
    compute_final(m)
    m.bearish_classification = classify(m.bearish_final)
    m.put_entry, m.put_stop, m.put_target, m.put_rr = put_levels(m)


def compute_final(m):
    """final_bearish = 0.30 market + 0.20 sector + 0.25 pattern + 0.15 relweak +
    0.10 liq (liq uses a neutral 50 prior until a put chain is fetched)."""
    liq = m.bear_liq_score if m.bear_liq_score >= 0 else LIQ_PRIOR
    m.bearish_final = round(
        FINAL_WEIGHTS["market"] * m.bear_market_score +
        FINAL_WEIGHTS["sector"] * m.bear_sector_score +
        FINAL_WEIGHTS["pattern"] * m.bearish_pattern_score +
        FINAL_WEIGHTS["relweak"] * m.bear_rel_weakness +
        FINAL_WEIGHTS["liq"] * liq, 1)
