#!/usr/bin/env python3
"""Options-focused bullish multi-pattern scanner for U.S. growth stocks.

Multi-LAYER pipeline: a bullish chart pattern is necessary but NOT sufficient.
Before any call is suggested the name must also clear market regime, sector
strength, relative strength, options liquidity, earnings risk, extension, and
risk-management filters. Layers are blended into one Final Trade Score and a
strict A+/A/B/Reject classification with position sizing and an exit plan.

Pattern layer (7 detectors): VCP, Pocket Pivot, Tight Weekly Close, Flat Base,
High Tight Flag, Three-Weeks-Tight, 21EMA/50SMA Moving-Average Bounce.

Filter layers:
  1 Market Regime   (SPY/QQQ/IWM/VIX)        30%
  2 Sector Strength (sector ETFs vs SPY/QQQ) 20%
  3 Pattern Score   (the 7 detectors)        20%
  4 Relative Strength(stock vs SPY/QQQ/ETF)  15%
  5 Options Liquidity(live chain)            10%
  6 Earnings Risk                             5%

All data is pulled FRESH each run. No hardcoded or remembered prices. The data
timestamp is printed. Educational only, NOT financial advice; long options can
expire worthless (-100%).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.simplefilter("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Universe (ticker -> (company, sector/theme))
# --------------------------------------------------------------------------
UNIVERSE_META: dict[str, tuple[str, str]] = {
    "NVDA": ("Nvidia", "AI/Semis"), "AVGO": ("Broadcom", "AI/Semis"),
    "AMD": ("Adv. Micro Devices", "AI/Semis"), "MRVL": ("Marvell", "AI/Semis"),
    "TSM": ("Taiwan Semi", "AI/Semis"), "MU": ("Micron", "AI/Semis"),
    "CRWD": ("CrowdStrike", "Cybersecurity"), "PANW": ("Palo Alto", "Cybersecurity"),
    "FTNT": ("Fortinet", "Cybersecurity"), "ZS": ("Zscaler", "Cybersecurity"),
    "CYBR": ("CyberArk", "Cybersecurity"),
    "PLTR": ("Palantir", "Cloud Software"), "DDOG": ("Datadog", "Cloud Software"),
    "MDB": ("MongoDB", "Cloud Software"), "NOW": ("ServiceNow", "Cloud Software"),
    "SNOW": ("Snowflake", "Cloud Software"),
    "ANET": ("Arista", "Data Center"), "VRT": ("Vertiv", "Data Center"),
    "META": ("Meta", "Mega-Cap"), "AMZN": ("Amazon", "Mega-Cap"),
    "MSFT": ("Microsoft", "Mega-Cap"), "GOOGL": ("Alphabet", "Mega-Cap"),
    "HOOD": ("Robinhood", "High-Beta"), "APP": ("AppLovin", "High-Beta"),
    "TSLA": ("Tesla", "High-Beta"), "COIN": ("Coinbase", "High-Beta"),
    "NFLX": ("Netflix", "High-Beta"),
}
UNIVERSE = list(UNIVERSE_META.keys())
BEAR_UNIVERSE: list[str] = []        # weakness/laggard list — PUT side only
BENCHMARKS = ["SPY", "QQQ", "IWM"]
VIX_SYMBOL = "^VIX"
RISK_FREE = 0.04

try:
    from zoneinfo import ZoneInfo
    _CT = ZoneInfo("America/Chicago")
except Exception:  # noqa: BLE001
    _CT = None


def _now_ct() -> str:
    """Timestamp in US Central time with an explicit CT label (the cloud server
    runs in UTC, so always render the user-facing time in CT)."""
    if _CT is not None:
        return dt.datetime.now(_CT).strftime("%Y-%m-%d %H:%M:%S CT")
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# Auto-tunable parameters (bounded). Defaults reproduce the original behavior;
# weekly_review.py may nudge them within TUNE_BOUNDS after backtest validation.
# --------------------------------------------------------------------------
TUNE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.json")
TUNE_DEFAULTS = {"chase_max_pct": 5.0, "vol_confirm_pct": 20.0}
TUNE_BOUNDS = {"chase_max_pct": (2.0, 8.0), "vol_confirm_pct": (10.0, 40.0)}


def _load_tuning() -> dict:
    t = dict(TUNE_DEFAULTS)
    try:
        import json as _json
        with open(TUNE_PATH) as fh:
            data = _json.load(fh)
        for k, (lo, hi) in TUNE_BOUNDS.items():
            if k in data:
                try:
                    t[k] = min(max(float(data[k]), lo), hi)
                except (TypeError, ValueError):
                    pass
    except Exception:  # noqa: BLE001
        pass
    return t


TUNE = _load_tuning()

# Options data source. Tradier returns real greeks + chains even after hours
# (last-close), so option details populate outside RTH for watchlisting.
TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")
OPTIONS_SOURCE = os.environ.get("OPTIONS_SOURCE", "auto")  # auto | tradier | yahoo
OHLCV_SOURCE = os.environ.get("OHLCV_SOURCE", "auto")      # auto | tradier | yahoo
# Committed (project root, not reports/) so the cloud ships with real earnings
# dates even when Yahoo blocks the datacenter IP. Yahoo updates it when reachable.
EARN_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "earnings_cache.json")

# Each ticker -> sector ETF used for sector-strength + relative-strength.
TICKER_ETF: dict[str, str] = {
    "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH", "MRVL": "SMH", "TSM": "SMH", "MU": "SMH",
    "CRWD": "CIBR", "PANW": "CIBR", "FTNT": "CIBR", "ZS": "CIBR", "CYBR": "CIBR",
    "PLTR": "IGV", "DDOG": "IGV", "MDB": "IGV", "NOW": "IGV", "SNOW": "IGV", "APP": "IGV",
    "ANET": "XLK", "VRT": "XLK", "MSFT": "XLK",
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
    "HOOD": "XLF", "COIN": "XLF",
}
SECTOR_ETFS = sorted(set(TICKER_ETF.values()))
ETF_NAME = {"SMH": "Semiconductors", "CIBR": "Cybersecurity", "IGV": "Software",
            "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer Disc.",
            "XLF": "Financials"}

# Pattern-score (layer 3) internal weights for the 8 detectors + trend.
# Backtest-rebalanced: PocketPivot demoted (weakest, 59% win) to fund
# Cup-with-Handle (strongest candidate, 87% win / +1.60R in the lab).
WEIGHTS = {
    "vcp": 0.25, "pocket": 0.10, "tight_weekly": 0.10, "flat_base": 0.10,
    "htf": 0.10, "three_weeks": 0.10, "ma_bounce": 0.10, "cup_handle": 0.10,
    "trend": 0.05,
}
PATTERN_FIELDS = ("vcp", "pocket", "tight_weekly", "flat_base", "htf",
                  "three_weeks", "ma_bounce", "cup_handle")
PATTERN_LABELS = {"vcp": "VCP", "pocket": "Pocket Pivot", "tight_weekly": "Tight Weekly",
                  "flat_base": "Flat Base", "htf": "High Tight Flag",
                  "three_weeks": "3-Weeks-Tight", "ma_bounce": "MA Bounce",
                  "cup_handle": "Cup w/ Handle"}

# Final Trade Score weights (Step 6).
FINAL_WEIGHTS = {"market": 0.30, "sector": 0.20, "pattern": 0.20,
                 "rs": 0.15, "liq": 0.10, "earn": 0.05}


def compute_combined(m: "M", mode: str) -> float:
    """Pattern Score (layer 3): blend the 7 detectors + trend into 0-100.

    weighted: fixed-weight additive sum. Patterns are largely mutually
        exclusive so scores compress.
    best:     drive off the single strongest pattern (one clean setup can score
        high), blend in trend, add a small confirmation bonus.
    """
    if mode == "best":
        pats = [getattr(m, f) for f in PATTERN_FIELDS]
        best = max(pats)
        confirms = sum(1 for p in pats if p >= 60) - (1 if best >= 60 else 0)
        bonus = min(max(confirms, 0) * 5, 10)
        return round(min(0.85 * best + 0.15 * m.trend + bonus, 100), 1)
    return round(
        WEIGHTS["vcp"] * m.vcp + WEIGHTS["pocket"] * m.pocket +
        WEIGHTS["tight_weekly"] * m.tight_weekly + WEIGHTS["flat_base"] * m.flat_base +
        WEIGHTS["htf"] * m.htf + WEIGHTS["three_weeks"] * m.three_weeks +
        WEIGHTS["ma_bounce"] * m.ma_bounce + WEIGHTS["trend"] * m.trend, 1)


def apply_universe(refresh: bool = False):
    """Replace the fixed 27-ticker source with the dynamic leadership universe.
    ONLY the ticker source (UNIVERSE + its company/sector/ETF lookups) changes;
    every downstream step is identical. Any failure -> keep the hardcoded 27."""
    global UNIVERSE, BEAR_UNIVERSE, UNIVERSE_META, TICKER_ETF, SECTOR_ETFS, ETF_NAME
    try:
        import universe
        res = universe.load_universe_for_scanner(refresh=refresh)
        if not res or len(res["tickers"]) < 10:
            raise RuntimeError("universe empty/too small")
        UNIVERSE = res["tickers"]
        BEAR_UNIVERSE = res.get("bear_tickers", [])
        UNIVERSE_META = res["meta"]
        TICKER_ETF = res["ticker_etf"]
        SECTOR_ETFS = res["sector_etfs"]
        ETF_NAME = {**ETF_NAME, **res["etf_name"]}
        print(f"  Universe: {len(UNIVERSE)} leadership + {len(BEAR_UNIVERSE)} "
              f"laggard (put-side) tickers (generated {res['generated']})", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        BEAR_UNIVERSE = []
        print(f"  ! Dynamic universe generation failed ({exc}). "
              f"Using fallback ticker list ({len(UNIVERSE)}).", file=sys.stderr)


# --------------------------------------------------------------------------
# Metrics container
# --------------------------------------------------------------------------
@dataclass
class M:
    ticker: str
    company: str
    sector: str
    price: float
    last_date: str
    df: pd.DataFrame
    wk: pd.DataFrame

    ema21: float = 0.0
    sma50: float = 0.0
    sma150: float = 0.0
    sma200: float = 0.0
    sma50_prev: float = 0.0
    avgvol10: float = 0.0
    avgvol20: float = 0.0
    avgvol50: float = 0.0
    atr14: float = 0.0
    dollar_vol: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    pct_from_high: float = 0.0

    rs_spy_1m: float = 0.0
    rs_spy_3m: float = 0.0
    rs_qqq_1m: float = 0.0
    rs_qqq_3m: float = 0.0
    rs_sec_1m: float = 0.0
    rs_sec_3m: float = 0.0

    wk10ma: float = 0.0

    # structure
    pivot: float = 0.0
    base_low: float = 0.0
    last_swing_low: float = 0.0
    dist_to_pivot: float = 0.0
    gap_down_20d: bool = False
    lower_lows: bool = False
    vol_vs_avg: float = 0.0

    # pattern scores (layer 3)
    trend: float = 0.0
    vcp: float = 0.0
    pocket: float = 0.0
    tight_weekly: float = 0.0
    flat_base: float = 0.0
    htf: float = 0.0
    three_weeks: float = 0.0
    ma_bounce: float = 0.0
    cup_handle: float = 0.0
    combined: float = 0.0          # = Pattern Score
    best_pattern: str = ""
    detected_patterns: list[str] = field(default_factory=list)

    # filter-layer scores
    market_regime: float = 0.0     # layer 1 (same for all names)
    sector_etf: str = ""
    sector_score: float = 0.0      # layer 2
    sector_rank: int = 0
    rs_score: float = 0.0          # layer 4
    liq_score: float = -1.0        # layer 5 (-1 = not assessed / no option fetched)
    earn_score: float = 100.0      # layer 6
    final_score: float = 0.0
    classification: str = ""

    # market event-risk filter (set by event_risk.apply_to_metric)
    event_risk_level: str = "LOW"
    event_risk_score: float = 100.0
    event_risk_reason: str = ""
    adjusted_final_score: float = 0.0
    position_size_multiplier: float = 1.0
    event_trade_allowed: bool = True

    # bearish PUT scanner (set by bearish.score_stock + put selection)
    bear_failed_breakout: float = 0.0
    bear_late_base: float = 0.0
    bear_distribution: float = 0.0
    bear_dma_breakdown: float = 0.0
    bear_rel_weakness: float = 0.0
    bear_flag: float = 0.0
    bear_gap_down: float = 0.0
    bearish_pattern_score: float = 0.0
    bear_market_score: float = 0.0
    bear_sector_score: float = 0.0
    bear_liq_score: float = -1.0
    bearish_final: float = 0.0
    bearish_classification: str = ""
    bear_best_pattern: str = ""
    bear_detected: list[str] = field(default_factory=list)
    do_not_chase_put: bool = False
    put_option: dict | None = None
    put_entry: float = 0.0
    put_stop: float = 0.0
    put_target: float = 0.0
    put_rr: float = 0.0
    put_contracts: int = 0
    put_position_note: str = ""
    direction: str = "CALL"        # primary opportunity direction for this name
    bear_only: bool = False        # from the weakness/laggard universe (PUT side only)
    put_exceptional: bool = False  # counter-market stock-specific breakdown path
    bear_warnings: list[str] = field(default_factory=list)
    flow: dict | None = None       # options-flow confirmation (flow.chain_flow)

    # context / flags
    contractions: list[float] = field(default_factory=list)
    liquid: bool = False
    extended: bool = False
    extension_flag: str = ""       # "", "extended", "very extended"
    below_200: bool = False
    earnings_date: str = "unknown"
    earnings_days: int | None = None
    earnings_within_7d: bool = False

    # levels & option
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0          # headline = 3R runner (backtest V6)
    target_trim: float = 0.0     # 2R scale-out level
    rr: float = 0.0
    triggered_weak_vol: bool = False   # breakout fired on < 1.2x avg volume (V3)
    option: dict | None = None
    options_liquidity: str = "n/a"
    contracts: int = 0
    position_note: str = ""
    exit_plan: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    group: str = ""

    # news/reddit sentiment confirmation (top names only)
    news_count: int = 0
    reddit_count: int = 0
    news_sent: float = -1.0
    reddit_sent: float = -1.0
    reddit_mentions: int = -1
    reddit_mentions_prev: int = -1
    reddit_source: str = ""
    sentiment_score: float = -1.0
    sentiment_verdict: str = "n/a"
    sentiment_headlines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    return int(_safe_float(v))


def _ret(close: pd.Series, days: int) -> float:
    if close is None or len(close) <= days:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def _sma(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n:
        return float("nan")
    return float(close.rolling(n).mean().iloc[-1])


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    high, low, close = df["High"], df["Low"], df["Close"]
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def _find_pivots(close: np.ndarray, w: int = 3):
    pivots = []
    n = len(close)
    for i in range(w, n - w):
        win = close[i - w:i + w + 1]
        if close[i] == win.max() and close[i] >= close[i - 1]:
            pivots.append((i, close[i], "H"))
        elif close[i] == win.min() and close[i] <= close[i - 1]:
            pivots.append((i, close[i], "L"))
    cleaned = []
    for p in pivots:
        if cleaned and cleaned[-1][2] == p[2]:
            if p[2] == "H" and p[1] >= cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == "L" and p[1] <= cleaned[-1][1]:
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def _contractions(close: np.ndarray, base_len: int = 100):
    seg = close[-base_len:] if len(close) > base_len else close
    pivots = _find_pivots(seg)
    depths, lows = [], []
    for a, b in zip(pivots, pivots[1:]):
        if a[2] == "H" and b[2] == "L" and a[1] > 0:
            depths.append(round((a[1] - b[1]) / a[1] * 100, 2))
            lows.append(b[1])
    return depths[-4:], lows[-4:]


def _upper_half(row) -> bool:
    rng = row["High"] - row["Low"]
    return rng > 0 and (row["Close"] - row["Low"]) / rng >= 0.5


def _rs_ok(m: M) -> bool:
    return (m.rs_spy_3m > 0 or m.rs_qqq_3m > 0)


# --------------------------------------------------------------------------
# Build metrics (pattern layer)
# --------------------------------------------------------------------------
def build_metrics(ticker: str, df: pd.DataFrame, bench: dict[str, pd.Series],
                  score_mode: str = "weighted") -> M | None:
    if df is None or len(df) < 200:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < 200:
        return None

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return None

    company, sector = UNIVERSE_META.get(ticker, (ticker, "?"))
    wk = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum"}).dropna()
    m = M(ticker=ticker, company=company, sector=sector, price=price,
          last_date=df.index[-1].date().isoformat(), df=df, wk=wk)
    m.sector_etf = TICKER_ETF.get(ticker, "")

    m.ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50s = close.rolling(50).mean()
    m.sma50 = float(sma50s.iloc[-1])
    m.sma50_prev = float(sma50s.iloc[-21]) if len(sma50s) > 21 else m.sma50
    m.sma150 = float(close.rolling(150).mean().iloc[-1])
    m.sma200 = float(close.rolling(200).mean().iloc[-1])
    m.avgvol10 = float(vol.rolling(10).mean().iloc[-1])
    m.avgvol20 = float(vol.rolling(20).mean().iloc[-1])
    m.avgvol50 = float(vol.rolling(50).mean().iloc[-1])
    m.atr14 = _atr(df)
    m.dollar_vol = price * m.avgvol20
    m.high_52w = float(close.tail(252).max())
    m.low_52w = float(close.tail(252).min())
    m.pct_from_high = (price - m.high_52w) / m.high_52w * 100 if m.high_52w else 0.0
    m.vol_vs_avg = (float(vol.iloc[-1]) / m.avgvol20 - 1) * 100 if m.avgvol20 else 0.0

    spy, qqq = bench.get("SPY"), bench.get("QQQ")
    if spy is not None:
        m.rs_spy_1m = _ret(close, 21) - _ret(spy, 21)
        m.rs_spy_3m = _ret(close, 63) - _ret(spy, 63)
    if qqq is not None:
        m.rs_qqq_1m = _ret(close, 21) - _ret(qqq, 21)
        m.rs_qqq_3m = _ret(close, 63) - _ret(qqq, 63)
    etf = bench.get(m.sector_etf)
    if etf is not None:
        m.rs_sec_1m = _ret(close, 21) - _ret(etf, 21)
        m.rs_sec_3m = _ret(close, 63) - _ret(etf, 63)

    if len(wk) >= 10:
        m.wk10ma = float(wk["Close"].rolling(10).mean().iloc[-1])

    depths, lows = _contractions(close.to_numpy())
    m.contractions = depths
    m.last_swing_low = float(lows[-1]) if lows else float(low.tail(20).min())
    m.base_low = float(low.tail(40).min())
    m.pivot = float(high.tail(40).max())
    m.dist_to_pivot = (m.pivot - price) / m.pivot * 100 if m.pivot else 0.0

    o, pc = df["Open"], close.shift(1)
    gap = (o < pc * 0.93) & (close < pc)
    m.gap_down_20d = bool(gap.tail(20).any())
    m.lower_lows = float(low.tail(20).min()) < float(low.iloc[-40:-20].min()) if len(low) > 40 else False

    m.below_200 = price < m.sma200
    m.liquid = m.dollar_vol >= 100_000_000

    # ---- pattern scores ----
    m.trend = trend_quality(m)
    m.vcp = score_vcp(m)
    m.pocket = score_pocket_pivot(m)
    m.tight_weekly = score_tight_weekly(m)
    m.flat_base = score_flat_base(m)
    m.htf = score_high_tight_flag(m)
    m.three_weeks = score_three_weeks_tight(m)
    m.ma_bounce = score_ma_bounce(m)
    m.cup_handle = score_cup_handle(m)
    m.combined = compute_combined(m, score_mode)

    pats = {"VCP": m.vcp, "PocketPivot": m.pocket, "TightWeekly": m.tight_weekly,
            "FlatBase": m.flat_base, "HighTightFlag": m.htf,
            "3WeeksTight": m.three_weeks, "MA-Bounce": m.ma_bounce,
            "CupHandle": m.cup_handle}
    m.best_pattern = max(pats, key=pats.get)
    m.detected_patterns = [PATTERN_LABELS[f] for f in PATTERN_FIELDS if getattr(m, f) >= 60]
    return m


# --------------------------------------------------------------------------
# Trend quality
# --------------------------------------------------------------------------
def trend_quality(m: M) -> float:
    checks = [
        m.price > m.sma50, m.price > m.sma150, m.price > m.sma200,
        m.sma50 > m.sma150, m.sma150 > m.sma200, m.sma50 > m.sma50_prev,
        m.rs_spy_1m > 0, m.rs_spy_3m > 0, m.rs_qqq_1m > 0, m.rs_qqq_3m > 0,
    ]
    score = sum(bool(c) for c in checks) * 10.0
    if m.below_200:
        score *= 0.5
    if m.sma50 < m.sma50_prev:
        score -= 10
    if m.lower_lows:
        score -= 10
    if m.gap_down_20d:
        score -= 10
    return float(np.clip(score, 0, 100))


# --------------------------------------------------------------------------
# Pattern 1: VCP
# --------------------------------------------------------------------------
def score_vcp(m: M) -> float:
    if m.price < m.sma50:
        return 0.0
    s = 0.0
    if m.price > m.sma50 and m.sma50 > m.sma150:
        s += 25
    n = len(m.contractions)
    if 2 <= n <= 4:
        s += 25
    elif n == 1:
        s += 10
    if n >= 2 and all(m.contractions[i] <= m.contractions[i - 1] + 1.0 for i in range(1, n)):
        s += 20
    vol = m.df["Volume"]
    if float(vol.tail(10).mean()) < float(vol.tail(50).mean()) * 0.9:
        s += 15
    if 0 <= m.dist_to_pivot <= 5:
        s += 15
    if m.dist_to_pivot > 15:
        s *= 0.5
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 2: Pocket Pivot
# --------------------------------------------------------------------------
def score_pocket_pivot(m: M) -> float:
    df = m.df
    if len(df) < 12:
        return 0.0
    today, prev = df.iloc[-1], df.iloc[-2]
    prior10 = df.iloc[-11:-1]
    down_days = prior10[prior10["Close"] < prior10["Close"].shift(1)]
    max_down_vol = float(down_days["Volume"].max()) if not down_days.empty else 0.0
    s = 0.0
    if today["Volume"] > max_down_vol and max_down_vol > 0:
        s += 25
    if today["Close"] > prev["Close"] and _upper_half(today):
        s += 20
    near = lambda ma: abs(m.price - ma) / ma * 100 <= 2 if ma else False
    if m.price >= m.ema21 and (near(m.ema21) or near(m.sma50) or m.price >= m.sma50):
        s += 20
    if m.pct_from_high >= -10:
        s += 20
    if _rs_ok(m):
        s += 15
    if m.sma50 < m.sma50_prev:
        s *= 0.4
    if not _upper_half(today):
        s *= 0.6
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 3: Tight Weekly Close
# --------------------------------------------------------------------------
def score_tight_weekly(m: M) -> float:
    wk = m.wk
    if len(wk) < 4:
        return 0.0
    closes = wk["Close"].iloc[-3:]
    rng = (closes.max() - closes.min()) / closes.mean() * 100 if closes.mean() else 99
    s = 0.0
    if rng <= 3:
        s += 35
    elif rng <= 5:
        s += 15
    lows = wk["Low"].iloc[-3:].to_numpy()
    if len(lows) >= 2 and all(lows[i] >= lows[i - 1] * 0.98 for i in range(1, len(lows))):
        s += 20
    if m.wk10ma and m.price > m.wk10ma:
        s += 20
    if float(wk["Volume"].iloc[-3:].mean()) < float(wk["Volume"].tail(12).mean()):
        s += 15
    if m.pct_from_high >= -10:
        s += 10
    if m.below_200:
        s *= 0.4
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 4: Flat Base
# --------------------------------------------------------------------------
def score_flat_base(m: M) -> float:
    df = m.df
    base = df.tail(35)
    bh, bl = float(base["High"].max()), float(base["Low"].min())
    depth = (bh - bl) / bh * 100 if bh else 99
    s = 0.0
    if depth < 15:
        s += 30
    if 25 <= len(base) <= 40:
        s += 20
    if m.price >= m.sma50 * 0.98:
        s += 20
    if float(base["Volume"].tail(10).mean()) < float(base["Volume"].mean()):
        s += 15
    if 0 <= m.dist_to_pivot <= 5:
        s += 15
    if depth > 20 or (m.price < m.sma50 and m.sma50 < m.sma50_prev):
        s *= 0.4
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 5: High Tight Flag
# --------------------------------------------------------------------------
def score_high_tight_flag(m: M) -> float:
    close = m.df["Close"]
    if len(close) < 60:
        return 0.0
    window = close.tail(40)
    runup = (window.max() / window.min() - 1) * 100 if window.min() else 0
    s = 0.0
    if runup >= 75:
        s += 35
    elif runup >= 50:
        s += 15
    peak = float(window.max())
    pullback = (peak - m.price) / peak * 100 if peak else 0
    if 10 <= pullback <= 25:
        s += 25
    recent = m.df.tail(15)
    if float(recent["Volume"].mean()) < float(m.df["Volume"].tail(40).mean()):
        s += 15
    if m.price > m.ema21 or m.price > m.sma50:
        s += 15
    flag_high = float(recent["High"].max())
    dist = (flag_high - m.price) / flag_high * 100 if flag_high else 99
    if 0 <= dist <= 7:
        s += 10
    if pullback > 35:
        s *= 0.3
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 6: Three-Weeks-Tight
# --------------------------------------------------------------------------
def score_three_weeks_tight(m: M) -> float:
    wk = m.wk
    if len(wk) < 4:
        return 0.0
    closes = wk["Close"].iloc[-3:]
    tight = (closes.max() - closes.min()) / closes.mean() * 100 if closes.mean() else 99
    s = 0.0
    if tight <= 1.5:
        s += 40
    elif tight <= 2.5:
        s += 20
    if m.wk10ma and m.price > m.wk10ma:
        s += 25
    if float(wk["Volume"].iloc[-3:].mean()) < float(wk["Volume"].tail(12).mean()):
        s += 15
    if m.pct_from_high >= -10:
        s += 10
    if _rs_ok(m):
        s += 10
    if m.price < m.sma50 or (m.wk10ma and m.price < m.wk10ma):
        s *= 0.4
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 7: 21 EMA / 50 SMA Moving-Average Bounce
# --------------------------------------------------------------------------
def score_ma_bounce(m: M) -> float:
    if m.price < m.sma150:
        return 0.0
    s = 0.0
    if m.price > m.sma50 and m.sma50 > m.sma150:
        s += 25
    d21 = abs(m.price - m.ema21) / m.ema21 * 100 if m.ema21 else 99
    d50 = abs(m.price - m.sma50) / m.sma50 * 100 if m.sma50 else 99
    near = min(d21, d50)
    if near <= 3:
        s += 20
    df = m.df
    if float(df["Volume"].iloc[-4:-1].mean()) < m.avgvol20:
        s += 20
    today, prev = df.iloc[-1], df.iloc[-2]
    if today["Close"] > prev["Close"] and _upper_half(today):
        s += 20
    if _rs_ok(m):
        s += 15
    if m.sma50 < m.sma50_prev:
        s *= 0.4
    if near > 7:
        s *= 0.6
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Pattern 8: Cup with Handle (backtest-validated addition: 87% win / +1.60R)
# --------------------------------------------------------------------------
def score_cup_handle(m: M) -> float:
    """O'Neil classic: 12-35% cup over ~6 months, right side rebuilt to near
    the rim, then a short quiet-volume handle drifting down. Entry = handle high."""
    c = m.df["Close"]
    if len(c) < 130 or m.price < m.sma200:
        return 0.0
    win = c.tail(120)
    rim = float(win.max())
    rim_pos = int(np.argmax(win.to_numpy()))
    after = win.iloc[rim_pos:]
    if len(after) < 25 or rim <= 0:
        return 0.0                       # rim too recent: no time for cup + handle
    trough = float(after.min())
    depth = (rim - trough) / rim * 100
    s = 0.0
    if 12 <= depth <= 35:
        s += 30                          # proper cup depth
    elif 8 <= depth < 12:
        s += 15
    if rim * 0.90 <= m.price < rim:
        s += 20                          # right side rebuilt, below the rim
    handle = c.tail(8)
    hh = float(handle.max())
    drift = (hh - m.price) / hh * 100 if hh else 99
    if 0 < drift <= 10:
        s += 15                          # handle drifting down, not breaking
    if float(m.df["Volume"].tail(8).mean()) < float(m.df["Volume"].tail(40).mean()):
        s += 20                          # quiet handle volume
    if m.price > m.sma50:
        s += 15
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Filter layer 1: Market Regime
# --------------------------------------------------------------------------
@dataclass
class Regime:
    score: float
    label: str
    vix: float
    size_factor: float
    detail: str


def market_regime(bench: dict[str, pd.Series], vix: pd.Series | None) -> Regime:
    spy, qqq, iwm = bench.get("SPY"), bench.get("QQQ"), bench.get("IWM")
    s = 0.0
    parts = []
    if spy is not None and spy.iloc[-1] > _sma(spy, 50):
        s += 20; parts.append("SPY>50SMA")
    if qqq is not None and qqq.iloc[-1] > _sma(qqq, 50):
        s += 20; parts.append("QQQ>50SMA")
    if qqq is not None and _sma(qqq, 20) > _sma(qqq, 50):
        s += 15; parts.append("QQQ 20>50")
    if spy is not None and spy.iloc[-1] > _sma(spy, 200):
        s += 15; parts.append("SPY>200SMA")
    if qqq is not None and qqq.iloc[-1] > _sma(qqq, 200):
        s += 15; parts.append("QQQ>200SMA")
    vix_v = float(vix.iloc[-1]) if vix is not None and len(vix) else float("nan")
    if math.isfinite(vix_v) and vix_v < 20:
        s += 10; parts.append("VIX<20")
    if iwm is not None and iwm.iloc[-1] > _sma(iwm, 50):
        s += 5; parts.append("IWM>50SMA")

    if s >= 80:
        label, size = "Bullish", 1.0
    elif s >= 60:
        label, size = "Neutral/Cautious", 0.5
    elif s >= 40:
        label, size = "Weak", 0.0
    else:
        label, size = "Bearish", 0.0
    return Regime(float(s), label, vix_v, size, ", ".join(parts) or "none")


# --------------------------------------------------------------------------
# Filter layer 2: Sector Strength
# --------------------------------------------------------------------------
def sector_strength(etf: pd.Series, spy: pd.Series, qqq: pd.Series) -> dict:
    s = 0.0
    price = float(etf.iloc[-1])
    if price > _sma(etf, 50):
        s += 20
    if price > _sma(etf, 200):
        s += 20
    r1, r3 = _ret(etf, 21), _ret(etf, 63)
    if spy is not None and r1 > _ret(spy, 21):
        s += 20
    if qqq is not None and r1 > _ret(qqq, 21):
        s += 15
    if spy is not None and r3 > _ret(spy, 63):
        s += 15
    if _sma(etf, 20) > _sma(etf, 50):
        s += 10
    return {"score": float(np.clip(s, 0, 100)), "ret1m": r1, "ret3m": r3}


def build_sector_table(bench: dict[str, pd.Series]) -> dict[str, dict]:
    spy, qqq = bench.get("SPY"), bench.get("QQQ")
    table: dict[str, dict] = {}
    for etf in SECTOR_ETFS:
        ser = bench.get(etf)
        if ser is None or len(ser) < 200:
            continue
        info = sector_strength(ser, spy, qqq)
        info["name"] = ETF_NAME.get(etf, etf)
        table[etf] = info
    for rank, etf in enumerate(sorted(table, key=lambda e: -table[e]["score"]), 1):
        table[etf]["rank"] = rank
    return table


# --------------------------------------------------------------------------
# Filter layer 4: Relative Strength
# --------------------------------------------------------------------------
def relative_strength_score(m: M) -> float:
    s = 0.0
    if m.rs_spy_1m > 0:
        s += 15
    if m.rs_spy_3m > 0:
        s += 15
    if m.rs_qqq_1m > 0:
        s += 15
    if m.rs_qqq_3m > 0:
        s += 15
    if m.rs_sec_1m > 0:
        s += 20
    if m.rs_sec_3m > 0:
        s += 20
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Filter layer 5: Options Liquidity score
# --------------------------------------------------------------------------
def options_liquidity_score(o: dict | None) -> float:
    if not o:
        return 0.0
    s = 10.0  # tradable monthly expiry was found
    if o["oi"] > 500:
        s += 20
    if o["oi"] > 1000:
        s += 10
    if o["volume"] > 100:
        s += 20
    if o["spread_pct"] < 10:
        s += 25
    if 0.60 <= o["delta"] <= 0.70:
        s += 15
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# Filter layer 6: Earnings Risk score
# --------------------------------------------------------------------------
def earnings_risk_score(days: int | None) -> float:
    if days is None:
        return 100.0
    if days <= 1:
        return 0.0
    if days <= 7:
        return 20.0
    if days <= 14:
        return 60.0
    return 100.0


# --------------------------------------------------------------------------
# Extension / do-not-chase flags (Step 8)
# --------------------------------------------------------------------------
def extension_flag(m: M) -> str:
    above_pivot = -m.dist_to_pivot                       # +ve when price above pivot
    above21 = (m.price - m.ema21) / m.ema21 * 100 if m.ema21 else 0
    above50 = (m.price - m.sma50) / m.sma50 * 100 if m.sma50 else 0
    if above50 > 15:
        return "very extended"
    if above_pivot > 5 or above21 > 10:
        return "extended"
    return ""


# --------------------------------------------------------------------------
# Final score + classification (Steps 6, 7, 14)
# --------------------------------------------------------------------------
LIQ_PRIOR = 50.0  # neutral assumption for liquidity that hasn't been assessed yet


def compute_final(m: M):
    """Literal spec blend (weights sum to 1.0, NO renormalization):
       0.30 market + 0.20 sector + 0.20 pattern + 0.15 rs + 0.10 liq + 0.05 earn.
    When no option has been fetched (liq_score < 0) we use a NEUTRAL 50 for the
    liquidity term — not 0 (which would fake-penalize) and not renormalized
    (which would inflate). Fetching the option then nudges Final by at most
    ±5 (=0.10·(real_liq−50)), up if liquid, down if thin."""
    liq = m.liq_score if m.liq_score >= 0 else LIQ_PRIOR
    m.final_score = round(
        FINAL_WEIGHTS["market"] * m.market_regime +
        FINAL_WEIGHTS["sector"] * m.sector_score +
        FINAL_WEIGHTS["pattern"] * m.combined +
        FINAL_WEIGHTS["rs"] * m.rs_score +
        FINAL_WEIGHTS["liq"] * liq +
        FINAL_WEIGHTS["earn"] * m.earn_score, 1)


def classify(m: M, allow_earnings: bool) -> str:
    # Hard rejects (Step 14 safety rules). NOTE: being extended above the 50SMA
    # is "wait for pullback" (watchlist), NOT a hard reject — only chasing the
    # entry (>5% ABOVE the pivot) is a hard reject here.
    chasing = (-m.dist_to_pivot) > TUNE["chase_max_pct"]
    if m.below_200 or chasing or m.market_regime < 40:
        return "REJECT"
    if m.earnings_within_7d and not allow_earnings:
        return "REJECT(earn)"
    if m.final_score < 65:
        return "REJECT"
    near_entry = (0 <= m.dist_to_pivot <= 5) or m.ma_bounce >= 70
    a_plus = (m.final_score >= 85 and m.market_regime >= 70 and m.sector_score >= 70
              and m.rs_score >= 70 and m.liq_score >= 70 and not m.earnings_within_7d
              and not m.extension_flag and near_entry)
    if a_plus:
        cls = "A+"
    # Backtest-validated (V1): extended entries won 63% vs 79% for non-extended.
    # Extended names are watch-only (B) — wait for the pullback/retest.
    elif m.extension_flag:
        cls = "B"
    elif m.final_score >= 75:   # >=85 but missing an A+ gate also lands here
        cls = "A"
    else:
        cls = "B"               # 65-74
    # Backtest-validated (V3): a breakout already triggered on weak volume is not
    # an A-grade entry — cap at B until volume confirms (80%->87% win rate).
    if m.triggered_weak_vol and cls in ("A+", "A"):
        cls = "B"
    return cls


def levels(m: M, atr_mult: float = 1.5):
    if m.best_pattern in ("MA-Bounce", "PocketPivot"):
        entry = float(m.df["High"].iloc[-1])
    else:
        entry = m.pivot
    cands = [entry * 0.92, m.last_swing_low, entry - atr_mult * m.atr14,
             min(m.ema21, m.sma50)]
    cands = [c for c in cands if 0 < c < entry]
    stop = max(cands) if cands else entry - atr_mult * m.atr14
    # Backtest-validated (V4/V11): tightest-of stops get shaken out. Floor the
    # stop at 2*ATR below entry — never tighter (win rate 68%->72-77%).
    if m.atr14 > 0:
        stop = min(stop, entry - 2.0 * m.atr14)
    if stop <= 0:
        stop = entry * 0.92
    risk = entry - stop
    # Backtest-validated (V6): a 3R runner earns +2.04R vs +1.35R for a flat 2R
    # target, consistent across 4/4 windows. Headline target = 3R; scale half at
    # the 2R trim level. (Cutting to 1.5R tested WORST, 0/4 — never do it.)
    target = entry + 3 * risk if risk > 0 else entry + (entry - m.base_low)
    m.target_trim = round(entry + 2 * risk, 2) if risk > 0 else round(entry, 2)
    rr = (target - entry) / risk if risk > 0 else 0.0
    m.entry, m.stop, m.target, m.rr = round(entry, 2), round(stop, 2), round(target, 2), round(rr, 2)
    # Backtest-validated (V3): a breakout that already triggered on weak volume
    # (< 1.2x 20-day avg) wins only ~baseline; volume-confirmed wins 80%->87%.
    # Flag it so classify() caps the grade below A.
    if len(m.df):
        triggered_today = m.entry > 0 and float(m.df["High"].iloc[-1]) >= m.entry
        m.triggered_weak_vol = bool(triggered_today and m.vol_vs_avg < TUNE["vol_confirm_pct"])


# --------------------------------------------------------------------------
# Position sizing (Step 9) + exit plan (Step 10)
# --------------------------------------------------------------------------
def position_size(m: M, account: float, max_risk_pct: float, max_prem_loss: float,
                  size_factor: float):
    if not m.option or account <= 0:
        m.contracts = 0
        m.position_note = ("set --account-size to size positions" if account <= 0
                           else "no option / premium")
        return
    prem = m.option["premium"]
    if prem <= 0:
        m.contracts = 0
        m.position_note = "no premium"
        return
    max_dollar_risk = account * max_risk_pct / 100.0
    per_contract_risk = prem * max_prem_loss * 100.0
    by_risk = math.floor(max_dollar_risk / per_contract_risk) if per_contract_risk > 0 else 0
    by_alloc = math.floor((account * 0.10) / (prem * 100.0))  # 10% max allocation
    contracts = max(min(by_risk, by_alloc), 0)
    if size_factor < 1.0:
        contracts = math.floor(contracts * size_factor)
    m.contracts = int(contracts)
    cap = "risk-capped" if by_risk <= by_alloc else "10%-alloc-capped"
    factor = f", regime x{size_factor:g}" if size_factor < 1.0 else ""
    m.position_note = (f"{m.contracts} ct ({cap}{factor}); "
                       f"risk ${max_dollar_risk:,.0f} @ -{max_prem_loss*100:.0f}% premium")


def build_exit_plan(m: M):
    if not m.option:
        m.exit_plan = []
        return
    prem = m.option["premium"]
    plan = [
        "ENTRY RULE: take the breakout only on volume > 1.2x 20-day avg "
        "(backtested edge: 80%->87% win — weak-volume triggers are capped at B)",
        f"Stop: option premium -30 to -40% (~${prem*0.65:.2f}) / stock below ${m.stop:.2f}",
        f"Scale out half at 2R — stock ${m.target_trim:.2f} (~option +50% ${prem*1.5:.2f}); "
        "move remaining stop to breakeven",
        f"Let the rest run to 3R — stock ${m.target:.2f} (backtest V6: 3R earns "
        "+2.04R vs +1.35R for a flat 2R)",
        f"Exit if stock closes below pivot ${m.pivot:.2f} after breakout",
        f"Consider exit if stock closes below 21 EMA ${m.ema21:.2f}",
        "Exit or roll if < 14 DTE remaining",
    ]
    m.exit_plan = plan


# --------------------------------------------------------------------------
# Alerts (Step 12)
# --------------------------------------------------------------------------
def build_alerts(m: M):
    a = []
    if 0 <= m.dist_to_pivot <= 2:
        a.append("Within 2% below pivot")
    if m.dist_to_pivot < 0 and m.extension_flag == "":
        a.append("Broke above pivot")
    # Backtest-validated (V3): breakouts on >1.2x avg volume win 73% vs 68%;
    # weak-volume triggers should be skipped.
    # df-dependent alerts — skipped on the on-demand option path (empty df).
    if len(m.df):
        if m.entry > 0 and float(m.df["High"].iloc[-1]) >= m.entry:
            if m.vol_vs_avg >= 20:
                a.append(f"Entry triggered TODAY with volume confirmation (+{m.vol_vs_avg:.0f}% vs avg)")
            else:
                a.append("Entry triggered today on WEAK volume — skip unless volume exceeds 1.2x avg")
        # Jun-10 forensic signature: that day's big losers were EXTENDED leaders
        # (avg +7.7% over 50DMA) already reversing (5d -3.4%) with weak closes.
        # Defensive only — protects longs; NOT a short signal (shorting it tested
        # negative across 97k name-days).
        vs50 = (m.price / m.sma50 - 1) * 100 if m.sma50 else 0.0
        ret5 = _ret(m.df["Close"], 5)
        weak_close = float(m.df["Close"].iloc[-1]) < float(m.df["Open"].iloc[-1])
        if vs50 > 5 and ret5 <= -3 and weak_close:
            a.append(f"DE-RISKING PROFILE: extended leader rolling over (5d {ret5:+.1f}%, "
                     f"weak close) — tighten stops on longs, do not add")
    if m.vol_vs_avg >= 40:
        a.append(f"Volume {m.vol_vs_avg:.0f}% above 20-day avg")
    if m.pocket >= 70:
        a.append("Pocket pivot triggered")
    if m.ma_bounce >= 70:
        a.append("Bounce from 21 EMA / 50 SMA")
    if m.final_score >= 85:
        a.append("Final score above 85")
    if m.extension_flag:
        a.append(f"{m.extension_flag.title()} — do not chase")
    if m.earnings_within_7d:
        a.append("Earnings within 7 days")
    m.alerts = a


# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
def fetch_histories_yahoo(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                sub = data[t].dropna(how="all")
                if not sub.empty:
                    out[t] = sub
    elif tickers:
        out[tickers[0]] = data.dropna(how="all")
    return out


def _tradier_history_one(ticker: str) -> pd.DataFrame | None:
    """1y daily OHLCV for one symbol from Tradier (^VIX -> 'VIX')."""
    sym = "VIX" if ticker == "^VIX" else ticker
    today = dt.date.today()
    start = (today - dt.timedelta(days=400)).isoformat()
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}

    def _go():
        r = requests.get(f"{TRADIER_BASE}/markets/history",
                         params={"symbol": sym, "interval": "daily",
                                 "start": start, "end": today.isoformat()},
                         headers=hdr, timeout=15)
        if not r.ok:
            return None
        days = (r.json().get("history") or {}).get("day")
        if days is None:
            return None
        return [days] if isinstance(days, dict) else days

    days = _retry(_go)
    if not days:
        return None
    df = pd.DataFrame(days)
    if "date" not in df or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns={"open": "Open", "high": "High",
                                              "low": "Low", "close": "Close", "volume": "Volume"})
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in df:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")


def fetch_histories_tradier(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, df in ex.map(lambda x: (x, _tradier_history_one(x)), tickers):
            if df is not None and not df.empty:
                out[t] = df
    return out


def fetch_histories(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """OHLCV dispatcher. Tradier (reliable, no Yahoo throttling) when a token is
    set; Yahoo otherwise. Any symbols Tradier misses fall back to Yahoo."""
    src = OHLCV_SOURCE
    if src == "auto":
        src = "tradier" if TRADIER_TOKEN else "yahoo"
    print(f"  Downloading FRESH 1y daily data for {len(tickers)} symbols ({src}) ...", file=sys.stderr)
    if src == "tradier" and TRADIER_TOKEN:
        out = fetch_histories_tradier(tickers)
        missing = [t for t in tickers if t not in out]
        if missing:
            print(f"    {len(missing)} not on Tradier -> Yahoo: {','.join(missing)}", file=sys.stderr)
            out.update(fetch_histories_yahoo(missing))
        return out
    return fetch_histories_yahoo(tickers)


def fetch_earnings(ticker: str) -> tuple[str, int | None]:
    """Return (next-earnings-date-iso or 'unknown', days_until or None)."""
    today = dt.date.today()
    try:
        tk = yf.Ticker(ticker)
        nxt = None
        try:
            cal = tk.get_earnings_dates(limit=12)
            if cal is not None and not cal.empty:
                fut = [i.date() for i in cal.index if i.date() >= today]
                if fut:
                    nxt = min(fut)
        except Exception:  # noqa: BLE001
            pass
        if nxt is None:
            cal = getattr(tk, "calendar", None)
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, (list, tuple)) and ed:
                    ed = ed[0]
                if isinstance(ed, dt.datetime):
                    nxt = ed.date()
                elif isinstance(ed, dt.date):
                    nxt = ed
        if nxt:
            return nxt.isoformat(), (nxt - today).days
    except Exception:  # noqa: BLE001
        pass
    return "unknown", None


def fetch_all_earnings(tickers: list[str]) -> dict[str, tuple[str, int | None]]:
    """Earnings dates with a day-long file cache. Dates don't change intraday,
    so the first scan of the day fetches (Yahoo), and every later scan that day
    reuses the cache — so frequent re-scans don't hammer/throttle Yahoo."""
    import json
    today = dt.date.today()
    today_iso = today.isoformat()
    cache = {}
    try:
        with open(EARN_CACHE_PATH) as fh:
            cache = json.load(fh)
    except Exception:  # noqa: BLE001
        cache = {}
    out: dict[str, tuple[str, int | None]] = {}
    to_fetch = []
    for t in tickers:
        c = cache.get(t)
        if c and c.get("fetched") == today_iso and "date" in c:
            ed = c["date"]
            days = ((dt.date.fromisoformat(ed) - today).days if ed and ed != "unknown" else None)
            out[t] = (ed, days)
        else:
            to_fetch.append(t)
    if to_fetch:
        print(f"  Fetching earnings for {len(to_fetch)} names ({len(tickers)-len(to_fetch)} cached) ...",
              file=sys.stderr)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for t, (ed, days) in ex.map(lambda x: (x, fetch_earnings(x)), to_fetch):
                # don't let a failed (blocked) fetch wipe a previously-known date
                if ed == "unknown":
                    prev = cache.get(t, {}).get("date")
                    if prev and prev != "unknown":
                        ed = prev
                        days = ((dt.date.fromisoformat(ed) - today).days
                                if ed != "unknown" else None)
                out[t] = (ed, days)
                cache[t] = {"date": ed, "fetched": today_iso}
        try:
            os.makedirs(os.path.dirname(EARN_CACHE_PATH), exist_ok=True)
            with open(EARN_CACHE_PATH, "w") as fh:
                json.dump(cache, fh)
        except Exception:  # noqa: BLE001
            pass
    else:
        print(f"  Earnings: all {len(tickers)} from today's cache.", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call_delta(S, K, T, sigma, r=RISK_FREE):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_put_delta(S, K, T, sigma, r=RISK_FREE):
    cd = bs_call_delta(S, K, T, sigma, r)
    return None if cd is None else cd - 1.0     # put delta in [-1, 0]


def _retry(fn, tries: int = 4, base: float = 0.8):
    """Call fn() with backoff; Yahoo rate-limits rapid option requests."""
    for i in range(tries):
        try:
            out = fn()
            if out is not None:
                return out
        except Exception:  # noqa: BLE001
            pass
        time.sleep(base * (i + 1))
    return None


def _pick_call(rows: list[dict], price: float, expiry: str, dte: int):
    """Common Δ~0.65 selection + enrichment shared by both providers."""
    if not rows:
        return None
    band = [r for r in rows if 0.60 <= r["delta"] <= 0.70]
    pick = min(band or rows, key=lambda r: abs(r["delta"] - 0.65))
    if abs(pick["delta"] - 0.65) > 0.25 or pick["premium"] <= 0:
        return None
    pick["expiry"], pick["dte"] = expiry, dte
    pick["breakeven"] = round(pick["strike"] + pick["premium"], 2)
    pick["breakeven_pct"] = round((pick["breakeven"] / price - 1) * 100, 1)
    pick["prem_stop"] = round(pick["premium"] * 0.65, 2)
    pick["prem_target"] = round(pick["premium"] * 1.75, 2)
    sp = pick["spread_pct"]
    good = sp <= 10 and pick["oi"] >= 500 and pick["volume"] >= 100
    ok = (sp <= 15 or sp >= 900) and pick["oi"] >= 100   # sp>=900 = spread unknown (closed)
    pick["liquidity"] = "good" if good else ("ok" if ok else "thin")
    return pick


def select_call_tradier(ticker: str, price: float, min_dte: int, max_dte: int,
                        target_dte: int = 40):
    """Tradier options chain (real greeks; works after hours = last-close)."""
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    today = dt.date.today()
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/expirations",
                         params={"symbol": ticker, "includeAllRoots": "true"},
                         headers=hdr, timeout=12)
        if not r.ok:
            return None
        exp = (r.json().get("expirations") or {}).get("date") or []
        if isinstance(exp, str):
            exp = [exp]
    except Exception:  # noqa: BLE001
        return None
    cands = []
    for e in exp:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except (ValueError, TypeError):
            continue
        if min_dte <= d <= max_dte:
            cands.append((abs(d - target_dte), d, e))
    if not cands:
        return None
    cands.sort()
    _, dte, expiry = cands[0]
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/chains",
                         params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
                         headers=hdr, timeout=15)
        if not r.ok:
            return None
        opts = (r.json().get("options") or {}).get("option") or []
    except Exception:  # noqa: BLE001
        return None
    rows = []
    for o in opts:
        if o.get("option_type") != "call":
            continue
        K = _safe_float(o.get("strike"))
        g = o.get("greeks") or {}
        delta = _safe_float(g.get("delta"))
        iv = _safe_float(g.get("mid_iv") or g.get("smv_vol"))
        if K <= 0 or delta <= 0:
            continue
        bid, ask = _safe_float(o.get("bid")), _safe_float(o.get("ask"))
        last = _safe_float(o.get("last"))
        close = _safe_float(o.get("close") or o.get("prevclose"))
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else (last if last > 0 else close)
        spread_pct = (ask - bid) / mid * 100 if (mid > 0 and bid > 0 and ask > 0) else 999
        rows.append({"strike": K, "delta": round(delta, 3), "iv": round(iv * 100, 1),
                     "bid": bid, "ask": ask, "premium": round(mid if mid > 0 else last, 2),
                     "spread_pct": round(spread_pct, 1),
                     "oi": _safe_int(o.get("open_interest")),
                     "volume": _safe_int(o.get("volume"))})
    return _pick_call(rows, price, expiry, dte)


def select_call(ticker: str, price: float, min_dte: int, max_dte: int, target_dte: int = 40):
    """Dispatch to the configured options source (auto -> Tradier if a token is
    set, else Yahoo; auto also falls back to Yahoo if Tradier returns nothing)."""
    src = OPTIONS_SOURCE
    if src == "auto":
        src = "tradier" if TRADIER_TOKEN else "yahoo"
    if src == "tradier" and TRADIER_TOKEN:
        o = select_call_tradier(ticker, price, min_dte, max_dte, target_dte)
        if o is not None or OPTIONS_SOURCE == "tradier":
            return o
        # auto mode: Tradier gave nothing -> try Yahoo
    return select_call_yahoo(ticker, price, min_dte, max_dte, target_dte)


def select_call_yahoo(ticker: str, price: float, min_dte: int, max_dte: int, target_dte: int = 40):
    tk = yf.Ticker(ticker)
    expiries = _retry(lambda: tk.options or None)
    if not expiries:
        return None
    today = dt.date.today()
    cands = []
    for e in expiries:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except ValueError:
            continue
        if min_dte <= d <= max_dte:
            cands.append((abs(d - target_dte), d, e))
    if not cands:
        return None
    cands.sort()
    _, dte, expiry = cands[0]
    chain = _retry(lambda: tk.option_chain(expiry))
    calls = chain.calls if chain is not None else None
    if calls is None or calls.empty:
        return None

    T = dte / 365.0
    rows = []
    for _, r in calls.iterrows():
        K = _safe_float(r.get("strike"))
        iv = _safe_float(r.get("impliedVolatility"))
        delta = bs_call_delta(price, K, T, iv)
        if delta is None:
            continue
        bid, ask = _safe_float(r.get("bid")), _safe_float(r.get("ask"))
        last = _safe_float(r.get("lastPrice"))
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        spread_pct = (ask - bid) / mid * 100 if mid > 0 and ask > 0 else 999
        rows.append({"strike": K, "delta": round(delta, 3), "iv": round(iv * 100, 1),
                     "bid": bid, "ask": ask, "premium": round(mid if mid > 0 else last, 2),
                     "spread_pct": round(spread_pct, 1),
                     "oi": _safe_int(r.get("openInterest")), "volume": _safe_int(r.get("volume"))})
    if not rows:
        return None
    band = [r for r in rows if 0.60 <= r["delta"] <= 0.70]
    pick = min(band or rows, key=lambda r: abs(r["delta"] - 0.65))
    # No reasonable ATM/slightly-ITM strike (e.g. only deep-ITM Δ~1.0 rows with
    # zero IV/OI) -> treat the chain as unusable.
    if abs(pick["delta"] - 0.65) > 0.25 or pick["premium"] <= 0:
        return None
    pick["expiry"], pick["dte"] = expiry, dte
    if pick["premium"] > 0:
        pick["breakeven"] = round(pick["strike"] + pick["premium"], 2)
        pick["breakeven_pct"] = round((pick["breakeven"] / price - 1) * 100, 1)
        pick["prem_stop"] = round(pick["premium"] * 0.65, 2)
        pick["prem_target"] = round(pick["premium"] * 1.75, 2)
    else:
        pick["breakeven"] = pick["breakeven_pct"] = pick["prem_stop"] = pick["prem_target"] = None
    good = pick["spread_pct"] <= 10 and pick["oi"] >= 500 and pick["volume"] >= 100
    ok = pick["spread_pct"] <= 15 and pick["oi"] >= 100
    pick["liquidity"] = "good" if good else ("ok" if ok else "thin")
    return pick


# --------------------------------------------------------------------------
# Put-option selection (bearish) — mirrors the call logic, target |Δ|~0.60
# --------------------------------------------------------------------------
def _pick_put(rows: list[dict], price: float, expiry: str, dte: int):
    if not rows:
        return None
    band = [r for r in rows if 0.50 <= abs(r["delta"]) <= 0.70]
    pick = min(band or rows, key=lambda r: abs(abs(r["delta"]) - 0.60))
    if abs(abs(pick["delta"]) - 0.60) > 0.25 or pick["premium"] <= 0:
        return None
    pick["expiry"], pick["dte"] = expiry, dte
    pick["breakeven"] = round(pick["strike"] - pick["premium"], 2)   # put: strike - premium
    pick["breakeven_pct"] = round((pick["breakeven"] / price - 1) * 100, 1)
    pick["prem_stop"] = round(pick["premium"] * 0.65, 2)
    pick["prem_target"] = round(pick["premium"] * 1.75, 2)
    sp = pick["spread_pct"]
    good = sp <= 10 and pick["oi"] >= 500 and pick["volume"] >= 100
    ok = (sp <= 15 or sp >= 900) and pick["oi"] >= 100
    pick["liquidity"] = "good" if good else ("ok" if ok else "thin")
    return pick


def select_put_tradier(ticker: str, price: float, min_dte: int, max_dte: int, target_dte: int = 40):
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    today = dt.date.today()
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/expirations",
                         params={"symbol": ticker, "includeAllRoots": "true"}, headers=hdr, timeout=12)
        if not r.ok:
            return None
        exp = (r.json().get("expirations") or {}).get("date") or []
        if isinstance(exp, str):
            exp = [exp]
    except Exception:  # noqa: BLE001
        return None
    cands = []
    for e in exp:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except (ValueError, TypeError):
            continue
        if min_dte <= d <= max_dte:
            cands.append((abs(d - target_dte), d, e))
    if not cands:
        return None
    cands.sort()
    _, dte, expiry = cands[0]
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/chains",
                         params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
                         headers=hdr, timeout=15)
        opts = (r.json().get("options") or {}).get("option") or []
    except Exception:  # noqa: BLE001
        return None
    rows = []
    for o in opts:
        if o.get("option_type") != "put":
            continue
        K = _safe_float(o.get("strike"))
        g = o.get("greeks") or {}
        delta = _safe_float(g.get("delta"))
        iv = _safe_float(g.get("mid_iv") or g.get("smv_vol"))
        if K <= 0 or delta == 0:
            continue
        bid, ask = _safe_float(o.get("bid")), _safe_float(o.get("ask"))
        last = _safe_float(o.get("last"))
        close = _safe_float(o.get("close") or o.get("prevclose"))
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else (last if last > 0 else close)
        spread_pct = (ask - bid) / mid * 100 if (mid > 0 and bid > 0 and ask > 0) else 999
        rows.append({"strike": K, "delta": round(delta, 3), "iv": round(iv * 100, 1),
                     "bid": bid, "ask": ask, "premium": round(mid if mid > 0 else last, 2),
                     "spread_pct": round(spread_pct, 1),
                     "oi": _safe_int(o.get("open_interest")), "volume": _safe_int(o.get("volume"))})
    return _pick_put(rows, price, expiry, dte)


def select_put_yahoo(ticker: str, price: float, min_dte: int, max_dte: int, target_dte: int = 40):
    tk = yf.Ticker(ticker)
    expiries = _retry(lambda: tk.options or None)
    if not expiries:
        return None
    today = dt.date.today()
    cands = []
    for e in expiries:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except ValueError:
            continue
        if min_dte <= d <= max_dte:
            cands.append((abs(d - target_dte), d, e))
    if not cands:
        return None
    cands.sort()
    _, dte, expiry = cands[0]
    chain = _retry(lambda: tk.option_chain(expiry))
    puts = chain.puts if chain is not None else None
    if puts is None or puts.empty:
        return None
    T = dte / 365.0
    rows = []
    for _, r in puts.iterrows():
        K = _safe_float(r.get("strike"))
        iv = _safe_float(r.get("impliedVolatility"))
        delta = bs_put_delta(price, K, T, iv)
        if delta is None:
            continue
        bid, ask = _safe_float(r.get("bid")), _safe_float(r.get("ask"))
        last = _safe_float(r.get("lastPrice"))
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
        spread_pct = (ask - bid) / mid * 100 if mid > 0 and ask > 0 else 999
        rows.append({"strike": K, "delta": round(delta, 3), "iv": round(iv * 100, 1),
                     "bid": bid, "ask": ask, "premium": round(mid if mid > 0 else last, 2),
                     "spread_pct": round(spread_pct, 1),
                     "oi": _safe_int(r.get("openInterest")), "volume": _safe_int(r.get("volume"))})
    return _pick_put(rows, price, expiry, dte)


def select_put(ticker: str, price: float, min_dte: int, max_dte: int, target_dte: int = 40):
    src = OPTIONS_SOURCE
    if src == "auto":
        src = "tradier" if TRADIER_TOKEN else "yahoo"
    if src == "tradier" and TRADIER_TOKEN:
        o = select_put_tradier(ticker, price, min_dte, max_dte, target_dte)
        if o is not None or OPTIONS_SOURCE == "tradier":
            return o
    return select_put_yahoo(ticker, price, min_dte, max_dte, target_dte)


# --------------------------------------------------------------------------
# News + Reddit sentiment confirmation (top names)
# --------------------------------------------------------------------------
BULL_WORDS = {
    "beat", "beats", "beating", "surge", "surged", "surges", "rally", "rallies",
    "rallied", "breakout", "breaks out", "upgrade", "upgraded", "upgrades", "buy",
    "bullish", "soar", "soars", "soared", "record", "strong", "growth", "gains",
    "gain", "gained", "outperform", "jumps", "jump", "jumped", "tops", "raises",
    "raised", "momentum", "accelerate", "accelerating", "wins", "win", "winning",
    "partnership", "expands", "expansion", "optimistic", "upside", "rebound",
    "rebounds", "all-time high", "highs", "leader", "demand", "profit", "boom",
    "moon", "calls", "long", "loading", "undervalued", "rocket",
}
BEAR_WORDS = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "drop", "drops",
    "dropped", "fall", "falls", "fell", "downgrade", "downgraded", "downgrades",
    "sell", "selling", "bearish", "crash", "crashes", "slump", "slumps", "weak",
    "weakness", "cut", "cuts", "lawsuit", "probe", "investigation", "warning",
    "warns", "warned", "loss", "losses", "decline", "declines", "declined",
    "fears", "selloff", "sell-off", "layoffs", "recall", "fraud", "puts", "short",
    "shorting", "dump", "dumps", "dumping", "overvalued", "bagholders", "bubble",
    "tumble", "tumbles", "sinks", "sink", "disappointing", "halted", "delisting",
}


def _lexicon_sentiment(texts: list[str]) -> tuple[float, int]:
    bull = bear = 0
    used = 0
    for t in texts:
        if not t:
            continue
        low = t.lower()
        b = sum(1 for w in BULL_WORDS if w in low)
        s = sum(1 for w in BEAR_WORDS if w in low)
        if b or s:
            used += 1
        bull += b
        bear += s
    if bull + bear == 0:
        return 50.0, used
    return round(100 * bull / (bull + bear), 1), used


def _google_news_titles(query: str, limit: int = 20) -> list[str]:
    titles: list[str] = []
    try:
        import xml.etree.ElementTree as ET
        q = requests.utils.quote(query)
        r = requests.get(
            f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if not r.ok:
            return titles
        for item in ET.fromstring(r.content).iter("item"):
            t = item.findtext("title")
            if t:
                titles.append(t)
            if len(titles) >= limit:
                break
    except Exception:  # noqa: BLE001
        pass
    return titles


def fetch_news_titles(ticker: str, company: str = "", limit: int = 25) -> list[str]:
    titles: list[str] = []
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        news = []
    for n in news[:15]:
        t = n.get("title")
        if not t and isinstance(n.get("content"), dict):
            c = n["content"]
            t = c.get("title")
            summ = c.get("summary") or c.get("description")
            if t and summ:
                t = f"{t} {summ}"
        if t:
            titles.append(t)
    titles += _google_news_titles(f"{ticker} stock")
    seen, uniq = set(), []
    for t in titles:
        k = t.lower()[:60]
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:limit]


_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120 Safari/537.36")
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


def _parse_atom(content: bytes, max_age_days: int = 14) -> list[str]:
    import re
    import xml.etree.ElementTree as ET
    out: list[str] = []
    try:
        root = ET.fromstring(content)
    except Exception:  # noqa: BLE001
        return out
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
    for e in root.findall("a:entry", _ATOM_NS):
        upd = e.findtext("a:updated", default="", namespaces=_ATOM_NS) or ""
        try:
            when = dt.datetime.fromisoformat(upd.replace("Z", "+00:00"))
            if when < cutoff:
                continue
        except ValueError:
            pass
        title = e.findtext("a:title", default="", namespaces=_ATOM_NS) or ""
        html = e.findtext("a:content", default="", namespaces=_ATOM_NS) or ""
        body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))[:400]
        txt = f"{title} {body}".strip()
        if txt:
            out.append(txt)
    return out


def _reddit_rss(ticker: str, sub: str | None, limit: int = 50) -> list[str]:
    base = (f"https://www.reddit.com/r/{sub}/search.rss" if sub
            else "https://www.reddit.com/search.rss")
    params = {"q": ticker, "sort": "new", "limit": limit, "t": "month"}
    if sub:
        params["restrict_sr"] = 1
    try:
        r = requests.get(base, params=params, headers={"User-Agent": _BROWSER_UA}, timeout=10)
        if r.ok:
            return _parse_atom(r.content)
    except Exception:  # noqa: BLE001
        pass
    return []


def fetch_apewisdom(pages: int = 3) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in range(1, pages + 1):
        try:
            r = requests.get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}",
                             headers={"User-Agent": _BROWSER_UA}, timeout=10)
            if not r.ok:
                break
            for row in r.json().get("results", []):
                out[row["ticker"]] = {
                    "mentions": _safe_int(row.get("mentions")),
                    "mentions_prev": _safe_int(row.get("mentions_24h_ago")),
                    "upvotes": _safe_int(row.get("upvotes")),
                }
        except Exception:  # noqa: BLE001
            break
    return out


def fetch_reddit_titles(ticker: str, limit: int = 50) -> tuple[list[str], str]:
    texts, seen = [], set()
    for sub in (None, "wallstreetbets", "stocks"):
        for t in _reddit_rss(ticker, sub):
            k = t.lower()[:80]
            if k not in seen:
                seen.add(k)
                texts.append(t)
    if texts:
        return texts[:limit], "rss"
    try:
        r = requests.get("https://api.pullpush.io/reddit/search/submission/",
                         params={"q": ticker, "size": limit, "sort": "desc",
                                 "sort_type": "created_utc"},
                         headers={"User-Agent": _BROWSER_UA}, timeout=8)
        if r.ok:
            for d in r.json().get("data", []):
                txt = f"{d.get('title','')} {(d.get('selftext','') or '')[:300]}".strip()
                if txt:
                    texts.append(txt)
            if texts:
                return texts, "pullpush"
    except Exception:  # noqa: BLE001
        pass
    return texts, ""


def analyze_sentiment(m: M):
    news = fetch_news_titles(m.ticker, m.company)
    reddit, m.reddit_source = fetch_reddit_titles(m.ticker)
    m.news_count = len(news)
    m.reddit_count = len(reddit)
    m.news_sent = _lexicon_sentiment(news)[0] if news else -1.0
    m.reddit_sent = _lexicon_sentiment(reddit)[0] if reddit else -1.0

    parts, weights = [], []
    if news:
        parts.append(m.news_sent); weights.append(0.5)
    if reddit:
        parts.append(m.reddit_sent); weights.append(0.5)
    m.sentiment_score = (round(sum(p * w for p, w in zip(parts, weights)) / sum(weights), 1)
                         if parts else -1.0)
    m.sentiment_headlines = news[:3]
    m.sentiment_verdict = _confirm_verdict(m)


def _confirm_verdict(m: M) -> str:
    if m.sentiment_score < 0:
        return "no data"
    tech_strong = m.trend >= 60 and m.combined >= 60
    s = m.sentiment_score
    if s >= 60 and tech_strong:
        return "CONFIRMS"
    if s >= 60 and not tech_strong:
        return "sentiment ahead of price"
    if 40 <= s < 60:
        return "NEUTRAL"
    if s < 40 and tech_strong:
        return "DIVERGES (price strong, chatter negative)"
    return "negative chatter"


# --------------------------------------------------------------------------
# Warnings + groups
# --------------------------------------------------------------------------
def build_warnings(m: M):
    w = []
    if m.extension_flag:
        w.append(f"{m.extension_flag.title()}: wait for pullback or new base — do not chase.")
    if m.below_200:
        w.append("Below 200 SMA — not a long-call candidate.")
    if m.market_regime < 40:
        w.append(f"Market regime bearish ({m.market_regime:.0f}/100) — no new bullish calls.")
    elif m.market_regime < 60:
        w.append(f"Weak market ({m.market_regime:.0f}/100) — watchlist only, no auto-trade.")
    if m.sector_score < 50:
        w.append(f"Weak sector ({m.sector_etf} {m.sector_score:.0f}/100) — capped below A+.")
    if m.earnings_within_7d:
        w.append("Earnings within 7 days: IV crush risk — avoid new calls.")
    elif m.earn_score == 60:
        w.append("Earnings in 8-14 days: smaller size / prefer post-earnings.")
    if m.option is None:
        w.append("No option recommendation: chain unavailable.")
    elif m.liq_score < 60:
        w.append(f"Options liquidity weak (score {m.liq_score:.0f}/100) — do not recommend option trade.")
    if m.rs_score < 40:
        w.append(f"Relative strength weak ({m.rs_score:.0f}/100).")
    if m.sentiment_verdict.startswith("DIVERGES") or m.sentiment_verdict == "negative chatter":
        w.append(f"News/Reddit sentiment {m.sentiment_score:.0f}/100 does not confirm the trend.")
    if m.group == "WATCHLIST":
        w.append("Watchlist / monitor only — not an active buy.")
    m.warnings = w


def assign_group(m: M):
    # ACTIVE (actionable BUY) = a recommend-grade setup (A/A+) that is at or near
    # its pivot and NOT extended — i.e. buyable now. This is the actionable subset
    # of the backtested recommendation set (class A/A+, final>=75). Recommend-grade
    # names that are extended or not yet near the pivot are WATCHLIST (wait for the
    # entry); B is watchlist; REJECT is AVOID.
    # (Previously only the near-impossible A+ mapped to ACTIVE — gated by liq>=70,
    # which most names never reach — so everything showed as "watchlist only".)
    if m.classification.startswith("REJECT"):
        m.group = "AVOID"
    elif (m.classification in ("A+", "A") and not m.extension_flag
          and (m.dist_to_pivot <= 5 or m.ma_bounce >= 70)):
        m.group = "ACTIVE"
    else:
        m.group = "WATCHLIST"


# --------------------------------------------------------------------------
# (Removed) Bearish put sizing / warnings / direction.
# The 7-pattern bearish trade engine and put-option recommendations were deleted:
# backtested no edge, and every short approach lost money across regimes
# (backtest_put_engine.py / backtest_defense.py). The downside is now a
# non-tradeable weakness radar (see bearish.py) for hedge/de-risk context only.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def print_header(data_date: str, score_mode: str, regime: Regime):
    now = _now_ct()
    print("\n" + "=" * 140)
    print(f"  MULTI-LAYER BULLISH OPTIONS SCANNER  (regime+sector+pattern+RS+liquidity+earnings -> final score; pattern mode={score_mode})")
    print(f"  Data through (last bar): {data_date}   |   Generated: {now}")
    print("  Data may be delayed (free EOD feed). FRESH each run — no hardcoded prices. "
          "Educational only, NOT financial advice.")
    print("=" * 140)


def print_context(regime: Regime, sectors: dict[str, dict]):
    size = {1.0: "normal position size", 0.5: "reduce size 50%"}.get(
        regime.size_factor, "no new trades")
    vix = f"{regime.vix:.1f}" if math.isfinite(regime.vix) else "n/a"
    print(f"\n  MARKET REGIME: {regime.score:.0f}/100  [{regime.label}]  VIX {vix}  -> {size}")
    print(f"    drivers: {regime.detail}")
    if regime.score < 40:
        print("    HARD RULE: regime bearish — no new bullish call trades today.")
    elif regime.score < 60:
        print("    RULE: weak market — watchlist only, no automatic trades.")
    print("\n  SECTOR STRENGTH (ranked; top 3 preferred):")
    for etf in sorted(sectors, key=lambda e: sectors[e]["rank"]):
        s = sectors[etf]
        star = " *" if s["rank"] <= 3 else "  "
        print(f"   {star}{s['rank']}. {etf:<5}{s['name']:<16} score {s['score']:>3.0f}"
              f"   1m {s['ret1m']:>+6.1f}%   3m {s['ret3m']:>+6.1f}%")


def print_table(rows: list[M]):
    print("\n  RANKED WATCHLIST  (layer scores 0-100; Final = weighted blend)")
    print("-" * 140)
    h = (f"{'Tkr':<6}{'Sector':<13}{'Price':>9}{'Mkt':>5}{'Sec':>5}{'Pat':>5}{'RS':>5}"
         f"{'Liq':>5}{'Ern':>5}{'Final':>7}{'Cls':>7}{'ToPiv':>7}{'OptLiq':>8}  Group")
    print(h)
    print("-" * 140)
    for m in rows:
        liq = f"{m.liq_score:.0f}" if m.liq_score >= 0 else "—"
        print(f"{m.ticker:<6}{m.sector:<13}{m.price:>9.2f}{m.market_regime:>5.0f}"
              f"{m.sector_score:>5.0f}{m.combined:>5.0f}{m.rs_score:>5.0f}{liq:>5}"
              f"{m.earn_score:>5.0f}{m.final_score:>7.1f}{m.classification:>7}"
              f"{m.dist_to_pivot:>+7.1f}{m.options_liquidity:>8}  {m.group}")
    print("-" * 140)


def print_group(title: str, rows: list[M], detail: bool):
    print("\n" + "=" * 140)
    print(f"  {title} — {len(rows)} name(s)")
    print("=" * 140)
    if not rows:
        print("  (none today)")
        return
    for i, m in enumerate(rows, 1):
        pats = ", ".join(m.detected_patterns) or m.best_pattern
        print(f"\n{i}. {m.ticker} — {m.company} ({m.sector} / {m.sector_etf})   ${m.price:.2f}   "
              f"FINAL {m.final_score} [{m.classification}]")
        if detail:
            liqd = f"{m.liq_score:.0f}" if m.liq_score >= 0 else "n/a"
            print(f"     Layers  Mkt {m.market_regime:.0f} | Sector {m.sector_score:.0f} | "
                  f"Pattern {m.combined:.0f} | RS {m.rs_score:.0f} | Liq {liqd} | "
                  f"Earn {m.earn_score:.0f}")
            print(f"     Patterns: {pats}")
            print(f"     Entry ${m.entry:.2f} | Stop ${m.stop:.2f} | Target ${m.target:.2f} "
                  f"| R/R {m.rr:.1f}:1 | dist-to-pivot {m.dist_to_pivot:+.1f}% | earnings {m.earnings_date}")
            o = m.option
            if o:
                be = f"${o['breakeven']:.2f} ({o['breakeven_pct']:+.1f}%)" if o["breakeven"] else "n/a"
                print(f"     CALL  BUY {m.ticker} {o['expiry']} ${o['strike']:.2f}C "
                      f"({o['dte']} DTE, Δ{o['delta']:.2f})  premium ${o['premium']:.2f} "
                      f"(${o['premium']*100:.0f}/ct)")
                print(f"           bid/ask ${o['bid']:.2f}/${o['ask']:.2f} (spread {o['spread_pct']:.0f}%) "
                      f"| breakeven {be} | IV {o['iv']:.0f}% | OI {o['oi']:,} | vol {o['volume']:,} "
                      f"| liq:{o['liquidity']} (score {m.liq_score:.0f})")
                print(f"     Size: {m.position_note}")
                if m.exit_plan:
                    print("     Exit plan:")
                    for step in m.exit_plan:
                        print(f"        - {step}")
            else:
                print("     No option recommendation: chain unavailable.")
            if m.sentiment_score >= 0:
                hl = m.sentiment_headlines[0] if m.sentiment_headlines else ""
                hl = (hl[:90] + "...") if len(hl) > 90 else hl
                ns = f"{m.news_sent:.0f} ({m.news_count})" if m.news_sent >= 0 else "n/a"
                rs = f"{m.reddit_sent:.0f} ({m.reddit_count})" if m.reddit_sent >= 0 else "n/a"
                mv = f" | reddit mentions {m.reddit_mentions}" if m.reddit_mentions >= 0 else ""
                print(f"     Sentiment {m.sentiment_score:.0f}/100 [{m.sentiment_verdict}]  "
                      f"news {ns} | reddit {rs}{mv}")
                if hl:
                    print(f"           latest: {hl}")
            if m.alerts:
                print(f"     Alerts: {' | '.join(m.alerts)}")
        for warn in m.warnings:
            print(f"     ! {warn}")


def print_sentiment(rows: list[M]):
    print("\n" + "=" * 140)
    print("  SENTIMENT CONFIRMATION — top names vs FRESH news + Reddit (past week)")
    print("=" * 140)
    if not rows:
        print("  (no sentiment data — sources unavailable or no qualifying names)")
        return
    print(f"  {'Tkr':<6}{'Final':>6}{'Pat':>5}{'News':>10}{'Reddit':>10}{'RdtMentions':>13}"
          f"{'Sent':>6}  Verdict")
    print("-" * 140)
    for m in rows:
        ns = f"{m.news_sent:.0f}({m.news_count})" if m.news_sent >= 0 else "n/a"
        rs = f"{m.reddit_sent:.0f}({m.reddit_count})" if m.reddit_sent >= 0 else "n/a"
        if m.reddit_mentions >= 0:
            chg = ""
            if m.reddit_mentions_prev > 0:
                chg = f" {'+' if m.reddit_mentions >= m.reddit_mentions_prev else ''}" \
                      f"{m.reddit_mentions - m.reddit_mentions_prev}"
            mv = f"{m.reddit_mentions}{chg}"
        else:
            mv = "-"
        print(f"  {m.ticker:<6}{m.final_score:>6.0f}{m.combined:>5.0f}{ns:>10}{rs:>10}{mv:>13}"
              f"{m.sentiment_score:>6.0f}  {m.sentiment_verdict}")
    print("-" * 140)
    print("  Sent 0-100 (50=neutral), News + Reddit weighted 50/50. CONFIRMS = chatter supports "
          "strong technicals; DIVERGES = price strong but news/Reddit negative.")
    print("  News = yfinance + Google News RSS. Reddit text = Reddit RSS search. "
          "RdtMentions = ApeWisdom 24h volume (+/- vs prior day).")


# --------------------------------------------------------------------------
# JSON export for the dashboard
# --------------------------------------------------------------------------
def metric_to_dict(m: M) -> dict:
    return {
        "ticker": m.ticker, "company": m.company, "sector": m.sector,
        "sector_etf": m.sector_etf, "price": round(m.price, 2), "last_date": m.last_date,
        "group": m.group, "classification": m.classification, "final_score": m.final_score,
        "event_risk_level": m.event_risk_level, "adjusted_final_score": m.adjusted_final_score,
        "position_size_multiplier": m.position_size_multiplier,
        "event_trade_allowed": m.event_trade_allowed,
        "direction": "CALL", "bear_only": m.bear_only, "flow": m.flow,
        "downside": {
            "weakness": m.bearish_final, "band": m.bearish_classification,
            "flags": m.bear_detected, "market": m.bear_market_score,
            "sector": m.bear_sector_score,
        },
        "trend": m.trend, "best_pattern": m.best_pattern,
        "detected_patterns": m.detected_patterns,
        "layers": {"market": m.market_regime, "sector": m.sector_score,
                   "pattern": m.combined, "rs": m.rs_score, "liq": m.liq_score,
                   "earn": m.earn_score},
        "pattern_scores": {"vcp": m.vcp, "pocket": m.pocket, "tight_weekly": m.tight_weekly,
                           "flat_base": m.flat_base, "htf": m.htf,
                           "three_weeks": m.three_weeks, "ma_bounce": m.ma_bounce,
                           "cup_handle": m.cup_handle},
        "pivot": m.pivot, "entry": m.entry, "stop": m.stop, "target": m.target,
        "target_trim": m.target_trim, "triggered_weak_vol": m.triggered_weak_vol, "rr": m.rr,
        "ema21": m.ema21, "below_200": m.below_200, "vol_vs_avg": m.vol_vs_avg,
        "dist_to_pivot": round(m.dist_to_pivot, 2), "extension_flag": m.extension_flag,
        "earnings_date": m.earnings_date, "earnings_days": m.earnings_days,
        "earnings_within_7d": m.earnings_within_7d,
        "option": m.option, "options_liquidity": m.options_liquidity,
        "contracts": m.contracts, "position_note": m.position_note,
        "exit_plan": m.exit_plan,
        "sentiment": {"score": m.sentiment_score, "verdict": m.sentiment_verdict,
                      "news_sent": m.news_sent, "news_count": m.news_count,
                      "reddit_sent": m.reddit_sent, "reddit_count": m.reddit_count,
                      "reddit_mentions": m.reddit_mentions,
                      "headlines": m.sentiment_headlines},
        "alerts": m.alerts, "warnings": m.warnings,
    }


def recompute_with_option(t: dict, option: dict | None, account: float,
                          max_risk_pct: float, max_prem_loss: float,
                          size_factor: float, allow_earnings: bool = False) -> dict:
    """Re-run the option-dependent layers for one ticker after an on-demand
    option fetch, using the SAME functions as a full scan (single source of
    truth). `t` is the exported ticker dict; returns the changed fields."""
    m = M(ticker=t["ticker"], company=t.get("company", ""), sector=t.get("sector", ""),
          price=t.get("price", 0.0), last_date=t.get("last_date", ""),
          df=pd.DataFrame(), wk=pd.DataFrame())
    m.sector_etf = t.get("sector_etf", "")
    L = t.get("layers", {})
    m.market_regime = L.get("market", 0.0); m.sector_score = L.get("sector", 0.0)
    m.combined = L.get("pattern", 0.0); m.rs_score = L.get("rs", 0.0)
    m.earn_score = L.get("earn", 100.0)
    ps = t.get("pattern_scores", {})
    for k in PATTERN_FIELDS:
        setattr(m, k, ps.get(k, 0.0))
    m.trend = t.get("trend", 0.0); m.best_pattern = t.get("best_pattern", "")
    m.detected_patterns = t.get("detected_patterns", [])
    m.pivot = t.get("pivot", 0.0); m.base_low = t.get("stop", 0.0)
    m.entry = t.get("entry", 0.0); m.stop = t.get("stop", 0.0)
    m.target = t.get("target", 0.0); m.rr = t.get("rr", 0.0)
    m.target_trim = t.get("target_trim", 0.0)
    m.triggered_weak_vol = bool(t.get("triggered_weak_vol", False))
    m.dist_to_pivot = t.get("dist_to_pivot", 0.0)
    m.extension_flag = t.get("extension_flag", ""); m.extended = bool(m.extension_flag)
    m.below_200 = t.get("below_200", False); m.ema21 = t.get("ema21", 0.0)
    m.vol_vs_avg = t.get("vol_vs_avg", 0.0)
    m.earnings_date = t.get("earnings_date", "unknown")
    m.earnings_days = t.get("earnings_days")
    m.earnings_within_7d = bool(t.get("earnings_within_7d", False))
    se = t.get("sentiment") or {}
    m.sentiment_score = se.get("score", -1.0); m.sentiment_verdict = se.get("verdict", "n/a")

    m.option = option
    m.options_liquidity = option["liquidity"] if option else "n/a"
    m.liq_score = options_liquidity_score(option) if option else -1.0

    compute_final(m)
    m.classification = classify(m, allow_earnings)
    assign_group(m)
    position_size(m, account, max_risk_pct, max_prem_loss, size_factor)
    build_exit_plan(m)
    build_alerts(m)
    build_warnings(m)

    return {
        "option": m.option, "options_liquidity": m.options_liquidity,
        "layers": {"market": m.market_regime, "sector": m.sector_score,
                   "pattern": m.combined, "rs": m.rs_score, "liq": m.liq_score,
                   "earn": m.earn_score},
        "final_score": m.final_score, "classification": m.classification, "group": m.group,
        "contracts": m.contracts, "position_note": m.position_note,
        "exit_plan": m.exit_plan, "alerts": m.alerts, "warnings": m.warnings,
    }


def write_dashboard_json(path: str, metrics: list[M], regime: Regime,
                         sectors: dict[str, dict], data_date: str, args, event=None):
    import json
    payload = {
        "generated": _now_ct(),
        "data_date": data_date,
        "score_mode": args.score_mode,
        "account_size": args.account_size,
        "options_source": (OPTIONS_SOURCE if OPTIONS_SOURCE != "auto"
                           else ("tradier" if TRADIER_TOKEN else "yahoo")),
        "risk": {"max_risk_pct": args.max_risk_pct, "max_prem_loss": args.max_prem_loss},
        "event_risk": ({"level": event.level, "score": event.score,
                        "multiplier": event.multiplier, "size_mult": event.size_mult,
                        "mega": event.mega, "scope": event.scope,
                        "confirmed": event.confirmed, "reason": event.reason}
                       if event is not None else None),
        "any_option": any(m.option for m in metrics),
        "regime": {"score": regime.score, "label": regime.label, "vix": regime.vix,
                   "size_factor": regime.size_factor, "detail": regime.detail},
        "sectors": [{"etf": e, **sectors[e]} for e in sorted(sectors, key=lambda x: sectors[x]["rank"])],
        "patterns": [{"key": k, "label": PATTERN_LABELS[k]} for k in PATTERN_FIELDS],
        "bear_market_score": (metrics[0].bear_market_score if metrics else 0.0),
        "tickers": [metric_to_dict(m) for m in metrics],
    }
    body = json.dumps(payload, default=lambda o: None, indent=2)
    if path.endswith(".js"):
        body = "window.SCAN_DATA = " + body + ";\n"
    with open(path, "w") as fh:
        fh.write(body)
    print(f"  Saved dashboard data -> {path}", file=sys.stderr)


def analyze_ticker(ticker: str, account: float = 0.0, max_risk_pct: float = 1.5,
                   max_prem_loss: float = 0.35, score_mode: str = "best",
                   event: dict | None = None, want_sentiment: bool = True) -> dict:
    """Run the full multi-layer analysis for ONE arbitrary ticker (used by the
    dashboard lookup). Reuses every existing scoring step; returns the same dict
    shape as a scan row. `event` = the market-level event_risk dict from the last
    scan (so the ad-hoc ticker is judged against the same tape)."""
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"error": "no ticker"}
    # resolve sector ETF + company name
    gics, company, etf = "", ticker, "XLK"
    try:
        import universe
        info = yf.Ticker(ticker).info or {}
        gics = info.get("sector") or ""
        company = info.get("shortName") or info.get("longName") or ticker
        etf = universe._gics_etf(gics) if gics else "XLK"
    except Exception:  # noqa: BLE001
        pass
    syms = list(dict.fromkeys([ticker, "SPY", "QQQ", "IWM", etf]))
    try:
        hist = fetch_histories(syms + [VIX_SYMBOL])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"data fetch failed: {exc}"}
    if ticker not in hist:
        return {"error": f"no price data for {ticker}"}
    bench = {b: hist[b]["Close"].astype(float) for b in ("SPY", "QQQ", "IWM", etf) if b in hist}
    vix = hist[VIX_SYMBOL]["Close"].astype(float) if VIX_SYMBOL in hist else None
    UNIVERSE_META[ticker] = (company, gics or "?")
    TICKER_ETF[ticker] = etf
    try:
        m = build_metrics(ticker, hist[ticker], bench, score_mode)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"analysis failed: {exc}"}
    if m is None:
        return {"error": f"insufficient history for {ticker} (need ~1y daily)"}

    regime = market_regime(bench, vix)
    m.market_regime = regime.score
    etf_s = bench.get(etf)
    m.sector_score = (sector_strength(etf_s, bench.get("SPY"), bench.get("QQQ"))["score"]
                      if etf_s is not None else 50.0)
    m.rs_score = relative_strength_score(m)
    m.extension_flag = extension_flag(m)
    m.extended = bool(m.extension_flag)
    levels(m, 1.5)
    ed, days = fetch_earnings(ticker)
    m.earnings_date, m.earnings_days = ed, days
    m.earnings_within_7d = days is not None and 0 <= days <= 7
    m.earn_score = earnings_risk_score(days)
    o = select_call(ticker, m.price, 30, 45)
    m.option = o
    m.options_liquidity = o["liquidity"] if o else "n/a"
    m.liq_score = options_liquidity_score(o) if o else -1.0
    compute_final(m)
    m.classification = classify(m, False)
    assign_group(m)
    position_size(m, account, max_risk_pct, max_prem_loss, regime.size_factor)
    build_exit_plan(m)
    build_alerts(m)
    if want_sentiment:
        try:
            analyze_sentiment(m)
        except Exception:  # noqa: BLE001
            pass
    build_warnings(m)
    try:
        import event_risk
        if event:
            er = event_risk.EventRisk(
                level=event.get("level", "LOW"), score=event.get("score", 100.0),
                multiplier=event.get("multiplier", 1.0), size_mult=event.get("size_mult", 1.0),
                reason=event.get("reason", ""), mega=event.get("mega", False),
                scope=event.get("scope", "broad"), confirmed=event.get("confirmed", True))
        else:
            er = event_risk.assess(bench, vix, [m], regime.score)
        event_risk.apply_to_metric(m, er)
    except Exception:  # noqa: BLE001
        pass

    try:
        import bearish
        bmkt = bearish.bearish_market_score(bench, vix, [m])
        bsec = bearish.bearish_sector_table(bench, [etf])
        bearish.score_stock(m, bench, bmkt, bsec)   # weakness only; puts not recommended
        m.direction = "CALL"
    except Exception:  # noqa: BLE001
        pass

    out = metric_to_dict(m)
    out["regime"] = {"score": regime.score, "label": regime.label, "vix": regime.vix}
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-layer bullish options scanner (fresh data).")
    ap.add_argument("--min-dte", type=int, default=30)
    ap.add_argument("--max-dte", type=int, default=45)
    ap.add_argument("--allow-earnings", action="store_true",
                    help="keep setups with earnings within 7 days (flagged as earnings risk)")
    ap.add_argument("--atr-mult", type=float, default=1.5, help="ATR multiple for the ATR stop")
    ap.add_argument("--top-options", type=int, default=30,
                    help="how many top-ranked names to pull option chains for")
    ap.add_argument("--score-mode", choices=["weighted", "best"], default="weighted",
                    help="pattern-score method: 'weighted' (additive) or 'best' (strongest pattern)")
    ap.add_argument("--sentiment", action=argparse.BooleanOptionalAction, default=True,
                    help="confirm top names against fresh news + Reddit chatter (default on)")
    ap.add_argument("--sentiment-top", type=int, default=12,
                    help="how many top-ranked names to run sentiment confirmation on")
    ap.add_argument("--account-size", type=float, default=0.0,
                    help="account size in $ for position sizing (0 = skip sizing)")
    ap.add_argument("--max-risk-pct", type=float, default=1.5,
                    help="max %% of account risked per trade (default 1.5)")
    ap.add_argument("--max-prem-loss", type=float, default=0.35,
                    help="max option premium loss fraction used as the stop (default 0.35)")
    ap.add_argument("--max-open-trades", type=int, default=5,
                    help="max active buy candidates surfaced (default 5)")
    ap.add_argument("--max-per-sector", type=int, default=2,
                    help="max active buy candidates per sector (default 2)")
    ap.add_argument("--refresh-universe", action="store_true",
                    help="force-regenerate the dynamic leadership universe now "
                         "(otherwise uses the cached one, refreshed weekly)")
    ap.add_argument("--options-source", choices=["auto", "tradier", "yahoo"], default=None,
                    help="options data provider (default: env OPTIONS_SOURCE or 'auto'; "
                         "Tradier works after-hours and needs TRADIER_TOKEN)")
    ap.add_argument("--ohlcv-source", choices=["auto", "tradier", "yahoo"], default=None,
                    help="daily OHLCV provider (default: env OHLCV_SOURCE or 'auto'; "
                         "Tradier avoids Yahoo throttling on frequent refreshes)")
    ap.add_argument("--csv", default="", help="save full table to CSV")
    ap.add_argument("--json", default="", help="save full structured results for the "
                    "dashboard (JSON; if path ends in .js it is wrapped as window.SCAN_DATA)")
    args = ap.parse_args()

    apply_universe(refresh=args.refresh_universe)

    global OPTIONS_SOURCE, OHLCV_SOURCE
    if args.options_source:
        OPTIONS_SOURCE = args.options_source
    if args.ohlcv_source:
        OHLCV_SOURCE = args.ohlcv_source
    eff_src = OPTIONS_SOURCE if OPTIONS_SOURCE != "auto" else ("tradier" if TRADIER_TOKEN else "yahoo")
    print(f"  Options source: {eff_src}"
          + ("" if TRADIER_TOKEN or eff_src == "yahoo" else " (no TRADIER_TOKEN set)"), file=sys.stderr)

    all_symbols = UNIVERSE + BEAR_UNIVERSE + BENCHMARKS + SECTOR_ETFS + [VIX_SYMBOL]
    histories = fetch_histories(all_symbols)
    bench = {b: histories[b]["Close"].astype(float)
             for b in BENCHMARKS + SECTOR_ETFS if b in histories}
    vix = histories[VIX_SYMBOL]["Close"].astype(float) if VIX_SYMBOL in histories else None

    dates = [df.index[-1].date() for df in histories.values() if not df.empty]
    data_date = max(dates).isoformat() if dates else "unknown"

    # Layer 1: market regime (once). Layer 2: sector strength table (once).
    regime = market_regime(bench, vix)
    sectors = build_sector_table(bench)

    metrics: list[M] = []
    bear_set = set(BEAR_UNIVERSE)
    for t in UNIVERSE + BEAR_UNIVERSE:
        if t not in histories:
            continue
        try:
            m = build_metrics(t, histories[t], bench, args.score_mode)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {t}: {exc}", file=sys.stderr)
            m = None
        if m:
            m.bear_only = t in bear_set
            metrics.append(m)
    if not metrics:
        print("No data analyzed.", file=sys.stderr)
        return 1

    # Earnings (parallel).
    earn_map = fetch_all_earnings([m.ticker for m in metrics])
    for m in metrics:
        ed, days = earn_map.get(m.ticker, ("unknown", None))
        m.earnings_date, m.earnings_days = ed, days
        m.earnings_within_7d = days is not None and 0 <= days <= 7
        m.earn_score = earnings_risk_score(days)

    # Assign filter-layer scores (regime/sector/RS/extension) + levels.
    for m in metrics:
        m.market_regime = regime.score
        sinfo = sectors.get(m.sector_etf)
        if sinfo:
            m.sector_score, m.sector_rank = sinfo["score"], sinfo["rank"]
        m.rs_score = relative_strength_score(m)
        m.extension_flag = extension_flag(m)
        m.extended = bool(m.extension_flag)
        levels(m, args.atr_mult)

    # Preliminary Final score (neutral liquidity prior) so we fetch option chains
    # for the names the MASTER list will actually rank highest — not a separate
    # pattern-only order (which left top recommendations without a contract).
    # Final is re-computed after the fetch with the real liquidity score.
    for m in metrics:
        compute_final(m)
    metrics.sort(key=lambda m: (-m.final_score, abs(m.dist_to_pivot)))
    # Fetch chains for the strongest names so the watchlist shows contracts.
    # Extended names ARE included (for watchlisting) — they still can't become
    # A+ (the extension gate blocks that); the contract is informational.
    opt_targets = [m for m in metrics if not m.below_200 and m.liquid
                   and not m.bear_only][:args.top_options]
    if opt_targets:
        print("  Verifying live option chains for top setups ...", file=sys.stderr)

        def _pull(mm):
            return mm, select_call(mm.ticker, mm.price, args.min_dte, args.max_dte)

        with ThreadPoolExecutor(max_workers=2) as ex:  # low concurrency: avoid Yahoo throttle
            for m, o in ex.map(_pull, opt_targets):
                m.option = o
                m.options_liquidity = o["liquidity"] if o else "n/a"
    for m in metrics:
        m.liq_score = options_liquidity_score(m.option) if m.option else -1.0

    # Final score + classification + sizing + exits + alerts.
    for m in metrics:
        compute_final(m)
        m.classification = classify(m, args.allow_earnings)
        assign_group(m)
        position_size(m, args.account_size, args.max_risk_pct, args.max_prem_loss,
                      regime.size_factor)
        build_exit_plan(m)
        build_alerts(m)

    # Rank by final score.
    metrics.sort(key=lambda m: (-m.final_score, -m.market_regime, -m.combined,
                                abs(m.dist_to_pivot)))

    # Sentiment on top non-rejected names.
    if args.sentiment and args.sentiment_top > 0:
        sent_targets = [m for m in metrics if not m.bear_only
                        and not m.classification.startswith("REJECT")][:args.sentiment_top]
        if sent_targets:
            print(f"  Confirming top {len(sent_targets)} names vs fresh news + Reddit ...",
                  file=sys.stderr)
            ape = fetch_apewisdom()
            for m in sent_targets:
                a = ape.get(m.ticker)
                if a:
                    m.reddit_mentions = a["mentions"]
                    m.reddit_mentions_prev = a["mentions_prev"]
            with ThreadPoolExecutor(max_workers=5) as ex:
                list(ex.map(analyze_sentiment, sent_targets))
    for m in metrics:
        build_warnings(m)

    # Market Event Risk filter — runs after scores, before final recommendation.
    # Downgrades over-eager buys during risk-off / liquidity events; never blocks
    # the scan. Failure is non-fatal (falls back to no event adjustment).
    event = None
    try:
        import event_risk
        event = event_risk.assess(bench, vix, metrics, regime.score)
        for m in metrics:
            event_risk.apply_to_metric(m, event)
        print(f"  Event risk: {event.level} (score {event.score:.0f}, x{event.multiplier:g}, "
              f"size x{event.size_mult:g}{', MEGA' if event.mega else ''})", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Event-risk module skipped ({exc}).", file=sys.stderr)

    # Downside Radar — transparent per-name WEAKNESS ranking (NOT a trade signal).
    # The former 7-pattern bearish trade engine + put-option recommendations were
    # REMOVED: backtested no edge, and every short approach lost money across
    # regimes (see backtest_put_engine.py / backtest_defense.py). The validated
    # bear-market action is CASH, not shorting. This only flags weak names for
    # hedging / de-risking context.
    try:
        import bearish
        bmkt = bearish.bearish_market_score(bench, vix, metrics)
        bsec = bearish.bearish_sector_table(bench, SECTOR_ETFS)
        for m in metrics:
            bearish.score_stock(m, bench, bmkt, bsec)
            m.direction = "CALL"           # puts are not recommended; trade side is long-only
        n_weak = sum(1 for m in metrics if m.bearish_final >= 60)
        print(f"  Downside Radar: market weakness {bmkt:.0f}/100, {n_weak} weak name(s) "
              f"(hedge/de-risk context only — no put trades)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Downside Radar skipped ({exc}).", file=sys.stderr)

    # Laggard-universe names exist only as weakness context — never call candidates.
    for m in metrics:
        if m.bear_only:
            m.group = "AVOID"

    # Options-flow confirmation (informational) for names with a contract.
    try:
        import flow as flow_mod
        flow_targets = [m for m in metrics if m.option][:20]
        if flow_targets:
            print(f"  Options-flow check on {len(flow_targets)} names ...", file=sys.stderr)
            with ThreadPoolExecutor(max_workers=3) as ex:
                for m, f in ex.map(lambda mm: (mm, flow_mod.chain_flow(mm.ticker, mm.price,
                                                                       args.min_dte, args.max_dte)),
                                   flow_targets):
                    m.flow = f
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Flow check skipped ({exc}).", file=sys.stderr)

    watchlist = metrics[:20]
    active_all = [m for m in metrics if m.group == "ACTIVE"]

    # Portfolio caps (Step 9): max N open, max K per sector.
    active, per_sec = [], {}
    for m in active_all:
        if len(active) >= args.max_open_trades:
            break
        if per_sec.get(m.sector_etf, 0) >= args.max_per_sector:
            continue
        per_sec[m.sector_etf] = per_sec.get(m.sector_etf, 0) + 1
        active.append(m)
    overflow = [m for m in active_all if m not in active]
    watch = [m for m in watchlist if m.group == "WATCHLIST"] + overflow
    avoid = [m for m in watchlist if m.group == "AVOID"]

    print_header(data_date, args.score_mode, regime)
    print_context(regime, sectors)
    print_table(watchlist)
    print_group("GROUP 1 — ACTIVE BUY CANDIDATES (A+, all filters aligned)", active, detail=True)
    print_group("GROUP 2 — WATCHLIST / MONITOR (good setup, one+ condition short)", watch, detail=True)
    print_group("GROUP 3 — AVOID / DO NOT CHASE", avoid, detail=False)

    if args.sentiment:
        scored = [m for m in watchlist if m.sentiment_score >= 0]
        print_sentiment(scored)

    any_option = any(m.option for m in metrics)
    print("\n" + "=" * 140)
    print("  Top active focus: " + (", ".join(m.ticker for m in active) or
                                     "none — no setup cleared all layers today"))
    if not active and active_all:
        print("  (A+ names exist but were capped by max-open/per-sector limits — see watchlist.)")
    if not any_option:
        print("  NOTE: no live option quotes returned (markets closed / outside 9:30-16:00 ET). "
              "The liquidity layer scored 0, so no A+/Active calls can be confirmed now.")
        print("        All other layers (regime/sector/pattern/RS/earnings) are valid — re-run "
              "during market hours to verify chains and surface Active buy candidates.")
    print("  CORE RULE: a pattern alone is not a trade. Market + sector + stock + pattern + "
          "liquidity + risk must all align.")
    print("  SAFETY: confirm quotes in your broker. Long calls can expire worthless. Educational only.")
    print("=" * 140 + "\n")

    if args.csv:
        rows = []
        for m in watchlist:
            d = {
                "ticker": m.ticker, "company": m.company, "sector": m.sector,
                "sector_etf": m.sector_etf, "price": m.price,
                "market_regime": m.market_regime, "sector_score": m.sector_score,
                "pattern_score": m.combined, "rs_score": m.rs_score,
                "liq_score": m.liq_score, "earn_score": m.earn_score,
                "final_score": m.final_score, "classification": m.classification,
                "detected_patterns": "; ".join(m.detected_patterns),
                "entry": m.entry, "stop": m.stop, "target": m.target, "rr": m.rr,
                "dist_to_pivot": round(m.dist_to_pivot, 2), "extension": m.extension_flag,
                "earnings": m.earnings_date, "earnings_days": m.earnings_days,
                "options_liquidity": m.options_liquidity, "contracts": m.contracts,
                "group": m.group, "sentiment": m.sentiment_score,
                "sentiment_verdict": m.sentiment_verdict,
                "alerts": " | ".join(m.alerts), "warnings": " | ".join(m.warnings),
            }
            if m.option:
                d.update({"opt_strike": m.option["strike"], "opt_expiry": m.option["expiry"],
                          "opt_delta": m.option["delta"], "opt_premium": m.option["premium"],
                          "opt_breakeven": m.option["breakeven"]})
            rows.append(d)
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"  Saved table -> {args.csv}", file=sys.stderr)

    if args.json:
        write_dashboard_json(args.json, metrics, regime, sectors, data_date, args, event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
