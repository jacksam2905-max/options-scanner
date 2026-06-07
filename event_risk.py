#!/usr/bin/env python3
"""MarketEventRiskEngine — broad market-event / liquidity-risk filter.

Runs AFTER the existing scores and BEFORE the final recommendation. It does not
change pattern detection, the core scoring formula, option selection, or the
output format — it only (a) measures how risk-off the tape is, (b) folds that
into an adjusted_final_score + position-size multiplier, and (c) downgrades
over-eager buy recommendations to watchlist/avoid during event risk.

Two inputs, with graceful fallback:
  1. Optional hand-maintained calendar  market_events.json  (CPI/FOMC/jobs/
     mega-IPO/etc.). If absent, skipped.
  2. Market-behavior PROXY computed from data already fetched (indexes, VIX,
     universe breadth). This always works and is the fallback per spec §13.

Core principle: technical patterns fail during liquidity events, so during
major events the scanner becomes more selective (higher bar, smaller size,
stronger confirmation, more watchlist-only).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass, field

import pandas as pd

PROJECT = os.path.dirname(os.path.abspath(__file__))
EVENTS_PATH = os.path.join(PROJECT, "market_events.json")

# High-beta names get an extra penalty under event risk (spec §9).
HIGH_BETA = {"TSLA", "PLTR", "COIN", "HOOD", "APP", "SMCI", "ARM", "MSTR", "RIVN", "AFRM"}
TECH_ETFS = {"XLK", "SMH", "IGV", "CIBR", "WCLD", "SKYY"}

LEVELS = ["LOW", "MEDIUM", "HIGH", "EXTREME"]
MULT = {"LOW": 1.00, "MEDIUM": 0.90, "HIGH": 0.75, "EXTREME": 0.50}
SIZE = {"LOW": 1.00, "MEDIUM": 0.75, "HIGH": 0.50, "EXTREME": 0.00}
SCORE = {"LOW": 100.0, "MEDIUM": 75.0, "HIGH": 50.0, "EXTREME": 25.0}
REQ_REGIME = {"LOW": 70, "MEDIUM": 80, "HIGH": 80, "EXTREME": 80}


@dataclass
class EventRisk:
    level: str = "LOW"
    score: float = 100.0
    multiplier: float = 1.0
    size_mult: float = 1.0
    reason: str = ""
    mega: bool = False
    scope: str = "broad"        # broad | tech | financials | energy
    confirmed: bool = True      # market confirms bullish trades
    signals: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
def _chg(series: pd.Series) -> float:
    """Latest move: prior close -> last (today's intraday move during RTH)."""
    if series is None or len(series) < 2:
        return 0.0
    try:
        return float(series.iloc[-1] / series.iloc[-2] - 1) * 100
    except Exception:  # noqa: BLE001
        return 0.0


def _sma(series: pd.Series, n: int) -> float:
    if series is None or len(series) < n:
        return float("nan")
    return float(series.rolling(n).mean().iloc[-1])


def _above_50(series: pd.Series) -> bool:
    s = _sma(series, 50)
    return series is not None and len(series) and math.isfinite(s) and float(series.iloc[-1]) >= s * 0.999


def _level_max(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


# --------------------------------------------------------------------------
def load_calendar(path: str = EVENTS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else data.get("events", [])
    except Exception:  # noqa: BLE001
        return []


def _calendar_assessment(events: list[dict]):
    """Return (level, mega, scope, notes) from scheduled events near today."""
    today = dt.date.today()
    level, mega, scope, notes = "LOW", False, None, []
    for e in events:
        try:
            d = (dt.date.fromisoformat(str(e.get("date"))) - today).days
        except (ValueError, TypeError):
            continue
        if d < -1 or d > 5:                       # only near-term events
            continue
        sev = str(e.get("severity", "medium")).upper()
        if sev not in LEVELS:
            sev = "MEDIUM"
        # closer events weigh full; 3-5 days out soften by one notch
        if d > 3 and LEVELS.index(sev) > 0:
            sev = LEVELS[LEVELS.index(sev) - 1]
        level = _level_max(level, sev)
        if e.get("scope"):
            scope = str(e.get("scope"))
        nm = e.get("ipo") or e.get("type") or "event"
        notes.append(f"{nm} in {d}d ({sev.lower()})")
        # mega IPO / liquidity drain (spec §4)
        if (e.get("mega") or float(e.get("valuation_b", 0) or 0) > 100
                or float(e.get("raise_b", 0) or 0) > 10):
            mega = True
            level = _level_max(level, "HIGH")
            notes.append(f"MEGA: {nm}")
    return level, mega, scope, notes


# --------------------------------------------------------------------------
def assess(bench: dict, vix: pd.Series, metrics: list, regime_score: float,
           events_path: str = EVENTS_PATH) -> EventRisk:
    spy, qqq, iwm = bench.get("SPY"), bench.get("QQQ"), bench.get("IWM")
    spy_c, qqq_c, iwm_c, vix_c = _chg(spy), _chg(qqq), _chg(iwm), _chg(vix)

    # universe breadth (from today's daily bar)
    red = below_open = total = 0
    for m in metrics:
        df = getattr(m, "df", None)
        if df is None or len(df) < 2:
            continue
        total += 1
        today, prev = df.iloc[-1], df.iloc[-2]
        if float(today["Close"]) < float(prev["Close"]):
            red += 1
        if float(today["Close"]) < float(today["Open"]):
            below_open += 1
    pct_red = (red / total * 100) if total else 0.0
    pct_below_open = (below_open / total * 100) if total else 0.0

    # tech leadership vs SPY (for scope = tech rotation)
    tech_c = [(_chg(bench.get(e))) for e in ("XLK", "SMH", "IGV") if bench.get(e) is not None]
    tech_avg = sum(tech_c) / len(tech_c) if tech_c else 0.0

    # ---- proxy severity points (spec §2D / §3) ----
    sev = 0
    if qqq_c < -1: sev += 2
    elif qqq_c < -0.5: sev += 1
    if spy_c < -0.75: sev += 2
    elif spy_c < -0.4: sev += 1
    if iwm_c < -1: sev += 1
    if vix_c > 8: sev += 2
    elif vix_c > 5: sev += 1
    if pct_red > 80: sev += 3
    elif pct_red > 60: sev += 2
    elif pct_red > 55: sev += 1
    if pct_below_open > 60: sev += 2

    # EXTREME is reserved for a genuine panic/flush — a real VIX spike or a large
    # gap-down — NOT just a broad-but-orderly red day (that tops out at HIGH).
    panic = (vix_c > 18) or (qqq_c < -3.5) or (spy_c < -3.0) or (vix_c > 12 and qqq_c < -2.5)

    proxy_level = ("EXTREME" if panic else
                   "HIGH" if sev >= 5 else
                   "MEDIUM" if sev >= 2 else "LOW")

    # ---- calendar ----
    events = load_calendar(events_path)
    cal_level, mega, cal_scope, cal_notes = _calendar_assessment(events)

    level = _level_max(proxy_level, cal_level)

    # scope: tech rotation if tech notably weaker than SPY
    scope = cal_scope or ("tech" if (tech_avg < spy_c - 0.4 and tech_avg < 0) else "broad")

    # market confirmation for bullish trades (spec §5)
    req = REQ_REGIME[level]
    confirmed = (regime_score >= req
                 and vix_c < 10
                 and (spy is None or _above_50(spy))
                 and (qqq is None or _above_50(qqq)))

    size = SIZE[level]
    if mega:
        size *= 0.5

    reason_bits = [f"QQQ {qqq_c:+.2f}%", f"SPY {spy_c:+.2f}%", f"IWM {iwm_c:+.2f}%",
                   f"VIX {vix_c:+.1f}%", f"{pct_red:.0f}% red", f"{pct_below_open:.0f}% below open"]
    if cal_notes:
        reason_bits.append("cal: " + "; ".join(cal_notes))
    reason = f"{level} ({scope}) — " + ", ".join(reason_bits)
    if not confirmed:
        reason += " | market NOT confirming"

    return EventRisk(level=level, score=SCORE[level], multiplier=MULT[level], size_mult=size,
                     reason=reason, mega=mega, scope=scope, confirmed=confirmed,
                     signals={"spy": spy_c, "qqq": qqq_c, "iwm": iwm_c, "vix": vix_c,
                              "pct_red": pct_red, "pct_below_open": pct_below_open,
                              "sev": sev, "regime": regime_score})


# --------------------------------------------------------------------------
def apply_to_metric(m, er: EventRisk):
    """Fold event risk into one stock: adjusted score, penalties, trade gate,
    group downgrade, internal fields, and warnings. Leaves final_score and
    classification (the displayed scores) untouched."""
    hb = m.ticker in HIGH_BETA
    adj = m.final_score * er.multiplier
    if er.level in ("HIGH", "EXTREME") and hb:
        adj -= 10                                   # high-beta penalty
    if er.scope == "tech" and m.sector_etf in TECH_ETFS and er.level in ("HIGH", "EXTREME"):
        adj -= 5                                     # sector-specific penalty
    m.adjusted_final_score = round(max(adj, 0.0), 1)
    m.position_size_multiplier = er.size_mult
    m.event_risk_level = er.level
    m.event_risk_score = er.score
    m.event_risk_reason = er.reason

    # ---- trade-allowed gate (spec §8 / §12) ----
    allowed = True
    if er.level == "EXTREME":
        allowed = False
    elif er.level == "HIGH":
        allowed = m.adjusted_final_score >= 90 and er.confirmed
    elif er.level == "MEDIUM":
        allowed = (m.classification == "A+") and er.confirmed
    if er.mega:
        allowed = allowed and m.adjusted_final_score >= 90 and er.confirmed
    if hb and er.level == "EXTREME":
        allowed = False
    if m.adjusted_final_score < 75:
        allowed = False
    if not er.confirmed and er.level != "LOW":
        allowed = False
    m.event_trade_allowed = allowed

    if er.level == "LOW":
        return  # normal mode: no downgrades, no event warnings

    # ---- downgrade the recommendation + warn ----
    if m.group == "ACTIVE" and not allowed:
        m.group = "AVOID" if er.level == "EXTREME" else "WATCHLIST"

    w = m.warnings
    if er.mega:
        w.insert(0, "Mega liquidity event risk: bullish setups may fail due to broad de-risking.")
    if er.level == "EXTREME":
        w.insert(0, "No trade due to extreme market event risk.")
    elif not allowed and m.classification in ("A+", "A", "B"):
        tag = "market event risk blocks trade" if not er.confirmed else "downgraded to watchlist"
        w.insert(0, f"{er.level.title()} market event risk: {tag} "
                    f"(adj {m.adjusted_final_score:.0f}, x{er.multiplier:g}).")
    if hb and er.level in ("HIGH", "EXTREME"):
        w.append("High-beta name penalized under event risk.")
    if not er.confirmed and er.level != "LOW":
        w.append(f"Market not confirming (regime/VIX/SPY/QQQ) — need regime ≥ {REQ_REGIME[er.level]}.")
