#!/usr/bin/env python3
"""Options-flow confirmation score (UOA-style, adapted for premium BUYERS).

One Tradier chains call per ticker (both calls+puts) -> near-the-money
aggregates -> a 0-100 flow score + a directional tilt:

  40%  call vol/OI ratio   (new positioning — the heart of UOA)
  30%  call volume         (log scale; activity has to be real)
  20%  tight spread        (tradeability)
  10%  LOW implied vol     (inverted vs classic UOA: we BUY premium,
                            so rich IV is a cost, not a virtue)

Tilt = put/call volume ratio near the money: <0.6 call-tilted, >1.5 put-tilted.
This is a CONFIRMATION layer (shown as info), not a selector — pending the
backtest verdict in backtest_flow.py.
"""
from __future__ import annotations

import datetime as dt
import math
import os

import requests

TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")


def _sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def chain_flow(ticker: str, price: float, min_dte: int = 30, max_dte: int = 45):
    """Aggregate near-the-money (±10%) call+put activity for one expiry."""
    if not TRADIER_TOKEN or price <= 0:
        return None
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    today = dt.date.today()
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/expirations",
                         params={"symbol": ticker}, headers=hdr, timeout=10)
        exp = (r.json().get("expirations") or {}).get("date") or [] if r.ok else []
        if isinstance(exp, str):
            exp = [exp]
    except Exception:  # noqa: BLE001
        return None
    pick = None
    best = 1e9
    for e in exp:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except (ValueError, TypeError):
            continue
        if min_dte <= d <= max_dte and abs(d - 40) < best:
            best, pick = abs(d - 40), e
    if not pick:
        return None
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/chains",
                         params={"symbol": ticker, "expiration": pick, "greeks": "true"},
                         headers=hdr, timeout=15)
        opts = (r.json().get("options") or {}).get("option") or [] if r.ok else []
    except Exception:  # noqa: BLE001
        return None

    cv = co = pv = po = 0
    spreads, ivs = [], []
    for o in opts:
        K = _sf(o.get("strike"))
        if not (price * 0.90 <= K <= price * 1.10):
            continue
        vol = int(_sf(o.get("volume")))
        oi = int(_sf(o.get("open_interest")))
        if o.get("option_type") == "call":
            cv += vol
            co += oi
        else:
            pv += vol
            po += oi
        bid, ask = _sf(o.get("bid")), _sf(o.get("ask"))
        mid = (bid + ask) / 2
        if mid > 0 and bid > 0:
            spreads.append((ask - bid) / mid * 100)
        g = o.get("greeks") or {}
        iv = _sf(g.get("mid_iv") or g.get("smv_vol"))
        if iv > 0:
            ivs.append(iv * 100)

    vol_oi = cv / co if co > 0 else 0.0
    med_spread = sorted(spreads)[len(spreads) // 2] if spreads else 999
    atm_iv = sorted(ivs)[len(ivs) // 2] if ivs else 0.0
    pcr = pv / cv if cv > 0 else 9.9

    s = 0.0
    s += 40 * min(vol_oi, 2.0) / 2.0                       # new positioning
    s += 30 * min(math.log10(cv + 1) / 4.5, 1.0)           # real activity (log)
    s += 20 * max(0.0, 1 - min(med_spread, 20) / 20)       # tight spreads
    s += 10 * max(0.0, 1 - min(atm_iv, 120) / 120)         # cheap premium (inverted IV)
    tilt = "CALL" if pcr < 0.6 else ("PUT" if pcr > 1.5 else "neutral")
    return {"score": round(s, 1), "vol_oi": round(vol_oi, 2), "call_vol": cv,
            "put_vol": pv, "pcr": round(pcr, 2), "spread": round(med_spread, 1),
            "atm_iv": round(atm_iv, 1), "tilt": tilt, "expiry": pick,
            "unusual": vol_oi >= 1.0 and cv >= 1000}
