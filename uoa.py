#!/usr/bin/env python3
"""STANDALONE Unusual Options Activity (UOA) scanner.

Completely separate from the pattern scanner — no shared scoring, no effect on
recommendations. Implements the classic UOA rubric as specified:

    30%  Vol/OI ratio          (activity relative to open interest)
    20%  Implied volatility    (demand pumping premium — detector logic,
                                NOT a buy-quality signal)
    20%  Volume
    15%  Absolute % price change of the underlying (momentum)
    15%  Tight bid/ask spread  (liquidity)

Universe: the option-active pond — current leadership + laggard lists plus a
curated set of the most actively-traded option names. UOA needs busy chains;
this is the universe that works best for it.

Scan: one front expiry (>=5 DTE, where flow concentrates) per name, both calls
and puts, near-the-money ±15%. Flags contract-level standouts
(vol >= 500 and vol/OI >= 2 — the classic UOA contract test).

Output: dashboard/uoa.json for the standalone UOA tab.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dashboard", "uoa.json")

# Curated most-active option names (flow lives here even when they're not
# technical leaders/laggards).
ACTIVE_OPTION_NAMES = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX",
    "AVGO", "PLTR", "HOOD", "COIN", "SOFI", "INTC", "BAC", "F", "RIVN",
    "MARA", "RIOT", "SMCI", "MU", "BABA", "UBER", "DIS", "PYPL", "SNAP",
    "CCL", "PFE", "T", "VZ", "WMT", "XOM", "CVX", "JPM", "GS", "SHOP",
    "ARM", "DELL", "ORCL", "QCOM", "CRM", "BA", "NKE", "GM",
]


def _sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _hdr():
    return {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}


# Global soft rate limiter: keep total request rate under Tradier's ~120/min
# even with parallel workers (a 300-name scan = ~600 calls ≈ 5-6 min).
import threading
_RL = threading.Lock()
_RL_LAST = [0.0]
_RL_MIN_INTERVAL = 0.55


def _get(url, **kw):
    with _RL:
        wait = _RL_LAST[0] + _RL_MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _RL_LAST[0] = time.time()
    return requests.get(url, **kw)


def uoa_universe(target: int = 300) -> list[str]:
    """The option-active pond, expanded for coverage (Jun-10 forensic: 18/24
    big losers weren't scanned at 116 names). Core = leaders + laggards +
    curated actives; filled to ~`target` with the highest dollar-volume S&P
    names today (liquid stock ~ liquid options)."""
    names = list(ACTIVE_OPTION_NAMES)
    try:
        import vcp_tracker as V
        V.apply_universe()
        names = list(V.UNIVERSE) + list(V.BEAR_UNIVERSE) + names
    except Exception:  # noqa: BLE001
        pass
    names = list(dict.fromkeys(names))
    try:
        import universe as U
        rest = [s for s in U.fetch_constituents() if s not in set(names)]
        q = batch_quotes(rest)
        ranked = sorted(((t, _sf(v.get("last")) * max(_sf(v.get("average_volume")),
                                                      _sf(v.get("volume"))))
                         for t, v in q.items()), key=lambda x: -x[1])
        for t, dv in ranked:
            if len(names) >= target:
                break
            if dv >= 50_000_000:          # >= $50M avg daily dollar volume
                names.append(t)
    except Exception as exc:  # noqa: BLE001
        print(f"  [uoa] universe expansion skipped ({exc})", file=sys.stderr)
    return names


def batch_quotes(symbols: list[str]) -> dict:
    out = {}
    for i in range(0, len(symbols), 100):
        part = ",".join(symbols[i:i + 100])
        try:
            r = requests.get(f"{TRADIER_BASE}/markets/quotes",
                             params={"symbols": part}, headers=_hdr(), timeout=15)
            data = (r.json().get("quotes") or {}).get("quote") if r.ok else None
            if isinstance(data, dict):
                data = [data]
            for q in data or []:
                out[q.get("symbol")] = q
        except Exception:  # noqa: BLE001
            pass
    return out


def scan_one(ticker: str, price: float, chg_pct: float):
    """Front-expiry chain -> UOA rubric score + contract standouts."""
    if price <= 0:
        return None
    today = dt.date.today()
    try:
        r = _get(f"{TRADIER_BASE}/markets/options/expirations",
                 params={"symbol": ticker}, headers=_hdr(), timeout=10)
        exp = (r.json().get("expirations") or {}).get("date") or [] if r.ok else []
        if isinstance(exp, str):
            exp = [exp]
    except Exception:  # noqa: BLE001
        return None
    front = None
    for e in exp:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except (ValueError, TypeError):
            continue
        if d >= 5:
            front = (e, d)
            break
    if not front:
        return None
    expiry, dte = front
    try:
        r = _get(f"{TRADIER_BASE}/markets/options/chains",
                 params={"symbol": ticker, "expiration": expiry, "greeks": "true"},
                 headers=_hdr(), timeout=15)
        opts = (r.json().get("options") or {}).get("option") or [] if r.ok else []
    except Exception:  # noqa: BLE001
        return None

    cv = co = pv = po = 0
    spreads, ivs, standouts = [], [], []
    for o in opts:
        K = _sf(o.get("strike"))
        if not (price * 0.85 <= K <= price * 1.15):
            continue
        vol = int(_sf(o.get("volume")))
        oi = int(_sf(o.get("open_interest")))
        typ = o.get("option_type")
        if typ == "call":
            cv += vol; co += oi
        else:
            pv += vol; po += oi
        bid, ask = _sf(o.get("bid")), _sf(o.get("ask"))
        mid = (bid + ask) / 2
        if mid > 0 and bid > 0:
            spreads.append((ask - bid) / mid * 100)
        g = o.get("greeks") or {}
        iv = _sf(g.get("mid_iv") or g.get("smv_vol"))
        if iv > 0:
            ivs.append(iv * 100)
        # classic contract-level UOA test
        if vol >= 500 and oi > 0 and vol / oi >= 2.0:
            standouts.append({"type": typ, "strike": K, "vol": vol, "oi": oi,
                              "vol_oi": round(vol / oi, 1),
                              "last": _sf(o.get("last")), "delta": round(_sf(g.get("delta")), 2)})

    tot_vol, tot_oi = cv + pv, co + po
    vol_oi = tot_vol / tot_oi if tot_oi > 0 else 0.0
    med_spread = sorted(spreads)[len(spreads) // 2] if spreads else 999
    atm_iv = sorted(ivs)[len(ivs) // 2] if ivs else 0.0

    # ---- the specified rubric, normalized 0-1 then weighted to 0-100 ----
    n_voloi = min(vol_oi, 2.0) / 2.0
    n_iv = min(atm_iv, 150) / 150
    n_vol = min(math.log10(tot_vol + 1) / 5.0, 1.0)
    n_chg = min(abs(chg_pct), 5.0) / 5.0
    n_spr = max(0.0, 1 - min(med_spread, 20) / 20)
    score = round(100 * (0.30 * n_voloi + 0.20 * n_iv + 0.20 * n_vol
                         + 0.15 * n_chg + 0.15 * n_spr), 1)
    pcr = pv / cv if cv > 0 else 9.9
    direction = "CALL" if pcr < 0.7 else ("PUT" if pcr > 1.4 else "mixed")
    standouts.sort(key=lambda s: -s["vol"])
    return {"ticker": ticker, "price": round(price, 2), "chg_pct": round(chg_pct, 2),
            "expiry": expiry, "dte": dte, "score": score, "vol_oi": round(vol_oi, 2),
            "call_vol": cv, "put_vol": pv, "pcr": round(pcr, 2), "atm_iv": round(atm_iv, 1),
            "spread": round(med_spread, 1), "direction": direction,
            "unusual": vol_oi >= 1.0 and tot_vol >= 2000,
            "standouts": standouts[:3]}


def scan(max_workers: int = 4) -> dict:
    if not TRADIER_TOKEN:
        return {"error": "TRADIER_TOKEN required", "rows": []}
    names = uoa_universe()
    print(f"  [uoa] scanning {len(names)} option-active names ...", file=sys.stderr)
    quotes = batch_quotes(names)
    rows = []
    def work(t):
        q = quotes.get(t) or {}
        return scan_one(t, _sf(q.get("last")), _sf(q.get("change_percentage")))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(work, names):
            if res:
                rows.append(res)
    rows.sort(key=lambda r: -r["score"])
    payload = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "count": len(rows), "universe": len(names),
               "weights": "30% vol/OI · 20% IV · 20% volume · 15% |Δprice| · 15% spread",
               "rows": rows}
    # Pre-market / closed-market scans see zero contract volume — keep the
    # last session's rich data instead of clobbering it with an empty scan.
    if rows and sum(r["call_vol"] + r["put_vol"] for r in rows) == 0:
        print("  [uoa] zero option volume (market closed) — keeping previous cache",
              file=sys.stderr)
        try:
            with open(OUT_PATH) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return payload
    try:
        with open(OUT_PATH, "w") as fh:
            json.dump(payload, fh)
    except Exception:  # noqa: BLE001
        pass
    print(f"  [uoa] {len(rows)} scored; {sum(1 for r in rows if r['unusual'])} unusual",
          file=sys.stderr)
    return payload


if __name__ == "__main__":
    p = scan()
    for r in p.get("rows", [])[:15]:
        st = " | ".join(f"{s['type'][0].upper()}{s['strike']:g} v{s['vol']}/oi{s['oi']}"
                        for s in r["standouts"])
        print(f"{r['ticker']:6} {r['score']:>5}  {r['direction']:<5} vol/OI {r['vol_oi']:>5} "
              f"cv {r['call_vol']:>7,} pv {r['put_vol']:>7,} IV {r['atm_iv']:>5}% "
              f"chg {r['chg_pct']:>+5.1f}%  {'UNUSUAL ' if r['unusual'] else ''}{st}")
