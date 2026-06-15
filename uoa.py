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
OI_SNAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "reports", "uoa_oi_snapshot.json")
EARN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "earnings_cache.json")

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


def _occ(ticker: str, expiry: str, strike: float, typ: str) -> str:
    """OCC option symbol, e.g. NVDA260619P00120000."""
    e = dt.date.fromisoformat(expiry).strftime("%y%m%d")
    cp = "C" if typ == "call" else "P"
    return f"{ticker}{e}{cp}{int(round(strike * 1000)):08d}"


# Backtest finding (backtest_uoa_enh.py): the EDGE lives in MODERATE volume
# spikes vs a contract's own recent average — 3-7x is the sweet spot (+15pp),
# while 10x+ monster spikes have NO edge (likely hedging / rebalancing / reaction
# to public news). Rank reflects edge, not raw size.
def _spike_band(ratio):
    if ratio is None:
        return ("", 1)            # unknown -> neutral
    if ratio >= 10:
        return ("monster", 0)     # no edge — down-rank
    if ratio >= 7:
        return ("strong", 2)
    if ratio >= 3:
        return ("sweet", 3)       # the validated +15pp band
    return ("mild", 1)


def _contract_vol_ratio(ticker, expiry, strike, typ, today_vol):
    """today's contract volume / its own prior ~10-session average volume."""
    try:
        r = _get(f"{TRADIER_BASE}/markets/history",
                 params={"symbol": _occ(ticker, expiry, strike, typ), "interval": "daily",
                         "start": (dt.date.today() - dt.timedelta(days=25)).isoformat(),
                         "end": dt.date.today().isoformat()},
                 headers=_hdr(), timeout=10)
        days = (r.json().get("history") or {}).get("day") if r.ok else None
    except Exception:  # noqa: BLE001
        return None
    if not days:
        return None
    if isinstance(days, dict):
        days = [days]
    vols = [_sf(d.get("volume")) for d in days]
    prior = [v for v in vols[-11:-1] if v > 0]      # ~10 prior sessions (exclude latest)
    if len(prior) < 3:
        return None
    base = sum(prior) / len(prior)
    return round(today_vol / base, 1) if base > 0 else None


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
    spreads, ivs, standouts, _contracts = [], [], [], []
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
        # classic contract-level UOA test + notional filter: >= $250K premium
        # actually traded (kills penny-contract noise)
        last_px = _sf(o.get("last"))
        if (vol >= 500 and oi > 0 and vol / oi >= 2.0
                and vol * last_px * 100 >= 250_000):
            standouts.append({"type": typ, "strike": K, "vol": vol, "oi": oi,
                              "vol_oi": round(vol / oi, 1),
                              "last": last_px, "delta": round(_sf(g.get("delta")), 2)})
        if vol >= 200:                       # compact map for next-day OI confirmation
            _contracts.append((typ, K, vol, oi))

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
    # Direction from PER-SIDE vol/OI (activity relative to each side's own open
    # interest) — self-normalizes names with a structurally put-heavy skew.
    c_voloi = cv / co if co > 0 else 0.0
    p_voloi = pv / po if po > 0 else 0.0
    if p_voloi >= 1.5 * max(c_voloi, 0.01) and pv >= 500:
        direction = "PUT"
    elif c_voloi >= 1.5 * max(p_voloi, 0.01) and cv >= 500:
        direction = "CALL"
    else:
        direction = "mixed"
    standouts.sort(key=lambda s: -s["vol"])
    standouts = standouts[:3]
    # ---- spike-quality: each standout's volume vs its OWN recent average.
    # Backtest: 3-7x = the edge; 10x+ monsters = none. spike_rank ranks by edge.
    spike_rank = 1
    for st in standouts:
        ratio = _contract_vol_ratio(ticker, expiry, st["strike"], st["type"], st["vol"])
        band, rank = _spike_band(ratio)
        st["vol_ratio"], st["band"] = ratio, band
        spike_rank = max(spike_rank, rank)
    unusual = vol_oi >= 1.0 and tot_vol >= 2000
    # PRE-MOVE flag retained for reference only. NOTE (backtest_uoa_enh.py):
    # PRE-MOVE was NOT predictive — it did not beat reactive flow, so it is no
    # longer prioritised in ranking or badged as "the predictive subset".
    pre_move = abs(chg_pct) < 1.0 and (unusual or vol_oi >= 0.8 or bool(standouts))
    return {"ticker": ticker, "price": round(price, 2), "chg_pct": round(chg_pct, 2),
            "expiry": expiry, "dte": dte, "score": score, "vol_oi": round(vol_oi, 2),
            "call_vol": cv, "put_vol": pv, "pcr": round(pcr, 2),
            "c_voloi": round(c_voloi, 2), "p_voloi": round(p_voloi, 2),
            "atm_iv": round(atm_iv, 1), "spread": round(med_spread, 1),
            "direction": direction, "unusual": unusual, "pre_move": pre_move,
            "spike_rank": spike_rank, "standouts": standouts, "_contracts": _contracts}


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
    # Rank by spike-quality first (sweet 3-7x up, monster 10x+ down), then score.
    rows.sort(key=lambda r: (-r.get("spike_rank", 1), -r["score"]))

    # Pre-market / closed-market scans see zero contract volume — bail out
    # BEFORE touching the OI snapshot or the cache (a zero scan must never
    # wipe the OI-confirmation baseline).
    if rows and sum(r["call_vol"] + r["put_vol"] for r in rows) == 0:
        print("  [uoa] zero option volume (market closed) — keeping previous cache",
              file=sys.stderr)
        try:
            with open(OUT_PATH) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            pass

    # ---- OI confirmation: did yesterday's spike volume become NEW open
    # interest? (OI only updates overnight, so confirmation appears the next
    # trading day: oi_today > oi_then + 30% of the spiked volume = OPENED.)
    today_iso = dt.date.today().isoformat()
    try:
        with open(OI_SNAP_PATH) as fh:
            snap = json.load(fh)
    except Exception:  # noqa: BLE001
        snap = {}
    new_snap = {}
    for r in rows:
        for typ, K, vol, oi in r.pop("_contracts", []):
            key = f"{r['ticker']}|{r['expiry']}|{typ}|{K:g}"
            new_snap[key] = {"oi": oi, "vol": vol, "date": today_iso}
        for st in r["standouts"]:
            key = f"{r['ticker']}|{r['expiry']}|{st['type']}|{st['strike']:g}"
            prev = snap.get(key)
            if prev and prev.get("date", "") < today_iso:
                if st["oi"] >= prev["oi"] + max(0.3 * prev["vol"], 100):
                    st["oi_confirmed"] = True       # volume became open interest
                elif st["oi"] <= prev["oi"]:
                    st["oi_confirmed"] = False      # likely closing/day-trade churn
    try:
        os.makedirs(os.path.dirname(OI_SNAP_PATH), exist_ok=True)
        with open(OI_SNAP_PATH, "w") as fh:
            json.dump(new_snap, fh)
    except Exception:  # noqa: BLE001
        pass

    # ---- earnings annotation (pre-earnings flow is mostly hedging) ----
    try:
        with open(EARN_PATH) as fh:
            ecache = json.load(fh)
        today_d = dt.date.today()
        for r in rows:
            ed = (ecache.get(r["ticker"]) or {}).get("date")
            if ed and ed != "unknown":
                days = (dt.date.fromisoformat(ed) - today_d).days
                if 0 <= days <= 30:
                    r["earn_days"] = days
    except Exception:  # noqa: BLE001
        pass

    payload = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "count": len(rows), "universe": len(names),
               "weights": "30% vol/OI · 20% IV · 20% volume · 15% |Δprice| · 15% spread",
               "rows": rows}
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
