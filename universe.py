#!/usr/bin/env python3
"""Dynamic leadership ticker universe for the scanner.

Replaces the scanner's fixed 27-ticker list with a weekly-refreshed leadership
universe built from the Nasdaq-100 + S&P 500: filter for liquid, trending,
above-200-SMA names, score each on relative strength / trend quality / sector
strength / institutional demand / options liquidity, take the top 50, and merge
a core backup list. Cached to dynamic_leadership_universe.json (regenerated when
older than 7 days or on force-refresh). The scanner only consumes the resulting
ticker list + per-ticker (company, sector, sector-ETF) mapping — nothing else in
the scanner changes. If anything here fails, the caller falls back to the fixed
27-ticker list.

This module is self-contained (no import of vcp_tracker) to avoid cycles.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None

PROJECT = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(PROJECT, "dynamic_leadership_universe.json")
MAX_AGE_DAYS = 7
TOP_N = 50

TRADIER_TOKEN = os.environ.get("TRADIER_TOKEN", "")
TRADIER_BASE = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")

BENCH = ["SPY", "QQQ"]

# GICS sector -> SPDR sector ETF (used for the sector-strength layer).
GICS_ETF = {
    # GICS names (Wikipedia) + yfinance .info sector names (lookup path)
    "Information Technology": "XLK", "Technology": "XLK",
    "Communication Services": "XLC", "Communications": "XLC",
    "Consumer Discretionary": "XLY", "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP", "Consumer Defensive": "XLP",
    "Financials": "XLF", "Financial Services": "XLF",
    "Health Care": "XLV", "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB", "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}
ETF_NAME = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate",
}

# Core backup list (the original fixed universe) — always merged in if liquid.
CORE_BACKUP = [
    "NVDA", "AVGO", "AMD", "MRVL", "MU",
    "CRWD", "PANW", "FTNT", "ZS", "CYBR",
    "PLTR", "DDOG", "NOW", "SNOW", "MDB",
    "ANET", "VRT",
    "META", "AMZN", "MSFT", "GOOGL",
    "HOOD", "APP", "TSLA", "COIN", "NFLX",
]

LEAD_WEIGHTS = {"rs": 0.35, "trend": 0.25, "sector": 0.20, "inst": 0.10, "opt": 0.10}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _ret(close: pd.Series, days: int) -> float:
    if close is None or len(close) <= days:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def _sma(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n:
        return float("nan")
    return float(close.rolling(n).mean().iloc[-1])


def _gics_etf(sector: str) -> str:
    return GICS_ETF.get(str(sector).strip(), "XLK")


# --------------------------------------------------------------------------
# starting universe: Nasdaq-100 + S&P 500 constituents (ticker -> company, sector)
# --------------------------------------------------------------------------
def _read_tables(url: str):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25).text
    return pd.read_html(io.StringIO(html))


def fetch_constituents() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    # S&P 500
    try:
        for df in _read_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
            cols = {str(c).lower(): c for c in df.columns}
            if "symbol" in cols and any("sector" in c for c in cols):
                scol = next(cols[c] for c in cols if "gics sector" in c) if any("gics sector" in c for c in cols) \
                    else next(cols[c] for c in cols if "sector" in c)
                ncol = cols.get("security") or cols.get("company")
                for _, r in df.iterrows():
                    sym = str(r[cols["symbol"]]).replace(".", "-").strip().upper()
                    if sym and sym != "NAN":
                        out[sym] = (str(r[ncol]) if ncol else sym, str(r[scol]))
                break
    except Exception:  # noqa: BLE001
        pass
    # Nasdaq-100
    try:
        for df in _read_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
            cols = {str(c).lower(): c for c in df.columns}
            tkey = next((cols[c] for c in cols if "ticker" in c or "symbol" in c), None)
            skey = next((cols[c] for c in cols if "sector" in c), None)
            if tkey and skey:
                nkey = next((cols[c] for c in cols if "company" in c or "name" in c), None)
                for _, r in df.iterrows():
                    sym = str(r[tkey]).replace(".", "-").strip().upper()
                    if sym and sym != "NAN":
                        out.setdefault(sym, (str(r[nkey]) if nkey else sym, str(r[skey])))
                break
    except Exception:  # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------
# leadership component scores (all from daily OHLCV)
# --------------------------------------------------------------------------
def _rs_score(close, spy, qqq, etf) -> float:
    s = 0.0
    if _ret(close, 21) > _ret(spy, 21): s += 15
    if _ret(close, 63) > _ret(spy, 63): s += 15
    if _ret(close, 21) > _ret(qqq, 21): s += 15
    if _ret(close, 63) > _ret(qqq, 63): s += 15
    if etf is not None:
        if _ret(close, 21) > _ret(etf, 21): s += 20
        if _ret(close, 63) > _ret(etf, 63): s += 20
    return float(np.clip(s, 0, 100))


def _trend_score(close, price, hi52, lo52) -> float:
    s50, s150, s200 = _sma(close, 50), _sma(close, 150), _sma(close, 200)
    s50_prev = float(close.rolling(50).mean().iloc[-21]) if len(close) > 71 else s50
    checks = [
        price > s50, price > s150, price > s200,
        s50 > s150, s150 > s200, s50 > s50_prev,
        hi52 > 0 and price >= hi52 * 0.85,
        lo52 > 0 and price >= lo52 * 1.30,
    ]
    return float(sum(bool(c) for c in checks) / len(checks) * 100)


def _sector_score(etf, spy, qqq) -> float:
    if etf is None:
        return 50.0
    s = 0.0
    price = float(etf.iloc[-1])
    if price > _sma(etf, 50): s += 20
    if price > _sma(etf, 200): s += 20
    if _ret(etf, 21) > _ret(spy, 21): s += 20
    if _ret(etf, 63) > _ret(spy, 63): s += 20
    if _ret(etf, 21) > _ret(qqq, 21): s += 20
    return float(np.clip(s, 0, 100))


def _inst_score(df: pd.DataFrame) -> float:
    d = df.tail(25)
    ch = d["Close"].diff()
    upvol = float(d.loc[ch > 0, "Volume"].sum())
    dnvol = float(d.loc[ch < 0, "Volume"].sum())
    acc = int(((ch > 0) & (d["Volume"] > d["Volume"].shift(1))).sum())
    dist = int(((ch < 0) & (d["Volume"] > d["Volume"].shift(1))).sum())
    up_avg = float(d.loc[ch > 0, "Volume"].mean() or 0)
    dn_avg = float(d.loc[ch < 0, "Volume"].mean() or 0)
    s = 0.0
    if upvol > dnvol: s += 34
    if acc > dist: s += 33
    if 0 < dn_avg < up_avg: s += 33
    return float(np.clip(s, 0, 100))


def _options_score(ticker: str, price: float) -> float:
    """Best-effort options-liquidity score via Tradier (neutral 50 if no token
    or lookup fails). Mirrors the spec's components."""
    if not TRADIER_TOKEN or price <= 0:
        return 50.0
    hdr = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
    today = dt.date.today()
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/expirations",
                         params={"symbol": ticker}, headers=hdr, timeout=10)
        if not r.ok:
            return 0.0
        exp = (r.json().get("expirations") or {}).get("date") or []
        if isinstance(exp, str):
            exp = [exp]
    except Exception:  # noqa: BLE001
        return 0.0
    pick = None
    for e in exp:
        try:
            d = (dt.date.fromisoformat(e) - today).days
        except (ValueError, TypeError):
            continue
        if 30 <= d <= 60:
            pick = e
            break
    if not pick:
        return 10.0  # optionable but no 30-60 DTE expiry
    try:
        r = requests.get(f"{TRADIER_BASE}/markets/options/chains",
                         params={"symbol": ticker, "expiration": pick, "greeks": "false"},
                         headers=hdr, timeout=12)
        opts = (r.json().get("options") or {}).get("option") or []
    except Exception:  # noqa: BLE001
        return 10.0
    # nearest-the-money call
    best, bestd = None, 1e9
    for o in opts:
        if o.get("option_type") != "call":
            continue
        k = float(o.get("strike") or 0)
        if abs(k - price) < bestd:
            bestd, best = abs(k - price), o
    s = 10.0  # has options + 30-60 DTE
    if best:
        oi = int(best.get("open_interest") or 0)
        vol = int(best.get("volume") or 0)
        bid, ask = float(best.get("bid") or 0), float(best.get("ask") or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread = (ask - bid) / mid * 100 if mid > 0 else 999
        if oi > 500: s += 35
        if vol > 100: s += 30
        if spread < 10: s += 25
    return float(np.clip(s, 0, 100))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _download(tickers: list[str], chunk: int = 40, pause: float = 1.0) -> dict[str, pd.DataFrame]:
    """Chunked download — a single 500+ ticker call gets heavily throttled by
    Yahoo (~30% spurious failures), so fetch in small batches with retries."""
    out: dict[str, pd.DataFrame] = {}
    uniq = list(dict.fromkeys(tickers))
    for i in range(0, len(uniq), chunk):
        part = uniq[i:i + chunk]
        for attempt in range(2):
            try:
                data = yf.download(part, period="1y", interval="1d", group_by="ticker",
                                   auto_adjust=True, threads=True, progress=False)
            except Exception:  # noqa: BLE001
                data = None
            got = 0
            if data is not None and len(part) > 1 and isinstance(data.columns, pd.MultiIndex):
                for t in part:
                    if t in data.columns.get_level_values(0):
                        sub = data[t].dropna(how="all")
                        if not sub.empty:
                            out[t] = sub; got += 1
            elif data is not None and len(part) == 1:
                sub = data.dropna(how="all")
                if not sub.empty:
                    out[part[0]] = sub; got += 1
            if got:
                break
            time.sleep(pause * 1.5)
        time.sleep(pause)
    print(f"  [universe] downloaded {len(out)}/{len(uniq)} symbols", file=sys.stderr)
    return out


def generate(limit: int | None = None) -> dict:
    if yf is None:
        raise RuntimeError("yfinance unavailable")
    cons = fetch_constituents()
    if len(cons) < 50:
        raise RuntimeError(f"constituent fetch too small ({len(cons)})")
    tickers = list(cons)
    if limit:
        tickers = tickers[:limit]

    sector_etfs = sorted({_gics_etf(cons[t][1]) for t in tickers})
    print(f"  [universe] {len(tickers)} candidates; downloading data ...", file=sys.stderr)
    hist = _download(tickers + BENCH + sector_etfs)
    spy = hist["SPY"]["Close"].astype(float) if "SPY" in hist else None
    qqq = hist["QQQ"]["Close"].astype(float) if "QQQ" in hist else None
    etf_close = {e: hist[e]["Close"].astype(float) for e in sector_etfs if e in hist}

    rows = []
    for t in tickers:
        df = hist.get(t)
        if df is None or len(df) < 200:
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(df) < 200:
            continue
        close = df["Close"].astype(float)
        price = float(close.iloc[-1])
        if not math.isfinite(price) or price <= 0:
            continue
        avgvol20 = float(df["Volume"].rolling(20).mean().iloc[-1])
        dollar_vol = price * avgvol20
        sma200 = _sma(close, 200)
        hi52 = float(close.tail(252).max())
        lo52 = float(close.tail(252).min())
        # ---- initial filters ----
        if price < 20:                       continue
        if dollar_vol < 100_000_000:         continue
        if not (price > sma200):             continue
        if hi52 > 0 and (price < hi52 * 0.70):  continue   # not >30% below 52w high
        etf = etf_close.get(_gics_etf(cons[t][1]))
        rows.append({
            "ticker": t, "company": cons[t][0], "gics": cons[t][1],
            "etf": _gics_etf(cons[t][1]), "price": price,
            "rs": _rs_score(close, spy, qqq, etf),
            "trend": _trend_score(close, price, hi52, lo52),
            "sector": _sector_score(etf, spy, qqq),
            "inst": _inst_score(df),
            "opt": 50.0,   # neutral until checked for top candidates below
        })

    if not rows:
        raise RuntimeError("no candidates passed initial filters")

    # preliminary leadership (opt neutral) to pick who to options-check
    for r in rows:
        r["prelim"] = (LEAD_WEIGHTS["rs"] * r["rs"] + LEAD_WEIGHTS["trend"] * r["trend"]
                       + LEAD_WEIGHTS["sector"] * r["sector"] + LEAD_WEIGHTS["inst"] * r["inst"]
                       + LEAD_WEIGHTS["opt"] * r["opt"])
    rows.sort(key=lambda r: -r["prelim"])

    # options liquidity for the strongest ~70 (best-effort; bounded API use)
    check = rows[:max(TOP_N + 20, 70)]
    if TRADIER_TOKEN and check:
        print(f"  [universe] options-liquidity check on top {len(check)} (Tradier) ...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for r, sc in zip(check, ex.map(lambda x: _options_score(x["ticker"], x["price"]), check)):
                r["opt"] = sc

    for r in rows:
        r["leadership"] = round(
            LEAD_WEIGHTS["rs"] * r["rs"] + LEAD_WEIGHTS["trend"] * r["trend"]
            + LEAD_WEIGHTS["sector"] * r["sector"] + LEAD_WEIGHTS["inst"] * r["inst"]
            + LEAD_WEIGHTS["opt"] * r["opt"], 1)
    rows.sort(key=lambda r: -r["leadership"])

    ts = dt.datetime.now().isoformat(timespec="seconds")
    selected: dict[str, dict] = {}
    for r in rows[:TOP_N]:
        selected[r["ticker"]] = {
            "ticker": r["ticker"], "company": r["company"], "gics_sector": r["gics"],
            "etf": r["etf"], "leadership_score": r["leadership"],
            "reason_selected": (f"RS {r['rs']:.0f} / Trend {r['trend']:.0f} / "
                                f"Sector {r['sector']:.0f} / Inst {r['inst']:.0f} / "
                                f"Opt {r['opt']:.0f}"),
            "generated_timestamp": ts, "source_universe": "sp500+ndx",
            "core_backup_flag": False,
        }

    # always merge core backup names that exist + pass basic liquidity
    by_t = {r["ticker"]: r for r in rows}
    for t in CORE_BACKUP:
        if t in selected:
            selected[t]["core_backup_flag"] = True
            continue
        r = by_t.get(t)
        if r:  # passed initial filters already
            selected[t] = {
                "ticker": t, "company": r["company"], "gics_sector": r["gics"],
                "etf": r["etf"], "leadership_score": r["leadership"],
                "reason_selected": "core backup", "generated_timestamp": ts,
                "source_universe": "core_backup", "core_backup_flag": True,
            }

    records = sorted(selected.values(), key=lambda x: -x["leadership_score"])

    # ---- WEAKNESS (laggard) universe for the PUT side --------------------
    # Backtest finding: puts on leaders lose (7-18% win); puts only worked on
    # laggards (below 50DMA + lagging SPY) with the exceptional-breakdown gate.
    bear_records = []
    spy_c = hist.get("SPY")
    spy63 = _ret(spy_c["Close"].astype(float), 63) if spy_c is not None else 0.0
    for t in tickers:
        df = hist.get(t)
        if df is None or len(df) < 200:
            continue
        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < 200:
            continue
        close = df["Close"].astype(float)
        price = float(close.iloc[-1])
        if not math.isfinite(price) or price < 10:
            continue
        avgvol20 = float(df["Volume"].rolling(20).mean().iloc[-1])
        if price * avgvol20 < 50_000_000:
            continue
        sma50 = _sma(close, 50)
        if not (math.isfinite(sma50) and price < sma50):
            continue
        lag = _ret(close, 63) - spy63
        if lag > -5:
            continue                                   # must lag SPY by >5pp over 3m
        weakness = round(-lag + (sma50 / price - 1) * 100, 1)
        bear_records.append({"ticker": t, "company": cons[t][0],
                             "gics_sector": cons[t][1], "etf": _gics_etf(cons[t][1]),
                             "weakness_score": weakness,
                             "generated_timestamp": ts})
    bear_records.sort(key=lambda x: -x["weakness_score"])
    bear_records = bear_records[:25]
    print(f"  [universe] weakness list: {len(bear_records)} laggards for the put side",
          file=sys.stderr)

    payload = {"generated_timestamp": ts, "count": len(records),
               "weights": LEAD_WEIGHTS, "tickers": records,
               "bear_tickers": bear_records}
    with open(CACHE_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [universe] selected {len(records)} tickers -> {CACHE_PATH}", file=sys.stderr)
    return payload


def _is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < MAX_AGE_DAYS * 86400


def get_dynamic_leadership_universe(refresh: bool = False, limit: int | None = None) -> dict:
    """Return the cached universe, regenerating if missing / stale / forced."""
    if not refresh and _is_fresh(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    return generate(limit=limit)


def load_universe_for_scanner(refresh: bool = False, limit: int | None = None):
    """Shape the universe for the scanner: ticker list + lookup maps
    (+ the weakness/laggard list used by the PUT side)."""
    payload = get_dynamic_leadership_universe(refresh=refresh, limit=limit)
    recs = payload.get("tickers", [])
    if len(recs) < 10:
        raise RuntimeError("universe too small")
    tickers = [r["ticker"] for r in recs]
    meta = {r["ticker"]: (r["company"], r["gics_sector"]) for r in recs}
    ticker_etf = {r["ticker"]: r["etf"] for r in recs}
    bears = [r for r in payload.get("bear_tickers", []) if r["ticker"] not in set(tickers)]
    bear_tickers = [r["ticker"] for r in bears]
    meta.update({r["ticker"]: (r["company"], r["gics_sector"]) for r in bears})
    ticker_etf.update({r["ticker"]: r["etf"] for r in bears})
    sector_etfs = sorted(set(ticker_etf.values()))
    etf_name = {e: ETF_NAME.get(e, e) for e in sector_etfs}
    return {"tickers": tickers, "bear_tickers": bear_tickers, "meta": meta,
            "ticker_etf": ticker_etf, "sector_etfs": sector_etfs, "etf_name": etf_name,
            "generated": payload.get("generated_timestamp", "?"), "count": len(tickers)}


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    lim = None
    for a in sys.argv:
        if a.startswith("--limit="):
            lim = int(a.split("=")[1])
    p = generate(limit=lim) if force else get_dynamic_leadership_universe(limit=lim)
    print(f"\nUniverse: {p['count']} tickers (generated {p['generated_timestamp']})")
    for r in p["tickers"][:20]:
        flag = " *core" if r["core_backup_flag"] else ""
        print(f"  {r['ticker']:6} {r['leadership_score']:>5}  {r['etf']:4} {r['gics_sector'][:22]:22} {r['reason_selected']}{flag}")
