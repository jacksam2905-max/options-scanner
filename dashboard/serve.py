#!/usr/bin/env python3
"""Local dashboard server with a live-refresh endpoint.

Serves dashboard/index.html and exposes POST/GET /api/refresh, which re-runs
the scanner against the CURRENT market, rewrites data.js, and returns the fresh
JSON to the page. Stdlib only. Bind is localhost-only.

    python3 dashboard/serve.py            # starts server + opens browser
    PORT=8787 python3 dashboard/serve.py  # custom port
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

import base64

ROOT = os.path.dirname(os.path.abspath(__file__))      # .../dashboard
PROJECT = os.path.dirname(ROOT)                         # project root
sys.path.insert(0, PROJECT)                             # so we can import vcp_tracker
PY = sys.executable or "python3"
PORT = int(os.environ.get("PORT", "8787"))
HOST = os.environ.get("HOST", "127.0.0.1")             # cloud: set HOST=0.0.0.0
OPEN = "--no-open" not in sys.argv
# Optional HTTP basic auth (set both to protect a public deployment).
DASH_USER = os.environ.get("DASH_USER", "")
DASH_PASS = os.environ.get("DASH_PASS", "")
# Optional server-side scan loop (seconds) so cloud data stays fresh w/o a browser.
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "0"))
SCAN_MODE = os.environ.get("SCAN_MODE", "best")
SCAN_ACCOUNT = os.environ.get("SCAN_ACCOUNT", "0")
QUOTE_TTL = float(os.environ.get("QUOTE_TTL", "2"))    # server-side live-quote cache (s)
# Background UOA auto-scan (seconds between runs; 0 = off). Separate lock so a
# long UOA scan never blocks the 5-min pattern re-scan.
UOA_INTERVAL = int(os.environ.get("UOA_INTERVAL", "1800"))
_lock = threading.Lock()
_ULOCK = threading.Lock()
_QLOCK = threading.Lock()
_QCACHE = {}                                            # symbols -> (ts, body) shared snapshot


def run_scan(mode: str, account: str) -> str:
    """Run the scanner, write data.json + data.js, return the JSON text."""
    data_json = os.path.join(ROOT, "data.json")
    cmd = [PY, "vcp_tracker.py", "--score-mode", mode, "--json", data_json]
    if account:
        try:
            if float(account) > 0:
                cmd += ["--account-size", str(float(account))]
        except ValueError:
            pass
    proc = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "scanner failed")[-1500:])
    with open(data_json) as fh:
        body = fh.read()
    with open(os.path.join(ROOT, "data.js"), "w") as fh:   # keep file:// mode working
        fh.write("window.SCAN_DATA = " + body + ";\n")
    return body


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def _send(self, code: int, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _auth(self) -> bool:
        """HTTP basic auth gate — active only when DASH_USER & DASH_PASS are set."""
        if not (DASH_USER and DASH_PASS):
            return True
        hdr = self.headers.get("Authorization", "")
        if hdr.startswith("Basic "):
            try:
                u, p = base64.b64decode(hdr[6:]).decode("utf-8").split(":", 1)
                if u == DASH_USER and p == DASH_PASS:
                    return True
            except Exception:  # noqa: BLE001
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="VCP Scanner"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def do_POST(self):
        if not self._auth():
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/refresh":
            return self._refresh()
        if path == "/api/option":
            return self._option()
        if path == "/api/analyze":
            return self._analyze()
        if path == "/api/uoa":
            return self._uoa(run=True)
        self._send(404, json.dumps({"error": "not found"}))

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/healthz", "/health"):            # unauthenticated health check
            return self._send(200, "ok", "text/plain")
        if not self._auth():
            return
        if path == "/api/refresh":
            return self._refresh()
        if path == "/api/quotes":
            return self._quotes()
        if path == "/api/uoa":
            return self._uoa(run=False)
        return super().do_GET()

    def _uoa(self, run: bool):
        """Standalone UOA scan. GET returns the cached result (if <15 min old);
        POST runs a fresh scan (~1-2 min, one chains call per name)."""
        cache = os.path.join(ROOT, "uoa.json")
        if not run:
            try:
                with open(cache) as fh:
                    return self._send(200, fh.read())   # client shows the run timestamp
            except FileNotFoundError:
                return self._send(200, json.dumps({"stale": True}))
        if not _ULOCK.acquire(blocking=False):
            return self._send(409, json.dumps({"error": "UOA scan already running"}))
        try:
            import uoa
            payload = uoa.scan()
            self._send(200, json.dumps(payload, default=lambda x: None))
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)[-600:]}))
        finally:
            _ULOCK.release()

    def _quotes(self):
        """Live batch quotes via Tradier, with a short server-side cache so the
        feed scales to many viewers: the server hits Tradier at most once per
        QUOTE_TTL seconds per symbol-set, no matter how many browsers poll."""
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        syms = (q.get("symbols", [""])[0] or "").strip()
        if not syms:
            return self._send(400, json.dumps({"error": "symbols required"}))
        now = time.time()
        with _QLOCK:                                    # serve a fresh-enough cached snapshot
            c = _QCACHE.get(syms)
            if c and (now - c[0]) < QUOTE_TTL:
                return self._send(200, c[1])
        token = os.environ.get("TRADIER_TOKEN", "")
        base = os.environ.get("TRADIER_BASE", "https://api.tradier.com/v1")
        if not token:
            return self._send(200, json.dumps({"error": "no tradier token", "quotes": {}}))
        try:
            import requests
            r = requests.get(f"{base}/markets/quotes",
                             params={"symbols": syms, "greeks": "false"},
                             headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                             timeout=10)
            data = (r.json().get("quotes") or {}).get("quote")
            if isinstance(data, dict):
                data = [data]
            out = {}
            for qd in data or []:
                out[qd.get("symbol")] = {
                    "last": qd.get("last"), "chg_pct": qd.get("change_percentage"),
                    "change": qd.get("change"), "open": qd.get("open"),
                    "prevclose": qd.get("prevclose"), "volume": qd.get("volume"),
                }
            body = json.dumps({"quotes": out, "ts": now})
            with _QLOCK:
                _QCACHE[syms] = (now, body)
            self._send(200, body)
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)[-300:], "quotes": {}}))

    def _option(self):
        """On-demand single-ticker option fetch + recompute of the option-
        dependent layers (one chain fetch only). Body: {ticker, data, account,
        size_factor, risk:{max_risk_pct,max_prem_loss}}."""
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception:  # noqa: BLE001
            req = {}
        t = req.get("data") or {}
        ticker = (req.get("ticker") or t.get("ticker") or "").upper()
        if not ticker:
            return self._send(400, json.dumps({"error": "ticker required"}))
        account = float(req.get("account", 0) or 0)
        size_factor = float(req.get("size_factor", 1) or 1)
        risk = req.get("risk") or {}
        mrp = float(risk.get("max_risk_pct", 1.5) or 1.5)
        mpl = float(risk.get("max_prem_loss", 0.35) or 0.35)
        price = float(t.get("price", 0) or req.get("price", 0) or 0)
        try:
            import vcp_tracker as V
            if price <= 0:
                try:
                    import yfinance as yf
                    price = float(yf.Ticker(ticker).fast_info["lastPrice"])
                except Exception:  # noqa: BLE001
                    price = 0.0
            o = V.select_call(ticker, price, 30, 45)
            upd = V.recompute_with_option(t or {"ticker": ticker, "price": price},
                                          o, account, mrp, mpl, size_factor)
            self._send(200, json.dumps({"ticker": ticker, "price": price, **upd},
                                       default=lambda x: None))
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)[-800:]}))

    def _refresh(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        mode = q.get("mode", ["best"])[0]
        account = q.get("account", ["0"])[0]
        if mode not in ("best", "weighted"):
            mode = "best"
        if not _lock.acquire(blocking=False):
            return self._send(409, json.dumps({"error": "a refresh is already running"}))
        try:
            sys.stderr.write(f"  [refresh] running scanner (mode={mode}, account={account}) ...\n")
            body = run_scan(mode, account)
            self._send(200, body)
            sys.stderr.write("  [refresh] done.\n")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)[-1500:]}))
        finally:
            _lock.release()

    def _analyze(self):
        """On-demand full analysis of one arbitrary ticker."""
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception:  # noqa: BLE001
            req = {}
        ticker = (req.get("ticker") or "").upper().strip()
        if not ticker:
            return self._send(400, json.dumps({"error": "ticker required"}))
        account = float(req.get("account", 0) or 0)
        risk = req.get("risk") or {}
        mrp = float(risk.get("max_risk_pct", 1.5) or 1.5)
        mpl = float(risk.get("max_prem_loss", 0.35) or 0.35)
        # reuse the last scan's market-level event risk so the lookup matches the board
        event = None
        try:
            with open(os.path.join(ROOT, "data.js")) as fh:
                js = fh.read()
            d = json.loads(js[js.index("{"):js.rstrip().rstrip(";").rindex("}") + 1])
            event = d.get("event_risk")
        except Exception:  # noqa: BLE001
            pass
        if not _lock.acquire(blocking=False):
            return self._send(409, json.dumps({"error": "scanner busy, try again"}))
        try:
            import vcp_tracker as V
            res = V.analyze_ticker(ticker, account, mrp, mpl, event=event)
            self._send(200, json.dumps(res, default=lambda x: None))
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(exc)[-800:]}))
        finally:
            _lock.release()

    def log_message(self, *a):  # quiet
        pass


def _uoa_loop():
    """Background UOA auto-refresh: keeps the put-flow banner current without
    anyone pressing the button. Staggered start to avoid colliding with the
    first pattern scan's API burst."""
    time.sleep(120)
    while True:
        try:
            if _ULOCK.acquire(blocking=False):
                try:
                    import uoa
                    uoa.scan()
                    print("  [uoa-loop] refreshed", file=sys.stderr)
                finally:
                    _ULOCK.release()
        except Exception as exc:  # noqa: BLE001
            print(f"  [uoa-loop] {exc}", file=sys.stderr)
        time.sleep(max(UOA_INTERVAL, 600))


def _scan_loop():
    """Optional server-side scan loop (cloud): keeps data fresh without a browser."""
    while True:
        try:
            if _lock.acquire(blocking=False):
                try:
                    run_scan(SCAN_MODE, SCAN_ACCOUNT)
                    print("  [scan-loop] refreshed", file=sys.stderr)
                finally:
                    _lock.release()
        except Exception as exc:  # noqa: BLE001
            print(f"  [scan-loop] {exc}", file=sys.stderr)
        time.sleep(max(SCAN_INTERVAL, 60))


def main():
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    if SCAN_INTERVAL > 0:
        print(f"  server-side scan loop every {SCAN_INTERVAL}s", file=sys.stderr)
        threading.Thread(target=_scan_loop, daemon=True).start()
    if UOA_INTERVAL > 0:
        print(f"  UOA auto-scan every {UOA_INTERVAL}s", file=sys.stderr)
        threading.Thread(target=_uoa_loop, daemon=True).start()
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as srv:
        shown = "127.0.0.1" if HOST in ("127.0.0.1", "0.0.0.0") else HOST
        url = f"http://{shown}:{PORT}/"
        print(f"VCP dashboard -> {url}   (Ctrl-C to stop)"
              + (f"   [auth: {DASH_USER}]" if DASH_USER and DASH_PASS else "   [no auth]"))
        if OPEN and HOST == "127.0.0.1":
            threading.Timer(0.7, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
