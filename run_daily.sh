#!/bin/bash
# Daily VCP + options watchlist run. Pulls FRESH data and writes a dated
# report + CSV into ./reports, plus a 'latest' copy. Intended to be launched
# by the launchd agent before the US market open, but can be run manually.
set -euo pipefail

DIR="/Users/jacob/Code/VCP tracker"
PY="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
cd "$DIR"
mkdir -p reports

# Optional: load Tradier options-data token (chains/greeks even after hours).
[ -f "$DIR/tradier_creds.sh" ] && source "$DIR/tradier_creds.sh"

DATE="$(date +%Y-%m-%d)"
TS="$(date +'%Y-%m-%d %H:%M:%S %Z')"
TXT="reports/watchlist_${DATE}.txt"
CSV="reports/watchlist_${DATE}.csv"
LOG="reports/run_${DATE}.log"

echo "[$TS] starting daily VCP scan" >> "$LOG"

# stdout -> dated report; stderr (progress/errors) -> dated log.
if "$PY" vcp_tracker.py --csv "$CSV" >"$TXT" 2>>"$LOG"; then
    cp "$TXT" reports/latest.txt
    [ -f "$CSV" ] && cp "$CSV" reports/latest.csv
    echo "[$TS] done -> $TXT" >> "$LOG"
else
    echo "[$TS] FAILED (see log above)" >> "$LOG"
    exit 1
fi
