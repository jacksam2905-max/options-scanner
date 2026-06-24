#!/bin/bash
# Weekly review + gated auto-tune. Saturdays 09:00 CT via
# com.jacob.vcptracker-weekly.plist. Analyzes the daily journal, validates any
# persistent proposal with backtest_lab, and (behind strict gates + smoke test)
# may auto-tune one bounded parameter and push to main. Kill switch: set
# WEEKLY_AUTOAPPLY=0 to make it report-only.
set -uo pipefail

DIR="/Users/jacob/Code/VCP tracker"
PY="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
cd "$DIR"
mkdir -p reports

if [ -f "$DIR/tradier_creds.sh" ]; then
    source "$DIR/tradier_creds.sh"
elif [ -f "$DIR/../Candle/.env" ]; then
    set -a; source "$DIR/../Candle/.env"; set +a
fi

DATE="$(date +%Y-%m-%d)"
LOG="reports/weekly_${DATE}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly review start" >> "$LOG"
"$PY" weekly_review.py >> "$LOG" 2>&1 || echo "[$(date)] weekly review FAILED" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] weekly review done" >> "$LOG"
