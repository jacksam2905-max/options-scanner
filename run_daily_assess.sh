#!/bin/bash
# Daily SWING self-assessment. Grades the cohort of recommendations the scanner
# would have made ~25 trading days ago (resolved by now), appends one line to
# backtest_journal.md, and prints an analysis + proposals (never auto-applied).
# Launched weekdays 15:15 CT (= 4:15pm ET) by com.jacob.vcptracker-assess.plist.
set -uo pipefail

DIR="/Users/jacob/Code/VCP tracker"
PY="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
cd "$DIR"
mkdir -p reports

# Creds (token kept out of git): local tradier_creds.sh, else the shared
# gitignored .env from the sibling intraday project.
if [ -f "$DIR/tradier_creds.sh" ]; then
    source "$DIR/tradier_creds.sh"
elif [ -f "$DIR/../Candle/.env" ]; then
    set -a; source "$DIR/../Candle/.env"; set +a
fi

DATE="$(date +%Y-%m-%d)"
LOG="reports/assess_${DATE}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] daily assess start" >> "$LOG"

"$PY" daily_assess.py >> "$LOG" 2>&1 || echo "[$(date)] assess FAILED (see above)" >> "$LOG"

# Preserve the journal line in git history WITHOUT pushing (no daily Render
# redeploy). Scoped to the journal file only, so it never touches other work.
git add backtest_journal.md >> "$LOG" 2>&1 \
  && git commit -m "journal: daily assess $DATE" >> "$LOG" 2>&1 || true

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] daily assess done" >> "$LOG"
