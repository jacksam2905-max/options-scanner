#!/bin/bash
# Keep the Render free-tier instance warm so it doesn't cold-start (30-60s wake)
# when you open the dashboard. Pings the unauthenticated /healthz. Scheduled every
# 10 min, ~8am-4pm CT weekdays by com.jacob.vcptracker-keepalive.plist.
URL="https://vcp-scanner-3txh.onrender.com/healthz"
LOG="/Users/jacob/Code/VCP tracker/reports/keepalive.log"
mkdir -p "$(dirname "$LOG")"

# Only keep warm during usage hours (Mon-Fri 08:00-16:59 CT). Outside the window
# do nothing, so Render sleeps nights/weekends and stays within the free-tier
# hour budget. (This Mac is on CT; %u = 1..7 Mon..Sun, %H = local hour.)
dow=$(date +%u); hour=$(date +%H)
if [ "$dow" -gt 5 ] || [ "$hour" -lt 8 ] || [ "$hour" -gt 16 ]; then
    exit 0
fi
code=$(curl -s -m 40 -o /dev/null -w "%{http_code}" "$URL")
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] healthz -> $code" >> "$LOG"
# keep the log from growing forever (last 500 lines)
tail -n 500 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
