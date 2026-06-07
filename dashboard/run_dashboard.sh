#!/bin/bash
# Launch the dashboard server with options-data credentials loaded, so the
# Refresh + Get-option buttons use Tradier (chains/greeks even after hours).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
[ -f tradier_creds.sh ] && source tradier_creds.sh
exec python3 dashboard/serve.py "$@"
