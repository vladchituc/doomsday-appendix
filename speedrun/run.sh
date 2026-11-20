#!/bin/bash
# Rebuild the speedrun estimate from the committed data (about three minutes on a laptop).
# To refresh the data from GitHub first:  python3 fetch_github.py && python3 build_series.py
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -f stable_table.npz ] || "$PY" jones_fit.py
"$PY" estimate_r.py "${1:-500}"
"$PY" fig8_returns_to_research.py
