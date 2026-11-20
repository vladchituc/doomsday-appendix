#!/bin/bash
# Regenerate every figure and results table from the included data (a few minutes on a laptop; most of it is the
# Figure 7 sweep, which is cached in results/singularity_sweep.npz; delete that file to recompute the sweep).
# Optional, slower: python3 scripts/csn_tests.py [reps]   (CSN goodness-of-fit bootstraps)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p figures results
run() { echo; echo "=== $*"; "$PY" "$@"; }
run scripts/fig1_children_per_woman.py
run scripts/fig2_births_halved.py
run scripts/davidson_model.py
run scripts/fig3_compute_per_researcher.py
run scripts/fig4_tail_fits.py
run scripts/fig5_rising_beta.py
run scripts/fig6_mechanism.py
run scripts/fig7_singularity_map.py
run speedrun/fig8_returns_to_research.py   # plots the committed speedrun estimates; bash speedrun/run.sh refits them
run scripts/law_comparison.py
run scripts/search_simulation.py
run scripts/check_numbers.py
