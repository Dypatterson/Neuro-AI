#!/usr/bin/env bash
# Fetch the SCAN benchmark splits used by the Bet-B discriminating regime (2026-06-15).
# Data is gitignored (re-downloadable); this script is the tracked, reproducible source.
# Usage:  bash scripts/fetch_scan.sh
set -euo pipefail

BASE="https://raw.githubusercontent.com/brendenlake/SCAN/master"
DEST="data/scan"
mkdir -p "$DEST"

FILES=(
  "add_prim_split/tasks_train_addprim_jump.txt"   # train: jump ONLY isolated, 0 compositions
  "add_prim_split/tasks_test_addprim_jump.txt"    # test:  all compose jump (the discriminating split)
  "simple_split/tasks_train_simple.txt"           # random-split train (sanity ceiling)
  "simple_split/tasks_test_simple.txt"            # random-split test  (sanity ceiling)
)

for f in "${FILES[@]}"; do
  out="$DEST/${f##*/}"
  curl -fsSL --max-time 60 "$BASE/$f" -o "$out"
  echo "fetched ${f##*/} ($(wc -l < "$out") lines)"
done

echo "SCAN data in $DEST/ — see notes/betb-scan-discriminating-regime-precommit.md"
