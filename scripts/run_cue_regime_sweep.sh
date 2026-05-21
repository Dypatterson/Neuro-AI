#!/usr/bin/env bash
# Tier-1 cue-regime sweep against the existing seed-17 A1' substrate.
#
# Per the 2026-05-20 brainstorm + research E §3.1: the four 2026-05-20
# design notes are all substrate-side fixes. The cue regime (binding-
# noise-std × content-distortion) was never swept. This sweep tells us
# whether the β failure is substrate-side (the substrate has no f_i
# variance to expose) OR operating-point-side (the cue regime is at a
# corner where binding decomposition collapses).
#
# Pre-committed pass threshold (binding): if std(f_i) > 0.02 for any
# cue cell across the 36-cell grid, β is rescued by cue choice.
#
# Wall-time: ~20s per cell × 36 ≈ 12 min total at n_cues=20 (smoke
# headline). Sequential because exp 40 doesn't take parallel input.
#
# Usage:
#   bash scripts/run_cue_regime_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP="reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt"
OUT_ROOT="reports/phase5_a1prime_pilot_seed17/cue_sweep"
mkdir -p "$OUT_ROOT"

# 4 × 6 = 24-cell grid (smaller than 36 for wall-time; covers the
# space) of binding-noise-std × content-distortion.
BNS_VALUES=("0.01" "0.05" "0.10" "0.20")
CD_VALUES=("0.0" "0.2" "0.4" "0.6" "0.8" "1.0")

echo "cell,binding_noise_std,content_distortion,f_mean,f_std,f_min,f_max" > "$OUT_ROOT/sweep_summary.csv"

CELL=0
for BNS in "${BNS_VALUES[@]}"; do
  for CD in "${CD_VALUES[@]}"; do
    CELL=$((CELL+1))
    CELL_DIR="$OUT_ROOT/cell_bns${BNS}_cd${CD}"
    echo "[$CELL/24] binding_noise_std=$BNS, content_distortion=$CD"
    PYTHONPATH=src .venv/bin/python experiments/40_phase5_branching.py \
      --mode headline \
      --substrate-snapshot "$SNAP" \
      --output-dir "$CELL_DIR" \
      --seed 17 \
      --n-cues 20 \
      --binding-noise-std "$BNS" \
      --content-distortion "$CD" \
      2>&1 | tail -3

    # Extract the β fidelity stats line (logged by exp 40 when β runs).
    # Format: "[β] full-substrate fidelities computed: N=..., mean(f)=..., std(f)=..., min=..., max=..."
    JSON_FILE="$CELL_DIR/phase5_headline_seed17.json"
    if [ -f "$JSON_FILE" ]; then
      # We don't currently log per-cell f_i stats; just record that
      # the cell ran. The post-processing aggregator pulls stats out.
      echo "$CELL,$BNS,$CD,run,run,run,run" >> "$OUT_ROOT/sweep_summary.csv"
    fi
  done
done

echo
echo "Sweep complete. Aggregate via:"
echo "  PYTHONPATH=src .venv/bin/python scripts/aggregate_cue_sweep.py \\"
echo "    --sweep-root $OUT_ROOT \\"
echo "    --output $OUT_ROOT/sweep_aggregate.json"
