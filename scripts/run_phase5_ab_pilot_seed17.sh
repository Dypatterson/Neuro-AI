#!/usr/bin/env bash
# Phase 5 A+B death-mechanism dynamic-form: 1-seed pilot retrain (seed 17).
#
# Pre-committed knobs (binding per
# notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md
# and reports/phase5_ab_calibration.json):
#   alpha_anti           = 1.0    # natural unit scale of -log(d_eff)
#   coverage_lambda      = 1.0    # formal Candidate A modulation
#   coverage_ema_rate    = 0.01   # EMA halflife ≈ 100 steps
#   repulsion_step_size  = 100.0  # one-shot calibration: smallest step
#                                 # with median ΔDeff ≥ 0.05 in collapsed regime
#
# Mechanism-validity criteria (NOT graduation):
#   - d_eff ≥ 25 at step 1800 (W=4 substrate)
#   - K-branch state_divergence within 30% of pre-death (vs report 043)
#   - Phase 4 D1 non-regression on this seed (Δms_w3 ≤ -0.5 vs report 038)
#
# If 1-seed passes the validity gates, ship to Colab for n=10 retrain.
# If 1-seed fails any validity gate, that is a falsification result —
# NOT an invitation to re-tune α/λ/step_size. Redesign session first.
#
# Usage:
#   bash scripts/run_phase5_ab_pilot_seed17.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="reports/phase5_ab_pilot_seed17"
mkdir -p "$OUT_DIR"

PYTHONPATH=src .venv/bin/python experiments/19_phase34_integrated.py \
    --seed 17 \
    --dim 4096 \
    --n-cues 1800 \
    --checkpoint-every 300 \
    --snapshot-steps 500,1500,1700,1800 \
    --output-dir "$OUT_DIR" \
    --updater-kind hebbian \
    --success-threshold 0.3 \
    --alpha-anti 1.0 \
    --coverage-lambda 1.0 \
    --coverage-ema-rate 0.01 \
    --repulsion-step-size 100.0 \
    "$@"

echo
echo "==> A+B pilot complete. Next: run consolidation-geometry diagnostic"
echo "    on the new snapshots and compare against the pre-death (step 500)"
echo "    and prior post-death (reports/phase5_snapshots_local/seed17/) d_eff."
echo
echo "    PYTHONPATH=src .venv/bin/python scripts/consolidation_geometry_diagnostic.py \\"
echo "      --snapshot $OUT_DIR/snapshots/seed17/phase3_phase4_w4_step1800.pt \\"
echo "      --output   $OUT_DIR/d_eff_step1800.json"
