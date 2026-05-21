#!/usr/bin/env bash
# Phase 5 A1 (substrate-derived r_ema init) 1-seed pilot retrain (seed 17).
#
# A+B+step3 with A1 (notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md):
# r_ema[new] ← _coverage_redundancy_instantaneous(P ∪ {new}) at add-time,
# replacing the implicit `r_ema = 0` that left discovery-channel atoms
# unmodulated for ~100 steps (the EMA half-life). With A1, near-duplicate
# discovery atoms start at r_ema ≈ 1 and self-throttle immediately.
#
# All A+B+step3 knobs pre-committed (binding per the death-dynamic note).
# A1 introduces no new tunable parameters.
#
# Six pre-committed mechanism-validity criteria (per the A1 design note):
#   1. Top-8 schema pairwise FHRR similarity in band [0.10, 0.60] at n=5
#   2. K-branch state_divergence within 30% of pre-death at n=5
#   3. d_eff ≥ 25 at step 1800
#   4. Phase 4 D1 non-regression: Δms_w3 ≤ -0.5 at n=10
#   5. r_ema init done once at add (code-level — verified by audit)
#   6. coverage_ema_rate NOT retuned (still 0.01)
#
# If 1-seed passes criteria 1+3, ship to Colab for n=5 K-branch (criterion 2)
# and n=10 retrain for criterion 4.
#
# Usage:
#   bash scripts/run_phase5_a1_pilot_seed17.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="reports/phase5_a1_pilot_seed17"
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
echo "==> A1 pilot complete. Next steps:"
echo "  1. Substrate diagnostic:"
echo "     PYTHONPATH=src .venv/bin/python scripts/consolidation_geometry_diagnostic.py \\"
echo "       --snapshot $OUT_DIR/snapshots/seed17/phase3_phase4_w4_step1800.pt \\"
echo "       --output   $OUT_DIR/d_eff_step1800.json"
echo
echo "  2. K-branch divergence diagnostic (criterion #2 at n=1):"
echo "     PYTHONPATH=src .venv/bin/python experiments/40_phase5_branching.py \\"
echo "       --mode headline \\"
echo "       --snapshot $OUT_DIR/snapshots/seed17/phase3_phase4_w4_step1800.pt \\"
echo "       --output   $OUT_DIR/branch_diag_w4_step1800/phase5_headline_seed17.json"
echo
