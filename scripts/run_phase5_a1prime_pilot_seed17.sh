#!/usr/bin/env bash
# Phase 5 A1' (max-over-others r_inst measure) 1-seed pilot retrain (seed 17).
#
# A1' (notes/notes/2026-05-20-r-inst-measure-dynamic-form.md, anti-
# homunculus PASS): replaces the Gram-row mean-RMS proxy in
# _coverage_redundancy_instantaneous with max-over-others. The patch
# applies on top of A+B+step3 + A1 — A1's geometric init at add-time
# now draws from a measurement that correctly identifies sparse
# duplicates.
#
# No new tunables. All A+B+step3 knobs unchanged.
#
# Six pre-committed mechanism-validity criteria (per the A1' design
# note, identical to the A1 criteria):
#   1. Top-8 schema pairwise FHRR similarity in band [0.10, 0.60] at n=5
#   2. K-branch state_divergence within 30% of pre-death at n=5
#   3. d_eff ≥ 25 at step 1800
#   4. Phase 4 D1 non-regression: Δms_w3 ≤ -0.5 at n=10
#   5. coverage_lambda/coverage_ema_rate/alpha_anti/repulsion_step_size
#      all unchanged (code-level)
#   6. `max` is the only reduction change (code-level)
#
# Usage:
#   bash scripts/run_phase5_a1prime_pilot_seed17.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="reports/phase5_a1prime_pilot_seed17"
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
echo "==> A1' pilot complete. Diagnostics:"
echo "  1. Top-8 schema diversity (criterion #1):"
echo "     PYTHONPATH=src .venv/bin/python scripts/inspect_topk_diversity.py \\"
echo "       --snapshot $OUT_DIR/snapshots/phase3_phase4_w4_step1800.pt \\"
echo "       --output   $OUT_DIR/topk_diversity_w4_step1800.json"
echo
echo "  2. Substrate d_eff (criterion #3):"
echo "     PYTHONPATH=src .venv/bin/python scripts/analyze_phase5_ab_pilot.py \\"
echo "       --pilot-dir $OUT_DIR --output $OUT_DIR/analysis.json"
echo
echo "  3. K-branch state_divergence (criterion #2):"
echo "     PYTHONPATH=src .venv/bin/python experiments/40_phase5_branching.py \\"
echo "       --mode headline \\"
echo "       --substrate-snapshot $OUT_DIR/snapshots/phase3_phase4_w4_step1800.pt \\"
echo "       --output-dir $OUT_DIR/branch_diag_w4_step1800 \\"
echo "       --seed 17"
echo
