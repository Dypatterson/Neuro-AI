#!/usr/bin/env bash
# Local cue-regime profile harness.
#
# This is intentionally NOT the cross-seed cue-regime sweep. Local runs are
# for timing and debugging one seed × one cue-regime cell only; the full
# 10-seed grid runs in Colab so CUDA timing, Drive snapshots, and resume
# behavior match the real execution environment.
#
# Defaults mirror the current Colab entry-point cell/knobs, with reduced
# n_cues for local smoke timing:
#   seed 17, binding_noise_std=0.05, content_distortion=0.0, n_cues=20,
#   beta=10, gamma=0.5, K=1.
#
# Usage:
#   bash scripts/run_cue_regime_sweep.sh
#
# Optional overrides:
#   SNAP=... BNS=0.10 CD=0.4 N_CUES=50 DEVICE=cpu bash scripts/run_cue_regime_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP="${SNAP:-reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt}"
OUT_ROOT="${OUT_ROOT:-reports/phase5_cue_regime_profile}"
SEED="${SEED:-17}"
BNS="${BNS:-0.05}"
CD="${CD:-0.0}"
N_CUES="${N_CUES:-20}"
DEVICE="${DEVICE:-cpu}"

BETA="10.0"
GAMMA="0.5"
K_MAIN="1"

mkdir -p "$OUT_ROOT"
OUT="$OUT_ROOT/seed${SEED}_bns${BNS}_cd${CD}_n${N_CUES}.json"

echo "[profile] local cue-regime harness"
echo "  snapshot: $SNAP"
echo "  output:   $OUT"
echo "  cell:     binding_noise_std=$BNS content_distortion=$CD"
echo "  fixed:    beta=$BETA gamma=$GAMMA K=$K_MAIN n_cues=$N_CUES device=$DEVICE"
echo

SECONDS=0
PYTHONPATH=src .venv/bin/python scripts/phase5_frozen_snapshot_audit.py \
  --snapshot "$SNAP" \
  --output "$OUT" \
  --cue-regime-sweep \
  --cue-regime-binding-noise "$BNS" \
  --cue-regime-content-distortion "$CD" \
  --cue-regime-beta "$BETA" \
  --cue-regime-n-cues "$N_CUES" \
  --headline-k-main "$K_MAIN" \
  --headline-gamma "$GAMMA" \
  --device "$DEVICE"

echo
echo "[profile] completed in ${SECONDS}s"
echo "[profile] full 10-seed cue-regime sweep should be run from Colab, not this script."
