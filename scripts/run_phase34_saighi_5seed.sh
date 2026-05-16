#!/bin/bash
# 5-seed Phase 3+4 integration sweep with Saighi A_k self-inhibition enabled.
# Verification of report 034's seed-1 result across the canonical seed set.
# See STATUS.md blocker #2 and reports/034_saighi_ak_seed1_prototype.md.
set -u
cd /Users/dypatterson/Desktop/Neuro-AI
export PYTHONPATH=/Users/dypatterson/Desktop/Neuro-AI/src

SEEDS=(17 11 23 1 2)
LOG_ROOT=reports/phase34_saighi_5seed
mkdir -p "$LOG_ROOT"

for s in "${SEEDS[@]}"; do
  OUT="reports/phase34_saighi_seed${s}"
  LOG="${LOG_ROOT}/seed${s}.log"
  echo "=== seed ${s} -> ${OUT} ===" | tee -a "${LOG_ROOT}/run.log"
  date | tee -a "${LOG_ROOT}/run.log"
  mkdir -p "${OUT}"
  .venv/bin/python experiments/19_phase34_integrated.py \
    --updater-kind hebbian \
    --seed "${s}" \
    --success-threshold 0.3 \
    --death-threshold 0.05 \
    --death-window 10 \
    --inhibition-gain 0.01 \
    --inhibition-decay 0.0 \
    --output-dir "${OUT}" \
    >"${LOG}" 2>&1
  rc=$?
  echo "  seed ${s} exit=${rc}" | tee -a "${LOG_ROOT}/run.log"
  date | tee -a "${LOG_ROOT}/run.log"
done

echo "ALL DONE" | tee -a "${LOG_ROOT}/run.log"
