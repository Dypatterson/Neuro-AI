---
name: landscape-sweep-diagnostic-precommit
date: 2026-05-30
project: personal-ai
phase: Phase 3 — Growing Codebook (Frame B)
status: PRECOMMIT — written BEFORE results so interpretation is fixed; user-authorized 2026-05-30
parent: reports/117_frameb_leveldid_feasibility_consolidation_variance.md
tags: [notes, subject/phase-3, subject/continual-learning, diagnostic]
---

# Landscape-size diagnostic — pre-committed read protocol

**Drill-down, NOT a graduation run.** Authorized by the user 2026-05-30 as a
labeled diagnostic. The Gate-0 precommit's "no operating-point sweep" rule
(`2026-05-28-gate0-frame-a-valid-control-precommit.md:184`) governs the
*graduation* run; this is an explicitly-labeled diagnostic with sign-off, and it
does **not** set or pin a new operating point — it only tests whether the
consolidation-injected variance is *reducible at source*.

## The one question this answers

[Report 117 §3](../../reports/117_frameb_leveldid_feasibility_consolidation_variance.md)
found that **consolidation injects the per-seed variance**: at the current
landscape_size=64, the consolidated real arm has σ_A=0.157 vs the frozen real arm
σ_C=0.069 (2.3×). Hypothesis: the variance comes from the **tiny 64-window
landscape** — each seed memorizes a different 64-of-~220k-window landscape, and
consolidation amplifies that draw-to-draw difference. **Does a larger memorized
landscape collapse σ_A toward the frozen floor (~0.07)?**

- If **yes** → the variance is reducible; a larger-landscape operating point makes
  *any* downstream estimand (level-DiD or slope) materially cheaper, and the next
  step is a powered run at the better landscape (with op-point sign-off).
- If **no** → the variance is irreducible at this architecture; corpus-specificity
  is structurally hard to detect here → Frame B mechanism / operating-point rethink.

## Design (zero new code)

3 runs of the **existing** `gate0_frame_a.py` at the recovered op point, varying
only `--landscape-size ∈ {64, 256, 512}`, n=10 seeds (0..9), `--device cuda`.
**L=64 is the recovered run** (`reports/gate0_2026-05-28/`, σ_A=0.157) and is
re-run only as a same-environment reproducibility anchor (it MUST reproduce
σ_A≈0.157). So the new compute is **L=256 + L=512**. Per-cell retrieve work is
linear in L, so serial cost ≈ (1+4+8)=13× one consolidating arm-column ≈ **~1.8
GPU-hr** (the n=10 run was 506 s CUDA).

**Two equivalent runners** (the diagnostic is method-fixed; the runner is not):
- **Serial:** `scripts/colab_landscape_sweep_2026-05-30.ipynb` — full `gate0_frame_a.py`
  ×3 landscapes; read via `scripts/landscape_sweep_read.py`. ~1.8 GPU-hr.
- **Parallel (preferred):** `scripts/colab_landscape_sweep_parallel_2026-05-30.ipynb`
  + `scripts/landscape_sweep_parallel.py` — fans 3 L × 10 seeds out to 30 single-cell
  CUDA workers, running **only arms A + C** (all the σ read needs). **Byte-identical**
  to the serial path because cells are **order-independent** (verified 2026-05-30:
  arm-A recall is the same whether a seed runs fresh or after others), so per-seed
  fan-out cannot change any number. ~5–10 min on an idle A100. The σ read is the
  same thresholds; the harness emits `landscape_sweep_summary.json` directly.

## Pre-committed read (apply verbatim — no rationalizing)

Compute from each run's `gate0_summary.json`:
- **σ_A(L)** = sample SD over the 10 seeds of `per_seed_recall["A"]` (consolidated, real).
- **σ_C(L)** = sample SD of `per_seed_recall["C"]` (frozen, real) — the floor (~0.069, should be ~L-independent).
- **mean(A−C)(L)** = mean per-seed consolidation lift (the *signal* — must not collapse).

| condition | verdict | action |
|---|---|---|
| σ_A(512) ≤ ~0.10 (≥35% drop toward the ~0.07 floor) **AND** mean(A−C)(512) ≥ ~0.020 | **VARIANCE REDUCIBLE** | bigger landscape works; propose a powered run at the best L (level-DiD n drops ~424→~100–187, or revisit the slope). Needs op-point sign-off as a new headline point. |
| σ_A(512) ≥ ~0.13 (<20% drop) | **VARIANCE IRREDUCIBLE** | landscape is not the knob; the effect is structurally hard to detect at this architecture → Frame B mechanism / op-point rethink. |
| 0.10 < σ_A(512) < 0.13 (partial) | **PARTIAL** | read the full σ_A(L) curve; consider pushing L higher (1024) before deciding, or combine modest L↑ with modest n↑. |
| mean(A−C)(512) < ~0.015 at ANY L (signal collapses) | **CAPACITY WALL** | the bigger landscape exceeded Hopfield capacity at β=10 (metastable regime, deep-dive:291) and killed the consolidation signal — bigger L is NOT a usable fix even if σ dropped. Overrides the σ read. |

## Anti-homunculus

`landscape_size` is a measurement/operating-point parameter (how many windows
seed the Hopfield landscape) — **not a new mechanism**. No supervisor, no
metric-reads-then-acts branch; consolidation still fires on the local buffer-fill
event. **No anti-homunculus reviewer pass required** (measurement-only change).

## Guardrails

- Report numbers as returned; quote the `per_seed_recall` arrays. The σ read is a
  pure re-analysis of the emitted JSON — no new estimator.
- This diagnostic does NOT graduate any phase and does NOT pin a new op point; a
  positive result only *licenses proposing* a powered run at a new L (separate sign-off).
- If σ_A(64) does NOT reproduce ~0.157, STOP — environment/repro problem, re-derive.
