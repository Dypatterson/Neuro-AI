# Project STATUS

**Last updated:** 2026-05-14 (post-report-029 — st03 variant exposes drift-real Phase 4)

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

---

## Active phase

**Phase 4 — verification in progress.** One design-spec headline verified
(ΔR@K), one not (Δcap-coverage). Architecturally-intended Phase 3+4
integration test has not been multi-seed run.

---

## Current headline metric (per [phase-4-unified-design.md:276-282](notes/emergent-codebook/phase-4-unified-design.md))

> **Recall@K AND cap-coverage on masked-token contextual completion,
> measured before vs. after N consolidation cycles, with active codebook
> drift between cycles.**

Both metrics. Not one, not top1. Top1 is a drill-down.

## Required controls (per [phase-4-unified-design.md:318-323](notes/emergent-codebook/phase-4-unified-design.md))

- No-replay baseline — present in every exp 18 / exp 19 run ✓
- Random-codebook control — verified once at rt=0.85, drift=0.30 (report 026) ✓

## Last verified results

**[Report 026](reports/026_phase4_verification_design_spec.md)** (Phase 4 in
isolation, frozen Phase 3c codebook + synthetic drift, rt=0.85, drift=0.30,
β=10, 5 seeds {17, 11, 23, 1, 2}):
- Δ Recall@10 at step 2000: **+0.010, CI [+0.001, +0.022], 5/5 ≥ 0** ✓
- Δ cap-coverage @ τ=0.5 at step 2000: -0.004, CI [-0.030, +0.022] ✗
- Random-codebook control: 0 candidates, Δ = 0.000 exactly ✓

**[Report 028](reports/028_phase34_integration_5seed.md)** (Phase 3+4
integration via online Hebbian @ success_threshold=0.5, 5 seeds, Colab A100):
- Δ Recall@10 at step 1500: +0.004, CI (t,4df) [-0.012, +0.021] ✗ (includes 0)
- The Hebbian updater fired on ~1% of cues, codebook drift was effectively
  zero (mean 1.6e-6). The test did not exercise drift; superseded by 029.

**[Report 029](reports/029_phase34_integration_st03.md)** (Phase 3+4
integration via online Hebbian @ success_threshold=0.3, 5 seeds, Colab A100):
- Δ Recall@10 at step 1500: **+0.0145, CI (t,4df) [-0.005, +0.034]**, 4/5
  seeds ≥ 0; CI *barely* includes 0 due to recurring seed-23 outlier.
  Pooled-Wilson Δ = +0.0144 (388/1111 vs 372/1111). Monotonically growing
  trajectory matching report-026 shape. ✓ (mean exceeds report 026's +0.010)
- Δ cap-coverage @ τ=0.5: +0.006, CI [-0.057, +0.069], 3/5 ≥ 0 ✗
- **Δtop1: −0.026, 4/5 NEGATIVE** ✗ — Phase 4 *hurts* rank-1 when codebook
  drifts. Mechanism: discovered patterns stored with `source_windows=None`
  are never re-encoded against the current codebook
  ([exp19:355-356](experiments/19_phase34_integrated.py), [reencoding.py](src/energy_memory/phase34/reencoding.py)),
  so they go stale as Hebbian updates accumulate.
- Hebbian fired on 14.5% of cues, mean drift 1.2e-5 (15× / 8× the 028 run).
- Strongest support for stale-discovered-patterns hypothesis: seed 1 had
  12 consolidations and largest C-B advantage (+0.031); seeds with 200+
  consolidations had C-B near zero or negative.

**[D1 5-seed aggregation](reports/d1_metastable_5seed.json)** from existing
report-026 JSON checkpoints (not re-run, drilling into existing data):
- W=3 meta-stable rate: baseline 0.67±0.19 → phase4 0.16±0.21; **Δ = -0.51,
  per-seed std = 0.038** (every seed shows ~-0.5 reduction).
- W=4 meta-stable rate: Δ = -0.57±0.21.
- W=2: ≈0 in both (already committed).
- Read: Phase 4 makes higher-scale retrievals significantly more decisive.
  Unreported in 026; load-bearing drill-down.

---

## Active blockers / must-do before Phase 4 can graduate

| # | Item | Why | Source |
| - | --- | --- | --- |
| 1 | 5-seed Phase 3+4 integration with active drift, ΔR@10 verified | **Conditionally closed** by report 029: ΔR@10 = +0.0145, 4/5 positive, exceeds report 026. Per-seed CI barely includes 0 due to seed-23 outlier. Fully closes once blocker #6 (reencoding fix) is in. | Report 029 |
| 6 | **NEW: Fix stale-discovered-patterns reencoding gap** | Discovered patterns are stored with `source_windows=None` → never re-encoded against the current codebook → go stale under drift. Causes 4/5 negative Δtop1 in report 029. | Report 029 §Top1 regression |
| 2 | Fix death mechanism settings, rerun rt=0.85 5-seed | Death mechanism vacuous at current `death_window=1000`+`death_threshold=0.005`; zero patterns died in any session run | Report 026 §Pattern death |
| 3 | Δcap-coverage second headline | One of two design-spec headlines; not verified in 026, 028, or 029; per-seed variance 5–10× larger than ΔR@K | Reports 026, 028, 029 |
| 4 | Diagnostic-actuator dynamic-form session | Named "next major architectural threshold" 2026-05-09, never held | [2026-05-09 paper synthesis](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md) |
| 5 | Seed-23 diagnostic | **Three** independent runs (026, 028, 029) all identify seed 23 as the cap_t05 / R@10 outlier. Idiosyncratic geometry, not noise. Discipline problem — continued tolerance without diagnosis is the bottleneck on tightening CIs. | Reports 026, 028, 029 |

---

## Audits passed / failed for current phase

| Audit | Status | Evidence |
| --- | --- | --- |
| FEP / anti-homunculus on Phase 4 mechanisms | Passed | Report 026 §FEP audit |
| Random-codebook control | Passed | Report 026 |
| Multi-seed Phase 3+4 integration (drift-effective) | **Conditionally passed** — report 029 (st=0.3) produces real drift; ΔR@10 4/5 positive, mean +0.014; top1 regresses due to stale-discovered-patterns gap (blocker #6) | Report 029 |
| Shuffled-token control | Not run (Phase 3 discipline; Phase 4 design doesn't require) | — |
| Pattern death (architectural component) | **Not exercised** in any session run | Report 026 §Pattern death |
| Entropy exit criterion ("repeated paths lower entropy") | **Failed** at W=4 | Report 026 §Engagement / entropy |

---

## Live operational policies

- **HAM regime split** (from report 022): β=30 + summed scores for retrieval; β=10 + HAM-arithmetic for replay diagnostics.
- **Online error-driven codebook updates: BANNED.** Use Hebbian for runtime, error-driven only in batch offline passes.
- **Drift sources sanctioned by design** (phase-4-unified-design.md:296-309): periodic batch retrains, online Hebbian reinforcement, or simulated synthetic perturbation.

---

## Pre-phase commitments still open (deferred but documented)

These were specified as required before claiming a phase done; they have not been.

- Consolidation-geometry regime classifier (d̄, d_eff per atom) — pre-Phase-3 spec, not built. [consolidation-geometry-diagnostic.md](notes/emergent-codebook/consolidation-geometry-diagnostic.md)
- Empirical θ′(β) calibration spike — recommended pre-Phase-3 (2026-05-09), not done.
- High-leverage brainstorm idea 5 (frequency-weighted Benna-Fusi α) — never built; named as the key experiment for the architecture's "compression → abstraction" claim. [brainstorm doc](brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md)

If you graduate Phase 4 without these, document why.

---

## Reading order for someone catching up

1. This file (STATUS.md).
2. [reports/029_phase34_integration_st03.md](reports/029_phase34_integration_st03.md) — latest, ΔR@10 conditionally verified under real drift; surfaces stale-discovered-patterns gap.
3. [reports/028_phase34_integration_5seed.md](reports/028_phase34_integration_5seed.md) — prior st=0.5 run; drift didn't fire (superseded by 029, but useful context).
4. [reports/027_full_repo_audit_synthesis.md](reports/027_full_repo_audit_synthesis.md) — full audit.
5. [reports/026_phase4_verification_design_spec.md](reports/026_phase4_verification_design_spec.md) — Phase 4 isolation verified.
6. [reports/d1_metastable_5seed.json](reports/d1_metastable_5seed.json) — D1 drill-down (Phase 4 collapses W=3/W=4 meta-stable rate by ~50pp).
7. [notes/emergent-codebook/phase-4-checklist.md](notes/emergent-codebook/phase-4-checklist.md) — exit criteria as line items.
8. [notes/emergent-codebook/phase-4-unified-design.md](notes/emergent-codebook/phase-4-unified-design.md) — original design spec.
9. [CLAUDE.md](CLAUDE.md) — agent working rules.

Anything older than report 022 is context, not load-bearing for current Phase 4 work.

---

## Update rule for this file

- Update at the end of every working session.
- Promote a "blocker" to "done" only when there is a report with multi-seed CI evidence.
- If a session walks back something currently in this file, that walk-back is the **first** edit of the session.
