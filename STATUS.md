# Project STATUS

**Last updated:** 2026-05-14 (post-audit, supersedes report 026)

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

## Last verified result

**[Report 026](reports/026_phase4_verification_design_spec.md)** —
rt=0.85, drift=0.30, frozen Phase 3c codebook + synthetic drift, β=10,
5 seeds {17, 11, 23, 1, 2}.

- Δ Recall@10 at step 2000: **+0.010, CI [+0.001, +0.022], 5/5 ≥ 0** ✓
- Δ cap-coverage @ τ=0.5 at step 2000: -0.004, CI [-0.030, +0.022] ✗
- Random-codebook control: 0 candidates, Δ = 0.000 exactly ✓

This tests Phase 4 in isolation with synthetic drift — not the
architecturally-intended Phase 3+4 concurrency.

---

## Active blockers / must-do before Phase 4 can graduate

| # | Item | Why | Source |
| - | --- | --- | --- |
| 1 | 5-seed Phase 3+4 integration (exp 19) with active drift | Architecturally-intended test never multi-seed run; exp 19's prior run was single-seed, no drift, Hebbian fired 1.8% | Report 027 §IV Tier 1 #1 |
| 2 | Fix death mechanism settings, rerun rt=0.85 5-seed | Death mechanism vacuous at current `death_window=1000`+`death_threshold=0.005`; zero patterns died in any session run | Report 026 §Pattern death |
| 3 | Δcap-coverage second headline | One of two design-spec headlines; not verified, per-seed variance 5× larger than ΔR@K | Report 026 §Δcap-coverage |
| 4 | Diagnostic-actuator dynamic-form session | Named "next major architectural threshold" 2026-05-09, never held | [2026-05-09 paper synthesis](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md) |

---

## Audits passed / failed for current phase

| Audit | Status | Evidence |
| --- | --- | --- |
| FEP / anti-homunculus on Phase 4 mechanisms | Passed | Report 026 §FEP audit |
| Random-codebook control | Passed | Report 026 |
| Multi-seed Phase 3+4 integration | **Not run** | Phase 3+4 audit |
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
2. [reports/027_full_repo_audit_synthesis.md](reports/027_full_repo_audit_synthesis.md) — full audit.
3. [reports/026_phase4_verification_design_spec.md](reports/026_phase4_verification_design_spec.md) — current verified state.
4. [notes/emergent-codebook/phase-4-checklist.md](notes/emergent-codebook/phase-4-checklist.md) — exit criteria as line items.
5. [notes/emergent-codebook/phase-4-unified-design.md](notes/emergent-codebook/phase-4-unified-design.md) — original design spec.
6. [CLAUDE.md](CLAUDE.md) — agent working rules.

Anything older than report 022 is context, not load-bearing for current Phase 4 work.

---

## Update rule for this file

- Update at the end of every working session.
- Promote a "blocker" to "done" only when there is a report with multi-seed CI evidence.
- If a session walks back something currently in this file, that walk-back is the **first** edit of the session.
