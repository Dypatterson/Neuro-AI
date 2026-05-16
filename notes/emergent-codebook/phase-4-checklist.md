# Phase 4 exit checklist

Companion to [phase-4-unified-design.md](phase-4-unified-design.md). Every
component, exit criterion, drill-down, and control specified by the design
appears here as a line item with explicit status and evidence link.

**Phase 4 graduates when every "must-have" item is verified. Each
"verified" status requires a multi-seed report with CI evidence.**

Single-seed results are not "verified." Mechanism-validated runs are not
"verified." The bar is multi-seed + CI.

---

## Status legend

- ✅ **verified** — multi-seed + CI evidence in a report
- 🟨 **partial** — implemented and tested, but not multi-seed or with caveats
- ❌ **open** — required but no evidence yet
- ⚠️  **broken** — mechanism exists but isn't actually exercising as designed
- 🟦 **deferred** — explicitly noted in design as deferred (not blocking)

---

## A. Component implementations (per design spec §1–5)

| # | Component | Spec source | Implemented? | Tested? | Status | Evidence |
| -- | --- | --- | :---: | :---: | :---: | --- |
| A1 | Trajectory trace | design §1 | yes | yes | ✅ | [src/energy_memory/phase4/trajectory.py](../../src/energy_memory/phase4/trajectory.py) |
| A2 | Engagement metric (entropy-based) | design §2 | yes | yes | ✅ | [report 023](../../reports/023_drift_sweep_and_gate_audit.md) gate audit |
| A3 | Resolution metric | design §2 | yes | yes | ✅ | report 023 |
| A4 | Gate signal = eng × (1-res) | design §2 | yes | yes | ✅ | report 023 (gate fires 59% at threshold 0.05) |
| A5 | Replay store (bounded buffer) | design §3 | yes | yes | ✅ | [src/energy_memory/phase4/replay_loop.py](../../src/energy_memory/phase4/replay_loop.py) |
| A6 | Replay re-settle through current landscape | design §3 | yes | yes | ✅ | report 026 |
| A7 | Candidate emission (resolve_threshold) | design §3 | yes | yes | ✅ | report 024 (rt sweep) |
| A8 | Trace age + decay | design §3 | yes | partial | 🟨 | mechanism present; age distribution never aggregated |
| A9 | Benna-Fusi u-chain (Eq. 10/11) | design §4 | yes | yes | ✅ | report 026 §u_k drill-down |
| A10 | u_1 novelty input + retrieval reinforcement | design §4 | yes | yes | ✅ | report 026 |
| A11 | Effective strength weighted sum | design §4 | yes | yes | ✅ | mean_strength reported per checkpoint |
| A12 | Pattern death (strength < threshold for window) | design §4 | yes | **not exercised** | ⚠️ | mechanism FEP-clean; report 033 shows it cannot fire in n_cues=1500 at canonical config, and >99.9% of vocab atoms are never reinforced — naive tuning produces mass-death of unreached initial atoms, not the architecturally-intended stale-discovered-pattern purge. STATUS blocker #2 reshaped to scope decision. |
| A13 | SQ-HN sparse-update principle | design §5 | yes | not ablated | 🟨 | code respects sparsity; no ablation showing it matters |
| A14 | Re-encode stored patterns through current codebook | project plan exit | yes | yes (knob enabled) | 🟨 | `reencode_every=100`, contribution never ablated |

---

## B. Headline metrics (per design §"Headline metric for Phase 4")

The design names **two headlines**, not one. Phase 4 graduates when *both* are verified.

| # | Headline | Status | Evidence |
| -- | --- | :---: | --- |
| B1 | Δ Recall@K under active drift, multi-seed, CI disjoint from 0 | ✅ | [report 026](../../reports/026_phase4_verification_design_spec.md): +0.010 CI [+0.001, +0.022] at step 2000 |
| B2 | Δ cap-coverage under active drift, multi-seed, CI disjoint from 0 | ❌ | report 026: all CIs cross zero; per-seed variance 5× larger than B1 |

---

## C. Required controls (per design §"Control conditions")

| # | Control | Status | Evidence |
| -- | --- | :---: | --- |
| C1 | No-replay baseline | ✅ | every exp 18 / exp 19 run |
| C2 | Random-codebook control | ✅ | report 026: 0 candidates, Δ = 0.000 exactly |

---

## D. Drill-downs (per design §"Drill-downs")

| # | Drill-down | Aggregated multi-seed? | Status | Evidence |
| -- | --- | :---: | :---: | --- |
| D1 | Meta-stable rate over time (should stay lower with replay) | no | ❌ | data in JSON, never aggregated |
| D2 | Mean engagement over time (should drop as landscape stabilizes) | yes | 🟨 | aggregated in report 026; entropy not lower (exit criterion E3 below) |
| D3 | Pattern age distribution (some short-lived, some long-lived) | **never measured** | ❌ | death mechanism vacuous; can't measure age distribution while no pattern dies |
| D4 | Number of discovered patterns per cycle (per scale) | yes | ✅ | report 026: W=2 9.0 ± 11.1, W=3 56.4 ± 27.2, W=4 179.6 ± 97.7 |
| D5 | u_k variable distributions (graduated consolidation) | yes (seed 17 trajectory) | ✅ | report 026 §u_k drill-down — bump centered on u_3/u_4, matches Benna-Fusi expectation |

---

## E. Project plan exit criteria (PROJECT_PLAN.md, Phase 4)

| # | Criterion | Status | Evidence |
| -- | --- | :---: | --- |
| E1 | Replayed trajectories improve future retrieval | ✅ | report 026 ΔR@K verified |
| E2 | Stale atom drift can be corrected without global rewrites | 🟨 | reencode_every=100 is doing this in the runs, but its specific contribution is not ablated |
| E3 | Repeated solution paths become faster and lower entropy | ❌ | report 026 §entropy: flat at W=2; **rises above baseline by step 2000 at W=4** |

---

## F. Architectural integration (the integration the audit surfaced)

This was assumed-tested but wasn't really.

| # | Integration | Status | Evidence |
| -- | --- | :---: | --- |
| F1 | Phase 3 Hebbian + Phase 4 replay running concurrently, multi-seed, with active codebook evolution | ❌ | exp 19 was run **single-seed, no drift, Hebbian fired at 1.8% of cues = effectively static codebook**. Mechanism validated (63× more discoveries vs broken error-driven baseline) but not headline-validated. |
| F2 | Phase 4 with batch codebook retrains as drift source | ❌ | sanctioned drift source per design, never tested |

---

## G. Cross-spec discipline (from notes/notes/2026-05-09)

Per the headline-vs-drill-down framework, the Phase 4 headline must:
- (G1) be CI-disjoint from zero ✅ for B1 / ❌ for B2
- (G2) survive the control comparison (random-codebook) ✅

---

## What's missing for Phase 4 to graduate

| Required | Effort | Priority |
| --- | --- | --- |
| F1 — multi-seed Phase 3+4 integration with drift | ~3h | **must-do** |
| A12 — fix death settings (or document that turnover isn't load-bearing); rerun 5-seed | ~3h | **must-do** |
| B2 — Δcap-coverage at design-spec disjoint-CI bar | depends on whether A12/F1 fix it | **must-do** |
| E3 — entropy exit criterion (resolve W=4 entropy rise) | depends on W=4 candidate count investigation | **must-do or document as deferred** |
| D1, D3 — meta-stable rate + age distribution aggregation | ~30 min total | should-do for completeness |
| A13, A14 — ablate sparse-update + reencode contributions | ~2h each | should-do |

---

## What's deferred (with explicit acknowledgement)

- **Adaptive m / store_threshold / cross-pattern coupling** — design §"deferred from this design," start with fixed values.
- **Sleep/wake cycles** — design §deferred.
- **Multi-scale interaction (do scales share traces?)** — design §deferred; start with per-scale replay.

These don't block Phase 4 graduation per the design itself.

---

## Status banner (one line for STATUS.md)

> Phase 4: 1 of 2 headlines verified, 1 of 3 exit criteria met, integration test not multi-seed run, death mechanism not exercised.
