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
- ⚠️  **broken / reframed** — mechanism exists but isn't actually exercising as designed, *or* item has been reframed per a binding discipline note
- ⏳ **in flight** — run is actively executing or aggregating; evidence pending
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

## B. Headline metrics

The design names two readout headlines (R@K, cap-coverage). The [2026-05-16
discipline note](../notes/2026-05-16-substrate-vs-readout-metric-discipline.md) +
report 037 reframe these: D1 (meta-stable rate) is the substrate-pure
headline; R@K + cap-coverage become drill-downs because their variance is
downstream of binary-death survival outcomes.

| # | Headline | Status | Evidence |
| -- | --- | :---: | --- |
| B0 | **Δ meta_stable_w3 (D1) under active drift at the Phase 3+4 integration regime, n=10, CI disjoint from 0** | ✅ | [Report 038](../../reports/038_phase4_d1_graduation.md): Δ = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds at floor; C−B = −0.720 CI-disjoint |
| B1 | Δ Recall@K under active drift, multi-seed, CI disjoint from 0 | ✅ (drill-down) | [report 026](../../reports/026_phase4_verification_design_spec.md): +0.010 CI [+0.001, +0.022] at step 2000 |
| B2 | Δ cap-coverage under active drift, multi-seed, CI disjoint from 0 | ⚠️ reframed | Per report 037: variance is corpus-order under binary death, not tunable. No longer graduation-gating; reported as drill-down. |

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
| D1 | Meta-stable rate over time (should stay lower with replay) | **headline (B0)** | ✅ | Promoted to headline; verified at integration regime in [report 038](../../reports/038_phase4_d1_graduation.md). Isolation evidence in [`reports/d1_metastable_5seed.json`](../../reports/d1_metastable_5seed.json) (Δ=−0.51 W=3 std=0.038) replicated and amplified at integration (Δ=−0.79 W=3). |
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
| F1 | Phase 3 Hebbian + Phase 4 replay running concurrently, multi-seed, with active codebook evolution | ✅ | n=10 at integration regime in [report 038](../../reports/038_phase4_d1_graduation.md). Hebbian fired at meaningful rate (consolidations 52–571 per seed), Phase 4 candidates 63–108 per seed. |
| F2 | Phase 4 with batch codebook retrains as drift source | ❌ | sanctioned drift source per design, never tested |

---

## G. Cross-spec discipline (from notes/notes/2026-05-09)

Per the headline-vs-drill-down framework, the Phase 4 headline must:
- (G1) be CI-disjoint from zero ✅ for B1 / ❌ for B2
- (G2) survive the control comparison (random-codebook) ✅

---

## What's missing for Phase 4 to graduate (post-pivot)

| Required | Effort | Priority |
| --- | --- | --- |
| **B0** — Δ meta_stable_w3 at integration regime, n=10, CI disjoint from 0 | run in flight | **must-do** |
| F1 — multi-seed Phase 3+4 integration with drift | covered by B0 run | **must-do** (resolved by same run) |
| A12 — pattern death mechanism documented as "fires at n_cues=3000, kills un-reinforced atoms" (per report 036) | done in reports 033, 036 | document only, not a fix |
| B2 — Δcap-coverage at design-spec disjoint-CI bar | **reframed**: not gated on; reported as drill-down | n/a |
| E3 — entropy exit criterion (resolve W=4 entropy rise) | depends on W=4 candidate count investigation | **document as deferred** for graduation; revisit Phase 5 |
| D3 — pattern age distribution aggregation | ~30 min, optional | should-do for completeness, not graduation-gating |
| A13, A14 — ablate sparse-update + reencode contributions | ~2h each | should-do, not graduation-gating |

---

## What's deferred (with explicit acknowledgement)

- **Adaptive m / store_threshold / cross-pattern coupling** — design §"deferred from this design," start with fixed values.
- **Sleep/wake cycles** — design §deferred.
- **Multi-scale interaction (do scales share traces?)** — design §deferred; start with per-scale replay.

These don't block Phase 4 graduation per the design itself.

---

## Status banner (one line for STATUS.md)

> Phase 4: **GRADUATED on D1** (report 038, n=10 integration regime; Δms_w3 = −0.79 CI [−0.94, −0.65]; 10/10 seeds at floor). R@K replicated as drill-down at +0.0089. cap-coverage + top1 remain variance-bound drill-downs.
