# Project STATUS

**Last updated:** 2026-05-17 (**Phase 5 implementation unblocked.** [Report 040](reports/040_freq_weighted_alpha_sweep.md): freq-weighted Benna-Fusi α (brainstorm idea 5) sweep at n=10 × 4 λ values produces no measurable change in Phase 4 production metrics for λ ∈ {0, 0.5, 1.0}; λ=2.0 is supercritical (cascade strengths explode to 10⁹, W=2 corr inverts to −0.52). Mass death already does the filtering work freq-α was designed for, so the additional graded filter is redundant at production scale. Phase 5 design decision #1 RESOLVED: schema source = post-death substrate (5–30 surviving atoms per seed at W=2). Working tree has: Phase 5 unified design doc, exp 40 skeleton (signatures + dataclasses + atom-split signal + surprise prior), test scaffold (11 real tests + 28 skipped-with-reason). Earlier session: Phase 4 graduated on D1 ([report 038](reports/038_phase4_d1_graduation.md)) at Δ meta_stable_w3 = −0.7920, CI [−0.9376, −0.6463]; report 037 closed seed-3 collapse as corpus-order under binary death.)

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

---

## Active phase

**Phase 5 — implementation unblocked** ([phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md)).
HAM × energy-guided structural branching wrapped around the post-death
substrate. Schema source resolved by [report 040](reports/040_freq_weighted_alpha_sweep.md):
the 5–30 surviving atoms per seed at W=2 (post mass-death) are already
filtered by retrieval frequency through the binary death mechanism, so no
freq-α layer is needed. Branching seeds K_main schemas + 1 surprise branch;
combines via energy-weighted bundle + re-settle; atom-splitting diagnostic
on joint criterion (similar low energies AND substantial state divergence).

**Phase 4** remains graduated on D1 ([report 038](reports/038_phase4_d1_graduation.md));
no regression. Open next-step questions for Phase-4-revision (gradient
death, top1 regression mechanism, cross-corpus generalization) are parked
behind Phase 5 implementation.

---

## Current headline metric (post-pivot per reports 036 + 037 + [2026-05-16 discipline note](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md))

> **Δ meta-stable rate at W=3 (D1) under active codebook drift, multi-seed,
> CI disjoint from zero.**

Rationale: the design-spec headlines (R@K, cap-coverage) are *readouts*
over a substrate that varies wildly per seed under binary mass death.
D1 measures basin-geometry of whatever substrate survives — a substrate-pure
property invariant to which atoms happen to be reinforced. Report 026
already showed Δ=−0.51 at W=3 with per-seed std=0.038 in Phase 4 isolation.
D1 at the integration regime is the in-flight verification.

Drill-downs: ΔR@K (still reported; report 026's +0.010 stands as evidence
of record), Δcap-coverage (variance-bound), Δtop1 (drill-down per discipline
note), D1 at W=2 and W=4 (scale variation), D1 C−B (Phase 4 above
phase3-only).

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
  seeds ≥ 0. Pooled-Wilson Δ = +0.0144 (388/1111 vs 372/1111). Monotonically
  growing trajectory. ✓ (mean exceeds report 026's +0.010)
- Δ cap-coverage @ τ=0.5: +0.006, 3/5 ≥ 0 ✗
- **Δtop1: −0.026, 4/5 NEGATIVE** ✗ — initially attributed to stale
  discovered patterns; report 030 falsified that hypothesis (see next).

**[Report 032](reports/032_phase34_n10_verification.md)** (n=10 verification,
same config as 029):
- ΔR@10 at step 1500, n=10: **+0.009, 95% CI [−0.003, +0.021]**, 5+/3(0)/2−.
  CI still includes 0. Report 029's +0.0145 was sample-lucky; new 5 seeds
  produced +0.004 alone.
- Δtop1: −0.018, 7/10 negative. Robust Phase-3-driven regression.
- Pooled-Wilson R@10: A 0.329 [0.310, 0.349], C 0.339 [0.320, 0.359].
  Windows overlap heavily.
- C − B R@10: +0.004 mean — the clean architectural test of Phase 4 over
  phase3-only. Positive sign, CI includes 0.
- Bimodal seed distribution: strong-positives {1, 7}, near-zeros middle 6,
  negatives {3, 23}. Variance is dynamics-trajectory, not substrate.
- **Implication:** Phase 4's R@10 contribution under sanctioned online
  drift is real but small (+0.005 to +0.015 plausible range). Architecture
  is not falsified; effect size is just smaller than report 029 suggested.

**[Report 030](reports/030_phase34_rfix_5seed.md)** (st=0.3 + `--reencode-discovered`
fix, 5 seeds, Colab A100):
- Δ Recall@10: +0.0109 (DOWN from 029's +0.0145; fix erodes the headline)
- Δtop1: −0.0229 (essentially unchanged; regression persists)
- Δcap_t05: −0.0043 (flipped negative; fix made capt5 worse)
- **Verdict: fix was wrong-shaped.** Re-settling discovered queries pulls
  them toward existing attractors, destroying the geometric property the
  discovery channel was meant to capture. Default flipped to off; flag kept
  as opt-in for future selective-refresh variants.
- **Re-frame:** condition B (phase3-only, no Phase 4) *also* shows top1
  collapse under drift (seed 11: B_top1 0.110→0.081). So the top1
  regression is a **Phase 3 / Hebbian-codebook-reshaping** property, not
  a Phase 4 stale-pattern property. The Phase 4 discovery channel adds a
  small *additional* top1 cost on top, but the bulk is upstream.
- Hebbian fired ~17% (slightly higher than 029's 14.5%), drift comparable.

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
| 1 | ~~D1 at Phase 3+4 integration regime~~ — **CLOSED, graduates** | Δms_w3 = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds → 0. Also C−B = −0.720, CI-disjoint. R@10 +0.0089 replicates report 026. Phase 4 graduates. | [Report 038](reports/038_phase4_d1_graduation.md) |
| 2 | ~~A_k mechanism~~ — **CLOSED** | Falsified at gain=0.01 across decay ∈ {0.0, 0.02, 0.05, 0.10, 0.20}; orthogonal to the dominant death-driven substrate-shaping force. Mechanism kept implemented but not used in graduation runs. | Reports 026, 030, 033, 034, 035, 036 |
| 2a | ~~Option 1: diagnose seed 3 collapse~~ — **CLOSED** | Same mass-death survivor mechanism as seed 17; outcome direction set by whether the 5–32 surviving atoms cover the test set. A_nz at step 1500 predicts step-2000 survivor count exactly. Corpus-order variance under binary death; not tunable. | Report 037 |
| 6 | ~~Fix stale-discovered-patterns reencoding gap~~ | **CLOSED wrong-shaped** by report 030: the rfix variant erodes ΔR@10 (+0.0145 → +0.0109) and flips Δcapt5 negative (+0.006 → −0.004). Code kept as opt-in (default off); not the right primitive. | Report 030 |
| 6′ | **Top1 regression is Phase 3 not Phase 4 — and A_k AMPLIFIES it** | Report 030 §re-frame: condition B (phase3-only, no Phase 4) also shows top1 collapse under drift. The regression is a Hebbian-codebook-reshaping property, not a Phase 4 architectural gap. Action: characterize whether the top1 regression is an inherent online-Hebbian tradeoff (decide accept) or a fixable issue (decide investigate). **2026-05-15:** Ganesan-style FHRR unitarity audit closed (invariant holds within 2.4e-7); regression is a real mechanism property. **2026-05-16 (report 035):** A_k at gain=0.01, decay=0.0 at n=10 produced Δtop1 = −0.051 (CI strictly negative, 1/10 positive) — *worse* than baseline (~−0.018). So basin-narrowing mechanisms aren't a cure for #6'; they make it worse. Now investigating whether rank-1-vs-neighborhood is a fundamental tradeoff at this corpus size. | Reports 030, 035 + 2026-05-15 notes |
| 3 | ~~Δcap-coverage second headline~~ — **REFRAMED as drill-down** | Per [2026-05-16 discipline note](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md) + report 037: cap-coverage variance is downstream of binary death (corpus-order survival), not tunable without a death-mechanism redesign. Still reported as drill-down; no longer graduation-gating. | Reports 026, 028, 029, 037 |
| 4 | Diagnostic-actuator dynamic-form session | Named "next major architectural threshold" 2026-05-09, never held | [2026-05-09 paper synthesis](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md) |
| 5 | Seed-23 diagnostic | **Three** independent runs (026, 028, 029) all identify seed 23 as the cap_t05 / R@10 outlier. Idiosyncratic geometry, not noise. Discipline problem — continued tolerance without diagnosis is the bottleneck on tightening CIs. | Reports 026, 028, 029 |
| 7 | ~~Phase 3 codebook-comparison data integrity~~ — **CLOSED** | [Report 039](reports/039_phase3_codebook_comparison_integrity.md): labeling bug, not data integrity. Phase3b and phase3c each load phase3a's `random` and `learned` artifacts and save them back unchanged for per-directory self-containment, so the comparison script enumerates 6 condition labels backed by only 4 distinct tensors. Audit overclaimed on `reconstruction ≡ error_driven` (they are genuinely distinct). **Phase 4 uses `phase3c_codebook_reconstruction.pt` which is byte-distinct from every other Phase 3-era codebook** — graduation result unaffected. Report 017 headline stands; §3 interpretation corrected. Low-priority cleanup: deduplicate-by-tensor-hash in `experiments/31_phase3_comparison.py` so future runs aren't misleading. | audit-report-2026-05-14.md §2.3, §Appendix A; report 039 |

---

## Audits passed / failed for current phase

| Audit | Status | Evidence |
| --- | --- | --- |
| FEP / anti-homunculus on Phase 4 mechanisms | Passed | Report 026 §FEP audit |
| Random-codebook control | Passed | Report 026 |
| Multi-seed Phase 3+4 integration (drift-effective) | **Conditionally passed** — report 029 (st=0.3) produces real drift; ΔR@10 4/5 positive, mean +0.014; top1 regresses due to stale-discovered-patterns gap (blocker #6) | Report 029 |
| Shuffled-token control | Not run (Phase 3 discipline; Phase 4 design doesn't require) | — |
| Pattern death (architectural component) | **Not exercised** in any session run; report 033 mechanism diagnostic explains why and reshapes blocker #2 | Reports 026, 033 |
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
2. [notes/emergent-codebook/phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md) — **Phase 5 design**. HAM × energy-guided structural branching; decision #1 closed (schema source = post-death substrate).
3. [reports/040_freq_weighted_alpha_sweep.md](reports/040_freq_weighted_alpha_sweep.md) — **closes the freq-α bridge experiment**; resolves Phase 5 decision #1; documents the supercritical λ ceiling.
4. [reports/038_phase4_d1_graduation.md](reports/038_phase4_d1_graduation.md) — **Phase 4 graduation result**. n=10 integration regime, Δms_w3 = −0.7920 CI-disjoint; 10/10 seeds. Trajectory analysis (seeds 17, 3) shows mechanism heterogeneity.
3. [reports/037_seed3_collapse_diagnostic.md](reports/037_seed3_collapse_diagnostic.md) — **closes option 1**: seed 3 collapse is same death-survivor mechanism as seed 17; corpus-order variance under binary death. Establishes the rationale for the D1 pivot.
4. [reports/036_decay_sweep_and_mass_death_finding.md](reports/036_decay_sweep_and_mass_death_finding.md) — A_k decay sweep + seed-17 trajectory; identifies death as the dominant substrate-shaping force.
5. [notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md) — **binding standing discipline**: top1 demoted to drill-down; substrate-pure metrics (D1) preferred when readout × substrate variance is large.
4. [reports/035_saighi_ak_n10_falsification.md](reports/035_saighi_ak_n10_falsification.md) — n=10 verification of A_k at decay=0; falsified report 034. (Framed pre-discipline-shift around Δtop1; reading order item 3 reframes.)
5. [reports/034_saighi_ak_seed1_prototype.md](reports/034_saighi_ak_seed1_prototype.md) — A_k seed-1 prototype (falsified; preserved as the optimistic single-seed precedent).
6. [notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md](notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md) — paper synthesis that proposed A_k + FHRR audit.
7. [reports/033_phase4_death_mechanism_diagnostic.md](reports/033_phase4_death_mechanism_diagnostic.md) — blocker #2 mechanism diagnostic.
8. [reports/032_phase34_n10_verification.md](reports/032_phase34_n10_verification.md) — n=10 baseline (no A_k); comparison for report 035.
3. [reports/030_phase34_rfix_5seed.md](reports/030_phase34_rfix_5seed.md) — rfix did not work; hypothesis revised; death mechanism now load-bearing.
3. [reports/029_phase34_integration_st03.md](reports/029_phase34_integration_st03.md) — ΔR@10 conditionally verified under real drift via st=0.3.
4. [reports/028_phase34_integration_5seed.md](reports/028_phase34_integration_5seed.md) — st=0.5 run; drift didn't fire; superseded by 029.
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
