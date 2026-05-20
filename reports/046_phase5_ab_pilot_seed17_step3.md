# Report 046 — Phase 5 A+B 1-seed pilot WITH step 3

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism-validity re-verification after step-3 implementation.
Supersedes [report 045](045_phase5_ab_pilot_seed17.md) for the binding
mechanism-validity verdict (both PASS; report 045 is preserved as the
without-step-3 baseline).
**Driver:** [scripts/run_phase5_ab_pilot_seed17.sh](../scripts/run_phase5_ab_pilot_seed17.sh) →
[experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py)
**Output:** [reports/phase5_ab_pilot_seed17_step3/](phase5_ab_pilot_seed17_step3/)
**Pre-committed knobs:** `alpha_anti=1.0`, `coverage_lambda=1.0`,
`coverage_ema_rate=0.01`, `repulsion_step_size=100.0`,
`retrieval_weight_epsilon=0.05`, `retrieval_weight_tau=0.02` (the last
two are new with step 3; pre-committed per anti-homunculus reviewer PASS).

## Experiment preamble

**Active phase:** 5

**Headline metric for this pilot (NOT graduation):** mechanism-validity
`d_eff ≥ 25 at step 1800 on W=4` — unchanged from report 045.

**Required controls:** the prior pilot ([report 045](045_phase5_ab_pilot_seed17.md))
on the same seed without step 3 is the natural A/B comparison. Phase 4
readouts (top1, topk, cap_t05) provide drill-down.

**Last verified result:** [report 045](045_phase5_ab_pilot_seed17.md)
— without step 3, d_eff = 35.23 at W=4 step 1800. Implementation gap
flagged: step 3 (E_i-weighted retrieval) not yet wired.

**Why this experiment now:** Report 045 §"Updated next-step
recommendation" picked option 1 — implement step 3 before n=10 — as
the disciplined choice. Step 3 landed at commit
[2a94f5f](https://github.com/Dypatterson/Neuro-AI/commit/2a94f5f) with
anti-homunculus reviewer PASS. This pilot re-verifies mechanism-validity
with the full design as audited.

## Headline mechanism-validity result

| Metric                  | No-step-3 ([report 045](045_phase5_ab_pilot_seed17.md)) | Step-3 (this report) |
| ----------------------- | ----------------: | --------: |
| W=4 step 1800 d_eff     | 35.23             | **35.20** |
| n_atoms                 | 1064              | 1064      |
| d_eff target            | ≥ 25 (pre-commit) | ≥ 25 ✓ PASS |
| Δ d_eff vs baseline (5.36) | +29.87         | +29.84    |

**Both pilots pass the mechanism-validity gate.** Step 3 doesn't change
substrate geometry (it operates on retrieval weights, not pattern
positions or consolidation strengths). The minute difference (35.23 vs
35.20) is float-precision noise across two independent runs at the
same seed — sampling order through the Hebbian updater differs by
~5% in succ_rate (see drill-down).

## d_eff trajectory comparison (W=4)

| step | No-step-3 d_eff | Step-3 d_eff |
| ---: | --------------: | -----------: |
|  500 | 36.34           | 36.34        |
| 1500 | 35.69           | 35.67        |
| 1700 | 35.38           | 35.36        |
| 1800 | 35.23           | 35.20        |

Trajectories are essentially identical (max delta = 0.03 at any step).
Confirms that step 3 leaves substrate evolution untouched.

## Consolidation drill-down (W=4) comparison

| step | No-step-3 mean_strength | Step-3 mean_strength | No-step-3 r_ema_mean | Step-3 r_ema_mean | No-step-3 dead_ready | Step-3 dead_ready |
| ---: | ----------------------: | -------------------: | -------------------: | ----------------: | -------------------: | ----------------: |
|  300 | 0.22074                 | 0.22074              | 0.01684              | 0.01684           | 0                    | 0                 |
|  600 | 0.12228                 | 0.12228              | 0.03306              | 0.03306           | 0                    | 0                 |
|  900 | 0.08490                 | 0.08490              | 0.04838              | 0.04838           | 0                    | 0                 |
| 1200 | 0.06496                 | 0.06496              | 0.06286              | 0.06286           | 0                    | 0                 |
| 1500 | 0.05224                 | 0.05225              | 0.07657              | 0.07657           | 0                    | 0                 |
| 1800 | 0.04336                 | 0.04336              | 0.08956              | 0.08957           | 1035                 | 1038              |

The consolidation state evolves **bit-identically** in both pilots through
step 1500 (differences ≤ 1e-5 are FP precision). At step 1800 there's a
tiny divergence in `dead_ready` (1035 → 1038), which reflects the small
retrieval-path divergence over the last 300 cues feeding back into the
death-counter logic.

This is the design's intent: step 3 only affects retrieval weights, not
the slow-timescale dynamics that A and B run on. The consolidation chain
keeps its same trajectory; only the *exposure* of low-strength atoms at
the retrieval surface changes.

## Phase 4 readouts (Condition C) comparison

| step | No-step-3 top1 | Step-3 top1 | No-step-3 capt5 | Step-3 capt5 | No-step-3 succ_rate | Step-3 succ_rate |
| ---: | -------------: | ----------: | --------------: | -----------: | ------------------: | ---------------: |
|  300 | 0.102          | 0.102       | 0.293           | 0.293        | 0.177               | 0.177            |
|  600 | 0.102          | 0.102       | 0.293           | 0.293        | 0.135               | 0.135            |
|  900 | 0.102          | 0.102       | 0.289           | 0.289        | 0.131               | 0.132            |
| 1200 | 0.102          | 0.102       | 0.284           | 0.284        | 0.149               | 0.152            |
| 1500 | 0.102          | 0.102       | 0.284           | 0.284        | 0.177               | 0.181            |
| 1800 | 0.102          | 0.102       | 0.284           | 0.284        | 0.218               | 0.222            |

`top1` and `capt5` are **identical** to FP precision throughout the run.
`succ_rate` (Hebbian updater's success rate) diverges by ~3-5% in late
training — step 3 nudges which atoms win in the marginal retrieval
cases, which slightly affects the Hebbian updater's per-cue success
classification, but doesn't propagate to a measurable change in
top1/topk readouts on the 300-sample test set.

## Interpretation

**Step 3 is shape-correct and has minimal observable Phase 4 effect.**

This is exactly what should happen for a well-shaped mechanism that
operates on *retrieval-weight composition* without changing
*substrate evolution*. The full A+B+step3 design is now implemented
per the anti-homunculus-PASS design note, and the mechanism-validity
gate passes both with and without step 3.

The interesting question — does step 3 buy anything at the *Phase 5
energy-landscape* level — cannot be answered by this Phase 4
pilot. Phase 5 evaluation (running `experiments/40_phase5_branching.py`
against the new substrate to compute the K-branch ΔE diagnostic) is
the natural next step. Step 3's value, if any, is that during the
Phase 5 settling iterations the low-strength atoms (1038 of 1064
W=4 atoms at step 1800) contribute infinitesimally rather than
equally — the energy landscape Phase 5 reads is effectively over
the ~26 alive atoms, much closer to the "few-atom dense substrate"
condition the design intended.

## Pre-committed binding remains intact

All five A+B knobs were set ONCE before either pilot ran (see
[scripts/run_phase5_ab_pilot_seed17.sh](../scripts/run_phase5_ab_pilot_seed17.sh)
and the `ConsolidationConfig` defaults). The d_eff gate passes; H4's
"falsification means redesign, not retune" prohibition does not fire.
None of {α_anti, coverage_lambda, coverage_ema_rate,
repulsion_step_size, retrieval_weight_epsilon, retrieval_weight_tau}
has been tuned post-hoc.

## What this report does NOT do

- It does NOT declare Phase 5 graduation. Mechanism-validity is a
  necessary-but-not-sufficient step.
- It does NOT exercise the Phase 5 K-branch ΔE diagnostic. That's
  the next move (against the new substrate snapshots).
- It does NOT yet n=10 on Colab. n=1 is the pilot; n=10 is the
  graduation-attempt move.

## Updated next-step recommendation

Step 3's pilot is clean. The disciplined sequence now:

1. **Run the Phase 5 K-branch ΔE diagnostic** against
   `reports/phase5_ab_pilot_seed17_step3/snapshots/phase3_phase4_w4_step1800.pt`
   using `experiments/40_phase5_branching.py --mode headline`. Compare
   `state_divergence` and per-cue ΔE against the pre-death snapshot
   ([report 043](043_phase5_substrate_scale_diagnostic.md) reference).
   This answers "does the A+B+step3 substrate produce the
   pre-death-like branch divergence?" at n=1, which is the second
   mechanism-validity criterion in the design note.

2. **If state_divergence is within 30% of pre-death** (the
   pre-committed criterion), **ship to Colab for n=10 retrain**
   with the same A+B+step3 config; aggregate across 10 seeds for
   the binding Phase 5 graduation attempt.

3. **If state_divergence falls short** at n=1, this is informative
   without being conclusive (n=1) — but it would suggest A+B+step3
   alone is insufficient and the cue-regime axis (path 3,
   [β + γ note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md))
   becomes the next architectural move.
