# Report 045 — Phase 5 A+B 1-seed pilot retrain

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism-validity verdict on the A+B death-mechanism dynamic-form.
**Driver:** [scripts/run_phase5_ab_pilot_seed17.sh](../scripts/run_phase5_ab_pilot_seed17.sh) →
[experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py)
**Output:** [reports/phase5_ab_pilot_seed17/](phase5_ab_pilot_seed17/)
**Pre-committed knobs:**
`alpha_anti=1.0`, `coverage_lambda=1.0`, `coverage_ema_rate=0.01`,
`repulsion_step_size=100.0`
(per [phase5_ab_calibration.json](phase5_ab_calibration.json) and
[2026-05-20 diagnostic-actuator note](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)).

## Experiment preamble

**Active phase:** 5

**Headline metric for this pilot (NOT graduation):** mechanism-validity
d_eff ≥ 25 at step 1800 on W=4
([phase5-checklist.md §A1 falsification design](../notes/emergent-codebook/phase-5-checklist.md)
+ [2026-05-20 design note "Pre-committed falsification criteria"](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)).

**Required controls:** none on n=1 — the controls (random codebook,
γ=0, K=1) are graduation-time, not mechanism-validity-time. n=1
suffices to falsify A+B if d_eff < 25.

**Last verified result:** [report 044](044_consolidation_geometry_diagnostic.md)
— pre-A+B baseline at W=4 step 1800: 12 atoms, d_eff = 5.36.

**Why this experiment now:** A+B is the
[2026-05-20 dynamic-form re-expression](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)
of the binary mass-death step. The pilot is the first empirical test
of the design's claim that d_eff is preserved without binary deletion.
Falsification result (d_eff < 25) sends A+B back to design; pass ships
to Colab n=10.

## Headline mechanism-validity result

| Metric                 | Pre-A+B baseline ([report 044](044_consolidation_geometry_diagnostic.md)) | A+B pilot |
| ---------------------- | -------------: | -------: |
| n_atoms (W=4 step 1800)| 12             | 1064     |
| d_eff (W=4 step 1800)  | 5.36           | **35.23** |
| d_eff target           | ≥ 25 (pre-commit) | ✓ PASS  |
| Δ d_eff vs baseline    | —              | +29.9     |

`d_eff = 35.23` is **comfortably above the validity threshold** (25)
and within shouting distance of the pre-death d_eff (~40 at step 1500).
Independently confirmed via the canonical
[consolidation_geometry_diagnostic.py](../scripts/consolidation_geometry_diagnostic.py):
35.23 (matches the analyzer to FP). Substrate-level d̄ = 0.7265 ± 0.13,
regime = `spread` (d̄ / θ' = 7.27 with β=10).

## d_eff trajectory across the snapshot grid

Pilot snapshots span {W=2, W=3, W=4} × {step 500, 1500, 1700, 1800}.
`scripts/analyze_phase5_ab_pilot.py` reports:

| W  | step 500 | step 1500 | step 1700 | step 1800 |
| -- | -------: | --------: | --------: | --------: |
| 2  | 21.15    | 20.78     | 20.61     | 20.52     |
| 3  | 28.32    | 27.67     | 27.39     | 27.23     |
| 4  | 36.34    | 35.69     | 35.38     | 35.23     |

The trajectory is **essentially flat at each scale** — d_eff loses ~1
unit between step 500 and 1800 at W=4. The mass-death event that
collapsed the baseline from ~40 to ~5 between step 1500 and 1800
**does not occur** because A+B's `garbage_collect()` no-op guard fires
when `coverage_lambda > 0`.

## A+B consolidation drill-down (W=4 substrate)

The interesting drill-down — what A+B's continuous dynamics are
actually doing per checkpoint:

| step | n_pat | mean_strength | r_ema_mean | r_ema_max | patterns_below_threshold | dead_ready |
| ---: | ----: | ------------: | ---------: | --------: | -----------------------: | ---------: |
|  300 | 1064  | 0.221         | 0.017      | 0.024     | 0                        | 0          |
|  600 | 1064  | 0.122         | 0.033      | 0.048     | 0                        | 0          |
|  900 | 1064  | 0.085         | 0.048      | 0.070     | 0                        | 0          |
| 1200 | 1064  | 0.065         | 0.063      | 0.090     | 1023                     | 0          |
| 1500 | 1064  | 0.052         | 0.077      | 0.110     | 1044                     | 0          |
| 1800 | 1064  | 0.043         | 0.090      | 0.129     | 1041                     | **1035**   |

Reading the table:

- **`n_pat` stays at 1064 throughout.** No atom is ever deleted. The
  binary `garbage_collect()` path is correctly suppressed by A+B's
  guard (commit [2c50f02](https://github.com/Dypatterson/Neuro-AI/commit/2c50f02)).
- **`r_ema_mean` rises monotonically** from 0.017 → 0.090 (~5×).
  Candidate A's continuous redundancy EMA is tracking the substrate's
  geometry and modulating reinforcement input by `(1 - r_ema)`.
- **`mean_strength` falls 5×** (0.221 → 0.043). The Benna-Fusi chain
  decay + Candidate A's modulation is reducing effective strength
  continuously. No discrete event.
- **`patterns_below_threshold = 1041` at step 1800.** Nearly all atoms
  are below the legacy `death_threshold=0.05`.
- **`dead_ready = 1035` at step 1800** — the count of atoms that
  *would have been* binary-culled. The substrate is asymptotically
  approaching the "death" boundary at the continuous level.

This is the dynamic the design specified: death as the asymptotic
limit of a continuous decay, not a discrete delete event.

## The implementation gap (transparent disclosure)

Per the [2026-05-20 design note §"Combining candidates"](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md):

> 3. **Each atom's contribution to retrieval is continuously weighted
>    by its current `E_i`** (no membership flag, no threshold). Atoms
>    whose `E_i` decays toward zero contribute infinitesimally to
>    retrieval by construction.

**This is not yet implemented.** The current `TorchHopfieldMemory.retrieve()`
weights patterns by softmax over similarity only, with no `E_i` factor.
Consequence: at step 1800 of this pilot, the 1035 atoms below the
death threshold (effective_strength ≈ 0) still contribute *equally* in
retrieval to the few atoms with strength ≈ 0.05. The "asymptotic
death" property — atoms fade in their contribution as their strength
decays — is unexpressed at retrieval time.

The mechanism-validity gate (d_eff ≥ 25) passes for a real reason
(A+B preserved substrate density by suppressing binary death), but
the deeper "atoms with zero strength contribute infinitesimally"
property requires the step-3 wiring: a continuous sigmoidal weighting
`w_i = σ((E_i − ε) / τ)` on each pattern's contribution to retrieval
softmax. Without it, the substrate's effective behavior under A+B is
"all atoms remain active, just with low Benna-Fusi strength."

## Auxiliary observations (not graduation-gating)

- **Replay store stayed at 0 throughout.** `gate_signal = engagement
  × (1 - resolution)`, and resolution stayed at 0.293 (relatively
  high) so the gate threshold was rarely crossed. Replay cycles fired
  (per the `replay_every=50` schedule) but the substrate dynamics
  step proceeded over empty traces. This is independent of A+B.
- **Top1 dropped from 0.129 (static baseline) → 0.102** in
  Conditions B and C. Same direction and magnitude as
  [report 030](030_phase34_rfix_5seed.md)'s "top1 regression is a
  Phase 3 / Hebbian-codebook-reshaping property" finding. Not a Phase 4
  or A+B regression; Phase 3 alone produces it.
- **Substrate d̄ = 0.73 at W=4 step 1800.** Far above θ' = 0.10
  (regime classifier threshold). Substrate is in the "spread" regime
  — patterns are well-distributed in FHRR space.

## Interpretation

**A+B passes the mechanism-validity gate as designed but partially.**

What works:
- Death-as-asymptotic-limit is correctly expressed in the consolidation
  dynamics (mean_strength decay, r_ema rise, patterns_below_threshold
  rising without n_atoms changing).
- The architectural guard ensures binary death cannot run alongside
  A+B (no `n_pat` drop, `dead_ready=1035` but no actual deletion).
- d_eff stays in the spread-regime range (35.23 at step 1800), far
  above both the post-death baseline (5.36) and the validity gate
  (25).

What's missing:
- **Step 3 (E_i-weighted retrieval contribution).** The asymptotic
  death property is *computed* in consolidation but not *exposed* at
  retrieval. A retrieval-time sigmoidal weighting `w_i = σ((E_i − ε)
  / τ)` is the missing piece.

## Pre-committed binding (binding decision)

Per the design note's H4 (lines 345-353):

> If the first retrain misses the d_eff criterion, that is a
> falsification result, not an invitation to re-tune α. ...

The pilot **did not miss** the criterion. So H4's re-tuning prohibition
does not fire. The decision is structural, not parameter:

- **Implement step 3** (the missing piece, not a tuning move) before
  shipping n=10 to Colab, OR
- **Ship n=10 as-is** with the gap documented, treating the current
  A+B as "death-suppression + slow continuous strength decay, without
  retrieval-time fade-out" — a deliberate partial implementation.

This is a design judgment call. The conservative read is that step 3
is **part of the design that passed anti-homunculus audit**, so
shipping without it is shipping an incomplete mechanism. The liberal
read is that the validity gate measures d_eff, and d_eff is in range,
so the next step is graduation testing.

## What this report does NOT do

- It does NOT declare Phase 5 graduation. Mechanism-validity is one
  step of three (d_eff ✓, branch_divergence sanity ?, Δms_w3 D1 ?).
- It does NOT claim A+B is "fully implemented." See §implementation gap.
- It does NOT report n=10. n=1 is the pilot; n=10 is the next move
  contingent on the decisions above.
- It does NOT exercise the path-3 cue-regime mechanism (β + γ from
  the [2026-05-20 cue-regime note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md)).
  Path 3 is contingent on this pilot's pass; if step 3 is implemented
  and n=10 also passes mechanism-validity, then path 3 becomes the
  next architectural axis to attempt.

## Updated next-step recommendation

Three options for the next move, in order of architectural
disciplined-ness:

1. **Implement step 3** (sigmoidal `w_i = σ((E_i − ε) / τ)` weighting
   in `TorchHopfieldMemory.retrieve()` and the related retrieval
   paths), re-run the 1-seed pilot, verify d_eff stays in range AND
   that the substrate's *effective* retrieval set is no longer
   dominated by zero-strength atoms. Then n=10 on Colab. *Most
   faithful to the design.*
2. **Ship n=10 as-is**, document the partial implementation, and let
   the n=10 result drive whether step 3 is required. *Fastest to a
   Phase 5 attempt, but partial mechanism.*
3. **Pause, redesign step 3's interaction with retrieval-softmax**
   from scratch, then implement and pilot. *Most conservative; most
   delay.*

Recommended: option 1.
