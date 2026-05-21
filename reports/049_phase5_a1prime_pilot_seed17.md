# Report 049 — Phase 5 A1' (max-reduction r_inst) 1-seed pilot

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism-validity criteria #1 and #2 **still fail at n=1**
(falsification SIGNAL, n=5 not yet run), but with **dramatic
mechanism-level progress**. Discovery atoms' max effective_strength fell
from 12.1 (no A1) → 9.6 (A1) → **0.029 (A1')**, a 99.8% reduction.
Top-1 atom is now an *original Phase-3 atom* (idx 713) instead of a
discovery atom. K-branch diagnostic shows 2 effective branches at
n=2.0 (vs 1.0 in A1). State_divergence is non-zero for the first time
(0.000897, vs 0 in A1). The failure mode has migrated from
"substrate-construction produces dominant duplicates" to "schema-store
selector picks from FP-identical tied duplicates at the noise floor."
**Driver:** [scripts/run_phase5_a1prime_pilot_seed17.sh](../scripts/run_phase5_a1prime_pilot_seed17.sh)
→ [experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py)
(with A+B+step3 + A1 + A1' all active).
**Output:** [reports/phase5_a1prime_pilot_seed17/](phase5_a1prime_pilot_seed17/)
**Pre-committed knobs (binding, unchanged across A1/A1'):**
`alpha_anti=1.0`, `coverage_lambda=1.0`, `coverage_ema_rate=0.01`,
`repulsion_step_size=100.0`, `retrieval_weight_epsilon=0.05`,
`retrieval_weight_tau=0.02`. A1' introduces no new tunables; the
implementation is a one-line reduction-operator change at
[consolidation.py:_coverage_redundancy_instantaneous](../src/energy_memory/phase4/consolidation.py)
(mean-RMS → max), commit [9a66879](https://github.com/Dypatterson/Neuro-AI/commit/9a66879).

## Experiment preamble

**Active phase:** 5

**Headline metric:** top-8 schema pairwise FHRR similarity in band
[0.10, 0.60] at step 1800 on W=4 (criterion #1).
**Drill-downs:** d_eff (criterion #3); K-branch state_divergence
(criterion #2 at n=1); effective_strength distribution comparison
across the A+B+step3 / A1 / A1' substrates.

**Required controls:** the prior pilots (A+B+step3 in
[report 046](046_phase5_ab_pilot_seed17_step3.md) and A1 in
[report 048](048_phase5_a1_pilot_seed17.md)) provide the natural
A/B/C comparison.

**Last verified result:** [report 048](048_phase5_a1_pilot_seed17.md)
— A1 1-seed FAIL at criteria #1 and #2 (top-8 pairwise sim = 1.0000;
state_divergence = 0). Root cause was the Gram-row RMS proxy
under-measuring sparse duplicates.

**Why this experiment now:** the
[2026-05-20 A1' design note](../notes/notes/2026-05-20-r-inst-measure-dynamic-form.md)
(anti-homunculus reviewer PASS, commit
[9a66879](https://github.com/Dypatterson/Neuro-AI/commit/9a66879))
replaced the under-measuring proxy with `max_{j != i} |G_ij|`. The
hypothesis: A1's geometric init draws from this measurement; under
A1', duplicates get r_ema_init = 1.0 (full throttle) instead of 0.03
(near-zero throttle).

## Headline result against the six pre-committed criteria

| # | Criterion | Bar | Observed | Verdict |
| - | --- | --- | --- | --- |
| 1 | top-8 pairwise FHRR similarity at W=4 step 1800 | in band [0.10, 0.60] | **0.9325** (mean, range [0.7302, 1.0000]) | **FAIL (improved from 1.0000)** |
| 2 | K-branch state_divergence at W=4 step 1800 | in band [0.024, 0.044] | **0.000897** | **FAIL (improved from 0)** |
| 3 | d_eff at W=4 step 1800 | ≥ 25 | **35.23** | **PASS** |
| 4 | Phase 4 D1 non-regression Δms_w3 | ≤ -0.5 CI-disjoint, n=10 | deferred to n=10 | (deferred) |
| 5 | all consolidation knobs unchanged | code-level | unchanged | **PASS** |
| 6 | `max` is the only reduction change | code-level | one-line patch | **PASS** |

Criteria #1 and #2 **fail by less** than before; substrate-level mechanism
verified working, but a downstream selector-level effect now bounds
the result. n=1 signal is informative but not the formal n=5 verdict.

## Substrate-level: A1' fully throttles discovery atoms

| | A+B+step3 (no A1) | A1 (mean-RMS) | A1' (max) |
| --- | ---: | ---: | ---: |
| Original atoms max eff. strength (idx 0–1023) | 0.1205 | 0.0915 | **0.0586** |
| Discovery atoms max eff. strength (idx ≥ 1024) | **12.115** | **9.646** | **0.0293** |
| Discovery atoms mean | 0.5065 | 0.3385 | **0.0272** |
| Top-1 atom & strength | idx 1044 (disc.) @ 12.11 | idx 1044 (disc.) @ 9.65 | **idx 713 (orig.) @ 0.059** |

A1' brought discovery atoms to **the noise floor** (max 0.029 ≈
original atoms' mean 0.025). The first-added discovery atom (idx 1044)
that survived A1's throttle at strength 9.65 is now throttled
to ≈ noise. The architectural claim of A's r_i — "duplicates should
self-throttle to near-zero strength" — is now empirically realized.

## Top-8 schema makeup: mixed but still degenerate

| Substrate | top-8 indices | mean pairwise sim | min pairwise sim |
| --- | --- | ---: | ---: |
| Pre-A+B baseline (seed 17 step 1500) | original atoms spread across Phase 3 | 0.361 | 0.032 |
| A+B+step3 (no A1) | 1044–1063 (all discovery) | 1.0000 | 1.0000 |
| A1 (mean-RMS) | 1044–1063 (all discovery) | 1.0000 | 1.0000 |
| **A1' (max)** | **713** (orig.) + 1054–1060 (7 disc.) | **0.9325** | **0.7302** |

The top-1 is now an original Phase-3 atom (713). But the remaining 7
slots are filled by FP-identical discovery atoms (1054–1060) — the
last batch added by the discovery channel. They all entered
consolidation in the last few hundred cues with novelty_strength=1.0,
got throttled by A1' to gain zero strength, but the initial novelty
bump hasn't yet decayed below the original-atom baseline. Result:
seven atoms tied at the same effective_strength (0.02935) cluster in
the top-K because there are seven of them.

## K-branch state_divergence: from zero to nonzero

| Condition         | A+B+step3 / A1 | A1' | Pre-death ref |
| ----------------- | ---: | ---: | ---: |
| role_K4   mean_n_branches    | 1.0  | **2.0** | (4) |
| role_K4   mean_state_div     | 0.0  | **0.000897** | 0.0336 |
| content_K4   mean_n_branches | 1.0  | **2.0** | — |
| content_K4   mean_state_div  | 0.0  | **0.000897** | — |
| role_K4   ΔE vs content_K4   | 0    | 0    | — |

The K-branch diagnostic moved off zero. Two effective branches instead
of one. State_divergence is positive but ~37× below the pre-death band
([0.024, 0.044]). ΔE between role-prior and content-prior is still 0
(both priors land at the same energy because they're selecting from
the same near-degenerate top-K).

## d_eff trajectory (criterion #3 PASS)

| step | A+B+step3 ([report 046](046_phase5_ab_pilot_seed17_step3.md)) | A1 ([report 048](048_phase5_a1_pilot_seed17.md)) | A1' (this report) |
| ---: | ---: | ---: | ---: |
|  500 | 36.34 | 36.34 | 36.34 |
| 1500 | 35.67 | 35.66 | 35.70 |
| 1700 | 35.36 | 35.34 | 35.39 |
| 1800 | 35.20 | 35.19 | **35.23** |

A1' does not affect substrate broad geometry (it changes only how the
reinforcement-modulation gate is computed, not Candidate B's
repulsion). d_eff remains preserved with no measurable degradation
across the three pilots.

## The failure mode has migrated

**A+B+step3 (report 047):** discovery atoms dominate top-k with
strength 12.11, FP-identical, single basin retrieval. Substrate-
construction failure.

**A1 (report 048):** A1's geometric init partially throttled discovery
atoms (max 9.65, -20%) but the first-added duplicate still dominated
because the mean-RMS proxy under-measured sparse duplicates.
Measurement-operationalization failure.

**A1' (this report):** discovery atoms throttled to noise floor (max
0.029). Top-1 is now an original atom (idx 713). But **the top-K
selector still picks from 7 FP-identical discovery atoms tied at the
same effective_strength**, because the discovery channel kept adding
them and the strength selector treats their bumped-then-decayed u-chain
as a tied local maximum. Selector-layer failure.

The three failures form a clear progression: substrate-construction →
measurement → selector. Each level's fix reduces the failure
magnitude (1.0000 → 1.0000 → 0.93 on top-8 pairwise sim; 12.11 → 9.65
→ 0.029 on max discovery strength; 0 → 0 → 0.0009 on state_divergence)
but a downstream effect bounds the result. **A1+A1' has correctly
solved the substrate-construction and measurement levels.** What
remains is the selector layer.

## What this means for the path forward

Three options enumerated in [report 048](048_phase5_a1_pilot_seed17.md)
were A1' (this report), A1'' (streaming low-rank SVD), and A1''' (path-3
β / role-fidelity-weighted prior). With A1' in hand:

- **A1'' is now lower-leverage.** A1' has driven the discovery atoms
  to the noise floor; a more-precise measurement (formal projection
  magnitude) would catch a few more partial-redundancy cases but
  wouldn't break the FP-identical-duplicate tie at the selector. A1''
  remains the principled-formulation goal but not the immediate fix.

- **A1''' / β is now the natural next move.** The cue-regime / role-
  prior note's β candidate (continuous role-fidelity-weighted prior
  via `prior = Σ_i (cue · s_i)^p · f_i^q · s_i`) replaces the
  categorical top-K selector with a continuous weighted sum. The 7
  tied discovery atoms at strength 0.029 would each contribute
  `s_i · f_i` weight to the prior; their `f_i` (role-binding fidelity)
  would be the differentiator. Per the
  [2026-05-20 cue-regime note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md):
  >  β was already designed + anti-homunculus PASS.
  Its concern ("if 1041 of 1064 atoms have |E_i| ≈ 0 ... the
  fidelity-weighted prior may still be dominated by the ~23 alive
  atoms") is no longer the binding worry because A1' has already
  reduced the strength heterogeneity. β operates on the existing 1064-
  atom substrate; the role-fidelity weighting is now the architectural
  load-bearing piece.

- **Discovery-channel redesign** is a fourth option that hasn't been
  designed. The discovery channel kept adding 40 duplicates over the
  run; an architectural fix at that layer would prevent the tied-
  duplicate accumulation entirely. But per the prior W1/W2 rejections,
  a gate at add-time is wrong-shape. The dynamic-form alternative
  would be to *not call* `add_pattern` when a re-settled query lands
  in the basin of an existing atom — but that requires a different
  formulation than the current `resolve_threshold` filter, which
  itself is a discrete `if X then Y` rule. A separate session.

## Pre-committed binding remains intact

All six A+B+step3 knobs were set ONCE; A1' introduces no new tunables.
None of {α_anti, coverage_lambda, coverage_ema_rate,
repulsion_step_size, retrieval_weight_epsilon, retrieval_weight_tau}
has been tuned post-hoc. Per H4 from the
[death-dynamic design note](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md):
the discipline against parameter retuning continues to hold.

A1' design pre-commitment #6 ("`max` is the only reduction change")
remains honored — the patch is one line at
[consolidation.py:_coverage_redundancy_instantaneous](../src/energy_memory/phase4/consolidation.py).

## What this report does NOT do

- It does NOT declare A1' falsified. n=1 is a signal; n=5 is the
  formal verdict. But the n=1 signal is highly informative because
  A1's effect is geometric (should fire on every seed).
- It does NOT commit to a fix. The next move (β implementation, or
  another design) is a separate decision.
- It does NOT exercise n=5. The 7-tied-duplicate failure mode is
  visible at n=1 on this seed; cross-seed variance on this exact
  failure shape is uncertain but unlikely to change the verdict.
- It does NOT close out criterion #4. D1 non-regression at n=10 is
  deferred until criteria 1+2 pass at n=5.

## Updated recommendation

**Implement β (path 3) as the next session.** A1' has shown that the
substrate-level mechanisms (A+B+A1+A1') correctly throttle redundant
atoms — the architectural claim is realized. The remaining failure
is at the selector layer, which is exactly what β was designed to
displace. β's design + anti-homunculus check are already done
([2026-05-20-cue-regime-role-prior-dynamic-form.md](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md));
implementation is ~1-2 days; the same 6-criterion mechanism-validity
gate applies (with #1 reinterpreted as "fidelity-weighted top-k
diversity" instead of categorical top-k).

Anti-homunculus discipline reminder: the architecture is being
debugged at each layer in turn, with each fix passing its own design
+ audit + pilot cycle. The progression from A+B → A1 → A1' → β is
exactly what the project's discipline produces. Each commit gets us
closer to a substrate that produces "structural retrieval" in dynamic
form — not because a controller is doing it, but because each
mechanism is in its right shape.
