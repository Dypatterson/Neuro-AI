# Report 047 — Phase 5 A+B+step3 K-branch divergence: FAIL at n=1

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism-validity criterion #2 fails at n=1 on seed 17.
Falsification SIGNAL, not final falsification (n=5 needed per
[design note pre-commitment](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)).
Surfaces a substrate-construction failure mode A+B does not address.
**Driver:** `experiments/40_phase5_branching.py --mode headline`
against [reports/phase5_ab_pilot_seed17_step3/snapshots/phase3_phase4_w4_step1800.pt](phase5_ab_pilot_seed17_step3/snapshots/phase3_phase4_w4_step1800.pt).
**Output:** [reports/phase5_ab_pilot_seed17_step3/branch_diag_w4_step1800/phase5_headline_seed17.json](phase5_ab_pilot_seed17_step3/branch_diag_w4_step1800/phase5_headline_seed17.json)

## Experiment preamble

**Active phase:** 5

**Headline metric for this report:** K-branch state divergence within
30% of pre-death state_divergence (mechanism-validity criterion #2 per
the [2026-05-20 diagnostic-actuator note](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)).
Pre-death seed 17 reference (from [report 043](043_phase5_substrate_scale_diagnostic.md)):
`role_K4 state_divergence = 0.0336`. Target band: [0.024, 0.044].

**Required controls:** content_K4 (paired), random_K4 (sanity),
role_K4_g0 (γ=0 collapse sanity), role_K1 (single-branch baseline).

**Last verified result:** [report 046](046_phase5_ab_pilot_seed17_step3.md)
— A+B+step3 1-seed pilot passes mechanism-validity criterion #1
(d_eff ≥ 25 at W=4 step 1800; observed 35.20).

**Why this experiment now:** the design's pre-committed second
mechanism-validity criterion must pass before n=10. n=1 is a signal,
not the criterion's final test (which is n=5).

## Headline mechanism-validity result

| Condition         | state_divergence | n_branches (mean) | ΔE_K4 vs pre-death band [0.024, 0.044] |
| ----------------- | ---------------: | ----------------: | -------------------------------------- |
| role_K4 (pilot)   | **0.000**        | **1.00**          | FAIL (outside band)                    |
| content_K4        | 0.000            | 1.00              | (collapsed same as role_K4)            |
| random_K4         | 0.000            | 1.00              | (sanity: all collapse identically)     |
| role_K4_g0 (γ=0)  | 0.000            | 1.00              | (γ=0 sanity)                           |
| role_K1           | 0.000            | 1.00              | (K=1 baseline)                         |
| **Pre-death** ref | 0.0336           | (4)               | (target reference)                     |

`mean_n_branches = 1.00` for every condition — the K-branch mechanism
is producing exactly one effective branch per cue. State divergence is
identically 0 because there's only one branch to compare against itself.

Per-cue ΔE (E_content − E_role) at K=4: mean = +5.96e-09 ± 4.7e-08;
2 of 20 cues positive. Pure FP noise around zero — the role-prior and
content-prior branches are returning the *same* settled state because
the underlying schema store is degenerate (see below).

## Cause: schema-store collapse on the A+B+step3 substrate

The schema store at `experiments/40_phase5_branching.py` selects top-k
by effective_strength. On the A+B+step3 substrate:

| Snapshot | n_atoms | top-8 indices | top-8 pairwise sim (mean off-diag) |
| --- | ---: | --- | ---: |
| Pre-A+B baseline (seed 17 step 1500) | 1024 | 167, 205, 211, 451, 634, 713, 720, 775 (spread across original Phase-3 atoms) | **0.361** (range [0.032, 0.842]) |
| A+B+step3 pilot (seed 17 step 1800)  | 1064 | 1044, 1045, 1046, 1047, 1048, 1049, 1055, 1063 (**all discovery-channel atoms**) | **1.0000** (FP-precision identical) |

The pilot's top-8 schemas are **near-duplicate atoms** at FP precision.
The baseline's were diverse. The A+B+step3 substrate has d_eff = 35.20
(passes criterion #1), but the *strength-ranking-by-effective_strength*
selects atoms that collapse to a single FHRR direction.

### Why the discovery-channel atoms dominate

Candidate A's reinforcement modulation is
`reinforce_input *= (1 − coverage_lambda · r_ema)`. For new atoms added
by the Phase 4 discovery channel:

- **Initial state:** new atoms enter consolidation at `r_ema[idx] = 0`
  (initialized to zero in `add_pattern()`).
- **First reinforcement window:** with `r_ema ≈ 0`, the (1 − r_ema)
  factor is ≈ 1 → full reinforcement on every retrieval.
- **r_ema EMA rate is 0.01** (slow) → new atoms gain strength faster
  than r_ema catches up.

For original Phase-3 atoms (already in the substrate at Phase 4 start):

- They have been reinforced over the entire Phase 4 run.
- `r_ema` for them has accumulated to the substrate's bulk redundancy
  level (≈ 0.09 at step 1800, per [report 046](046_phase5_ab_pilot_seed17_step3.md)
  drill-down).
- Their reinforcement input is throttled by (1 − 0.09) ≈ 0.91.

The result: **discovery-channel atoms accumulate strength faster than
original atoms** because their `r_ema` starts at 0. This is an
unanticipated consequence of A's design: the modulation systematically
advantages newly-added atoms over older atoms.

### Why the discovery atoms are near-duplicates

The Phase 4 discovery channel (`experiments/19_phase34_integrated.py`
candidate_handler) adds an atom whenever a re-settling pass in the
replay loop produces a candidate with `final_top_score ≥
resolve_threshold`. On the A+B+step3 substrate, repeat retrievals of
similar cues all converge to similar settled states — when the
discovery channel adds a new atom from one of these convergent settles,
it adds a near-copy of the convergent state. With ~40 discovery atoms
across the run (1024 → 1064), most are near-duplicates of the same
"basin of attraction" the substrate produces.

## What this means for A+B's mechanism-validity status

**A+B+step3 passes criterion #1 (d_eff preservation) but produces a
substrate with a degenerate strength-ranked schema store.** The
top-k-by-effective-strength selector picks essentially-one atom
repeated K times; no K-branch diversity is possible.

The mechanism-validity gate failure at n=1 is **not** caused by:

- Insufficient d_eff (it's 35, well above 25)
- Step 3 being wrong (the consolidation drill-downs are identical
  between the with/without step 3 pilots — see [report 046](046_phase5_ab_pilot_seed17_step3.md))
- Pattern matrix collapse at the substrate level (the broader 1064
  atoms ARE diverse; it's only the top-strength subset that collapses)

It IS caused by:

- A's (1 − r_ema) modulation systematically advantaging new atoms
  (with r_ema = 0 init) over original atoms (with r_ema accumulated),
  which causes discovery-channel atoms to dominate strength rankings.
- The discovery-channel atoms being near-duplicates of each other
  because they come from convergent retrieval settles on a substrate
  with a few dominant basins.

## Three possible fixes (each requires a separate design + anti-homunculus check)

These are sketches, NOT implementation commitments:

### Fix A1 — Substrate-aware r_ema initialization for new atoms

When the discovery channel adds a new atom, initialize `r_ema[new] =
_coverage_redundancy_instantaneous(P)[new]` instead of 0. New atoms
that are duplicates of the existing substrate would get `r_ema ≈ 1` at
add-time and be throttled to near-zero reinforcement immediately.
Novel atoms would get `r_ema ≈ 0` and gain strength normally.

**Anti-homunculus check sketch:** computing r_inst at add-time is the
same local geometric operation as the periodic EMA update. Adding it
as an initialization step (not a "decide whether to add" step) keeps
the candidate-handler shape unchanged. PASS-shaped on first read.

**Drawback:** still uses categorical add/don't-add for new atoms (the
existing candidate_handler). A more dynamic-form move would be to let
all candidates be added with continuous initial r_ema, accepting that
duplicate-candidates self-throttle.

### Fix A2 — Schema-store selection by combined strength + diversity

Replace `top_k_by_effective_strength` with a selection rule that
balances strength and diversity (e.g., diversified top-k via
Maximum Marginal Relevance or determinantal point process). This is
strictly a Phase 5 change; A+B substrate unchanged.

**Anti-homunculus check sketch:** a selection rule reading both
strength and pairwise similarity is a measurement-time mechanism over
the schema store. It's not in the runtime architecture's
critical path; the system "uses one prior" (the diversified store)
and that prior is a substrate-derived geometric quantity. Possibly
PASS-shaped depending on exact form.

**Drawback:** patches the symptom (schema-store degeneracy) without
addressing the underlying issue (new atoms strength-advantage).

### Fix A3 — Path 3 β: continuous role-fidelity weighting

Per [2026-05-20 cue-regime design note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md):
replace categorical top-k schema selection with a continuous weighted
sum `prior = Σ_i (cue · s_i)^p · f_i^q · s_i` where `f_i` is the
per-schema role-binding fidelity. This is a more substantial Phase 5
redesign and was anti-homunculus-PASS audited.

**Anti-homunculus check:** already PASSED in
[bddd0b2 commit message](https://github.com/Dypatterson/Neuro-AI/commit/bddd0b2).

**Drawback / concern:** β operates over the same atoms that the
top-k selector saw. If 1041 of 1064 atoms have `|E_i| ≈ 0` and step 3
weighting suppresses them in retrieval already (via `w_i = σ((|E_i|
− ε)/τ)`), the role-fidelity weighted prior may still be dominated by
the ~23 alive atoms. β alone may not fix the discovery-channel-
duplication problem.

## Pre-committed binding remains intact

Per H4 of the [design note](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md):

> If the mechanism fails D1 preservation, the candidate is wrong-shaped
> and we go back to design.

D1 preservation (Phase 4 D1 non-regression) is criterion #3 (still
unmeasured at n=1; bounded by the seed-17 single-run readouts in
report 046). Criterion #2 is K-branch state_divergence, NOT D1.

**No re-tuning of α_anti, coverage_lambda, coverage_ema_rate,
repulsion_step_size, ε, or τ.** The failure at criterion #2 is at the
SUBSTRATE-CONSTRUCTION level (discovery channel + r_ema init), not at
the parameter-setting level. The right move is a design-level fix
(one of A1–A3), not parameter retuning.

## What this report does NOT do

- It does NOT declare A+B+step3 falsified. n=1 is a signal, not the
  pre-committed n=5 threshold for criterion #2.
- It does NOT commit to a fix. A1, A2, A3 are sketches; each requires
  a separate design + anti-homunculus check + decision.
- It does NOT block other paths. Path 3 (cue-regime design) was
  already filed contingent on this pilot; this report shows that path 3
  may be necessary BUT may also be insufficient (Fix A3 drawback).
- It does NOT exercise n=5. The local-snapshot grid has 5 seeds; if
  the user wants the formal n=5 result, exp 40 can be run against the
  4 other A+B+step3 retrains (which would require running pilots for
  seeds 1, 2, 11, 23 — out of scope for this report).

## Updated recommendation

Three options in disciplined order:

1. **Pause + design**. Surface this finding to the user; choose between
   fixes A1, A2, A3 (or a combination) at the design level; write the
   design note + anti-homunculus check; THEN implement and re-pilot.
   *Most faithful to the project's discipline.*
2. **Run n=5 first** — re-pilot the A+B+step3 retrain across seeds 1,
   2, 11, 23 and confirm the criterion #2 failure is consistent across
   seeds before designing a fix. *Slightly more expensive; tightens the
   diagnostic.*
3. **Try Fix A1 directly** (substrate-aware r_ema init) and re-pilot.
   It's the smallest mechanism patch and the most natural extension of
   A. If A1 closes the gap, A2/A3 can be deferred. *Fastest move; risks
   skipping the design step.*

Recommended: **option 1**. The discipline cost is one design note; the
information cost of skipping it is committing to a fix that doesn't
address the root cause.
