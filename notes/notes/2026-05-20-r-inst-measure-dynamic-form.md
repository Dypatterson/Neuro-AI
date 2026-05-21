---
date: 2026-05-20
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: design-note
session-closes: STATUS-report-048-redesign-decision
---

# `r_inst` Redundancy-Measure Dynamic Form (Fix A1')

Companion to:

- [2026-05-20 diagnostic-actuator death-dynamic note](2026-05-20-diagnostic-actuator-death-dynamic-form.md) (Path 1, A+B)
- [2026-05-20 cue-regime / role-prior note](2026-05-20-cue-regime-role-prior-dynamic-form.md) (Path 3, β + γ)
- [2026-05-20 discovery-channel r_ema init note](2026-05-20-discovery-channel-r-ema-init-dynamic-form.md) (Fix A1)

Held same day, immediately after
[report 048](../../reports/048_phase5_a1_pilot_seed17.md) found that A1
fired correctly but only *partially* worked — the first-added discovery
atom (idx 1044) still has effective_strength 9.65 (100× the top
original atom 0.0915), because the implementation's
`_coverage_redundancy_instantaneous` proxy under-measures sparse
duplicates.

Closes the report-048 redesign decision among A1' / A1'' / A1'''. This
note designs **A1'** — the measurement-level patch.

Like the prior three notes, this is **design-only**. No implementation
commitment until anti-homunculus reviewer audit + a separate decision.

## What this session is for

The 2026-05-20 death-dynamic note specified Candidate A's `r_i` at the
*formal* level:

> `r_i = ||proj_{P_¬i}(p_i)|| / ||p_i||`
>
> where `proj_X(v)` is the projection of `v` onto the column-span of
> `X`.

The implementation chose a proxy (per
[consolidation.py:_coverage_redundancy_instantaneous](../../src/energy_memory/phase4/consolidation.py)):

```python
G_ij = (1/D) * <p_i, p_j>             # complex; |G_ii| = 1
r_i  = sqrt( mean_{j != i} |G_ij|² )  # RMS off-diag, ∈ [0, 1]
```

The proxy's docstring claims:

> For unit-magnitude FHRR patterns this proxy saturates at 1 when atom
> i is identical to all other atoms and goes to 0 when atom i is
> orthogonal to all of them.

The first claim is correct; the second is correct; **but the proxy
collapses to ≈ 0 when atom i is identical to ONE other atom and
orthogonal to the rest** — exactly the failure mode of report 048. The
proxy's wording about "identical to all other atoms" implicitly assumes
uniform similarity, which is not the regime the discovery channel
produces.

The session's question: **what slow-timescale dynamic is "atom i's
redundancy given the rest of the substrate" a fast-timescale snapshot
of, such that the snapshot correctly identifies sparse duplicates?**

## What the diagnostic established

[Report 048](../../reports/048_phase5_a1_pilot_seed17.md) finding, A1-
enabled A+B+step3 substrate at seed 17 step 1800:

|  | A+B+step3 (no A1) | A1 (this report's predecessor) | Effect |
| --- | ---: | ---: | --- |
| Original atoms max eff. strength | 0.1205 | 0.0915 | -24% |
| Discovery atoms max | **12.115** | **9.646** | **-20%** |
| Discovery atoms mean | 0.5065 | 0.3385 | -33% |
| Top-5 discovery strengths | [12.11, 1.99, 1.19, 0.67, 0.50] | [9.65, 0.95, 0.29, 0.29, 0.20] | progressively reduced |
| Top-8 pairwise sim | 1.0000 | **1.0000** (unchanged) | FAIL |

A1 reduced the *bulk* discovery-atom dominance by 20–70% on the top-5,
but the very first discovery atom (idx 1044, strength 9.65) still
dominates the rest of the substrate by ~100×. Top-8 schema diversity
is unchanged at FP-identical pairwise similarity.

**Why atom 1044 survived A1's throttle.** It was added early in Phase 4,
against an essentially-orthogonal substrate of ~1024 original atoms.
At its add-time, the proxy gave:

```
r_inst[1044] ≈ sqrt( (1² + ε²·(N-2)) / (N-1) ) ≈ sqrt(1/1023) ≈ 0.031
```

A1's init set `r_ema[1044] ← 0.031`. The reinforcement modulation `(1
− 0.031) ≈ 0.97` is nearly unthrottled. Atom 1044 accumulated strength
through the remaining 1700 cues. The implementation's proxy **diluted
the duplicate signal across N=1024 atoms** even though the formal
projection-magnitude reading would have correctly returned ≈ 1.0.

This is the diagnostic. The dynamic-form question is what continuous
local quantity, expressible as a per-atom reduction of the
substrate's pairwise Gram-row, correctly identifies the "duplicate
of any one neighbor" case.

## The wrong-shape current measurement

The mean-RMS reduction is the wrong-shape:

```python
r_i = sqrt( mean_{j != i} |G_ij|² )
```

Shape diagnosis:

- It is a *measurement*, not an arbitration — that part is correct.
- But the *reduction* (`mean` of squared similarities) averages a
  single saturated entry (|G_{i,k}| = 1) with N-2 small entries.
- For sparse duplicates against an otherwise-orthogonal substrate,
  the mean dilutes the saturated entry to ≈ 1/(N-1), giving r ≈
  1/sqrt(N-1).
- The formal definition (projection magnitude onto the column-span of
  `P_¬i`) does NOT dilute: it captures "how well can atom i be
  reconstructed by some linear combination of the others," and for a
  duplicate of any one atom, the reconstruction is 100% accurate
  (magnitude 1).

Anti-homunculus filter: the measurement is the right *shape* (per-atom
reduction of a local geometric quantity, no comparator-and-act), but
the *operationalization* is too weak. This is a failure mode the 2026-
05-09 note named in shape #2:

> An implementer who hard-codes a constant where a measurement
> belongs.

— except here it's "an implementer who hard-codes the *wrong*
reduction where the correct one belongs." `mean` is a measurement, but
it is not the measurement the formal definition called for.

## Wrong-shape candidates to reject up front

Three "controllers in disguise" the design must reject:

### W1 — Discovery-channel gate at add time

"Modify `add_pattern` to refuse atoms whose `r_inst > threshold`."

Already rejected in the [A1 design note §W1](2026-05-20-discovery-channel-r-ema-init-dynamic-form.md).
The threshold-and-act shape is the textbook anti-homunculus failure.
Restating here so the door stays closed.

### W2 — Discovery-channel pre-filter on duplicate detection

"Run `_coverage_redundancy_instantaneous` *before* adding; if the
candidate is too similar to existing atoms, skip the add."

Shape: identical to W1 in another wrapping. The threshold is at the
add gate, the action is "skip the add." `if X then Y` over a metric.
Rejected.

### W3 — Adaptive `coverage_lambda` per atom

"Set `coverage_lambda` higher for atoms whose initial `r_inst` was
small (compensating for the proxy's under-measurement)."

Shape: a per-atom parameter that adapts to the observed `r_inst`. This
is exactly the H4 violation from the death-dynamic note's pre-committed
falsification criteria:

> α and λ are set once before the first retrain from theoretical
> considerations, NOT tuned to land d_eff in the target [25, 50] range.

A per-atom adaptive λ is the same shape as a per-atom tuned α.
Rejected.

## Candidates: stronger per-atom reductions

Three candidates re-express `r_i` as a stronger per-atom reduction of
the Gram row that correctly identifies sparse duplicates. Each is
local per-atom and a measurement (not an arbitration).

### Candidate α — `max` over off-diagonal similarities

> `r_i = max_{j != i} |G_ij|`

For an FHRR unit-magnitude pattern that's a perfect duplicate of any
one neighbor, `r_i = 1.0`. For an atom orthogonal to all neighbors,
`r_i ≈ 0`. The two saturating regimes are identified correctly.

**Diagnostic snapshot.** Per-atom max similarity across the substrate.
A histogram of `{r_i}` across the substrate distinguishes:

- Isolated atoms (low r_i — orthogonal to all neighbors)
- Duplicates of any one atom (high r_i — saturated)
- Atoms inside a tight cluster (high r_i — saturated by their
  cluster-mates)
- Atoms with partial similarity to many neighbors (moderate r_i, lower
  than the cluster case)

**Anti-homunculus check.** `max` is a per-atom reduction of the
substrate's pairwise Gram row. There is no decision being made over
the population; there is no `if-then`; the value is computed from
atom i's row of similarities to its neighbors. It is the same shape
as `mean` but a different reduction operator. The geometric quantity
it measures is "how strongly is atom i represented by its most-
similar neighbor."

**Note on piecewise-smoothness.** Unlike `mean`, `max` has rank-swap
kinks: as the substrate evolves, the *identity* of the argmax can
jump from neighbor j to neighbor k. The *value* of the max is
continuous (both neighbors must have equal similarity at the swap
instant), so the measurement itself is piecewise-smooth. The
death-dynamic note's anti-homunculus check didn't require smoothness,
only "continuous local dynamic." Piecewise-smooth functions of the
substrate state ARE continuous functions; they're just not C¹. The
2026-05-09 framing of "local geometric dynamic" doesn't distinguish
smooth from piecewise-smooth — both are continuous local geometry, as
opposed to discrete arbitrations.

**Drawback.** Loses partial-redundancy signal. An atom that's 50%
similar to 100 neighbors gets `r_i = 0.5` under both `mean` and `max`
(approximately). An atom that's 50% similar to 1 neighbor gets `r_i
= 0.5` under `max` but `r_i ≈ 0.05` under `mean`. The distinction
matters when the substrate's failure mode is "distributed
redundancy" — but the report-048 failure mode is "sparse duplicate,"
which is what `max` directly fixes.

### Candidate β — Soft-max (p-norm interpolation)

> `r_i = (Σ_{j != i} |G_ij|^p)^(1/p)` with `p` large (e.g., p=20)

A continuously-differentiable interpolation between `mean` (p=2,
current proxy) and `max` (p=∞, Candidate α). At large finite `p`, the
function is smooth and approximates `max`.

**Diagnostic snapshot.** Same as α at the extremes; in between, a
weighted sum that gives more weight to the largest similarities.

**Anti-homunculus check.** Per-atom reduction of the Gram row, no
arbitration. PASS-shaped. Smoother than α (no rank-swap kinks).

**Drawback.** Introduces a new parameter `p` that needs to be set.
The death-dynamic note's H4-style discipline binds: `p` must be set
from theoretical considerations before any retrain, NOT tuned to
land top-k diversity in target range. If `p` is mis-set, the result
is a falsification, not an invitation to re-tune.

### Candidate γ — Streaming low-rank SVD (formal definition)

> `r_i = ||proj_{V_¬i}(p_i)|| / ||p_i||`
>
> where `V_¬i` is a streaming low-rank approximation to the column-span
> of `P_¬i`.

This is the load-bearing recommendation in the
[2026-05-20 death-dynamic note §"Implementation precondition"](2026-05-20-diagnostic-actuator-death-dynamic-form.md).
A running estimate of the substrate's principal subspace via streaming
SVD; atom i's redundancy is its projection magnitude onto that
subspace.

**Diagnostic snapshot.** The formal definition; r_i = 1 iff atom i is
in the row-span of the rest of the substrate.

**Anti-homunculus check.** Already PASSED audit in the death-dynamic
note (line 158-165). The audit's load-bearing constraint —
"continuous per-atom running estimate" — is the streaming-SVD's
natural form.

**Drawback.** Implementation cost: ~1-2 days for a streaming low-rank
SVD primitive that doesn't currently exist in the codebase. Per-atom-
per-step cost depends on the retained rank `k`; at `k ≈ 40` (the
d_eff scale of the pre-death substrate), the cost is `O(N·k²)` per
update, which at N=4096 is 4096 × 1600 ≈ 6.5M ops per step. Whether
that's fast enough at the existing pilot scale is an open question.

## Recommended candidate

**Candidate α (max-over-others).**

α is the smallest patch (one line in
`_coverage_redundancy_instantaneous`), the fastest to validate, and
directly addresses the report-048 failure mode (sparse duplicate
against an otherwise-orthogonal substrate). The anti-homunculus shape
is clean (per-atom measurement of a local geometric quantity); the
piecewise-smoothness concern is contained.

β is a fallback if α fails the anti-homunculus audit on the
piecewise-smoothness grounds. γ is the formal-definition fallback if α
fails empirically (e.g., produces a different failure mode). The
disciplined sequencing is **α first**, then β or γ depending on which
failure surfaces.

## Pre-committed falsification criteria

Before any A1' retrain, pre-commit these to keep us out of "the run
worked, declare graduation":

1. **Top-k schema diversity at step 1800.** Across 5 seeds, the
   top-8 atoms by `effective_strength` should have mean pairwise FHRR
   similarity within band [0.10, 0.60] (the same criterion as the A1
   design note). The current A+B+step3 substrate's 1.0000 falls
   outside this band; A1's 1.0000 also falls outside; A1' must bring
   this into band. **Failure criterion: cross-seed median top-8
   similarity > 0.60 at step 1800.**

2. **K-branch state_divergence within 30% of pre-death** across 5
   seeds (matches the [death-dynamic note's criterion #2](2026-05-20-diagnostic-actuator-death-dynamic-form.md)).
   Pre-death reference is `state_divergence = 0.0336`; target band
   `[0.024, 0.044]` per seed. Failure if median across 5 seeds falls
   outside band.

3. **d_eff preservation (substrate-level).** Still ≥ 25 at step 1800
   across 5 seeds. A1' must not erode the substrate's broad
   dimensionality.

4. **Phase 4 D1 non-regression.** Δms_w3 ≤ -0.5 CI-disjoint from zero
   at n=10. The non-regression constraint.

5. **No retune to compensate.** `coverage_lambda` remains 1.0;
   `coverage_ema_rate` remains 0.01; `alpha_anti` remains 1.0;
   `repulsion_step_size` remains 100.0. Code-level check; any change
   to these requires a separate design note.

6. **`max` is the only reduction change.** A1' patches ONE line in
   `_coverage_redundancy_instantaneous`. No other measurement changes.
   If empirical results suggest a second measurement also needs fixing,
   that is a separate design session.

If A1' passes 1–4 (with 5–6 honored by construction), the next move
is Colab n=10 retrain.

If A1' fails criterion 1 (top-8 diversity stays high), the failure is
deeper than the `mean`-vs-`max` choice. β (smoothed p-norm) or γ
(streaming SVD) becomes the next session — but the failure would
suggest the substrate's discovery channel is producing duplicates at
a rate any measurement-level fix can't catch up to. That points to a
discovery-channel level redesign, which is a separate axis.

If A1' passes 1 but fails 2, the failure is at the prior-weighting
layer (the schema store's role-fidelity issue) — β from the cue-regime
note (path 3) becomes the next session.

If A1' passes 1+2 but fails 3, then A's modulation has become too
aggressive — `coverage_lambda` is over-calibrated for the stronger
measurement. Per H4, that is a falsification, not an invitation to
retune; back to design.

If A1' passes 1+2+3 but fails 4 (D1 regresses), A1' is wrong-shaped
at a level the candidate enumeration missed; redesign session.

## What this design note explicitly does NOT do

- It does NOT commit to a Phase 4 retrain. A1' implementation requires
  anti-homunculus reviewer audit + a separate decision.
- It does NOT supersede A1. A1's geometric init at add-time remains
  the correct shape; A1' fixes the measurement A1 draws from. Both
  apply together.
- It does NOT supersede the death-dynamic note (A+B). The substrate's
  slow-timescale dynamic is unchanged.
- It does NOT change the Phase 5 graduation criteria. Mechanism-
  validity criteria from the death-dynamic note remain unchanged; A1'
  is a fix to make A+B+A1 reach them at n=5.
- It does NOT modify any tunable parameter. A1' is a *shape* fix at
  the measurement level; no parameter changes.
- It does NOT introduce a new state variable. The Gram row is already
  computed at every EMA update; the patch reuses it.
- It does NOT replace γ (streaming low-rank SVD) as the long-term
  goal. If A1' passes its mechanism-validity criteria, the project
  has a working measurement at the proxy level; γ remains the more-
  principled formulation for any future refinement.

## Implementation sketch

Files that would change (sketch only, NOT to be implemented from this
note alone):

- `src/energy_memory/phase4/consolidation.py` — patch
  `_coverage_redundancy_instantaneous` to use `max` instead of mean-RMS:
  ```python
  # before
  gram_sq = gram.abs() * gram.abs()
  off_diag_sum = (gram_sq * mask.to(gram_sq.dtype)).sum(dim=1)
  return (off_diag_sum / (n - 1)).clamp(min=0.0, max=1.0).sqrt().to(torch.float32)

  # after (A1')
  gram_abs = gram.abs()
  gram_abs_masked = gram_abs.masked_fill(~mask, 0.0)
  return gram_abs_masked.max(dim=1).values.clamp(min=0.0, max=1.0).to(torch.float32)
  ```
  Docstring updated to reflect the formal claim ("saturates at 1 when
  atom i is identical to *any* other atom") instead of the mistaken
  earlier wording ("identical to *all* other atoms").

- `tests/test_phase5_ab_death_dynamic.py` — update
  `TestCoverageRedundancyInstantaneous` to reflect the stronger
  per-atom values. Specifically:
  - `test_orthogonal_patterns_have_low_redundancy` should still pass
    (max of small values ≈ 0).
  - `test_duplicate_patterns_have_high_redundancy` should still pass
    (max ≈ 1 when the substrate contains duplicates).
  - A new test: a single duplicate among N orthogonal atoms — should
    return r_i ≈ 1.0 for the duplicate atoms and r_i ≈ 0 for the
    others. This is the report-048 failure case; the test pins A1'
    against regression.
  - The A1 tests (`TestA1DiscoveryChannelInit`) should still pass; A1
    init draws from r_inst, and the stronger r_inst values flow
    through unchanged.

- `scripts/run_phase5_a1prime_pilot_seed17.sh` — copy of
  `run_phase5_a1_pilot_seed17.sh` with output dir
  `reports/phase5_a1prime_pilot_seed17/`. No config changes.

- `reports/049_phase5_a1prime_pilot_seed17.md` — n=1 result against
  the six pre-committed criteria, parallel structure to report 048.

Cost estimate: ~0.5 day for implementation + test updates; ~0.5 day
for 1-seed pilot re-run; ~0.5 day for diagnostics + report. Total ~1.5
days, same shape as the A1 cycle.

## Sequencing relative to A1, A+B, and path 3

A1' is **contingent** on A1 remaining the candidate at the init level.
Sequencing:

- **A1' lands as a patch on top of A1.** Both apply: A1' fixes the
  measurement; A1 uses it at add-time. The combined system has
  `r_ema_init = max_{j != i} |G_ij|` for new atoms.
- **A+B+step3 substrate continues unchanged.** All pre-committed
  parameters remain.
- **β (path 3) is deprioritized until A1' either passes or fails its
  K-branch criterion.**
- **γ (streaming low-rank SVD) remains as the long-term cleaner
  formulation** — implemented only if A1' fails empirically or if a
  Phase 6 architectural pass calls for it.

## Closing the loop on the 2026-05-09 prescription

The 2026-05-09 note's anti-homunculus filter named two failure modes:

1. A controller that reads a metric and triggers a discrete action.
2. An implementer who hard-codes a constant where a measurement
   belongs.

The A1 note added a third, by extension:

3. An implementer who picks the *wrong reduction* of a measurement,
   such that the reduction itself encodes an architectural claim that
   the substrate's geometry doesn't support.

The current `mean`-RMS proxy is failure mode 3: it implicitly claims
"redundancy is the average similarity to ALL other atoms," which the
geometry of FHRR + sparse duplicates does not match. A1' substitutes
`max`-over-others, which encodes the claim "redundancy is similarity
to the most-similar neighbor" — closer to the formal projection-
magnitude reading and correct for the failure mode observed.

If A1' passes mechanism-validity, the architecture's first diagnostic-
actuator pair in dynamic form has all three failure modes closed for
A's lifecycle: failure mode 1 closed by A+B (no controller), failure
mode 2 closed by A1 (no constant init), failure mode 3 closed by A1'
(correct reduction). The other diagnostic-actuator pairs from the
2026-05-09 note remain open.
