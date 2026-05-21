---
date: 2026-05-20
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: design-note
session-closes: STATUS-report-047-fix-selection
---

# Discovery-Channel `r_ema` Initialization Dynamic Form (Fix A1)

Companion to
[2026-05-20 diagnostic-actuator death-dynamic note](2026-05-20-diagnostic-actuator-death-dynamic-form.md)
(Path 1, A+B) and
[2026-05-20 cue-regime / role-prior note](2026-05-20-cue-regime-role-prior-dynamic-form.md)
(Path 3, β + γ). Held same day, immediately after
[report 047](../../reports/047_phase5_ab_branch_divergence_failure.md)
identified the failure mode of the A+B+step3 substrate at the K-branch
state-divergence diagnostic (n=1 signal): the top-8 schemas by
effective_strength are near-duplicate discovery-channel atoms with
FP-identical pairwise similarity (1.0000 vs the baseline substrate's
0.36).

Closes the "fix selection" decision flagged at
[report 047 §"Three possible fixes"](../../reports/047_phase5_ab_branch_divergence_failure.md).
Like the prior two notes, this is **design-only**. No implementation
commitment until anti-homunculus reviewer audit + a separate decision.

## What this session is for

Per the 2026-05-09 framing:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of.

The A+B design specified `r_ema` as "a continuous per-atom running
estimate" — an EMA over the per-atom redundancy `r_i`, with the update
rule local-per-atom (audit PASS, 2026-05-20). But the *initial condition*
of that EMA was left implicit. The implementation that landed
([consolidation.py:197-200](../../src/energy_memory/phase4/consolidation.py))
makes the choice explicit:

```python
self.r_ema = torch.cat([
    self.r_ema,
    torch.zeros(1, dtype=torch.float32, device=self.device),
])
```

Every new atom starts at `r_ema = 0`. That implicit choice carries a
categorical claim: **"a newly added atom is, by virtue of being newly
added, novel."**

The session's question: **what slow-timescale dynamic is "atom i's
redundancy at the moment of its addition" a fast-timescale snapshot of,
such that the initial condition is a local geometric quantity rather
than an implementer's declaration?**

## What the diagnostic established

[Report 047](../../reports/047_phase5_ab_branch_divergence_failure.md)
finding, A+B+step3 substrate at seed 17 step 1800, n=1:

| Substrate | top-8 indices | top-8 pairwise sim (mean off-diag) |
| --- | --- | ---: |
| Pre-A+B baseline (seed 17 step 1500) | spread across original Phase-3 atoms (167, 205, 211, ...) | **0.361** (range [0.032, 0.842]) |
| A+B+step3 pilot (seed 17 step 1800) | all discovery-channel atoms (1044–1063) | **1.0000** (FP-precision identical) |

The A+B+step3 substrate's broader 1064 atoms are geometrically diverse
(d_eff = 35.20, passing criterion #1). The top-by-effective_strength
selector — which Phase 5 uses for K-branch seeding — picks atoms that
are near-duplicates at FP precision. K-branch settling under these
seeds is degenerate by construction; `mean_n_branches = 1.00`,
`state_divergence = 0`.

The mechanism, traced in report 047:

1. The Phase 4 discovery channel adds new atoms when re-settled queries
   in the replay loop produce candidates exceeding `resolve_threshold`.
   On a substrate with a few dominant retrieval basins, repeat
   retrievals converge to similar settled states — new atoms are added
   as near-copies of those convergent states.
2. Each new atom enters consolidation at `r_ema[idx] = 0`.
3. With `coverage_lambda = 1.0`, A's reinforcement modulation
   `(1 - coverage_lambda · r_ema[idx]) = 1.0` for new atoms — full
   reinforcement at every retrieval.
4. Original Phase 3 atoms have accumulated `r_ema ≈ 0.09` over the
   Phase 4 run; their reinforcement is throttled to `(1 - 0.09) ≈ 0.91`.
5. The `r_ema` EMA rate is 0.01 (halflife ≈ 100 steps). The new atom
   accumulates ~100 reinforcements at full rate before r_ema reaches
   equilibrium.
6. By step 1800 the discovery atoms dominate `effective_strength`,
   pushing original atoms out of the top-8.

The diagnostic is the FP-identical top-8 similarity. The dynamic-form
question is: what continuous local process is `r_ema = 0` an
implementer-declared snapshot of, such that the snapshot itself can be
re-expressed as a geometric quantity?

## The wrong-shape current init

```python
# consolidation.py:197-200 (current behavior, all coverage_lambda regimes)
self.r_ema = torch.cat([
    self.r_ema,
    torch.zeros(1, dtype=torch.float32, device=self.device),
])
```

Shape diagnosis:

- The implementer chose the initial value (zero).
- That choice is uniform across atoms regardless of geometry.
- It is a categorical claim ("new = novel") that the existing
  `r_inst` measurement would have answered directly.
- The mismatch propagates: the EMA needs ~100 steps to relax to the
  geometric equilibrium, during which the atom accumulates strength
  at a rate the population-equilibrium dynamic does not endorse.

Anti-homunculus filter: **fails as currently written**, on a subtler
grounds than the binary death failure. There is no `if X then Y`
controller here — the failure is that an implementer-set constant
(`0`) substitutes for what should be a measurement. The constant
encodes a population-level claim ("new atoms are novel") that the
substrate's own geometry would have answered correctly.

The 2026-05-20 audit-PASS on A's `r_ema` rule said:

> The low-rank approximation of `r_i` MUST be a continuous per-atom
> running estimate (e.g., an EMA of projection contributions...). It
> MUST NOT be a scheduled "recompute global subspace every K steps and
> broadcast" — that is a controller in disguise.

The init at zero is the dual failure: a *scheduled-once initializer*
that broadcasts a constant. The audit caught the recompute-and-broadcast
shape; this note catches the init-by-constant shape — which has the
same form (a non-geometric value assigned by the implementer at a
specific moment).

## Wrong-shape candidates to reject up front

Three "controllers in disguise" the design must reject:

### W1 — Discovery-channel gate

"Modify `add_pattern` to refuse to add atoms whose `r_inst >
threshold`."

Shape: a controller reads `r_inst` for the candidate and decides
add-vs-no-add. The selection criterion is a threshold on a population-
derived metric. This is `if X then Y` in the explicit form. Rejected
on the same grounds as binary death.

### W2 — Schema-store selection by diversity

"Replace `top_k_by_effective_strength` with a diversified top-k
selector (Maximum Marginal Relevance, determinantal point process,
etc.). The Phase 5 branching code picks K atoms maximizing a
strength + diversity objective."

Shape: an arbitration over the population at measurement time. Even
if framed as "at the schema-store boundary," it inserts a chooser into
the system that reads pairwise similarity and decides which atoms
become branch seeds. The "decision" lives in the selector, not in any
local dynamic. Rejected — this is `argmax` over a population objective,
which is the textbook anti-homunculus failure.

(This is report 047's sketched Fix A2. The sketch noted it "patches the
symptom (schema-store degeneracy) without addressing the underlying
issue (new atoms strength-advantage)." The anti-homunculus reading is
why it's not just a worse patch — it's the wrong shape.)

### W3 — Post-hoc strength rebalancing

"Periodically rebalance `effective_strength` across atoms so the
discovery atoms don't dominate the top-k."

Shape: a scheduled global recompute-and-broadcast that reads the
population's strength distribution and adjusts to land in a target
shape. This is exactly the failure mode the original A's audit
flagged for `r_i` recomputation — applied here to a different
state variable. Rejected.

## Candidate: substrate-derived initial condition

A single candidate — A1, the smallest move that re-expresses the init
as a local geometric measurement.

### Candidate A1 — `r_ema` initialized at the geometric equilibrium

**Slow dynamic.** When the discovery channel adds atom i to the
substrate, initialize `r_ema[i]` from the substrate's *current
geometric* redundancy of i — the same `_coverage_redundancy_instantaneous`
computation the EMA update already uses:

> At add-time: `r_ema[i] ← r_inst(i, P ∪ {i})`
>
> where `r_inst(i, P)` is atom i's row-wise off-diagonal Gram RMS
> (the per-atom redundancy used in
> [consolidation.py:_coverage_redundancy_instantaneous](../../src/energy_memory/phase4/consolidation.py)).

The EMA update at subsequent `step_dynamics` calls is unchanged. The
*only* change is that the initial condition equals the EMA's geometric
equilibrium for the current substrate state — instead of zero.

**Why this is the equilibrium.** The EMA update is
`r_ema ← (1 − η) · r_ema + η · r_inst`. If `r_inst` is stationary, the
fixed point is `r_ema = r_inst`. Initializing at the fixed point means
the EMA starts at equilibrium for the current substrate; subsequent
updates respond to *changes* in `r_inst` rather than to the implementer's
initial guess.

**Diagnostic snapshot.** A new atom's `r_ema[i]` at the moment of
addition is the FHRR off-diagonal-Gram-RMS of atom i against the
existing substrate. For a near-duplicate added by the discovery channel
on a low-d_eff substrate, this is ≈ 1.0 (the row's off-diagonal entries
saturate near similarity 1). For a geometrically novel atom, this is
≈ 0. The diagnostic is the value of `r_ema` at step `t_add + 0`.

**Anti-homunculus check.** No new computation is introduced — the
function `_coverage_redundancy_instantaneous` already exists and is
used in the EMA update. The init draws from the same measurement.
There is no schedule (the computation happens at the add event, which
is itself a dynamic of the discovery channel, not a wall-clock or
step-count trigger). There is no decision (the value computed is used
as-is, not compared to a threshold). The categorical claim ("new =
novel") is replaced by a measurement ("how redundant is i, geometrically,
right now"). PASS-shaped on first reading.

**Mechanism downstream.** For discovery-channel duplicates at
add-time: `r_ema[i] ≈ 1` → reinforcement modulation `(1 - 1·r_ema)
≈ 0` → atom gains near-zero strength on every retrieval → never enters
the top-k. For geometrically novel atoms: `r_ema[i] ≈ 0` →
reinforcement modulation ≈ 1 → atom gains strength at full rate until
the substrate's collective r_inst around it rises. The substrate's
strength ranking ends up tracking *geometric uniqueness* rather than
*recency of addition*.

**Drawback.** The computation of `r_inst` at add-time requires the
full pattern matrix including the new atom. The existing add path
returns the new index *before* `step_dynamics` runs. Two
implementation shapes are available:

1. Defer the `r_ema` init to the first `step_dynamics` call after add
   that supplies `pattern_matrix`. At that call, detect "atoms with
   uninitialized r_ema" (sentinel value, e.g., NaN) and replace with
   the freshly-computed `r_inst` for those rows. Then run the standard
   EMA update. *(Implementation cost: small; sentinel handling is
   one extra branch in `step_dynamics`.)*
2. Have `add_pattern` accept an optional `pattern_matrix` argument and
   compute `r_inst` synchronously if supplied. *(Implementation cost:
   slightly larger API surface; pattern_matrix already lives in
   `UnifiedReplayMemory` and is constructible at the add call site in
   `replay_loop`.)*

Both shapes preserve the dynamic-form reading. The first is closer to
"the EMA dynamic is the only place r_ema is computed"; the second is
more atomic at the call site. *Either is acceptable; the implementer
picks based on which keeps `add_pattern`'s signature cleaner.* No
anti-homunculus distinction between them.

**Cost.** O(N²) per add (the existing `r_inst` cost). Adds at the
discovery channel are rare (~40 over 1800 cues per the seed-17 pilot).
Negligible.

### Why no other candidates

A2 (diversified selection) and a hypothetical strength-init coupling
were considered:

- **A2** is rejected as W2 above.
- **Strength-init coupling** ("init `effective_strength[new] = baseline ·
  (1 - r_inst)`") would set the strength itself based on r_inst at
  add-time. This works but has a heavier footprint: it changes how
  `effective_strength` is constructed for new atoms, affecting every
  place that reads it. A1 already produces the same eventual outcome
  (duplicates get near-zero strength growth) through the existing
  modulation, without changing the strength-init code path. A1 is the
  minimum-footprint move.

There is no third candidate worth proposing. The failure has one cause
(the categorical r_ema = 0 claim) and one fix (let geometry answer).

## Recommended candidate

**A1.** It is the smallest move that re-expresses the init as a
geometric measurement, uses the existing `_coverage_redundancy_instantaneous`
function, and changes no part of the EMA dynamic A+B already committed
to.

A1 is sufficient for the failure documented in report 047. β (from
the cue-regime note) remains the natural next architectural threshold
*after* A1 lands and the K-branch divergence diagnostic is
re-attempted. The two paths are independent: A1 fixes the substrate-
strength ranking; β fixes the role-prior weighting.

## Pre-committed falsification criteria

Before any A1 retrain, pre-commit these to keep us out of "the run
worked, declare graduation":

1. **Top-k schema diversity at step 1800.** Across 5 seeds, the
   top-8 atoms by `effective_strength` should have mean pairwise
   FHRR similarity within the band `[0.10, 0.60]` (matches the
   pre-A+B baseline 0.36 ± reasonable tolerance). The current A+B+step3
   substrate's 1.0000 falls outside this band; the failure criterion is
   *cross-seed median top-8 similarity > 0.60 at step 1800*.

2. **K-branch state_divergence within 30% of pre-death** across 5
   seeds (matches the [death-dynamic note's mechanism-validity
   criterion #2](2026-05-20-diagnostic-actuator-death-dynamic-form.md)).
   Pre-death seed-17 reference is `role_K4 state_divergence = 0.0336`;
   target band `[0.024, 0.044]` per seed; failure if median across
   5 seeds falls outside.

3. **d_eff preservation (substrate-level).** Still ≥ 25 at step 1800
   across 5 seeds. A1 must not erode the substrate's broad
   dimensionality (which currently passes at d_eff = 35.20). Failure
   if median d_eff < 25.

4. **Phase 4 D1 non-regression.** Δms_w3 ≤ -0.5 with CI-disjoint from
   zero across 10 seeds (per [report 038](../../reports/038_phase4_d1_graduation.md)).
   This is the non-regression constraint.

5. **r_ema init computation done once at add, not scheduled.** A
   *code-level* check, not a metric: there must be no scheduled
   recompute, no periodic re-init, no batch refresh. The
   anti-homunculus reviewer audits the diff against this rule.

6. **`coverage_ema_rate` is NOT retuned to compensate.** Currently 0.01
   (halflife ≈ 100 steps). If the team is tempted to "raise eta so
   r_ema reaches equilibrium faster" instead of A1, that is the
   wrong-shape patch — it accelerates the EMA but does not fix the
   categorical claim. Pre-commit: `coverage_ema_rate` remains 0.01;
   change requires its own design note.

If A1 passes 1–4 (with 5–6 honored by construction), the next move is
Colab n=10 retrain with the same config, then re-attempt the Phase 5
A1 headline.

If A1 fails criterion 1 (top-k diversity stays high), the failure is
deeper than the init — the discovery channel itself is adding too
many near-duplicates regardless of how their reinforcement is throttled.
That points to a different design session, not a retune.

If A1 passes 1 but fails 2 (K-branch divergence still degenerate
despite diverse top-k), then the failure is downstream of the schema
store — likely the role-prior weighting issue that β (path 3) is
designed for. β becomes the next session.

If A1 passes 1–2 but fails 3 (d_eff erodes), the modulation has become
*too* aggressive — the substrate stops growing because too many atoms
self-throttle. That would suggest `coverage_lambda` is overcalibrated,
which is a parameter-level falsification and requires the
death-dynamic note's redesign discipline (no retune; back to design).

If A1 passes 1–3 but fails 4 (D1 regresses), A1 is wrong-shaped at
a level the candidate enumeration missed; redesign session.

## What this design note explicitly does NOT do

- It does NOT commit to a Phase 4 retrain. A1 implementation requires
  anti-homunculus reviewer audit + a separate decision.
- It does NOT supersede the
  [2026-05-20 death-dynamic note](2026-05-20-diagnostic-actuator-death-dynamic-form.md).
  A+B is the substrate's slow-timescale dynamic; A1 is a correction
  to a specific *initial condition* in A's state. The combined
  mechanism remains A + B with A1 closing a gap in A's spec.
- It does NOT replace path 3 / β. Cue-regime / role-prior bimodality
  is a separate axis. A1 fixes "the substrate's strength ranking
  doesn't track geometric uniqueness." β fixes "the prior over schemas
  doesn't track role-binding fidelity." Both may end up necessary.
- It does NOT change the Phase 5 graduation criteria. The mechanism-
  validity criteria from the death-dynamic note remain unchanged; A1
  is a fix to make A+B reach them at n=5.
- It does NOT modify `coverage_lambda`, `coverage_ema_rate`,
  `alpha_anti`, `repulsion_step_size`, `retrieval_weight_epsilon`, or
  `retrieval_weight_tau`. All five remain at the values pre-committed
  in the death-dynamic note. A1 is a *shape* fix, not a *parameter*
  fix.
- It does NOT change the `add_pattern` decision (when to add). The
  discovery channel's gate (`resolve_threshold`) is unchanged.

## Implementation sketch

Files that would change (sketch only, NOT to be implemented from this
note alone):

- `src/energy_memory/phase4/consolidation.py` — two surgical changes:
  1. At [add_pattern lines 197-200](../../src/energy_memory/phase4/consolidation.py):
     replace `torch.zeros(1, ...)` with a sentinel (e.g., `torch.full(
     (1,), float('nan'), ...)`) if Shape 1 is chosen; or accept an
     optional `pattern_matrix` argument and compute `r_inst` for the
     new row directly if Shape 2 is chosen.
  2. At [step_dynamics lines 378-391](../../src/energy_memory/phase4/consolidation.py):
     if Shape 1, detect NaN entries in `r_ema` before the EMA update
     and replace them with the freshly-computed `r_inst[i]` for those
     rows. Then run the standard EMA update unchanged.
- `src/energy_memory/phase4/replay_loop.py` — if Shape 2 is chosen,
  pass `pattern_matrix` through to `add_pattern` at the discovery
  channel call site. No new logic; just plumbing.
- `tests/test_phase5_ab_death_dynamic.py` — add 3 tests:
  1. New atom added to a near-uniform substrate (r_inst per existing
     atom ≈ 0) gets r_ema ≈ 0.
  2. New atom added as a near-duplicate of an existing atom gets
     r_ema ≈ 1.
  3. With A1 active, a sequence of duplicate-adds over 100 steps
     produces a top-k by effective_strength that is geometrically
     diverse (no FP-identical entries).
- `experiments/19_phase34_integrated.py` — no flag changes; A1 is part
  of A's spec when `coverage_lambda > 0`. The current default off
  behavior (coverage_lambda=0 → no modulation, no init change) is
  preserved by construction (the sentinel/init logic only fires when
  coverage_lambda > 0).
- `scripts/run_phase5_ab_pilot_seed17.sh` — no changes; existing
  config is the A1 retrain config.

Cost estimate: 0.5–1 day implementation + tests; 0.5 day for 1-seed
pilot re-run against the existing seed-17 substrate; 1 day for the
Phase 5 K-branch ΔE re-diagnostic at n=5 using the local-snapshot grid.
Plus the falsification criteria as the discipline gates.

## Sequencing relative to A+B pilot and path 3

A1 is **contingent** on the A+B mechanism remaining the candidate at
the substrate level. Sequencing:

- **A1 lands first.** Library implementation + tests + 1-seed pilot
  re-run + n=5 K-branch ΔE diagnostic. If A1's pre-committed
  falsification criteria 1–4 all pass at n=5, ship to Colab n=10
  retrain.
- **β (path 3) is deprioritized until A1 either passes or fails its
  K-branch criterion.** If A1 passes criterion 2 (K-branch state_
  divergence in band), β may not be needed for Phase 5 graduation
  unless n=10 reveals residual bimodality across seeds. If A1 fails
  criterion 2 but passes 1 (diverse top-k), β becomes the next session.
- **The death-dynamic A+B substrate continues unchanged.**
  `coverage_lambda`, `alpha_anti`, etc. remain pre-committed.

## Closing the loop on the 2026-05-09 prescription

The 2026-05-09 note's anti-homunculus filter names two failure modes
that are visually distinct but architecturally identical:

1. A controller that reads a metric and triggers a discrete action.
2. An implementer who hard-codes a constant where a measurement
   belongs.

The binary death mechanism was failure mode 1. The `r_ema = 0` init was
failure mode 2. Both are arbitrations — one explicit, one implicit —
between "this is novel" and "this is redundant" that the substrate's
own geometry can answer.

A1 is the second concrete crossing from "geometry-as-observation" to
"geometry-as-endogenous-regulation" in the same shape the death-dynamic
note proposed. The death note made the *update* geometric; A1 makes
the *initial condition* geometric. With both fixed, A's full slow-
timescale dynamic over `r_ema` is uniformly anti-homunculus-compliant:
no implementer-set values at any point in the lifecycle of an atom.

If A1 passes its mechanism-validity gate, the architecture's first
diagnostic-actuator pair in dynamic form is *complete* in a strict
sense — no remaining implementer declarations in its lifecycle. The
2026-05-09 note's other four pairs remain open in their original
shape.
