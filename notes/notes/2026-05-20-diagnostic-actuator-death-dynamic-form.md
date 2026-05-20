---
date: 2026-05-20
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: design-note
session-closes: STATUS-blocker-4
---

# Diagnostic-Actuator Dynamic Form for the Death Mechanism

The session prescribed on [2026-05-09](2026-05-09-papers-diagnostics-and-actuator-dynamics.md)
as the "next major architectural threshold" and never held. Held now,
2026-05-20, immediately after [report 044](../../reports/044_consolidation_geometry_diagnostic.md)
identified substrate effective dimensionality (d_eff) collapse as the
geometric mechanism for the K-branch failure documented in
[report 042](../../reports/042_phase5_branching_collapse_diagnostic.md)
and [report 043](../../reports/043_phase5_substrate_scale_diagnostic.md).

Closes [STATUS](../../STATUS.md) blocker #4 as a design note. Does NOT
commit to implementation — that requires a separate decision after this
note has been audited by the anti-homunculus reviewer and accepted.

## What this session is for

Per [2026-05-09 §"Diagnostics vs Actuators":156-167](2026-05-09-papers-diagnostics-and-actuator-dynamics.md):

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of. Diagnostic and actuator are the
> same physical process viewed at different temporal resolutions.

For the death mechanism specifically: the question is **what is the
slow-timescale dynamic of which (d_eff, effective_strength, coverage)
is a fast-timescale snapshot?** Not "what rule should fire when d_eff
crosses a threshold."

The 2026-05-09 note explicitly warns that this is where the controller
re-enters under cover of a framing change. The current binary mass
death is exactly that failure mode: a step function on
`effective_strength < threshold` is a `if X then Y` rule even when the
metric X is a local geometric quantity. The action — atom deletion — is
not a continuous evolution of the metric; it is a discrete response
*to* the metric. That is the homunculus.

## What the diagnostic established

[Report 044](../../reports/044_consolidation_geometry_diagnostic.md)
finding, n=5 seeds × pre/post-death:

- Pre-death substrate d_eff ≈ 40–45 (out of 4096-dim space)
- Post-death substrate d_eff ≈ 2.5–6.4
- **The death event collapses substrate effective dimensionality by ~10× in a single step.**

The collapse is the geometric mechanism by which K-branch settling
becomes degenerate. With d_eff ~5 and K=4 branches, settling drains
diverse priors into the same low-dimensional well; the architecture's
"distinct schemas produce distinct settled states" claim fails for
geometric reasons.

This is the diagnostic. The dynamic-form question is: what slow-timescale
process is d_eff a fast-timescale snapshot of, such that the natural
evolution of that process produces an under-capacity substrate that
nonetheless preserves enough effective dimensionality to support K-branch
separation?

## The wrong-shape current mechanism

Current binary death in
[src/energy_memory/phase4/consolidation.py](../../src/energy_memory/phase4/consolidation.py)
and [src/energy_memory/phase4/replay_loop.py](../../src/energy_memory/phase4/replay_loop.py)
(approximately):

```
# Pseudocode of the current mechanism
for atom in atoms:
    if effective_strength(atom) < threshold for window steps:
        delete(atom)
```

Shape diagnosis:
- A controller reads `effective_strength` per atom.
- A threshold turns the reading into a binary signal.
- A discrete `delete` operation is triggered.
- The dimensionality collapse is the *side effect* of accumulated
  deletions, not a quantity any local dynamic is trying to preserve.

Anti-homunculus filter: **fails.** Even though `effective_strength` is a
local geometric quantity, the deletion action is an arbitration over
the population. The architecture has a small controller deciding who
lives and who dies, and that controller sees enough of the population
state to know the threshold is satisfied.

(Phase 4 design's anti-homunculus check at
[phase-4-unified-design.md:204-233](../emergent-codebook/phase-4-unified-design.md)
covers replay routing and the consolidation update rule, but does not
interrogate the death step itself. Per
[2026-05-16 graduation synthesis](2026-05-16-phase4-graduation-synthesis.md):
"Binary vs gradient death... not architecturally defended; a parameter
setting that happens to work for D1." It worked for Phase 4's D1
graduation. It does not work for Phase 5.)

## Candidates: continuous local dynamics

Three candidates that re-express death as a slow-timescale dynamic of
which a fast-timescale local-geometric quantity is a snapshot. None of
them has `if X then delete` in its formulation. None reads a population-
level metric and acts on a member. Each is local per-atom and continuous
in time.

### Candidate A — Coverage-weighted reinforcement rate

**Slow dynamic:** Each atom's reinforcement *rate* per retrieval is
modulated by how much of its variance is already covered by the
remainder of the substrate. Atoms in already-covered directions get
near-zero reinforcement per retrieval; atoms in uncovered directions
get full reinforcement. Atoms whose effective strength drops below the
substrate's stochastic noise floor stop participating in retrieval
— **death is the geometric limit of zero reinforcement, not a discrete
action.**

**Formulation.** Let `P` be the surviving pattern matrix and `P_¬i` be
the pattern matrix with row i removed. Define atom i's *coverage
redundancy*:

> `r_i = ||proj_{P_¬i}(p_i)|| / ||p_i||`

where `proj_X(v)` is the projection of `v` onto the column-span of `X`.
Then the reinforcement rate becomes:

> `dE_i/dt = (η · indicator(retrieval_i)) · (1 - r_i) - λ · E_i`

where `E_i` is atom i's effective strength, `η` is the base learning
rate, and `λ` is the natural decay rate. Highly redundant atoms (r_i →
1) gain almost nothing per retrieval; novel atoms (r_i → 0) gain at
full rate. The decay term λ ensures that any atom whose redundancy
stays high eventually loses its effective strength — but as a
continuous evolution, not a step.

**Diagnostic snapshot:** d_eff of the substrate at any moment is
exactly the sum of (1 - r_i) over surviving atoms, weighted by their
strengths. The diagnostic measures the cumulative state; the dynamic
runs continuously.

**Anti-homunculus check:** No global metric is read. Each atom's r_i
is computable from its own context (its pattern and the rest of the
substrate); the local rule is "your reinforcement gain is reduced by
your redundancy in the current substrate." No `if-then` step. No
deletion operation; atoms that lose strength simply stop firing in
retrieval (a softmax-floor effect).

**Drawback:** Computing `proj_{P_¬i}` per atom per step is O(N³) per
update. Will need a low-rank approximation or running estimate to make
the dynamic feasible at N≈4000.

**Implementation precondition (load-bearing — per anti-homunculus
audit 2026-05-20):** the low-rank approximation of `r_i` MUST be a
continuous per-atom running estimate (e.g., an EMA of projection
contributions, or a streaming SVD update). It MUST NOT be a scheduled
"recompute global subspace every K steps and broadcast" — that is a
controller in disguise (a schedule reading "expensive to compute" and
acting). The estimate is a local dynamic on i's own state, not a
snapshot the implementer hands i periodically.

### Candidate B — Dimensionality-preserving repulsion field

**Slow dynamic:** Add a repulsion term to the substrate's energy that
penalizes dimensionality collapse. Each atom experiences a continuous
force pushing it away from the current low-d subspace; the magnitude
of the force is proportional to how much that subspace shrinks if
atom i contracts toward existing atoms.

**Formulation.** Define the substrate's anti-collapse energy:

> `H_anti(P) = -α · log(d_eff(P))`

where `d_eff(P) = (Σ λ_k)² / Σ λ_k²` is the participation ratio of
P's Hermitian Gram matrix. Then each atom's update gets an additional
term:

> `dp_i/dt = -∂(H_main + H_anti)/∂p_i`

where `H_main` is the existing consolidation energy. The
`∂H_anti/∂p_i` term repels atom i from the directions already saturated
by other atoms; the gradient is continuous and local-per-atom (it
depends only on p_i and the substrate's spectral state).

**Diagnostic snapshot:** d_eff is *exactly* the quantity the dynamic
is conserving. The slow-timescale evolution of d_eff under this
dynamic is upper-bounded by α; collapse becomes thermodynamically
unfavorable.

**Anti-homunculus check:** No discrete decision is made. The repulsion
is a force in the substrate's energy landscape. Each atom feels it
through its local gradient. Nothing reads "d_eff is too low" and
triggers an action; d_eff is the integral of the dynamic.

**Drawback:** This is *not death at all*. Atoms aren't removed; they
just don't collapse together. To produce an under-capacity substrate
(which brainstorm Idea 5 / freq-weighted α argues is necessary for
abstraction), a separate mechanism for under-capacity is needed — see
the freq-weighted α work as the natural companion.

**Implementation precondition (load-bearing — per anti-homunculus
audit 2026-05-20):** `H_anti` is part of *the* substrate energy at
every call site that reads substrate energy (settling, replay scoring,
branch energy, consolidation). It is NOT a consolidation-time-only
term. The implementer must not add `H_anti` only to consolidation for
performance — that would re-introduce the controller (an energy term
that "fires" only when consolidation runs). α is fixed at training
start; α is NOT adapted from observed d_eff trajectories during
training. If α is wrong, the next retrain uses a different α; α is
not a feedback loop on d_eff.

### Candidate C — Per-atom inhibition with redundancy-coupled decay

**Slow dynamic:** Saighi-style per-attractor inhibition (the A_k that
was tried in [report 034](../../reports/034_saighi_ak_seed1_prototype.md)
and falsified at n=10 in [report 035](../../reports/035_saighi_ak_n10_falsification.md))
but with the inhibition decay coupled to the atom's *redundancy*, not
to time alone. Highly redundant atoms accumulate inhibition; novel
atoms shed it.

**Formulation.** Let `a_i` be atom i's adaptation variable:

> `da_i/dt = κ · indicator(retrieval_i) · r_i - μ · a_i · (1 - r_i)`

Redundant atoms (high r_i) gain inhibition on retrieval and shed it
slowly. Novel atoms gain little inhibition and shed it fast. Atom i's
*effective* contribution to retrieval is `E_i · (1 - a_i)`. As `a_i →
1`, the atom is functionally absent without being deleted; if context
later changes such that r_i drops, a_i decays and the atom reactivates.

**Diagnostic snapshot:** `a_i` is the per-atom adaptation. The
distribution `{a_i}` across the substrate is what would have been
"alive vs dead" in the binary scheme, now graded continuously.

**Anti-homunculus check:** Same shape as A_k except the time decay is
replaced by a redundancy-coupled decay. Local per-atom; no population
metric is read. The reason A_k was falsified at n=10 in report 035 was
that A_k didn't help the top1 regression — but that's an empirical
finding about *that specific* implementation. Coupling adaptation to
redundancy (not to time) is a different dynamic.

**Drawback:** Same r_i computation cost as Candidate A. Also, the n=10
A_k falsification raises the bar — any candidate in this family needs
to be paired with an upfront commitment about what would falsify it.

**Falsification pre-commitment (load-bearing — per anti-homunculus
audit 2026-05-20):** if C is ever advanced to implementation as a
primary candidate (rather than backup), it inherits the n=10
falsification bar that closed A_k in [report 035](../../reports/035_saighi_ak_n10_falsification.md).
*Before* any C retrain, pre-register a diagnostic prediction that
distinguishes redundancy-coupled C from time-decay A_k — e.g., "the
distribution of `a_i` across surviving atoms should be bimodal (low
for novel, high for redundant) for C, vs. unimodal under time-decay
A_k." Observe at step 1800. If the bimodality is absent, C reduces to
a renamed A_k and is falsified by report 035's existing data; no
retrain is needed and graduation chase stops.

## Combining candidates

Candidates A and B are not mutually exclusive — A modulates
*reinforcement* by redundancy; B adds a *repulsion force* in the
substrate energy. The natural combined system:

> 1. Each atom's reinforcement rate is multiplied by (1 - r_i). (A)
> 2. The substrate energy includes the -α log(d_eff) repulsion term. (B)
> 3. Each atom's contribution to retrieval is continuously weighted by
>    its current E_i (no membership flag, no threshold). Atoms whose
>    E_i decays toward zero contribute infinitesimally to retrieval by
>    construction — this is the geometric limit referred to in
>    Candidate A above, not a separate mechanism. **No "active-retrieval
>    set" exists as a categorical object.** If the implementer believes
>    a numerical floor is required for bounded retrieval-softmax
>    computation, use a smooth sigmoidal weighting `w_i = σ((E_i − ε)/τ)`
>    where τ is a fixed substrate parameter — NOT a two-threshold
>    hysteresis flag.

This combined system preserves d_eff via two complementary mechanisms
(modulated reinforcement *and* repulsion), and "death" is the
asymptotic limit of an atom whose strength has decayed toward zero
under (A)'s dynamics — i.e., the limit of a continuous process, not a
discrete event. No deletion. No membership flag. No supervisor. The
substrate's effective dimensionality is preserved at whatever level α
and λ jointly support.

**Why step 3's wording matters (per anti-homunculus audit 2026-05-20):**
The earlier wording in this note ("dropped from the active-retrieval
set ... a soft floor with hysteresis on ε") was flagged as a controller
in disguise — a two-state membership flag with threshold-crossing
transitions is exactly `if E_i < ε_low: drop_i ← True` followed by
`if E_i > ε_high: drop_i ← False`. That is a per-atom controller,
however small. The 2026-05-09 note's warning (line 158) that "the
temptation will be strongest exactly here, because the standard
reading is much easier to write down and code" was being illustrated
in real time. The rewrite eliminates the categorical membership object
entirely; retrieval contribution is a continuous function of E_i.

Candidate C is a third option that targets the *retrieval* dynamics
rather than the *reinforcement* dynamics. It could be combined with A
or used standalone. Given the falsification of the time-decay A_k,
proposing C requires re-doing the n=10 falsification setup with
redundancy-coupled decay; I would not commit to C without that test
first.

## Recommended candidate

**Combined A+B,** subject to anti-homunculus reviewer audit, then a
prototype implementation against the d_eff diagnostic, then a Phase 4
retrain on a 1-seed pilot to verify d_eff stays in the 30–50 range
(matching pre-death) without collapsing.

The combined dynamic has the cleanest dynamic-form reading: d_eff is
*literally* the quantity the substrate energy is preserving, and the
reinforcement-rate modulation is a continuous local rule on each
atom's own redundancy. No part of either component reads a population
metric and acts. The Saighi A_k variant (C) is preserved as an
alternative if A+B has implementation issues.

## Pre-committed falsification criteria

Before any retrain happens, pre-commit these to keep us out of the
"the run worked, declare graduation" failure mode:

- **D_eff preservation (substrate-level):** at step 1800 (or whenever
  the existing system would have died), substrate d_eff ≥ 25 on a
  W=4 substrate, across 5 seeds. (Pre-death pre-mechanism is ~40–45;
  post-death pre-mechanism is ~3–6. ≥ 25 is "halfway back to pre-death."
  Pre-commit before observing.)
- **K-branch state divergence rises:** report 043's pre/post-death
  ratio of state_divergence under K=4 was 3×–4978×. After mechanism
  swap, the post-mechanism state_divergence should fall within 30% of
  pre-death state_divergence — i.e. branches actually separate on the
  consolidated substrate. n=5 seeds is enough for this criterion.
- **Phase 4 D1 graduation preserved:** Δms_w3 ≤ -0.5 with CI-disjoint
  from zero across 10 seeds (report 038's bar). This is the
  non-regression constraint — the new mechanism must not break Phase 4.
- **Phase 5 A1 then attempted on n≥10:** ΔE_K4 CI disjoint from zero,
  n_seeds ≥ 10. **This is the graduation criterion**, not "any of the
  above passing" — those are mechanism-validity criteria, not
  graduation criteria.
- **α and λ are set once before the first retrain from theoretical
  considerations** (paper-derived or one-shot calibration), NOT tuned
  to land d_eff in the target [25, 50] range. If the first retrain
  misses the d_eff criterion, that is a falsification result, not an
  invitation to re-tune α. Retuning is permitted only after the
  retrain has been reported as falsified and a redesign session has
  re-derived α from first principles. Per H4 ([phase-5-checklist.md:195](../emergent-codebook/phase-5-checklist.md)):
  the d_eff ≥ 25 criterion must be a measurement, not a target the
  parameters are searched over.

If the mechanism passes the first three and A1 still fails, that's
information: the death-d_eff mechanism is correct in isolation but the
combiner / cue regime is the load-bearing remaining issue (per report
043's bimodal-ΔE finding). The discipline is to report that and re-scope,
not to chase a new mechanism revision.

If the mechanism fails D1 preservation, the candidate is wrong-shaped
and we go back to design.

## What this design note explicitly does NOT do

- It does NOT commit to a Phase 4 retrain. That requires the anti-
  homunculus reviewer audit of the candidates and then a separate
  decision.
- It does NOT close out the cue-regime / role-prior bimodality issue
  ([report 043](../../reports/043_phase5_substrate_scale_diagnostic.md)).
  That is a separate axis. A successful death-mechanism redesign is
  necessary but not sufficient for Phase 5 graduation.
- It does NOT replace [consolidation-geometry-diagnostic.md](../emergent-codebook/consolidation-geometry-diagnostic.md)
  as the pre-Phase-3 diagnostic spec; it builds on it.
- It does NOT calibrate θ′(β) (still open). The combined A+B candidate
  doesn't strictly need the calibration if r_i is computed directly
  from projections; the regime-classifier framework needs it.
- It does NOT propose to switch Phase 5's design to wrap the new
  substrate without re-checking the cue-regime axis. Report 043's
  bimodality persists across the d_eff range tested.

## Implementation sketch

Files that would change (sketch only, NOT to be implemented from this
note alone):

- `src/energy_memory/phase4/consolidation.py` — replace the binary
  death step with the Candidate A reinforcement-rate modulation. Add
  `r_i` computation as a method on the consolidation state.
- `src/energy_memory/substrate/torch_fhrr.py` — add the
  `H_anti = -α log(d_eff)` term to substrate energy (Candidate B). This
  affects every settling step that uses substrate energy; that surface
  is wide and needs an anti-homunculus check on the energy-landscape
  level, not just the consolidation level.
- `src/energy_memory/phase4/replay_loop.py` — delete the `_purge_dead`
  call. Atoms with E_i < ε are still in the substrate but don't
  participate in the active-retrieval set (a soft-floor flag, with
  hysteresis).
- `scripts/consolidation_geometry_diagnostic.py` — already built (this
  session); becomes the live diagnostic during the retrain to verify
  d_eff stays bounded.

Cost estimate: 1-2 days to write + test against existing 5-seed
substrates locally; 1 day for the Phase 4 retrain at n=10 on Colab;
1 day to run the full Phase 5 headline against the new substrate. Plus
the falsification criteria above as the discipline gates.

## What the next session should do

If the anti-homunculus reviewer audits the candidates A+B (combined)
and finds them PASS, the next session:

1. Implements the combined A+B mechanism as a single design unit, with
   the dynamic forms written into both `consolidation.py` and
   `torch_fhrr.py` simultaneously. Adds tests that the existing Phase 4
   D1 graduation is preserved.
2. Runs the consolidation-geometry diagnostic *during training* on a
   1-seed pilot. Verifies d_eff stays in the target range.
3. Re-runs the Phase 5 headline on the new substrate at n=5, then n=10.
4. Reports against the pre-committed falsification criteria above.

If the audit finds A+B FAIL on any anti-homunculus dimension, the
mechanism gets reframed before any code lands. C remains as a backup;
if all three fail, the right move is path (c) — decline Phase 5
graduation and re-scope.

## Closing the loop on the 2026-05-09 prescription

The 2026-05-09 note prescribed this session as the prerequisite for
crossing the threshold from "geometry-as-observation" to
"geometry-as-endogenous-regulation". The death mechanism is exactly
the first concrete crossing. The combined A+B candidate produces a
substrate whose effective dimensionality is *part of the substrate's
energy landscape*, not a metric some controller checks. d_eff becomes
endogenous: it is what the system *is*, not what a watcher *reports*.

If A+B passes review and the retrain produces d_eff ≥ 25 with
branch-divergence restored, the project will have its first
diagnostic-actuator pair in dynamic form. The other pairs from the
2026-05-09 §"Diagnostics vs. Actuators" table (drift / replay-pressure,
bimodality / splitting, metastability / replay-ranking) become the
next sessions in the same shape, each producing one local-dynamic
re-expression of a previously rule-based response.
