---
date: 2026-05-17
project: neuro-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - phase-5
  - design
---

# Phase 5 Unified Design — HAM × Energy-Guided Structural Branching

This document specifies the Phase 5 architecture as a refinement of
PROJECT_PLAN's Phase 5 ("Structure and Abstraction"). It commits to a single
unified mechanism — *energy-guided structural branching wrapped around HAM* —
that delivers role/filler binding, atom splitting, and hierarchical retrieval
through one substrate-level dynamic rather than three separate mechanisms.

Supersedes nothing (no prior Phase 5 design existed); read alongside
[phase-4-unified-design.md](phase-4-unified-design.md) and the
[2026-05-16 Phase 4 graduation synthesis](../notes/2026-05-16-phase4-graduation-synthesis.md).

---

## What this design does

Phase 5 takes the graduated Phase 4 substrate (a Benna-Fusi cascade producing
a filtered slow-variable store) and uses it as the source of *schema priors*
that seed multiple competing settling trajectories on the same cue. Each
trajectory ("branch") settles independently under a different prior; their
final energies define a posterior; the system either bundles them (preserving
ambiguity when the substrate genuinely supports multiple meanings) or
collapses to one (when one branch dominates).

The architectural claim:

- **Structural retrieval** emerges when role-binding priors produce
  lower-energy settling than content-similar priors. No structural-matcher
  module; the energy landscape is the matcher.
- **Atom splitting** emerges when two branches converge to substantially
  different states with comparable low energies. No polysemy-detector
  module; branch-energy-and-state geometry is the detector.
- **Hierarchical compression** emerges because schemas (the priors) come
  from the Benna-Fusi filtered slow store, which holds only frequently-
  retrieved patterns. No compression module; the cascade is the filter.

No supervisor decides which branch wins. The "decision" distributes into
local geometry: schemas seed branches, branches settle on energy, branches
combine algebraically.

---

## Architectural diagram

```
                          cue
                           │
                           ▼
                ┌────────────────────┐
                │  schema selector   │  (top-K_main schemas by similarity
                │  (top-K + surprise)│   to cue, + 1 high-novelty branch)
                └────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          branch_1     branch_2  …  branch_K
          (prior=s_1)  (prior=s_2)   (prior=s_K)
              │            │            │
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ HAM     │  │ HAM     │  │ HAM     │      each branch retrieves
        │ settle  │  │ settle  │  │ settle  │      with prior-biased energy
        │ q_1*    │  │ q_2*    │  │ q_K*    │
        └─────────┘  └─────────┘  └─────────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                ┌────────────────────┐
                │  branch combiner   │  (energy-weighted bundle + re-settle,
                │  + atom-splitting  │   greedy/Boltzmann as comparison)
                │  diagnostic        │
                └────────────────────┘
                           │
                           ▼
                    final state q*
                   (+ split signal
                    if applicable)
```

---

## Mechanism specification

### 1. Schema store

**RESOLVED 2026-05-17 by [report 040](../../reports/040_freq_weighted_alpha_sweep.md):**
**the schema source is the post-death substrate.**

Concretely: the surviving 5–30 atoms per seed at W=2 (and the larger surviving
sets at W=3 and W=4) after the Phase 4 mass-death event form the schema
population. These atoms are precisely the ones that received retrieval
reinforcement during the run — mass death is a binary retrieval-frequency
filter applied directly to the substrate. The freq-weighted α coupling
that was originally planned as the schema-filtering mechanism was tested
in report 040 and found redundant at the production regime: mass death
already produces the under-capacity slow store the architecture needs.

At Phase 5 retrieval time, the schema store is therefore just the set of
patterns alive in the consolidation state at all scales. Ranking within
the store (for top-K selection) uses `effective_strength` (the Benna-Fusi
weighted sum across u_1...u_m), preferring slow-variable contribution
via the default weight schedule `2^(1-k)` already in
`ConsolidationConfig.strength_weights`.

For research/comparison only (deferred, not load-bearing):

- **Co-occurrence clusters** — Phase 5 follow-up if the post-death
  substrate at n_cues=3000 turns out to be too small (<20 atoms per scale)
  to seed K=4 diverse schemas per cue.
- **Layer-2 HAM bindings** — exp 21 already produces composite patterns
  by construction; using them as additional priors is a clean Phase 5+
  extension once the basic post-death-substrate mechanism is validated.
- **Hybrid** — slow-store atoms ∪ ham_layer2; Phase 5+.

### 2. Branch seeding

For a cue `c`, generate K branches. **Main path: schema-prior branching.**

For `k = 1 ... K_main`:
- Compute similarity `sim(c, s_j)` against each schema `s_j` in the store.
  Similarity is FHRR cosine: `Re(⟨c, s_j⟩) / (‖c‖ · ‖s_j‖)`.
- Sort schemas by similarity, pick top-K_main *with diversity constraint*:
  if two top-ranked schemas have cosine similarity > δ_redundant to each
  other (e.g., 0.95), skip the lower-ranked one. This prevents the K
  branches from being K copies of the same schema.

For `k = K_main + 1` (the surprise branch):
- Find the pattern with maximum `u_1 / max(u_m, ε)` ratio — the most
  novel pattern that has not yet propagated through the cascade.
- This pattern is the surprise prior.

**Comparison conditions (run alongside in experiments, not in production):**

- **Stochastic perturbation**: cue is perturbed by Gaussian noise in
  FHRR space, K branches with K different noise draws.
- **Top-K commitment**: each branch is forced to commit to a different
  top-K Hopfield basin in its first settling step.

These are *baselines* against which schema-prior branching is compared.

### 3. Per-branch settling

Each branch `k` settles the HAM hierarchy with energy biased by its prior `p_k`.

For layer-1 (the existing Modern Hopfield substrate):
```
E_k(q) = -logsumexp(β · X q*) - γ · Re(⟨q, p_k⟩)
```

where `X` is the stored-pattern matrix, `*` is complex conjugate, and `γ`
is the prior-weight hyperparameter. The settled state is
`q_k* = argmin_q E_k(q)` reached by iterated Hopfield retrieval with the
prior term added to the score.

For layer-2 (HAM layer 2): standard HAM retrieval, but with `q_k*` from
layer-1 as input. The prior `p_k` does not enter layer 2 directly; it
already shaped what reached layer 2.

**Hyperparameter γ.** Default `γ = 0.5`. Sweep range `{0.0, 0.25, 0.5, 1.0, 2.0}`.
- `γ = 0`: prior is informational only (used to pick the branch, not to
  bias retrieval). Equivalent to top-K commitment.
- `γ = ∞`: prior dominates entirely; retrieval reduces to nearest-schema
  lookup.
- The interesting regime is γ ≈ 0.5 — prior nudges but doesn't dictate.

### 4. Branch scoring

**Primary score (used for selection and bundle weights):** the post-settling
**unbiased** Modern Hopfield energy:
```
E_k(q_k*) = -logsumexp(β · X q_k**)
```

The scoring energy *omits* the γ term. The prior was a seed, not
a measurement target — we score how well the settled state matches the
substrate, not how well it matches the prior. This is what makes the
branch-comparison fair: branches with very different priors compete on
the same substrate-level energy.

**Logged-but-not-used for selection:** the γ-biased settling energy
`E_k^biased(q_k*) = E_k(q_k*) - γ · Re(⟨q_k*, p_k⟩)`. The difference
between these two scores (i.e., `γ · Re(⟨q_k*, p_k⟩)`) is the per-branch
**prior-alignment** diagnostic — see §"Per-branch diagnostics" below.

Softmax weights over branches:
```
w_k = softmax_k(-E_k(q_k*) / τ)
```
where `τ` is the temperature (default `τ = 1.0`).

### 5. Branch combination

**Preferred: energy-weighted bundle + re-settle.**
```
q_bundle = Σ_k w_k · q_k*
q* = HopfieldRetrieve(q_bundle, β=β, max_iter=12)   # unbiased re-settle
```

The bundle is an FHRR-space weighted sum. The re-settle is a normal
Hopfield retrieval (no prior term) — it lands `q_bundle` onto a stable
attractor of the unbiased landscape. The whole pipeline reads as a single
posterior-mean approximation followed by gradient descent on the marginal
energy. No controller.

**Comparison conditions:**

- **Greedy argmin**: `q* = q_k*` for `k = argmin_k E_k(q_k*)`.
- **Boltzmann sample**: draw `k ~ Categorical(w_k)`; `q* = q_k*`.

Both are run as comparison conditions to test whether bundling preserves
ambiguity where it should and collapses where it should.

### 6. Atom-splitting diagnostic

After branch combination, compute the **joint criterion** over branches in
the low-energy set (branches with `E_k - E_min < δ_energy`):

```
split_signal = (
    energy_set_size(δ_energy) ≥ 2
    AND
    max_pairwise_state_distance(low_energy_branches) > δ_state
)
```

State distance is FHRR cosine distance: `1 - cos(q_i*, q_j*)`.

- `δ_energy` default: 0.1 (branches within 10% of best energy).
- `δ_state` default: 0.3 (substantial geometric divergence in FHRR space).

When `split_signal == True` for a cue, the cue's nearest schema in the
store is marked **split-eligible**. The actual split mechanism (create a
new schema by separating the two branch states into independent attractors)
is a Phase 5 sub-component spec deferred to a follow-up note. For now we
just measure split-eligibility frequency as a diagnostic.

**Why the joint criterion matters.** Energy-similarity alone would
over-fire on duplicates: two branches that converge to nearly the same
state are *redundant*, not polysemous. State-divergence alone would
over-fire on degenerate cases where one branch lands far from the manifold
at high energy. Both axes are needed.

---

## Headline metric

**Structural retrieval verified iff Δ final-state energy is CI-disjoint from
zero, role-prior branches vs content-prior branches, on a held-out cue set
designed for structural retrieval.**

Concretely:

- For each test cue:
  - **Condition A (content prior)**: branches seeded by top-K schemas by
    content similarity to cue (the default mechanism above).
  - **Condition B (role-prior)**: branches seeded by top-K schemas that
    share *role-fillers* with the cue. Role-fillers are FHRR-bound atoms;
    matching is performed in the binding-decomposed space.
- Compute the final state's unbiased energy `E(q*)` under each condition.
- The cue's score is `ΔE = E_A(q*) - E_B(q*)`. Positive means role-prior
  branching produced a lower-energy final state (better structural match).
- Headline: mean ΔE across n_seeds × n_cues, with 95% CI.

If CI excludes zero, the system retrieves structurally — role priors land
at lower-energy joint states than content priors. **This is the Phase 5
graduation criterion.**

Per the 2026-05-16 substrate-vs-readout discipline note: this headline is
substrate-pure (energy, not readout). R@K and cap-coverage become
drill-downs.

---

## Required controls

| Control | What it tests | Status |
|---|---|---|
| **Random-schema branches** | If random schemas as priors produce the same Δ as role-binding schemas, no structural retrieval is happening. | Required. |
| **K=1 (single-branch, no branching)** | Tests whether the branching adds value over standard HAM retrieval. If single-branch with role-prior matches K-branch with role-prior, branching is gratuitous. | Required. |
| **No-prior** (γ=0) | Tests whether the prior is doing the work, or whether top-K commitment alone would suffice. | Required. |
| **No-schema-store** | Tests whether the schema store is the right source. Run with priors drawn directly from the codebook (not the filtered slow store). | Should-do. |

---

## Drill-downs (substrate-pure)

Per the discipline note: read these to *explain* the headline, not to
replace it.

1. **Branch-energy dispersion distribution.** Entropy of the softmax
   weights `w_k` across cues. Predicted: high entropy on ambiguous cues,
   low entropy on unambiguous ones.
2. **Split-eligibility rate.** Fraction of cues for which the joint
   atom-splitting criterion fires. Predicted: rises with corpus complexity.
3. **Bundle convergence.** Fraction of bundle re-settlings that converge
   (energy below threshold) vs oscillate. If oscillation is common, the
   bundling mechanism is broken architecturally.
4. **Schema-store utilization.** What fraction of schemas in the store
   are ever picked as a top-K prior for any cue? If low (say <10%), most
   schemas are useless and the store needs pruning.
5. **Branch-state diversity.** Mean pairwise FHRR distance among the K
   settled states `{q_k*}`. Predicted: rises monotonically with cue
   complexity (a clean cue → all branches converge to one place; a
   complex cue → diverse branch endpoints).
6. **Prior-domination rate.** Fraction of branches with
   `prior_alignment(q_k*, p_k) > δ_dominated`. Sweep δ_dominated in
   {0.5, 0.75, 0.9}. Detects the γ regime where the prior overpowers
   memory support — if this rises sharply with γ, the system is reducing
   to nearest-schema lookup. Predicted: low under γ ≤ 0.5, rises with γ.

---

## Per-branch diagnostics (logged, not used for selection)

Per the substrate-vs-readout discipline note: selection stays
energy-based, but interpretation requires a multi-diagnostic per-branch
record. Each settled branch `q_k*` logs:

| Field | Definition |
|---|---|
| `energy_unbiased`        | `E_k(q_k*)` — the score used for selection (above). |
| `energy_biased`          | `E_k(q_k*) - γ · Re(⟨q_k*, p_k⟩)` — the settling-time energy. |
| `energy_drop`            | `E_k(q_k^{0}) - E_k(q_k*)` — how much energy dropped during settling. Measures basin strength. |
| `prior_alignment`        | `Re(⟨q_k*, p_k⟩) / (‖q_k*‖ · ‖p_k‖)` — cosine similarity to the seed prior. Higher = prior dominated. |
| `score_entropy_initial`  | `H(softmax(β · X q_k^{0}*))` — score distribution entropy at start of settling. |
| `score_entropy_final`    | `H(softmax(β · X q_k**))` — at convergence. |
| `entropy_collapse`       | `score_entropy_initial - score_entropy_final` — measures retrieval decisiveness. |
| `final_state_divergence` | Mean cosine distance to other branches' final states. |
| `recall_support`         | Boolean: does the target token appear in the top-decode of `q_k*`? |
| `cap_coverage_t05`       | Standard cap_t_05 readout applied to this branch's final state. |
| `meta_stable`            | Boolean: top decode score < 0.95 (meta-stable per Phase 2 metric). |
| `structural_match`       | FHRR-binding decomposition: unbind position vectors from `q_k*` and compare extracted filler atoms to the cue's bindings. High score = matching role structure, not just content. |
| `converged`              | Boolean: did the per-branch retrieval converge below an iter-count budget? |

These are recorded per (cue, branch, condition). The aggregate analysis
slices them by condition to interpret the headline. **None of them feed
back into the selection mechanism** — that's energy-only, by design, to
keep the anti-homunculus check clean.

---

## Mixed-branch policy

`K = K_main + 1`, where the +1 is the surprise branch.

**Default K_main = 4.** Total K = 5. Compute cost per cue: ~5× standard
HAM retrieval. Sweep range: `K_main ∈ {2, 4, 8}`.

The surprise branch is the architectural safety valve against the
freq-α slow store excluding rare-but-important patterns. **It is not
optional.** Without it, the system is structurally incapable of
surfacing a pattern that has never consolidated to the slow store but
is currently relevant to the cue.

Concrete novelty score for the surprise branch:
```
novelty_score(pattern) = log(u_1 + ε) - log(u_m + ε)
```
(Log-ratio form rather than raw ratio; raw `u_1 / max(u_m, ε)` is dominated
by tiny u_m values and produces artificial surprise explosions on patterns
whose slow variable has decayed numerically near zero. The log-difference
form is symmetric, bounded by the dynamic range of `u_k`, and well-behaved
for ε ≈ 1e-6.)

The pattern with the highest novelty score becomes the surprise prior.
Ties broken randomly.

**Surprise-branch instrumentation.** Track per-cue: did the surprise
branch win (lowest E_k)? Did it produce the largest energy drop during
settling? Did its prior alignment exceed a sanity threshold? Did its
re-settle converge? If the surprise branch consistently loses or
destabilizes the bundle, the safety-valve framing is wrong and we
revise; if it consistently wins, the system is over-weighting novelty
and we should reduce its share of K. **This diagnostic is required, not
optional**, because the surprise branch is the one place we add a
mechanism whose contribution can fail silently in either direction.

---

## Anti-homunculus check

Per the standing project rule: every mechanism must be a local geometric
dynamic or a measurement of one, never an arbitration over them.

| Component | Check |
|---|---|
| Schema selection | Pure similarity ranking (algebraic). Pass. |
| Surprise branch | Argmax over a local ratio `u_1 / u_m`. Pass as a measurement. |
| Per-branch settling | Standard Modern Hopfield retrieval with one extra scalar in the energy. Pass (well-known FEP-clean dynamic). |
| Branch scoring | Energy of the settled state. Pass (passive measurement). |
| Softmax weights | Standard probabilistic operation defined by energies. Pass. |
| **Bundle + re-settle** | Weighted FHRR sum (algebraic) followed by unbiased Hopfield retrieval (a known FEP-clean dynamic). **The pipeline reads as a single posterior-mean approximation followed by gradient descent on the marginal energy.** Pass cleanly. |
| Greedy argmin | Borderline: argmin is a measurement, but if its output controls a downstream code path, that's an arbitration. Only allowed as a comparison condition, NOT in the production architecture. |
| Atom-splitting diagnostic | Pure measurement (joint criterion on energy + state). Pass; the *split action* is a separate mechanism deferred to a follow-up. |

The greedy argmin caveat is critical: this is why bundling is the
*preferred* combination rule. Bundling reads as one continuous gradient
descent; greedy reads as an if-X-then-Y rule even if X is a measurement.

---

## What's in scope for Phase 5

- The full branching pipeline above, implemented as a wrapper around the
  existing HAM machinery (exp 21).
- The headline structural-retrieval experiment with required controls.
- All five drill-downs measured per-checkpoint.
- The atom-splitting diagnostic as a *measurement* (not an action).
- Mixed-branch policy with surprise branch.

## What's deferred to Phase 5+ follow-up notes

- **Atom-splitting *action* mechanism**: once an atom is marked split-
  eligible, what actually splits? Two candidate mechanisms (independent-
  attractor creation, schema-store partitioning) are sketched but not
  specified.
- **Cross-cue learning**: if branch energies are stable across many cues
  for a given schema, that schema becomes a *concept*. The promotion
  mechanism is Phase 5+.
- **Sleep/wake interleaving**: when branching runs (during retrieval) vs
  when bundling propagates back to substrate (during replay) — Phase 5
  sub-rhythm spec is deferred.
- **Phase 6 reuse**: this mechanism's machinery becomes Phase 6's temporal
  rollout substrate. The Phase 6 design will inherit but not duplicate.

---

## Open decisions before implementation start

| # | Decision | Status |
|---|---|---|
| 1 | What populates the schema store? | ✅ **CLOSED** by [report 040](../../reports/040_freq_weighted_alpha_sweep.md): post-death substrate (5–30 atoms per seed at W=2; more at W=3/W=4), ranked by `effective_strength`. |
| 2 | γ (prior weight in per-branch energy). Default 0.5; sweep `{0.0, 0.25, 0.5, 1.0, 2.0}`. | Settled at first experimental run; CPU spike. |
| 3 | K_main (branches per cue). Default 4; sweep `{2, 4, 8}`. | Settled at first experimental run; compute-cost question, not architectural. |
| 4 | Bundle re-settle convergence. Verify numerically that the bundled state lands cleanly on an attractor (no oscillation, no high-energy outcome). | Settled at implementation start; ~30-min CPU spike on synthetic cues. |

Decision 1 was the architectural blocker. With it closed, decisions 2–4 are
all addressable during implementation as numerical hyperparameters.
**Phase 5 implementation is unblocked.**

---

## Integration with Phase 4 substrate

Phase 5 *does not modify* the Phase 4 substrate. It wraps it:

- **Codebook**: unchanged. Still the phase3c reconstruction codebook (or
  whatever Phase 4 produces from online Hebbian).
- **Modern Hopfield retrieval**: unchanged in its mathematics; only the
  energy function is augmented with the prior term (one extra scalar add).
- **Benna-Fusi cascade**: unchanged. Its filtered slow store is *read by*
  Phase 5 (as the schema source) but not modified.
- **Pattern death**: unchanged.
- **A_k self-inhibition**: unchanged (off by default per report 036).
- **Online Hebbian**: unchanged.

This means a Phase 5 implementation can be developed and tested without
risking regression in Phase 4's graduated behavior. Phase 4 graduation
([report 038](../../reports/038_phase4_d1_graduation.md)) remains valid;
Phase 5 either passes its own graduation (the structural-retrieval headline)
or doesn't.

---

## Reading order for someone catching up to Phase 5

1. This file.
2. The 2026-05-16 Phase 4 graduation synthesis
   ([notes/notes/2026-05-16-phase4-graduation-synthesis.md](../notes/2026-05-16-phase4-graduation-synthesis.md))
   — what Phase 5 inherits.
3. The 2026-05-16 substrate-vs-readout discipline note
   ([notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md](../notes/2026-05-16-substrate-vs-readout-metric-discipline.md))
   — why the headline is energy, not R@K.
4. The freq-weighted α experiment plan
   ([notes/notes/2026-05-16-freq-weighted-alpha-experiment-plan.md](../notes/2026-05-16-freq-weighted-alpha-experiment-plan.md))
   and its eventual result — what schemas come from.
5. The Phase 4 unified design ([phase-4-unified-design.md](phase-4-unified-design.md))
   — the substrate this builds on.
6. Brainstorm idea 5 (freq-weighted α) and idea 7 (REM-phase generative
   replay) in [brainstorm-neuro-personal-ai.md](../../brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md) —
   the upstream design rationale.
