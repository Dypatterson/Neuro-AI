---
date: 2026-05-20
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: design-note
session-closes: STATUS-path-3-precommitment
---

# Cue-Regime / Role-Prior Dynamic Form (Path 3)

Companion to
[2026-05-20 diagnostic-actuator death-dynamic note](2026-05-20-diagnostic-actuator-death-dynamic-form.md)
(Path 1). Held immediately after that note's design + implementation
landed, while the 1-seed A+B retrain pilot runs in the background. Per
[report 043](../../reports/043_phase5_substrate_scale_diagnostic.md)
§"Implications for next steps":

> Death-mechanism redesign is necessary but **not sufficient** —
> even with a denser substrate, the role-prior asymmetry isn't
> consistently positive (3/5 K4, 2/5 K1).

That residual problem — the bimodal ΔE-sign across seeds, *persisting*
on the dense pre-death substrate where K-branch divergence is restored
— is what this note designs against. Closes the
[phase-5-checklist.md §J carried Phase 4 items](../emergent-codebook/phase-5-checklist.md)
"Path 3 design note pre-commitment" — the H1 binding requires this note
to exist *before* any cue-regime sweep is committed.

Like the death-dynamic note, this is **design-only**. No implementation
commitment until anti-homunculus reviewer audit + a separate decision.

## What this session is for

Per the 2026-05-09 framing:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of.

The cue-regime / role-prior question is: **what slow-timescale dynamic
of the substrate is "per-seed ΔE direction" a fast-timescale snapshot
of?** Not "what cue-parameter setting fixes the bimodality."

The 2026-05-09 note's warning at line 158 ("the temptation will be
strongest exactly here, because the standard reading is much easier to
write down and code") was illustrated in real time during the
death-dynamic session — a step-3 hysteresis flag had to be reframed
out. Expect the same temptation here. The wrong-shape candidate is:

> If seed produces negative ΔE on this cue regime, switch cue
> parameters / schema source / formulation until ΔE goes positive.

That is **H1 in disguise** (and would be a [phase-5-checklist.md:191](../emergent-codebook/phase-5-checklist.md)
H1 violation — schema-source ablation is diagnostic-only). The right
move is to identify the local dynamic, not the rule.

## What the diagnostic established

[Report 043](../../reports/043_phase5_substrate_scale_diagnostic.md)
finding, n=5 pre-death seeds at W=4 with role-binding cues
(`binding_noise_std=0.05`, `content_distortion=0.6`):

| Seed | ΔE_K4    | ΔE_K1     | K4 frac+ | K1 frac+ |
| ---: | -------: | --------: | -------: | -------: |
|  17  | +8.06e-4 | **+4.06e-3** | 0.20    | 0.45    |
|  11  | **+2.48e-3** | −4.75e-4 | 0.45    | 0.10    |
|  23  | −2.41e-3 | **+3.90e-3** | 0.40    | 0.55    |
|   1  | −7.60e-5 | −2.85e-5  | 0.15    | 0.15    |
|   2  | +1.60e-5 | −8.82e-4  | 0.35    | 0.35    |

ΔE-sign is bimodal across seeds even on the dense substrate. Seeds 17,
11, 23 show ≥ +2e-3 effects in at least one K; seeds 1 and 2 give
near-zero or negative ΔE on both. The K-branch divergence is restored
(report 043's discrimination finding) — branches DO separate; they
just *don't reliably separate role-prior-better-than-content-prior*.

Report 043 hypothesizes the cue⊛substrate interaction:

> The cue generator (`_build_role_binding_cues` with
> `binding_noise_std=0.05`, `content_distortion=0.6`) produces cues
> whose "role" component is extracted from cue ⊛ position⁻¹ — that
> decomposition's quality depends on the substrate's position-vector
> geometry, which is seed-dependent in subtle ways.

The seed-dependent geometric quantity is **how well the substrate's
position vectors `{position[r]}_{r=1..W}` themselves discriminate** —
i.e., per-seed `d_eff` and `d̄` over the W position vectors, *separate
from* the per-seed d_eff over patterns that A+B addresses.

This is the diagnostic. The dynamic-form question is what
slow-timescale process is the bimodal ΔE a snapshot of, such that the
natural evolution of that process produces a substrate where role-prior
vs content-prior is a *visible* property invariant to seed.

## The wrong-shape candidates

Three "controllers in disguise" the design must reject up front:

### W1 — Per-seed cue-parameter calibration

"For each seed, run a sweep over `binding_noise_std`,
`content_distortion`, and the role-extraction depth; pick the setting
that produces consistently positive ΔE."

Shape: a controller reads ΔE on a held-out cue set, sweeps cue
parameters, selects a winning configuration. The selection criterion
is the same metric the headline measures.

H1 violation: this is `if metric ≥ X: keep this cue regime`. The
parameter selection IS the controller. The 2026-05-09 note's "temptation
will be strongest exactly here" applies — this is the easy code to
write down and the hardest shape to defend.

### W2 — Adaptive prior weighting (γ schedule per seed)

"For seeds where ΔE is negative, increase γ until role-prior dominates
content-prior; for seeds where ΔE is positive, leave γ alone."

Shape: a per-seed feedback loop on γ. The γ value adapts to observed
ΔE during evaluation.

H4 violation: γ is set once before evaluation. Adapting γ from observed
ΔE is parameter-tuning to land the metric in target range.

### W3 — Categorical role/content branch selection

"Run both role-prior and content-prior branches; pick the lower-energy
one per cue." (This is in the existing design as the headline structure
but as *the test*, not as the *runtime* behavior.)

Shape: a per-cue argmin on energy across categorical branch types. The
"role vs content" is a discrete choice the runtime makes per cue.

This is more subtle — the existing design already does this for the
*test*. The wrong-shape version is when the runtime architecture (not
the test) adopts the same argmin shape. The system retrieving from its
own substrate must not be running per-cue branch arbitration.

## Candidates: continuous local dynamics

Three candidates re-express the bimodality as a slow-timescale dynamic
of which (per-seed ΔE direction) is a fast-timescale snapshot. None
reads ΔE during retrieval and acts. Each is local per-atom or per-cue
and continuous in form.

### Candidate α — Position-vector geometry as a second H_anti term

**Slow dynamic.** Add a second dimensionality-preserving repulsion
term to the substrate's energy:

> `H_pos(positions) = -α_pos · log(d_eff(positions))`

where `positions ∈ C^{W×D}` is the substrate's set of W role-marker
vectors. The substrate's total H_anti becomes `H_anti(P) + H_pos(R)`.
Positions evolve under `dr_w/dt = -∂H_pos/∂r_w` at the same
slow-timescale gradient flow as Candidate B for patterns.

**Diagnostic snapshot.** `d_eff(positions)` is the participation ratio
of the position-vector Gram. Seeds where positions are crowded (low
d_eff) have degraded role-decomposition quality; seeds where positions
are spread (high d_eff) have clean role-decomposition. The bimodal ΔE
is downstream of bimodal `d_eff(positions)`.

**Anti-homunculus check.** Identical shape to B for patterns:
α_pos is set once at substrate construction; no metric reads d_eff
during retrieval; the gradient flow is continuous and local-per-position.
PASS.

**Drawback.** Positions are the *reference frame* for binding/unbinding.
If positions evolve during retraining, the previously-bound patterns
must co-evolve so the bind/unbind invariant `unbind(bind(x, r), r) ≈ x`
holds across the trajectory. This is a non-trivial coupled-flow problem.
Practical risk: if positions move faster than patterns can co-adapt,
retrieval degrades catastrophically.

**Mitigation.** Couple the two gradient flows so positions and patterns
move *together*: each pattern is `p_i = Σ_r r_w ⊛ filler_{i,w}`. Apply
the gradient to fillers (the schema-content), not to the bound patterns
or the positions, and rebundle. The fillers become the substrate's
fundamental state; positions are fixed.

**This collapses Candidate α back to Candidate B over fillers.** The
position-vector geometry is then a *passive consequence* of the filler
geometry, not a separate dynamic. Candidate α as stated does not stand
on its own.

### Candidate β — Per-schema role-fidelity continuous weighting

**Slow dynamic.** Each schema atom `s_i` has a per-schema continuous
"role-fidelity" property `f_i ∈ [0, 1]`:

> `f_i = d̄_{positions decomposition of s_i}`

i.e., the mean pairwise FHRR distance among `{unbind(s_i, r_w)}_{w=1..W}`.
Schemas whose unbind decomposition produces distinct per-position
fillers have `f_i ≈ 1` (clean role-binding); schemas whose decomposition
collapses (fillers near-identical across positions) have `f_i ≈ 0`
(role-binding degenerate).

`f_i` is **not stored** as a per-atom variable. It is computed on
demand from the substrate's geometry — a property of the schema, not a
controller-set parameter.

Phase 5's prior over schemas becomes a *continuous weighted sum* over
the schema store:

> `prior(cue) = normalize(Σ_i (cue · s_i)^p · f_i^q · s_i)`

where `(cue · s_i)^p` is the cue-similarity weight (the role-prior
component) and `f_i^q` is the role-fidelity weight (substrate-derived).
At `q=0` the prior collapses to the existing content-prior; at `q=1`
the prior continuously up-weights schemas whose role decomposition is
clean. No categorical "role-prior vs content-prior" branch object.

**Diagnostic snapshot.** Histogram of `{f_i}` across schemas. Seeds
with bimodal {f_i} (some schemas clean, others degenerate) produce
bimodal contribution to the prior; seeds with uniform low or high
{f_i} produce uniform contribution. The bimodality across seeds is
re-expressed as a property of the distribution of `f_i` per substrate.

**Anti-homunculus check.** `f_i` is a deterministic geometric property
of each schema; the weighted sum is a continuous superposition; no
controller reads f_i and routes. The headline test changes from "is
ΔE > 0" (categorical) to "does the role-fidelity-weighted prior land
at lower energy than the content-only prior (q=0)" — but the *test*
applying an argmin to compare two priors is a measurement, not a
runtime mechanism. PASS, contingent on the runtime not making
per-cue choices over the prior. (The runtime uses ONE prior — the
fidelity-weighted one.)

**Drawback.** Changes the Phase 5 headline metric. The 2026-05-09
note's "headline metric principle" warns against this: phase
graduation requires a single fixed headline. Re-defining it
mid-flight is a discipline cost. Defensible only if the new headline
is substrate-pure (per the 2026-05-16 substrate-vs-readout discipline
note) and the old headline is *already* identified as readout-coupled,
which the bimodal ΔE finding suggests.

### Candidate γ — Cue-distribution averaging at evaluation

**Slow dynamic.** None — this is a *test discipline* candidate, not a
mechanism. Currently the Phase 5 headline measures ΔE on N cues
generated at fixed `binding_noise_std=0.05`, `content_distortion=0.6`.
The bimodality across seeds reflects the substrate's interaction with
*this specific cue regime*. Re-express the test as an average over a
*distribution* of cue regimes:

> ΔE_headline = E_{(σ,δ) ~ Uniform([0, 0.1] × [0.4, 0.8])} [ΔE(σ, δ)]

The cue regime becomes a distribution; the headline is the
substrate's expected behavior across that distribution.

**Diagnostic snapshot.** The per-cue-regime ΔE is the integrand; the
substrate's "structural retrieval" capability is the integral.

**Anti-homunculus check.** The cue regime IS the test-set
generator; randomizing it across a fixed distribution is the same
shape as multi-seed averaging — a measurement, not an arbitration.
The runtime doesn't read the cue regime; it just receives cues. PASS.

**Drawback.** This addresses the *variance* of the headline across
seeds, not the underlying mechanism. If the substrate genuinely has no
role-prior advantage on seeds 1 and 2 (rather than the cue regime
hitting their position-geometry pathology), averaging across cue
regimes won't reveal a positive effect — it'll show "the substrate
has no consistent role advantage", which is honest but doesn't
graduate Phase 5. This is the right answer if path α/β both fail;
it's not the right answer if the architecture can be reshaped to
produce a consistent effect.

## Recommended candidate

**Candidate β, with γ as a test-discipline companion.**

β is the cleanest dynamic-form reading: the substrate's per-schema
role-fidelity becomes a continuous weighted contribution to the
runtime prior, eliminating the categorical role-vs-content branch
that flips sign per seed. The headline test reshapes around a
substrate-pure quantity (the *fidelity-weighted prior's energy
advantage over q=0*) rather than a categorical sign.

γ should run *alongside* β as the new headline's test methodology —
averaging across a cue-regime distribution so the test doesn't lock
onto one regime's pathology.

α is reduced to a sub-component of B (filler-level repulsion already
covered by A+B) and does not add new mechanism.

## Pre-committed falsification criteria

Before any Phase 4 retrain with β + γ:

- **Per-seed `{f_i}` distribution coherence.** Across 5 seeds, the
  spread of `mean(f_i)` per substrate should be within 30% of the
  cross-seed mean. (If seeds vary by 5× in mean fidelity, the cue
  regime can't be the only source of bimodality.)
- **Fidelity-weighted prior vs q=0 ΔE.** On the cue-regime
  distribution from γ, the fidelity-weighted (`q=1`) prior must
  produce ΔE > 0 with 95% CI disjoint from zero, n_seeds ≥ 10.
  This is the new headline.
- **q-sweep monotonicity.** ΔE(q) is a continuous function of `q ∈
  [0, 1]`. Per-seed, ΔE(q) should be monotonically non-decreasing in
  q (the more weight on substrate-derived fidelity, the more energy
  advantage). Bimodal q-sweep curves (positive in some seeds,
  negative in others) flag the same problem one level deeper —
  substrate's fidelity *direction* is also seed-dependent, in which
  case the architecture has a deeper issue than this candidate
  addresses.
- **Anti-cherry-pick discipline.** The cue-regime distribution and
  the q-value used for headline are **pre-committed before
  observation**. Re-running with a different cue-regime distribution
  if the first one fails is H1.

If β passes all four, Phase 5 has a substrate-pure headline that
isn't seed-dependent. If β fails on any, candidate γ becomes the
fallback (declared Phase 5 outcome: "structural retrieval is real on
some seeds but the substrate doesn't produce it consistently") and
the project re-scopes.

## What this design note explicitly does NOT do

- It does NOT commit to a Phase 4 retrain. β implementation requires
  anti-homunculus reviewer audit + a separate decision.
- It does NOT supersede the 2026-05-20 death-dynamic note. The A+B
  pilot must complete and pass mechanism-validity before β is even
  worth implementing. If A+B fails, the substrate has more
  fundamental issues than β addresses.
- It does NOT change the Phase 5 headline mid-flight. β's
  substrate-pure headline (q-sweep monotonicity + fidelity-weighted
  ΔE) is a *successor* design; the existing ΔE headline remains the
  measurement for the A+B-only retrain.
- It does NOT calibrate `f_i^q`'s `q` value from observation. `q=1`
  is the principled choice for β (maximum substrate weight). A sweep
  at `q ∈ {0, 0.5, 1.0}` is a diagnostic only.
- It does NOT propose changing `binding_noise_std` or
  `content_distortion` to "fix" the bimodality. γ (cue distribution)
  is the only allowed move on those parameters, and it's only used
  as the *test distribution*, not as a runtime parameter selector.

## Implementation sketch

Files that would change (sketch only, NOT to be implemented from this
note alone):

- `experiments/40_phase5_branching.py` — add a fidelity-weighted prior
  construction path (`q ∈ [0, 1]`). Add a cue-regime distribution
  sampler for γ-style headline evaluation. Both as additional
  conditions, not replacing the existing categorical role/content
  branches.
- `src/energy_memory/phase5/` (new module) — `compute_role_fidelity(s,
  positions, substrate) → float` as a substrate-derived per-schema
  property. No state; pure function.
- `notes/emergent-codebook/phase-5-unified-design.md` — append §"§N
  Successor headline: fidelity-weighted prior" with the q-sweep
  protocol and the cue-regime distribution definition. Do NOT delete
  the existing categorical-branch §"Headline metric" — preserve as
  the A+B-only retrain headline until β is implemented and
  validated.
- `notes/emergent-codebook/phase-5-checklist.md` — add §K
  "Successor-headline pre-commitments" with the four pre-committed
  falsification criteria.

Cost estimate: 1-2 days to implement β + γ on a single seed; 1 day
for re-running across 5 seeds locally; the n=10 retrain rides the
existing infrastructure. Plus the falsification criteria as the
discipline gates.

## Sequencing relative to A+B pilot

This note is **contingent**:

- **If A+B pilot fails** (d_eff < 25 at step 1800): β is
  deprioritized. The substrate has more fundamental issues; back to
  death-mechanism design. This note remains a record of the path-3
  pre-commitment but is not the next move.
- **If A+B pilot passes** (d_eff ≥ 25): β becomes the next major
  architectural threshold. Anti-homunculus audit + implementation
  decision before any compute.
- **If A+B pilot passes and an n=10 Colab retrain is attempted with
  the existing ΔE headline**: per H1, the cue-regime sweep CANNOT be
  used to select a graduation-passing cue regime. The n=10 retrain
  with the existing headline runs unconditionally; β is the
  *successor design* if the existing headline still produces bimodal
  ΔE at n=10.

## Closing the loop on the 2026-05-09 prescription

The 2026-05-09 note named five diagnostic-actuator pairs as
threshold-crossings:

| Pair | Status |
| --- | --- |
| high drift ~ replay pressure | (open) |
| high spread ~ reduced consolidation | A+B closes this (`H_anti`) |
| bimodality ~ splitting pressure | **β addresses this for cue/role-prior bimodality** |
| metastability ~ replay prioritization | (open) |
| low cap-coverage ~ restructuring pressure | (open) |

A+B closed one pair. β would close a second. The remaining three
remain for later sessions, in the same shape — each producing a
local-dynamic re-expression of a previously rule-based response.
