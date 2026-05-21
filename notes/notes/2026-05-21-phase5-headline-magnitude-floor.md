---
date: 2026-05-21
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - project/personal-ai
status: pre-commitment
session-closes: phase-5-headline-magnitude-pre-commit
---

# Phase 5 ΔE Headline Magnitude-Floor Pre-Commit

This note pre-commits a **magnitude floor** for the Phase 5 graduation
headline `ΔE = E_content-prior − E_role-prior`, on top of the design
spec's existing "CI disjoint from zero" criterion
([phase-5-unified-design.md:256-281](../emergent-codebook/phase-5-unified-design.md)).

Held immediately before the n=10 Colab graduation run on the
A+B+A1' substrate. Closes the magnitude question raised in the
2026-05-21 sanity-check session.

## Why a magnitude floor is needed

The design spec's graduation criterion is *only* "CI disjoint from zero"
(per [§"Headline metric"](../emergent-codebook/phase-5-unified-design.md#L256)).
At n=10 with low-variance measurements, the substrate could produce a
CI like `[+1e-7, +5e-5]` — technically disjoint from zero, technically
graduating — but with a mean ΔE so small that the "structural retrieval"
property the headline is meant to verify is in practice **smaller than
the substrate's own energy noise floor**. That would be a vacuous
graduation: Phase 6 would inherit a "verified" structural-retrieval
property that has no practical magnitude.

The original Decision-5 spike
([phase-5-unified-design.md:501-534](../emergent-codebook/phase-5-unified-design.md#L501))
illustrates this risk directly:
- Observed ΔE at K=1, N=12 atoms: **+2.6e-5** with 98% directional
- Energy noise scale at N=12: **+5.8e-5**
- The observed signal was **0.45× the noise scale** — directional but
  sub-noise-magnitude

The design note acknowledges (line 525-531) that "larger N_schemas
should yield larger magnitude" and treats the small magnitude as "a
substrate-capacity observation, not a Phase 5 mechanism limitation."
That prediction is **untested** at n=10 on the A+B+A1' substrate. This
pre-commit gates the graduation criterion on the prediction holding.

## Derivation of the floor

For Modern Hopfield retrieval with softmax kernel
`E(state) = -logsumexp(β · scores) / β`:

For a settled state at pattern `p_winner`:
- `score_winner = 1.0` (cosine self-similarity)
- `score_others ≈ 1/√D` (FHRR crosstalk noise floor)

At D=4096, β=10, the per-substrate energy "noise scale" (the energy
mass contributed by non-winner atoms) is:

```python
noise_floor = 1.0 / sqrt(D)              # ≈ 0.0156 at D=4096
correction  = (1/β) · log(1 + (N-1) · exp(-β · (1 - noise_floor)))
```

Computed for several substrate sizes:

| N       | E_settled  | noise-energy scale |
|--------:|-----------:|-------------------:|
| 12      | −1.000058  | **5.8e-5**         |
| 100     | −1.000524  | 5.2e-4             |
| 500     | −1.002614  | 2.6e-3             |
| 1064    | −1.005489  | **5.5e-3**         |
| 4000    | −1.019249  | 1.9e-2             |

**Interpretation.** The noise-energy scale is the energy contribution
from non-winner atoms at the substrate's FHRR noise floor. Any ΔE
signal *below* this scale is statistically indistinguishable from
"role-prior happens to land at a slightly different basin in a way
that's within the substrate's own per-atom noise." A signal *at or
above* this scale is *architecturally distinguishable* — it requires
role-prior to systematically pull the settled state toward a basin
that's energetically distinct from content-prior's at a level the
substrate's geometry can support.

## Pre-committed magnitude floor (binding)

For the A+B+A1' substrate at N=1064 atoms, D=4096, β=10:

> **Mean ΔE ≥ 5.5e-3** (one-sided floor — must be in the *positive*
> direction, since the design spec defines ΔE = E_content − E_role and
> positive means role-prior gives lower energy)

Combined with the design-spec criterion:

> **Mean ΔE ≥ 5.5e-3 AND bootstrap 95% CI lower bound > 0**

The "AND" is binding. Both must hold for the headline to graduate.

## What the floor explicitly forecloses

| Observed outcome | Verdict |
|---|---|
| Mean ΔE ≥ 5.5e-3, CI lower > 0, controls clean | **Phase 5 graduates** (design-spec criterion + magnitude criterion both met) |
| Mean ΔE ∈ (0, 5.5e-3), CI lower > 0 | **Graduation-unattained — directional-but-sub-noise.** Substrate produces statistically detectable preference for role-prior but at magnitude below noise floor; not architecturally meaningful. Phase 5 closes as "spec criterion partially met; magnitude criterion failed." |
| Mean ΔE ≥ 5.5e-3, CI crosses zero | Graduation-unattained — magnitude is meaningful but signal not reliable across seeds. Investigate whether higher n_cues or different seed mix tightens CI. |
| Mean ΔE < 0, CI crosses zero | Graduation-unattained — content-prior outperforms or no signal. |
| Mean ΔE ≪ 0, CI lower < 0 | Phase 5 architectural assumption falsified — role-prior is actively worse than content-prior. |

The second row is the case this pre-commit specifically guards against.

## Why this floor is NOT post-hoc retuning

The discipline binding (H4 / audit constraint #10) forbids retuning
parameters in response to a falsified pre-commit. This magnitude floor
is **derived ex ante from the substrate's first-principles energy
formula** (FHRR noise floor + softmax kernel + substrate-N). It is:

- Not chosen to make the existing Decision-5 spike pass (which gave
  +2.6e-5 at N=12, well below the N=12 noise scale of 5.8e-5 — so the
  Decision-5 result would have *failed* this gate too).
- Not chosen to match an observed n=10 ΔE (no such observation exists yet).
- Derivable independently by anyone reading the design spec + the
  substrate's β, D, N parameters.

The design note itself predicted "larger N_schemas should yield larger
magnitude." This floor tests that prediction at the substrate-noise
scale. If the prediction holds, the floor is cleared. If not, the
prediction was wrong and the substrate doesn't actually produce
structural retrieval at a useful scale.

## Parameters at the run

Locking in here, pre-commit, before the n=10 launches:

- **D** = 4096 (substrate dimension)
- **β** = 10.0 (Hopfield inverse temperature)
- **N** = atoms per scale at step 1800 on A+B+A1' substrate (≈ 1064 for W=4)
- **K** = 1 (matching Decision-5; the design's K=1 control is also the
  cleanest signal per the spike's own analysis: line 513-517 explains
  why K=1 dominates K=4 at small N_schemas)
- **γ** = 0.5 (per Decision-5; the "no-prior" γ=0 condition is the
  required control)
- **Formulation** = per_pattern (the "production code" formulation per
  line 533)
- **n_seeds** = 10
- **n_cues** = 200 per seed (substantially above Decision-5's 50; high
  enough for the bootstrap CI to be tight)

## Required controls (from the design spec — unchanged)

Per [phase-5-unified-design.md:285-293](../emergent-codebook/phase-5-unified-design.md#L285):

1. **Random-schema branches:** role-prior schemas replaced by random
   atoms. Predicts: random ΔE ≈ 0 with CI containing zero. If random
   matches role-prior, the headline is measuring something other than
   role structure.
2. **K=1 (no branching):** already the headline condition above. The
   K-vs-1 comparison is itself a drill-down.
3. **No-prior (γ=0):** the priors don't fire at all. Predicts: ΔE = 0
   identically. If non-zero, the test setup is leaking signal somewhere.
4. **No-schema-store:** priors drawn from codebook directly, not from
   the filtered slow store. Tests whether the schema-filtering step is
   load-bearing.

All four pre-committed to run on the same n=10 seeds.

## What this pre-commit does NOT do

- It does **NOT** change κ, μ_obs, or any pair-#4 parameters — those are
  closed (falsified).
- It does **NOT** propose to test pair #2. That was the recommended
  pivot before the sanity-check session; the sanity-check found Phase
  5's actual headline was never tested. This pre-commit gates that
  un-run headline experiment.
- It does **NOT** modify the substrate or its training. The A+B+A1'
  substrate at the existing seed-17 snapshot
  ([reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt](../../reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt))
  is the substrate this experiment runs against. For n=10, we need 9
  more substrate snapshots — one per seed (or use existing if they
  exist; otherwise run them in the same Colab notebook).
- It does **NOT** reopen the K-branch state_divergence chase. That
  remains closed (a drill-down measured for diagnostic purposes only).

## Sequencing

1. Pre-commit (this note) — DONE.
2. Verify per-seed A+B+A1' snapshots exist or queue them for the
   same Colab notebook (cheaper than separate runs).
3. Colab notebook updated to run the headline experiment with:
   - all four required controls,
   - n=10 seeds × n_cues=200,
   - aggregator computes mean ΔE + bootstrap 95% CI + the magnitude-floor check.
4. Report 053 documents the verdict against the binding criteria above.
