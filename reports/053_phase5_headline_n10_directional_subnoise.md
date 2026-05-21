# Report 053 — Phase 5 ΔE Headline at n=10: Directional but Sub-Noise

**Date:** 2026-05-21
**Phase:** 5 (graduation attempt)
**Verdict:** **Graduation-unattained — directional but sub-noise.**
The pre-committed magnitude floor failed; the CI-disjoint-from-zero
half of the gate passed; controls behaved as predicted. Per the binding
pre-commit ([notes/notes/2026-05-21-phase5-headline-magnitude-floor.md](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md)),
this is **not** graduation.

---

## Headline metric and binding gate

Per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):

> ΔE = E_content-prior(q*) − E_role-prior(q*), mean across n_seeds ×
> n_cues with bootstrap 95% CI.

Pre-committed gate ([magnitude-floor note](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md)):

> **mean ΔE ≥ 5.5e-3 AND bootstrap 95% CI lower bound > 0.**

The magnitude floor (5.5e-3) is the substrate's energy noise scale at
D=4096, β=10, N≈1064 — `(1/β)·log(1 + (N−1)·exp(−β·(1−1/√D)))`.

## Run parameters

Substrate: A+B+A1' (alpha_anti=1.0, coverage_lambda=1.0, coverage_ema_rate=0.01,
repulsion_step_size=100.0); snapshots at W=4 step 1800.
Headline: K_main=1, γ=0.5, formulation=per_pattern, β=10.0, n_cues=200 per seed.
Seeds: {17, 11, 23, 1, 2, 3, 5, 7, 13, 29}. n=10.

## Result

| Quantity | Value |
|---|---|
| mean per-cue ΔE | **+0.00130** |
| std per-cue ΔE | 0.00968 |
| mean per-seed mean | +0.00130 |
| bootstrap 95% CI (per-seed means) | **[+0.00071, +0.00193]** |
| Gate 1: mean ΔE ≥ 5.5e-3 | **FAIL** (+0.00130, 4.2× below floor) |
| Gate 2: CI lower bound > 0 | **PASS** (+0.00071) |

Per-seed:

| Seed | n_cues | mean ΔE | frac_positive |
|---|---|---|---|
| 17 | 200 | +0.00267 | 0.30 |
| 11 | 200 | +0.00018 | 0.35 |
| 23 | 200 | +0.00166 | 0.49 |
| 1  | 200 | +0.00112 | 0.34 |
| 2  | 200 | +0.00284 | 0.41 |
| 3  | 200 | +0.00024 | 0.28 |
| 5  | 200 | +0.00113 | 0.32 |
| 7  | 200 | +0.00036 | 0.38 |
| 13 | 200 | +0.00246 | 0.38 |
| 29 | 200 | +0.00035 | 0.42 |

10/10 seeds positive in mean; per-cue frac_positive ranges 0.28–0.49 (all
below 0.50, meaning *most individual cues* see content ≤ role energy even
when the seed mean is positive — the signal is a small per-seed mean of a
high-variance per-cue quantity).

## Controls (all clean)

**Control 1 — random-schema (predicted: random ΔE ≈ 0, CI containing zero):**

mean(content − random) across seeds = **+0.00086 ± 0.00161**.
Random sits between content and role across seeds; the role-prior advantage
over random is comparable in magnitude to content's disadvantage versus
random. Behaves as designed.

**Control 2 — no-prior γ=0 (predicted: ΔE = 0 identically):**

mean ΔE_K1_g0 = **+0.000000** across all 10 seeds (exact). No leak in the
test setup; the prior fires only when γ>0.

## Interpretation against the pre-commit's outcome table

| Observed | Pre-committed verdict |
|---|---|
| mean ΔE ∈ (0, 5.5e-3) AND CI lower > 0 | **Graduation-unattained — directional but sub-noise.** Substrate produces statistically detectable preference for role-prior at magnitude below noise floor; not architecturally meaningful. |

This is the *exact* cell the magnitude-floor pre-commit was designed to
catch. The design-spec criterion ("CI disjoint from zero") is met; the
magnitude criterion is not. Phase 5's prediction that "larger N_schemas
should yield larger magnitude" (design doc lines 525-531) is partially
borne out — N=1064 produced +1.3e-3, vs N=12's +2.6e-5, a 50× scaling —
but the noise scale grew faster (5.8e-5 → 5.5e-3, 95×), so the
signal-to-noise *ratio* is essentially unchanged (0.45 at N=12 → 0.24 at
N=1064). Scaling N did not rescue magnitude in the way the design spec
anticipated.

## What this forecloses and what it doesn't

**Foreclosed (per the pre-commit):**
- "Phase 5 graduates because the headline CI is disjoint from zero."
  Without the magnitude floor, the FAIL row above would have been a vacuous
  graduation: a 4×-below-noise signal verified as "structural retrieval."
  The pre-commit guarded against exactly this.
- "Run more seeds to tighten the CI." The CI already excludes zero; that
  isn't where the gate fails.

**Not foreclosed:**
- The architectural claim that role-prior systematically outperforms
  content-prior at the energy level is consistent with this result —
  the *direction* is robust (10/10 seeds positive). What's foreclosed is
  the claim that the *magnitude* is large enough to constitute structural
  retrieval at a useful scale on this substrate.
- The design spec itself ([phase-5-unified-design.md:285-293](../notes/emergent-codebook/phase-5-unified-design.md))
  remains a valid theoretical framing; the substrate is the binding limit.

## Substrate-wide finding (added to the principles ledger)

The substrate-wide principle established in [report 050](050_phase5_beta_smoke_seed17.md)
and [report 052](052_phase5_pair4_smoke_falsification.md) — "softmax
fixed-point measurements collapse on sharp-basin substrates" — extends
here in a related but distinct form:

> **The substrate's clean-retrieval geometry (engineered through
> A+B+A1') produces sub-noise role-vs-content energy gaps. The same
> sharpness that gives clean retrievals (max_w ≈ 1, basin-self-retrieval
> ratio ≈ 1.0) also makes the role-prior advantage over content-prior a
> small perturbation on top of energies that are already saturated at
> the basin floor.**

This is the third independent substrate-shape finding pointing at the
same root cause: the A+B+A1' substrate is *too clean* for mechanism
classes whose signal depends on per-atom variance, basin-trajectory
diversity, or energy-margin between near-equivalent priors.

## Why this is NOT post-hoc retuning

Per audit constraint #10 ([2026-05-20 metastability note
ADDENDUM](../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md)):
no parameter retuning in response to a falsified pre-commit. None
applied. The graduation criterion was set in the magnitude-floor pre-commit
*before* this n=10 run, derived from the substrate's energy formula not
from observed data.

## Open questions surfaced (not addressed here)

1. **Lower-D substrate.** The 1/√D crosstalk noise floor sets the lower
   bound on per-atom energy ambiguity. A D=512 or D=1024 substrate
   would have a noise floor 8×–4× higher relative to retrieval energies,
   so the per-cue ΔE relative to noise would be ~3–5× larger. The
   magnitude-floor derivation scales with D and would need to be
   re-derived; the gate could still apply with substrate-specific
   numbers. Whether the A+B+A1' substrate-construction principles
   transfer cleanly to lower D is an open empirical question.
2. **Different priors over basins.** The current prior is per-pattern
   on existing atoms. A prior that operates over basin-of-attraction
   geometry rather than pattern identity might produce larger energy
   gaps. This is research-direction territory, not within Phase 5's scope.
3. **Direct cap-coverage / R@K measurement.** ΔE is the headline; the
   downstream phenomena (recall, retrieval accuracy) might still show
   measurable improvements from role-prior even when the energy gap is
   sub-noise. Not measured in this run; would require a different
   experiment specifically designed for that question.

## Pre-commitments still binding (carried forward)

- No retuning of κ, μ_obs, β, γ, K_main, formulation to make this
  graduate post-hoc.
- No re-running with adjusted magnitude floor to make it pass.
- No reframing "directional but sub-noise" as "partial graduation."
- Any Phase 6 work that depends on Phase 5's structural-retrieval
  property MUST cite this report and acknowledge the magnitude is
  sub-noise on this substrate.

## Sequencing / next step

This is a strategic decision point for the user, not an immediate next
experiment. The four substantively distinct directions are:

1. **Close Phase 5 as graduation-unattained, redesign substrate
   construction for a lower-D version, retry the headline there.** This
   is the cleanest research-direction path and the one most consistent
   with the substrate-wide finding.
2. **Close Phase 5 as graduation-unattained, accept the directional
   finding, advance to Phase 6 with explicit acknowledgment that the
   structural-retrieval claim has sub-noise magnitude at D=4096.**
   This treats the architectural prediction as "consistent with theory
   but quantitatively underwhelming on the current substrate."
3. **Reformulate the Phase 5 headline.** Replace ΔE with a downstream
   metric (e.g., R@K, basin-membership classification accuracy) that
   doesn't sit on top of the already-saturated basin-energy. This is
   a redesign of the phase's success criterion and requires user
   sign-off, not implementation in-session.
4. **Investigate a different prior class entirely** (e.g., basin-shape
   priors rather than pattern-identity priors) — research direction.

I do not recommend any of these without explicit user direction.

---

**Artifacts:** per-seed JSONs at
`reports/phase5_headline_n10_seed{17,11,23,1,2,3,5,7,13,29}/` and
worker logs at `reports/phase5_headline_n10_colab/seed{N}.log`. Substrate
snapshots at `reports/phase5_headline_substrate_seed{N}/snapshots/`.
Aggregator cells 7 and 7b of [scripts/colab_phase5_headline_n10.ipynb](../scripts/colab_phase5_headline_n10.ipynb).
