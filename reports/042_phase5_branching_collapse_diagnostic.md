# Report 042 — Phase 5 branching diagnostic: K-branches collapse onto a single attractor

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism diagnostic. **Not graduation, not falsification at the
n≥10 bar.** Decisive evidence on the source of [report 041](041_phase5_de_n5_partial.md)'s
K1≥K4 finding.
**Driver:** [experiments/40_phase5_branching.py](../experiments/40_phase5_branching.py)
(extended with per-cue branch diagnostics + `--k-main` CLI arg)
**Per-seed outputs:** [reports/phase5_headline_n5_diag/](phase5_headline_n5_diag/)
**Sweep outputs:** [reports/phase5_gamma_sweep_seed17/](phase5_gamma_sweep_seed17/),
[reports/phase5_kmain_sweep_seed17/](phase5_kmain_sweep_seed17/)

## Experiment preamble

**Active phase:** 5
**Headline metric per [phase-5-checklist.md:39](../notes/emergent-codebook/phase-5-checklist.md):**
Δ final-state energy CI disjoint from zero, n≥10.

**Required controls per checklist B1–B4:** logged where applicable.

**Last verified result:** [report 041](041_phase5_de_n5_partial.md) — at
n=5, K4 ΔE CI includes zero; K1 outperforms K4 (B2 fires).

**Why this experiment now:** Before extending to n=10 (~hours of Colab),
test whether the K1≥K4 finding is a mechanism failure (K=4 branches
collapse onto one attractor → branching adds nothing) or a sample-size
artifact (n=5 too noisy → n=10 may pull K4 positive). If the former, n=10
is wasted compute on a mechanism that isn't doing what the design claims.
The diagnostic is cheap (local CPU minutes vs Colab hours) and disciplinary.

## Method

1. Extended [experiments/40_phase5_branching.py](../experiments/40_phase5_branching.py)
   headline-mode to log, per condition, the mean across cues of:
   - `mean_branch_softmax_entropy` — entropy of `w_k = softmax(-E_k_unbiased / τ)`.
     Uniform → ln(K). The bundle re-settle weight discriminator.
   - `mean_branch_state_divergence` — mean pairwise cos-distance among
     `{q_k_settled}`. Zero → all branches converged.
   - `mean_prior_alignment` — mean cos(q_settled, prior). High → prior
     held the state.
   - `mean_prior_pairwise_distance` — mean pairwise cos-distance among
     the priors used for this cue.
   - `mean_energy_drop` — mean over branches of unbiased energy at
     q_initial minus unbiased energy at q_settled.
2. Re-ran all 5 seeds (`reports/phase5_snapshots_local/seed{N}/phase3_phase4_w4_step1800.pt`)
   with the diagnostic instrumentation; verified ΔE numerically identical
   to [report 041](041_phase5_de_n5_partial.md) (per-seed determinism intact).
3. Added `--k-main` CLI arg so the headline mode's "main" K can be swept.
4. Ran a γ sweep on seed 17 (γ ∈ {0.1, 0.25, 0.5, 1.0, 2.0}, K_main=4).
5. Ran a K_main sweep on seed 17 (K_main ∈ {1, 2, 3, 4, 6, 8}, γ=0.5).

## Headline diagnostic: branch states collapse

Per-seed `role_K4` diagnostics, dim=4096, 20 cues per seed:

| Seed | n_atoms | softmax_H | state_div (cos-dist) | prior_alignment | prior_pairwise_dist | energy_drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  17 | 12 | 1.3863 (= ln 4) | **6.7e-6** | 0.435 | 0.531 | 0.0713 |
|  11 | 10 | 1.3863 | **2.0e-4** | 0.510 | 0.384 | 0.0525 |
|  23 |  6 | 1.3863 | **1.6e-4** | 0.624 | 0.348 | 0.0445 |
|   1 | 12 | 1.3863 | **2.5e-3** | 0.553 | 0.321 | 0.0460 |
|   2 | 12 | 1.3863 | **2.7e-3** | 0.642 | 0.340 | 0.0359 |

The same for `content_K4`:

| Seed | softmax_H | state_div | prior_alignment | prior_pairwise_dist | energy_drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
|  17 | 1.3863 | 5.2e-5 | 0.614 | 0.510 | 0.0713 |
|  11 | 1.3863 | 4.3e-4 | 0.730 | 0.365 | 0.0525 |
|  23 | 1.3863 | 2.8e-4 | 0.758 | 0.364 | 0.0445 |
|   1 | 1.3863 | 5.5e-3 | 0.769 | 0.356 | 0.0462 |
|   2 | 1.3863 | 4.8e-3 | 0.749 | 0.355 | 0.0358 |

**Interpretation, item by item:**

- **Softmax entropy is exactly ln(K) across all 5 seeds × both prior types.**
  The 4 settled branch energies are numerically identical to floating-point
  precision; the softmax over them is perfectly uniform. The bundle re-settle
  weight `w_k` cannot prefer any branch — it averages 4 equal-weight copies.

- **State divergence is 10⁻⁶ to 10⁻³.** The cos-distance among the K=4
  settled states is 3–6 orders of magnitude smaller than the pairwise
  prior distance (0.32–0.53). The K=4 branches settle to nearly the
  same FHRR state regardless of which schema-prior seeded them.

- **Prior alignment for `role` < `content`.** Role priors are role-binding
  decompositions of cue ⊛ position⁻¹ — partial-content vectors with extra
  noise from the binding extraction. Content priors are the schema atom
  directly. The settled state ends up aligned with content priors more
  strongly because retrieving a stored pattern requires the cue to be
  pattern-shaped.

- **Energy drop is uniform across seeds and prior types** at the
  per-seed level (e.g., seed 17: 0.0713 for both role and content; seed 2:
  ~0.036 for both). The substrate's basin depth dominates; the prior
  doesn't change *how much* energy is dropped, just *which* basin the
  drop terminates in. Since all K branches end in the same basin, energy
  drop is identical across branches.

**Plain-language summary:** the K-branch mechanism does not produce K
distinct attractors. It produces K *seeds* that all flow into the same
basin under the substrate's energy landscape. The bundle+re-settle step
is then averaging K identical points and re-settling on that same basin.
This is structurally why K=4 doesn't beat K=1.

## γ sweep (seed 17, K_main=4)

| γ | ΔE_K4 | ΔE_K1 | K4 state_div | K4 softmax_H | K4 energy_drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | +4.2e-8 | **+8.3e-7** | 2.6e-7 | 1.3863 | 0.0713 |
| 0.25 | +1.9e-7 | **+5.2e-6** | 1.6e-6 | 1.3863 | 0.0713 |
| 0.5 | +8.4e-7 | **+2.3e-5** | 6.7e-6 | 1.3863 | 0.0713 |
| 1.0 | −2.8e-4 | −2.5e-4 | 3.0e-5 | 1.3863 | 0.0713 |
| 2.0 | −2.8e-4 | −2.6e-4 | 1.9e-4 | 1.3863 | 0.0712 |

**At every γ point, ΔE_K1 ≥ ΔE_K4 in the positive direction.** At γ=0.5,
K1 is 27× larger than K4. At γ≥1.0, both flip negative (the prior is
overpowering the substrate; the role prior over-distorts the cue
relative to content). State divergence remains negligible across the
entire γ range. The softmax entropy is exactly ln(4) at every γ — the
prior-biased settling lands branches in the same final unbiased energy
regardless of γ.

**Anti-homunculus note (per checklist H4):** I am not selecting the γ
that maximizes ΔE_K4 to declare graduation. I am observing that K-branch
ΔE is dominated by K=1 ΔE across the entire γ regime. The γ sweep
*falsifies* "γ was tuned wrong" as the cause of B2's fire.

## K_main sweep (seed 17, γ=0.5)

| K_main | ΔE | state_div | softmax_H | energy_drop | n_branches realized |
| ---: | ---: | ---: | ---: | ---: | ---: |
|   1 | **+2.29e-5** | 0.0 | 0.0 | 0.0713 | 1.0 |
|   2 | +2.07e-6 | 6.7e-6 | 0.6931 (= ln 2) | 0.0713 | 2.0 |
|   3 | +8.46e-7 | 6.6e-6 | 1.0986 (= ln 3) | 0.0713 | 3.0 |
|   4 | +8.40e-7 | 6.7e-6 | 1.3863 (= ln 4) | 0.0713 | 4.0 |
|   6 | +2.03e-7 | 1.0e-5 | 1.7918 (= ln 6) | 0.0713 | 6.0 |
|   8 | **0.0** | 2.7e-5 | 2.0794 (= ln 8) | 0.0713 | 8.0 |

**ΔE is monotonically decreasing in K_main.** K=1 is strictly best at
seed 17. K=8 produces ΔE = 0 exactly — because K=8 covers the entire
schema store of size 8, role and content priors select the same set of
schemas (just ranked differently), and after settling and bundling, the
two paired conditions land at numerically identical points.

State divergence stays ~10⁻⁵ across K. Softmax entropy is *exactly* ln(K)
for every K — the K settled branches are equi-energetic for K=2, 3, 4,
6, 8. This is robust evidence that the K branches converge to the same
basin: at minimum, their final unbiased energies agree to floating-point
precision, even with diverse seeds.

## What this rules out

- **Sample-size noise as the source of K1≥K4.** The diagnostic shows
  K=4 branches collapse onto one state across all 5 seeds. n=10 would
  observe the same collapse on 5 new seeds. The mechanism, not the
  sample size, is the issue.
- **γ tuning as the source.** Five γ values from 0.1 to 2.0 produce the
  same K1>K4 ordering. The γ regime is not the problem.
- **K_main=4 being a particular bad K.** K=2, 3, 4, 6, 8 all
  underperform K=1; ΔE is monotonically decreasing. The mechanism scales
  poorly with K on this substrate.
- **B3 (γ=0 control) was already passed:** ΔE_g0 = 0 exactly across all
  5 seeds × 20 cues × multiple γ runs.

## What this does NOT rule out

- **Larger substrates may break the collapse.** The post-death substrate
  has 6–12 atoms; Hopfield basins are wide and the few atoms cover the
  entire reachable space. A larger substrate (pre-death, n_atoms ~ 4000)
  might have narrower basins and produce genuinely distinct K-branch
  settling. **This is exactly the §C2/C4 schema-source comparison** —
  but framed as a diagnostic of whether the *mechanism behaves
  differently* at scale, not as a search for a substrate that produces
  positive ΔE_K4. (Anti-homunculus H4 still binds.)
- **Bundle+re-settle is the wrong combiner.** Even if branches settled
  to distinct basins, the energy-weighted bundle then re-settled might
  still collapse onto one basin. A different combiner (e.g., explicit
  superposition without re-settle, or branch-and-resample) might preserve
  branching's structural information.
- **The cue regime is too easy.** The role-binding cue generator with
  `binding_noise_std=0.05`, `content_distortion=0.6` may produce cues
  that the content-prior can solve in K=1; harder cues might require
  branching to disambiguate.

## Checklist updates

| Item | Pre-this-report | After this report |
| --- | :---: | :---: |
| A1 (ΔE CI-disjoint, n≥10) | 🟨 partial n=5 | ⚠️ partial n=5, **mechanism diagnostic explains why K≥2 underperforms K=1** |
| B2 (K=1 control) | ⚠️ fires at n=5 | ⚠️ fires + mechanism cause identified (branch collapse) |
| E1 (branch-energy dispersion, softmax entropy) | ❌ | 🟨 measured n=5, exactly ln(K) — pathologically uniform |
| E3 (bundle convergence) | ❌ | 🟨 not directly measured; energy_drop is uniform, suggesting yes |
| E5 (branch-state diversity) | ❌ | 🟨 measured n=5, ~10⁻⁵ — branches collapse |
| E6 (prior-domination at γ values) | ❌ | 🟨 γ sweep on seed 17: γ≥1.0 flips ΔE negative |

## Implications for next steps

Three plausible paths, in increasing order of architectural commitment.

1. **Substrate-scale diagnostic.** Re-run the K=4 headline on a pre-death
   (step-1500) snapshot of seed 17 — same substrate-source as the
   checklist §C4 condition, but reframed as "does the mechanism behave
   structurally differently when n_atoms is large enough to produce
   distinct basins?" If state_divergence is still ~10⁻⁵, the K-branch
   mechanism is structurally degenerate regardless of n_atoms. If
   state_divergence rises by orders of magnitude, the K-branch
   mechanism is not structurally degenerate at larger n_atoms — the
   post-death scale was a limiting factor for this particular diagnostic,
   nothing more. Per H4
   ([phase-5-checklist.md:195](../notes/emergent-codebook/phase-5-checklist.md)):
   a positive result here does **NOT** license declaring graduation by
   switching to the larger substrate; it only narrows the failure-mode
   hypothesis and reopens the path-2 question. Any subsequent claim of
   graduation must still satisfy §I.3 of the checklist on its own n≥10
   headline run, on the schema source that produced the diagnostic
   result. Cost: pull one snapshot from Drive + one exp 40 run. Fast.

2. **Combiner redesign.** Two sub-options:
   - **2(a) greedy-argmin** is **ruled out** as a production combiner.
     Per [phase-5-unified-design.md:404](../notes/emergent-codebook/phase-5-unified-design.md)
     ("argmin is a measurement, but if its output controls a downstream
     code path, that's an arbitration. Only allowed as a comparison
     condition, NOT in the production architecture") and
     [experiments/40_phase5_branching.py:850-861](../experiments/40_phase5_branching.py),
     greedy-argmin is already classified comparison-only. Promoting it
     to the production combiner is ruled out by H3
     ("Branch selection is energy-based only, never metric-based")
     *and* by the existing design verdict. Path 2(a) is therefore not
     a redesign path; it is at most a strengthened baseline reported
     alongside the bundle+re-settle headline.
   - **2(b) emit-K-states-as-superposition.** Emit `q_final` as the
     K branch states themselves, treating Phase 5's output as a
     continuous FHRR superposition over states rather than a single
     point. Passes the anti-homunculus check **only if** downstream
     consumers treat `{q_k*}` as a continuous superposition that
     participates in further dynamics, not as a ranked list to be
     selected from. If any downstream module does `argmax` over the
     K states or routes execution based on them, that is H3 by another
     name. This path requires both a downstream consumer spec and a
     fresh anti-homunculus check on the consumer.

3. **Decline Phase 5 graduation under this design.** The mechanism does
   not implement structural retrieval on the post-death substrate. This
   is the most disciplined read of the diagnostic. Phase 5 is re-scoped
   to either (a) the new combiner from path 2, or (b) a larger substrate
   per path 1 succeeding, or (c) a different mechanism entirely.

**My recommendation:** path 1, immediately. It's a 10-minute Colab+local
experiment that disambiguates "small-substrate artifact" from
"structurally degenerate mechanism." That decides whether to invest in
path 2(b) (combiner redesign with consumer spec) or pull back to path 3
(re-scope).

**Pre-commitment per H4
([phase-5-checklist.md:195](../notes/emergent-codebook/phase-5-checklist.md)):**
a positive path-1 result narrows the diagnosis but does **not**
constitute graduation. It triggers either path-2(b) design work, or a
fresh n≥10 headline run that must pass §I in full on the new schema
source. The diagnostic does not license skipping the graduation bar; it
only tells us which mechanism revision (if any) is worth committing the
graduation run on.

n=10 on the W=4 post-death substrate is *not* a useful next step until
the substrate-scale diagnostic tells us whether the mechanism behaves
differently at a different substrate scale.
