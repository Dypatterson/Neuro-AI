# Report 041 — Phase 5 headline ΔE, n=5 partial

**Date:** 2026-05-20
**Phase:** 5 (HAM × energy-guided structural branching, [design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** **PARTIAL — n=5, NOT graduation.** Phase 5 checklist D1 requires n_seeds ≥ 10.
**Aggregator:** [scripts/aggregate_phase5_de.py](../scripts/aggregate_phase5_de.py)
**Per-seed JSONs:** [reports/phase5_headline_n5/seed{1,2,11,17,23}/](phase5_headline_n5/)
**Aggregate JSON:** [reports/phase5_headline_n5/aggregate.json](phase5_headline_n5/aggregate.json)

## Experiment preamble

**Active phase:** 5
**Headline metric per [phase-5-checklist.md:39](../notes/emergent-codebook/phase-5-checklist.md):**
Δ final-state energy `E_A − E_B` (content-prior minus role-prior) > 0 with
95% CI disjoint from zero, on a held-out cue set designed for structural
retrieval, n_seeds ≥ 10.

**Required controls per [phase-5-checklist.md:52-58](../notes/emergent-codebook/phase-5-checklist.md):**
- B1 random-schema (logged as `random_K4` in exp 40 output; not paired here)
- B2 K=1 single-branch (logged as `K1` ΔE)
- B3 γ=0 no-prior (logged as `K4_g0` ΔE)
- B4 no-schema-store — *not run in this partial pass*

**Last verified result:** [report 038](038_phase4_d1_graduation.md) (Phase 4 D1
graduated, n=10).

**Why this experiment now:** STATUS.md "next session entry point" recommended
**C then B**: pull the five W=4 post-death snapshots locally then aggregate.
This run is the first multi-seed Phase 5 headline data; n=5 is half the
graduation bar but exercises the full pipeline (snapshot → headline cues
→ paired ΔE → aggregator) end-to-end and surfaces real signal.

## Substrate

Phase 4 post-death substrates, W=4, step 1800, captured via
`experiments/19_phase34_integrated.py --snapshot-steps` on Colab A100,
synced via `scripts/colab_phase5_snapshots.ipynb`. Five seeds {1, 2, 11,
17, 23}, dim=4096, 4 positions, n_atoms in {6, 10, 12, 12, 12}. Seed 23
has only 6 surviving atoms — the smallest post-death substrate in the
batch — and the .pt file is correspondingly smaller (332 kB vs 463–529
kB for the others).

## Conditions and exp 40 invocation

`experiments/40_phase5_branching.py --mode headline --formulation
per_pattern --gamma 0.5 --beta 10.0 --n-cues 20`. Per-pattern formulation
is the one closed by Decision-5 ([commit a9c576f](https://github.com/dypatterson/Neuro-AI/commit/a9c576f)).
Per-cue role-binding cue generation with `binding_noise_std=0.05` and
`content_distortion=0.6`. Three headline-relevant tags compared:

- **K4** — K_main=4 schema-prior branches + energy-weighted bundle +
  re-settle (the Phase 5 design as specified)
- **K1** — K_main=1 single schema-prior branch (no branching). Maps to
  checklist control B2
- **K4_g0** — K_main=4 with γ=0 (prior contributes nothing). Maps to
  checklist control B3

ΔE is paired per cue: `E_content_branch − E_role_branch` aggregated across
20 cues per seed. Positive means role-prior found a lower-energy state
(structural retrieval succeeded).

## Headline result — K4 (the Phase 5 design)

| Statistic | Value |
| --- | --- |
| Across-seed Δ (per-seed mean is the unit) | **−1.522e-4** |
| 95% t-CI (n=5, df=4) | [−5.192e-4, +2.148e-4] |
| **CI excludes zero** | **No** |
| Pooled per-cue Δ | −1.522e-4 |
| Pooled bootstrap 95% CI | [−4.453e-4, +5.176e-6] |
| Seeds positive | 3 / 5 (Wilson [0.566, 1.000]) |
| Cues positive | 38 / 100 (Wilson [0.291, 0.478]) |

Per-seed:

| Seed | n_atoms | Mean ΔE | Fraction+ |
| ---: | ---: | ---: | ---: |
|   1 | 12 | **−6.741e-04** | 0.35 |
|   2 | 12 | −1.037e-04 | 0.10 |
|  11 | 10 | +1.563e-05 | 0.70 |
|  17 | 12 | +8.404e-07 | 0.65 |
|  23 |  6 | +1.550e-07 | 0.10 |

The mean is dominated by seed 1's −6.7e-4 (4× the absolute magnitude of
any other seed). Seeds 11, 17, and 23 are weakly positive; seeds 1 and 2
are negative. Drilling into LOSO: removing seed 1 flips the n=4 mean from
−1.5e-4 to −2.2e-5; none of the n=4 LOSO subsets produce a CI excluding
zero, but the dispersion is large.

**Headline assessment:** **K4 ΔE CI includes zero at n=5.** The Phase 5
design as specified does not show structural retrieval beating
content-prior retrieval on the post-death W=4 substrate at this sample
size. This is a partial result, not a graduation result — n=10 is the
required bar per checklist D1 — but the direction at n=5 is *not* "small
positive trending right way." The sign across seeds is split, and seed 1
dominates the mean negatively.

## Control K1 (no branching, single schema-prior) — checklist B2

| Statistic | Value |
| --- | --- |
| Across-seed Δ | **+1.275e-04** |
| 95% t-CI (n=5, df=4) | [−1.130e-04, +3.680e-04] |
| Across-seed CI excludes zero | No |
| Pooled per-cue Δ | +1.275e-04 |
| **Pooled bootstrap 95% CI** | **[+3.494e-5, +2.840e-4]** (excludes 0) |
| Seeds positive | **5 / 5** (Wilson [0.566, 1.000]) |
| Cues positive | 83 / 100 (Wilson [0.745, 0.891]) |

Per-seed:

| Seed | Mean ΔE | Fraction+ |
| ---: | ---: | ---: |
|   1 | +4.075e-05 | 0.65 |
|   2 | +4.727e-04 | 0.80 |
|  11 | +6.737e-05 | 0.85 |
|  17 | +2.287e-05 | 0.95 |
|  23 | +3.368e-05 | 0.90 |

**Interpretation per checklist B2.** The B2 control prediction
([phase-5-checklist.md:55](../notes/emergent-codebook/phase-5-checklist.md))
is: "If single-branch with role-prior matches K-branch with role-prior,
branching is gratuitous." At n=5 here, K1 not only *matches* K4 — it
outperforms it. K1 is 5/5 seeds positive; K4 is 3/5. K1's pooled
bootstrap CI excludes zero; K4's does not. The role-prior is doing
useful work; the energy-weighted bundle over K=4 schema-prior branches
is destroying that effect on small post-death substrates (6–12 atoms,
schema store size = min(k=8, n_atoms) = 6–8). K=4 schema-priors covers
50–100% of the schema store, leaving no diversity headroom.

This is the kind of result H4 of the checklist exists to discipline:
**we do not declare graduation by switching to the condition that
worked**. The headline is K4. K1 is a control, and its passing while K4
fails is a falsification signal for the branching mechanism at this
substrate scale — not a graduation route.

## Control K4_g0 (γ=0, no prior) — checklist B3

ΔE = +0.000e+00 exactly, identical across all 5 seeds, every cue. With
γ=0 the prior weight vanishes from the biased energy and `content` and
`role` priors produce identical settling trajectories; the paired ΔE is
identically zero. **Sanity check on B3 passes by construction.** This
also implies the small but real K1 and K4 effects above are entirely
driven by the prior; nothing in the unbiased energy or settling
dynamics depends on which prior was chosen except through the γ-weighted
term.

## Per-seed substrate snapshot provenance

All five seeds load from
`reports/phase5_snapshots_local/seed{N}/phase3_phase4_w4_step1800.pt`,
each containing `patterns` (n_atoms × 4096 complex64), `positions` (4 ×
4096 complex64), `consolidation`, and `metadata`. Pulled from Drive
folders `Neuro-AI-Snapshots/phase5_snapshots_seed{N}/snapshots/` on
2026-05-20.

## Checklist status update

| Item | Pre-this-report | After this report |
| --- | :---: | :---: |
| A1 (ΔE CI-disjoint, n≥10) | ❌ | 🟨 partial (n=5, CI includes zero) |
| B1 (random-schema control) | ❌ | 🟨 logged but not paired against role |
| B2 (K=1 control) | ❌ | ⚠️ **fires** — K1 ≥ K4 at n=5 |
| B3 (γ=0 control) | ❌ | ✅ ΔE=0 by construction, 5/5 seeds |
| B4 (no-schema-store) | ❌ | ❌ not run |
| C1 (post-death source) | ❌ | 🟨 n=5 (this report) |
| C2–C5 (other schema sources) | ❌ | ❌ not run |
| D1 (n≥10) | ❌ | ❌ at n=5 |
| D2 (LOSO CI excludes zero) | ❌ | ❌ no LOSO subset excludes zero for K4 |
| D3 (seed-23 readout) | ❌ | ✅ +1.55e-7 (smallest positive; not outlier) |
| E1–E6 (drill-downs) | ❌ | ❌ not aggregated multi-seed |
| F1–F5 (readout audit) | ❌ | ❌ not run |
| G1 (W=3 D1 tiebreaker) | ❌ | ❌ not run (W=4 only here) |

## What this report does NOT claim

- It does **not** claim Phase 5 has graduated.
- It does **not** claim K1 is the right Phase 5 design; H4 forbids
  selecting the working condition as the headline.
- It does **not** claim K4 is falsified at the graduation bar — n=5 is
  half the bar; n=10 may produce a different picture. But the n=5
  direction is "split, possibly negative" not "small positive."
- It does **not** run schema-source robustness (section C), readout audit
  (section F), or the W=3 D1 tiebreaker (section G).

## Implications for next session

Three plausible reads:

1. **Real falsification of K-branch at small substrates.** With 6–12 atoms
   and schema store size 6–8, K=4 branches has no diversity headroom; the
   bundle+re-settle step is destabilizing rather than aggregating. This
   would be a Phase-5-design finding ("branching gratuitous when n_atoms
   < some threshold"), not a Phase-5-design success.
2. **Per-seed dynamics variance.** Seed 1's −6.7e-4 is 4× any other
   magnitude. Without seed 1, K4 across-seed mean is −2.2e-5 (one OOM
   smaller). The K4 effect could become genuinely small-positive at n=10
   if the next 5 seeds resemble {11, 17, 23} more than {1, 2}.
3. **Single hyperparameter point is undersampled.** γ=0.5 and K_main=4
   are the design defaults but have not been swept. A γ sweep (decision
   #2 in the design doc) or K_main sweep (decision #3) might reveal that
   the bundle+re-settle is well-behaved at smaller γ or larger K_main.

The disciplined next steps, in order:

- **n=10 first.** Per checklist D1 the graduation bar is n_seeds ≥ 10.
  Pull 5 more W=4 post-death snapshots (seeds 7, 3, 5, 13, 19 — matching
  the Phase 4 n=10 set from report 038), re-run exp 40, re-aggregate. If
  n=10 K4 CI excludes zero positive, A1 passes against the headline
  bar.
- **Then schema-source robustness (section C)** — five conditions on the
  same seeds. If C2 or C4 produce ΔE > 0 CI-disjoint where C1 does not,
  the H4 prohibition says report it; do not declare graduation.
- **Drill-downs (E1–E6) on the same data** — they may explain whether K1
  is winning because the bundle step is the wrong shape, or because the
  diversity filter is collapsing all four schema-priors onto the same
  schema.

The headline at n=5 does not graduate Phase 5. It also does not
falsify Phase 5. It produces enough signal to schedule the n=10
extension as the next concrete unit of work.
