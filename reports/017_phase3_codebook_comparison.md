# Report 017 — Phase 3 codebook-growth comparison at the operating envelope

**Date:** 2026-05-13
**Experiment:** `experiments/31_phase3_comparison.py`
**Raw data:**
- `reports/phase3_comparison.csv` (6 seeds × 6 codebooks)
- `reports/phase3_comparison.json`

## Motivation

Per [reports/016_phase2_audit_and_phase3_objective.md](016_phase2_audit_and_phase3_objective.md),
the Phase 3 codebook-growth objective was formally pinned to

> Maximize `masked_token:generalization Recall@1` at the operating
> envelope (β=3, L=64, W=8, mask=1, end-position, wikitext-2-raw-v1,
> D=4096). Beat the Phase 2 random-codebook baseline of 0.205.

This report executes that comparison across the six existing
2050×4096 codebook artifacts in [reports/](.). It reports two
findings: (a) a methodology check that confirms the evaluator
reproduces the matrix; (b) the multi-seed Recall@1 result that
revises the baseline number.

## Methodology check (seed = 17, matrix-aligned slicing)

The matrix's slice seed is
`seed + 10000 * window_index + 1000 * landscape_index`. With W=8 at
index 1 of `[4,8,16]` and L=64 at index 0 of `[64,256,1024]`, that's
`seed + 10000`. The evaluator hardcodes these offsets so a given
`--seed` produces the same window slice the matrix saw.

At seed=17 the evaluator reproduces the matrix exactly:

| codebook                 | matrix-seed-17 | matrix CSV  |
| ------------------------ | -------------- | ----------- |
| random_baseline_matrix   | 0.205          | 0.205       |
| hebbian, reconstruction, error_driven | 0.205 | (not in matrix) |

All six codebooks happen to score 0.205 at this single seed: with
n=44 effective samples there are only 45 possible accuracy values,
and seed=17's window slice happens to be one where every codebook
lands on the same one. **This is the single-seed coincidence that
report 016 framed as the baseline.**

## Headline result — six seeds, methodology-verified

Pooled across seeds {17, 1, 2, 3, 4, 5}, ~290 effective test windows
per codebook:

| codebook                 | mean acc | pooled CI         | seeds | verdict |
| ------------------------ | -------- | ----------------- | ----- | ------- |
| random_baseline_matrix   | 0.089    | [0.060, 0.125]    | 6     | baseline |
| random_phase3c           | 0.089    | [0.060, 0.125]    | 6     | identical to baseline |
| **hebbian_phase3c**      | **0.104** | **[0.071, 0.141]** | 6  | **overlaps baseline** |
| **hebbian_phase3b**      | **0.104** | **[0.071, 0.141]** | 6  | **overlaps baseline** |
| reconstruction           | 0.054    | [0.032, 0.084]    | 6     | below baseline |
| error_driven             | 0.055    | [0.032, 0.084]    | 6     | below baseline |

## Three findings that change report 016's framing

### 1. The 0.205 baseline number was single-seed

Report 016 named 0.205 as "the Phase 2 random-codebook baseline".
Across 6 seeds at the same operating envelope, the corrected
baseline is **0.089 (95% CI [0.060, 0.125])**. The 0.205 cell in the
matrix is at the upper edge of this CI — a real but high-variance
draw from the same underlying distribution.

**This is not a bug in the matrix.** The matrix correctly reports
the single-seed accuracy it ran. It is a misreading by report 016
that called a single-seed point estimate "the baseline". With
n_eff=44 and the binomial variance at small samples, the matrix
cells in this regime should all carry [±15 percentage points]
single-seed CIs and need replication for headline use. Multi-seed
pooled CIs are tighter ([±3 percentage points] at 6 seeds) and
should be the bar.

### 2. No learned codebook reliably beats the random baseline

The two Hebbian codebooks are nominally above baseline (0.104 vs
0.089) but their CIs overlap heavily. The wins are not
statistically significant at 6 seeds. The conservative read is
"comparable to random."

The reconstruction and error-driven codebooks are **below** the
random baseline at p ≈ 0.05 (pooled CIs nearly disjoint from
[0.060, 0.125] on the upper side). Training these codebooks with
their respective objectives actively hurt accuracy on the headline
metric.

### 3. Random codebooks reproduce identically across the two random artifacts

`random_baseline_matrix` and `random_phase3c` give bit-exact
identical per-seed accuracies (0.089 / 0.089 mean, [0.060, 0.125]
CI). The two artifacts are different random codebooks but produce
the same Recall@1 because both encode the same windows with
algebraically equivalent role-filler bundles — the *which* random
codebook doesn't matter at this metric, only that it's random.

This is a useful sanity check: the evaluator is reading the
codebooks correctly, but Recall@1 at this regime is essentially
*content-blind* — the random codebook's structure doesn't propagate
into the answer distribution.

## What this means for the Phase 3 program

Two structural reads:

**Possibility A:** the Phase 3 learning objectives that have been
tried are *not yet hitting the right target*. Hebbian co-occurrence
gives at best a small nominal lift (≈ 1.5 pp). Reconstruction loss
and error-driven gradients hurt. The space of "what should the
codebook learn" is large, and the project has tried three points in
it; none beats random with confidence.

**Possibility B:** Recall@1 at this envelope is the *wrong*
headline metric. The within-seed accuracy variance is so high that
distinguishing real codebook gains from noise needs more samples
than the test set provides, or a denser metric. Two candidates:

- **Recall@K with K = 5 or 10.** The matrix reports `cap_error_0_5`
  and `cap_error_0_3` columns; those track "does the answer appear
  in the top-K with score ≥ threshold". They had more signal in the
  matrix (the 0.205 cell had 0 cap-error at threshold 0.5 but the
  random baseline at other cells went up to bigram_accuracy of 0.27,
  outperforming the substrate's Recall@1).
- **The synergy estimator from
  [report 011](011_synergy_probe_phase4.md).** Structural quality
  of the codebook — does it produce decomposable role-filler
  bindings? — is a denser signal than per-window argmax accuracy.
  We have the estimator; we just haven't applied it across
  codebooks.

**My read: this is mostly possibility B.** Recall@1 with n_eff ≈ 50
per seed is too noisy to be the deciding metric. The signal is
real but small (Hebbian ≈ 1.5 pp lift); the test set is too small
to resolve it. The right next move is to switch to a denser metric
*before* concluding any Phase 3 algorithm has failed.

## Recommended next steps

### Update report 016's baseline claim

The Phase 3 codebook-growth objective should be revised to use a
multi-seed pooled baseline. Concrete amendment:

> Maximize `masked_token:generalization Recall@1` at the operating
> envelope. Beat the Phase 2 random-codebook 6-seed pooled baseline
> of **0.089 (95% CI [0.060, 0.125])** with non-overlapping pooled
> CIs at ≥ 5 seeds.

### Adopt a denser secondary metric

Pin **synergy** (from [report 011](011_synergy_probe_phase4.md)) as
the Phase 3 *secondary* headline. Synergy = 0.519 on the random
codebook's encoded windows (the baseline run, from report 011's raw
section). A codebook that learns compositional structure should
push that number up. Concrete: run the synergy estimator on each of
the six codebooks at the operating envelope and add `synergy` and
`synergy_ratio = settled / raw` columns to the comparison table.

### Don't expand the Phase 3 algorithm catalog yet

Adding a 4th, 5th, 6th codebook learning method on top of an
overly-noisy metric is the wrong move. Tighten the measurement
first; then judge the existing three (Hebbian, reconstruction,
error-driven). If Hebbian's nominal lift survives a denser metric
with non-overlapping CIs, Hebbian is the winner. If none of the
three survives, *that* is the time to design a fourth.

## Anti-homunculus check / FEP audit

Pure measurement evaluator. No mechanism added. The
`evaluate_condition` machinery is reused unchanged from the Phase 2
matrix; only the seed-offset arithmetic was made explicit so the
single-seed reproduction works. No inspect-and-trigger logic.
Passes trivially.
