# Report 059 — Phase 5 Log-Prior Spike Local Smoke

**Date:** 2026-05-24
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:269-292](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior - E_role-prior, paired per cue. The C-first log-prior spike is marked as an exploratory Phase 5' diagnostic in [phase-5-unified-design.md:258-267](../notes/emergent-codebook/phase-5-unified-design.md), not a graduation retune.
**Required controls per [phase-5-unified-design.md:296-304](../notes/emergent-codebook/phase-5-unified-design.md):** random-prior, K=1, no-prior, no-schema-store. This smoke includes random-prior and K=1 only; it is not a graduation run.
**Last verified result:** [Report 058](058_phase5_cross_seed_cue_regime_sweep.md) — best locked cue cell reached ΔE/floor = 0.45 with zero role-target basin hits.
**Why this experiment now:** Tier 0 selected Path C first: test whether a Varner-style per-pattern log-multiplicity boost can move the locked Phase 5 K=1 signal on the existing A+B+A1' substrate before choosing closure or scale-down work.

This is a **local smoke**, not a graduation experiment. No retraining, no beta/gamma/K/cue-regime retuning, no Phase 5 graduation claim.

## Setup

- Snapshots: cached A+B+A1' seeds 17, 11, 23 under `reports/phase5_a1prime_pilot_seed17/snapshots/`
- Fixed operating point: `beta=10`, `gamma=0.5`, `K=1`, `formulation=per_pattern`, `binding_noise_std=0.05`, `content_distortion=0.6`
- Gain grid: `log_prior_gain ∈ {0, 1, 2, 4, 6}`
- Cues: 50 per seed, fixed `cue_seed=117`
- Implementation: additive branch-local logit boost on the selected schema atom, with `gain=0` preserving the baseline path

## Aggregate Smoke Results

Mean across seeds 17/11/23:

| gain | mean ΔE | ΔE/floor | random_lowest | role<content | role_hit | role_rank |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.001847 | 0.336 | 0.367 | 0.453 | 0.013 | 475.0 |
| 1 | +0.006350 | 1.155 | 0.360 | 0.480 | 0.013 | 465.8 |
| 2 | +0.020195 | 3.672 | 0.413 | 0.527 | 0.013 | 434.7 |
| 4 | +0.020102 | 3.655 | 0.420 | 0.460 | 0.013 | 306.7 |
| 6 | +0.001434 | 0.261 | 0.327 | 0.300 | 0.013 | 214.1 |

Per-seed best magnitudes:

| seed | cleanest gain | ΔE/floor | largest gain cell | ΔE/floor | random control caveat |
|---:|---:|---:|---:|---:|---|
| 17 | 1 | 1.153 | 2 | 4.148 | gain 2 improves random_lowest vs baseline |
| 11 | 1 | 1.554 | 4 | 3.113 | gains 2/4 worsen random_lowest |
| 23 | 1 | 0.757 | 4 | 5.510 | gains 2/4 worsen random_lowest |

## Reading

**Magnitude moved.** Gain 1 is the cleanest cell: it clears the 5.5e-3 floor on seeds 17 and 11, reaches 0.76x floor on seed 23, and does not materially worsen random-prior domination. Gains 2 and 4 produce much larger ΔE on all three seeds, but the random-prior control worsens on seeds 11 and 23.

**This is a soft pass, not a hard pass.** The hard-pass gate required clearing the magnitude floor without random-prior domination worsening. Gain 1 is close to that standard but not fully cross-seed floor-clearing; gains 2/4 clear magnitude but carry the random-control caveat.

**Basin retrieval still does not recover.** Role-target basin hit stays ~0.013 across gains. Higher gains improve role-target rank, especially gain 4/6, but rank remains far from a credible retrieval claim. The spike moves the energy margin much more than it moves role-target basin membership.

## Recommendation

Run an n=10 Colab confirmation before interpreting Path C. Use the same locked operating point, same gain grid narrowed to `{0, 1, 2, 4}`, and report:

- mean ΔE with CI and the 5.5e-3 floor gate
- random_lowest and role<content<random rates
- role-target basin hit/rank
- per-seed table, because seed-level control behavior differs

If gain 1 clears or nearly clears the floor at n=10 without worsening random-prior domination, Path C remains live. If only gains 2/4 clear magnitude while random dominates more often, Path C should be treated as an energy-margin intervention that does not solve structural retrieval.

## Artifacts

- `reports/phase5_log_prior_spike_seed17.json`
- `reports/phase5_log_prior_spike_seed11.json`
- `reports/phase5_log_prior_spike_seed23.json`
