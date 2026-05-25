# Report 078: Phase 5′ Observed-Prefix Context-Size Curve

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5′ bundle-first follow-up to Report 077
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 077 showed that strict partial context works: excluding the queried
role/filler pair remains strong, and observed-prefix context has a graded
threshold. This follow-up narrows the K=16 hard cell and asks how much observed
context is needed.

The results below were pasted from the Colab notebook summary output. Raw JSON
was not provided in this session.

## Run Configuration

n=10 seeds, 512 queries per seed, `D=4096`, `K_roles=16`, `N=512`,
`cue_noise=0.15`, skewed co-occurrence, `scene_token=1`,
`scene_token_source=context_bundle_observed_prefix`, `scene_token_pool_size=0`.

Swept:

- `context_roles in {1, 2, 3, 4, 6, 8}`
- `scene_token_weight in {0.1, 0.25, 0.5}`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`

## Candidate Result

Observed context has a clean threshold curve. One observed role stays at the
skewed baseline. Two roles are enough at moderate weight. Three to four roles
are near-ceiling at moderate weight, and six to eight roles saturate.

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2316 | 0.2316 | 0.2316 |
| 2 | 0.4461 | 0.7650 | 0.9590 |
| 3 | 0.5412 | 0.9133 | 0.9969 |
| 4 | 0.6385 | 0.9793 | 1.0000 |
| 6 | 0.7760 | 0.9990 | 1.0000 |
| 8 | 0.8627 | 1.0000 | 1.0000 |

The scene/content split remains tight. Candidate `scene_tix` tracks top1 and
`content_tix` in every cell, so this is still a scene-completion threshold, not
a content-cleanup threshold.

## Controls

`content_cleanup_positive` is `1.0000` for every weight and context-size cell.

`random_role` stays near zero even as scene identification becomes perfect:

- w=0.1: top1 ranges from `0.0000` to `0.0008`.
- w=0.25: top1 ranges from `0.0000` to `0.0008`.
- w=0.5: top1 ranges from `0.0000` to `0.0008`.

`deranged_role` is the cleaner role negative control and also stays near zero:

- w=0.1: top1 ranges from `0.0000` to `0.0004`.
- w=0.25: top1 ranges from `0.0000` to `0.0004`.
- w=0.5: top1 ranges from `0.0000` to `0.0002`.

`shuffled_role` rises with context size and scene identification, but remains a
dirtier control than `deranged_role`:

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0045 | 0.0072 | 0.0168 |
| 2 | 0.0055 | 0.0166 | 0.0699 |
| 3 | 0.0072 | 0.0271 | 0.0791 |
| 4 | 0.0076 | 0.0396 | 0.0797 |
| 6 | 0.0098 | 0.0631 | 0.0797 |
| 8 | 0.0117 | 0.0764 | 0.0797 |

The `deranged_role` result is the load-bearing negative control: high scene
identification does not recover content without the correct role unbinding.

## Interpretation

The K=16 hard cell has a graded context-size threshold:

- `context_roles=1` is insufficient and stays at baseline.
- `context_roles=2` is a weak/minimal operating point: strong only at
  weight `0.5`.
- `context_roles=3` is a practical threshold: `0.9133` at weight `0.25` and
  `0.9969` at weight `0.5`.
- `context_roles=4` is the conservative fixed operating point: `0.9793` at
  weight `0.25` and `1.0000` at weight `0.5`.
- `context_roles>=6` saturates the hard cell.

The most defensible next fixed diagnostic setting is:

`scene_token_source=context_bundle_observed_prefix`,
`context_roles=4`, `scene_token_weight=0.25`.

It is high but not maximally saturated, and its shuffled-role residual
(`0.0396`) is lower than the saturated `w=0.5` residual (`0.0797`) while
`random_role` and `deranged_role` remain near zero.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No leave-one-seed-out sensitivity.
- No natural-corpus or Phase 3/4 learned-codebook version.
- No learned/replay-derived context trace yet.

## Next Work

Before the full matrix, run a fixed observed-prefix operating-point grid:

- `scene_token_source=context_bundle_observed_prefix`
- `context_roles=4`
- `scene_token_weight=0.25`
- `K_roles in {4, 8, 16}`
- `N in {128, 256, 512}`
- `cue_noise in {0.0, 0.10, 0.15}`
- `cooccurrence=skewed`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`

This tests whether the selected partial-context mechanism generalizes across
load and noise before spending time on the full Phase 5′ matrix.
