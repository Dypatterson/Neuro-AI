# Report 079: Phase 5' Fixed Observed-Prefix Grid

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5' bundle-first follow-up to Report 078
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 078 selected a conservative observed-prefix operating point for the
hard K=16 skewed cell:

`scene_token_source=context_bundle_observed_prefix`, `context_roles=4`,
`scene_token_weight=0.25`.

This follow-up asks whether that fixed partial-context setting generalizes
across role count, scene load, and cue noise before spending time on the full
all-controls matrix.

## Artifact

Raw merged Colab JSON is committed at:

`reports/phase5_prime_fixed_observed_prefix_grid_n10.json`

The artifact contains 135 aggregate rows:

- 5 conditions
- `K_roles in {4, 8, 16}`
- `N in {128, 256, 512}`
- `cue_noise in {0.0, 0.10, 0.15}`

Each aggregate is n=10 seeds with 512 queries per seed (`n_total=5120`).

## Run Configuration

- `D=4096`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_source=context_bundle_observed_prefix`
- `scene_token_weight=0.25`
- `scene_token_pool_size=0`
- `context_roles=4`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`

## Candidate Result

The fixed operating point remains high across the full grid. Candidate top1
ranges from `0.9787` to `1.0000`; the worst cell is the largest K/N load:

`K=16,N=512,noise=0.0`: top1 `0.9787`, CI `[0.9744,0.9823]`,
`scene_tix=0.9787`, `content_tix=0.9787`.

| K | N | noise=0.0 | noise=0.10 | noise=0.15 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 128 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 256 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 512 | 0.9998 | 0.9996 | 0.9996 |
| 8 | 128 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 256 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 512 | 0.9998 | 0.9996 | 0.9996 |
| 16 | 128 | 0.9953 | 0.9938 | 0.9939 |
| 16 | 256 | 0.9889 | 0.9883 | 0.9881 |
| 16 | 512 | 0.9787 | 0.9791 | 0.9793 |

Scene and content ticket rates track top1 tightly across candidate cells. The
residual error is therefore still scene/context completion, not content cleanup.

## Controls

`content_cleanup_positive` is `1.0000` for every cell.

The role-negative controls remain near zero:

| control | top1 range | max cell |
| --- | ---: | --- |
| `random_role` | `0.0000-0.0002` | `K=16,N=128,noise=0.0`, CI `[0.0000,0.0011]` |
| `deranged_role` | `0.0000-0.0006` | `K=4,N=512,noise=0.0`, CI `[0.0002,0.0017]` |

The `deranged_role` result is the cleaner load-bearing negative control: even
when scene/context identification is high, content is not recovered without the
correct role unbinding.

`shuffled_role` remains a dirty, nonzero control and is strongest at low load:

| K | shuffled_role top1 range |
| ---: | ---: |
| 4 | `0.2055-0.2617` |
| 8 | `0.1141-0.2234` |
| 16 | `0.0396-0.0439` |

The maximum shuffled-role cell is `K=4,N=128,noise=0.0`: top1 `0.2617`, CI
`[0.2499,0.2739]`. This is far below the weakest candidate cell (`0.9787`) but
high enough to keep `shuffled_role` classified as a dirty residual control
rather than the decisive negative. `random_role` and `deranged_role` carry the
role-unbinding falsification.

## Interpretation

The selected observed-prefix setting generalizes across this fixed grid:

- Candidate stays near-ceiling across `K={4,8}` and remains `>=0.9787` at
  `K=16`.
- Noise in `{0.0,0.10,0.15}` does not drive the dominant failure mode; load
  (`K=16,N=512`) is the harder axis.
- `random_role` and `deranged_role` stay effectively zero despite high
  scene/context identification.
- Content cleanup is not the bottleneck under the positive control.

This strengthens the bundle-first structural-memory interpretation:
context/scene completion can supply the scene trace, role unbinding remains
necessary, and content cleanup succeeds once the right role-specific trace is
present.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No leave-one-seed-out sensitivity.
- No natural-corpus or learned-codebook version.
- No learned/replay-derived context trace yet.
- The context trace is still synthetic observed-prefix context, not a
  production source.

## Next Work

Do not spend the full all-controls matrix yet. The next stricter diagnostic
should make the context trace less synthetic while preserving the same matched
controls:

- learned or replay-derived context trace;
- observed-prefix context from available prefix evidence rather than the full
  synthetic scene object;
- same role-negative controls: `random_role`, `deranged_role`, and
  `shuffled_role`;
- positives where relevant: perfect cue, bundle/content positives.

Only after that stricter context-source choice is fixed should the larger
matrix be worth running.
