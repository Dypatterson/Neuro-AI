# Report 080: Phase 5' Available-Prefix Context Source

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5' bundle-first follow-up to Report 079
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 079 showed that a fixed observed-prefix setting generalizes across a
limited K/N/noise grid:

`scene_token_source=context_bundle_observed_prefix`, `context_roles=4`,
`scene_token_weight=0.25`.

That source still selected the query-side observed-prefix roles deterministically
from the generated scene object. This follow-up asks whether the K=16
context-size threshold survives a stricter available-prefix source:

`scene_token_source=context_bundle_observed_prefix_plan`.

This source samples the available non-query context roles per query, always
including the known cue role first and then adding sampled observed roles from
the same scene. It is less synthetic about which roles are available at query
time, but it still uses generated scene contents rather than a learned or
replay-derived trace.

## Artifact

Raw merged Colab JSON is committed at:

`reports/phase5_prime_available_prefix_context_n10.json`

The artifact contains 90 aggregate rows:

- 5 conditions
- `scene_token_weight in {0.1, 0.25, 0.5}`
- `context_roles in {1, 2, 3, 4, 6, 8}`

Each aggregate is n=10 seeds with 512 queries per seed (`n_total=5120`).

The Colab direct CLI run checked out commit `0aa1649` before launching the
diagnostic.

## Run Configuration

- `D=4096`
- `K_roles=16`
- `N=512`
- `cue_noise=0.15`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_weight in {0.1, 0.25, 0.5}`
- `scene_token_source=context_bundle_observed_prefix_plan`
- `scene_token_pool_size=0`
- `context_roles in {1, 2, 3, 4, 6, 8}`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`

## Candidate Result

The available-prefix plan reproduces the Report 078 context-size threshold. One
observed role stays at the skewed baseline. Two roles are strong only at higher
weight. Three to four roles are practical, and six to eight roles saturate.

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2316 | 0.2316 | 0.2316 |
| 2 | 0.4453 | 0.7602 | 0.9553 |
| 3 | 0.5447 | 0.9102 | 0.9984 |
| 4 | 0.6389 | 0.9803 | 1.0000 |
| 6 | 0.7748 | 0.9982 | 1.0000 |
| 8 | 0.8639 | 1.0000 | 1.0000 |

The conservative fixed point remains defensible:

`context_roles=4`, `scene_token_weight=0.25`: top1 `0.9803`, CI
`[0.9761,0.9837]`, `scene_tix=0.9799`, `content_tix=0.9803`.

Scene and content ticket rates track candidate top1 across the curve, so the
remaining error remains scene/context completion rather than content cleanup.

## Controls

`content_cleanup_positive` is `1.0000` for every weight and context-size cell.

Role-negative controls stay near zero:

| control | max top1 | max Wilson hi | note |
| --- | ---: | ---: | --- |
| `random_role` | 0.0008 | 0.0020 | max at `context_roles=1`; content remains absent despite correct-scene signal |
| `deranged_role` | 0.0004 | 0.0014 | max at `w=0.25,context_roles=6`; scene_tix can be high without content recovery |

`deranged_role` remains the cleaner load-bearing negative: content does not
come back without the correct role-specific unbinding path.

`shuffled_role` remains bounded but dirty:

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0045 | 0.0072 | 0.0168 |
| 2 | 0.0057 | 0.0148 | 0.0686 |
| 3 | 0.0068 | 0.0277 | 0.0795 |
| 4 | 0.0078 | 0.0402 | 0.0797 |
| 6 | 0.0094 | 0.0625 | 0.0797 |
| 8 | 0.0117 | 0.0756 | 0.0797 |

The maximum shuffled-role cell is `0.0797`, CI `[0.0752,0.0874]`, far below
the candidate curve at the chosen operating point but still nonzero enough to
keep it classified as a residual/dirty control.

## Interpretation

The available-prefix plan is a stricter query-side source than the prior
deterministic observed-prefix curve, and it preserves the same basic result:

- candidate performance rises smoothly with available context size and weight;
- `context_roles=4,w=0.25` remains high without saturating every control;
- `random_role` and `deranged_role` stay effectively zero even when scene
  identification is high;
- content cleanup is solved once the right role-specific content query is
  presented.

This strengthens the bundle-first structural-memory interpretation:

`context/scene completion -> role unbinding -> content cleanup`

It does not make the context source production-clean. The trace is still
assembled from synthetic scene contents, and the stored scene token remains the
full scene bundle. The next source must be learned, replay-derived, or otherwise
available from a less synthetic passive trace before the full all-controls
matrix is worth running.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No leave-one-seed-out sensitivity.
- No natural-corpus or learned-codebook version.
- No learned or replay-derived context trace yet.

## Next Work

Do not run the full matrix yet. The next useful discriminator should keep the
same matched controls while replacing synthetic observed-prefix context with a
less synthetic trace source:

- learned or replay-derived context trace;
- passive context trace produced by the Phase 3/4 replay/trajectory substrate;
- fixed novelty-preserving presentation/storage if the trace is synthesized;
- controls: `random_role`, `deranged_role`, `shuffled_role`;
- positives where relevant: content cleanup, bundle/perfect cue.

Only after that context-source choice is fixed should the larger Phase 5'
matrix be run.
