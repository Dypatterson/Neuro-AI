# Report 081: Phase 5' Trace-Backed Context Source

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5' bundle-first follow-up to Report 080
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 080 showed that a sampled available-prefix query plan preserves the K=16
context-size threshold:

`scene_token_source=context_bundle_observed_prefix_plan`

That source is stricter than the deterministic observed-prefix curve, but the
query context is still assembled directly from generated scene contents. This
follow-up asks whether the same threshold survives a passive trace-backed
source:

`scene_token_source=context_trace_observed_prefix_plan`

This source builds a `TrajectoryTrace` from the available observed-prefix role
contents, inserts it into a `ReplayStore`, and then uses the retrieved trace
query as the context token. It exercises the Phase 2/4 provenance and replay
storage plumbing, but it is still not a learned or replay-discovered production
trace.

## Artifact

Raw merged Colab JSON is committed at:

`reports/phase5_prime_trace_context_n10.json`

The artifact contains 90 aggregate rows:

- 5 conditions
- `scene_token_weight in {0.1, 0.25, 0.5}`
- `context_roles in {1, 2, 3, 4, 6, 8}`

Each aggregate is n=10 seeds with 512 queries per seed (`n_total=5120`).

The Colab direct CLI run checked out commit `fec8860` before launching the
diagnostic.

## Run Configuration

- `D=4096`
- `K_roles=16`
- `N=512`
- `cue_noise=0.15`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_weight in {0.1, 0.25, 0.5}`
- `scene_token_source=context_trace_observed_prefix_plan`
- `scene_token_pool_size=0`
- `context_roles in {1, 2, 3, 4, 6, 8}`
- conditions: `candidate`, `random_role`, `shuffled_role`, `deranged_role`,
  `content_cleanup_positive`

## Candidate Result

The trace-backed source reproduces the available-prefix threshold from Report
080. One observed role stays at the skewed baseline. Two roles become useful
only when context weight rises. Three to four roles are practical, and six to
eight roles saturate.

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.2316 | 0.2316 | 0.2316 |
| 2 | 0.4453 | 0.7602 | 0.9553 |
| 3 | 0.5447 | 0.9102 | 0.9984 |
| 4 | 0.6389 | 0.9803 | 1.0000 |
| 6 | 0.7748 | 0.9982 | 1.0000 |
| 8 | 0.8639 | 1.0000 | 1.0000 |

The same conservative fixed point remains intact:

`context_roles=4`, `scene_token_weight=0.25`: top1 `0.9803`, CI
`[0.9761,0.9837]`, `scene_tix=0.9799`, `content_tix=0.9803`.

Scene and content ticket rates continue to track candidate top1, so the
remaining errors are scene/context-completion errors, not content-cleanup
failures.

## Controls

`content_cleanup_positive` is `1.0000` for every weight and context-size cell.

Role-negative controls stay near zero:

| control | max top1 | max Wilson hi | note |
| --- | ---: | ---: | --- |
| `random_role` | 0.0008 | 0.0020 | max at `context_roles=1`; correct-scene signal does not recover content under the wrong role |
| `deranged_role` | 0.0004 | 0.0014 | max at `w=0.25,context_roles=6`; high scene_tix is insufficient without the correct role-specific unbinding path |

`deranged_role` remains the cleaner load-bearing negative: content is not
recovered by scene/context completion alone.

`shuffled_role` remains bounded but dirty:

| context_roles | w=0.1 | w=0.25 | w=0.5 |
| ---: | ---: | ---: | ---: |
| 1 | 0.0045 | 0.0072 | 0.0168 |
| 2 | 0.0057 | 0.0148 | 0.0686 |
| 3 | 0.0068 | 0.0277 | 0.0795 |
| 4 | 0.0078 | 0.0402 | 0.0797 |
| 6 | 0.0094 | 0.0625 | 0.0797 |
| 8 | 0.0117 | 0.0756 | 0.0797 |

The maximum shuffled-role cell is `0.0797`, CI `[0.0726,0.0874]`, far below
the candidate curve at the selected operating point but still nonzero enough
to keep it classified as a residual/dirty control.

## Interpretation

The trace-backed source preserves the same bundle-first structural-memory
pattern:

- candidate performance rises smoothly with available context size and weight;
- `context_roles=4,w=0.25` remains high without saturating every control;
- `random_role` and `deranged_role` stay effectively zero even when scene
  identification is high;
- content cleanup remains solved once the right role-specific content query is
  presented.

This strengthens the leading path:

`context/scene completion -> role unbinding -> content cleanup`

The result is narrower than a production claim. The query-side context now
passes through `TrajectoryTrace` and `ReplayStore`, but the trace is still
constructed from generated scene contents and the storage-side scene token is
still the full scene bundle. This is trace-backed provenance plumbing, not a
learned, replay-derived, or naturally observed context trace.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No leave-one-seed-out sensitivity.
- No natural-corpus or learned-codebook version.
- No learned or replay-derived context trace yet.

## Next Work

Do not run the full matrix yet. The next useful discriminator should keep the
same matched controls while replacing synthetic trace contents with a less
synthetic source:

- actual replay-derived or trajectory-derived context trace from Phase 3/4;
- fixed novelty-preserving presentation/storage if a synthetic trace is still
  needed;
- learned or observed passive context trace before treating this as a Phase 5'
  architecture candidate;
- controls: `random_role`, `deranged_role`, `shuffled_role`;
- positives where relevant: content cleanup, bundle/perfect cue.

Only after that context-source choice is fixed should the larger Phase 5'
matrix be run.
