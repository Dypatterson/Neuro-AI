# Report 076: Phase 5′ Context-Bundle Anchor Follow-Up

**Date:** 2026-05-25
**Branch:** `phase5-m1-role-energy-stack`
**Scope:** Phase 5′ bundle-first follow-up to Report 075
**Status:** Diagnostic only. No Phase 5 graduation or Delta E claim.

## Question

Report 075 showed that unique random scene anchors rescue the hard skewed
bundle-first cell, while a global shared anchor stays near baseline. That made
scene/context completion the leading bottleneck interpretation, but random
anchors are not production-clean.

This follow-up replaces random scene IDs with `scene_token_source=context_bundle`:
the role-filler scene bundle itself is used as the substrate-derived context
trace.

## Artifact

Raw merged Colab JSON is committed at:

`reports/phase5_prime_context_anchor_followup_n10.json`

The notebook framing marks this as `not_graduation=true` and uses static
conditions and controls, with no metric-triggered routing or best-of-N
selection.

## Run Configuration

n=10 seeds, 512 queries per seed, `D=4096`, `N=512`, `cue_noise=0.15`,
skewed co-occurrence, `scene_token=1`, `scene_token_pool_size=0`,
`scene_token_source=context_bundle`.

Swept:

- `K_roles in {4, 8, 16}`
- `scene_token_weight in {0.1, 0.25, 0.5, 1.0}`
- conditions: `candidate`, `random_role`, `shuffled_role`,
  `content_cleanup_positive`

## Candidate Result

The context-bundle anchor saturates the hard skewed cells. Even the weakest
weight (`0.1`) is near ceiling.

| K | w | Top1 | Wilson CI | scene_tix | content_tix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.1 | 0.9912 | [0.9883, 0.9934] | 0.9908 | 0.9912 |
| 4 | 0.25 | 1.0000 | [0.9993, 1.0000] | 0.9998 | 1.0000 |
| 4 | 0.5 | 1.0000 | [0.9993, 1.0000] | 0.9998 | 1.0000 |
| 4 | 1.0 | 1.0000 | [0.9993, 1.0000] | 0.9998 | 1.0000 |
| 8 | 0.1 | 0.9965 | [0.9944, 0.9978] | 0.9963 | 0.9965 |
| 8 | 0.25 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| 8 | 0.5 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| 8 | 1.0 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| 16 | 0.1 | 0.9959 | [0.9937, 0.9973] | 0.9959 | 0.9959 |
| 16 | 0.25 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| 16 | 0.5 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| 16 | 1.0 | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |

`content_cleanup_positive` is `1.0000` for every K/weight cell.

## Matched Controls

`random_role` is exactly zero in every cell:

| K | w range | Top1 | Wilson CI upper | scene_tix range |
| ---: | --- | ---: | ---: | ---: |
| 4 | 0.1-1.0 | 0.0000 | 0.0007 | 0.9908-0.9998 |
| 8 | 0.1-1.0 | 0.0000 | 0.0007 | 0.9963-1.0000 |
| 16 | 0.1-1.0 | 0.0000 | 0.0007 | 0.9959-1.0000 |

This split matters: correct scene identification remains high, but the wrong
unbind role destroys content recovery. Scene recovery alone is not enough.

`shuffled_role` is bounded but nonzero, and rises when the context trace
dominates scene identification:

| K | w=0.1 | w=0.25 | w=0.5 | w=1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0951 | 0.2119 | 0.2121 | 0.2498 |
| 8 | 0.1373 | 0.2350 | 0.2354 | 0.2564 |
| 16 | 0.0361 | 0.0797 | 0.0830 | 0.0830 |

The residual is not a pass/fail blocker for this diagnostic, but it tightens
the next control requirement. The current `shuffled_role` implementation can
leave some roles fixed under the random permutation, so the harness now also
includes a `deranged_role` control with no fixed points.

## Interpretation

The result supports the bundle-first decomposition:

`context/scene completion -> role unbinding -> content cleanup`

The substrate-derived full-scene context trace is sufficient to recover the
hard skewed cells at n=10. This is stronger than the random-anchor result
because the anchor is derived from the same role-filler substrate instead of an
external random scene ID.

The result is still not production-clean. Current `context_bundle` uses the
full scene bundle as the context trace, including the queried role/filler.
Therefore it proves sufficiency for full-scene substrate context, not partial
observed-context completion.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No leave-one-seed-out sensitivity.
- No natural-corpus or Phase 3/4 learned-codebook version.
- No learned/replay-derived context trace yet.

## Next Work

Do not run the full matrix yet. The harness and Colab notebook now include
stricter context-anchor variants:

1. `context_bundle_exclude_query_role`: query context excludes the queried
   role/filler pair.
2. `context_bundle_observed_prefix`: query context uses only a fixed observed
   subset of non-query role/filler pairs.
3. `deranged_role`: a role-shuffle control with no fixed points.

Run those next. If the stricter partial-context variants keep candidate high while
`random_role`, `shuffled_role`, and `deranged_role` remain bounded, then run
the broader Phase 5′ matrix with matched controls. Only after that should this
be mapped back to the Phase 5 `Delta E` headline.
