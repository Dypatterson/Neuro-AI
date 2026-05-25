---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
---

# Phase 5 Prime Less-Synthetic Context Trace Gate

## Status

Report 082 has now run this narrow gate with
`scene_token_source=replay_observed_context_trace`. The result is diagnostic
only: no Phase 5 graduation, no Delta E headline, no full matrix, and no M2
commitment.

## Anchor

Report 081 is the current context-source anchor. Its hard cell is:

- `D=4096`
- `K_roles=16`
- `N=512`
- `cue_noise=0.15`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_source=context_trace_observed_prefix_plan`
- `context_roles=4`
- `scene_token_weight=0.25`
- `n_queries=512`
- `n_seeds=10`

At that cell, candidate top1 is `0.9803`, Wilson CI `[0.9761, 0.9837]`,
`scene_tix=0.9799`, `content_tix=0.9803`; `random_role` and `deranged_role`
remain near zero, `shuffled_role` is bounded but dirty, and
`content_cleanup_positive` is `1.0000`.

Report 081 still constructs the trace from generated scene contents. It
exercises `TrajectoryTrace` and `ReplayStore` plumbing, but it is not yet a
learned, replay-derived, trajectory-derived, or naturally observed context
trace.

## Next diagnostic

The specified narrow context-source discriminator before any full Phase 5'
matrix is:

```text
scene_token_source = replay_observed_context_trace
```

Definition:

1. Build or reuse a passive `ReplayStore` whose traces come from an actual
   Phase 3/4 trajectory-producing path, not from direct construction of a
   generated scene row inside the MQAR harness.
2. Select a fixed number of observed non-query roles from a stored trace's
   `encoder_terms`. For the Report 081 anchor, keep `context_roles=4`.
3. Encode the context token from the trace-derived role/atom terms only.
   The queried role/filler must not be included in the context token.
4. If the trace store does not contain enough support for a matched anchor
   cell, stop and report the support deficit. Do not silently fall back to
   direct generated-scene context.

This is a passive trace-source test. It is not a new routing mechanism and it
does not choose a source based on observed top1 or Delta E.

## Fixed run cell

Keep the Report 081 hard cell fixed:

```text
D=4096
K_roles=16
N=512
cue_noise=0.15
cooccurrence=skewed
scene_token=1
scene_token_weight=0.25
context_roles=4
n_queries=512
n_seeds=10
```

The point of this run is source provenance, not matrix coverage. Do not add
`K_roles`, `N`, `noise`, or weight sweeps until this source question is settled.

## Matched controls

Run all controls on the same seeds, query schedule, source, and support set:

- `candidate`
- `random_role`
- `deranged_role`
- `shuffled_role`
- `content_cleanup_positive`

Add `bundle_positive` or `perfect_cue` only if they are directly comparable to
the trace-source path. They are positives, not substitute evidence for the
candidate condition.

## Required readout

Report:

- top1 with Wilson 95% CI
- per-seed top1
- `scene_tix` and `content_tix`
- scene/content entropy
- scene/content margin
- support diagnostics for the passive trace source
- explicit boundary text: no Phase 5 graduation, no Delta E headline, no full
  matrix, no M2 commitment

## Stop conditions

Stop and ask before proceeding if the implementation requires any of these:

- adaptive source selection or metric-triggered fallback
- a full Phase 5' matrix
- M2, MQAR, bAbI, or other headline-pivot work
- repository-wide status claims about continuous learning across multiple
  domains or emergent selfhood
- reclassifying a diagnostic top1 result as Phase 5 graduation evidence

## Interpretation

Report 082 is a middle result: the signal survives a less-synthetic passive
trace source versus matched role-negative controls, but drops from Report 081's
`0.9803` to `0.6762`. That supports continuing the bundle-first path, but it
also makes the next bounded step artifact/per-seed diagnostics, not a full
matrix. Any broader claim remains blocked until the active Phase 5 spec and
checklist criteria are explicitly met or amended.
