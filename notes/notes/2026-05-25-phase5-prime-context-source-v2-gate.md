---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Context-Source V2 Gate

## Status

Reports 082-084 close the first less-synthetic replay-observed context-source
gate. They are diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment follows from them.

Current blocker from Report 084:

- the passive provenance snapshot has valid rows, but all rows cover only roles
  `{0,1,2,3}`;
- the Report 082/083 hard cell uses `K_roles=16`;
- hard-cell query roles are always outside passive row support;
- learned snapshot pattern vectors align to the synthetic hard-cell context at
  chance.

Do not run another context-source experiment until this v2 gate is implemented
exactly or replaced by a new precommit note.

## Decision

Use the matched 4-role gate first.

Rejected for the immediate next gate:

- **True 16-role provenance gate.** Scientifically cleaner for the original
  hard cell, but it requires generating or locating a new provenance source
  whose traces cover the 16-role universe. That is a larger data-generation
  step and would hide whether the current failure is simply role-universe
  mismatch.
- **Full Phase 5' matrix.** Blocked by Reports 082-084.
- **M2 or headline pivot.** Out of scope.

## Question

If the diagnostic scene role universe is matched to the passive trace role
universe, can replay-observed context recover the bundle-first scene/context
completion effect?

This is a falsifier for the Report 084 mismatch:

- If matched 4-role passive traces still fail, learned/passive trace geometry is
  the blocker.
- If matched 4-role passive traces recover, the K=16 role-universe mismatch was
  the dominant blocker and a true 16-role provenance source becomes the next
  required build item.

## Fixed Gate Cell

Use the same general shape as the Report 082/083 hard cell, but match the role
universe:

```text
D=4096
K_roles=4
N=512
cue_noise=0.15
cooccurrence=skewed
scene_token=1
scene_token_weight=0.25
context_roles=2
n_queries=512
n_seeds=10
C_codebook=2048
```

Rationale:

- `K_roles=4` matches the available passive row support `{0,1,2,3}`.
- `context_roles=2` preserves a genuine held-out query role and avoids handing
  the whole row to the cue. With `K_roles=4`, context sizes 3 or 4 would risk
  becoming a near-complete-row cue or invalid query setup.
- Keep `N=512`, `noise=0.15`, `w=0.25`, and skewed co-occurrence to preserve
  continuity with the Report 081-084 operating point.

## Sources

Run only these source families in the v2 gate:

1. `replay_observed_context_trace`
   - Re-encodes passive row terms through the synthetic gate role/content
     codebook.
   - Tests whether role-universe matching fixes the Report 082 degraded
     passive-term source.
2. `replay_observed_pattern_context_trace`
   - Uses learned snapshot `patterns` as fixed context tokens.
   - Tests whether learned pattern vectors become useful when the scene role
     universe matches the trace role universe.

Do not add generated-scene sources, random anchors, or the full fixed observed
prefix grid to this run. Reports 079-081 already cover those anchors.

## Required Preflight

Before any candidate/control run, write and inspect a preflight JSON with:

- `raw_rows`, `eligible_rows`, `rows_invalid`, `rows_too_short`;
- role counts over the selected source rows;
- role-set counts;
- atom distinct count and normalized atom entropy;
- query-role distribution for the planned matched 4-role query schedule;
- `query_fraction_in_observed_role_support`;
- learned-token pairwise nearest-neighbor stats;
- learned-token alignment to the matched synthetic full context;
- re-encoded partial-context alignment to the matched synthetic full context.

Pass criteria for running the gate:

- `eligible_rows >= N`;
- every selected row has at least `context_roles` valid terms;
- planned query roles are not systematically outside the source role universe;
- learned-token alignment and clustering are reported before any top1 claim.

If any preflight fails, stop and report the preflight. Do not change the gate
into a matrix.

## Conditions

For each source, run the same matched controls on the same seeds, selected
rows, query schedule, and support set:

- `candidate`
- `random_role`
- `deranged_role`
- `shuffled_role`
- `content_cleanup_positive`

Optional positives only if they are directly comparable and kept separate from
candidate evidence:

- `bundle_positive`
- `perfect_cue`

## Required Readout

For each source/condition:

- top1 with Wilson 95% CI;
- per-seed top1;
- leave-one-seed-out candidate sensitivity;
- `scene_tix` and `content_tix`;
- scene/content entropy;
- scene/content margin;
- passive support diagnostics;
- preflight geometry diagnostics.

## Decision Gate

After the matched 4-role run:

- If both passive sources remain degraded and learned-token alignment remains at
  chance, stop. The next work is learned-trace geometry analysis, not a larger
  matrix.
- If `replay_observed_context_trace` recovers but
  `replay_observed_pattern_context_trace` remains weak, the role-universe
  mismatch explains Report 082 but learned-vector geometry remains unsolved.
  Next work is a true 16-role provenance-source plan, not a matrix.
- If both recover with clean controls, then write a new precommit note for a
  true 16-role provenance gate. Do not jump directly to the full Phase 5'
  matrix.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No continuous-learning-across-domains claim.
- No emergent-self claim.
- No adaptive source selection or metric-triggered fallback.
