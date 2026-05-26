---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Natural Source Control-Cleanup Gate

## Status

Report 092 preflighted stricter source/control hygiene after Report 091 and
found the strictest tested protocol feasible:

`non_special_unique_target_freq_le_32`

The selected plans have:

- `512` queries per seed;
- no `<UNK>` / `<MASK>` targets;
- no same-row target duplicates;
- zero same-scene exact-match opportunity for `random_role`,
  `deranged_role`, and `fixedpoint_free_shuffled_role`;
- ample support (`min eligible triples = 13808` per seed).

## Decision

Run exactly one fixed gate for the Report 092 recommended protocol.

This is a changed query/control protocol, so it must be labeled separately
from Report 090. It tests whether non-synthetic candidate recovery survives
source/control hygiene.

## Fixed Cell

Use the same hard context-source operating point:

```text
D=4096
K_roles=16
N=512
cue_noise=0.15
scene_token=1
scene_token_weight=0.25
context_roles=4
n_queries=512
n_seeds=10
C_codebook=2048
cooccurrence=repo_sample_natural
```

Seeds:

```text
17 11 23 1 2 3 5 7 13 29
```

Source:

`trajectory_native_provenance_context_trace`

Query/control protocol:

`non_special_unique_target_freq_le_32`

## Conditions

Run exactly:

- `candidate`
- `random_role`
- `deranged_role`
- `fixedpoint_free_shuffled_role`
- `content_cleanup_positive`

Do not run legacy shuffled-role in this gate. Report 091 showed that the legacy
permutation contains fixed points and is a weaker control for natural sources.

## Readout

Report:

- top1 with Wilson 95% CI;
- per-seed top1;
- leave-one-seed-out sensitivity;
- `scene_tix` and `content_tix`;
- scene/content entropy;
- scene/content margin;
- source/preflight/query-plan SHA;
- explicit no-graduation boundary.

## Decision Gate

- **Clean recovery:** candidate remains high (`>=0.90`) and random/deranged/
  fixed-point-free shuffled controls are `<=0.005`, with positive cleanup
  solved. This permits a new integration precommit, not graduation.
- **Candidate preserved but controls dirty:** stop and analyze residuals again.
- **Candidate collapses:** source/control hygiene removed the apparent signal;
  stop and report the natural-source limitation.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No adaptive source selection or metric-triggered fallback.
