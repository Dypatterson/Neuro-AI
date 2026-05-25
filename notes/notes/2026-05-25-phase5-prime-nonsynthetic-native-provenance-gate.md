---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime Non-Synthetic Native Provenance Gate

## Status

Report 088 establishes strong recovery for controlled native provenance:

`native_provenance_context_trace`: candidate `0.9779`, CI
`[0.9735,0.9816]`, random/deranged controls `0.0000`, shuffled-role `0.0348`,
and positive cleanup `1.0000`.

This proves the K=16 context-source gate is recoverable when provenance rows
natively match the diagnostic role universe. It does not prove that learned or
natural traces already provide that source. The Report 087-088 source is
synthetic controlled provenance and must remain labeled as such.

No Phase 5 graduation, Delta E headline run, full matrix, M2 commitment, or
MQAR/bAbI pivot follows from Report 088.

## Decision

Do not run another context-source experiment until a non-synthetic native
provenance source is specified and preflighted.

The controlled `native_provenance_context_trace` artifact may be integrated
only as a positive-control source. It must not be used as a production or
learned-trace claim.

## Question

Can a non-synthetic trace source produce native K=16 encoder-term rows with
enough support and clean enough geometry to recover the Report 088 effect?

This is the next discriminator between:

- source-plumbing solved but real provenance missing; and
- a feasible path to learned/natural trace context once the source builder is
  fixed.

## Source Candidates

Exactly one source family should be selected before implementation:

1. **Trajectory-derived native source**
   - Generate or collect `TrajectoryTrace.encoder_terms` rows from the actual
     Phase 2/4 trace path using `K_roles=16`.
   - Preferred if the current pipeline can emit native 16-role windows without
     inventing new learned vectors.
2. **Replay-buffer native source**
   - Populate a `ReplayStore` with traces whose `encoder_terms` already cover
     the K=16 diagnostic role universe.
   - Use only stored trace provenance; no post-hoc row reconstruction from
     retrieval results.
3. **Learned-pattern source**
   - Deferred unless a preflight shows learned pattern vectors align
     above-chance to same-row full or partial context.
   - If used as anchors, label them as scene-anchor diagnostics rather than
     semantic context tokens.

Do not mix these source families in one first run. Pick one, preflight it, and
stop if support or geometry fails.

## Fixed Diagnostic Cell

Use the same hard context-source operating point:

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
C_codebook=2048
```

Seeds:

```text
17 11 23 1 2 3 5 7 13 29
```

## Required Source Manifest

Commit a source manifest before retrieval:

```text
source_name
source_family
source_build_command
source_artifact_path
source_artifact_sha256
D
K_roles
C_codebook
N_raw_rows
role_universe
role_counts
role_set_counts
atom_distinct
atom_entropy
row_selection_seed_or_rule
query_plan_seed_or_rule
pattern_token_shape, if used
```

The manifest must say whether the source is trajectory-derived, replay-buffer
native, learned-pattern, or controlled-positive. Ambiguous "less synthetic"
language is not enough.

## Required Preflight

Before any candidate/control run, commit a preflight JSON with:

- raw row count, eligible row count, used row count, invalid row count, and
  too-short row count;
- source role counts and role-set counts;
- query-role counts;
- `query_fraction_in_source_role_support`;
- `query_fraction_in_own_observed_context`;
- atom distinct count, atom entropy, and top atoms;
- source observed context -> matched full scene context geometry;
- learned-token alignment and pairwise nearest-neighbor stats, if learned
  tokens are used;
- raw artifact SHA-256.

Preflight pass criteria:

- `eligible_rows >= N`;
- `used_rows == N`;
- `rows_invalid == 0`;
- `rows_too_short == 0`;
- `query_fraction_in_source_role_support == 1.0000`;
- `query_fraction_in_own_observed_context == 0.0000`;
- source geometry is reported before any top1 claim.

If the preflight fails, stop and report it. Do not patch the gate into a matrix
or change source families mid-run.

## Candidate/Control Conditions

If and only if the preflight passes, run exactly:

- `candidate`
- `random_role`
- `deranged_role`
- `shuffled_role`
- `content_cleanup_positive`

All conditions must share source rows, selected observed roles, query schedule,
seeds, and support set.

## Readout

Report:

- top1 with Wilson 95% CI;
- per-seed top1;
- leave-one-seed-out candidate sensitivity;
- `scene_tix` and `content_tix`;
- scene/content entropy;
- scene/content margin;
- support and source-geometry diagnostics;
- raw artifact paths and SHA-256.

## Decision Gate

- **Strong non-synthetic recovery:** candidate `>=0.90`, random/deranged
  controls `<=0.005`, bounded shuffled-role residual, positive cleanup solved,
  and stable seeds. This permits a new integration precommit, not a Phase 5
  graduation claim.
- **Partial recovery:** candidate above controls but `<0.90`, unstable seeds,
  or nontrivial shuffled-role residual. Stop and analyze source geometry.
- **Failure:** candidate near controls, dirty random/deranged controls, failed
  positive cleanup, or failed support. Stop and localize the source builder.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No continuous-learning-across-domains claim.
- No emergent-self claim.
- No adaptive source selection or metric-triggered fallback.
