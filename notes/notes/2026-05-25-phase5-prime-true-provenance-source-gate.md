---
date: 2026-05-25
project: neuro-ai
tags:
  - notes
  - phase-5-prime
  - planning
  - precommit
---

# Phase 5 Prime True Provenance-Source Gate

## Status

Reports 082-086 close the first replay-observed context-source sequence:

- Report 082 showed a real but degraded replay-observed passive signal at the
  K=16 hard cell.
- Report 083 showed learned snapshot pattern tokens are weaker than re-encoded
  passive context.
- Report 084 localized the first mismatch to role-universe support and learned
  token geometry.
- Report 085 passed the matched 4-role support/query preflight but showed
  learned tokens still align to matched synthetic context at chance.
- Report 086 ran the matched 4-role gate and found only partial recovery:
  `replay_observed_context_trace=0.7668` and
  `replay_observed_pattern_context_trace=0.5219`.

These are diagnostic results only. Phase 5 has not graduated. Do not run a
Phase 5' matrix, M2, MQAR/bAbI headline pivot, or Delta E headline run from
this sequence.

## Decision

Stop ad-hoc context-source expansion.

The next context-source experiment, if any, must be a true provenance-source
gate: the context source must be generated or recovered in the same role
universe as the diagnostic scene before candidate/control retrieval is run.

Do not reuse the Report 082-086 workaround as headline evidence:

- do not treat passive 4-role rows as evidence for a 16-role source;
- do not re-encode a mismatched passive row distribution into a synthetic scene
  geometry and call that a learned trace source;
- do not treat learned snapshot `patterns` as semantic context tokens unless
  their geometry is specified and preflighted.

## Question

Can a provenance source whose trace role universe natively matches the
diagnostic role universe recover the bundle-first context-completion effect
without relying on synthetic re-encoding after the fact?

This is narrower than Phase 5 graduation. It tests whether the context-source
blocker is source provenance or a deeper learned-vector/context-geometry
failure.

## Required Source Contract

A proposed source must ship a manifest before any retrieval run:

```text
source_name
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
pattern_token_shape, if used
```

Minimum source requirements:

- trace rows must be produced or selected under the same `K_roles` as the gate;
- every planned query role must be in source role support;
- the query target role must be held out from that query's observed context;
- row selection must be fixed before conditions are run;
- candidate and controls must share the same rows, query schedule, seeds, and
  support set;
- source construction must not use candidate/control outcomes.

## First Allowed Gate

Use the original hard context-source operating point unless a later precommit
replaces it:

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

Allowed first source family:

1. `native_provenance_context_trace`
   - A passive or trajectory-derived source whose encoder terms already live
     in the gate role universe.
   - Query context must be built from the source's observed terms for that
     scene or trace row.
   - It may be synthetic as a controlled diagnostic, but its provenance rows
     must be generated in the same role universe before retrieval evaluation.

Do not include generated-scene context bundles, random anchors, fixed-prefix
synthetic context, or the full fixed observed-prefix grid in this gate. Reports
079-081 already cover those cleaner sources.

Learned pattern vectors are not an allowed candidate source in the first true
provenance gate unless the preflight defines their role:

- **Semantic context-token interpretation:** must show above-chance alignment
  to same-row full/partial context before the retrieval run.
- **Scene-anchor interpretation:** must be labeled as an anchor diagnostic, not
  as learned semantic context.

## Required Preflight

Run and commit a preflight JSON before any candidate/control result:

- raw row count, eligible row count, used row count, invalid row count, and
  too-short row count;
- source role counts and role-set counts;
- query-role counts;
- `query_fraction_in_source_role_support`;
- `query_fraction_in_own_observed_context`;
- atom distinct count, atom entropy, and top atoms;
- source/context geometry:
  - source observed context -> matched full scene context;
  - learned pattern token -> matched full scene context, if learned tokens are
    used;
  - learned pattern token pairwise nearest-neighbor stats, if learned tokens
    are used;
- raw artifact SHA-256.

Preflight pass criteria:

- `eligible_rows >= N`;
- `used_rows == N`;
- `rows_invalid == 0`;
- `rows_too_short == 0`;
- `query_fraction_in_source_role_support == 1.0000`;
- `query_fraction_in_own_observed_context == 0.0000`;
- source geometry is reported before any top1 claim.

If any support/query criterion fails, stop and report the preflight. Do not
change the gate into a matrix.

## Conditions

If and only if the preflight passes, run exactly these conditions:

- `candidate`
- `random_role`
- `deranged_role`
- `shuffled_role`
- `content_cleanup_positive`

All conditions must share the same source rows, query schedule, seeds, and
support set.

## Required Readout

Report:

- top1 with Wilson 95% CI;
- per-seed top1;
- leave-one-seed-out candidate sensitivity;
- `scene_tix` and `content_tix`;
- scene/content entropy;
- scene/content margin;
- all preflight support diagnostics;
- raw JSON path and SHA-256.

## Decision Gate

Readout bands are diagnostic, not graduation criteria:

- **Strong source recovery:** candidate `>=0.90`, random/deranged controls
  `<=0.005`, shuffled-role residual bounded and interpreted, positive cleanup
  solved, and no unstable seed. This permits writing a new architecture or
  integration precommit, not a full matrix.
- **Partial source recovery:** candidate above controls but `<0.90`, or a
  nontrivial shuffled-role residual. Stop and analyze source geometry/support.
- **Failure:** candidate near controls, unstable seeds, dirty random/deranged
  controls, or failed positive cleanup. Stop and localize the source failure.

Do not jump from any outcome here to Phase 5 graduation.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No continuous-learning-across-domains claim.
- No emergent-self claim.
- No adaptive source selection or metric-triggered fallback.
