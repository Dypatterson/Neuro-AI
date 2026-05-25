# Report 084: Phase 5' Passive Trace Mismatch Analysis

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-replay-context-source`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Analysis-only follow-up to Reports 082-083
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 082 showed that a replay-observed passive context source preserves a
real but degraded signal:

`replay_observed_context_trace`: candidate `0.6762`, CI `[0.6632,0.6889]`

Report 083 spent the single fixed-source discriminator and weakened further:

`replay_observed_pattern_context_trace`: candidate `0.4463`, CI
`[0.4327,0.4599]`

This report performs the bounded next step: analyze the passive trace mismatch
without running a new condition, matrix, or mechanism.

## Artifact

Analysis script:

`scripts/phase5_prime_passive_trace_mismatch.py`

Raw analysis JSON:

`reports/phase5_prime_passive_trace_mismatch_analysis.json`

SHA-256:

`6e39208bd69c5bf404e6f2ce57603304de3e5e9ed70d7b2dfddfca9df6002905`

Inputs:

- `reports/phase5_prime_replay_observed_context_hard_cell.json`
- `reports/phase5_prime_replay_observed_pattern_context_hard_cell.json`
- `reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`

## Snapshot Row Support

The snapshot has full row support for the hard-cell filter, but only over the
Phase 3/4 window roles:

| metric | value |
| --- | ---: |
| raw rows | 1064 |
| eligible rows at `C_codebook=2048,context_roles=4` | 1064 |
| rows invalid | 0 |
| rows too short | 0 |
| row role-set support | `{0,1,2,3}` for all 1064 rows |

Role counts:

| role | count |
| ---: | ---: |
| 0 | 1064 |
| 1 | 1064 |
| 2 | 1064 |
| 3 | 1064 |
| 4-15 | 0 |

The hard cell has `K_roles=16`. Because the passive rows always provide
observed context roles `{0,1,2,3}`, the harness correctly chooses query roles
outside the observed context. Across all reconstructed seeds:

`query_fraction_in_observed_role_support = 0.0000`

So the passive source is not testing a complete 16-role trace. It is testing
whether a 4-role Phase 3/4 window can identify a 16-role synthetic scene whose
queried target role is always outside the passive row's role support.

## Atom Distribution

The atom support is broad but skewed:

| metric | value |
| --- | ---: |
| distinct atoms | 945 |
| normalized atom entropy | 0.6828 |
| fraction of atom entries `>=1024` | 0.0780 |

Most frequent atoms across all rows:

| atom | count |
| ---: | ---: |
| 0 | 1107 |
| 2 | 318 |
| 3 | 154 |
| 5 | 116 |
| 4 | 110 |
| 7 | 89 |
| 6 | 85 |
| 8 | 53 |
| 10 | 44 |
| 1 | 40 |

The source is valid at `C_codebook=2048`, but it is not a balanced hard-cell
source. It is a real Phase 3/4 window distribution with a strong low-token
skew.

## Geometry

The key split is geometric alignment to the synthetic hard-cell scene space.

| comparison | diag mean | offdiag mean | diag top1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| Report 082 partial synthetic context -> synthetic full context | 0.4100 | 0.0312 | 0.9832 | 1.02 |
| learned snapshot pattern -> synthetic full context | -0.0001 | -0.0001 | 0.0014 | 256.13 |
| learned snapshot pattern -> partial synthetic context | -0.0005 | -0.0005 | 0.0010 | 255.53 |
| learned snapshot pattern -> Report 083 stored scene matrix | 0.0546 | 0.0149 | 0.5285 | 6.34 |

Report 082 works as well as it does because the re-encoded passive partial
context has strong same-scene overlap with the synthetic full context:
same-scene top1 is `0.9832`.

Report 083 fails to rescue because the learned snapshot pattern vectors are
geometrically unrelated to the synthetic hard-cell context. Their alignment to
the synthetic full context is chance-level: same-row top1 is `0.0014`, close to
the `1/512` chance scale.

When the learned pattern token is inserted into the Report 083 stored scene
matrix, it becomes a weak scene-specific anchor (`diag_top1=0.5285`) rather
than a semantic context match. That explains why Report 083 remains positive
but weaker than Report 082.

The learned snapshot tokens are also highly clustered:

| learned-token pairwise metric | value |
| --- | ---: |
| offdiag mean | 0.2726 |
| offdiag std | 0.1270 |
| nearest-neighbor mean | 0.6261 |
| nearest-neighbor max | 1.0000 |

So the learned pattern tokens are not clean unique scene IDs either; some are
near duplicates at the scene-anchor level.

## Interpretation

The passive trace mismatch has two separable causes:

1. **Role-support mismatch.** The available passive trace rows are 4-role
   Phase 3/4 windows over roles `0-3`, while the diagnostic hard cell is a
   16-role scene and always queries roles outside that passive support.
2. **Geometry mismatch.** Re-encoding the passive terms through the synthetic
   hard-cell role/content codebook creates a synthetic context that aligns with
   the synthetic scene. Using the learned snapshot pattern vectors does not;
   learned-to-synthetic alignment is at chance.

This means Report 082's `0.6762` is not evidence that the actual learned
snapshot vectors already provide a production-ready context source. It is
evidence that passive row terms can still act as a partial synthetic context
when re-encoded into the same synthetic geometry.

Report 083's `0.4463` is best read as weak scene-anchor behavior from learned
tokens, not as a solved learned-trace context mechanism.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Keep experimental expansion blocked. The next useful work is design-level:

1. Define a precommitted context-source gate whose learned/passive trace uses
   the same role universe as the diagnostic scene, or downscale the diagnostic
   to the 4-role window support before making a new claim.
2. Before running anything, specify whether the source is testing semantic
   context overlap, scene-specific anchoring, or learned trace geometry.
3. Do not run a Phase 5' full matrix, M2, or headline pivot from Reports
   082-084.
