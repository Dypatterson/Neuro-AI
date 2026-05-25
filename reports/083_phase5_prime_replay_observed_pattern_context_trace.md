# Report 083: Phase 5' Replay-Observed Pattern Context Trace

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-replay-context-source`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Single fixed-source discriminator after Report 082
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 082 tested:

`scene_token_source=replay_observed_context_trace`

That source reads `pattern_encoder_terms` from the Phase 3/4 provenance-bearing
snapshot, overlays those observed role/atom terms into the synthetic hard cell,
and rebuilds the query-side context with the diagnostic harness's synthetic
role/content codebook. Candidate top1 was `0.6762`, well above controls but far
below Report 081's trace-backed synthetic-context `0.9803`.

This report runs the one fixed-source discriminator allowed by Report 082:

`scene_token_source=replay_observed_pattern_context_trace`

The new source uses the same passive snapshot rows, support filter, and query
plan, but uses each selected snapshot row's actual learned `patterns` vector as
the fixed context token. This asks whether a learned/trajectory-derived context
vector from the Phase 3/4 snapshot repairs the passive trace mismatch.

## Artifact

Raw hard-cell evidence is committed at:

`reports/phase5_prime_replay_observed_pattern_context_hard_cell.json`

SHA-256:

`b247cbba6a439c587c5608453a6dab0094423c547917c5bc75f9fb2ebe59de74`

The run used the same local provenance-bearing snapshot lineage as Report 082:

`reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`

## Implementation

`experiments/44_phase5_prime_bundle_first.py` now supports:

`replay_observed_pattern_context_trace`

The source is static:

- load eligible `pattern_encoder_terms` rows from `--context_trace_snapshot`;
- require enough rows with `context_roles` valid role/atom terms;
- select rows with the existing seeded passive-source path;
- overlay those observed terms into the synthetic hard cell exactly as Report
  082 did;
- use the corresponding snapshot `patterns` rows as the scene/query context
  tokens.

No downstream metric controls routing or condition selection.

## Run Configuration

- `D=4096`
- `K_roles=16`
- `N=512`
- `cue_noise=0.15`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_weight=0.25`
- `scene_token_source=replay_observed_pattern_context_trace`
- `context_roles=4`
- `n_queries=512`
- `seeds={17,11,23,1,2,3,5,7,13,29}`
- conditions: `candidate`, `random_role`, `deranged_role`, `shuffled_role`,
  `content_cleanup_positive`

## Result

The learned snapshot pattern token does not rescue the passive context-source
drop. It weakens the hard cell from Report 082's `0.6762` to `0.4463`.

| condition | top1 | Wilson 95% CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| `candidate` | 0.4463 | [0.4327, 0.4599] | 0.4385 | 0.4463 |
| `content_cleanup_positive` | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| `deranged_role` | 0.0004 | [0.0001, 0.0014] | 0.0016 | 0.0004 |
| `random_role` | 0.0008 | [0.0003, 0.0020] | 0.4385 | 0.0008 |
| `shuffled_role` | 0.0045 | [0.0030, 0.0067] | 0.0262 | 0.0045 |

All aggregate rows reported passive trace support available as 1064 rows,
support used as 512 rows, and zero invalid or too-short rows after the
`C_codebook=2048` selection.

### Per-Seed Candidate Diagnostics

| seed | top1 | correct / 512 | scene_tix | content_tix | scene_margin | content_margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.4551 | 233 | 0.4375 | 0.4551 | 0.8303 | 0.9616 |
| 11 | 0.4238 | 217 | 0.4180 | 0.4238 | 0.8272 | 0.9623 |
| 23 | 0.4727 | 242 | 0.4648 | 0.4727 | 0.8287 | 0.9626 |
| 1 | 0.4160 | 213 | 0.4121 | 0.4160 | 0.8164 | 0.9617 |
| 2 | 0.4473 | 229 | 0.4375 | 0.4473 | 0.8312 | 0.9621 |
| 3 | 0.4180 | 214 | 0.4160 | 0.4180 | 0.8265 | 0.9615 |
| 5 | 0.4453 | 228 | 0.4375 | 0.4453 | 0.8303 | 0.9619 |
| 7 | 0.4883 | 250 | 0.4766 | 0.4883 | 0.8228 | 0.9622 |
| 13 | 0.4395 | 225 | 0.4336 | 0.4395 | 0.8335 | 0.9620 |
| 29 | 0.4570 | 234 | 0.4512 | 0.4570 | 0.8290 | 0.9622 |

Candidate top1 is positive on every seed (`0.4160-0.4883`), but the whole
distribution is weaker than Report 082. Leave-one-seed-out candidate top1
ranges from `0.4416` to `0.4497`, so the aggregate is not a single-seed
artifact.

### Entropy, Margin, and Support

| condition | scene_entropy | content_entropy | scene_margin | content_margin | support available / used / invalid / too_short |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate` | 3.18e-09 | 7.42e-09 | 0.8276 | 0.9620 | 1064 / 512 / 0 / 0 |
| `random_role` | 3.18e-09 | 7.42e-09 | 0.8276 | 0.9621 | 1064 / 512 / 0 / 0 |
| `deranged_role` | 4.89e-09 | 7.42e-09 | 0.7765 | 0.9617 | 1064 / 512 / 0 / 0 |
| `shuffled_role` | 4.62e-09 | 7.42e-09 | 0.7845 | 0.9621 | 1064 / 512 / 0 / 0 |
| `content_cleanup_positive` | 0.00e+00 | 7.42e-09 | 0.0000 | 0.9619 | 1064 / 512 / 0 / 0 |

The entropy/margin/support diagnostics remain clean. The drop is not caused by
support loss, diffuse decode, or content-cleanup failure.

## Interpretation

This discriminator weakens the bundle-first context-source story:

- A learned snapshot pattern token is worse than the Report 082 re-encoded
  passive terms (`0.4463` vs `0.6762`).
- `random_role` preserves high scene identification but near-zero content
  recovery, so the role-unbinding control remains load-bearing.
- `deranged_role` and `shuffled_role` remain near zero.
- `content_cleanup_positive` remains solved.
- Candidate errors track scene/context identification rather than content
  cleanup.

The one allowed fixed-source discriminator therefore points to a passive trace
source-provenance mismatch, not a simple artifact-recovery or support issue.
The learned trajectory vector is not aligned enough with the synthetic hard
cell's scene-MHN geometry to serve as a stronger context token.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Stop the experimental expansion here. The Report 082 decision gate has now been
spent.

Next work should be analysis-only unless a new precommitted gate is written:

1. Analyze the passive trace mismatch: role coverage, atom distribution,
   snapshot-token geometry, and similarity between learned snapshot tokens and
   synthetic scene bundles.
2. Keep the Phase 5' matrix blocked until the context-source mismatch is
   understood.
3. Do not start M2, a full matrix, or a headline pivot from this result.
