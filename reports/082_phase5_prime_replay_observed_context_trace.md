# Report 082: Phase 5' Replay-Observed Context Trace

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-replay-context-source`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Single Phase 5' less-synthetic context-source gate after Report 081
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 081's hard cell used:

`scene_token_source=context_trace_observed_prefix_plan`

That result was strong, but the trace contents were still assembled from the
generated scene row. This follow-up asks whether the same hard-cell anchor
survives a less-synthetic passive trace source:

`scene_token_source=replay_observed_context_trace`

The source reads `pattern_encoder_terms` from the Phase 3/4 provenance-bearing
snapshot and builds the context token from stored role/atom terms rather than
from the MQAR harness's generated scene contents.

## Artifact

Compact aggregate evidence is committed at:

`reports/phase5_prime_replay_observed_context_hard_cell_summary.json`

Raw hard-cell evidence is committed at:

`reports/phase5_prime_replay_observed_context_hard_cell.json`

The raw JSON was recovered from the existing Colab runtime through a scratch-cell
Drive copy. No experiment cell was rerun and the notebook was not saved back to
GitHub. SHA-256:

`d587d2bc33e60b9665764be6b1094af0cb6fee036ee106497fbcc19a227c1120`

The completed Colab runtime path was:

`/content/phase5_prime_replay_observed_context_hard_cell.json`

Notebook URL:

`https://colab.research.google.com/github/Dypatterson/Neuro-AI/blob/codex/phase5-prime-replay-context-source/scripts/colab_phase5_prime_replay_observed_context.ipynb`

## Snapshot Validation

The Drive snapshots found by Cell 2 were stale and did not contain
`pattern_encoder_terms`. The local provenance snapshot was uploaded to the
runtime and validated before running the experiment:

- runtime path: `/content/phase3_phase4_w4_step1800.pt`
- byte size: `35111853`
- `pattern_encoder_terms` rows: `1064`

Support probe:

| C_codebook | eligible_rows | rows_invalid | rows_too_short | max_atom_seen |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 766 | 298 | 298 | 2047 |
| 2048 | 1064 | 0 | 0 | 2047 |

The hard cell used `C_codebook=2048`. Using `C_codebook=1024` would have made
298 rows invalid because the passive trace atoms extend to `2047`.

## Run Configuration

- `D=4096`
- `K_roles=16`
- `N=512`
- `cue_noise=0.15`
- `cooccurrence=skewed`
- `scene_token=1`
- `scene_token_weight=0.25`
- `scene_token_source=replay_observed_context_trace`
- `context_roles=4`
- `n_queries=512`
- `seeds={17,11,23,1,2,3,5,7,13,29}`
- conditions: `candidate`, `random_role`, `deranged_role`, `shuffled_role`,
  `content_cleanup_positive`

## Result

The candidate signal survives the less-synthetic passive trace source, but it
does not reproduce Report 081's near-ceiling hard-cell result. Report 081 had
candidate top1 `0.9803` at the same `context_roles=4,w=0.25` anchor. The
replay-observed context trace lands at `0.6762`.

| condition | top1 | Wilson 95% CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| `candidate` | 0.6762 | [0.6632, 0.6889] | 0.6715 | 0.6762 |
| `content_cleanup_positive` | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |
| `deranged_role` | 0.0006 | [0.0002, 0.0017] | 0.0039 | 0.0006 |
| `random_role` | 0.0014 | [0.0007, 0.0028] | 0.6715 | 0.0014 |
| `shuffled_role` | 0.0061 | [0.0043, 0.0086] | 0.0412 | 0.0061 |

All hard-cell aggregate rows reported passive trace support available as 1064
rows, support used as 512 rows, and zero invalid or too-short rows after the
`C_codebook=2048` selection.

### Per-Seed Candidate Diagnostics

| seed | top1 | correct / 512 | scene_tix | content_tix | scene_margin | content_margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.6855 | 351 | 0.6836 | 0.6855 | 0.8362 | 0.9617 |
| 11 | 0.6543 | 335 | 0.6504 | 0.6543 | 0.8304 | 0.9621 |
| 23 | 0.7363 | 377 | 0.7324 | 0.7363 | 0.8335 | 0.9623 |
| 1 | 0.6387 | 327 | 0.6348 | 0.6387 | 0.8215 | 0.9618 |
| 2 | 0.6895 | 353 | 0.6836 | 0.6895 | 0.8353 | 0.9621 |
| 3 | 0.6504 | 333 | 0.6465 | 0.6504 | 0.8322 | 0.9615 |
| 5 | 0.6875 | 352 | 0.6797 | 0.6875 | 0.8319 | 0.9620 |
| 7 | 0.6680 | 342 | 0.6641 | 0.6680 | 0.8281 | 0.9625 |
| 13 | 0.6738 | 345 | 0.6680 | 0.6738 | 0.8351 | 0.9620 |
| 29 | 0.6777 | 347 | 0.6719 | 0.6777 | 0.8319 | 0.9622 |

Candidate top1 is positive on every seed (`0.6387-0.7363`). Leave-one-seed-out
candidate top1 ranges from `0.6695` to `0.6803`, so the aggregate is not carried
by a single seed.

### Entropy, Margin, and Support

| condition | scene_entropy | content_entropy | scene_margin | content_margin | support available / used / invalid / too_short |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate` | 3.08e-09 | 7.42e-09 | 0.8316 | 0.9620 | 1064 / 512 / 0 / 0 |
| `random_role` | 3.08e-09 | 7.42e-09 | 0.8316 | 0.9621 | 1064 / 512 / 0 / 0 |
| `deranged_role` | 4.76e-09 | 7.42e-09 | 0.7790 | 0.9618 | 1064 / 512 / 0 / 0 |
| `shuffled_role` | 4.50e-09 | 7.42e-09 | 0.7867 | 0.9621 | 1064 / 512 / 0 / 0 |
| `content_cleanup_positive` | 0.00e+00 | 7.42e-09 | 0.0000 | 0.9619 | 1064 / 512 / 0 / 0 |

The entropy and margin diagnostics do not show a diffuse decode collapse.
Support is matched across conditions and seeds.

## Interpretation

This is a useful but narrower result than Report 081:

- The replay-observed context source is substantially above all role-negative
  controls.
- `random_role` has high scene identification but near-zero content recovery,
  preserving the role-unbinding control.
- `content_cleanup_positive` stays solved, so the candidate drop is not a
  content-cleanup failure.
- The lower candidate top1 localizes a new source-provenance cost: direct
  generated-scene context and trace-backed observed-prefix context were much
  cleaner than this passive replay-observed trace.
- Raw inspection removes the immediate artifact caveat. Per-seed and
  leave-one-seed-out diagnostics are stable enough for a single next
  discriminator, and entropy/margin/support diagnostics are clean enough to
  rule out a support or decode-collapse explanation for the drop.

The result supports continuing the bundle-first path, but it is not strong
enough to justify a headline pivot or full matrix. The next defensible work is
exactly one more fixed-source discriminator, such as a trajectory-derived or
learned trace source, on the same hard cell with matched controls. If that
weakens or repeats the provenance gap, stop and analyze the passive trace
mismatch instead of expanding to a matrix.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Do not run the full matrix yet. The next step should stay narrow:

1. Keep the recovered raw Colab hard-cell JSON at
   `reports/phase5_prime_replay_observed_context_hard_cell.json`.
2. If continuing, run exactly one fixed-source discriminator
   against a trajectory-derived or learned trace source. Keep the same hard
   cell and controls.
3. If that discriminator weakens or shows the same provenance mismatch, stop
   and analyze the passive trace mismatch.
4. Only after the context-source question is fixed should a larger Phase 5'
   matrix be considered.
