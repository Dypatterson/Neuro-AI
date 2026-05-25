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

The completed Colab runtime also wrote:

`/content/phase5_prime_replay_observed_context_hard_cell.json`

That raw runtime JSON was not downloaded into the repo because Safari requested
a site download permission prompt. I did not accept that browser permission
without an explicit action-time approval. The committed JSON is therefore a
compact aggregate capture from the completed Colab output, not the raw Colab
artifact.

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

The result supports continuing the bundle-first path, but it is not strong
enough to justify a headline pivot or full matrix. The next defensible work is
to capture the raw runtime JSON or rerun with an approved artifact path, then
inspect per-seed top1 plus entropy/margin diagnostics before deciding whether
the passive trace source is ready for a broader matrix.

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

1. Preserve the raw Colab hard-cell JSON through an approved artifact path
   rather than a browser permission prompt.
2. Verify per-seed top1, scene/content entropy, and scene/content margin from
   the raw JSON.
3. If those diagnostics are clean, consider one more fixed-source discriminator
   against a trajectory-derived or learned trace source. Keep the same hard
   cell and controls.
4. Only after the context-source question is fixed should a larger Phase 5'
   matrix be considered.
