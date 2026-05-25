# Report 088: Phase 5' True Provenance-Source Gate

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-true-provenance-gate`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Fixed candidate/control gate after Report 087 preflight
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 087 preflighted a controlled `native_provenance_context_trace` source:

- native `K_roles=16` source rows are fixed before retrieval evaluation;
- observed roles and query plans are committed in the source artifact;
- query roles are inside source support and held out from their own observed
  context;
- source observed context aligns with matched full context (`diag_top1=1.0000`).

This report runs the fixed candidate/control gate allowed by that preflight.

## Artifact

Gate script:

`scripts/phase5_prime_true_provenance_gate.py`

Raw gate JSON:

`reports/phase5_prime_true_provenance_gate.json`

Gate JSON SHA-256:

`c8ece06a1078aa09a50987924fe73e090687642181e1dfa56f987267ccde8a31`

Source artifact:

`reports/phase5_prime_native_provenance_context_source.json`

Source artifact SHA-256:

`bce6c08de6558cc9d01df51ca5eb53c26b1f8fea9c239687a26fa5196379b7b8`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_true_provenance_gate.py
```

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_true_provenance_gate.py
jq empty reports/phase5_prime_true_provenance_gate.json
shasum -a 256 reports/phase5_prime_true_provenance_gate.json
```

## Fixed Cell

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

`17, 11, 23, 1, 2, 3, 5, 7, 13, 29`

Source:

`native_provenance_context_trace`

Source kind:

`synthetic_controlled_native_provenance`

## Results

| condition | top1 | Wilson 95% CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| candidate | 0.9779 | [0.9735, 0.9816] | 0.9777 | 0.9779 |
| random_role | 0.0000 | [0.0000, 0.0007] | 0.9777 | 0.0000 |
| deranged_role | 0.0000 | [0.0000, 0.0007] | 0.4061 | 0.0000 |
| shuffled_role | 0.0348 | [0.0301, 0.0401] | 0.4467 | 0.0348 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |

Candidate per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.9863 |
| 11 | 0.9707 |
| 23 | 0.9688 |
| 1 | 0.9746 |
| 2 | 0.9805 |
| 3 | 0.9824 |
| 5 | 0.9707 |
| 7 | 0.9727 |
| 13 | 0.9941 |
| 29 | 0.9785 |

Leave-one-seed-out candidate top1 range: `0.9761-0.9789`.

Shuffled-role per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.0254 |
| 11 | 0.0742 |
| 23 | 0.0000 |
| 1 | 0.0254 |
| 2 | 0.0254 |
| 3 | 0.0000 |
| 5 | 0.0312 |
| 7 | 0.0410 |
| 13 | 0.1250 |
| 29 | 0.0000 |

Shuffled-role leave-one-seed-out top1 range: `0.0247-0.0386`.

## Support Diagnostics

The gate uses the committed Report 087 source rows and query plans:

| metric | value |
| --- | ---: |
| source rows available | 512 per seed |
| source rows used | 512 per seed |
| source rows invalid | 0 per seed |
| source rows too short | 0 per seed |

Mean entropy is effectively zero in all retrieval cells, and content cleanup
margins stay high (`~0.9619`). Failures, where present, are scene/context
identification failures rather than content cleanup failures.

## Interpretation

This is the first strong source-recovery result after Reports 082-087:

- native role-universe provenance restores the K=16 context-source gate to
  near the Report 080-081 operating point;
- random-role and deranged-role controls remove the candidate signal;
- content cleanup is solved;
- per-seed and leave-one-seed-out sensitivity are stable.

The shuffled-role residual is nonzero (`0.0348`, one seed at `0.1250`) but
bounded below the candidate by a wide margin. It should be treated as residual
source/provenance structure, not ignored.

The result is still diagnostic. The source is synthetic controlled native
provenance, not learned or natural trace evidence. It shows that the
source-plumbing and role-universe problem can be solved when the provenance
rows natively match the diagnostic scene role universe. It does not prove that
the existing learned snapshot patterns or passive real-corpus traces provide a
production context source.

## Comparison

| report/source | setup | candidate top1 |
| --- | --- | ---: |
| Report 081 `context_trace_observed_prefix_plan` | generated trace plumbing, K=16 | 0.9803 |
| Report 082 `replay_observed_context_trace` | passive rows, K=16 mismatch | 0.6762 |
| Report 086 `replay_observed_context_trace` | matched 4-role passive rows | 0.7668 |
| Report 088 `native_provenance_context_trace` | controlled native K=16 rows | 0.9779 |

The pattern is consistent: the context-completion mechanism works when the
source rows natively match the diagnostic role universe, but degraded or
mismatched provenance weakens the effect.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Stop experiment expansion here.

The result permits a new design/precommit step, not a matrix:

1. Specify a non-synthetic provenance source whose rows natively cover the
   K=16 diagnostic role universe.
2. Or integrate `native_provenance_context_trace` as a controlled positive
   source in the harness, explicitly labeled as synthetic provenance.
3. Preflight support/query/geometry before any new retrieval result.

Do not proceed to a Phase 5' matrix, M2, or headline framing from this result.
