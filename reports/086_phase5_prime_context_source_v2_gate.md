# Report 086: Phase 5' Context-Source V2 Gate

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-context-source-v2-gate-run`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Fixed matched 4-role candidate/control discriminator from the
precommitted v2 gate
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 085 passed the required preflight for the matched
`K_roles=4,context_roles=2` context-source v2 gate:

- passive support is sufficient and matched to roles `{0,1,2,3}`;
- query roles are inside source support;
- each query role is held out from that query's observed context;
- re-encoded passive partial context aligns with matched synthetic full context
  (`diag_top1=0.8668`);
- learned snapshot patterns remain at chance against matched synthetic context
  (`diag_top1=0.0031`).

This report runs the next and only allowed fixed discriminator:

1. `replay_observed_context_trace`
2. `replay_observed_pattern_context_trace`

Both sources use the same seeds, rows, query schedule, and matched controls.

## Artifact

Raw gate JSON:

`reports/phase5_prime_context_source_v2_gate.json`

SHA-256:

`4f1b44f678ac80355decb482456986b2b7a47797d82d17084e5337f36e59cdd9`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/44_phase5_prime_bundle_first.py \
  --Ds 4096 \
  --Ns 512 \
  --K_roles 4 \
  --cue_noise 0.15 \
  --seeds 17 11 23 1 2 3 5 7 13 29 \
  --n_queries 512 \
  --C_codebook 2048 \
  --conditions candidate random_role deranged_role shuffled_role content_cleanup_positive \
  --scene_token 1 \
  --scene_token_weight 0.25 \
  --scene_token_source replay_observed_context_trace replay_observed_pattern_context_trace \
  --context_roles 2 \
  --cooccurrence skewed \
  --context_trace_snapshot reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt \
  --out reports/phase5_prime_context_source_v2_gate.json
```

The run used CPU because neither MPS nor CUDA was available in the local
environment.

## Fixed Cell

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

Seeds:

`17, 11, 23, 1, 2, 3, 5, 7, 13, 29`

## Results

### `replay_observed_context_trace`

| condition | top1 | Wilson 95% CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| candidate | 0.7668 | [0.7550, 0.7782] | 0.7596 | 0.7668 |
| random_role | 0.0012 | [0.0005, 0.0026] | 0.7596 | 0.0012 |
| deranged_role | 0.0014 | [0.0007, 0.0028] | 0.0064 | 0.0014 |
| shuffled_role | 0.0439 | [0.0387, 0.0499] | 0.2088 | 0.0439 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |

Candidate per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.7344 |
| 11 | 0.7773 |
| 23 | 0.7461 |
| 1 | 0.7871 |
| 2 | 0.7227 |
| 3 | 0.7715 |
| 5 | 0.7637 |
| 7 | 0.8066 |
| 13 | 0.7598 |
| 29 | 0.7988 |

Leave-one-seed-out candidate top1 range: `0.7624-0.7717`.

### `replay_observed_pattern_context_trace`

| condition | top1 | Wilson 95% CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| candidate | 0.5219 | [0.5082, 0.5355] | 0.5148 | 0.5219 |
| random_role | 0.0016 | [0.0008, 0.0031] | 0.5148 | 0.0016 |
| deranged_role | 0.0016 | [0.0008, 0.0031] | 0.0027 | 0.0016 |
| shuffled_role | 0.0270 | [0.0229, 0.0318] | 0.1354 | 0.0270 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 |

Candidate per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.5293 |
| 11 | 0.5508 |
| 23 | 0.4863 |
| 1 | 0.5234 |
| 2 | 0.5156 |
| 3 | 0.5215 |
| 5 | 0.5371 |
| 7 | 0.5039 |
| 13 | 0.5137 |
| 29 | 0.5371 |

Leave-one-seed-out candidate top1 range: `0.5187-0.5258`.

## Support Diagnostics

Support is matched across all conditions and both sources:

| metric | value |
| --- | ---: |
| passive trace rows available | 1064 per seed |
| passive trace rows used | 512 per seed |
| passive trace rows too short | 0 per seed |
| passive trace rows invalid | 0 per seed |

Source snapshot:

`reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`

Mean entropy is effectively zero in all cells, and content cleanup margins stay
high (`~0.962`). The failure mode is scene/context identification, not content
cleanup capacity.

## Comparison To Reports 082-085

The matched 4-role gate improves the re-encoded passive context source relative
to the Report 082 hard cell:

| source | setup | candidate top1 |
| --- | --- | ---: |
| Report 082 `replay_observed_context_trace` | `K_roles=16,context_roles=4` | 0.6762 |
| Report 086 `replay_observed_context_trace` | `K_roles=4,context_roles=2` | 0.7668 |

So Report 084's role-universe mismatch was a real part of the degradation.

But the effect does not return to the cleaner synthetic/trace-backed operating
point from Reports 080-081 (`0.9803`). Matching the role universe only partially
recovers the passive source.

The learned-pattern source improves only modestly relative to Report 083:

| source | setup | candidate top1 |
| --- | --- | ---: |
| Report 083 `replay_observed_pattern_context_trace` | `K_roles=16,context_roles=4` | 0.4463 |
| Report 086 `replay_observed_pattern_context_trace` | `K_roles=4,context_roles=2` | 0.5219 |

That matches Report 085's preflight: learned snapshot pattern vectors remain
geometrically misaligned with matched synthetic context and behave as weak
scene anchors rather than solved semantic context tokens.

## Interpretation

The v2 gate resolves the narrow role-support concern but does not solve the
context-source problem.

What is now supported:

- passive rows are a real and usable less-synthetic context source when the
  diagnostic role universe matches the trace role universe;
- role-negative controls remove the candidate signal;
- content cleanup is not the bottleneck;
- per-seed and leave-one-seed-out sensitivity are stable.

What remains unsolved:

- the re-encoded passive source is still far below the synthetic/trace-backed
  ceiling;
- learned snapshot pattern tokens remain weak despite the matched role
  universe;
- the nonzero shuffled-role residual shows source/provenance structure is not
  perfectly clean.

This is therefore a partial recovery, not a Phase 5 graduation route.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Stop experimental expansion here.

The next useful step is design/precommit work for a true provenance-source
gate, not another ad-hoc run:

1. Specify how to obtain a passive/trajectory source whose trace role universe
   matches the diagnostic role universe without re-encoding synthetic
   identities after the fact.
2. Decide whether learned pattern vectors are supposed to act as semantic
   context terms or only as scene anchors.
3. If a true 16-role provenance source is proposed, write a new precommit note
   with its support diagnostics and controls before running it.

Do not proceed to a Phase 5' matrix, M2, or headline framing from this result.
