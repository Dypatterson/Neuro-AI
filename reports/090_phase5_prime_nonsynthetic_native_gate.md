# Report 090: Phase 5' Non-Synthetic Native Provenance Gate

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-nonsynthetic-native-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Fixed candidate/control gate after Report 089 preflight
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 089 preflighted `trajectory_native_provenance_context_trace`, a
repo-sample source built from Phase 2 windows encoded through
`encode_window_with_provenance()` and stored as `TrajectoryTrace.encoder_terms`
in a `ReplayStore`.

The preflight passed support/query/geometry checks:

- `5352` eligible repo-sample windows;
- `512` used source rows per seed;
- invalid/too-short rows `0`;
- query role support `1.0000`;
- query role leakage into its own observed context `0.0000`;
- observed-context to matched full-context `diag_top1=0.9998`.

This report runs only the fixed candidate/control gate allowed by that
preflight.

## Artifact

Gate script:

`scripts/phase5_prime_nonsynthetic_native_gate.py`

Raw gate JSON:

`reports/phase5_prime_nonsynthetic_native_gate.json`

Gate JSON SHA-256:

`8ada5e4c01104f79545e9da156c3b870279fdff0da214d54d969f1d85a2ef280`

Source artifact:

`reports/phase5_prime_nonsynthetic_native_context_source.json`

Source artifact SHA-256:

`2200f3c1b1599c478ce6ecc6a3cb1e783c3bfd51ceda3f335642bb9158989398`

Preflight artifact:

`reports/phase5_prime_nonsynthetic_native_preflight.json`

Preflight artifact SHA-256:

`868ac20feb75169eaeb6b4440bc32aeb4d336a74af6c40dbd793aadebeca9bab`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_nonsynthetic_native_gate.py
```

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_nonsynthetic_native_gate.py tests/test_phase5_prime_nonsynthetic_native_gate.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase5_prime_nonsynthetic_native_gate -v
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_nonsynthetic_native_gate.py
jq empty reports/phase5_prime_nonsynthetic_native_gate.json
shasum -a 256 reports/phase5_prime_nonsynthetic_native_gate.json
```

## Fixed Cell

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
beta=30
cooccurrence=repo_sample_natural
```

Seeds:

`17, 11, 23, 1, 2, 3, 5, 7, 13, 29`

Source:

`trajectory_native_provenance_context_trace`

Source kind:

`repo_sample_phase2_window_trace`

## Results

| condition | top1 | Wilson 95% CI | scene_tix | content_tix | LOO top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate | 0.9539 | [0.9478, 0.9593] | 0.9525 | 0.9539 | 0.9520-0.9568 |
| random_role | 0.0129 | [0.0101, 0.0164] | 0.9525 | 0.0129 | 0.0122-0.0139 |
| deranged_role | 0.0250 | [0.0211, 0.0296] | 0.1076 | 0.0250 | 0.0239-0.0258 |
| shuffled_role | 0.0441 | [0.0388, 0.0501] | 0.1732 | 0.0441 | 0.0380-0.0462 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 | 1.0000-1.0000 |

Candidate per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.9453 |
| 11 | 0.9434 |
| 23 | 0.9629 |
| 1 | 0.9648 |
| 2 | 0.9277 |
| 3 | 0.9707 |
| 5 | 0.9590 |
| 7 | 0.9629 |
| 13 | 0.9434 |
| 29 | 0.9590 |

Control per-seed top1:

| seed | random_role | deranged_role | shuffled_role |
| ---: | ---: | ---: | ---: |
| 17 | 0.0137 | 0.0215 | 0.0273 |
| 11 | 0.0098 | 0.0176 | 0.0488 |
| 23 | 0.0039 | 0.0273 | 0.0430 |
| 1 | 0.0137 | 0.0215 | 0.0371 |
| 2 | 0.0195 | 0.0352 | 0.0469 |
| 3 | 0.0098 | 0.0273 | 0.0293 |
| 5 | 0.0156 | 0.0176 | 0.0254 |
| 7 | 0.0117 | 0.0312 | 0.0508 |
| 13 | 0.0117 | 0.0332 | 0.0996 |
| 29 | 0.0195 | 0.0176 | 0.0332 |

## Support Diagnostics

The gate used the committed Report 089 source rows and query plans:

| metric | value |
| --- | ---: |
| source rows available | 512 per seed |
| source rows used | 512 per seed |
| source rows invalid | 0 per seed |
| source rows too short | 0 per seed |
| source artifact SHA | `2200f3c1...9989398` |

Content cleanup is solved independently of scene retrieval (`1.0000`). In the
candidate condition, scene and content rates track each other closely
(`0.9525` vs `0.9539`), so candidate misses are primarily scene/context
identification misses.

## Interpretation

This is the first high candidate recovery from a non-synthetic native
provenance source at the K=16 hard cell: `candidate=0.9539`, with stable seeds
and stable leave-one-seed-out sensitivity.

However, the precommitted strong-recovery criterion required random/deranged
controls `<=0.005`. Report 090 does not meet that bar:

- `random_role=0.0129`;
- `deranged_role=0.0250`;
- `shuffled_role=0.0441`, with seed 13 at `0.0996`.

This is therefore a positive but dirty-control recovery, not a clean solved
source mechanism. The natural repo co-occurrence source recovers most of the
controlled-source effect, but it also carries enough residual role/content
structure that wrong-role controls can score above the precommitted clean
threshold.

## Comparison

| report/source | source kind | candidate | random | deranged | shuffled |
| --- | --- | ---: | ---: | ---: | ---: |
| Report 088 `native_provenance_context_trace` | synthetic controlled native provenance | 0.9779 | 0.0000 | 0.0000 | 0.0348 |
| Report 090 `trajectory_native_provenance_context_trace` | repo-sample native trajectory source | 0.9539 | 0.0129 | 0.0250 | 0.0441 |

The non-synthetic source preserves most of the candidate signal, but the
controlled source remains cleaner under role-negative controls.

## Decision Read

Classify this as **partial / dirty-control recovery**.

Do not proceed to a Phase 5' full matrix, M2 run, MQAR/bAbI headline pivot, or
graduation-style claim from this result. The next useful step is an
analysis-only localization of the control residual: quantify whether
wrong-role hits are explained by repeated atoms, natural adjacent-token
co-occurrence, high-frequency function tokens, or role-shift structure in the
repo-sample windows.

Any rerun that changes source construction, filtering, balancing, or controls
needs a new precommit and preflight before candidate/control top1 is run.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.
