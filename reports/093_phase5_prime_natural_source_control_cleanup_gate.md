# Report 093: Phase 5' Natural Source/Control Cleanup Gate

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-nonsynthetic-native-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Fixed gate for Report 092 cleaned natural-source protocol
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 090 showed high non-synthetic candidate recovery but dirty controls.
Report 091 localized those controls to `<UNK>`, high-frequency tokens,
same-row duplicate atoms, and legacy shuffled-role fixed points. Report 092
preflighted a stricter natural-source query/control protocol and found the
strictest tested protocol feasible:

`non_special_unique_target_freq_le_32`

This report runs exactly the fixed gate precommitted for that cleaned protocol.

## Artifact

Gate precommit:

`notes/notes/2026-05-25-phase5-prime-natural-source-control-cleanup-gate.md`

Gate script:

`scripts/phase5_prime_natural_source_control_cleanup_gate.py`

Raw gate JSON:

`reports/phase5_prime_natural_source_control_cleanup_gate.json`

Gate JSON SHA-256:

`54d95700df01a43bb3c5d28a5f72f3f80cf49f2dd9f848ba860da90bebea98b9`

Cleanup preflight JSON:

`reports/phase5_prime_natural_source_control_cleanup_preflight.json`

Cleanup preflight JSON SHA-256:

`3c3a5c9781c7cb2f2a052923703ad8d4d60a84fcae5a590e82a1c8f29cf40aaf`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_natural_source_control_cleanup_gate.py
```

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_natural_source_control_cleanup_gate.py tests/test_phase5_prime_natural_source_control_cleanup_gate.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase5_prime_natural_source_control_cleanup_gate -v
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_natural_source_control_cleanup_gate.py
jq empty reports/phase5_prime_natural_source_control_cleanup_gate.json
shasum -a 256 reports/phase5_prime_natural_source_control_cleanup_gate.json
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

Source:

`trajectory_native_provenance_context_trace`

Query/control protocol:

`non_special_unique_target_freq_le_32`

Conditions:

- `candidate`
- `random_role`
- `deranged_role`
- `fixedpoint_free_shuffled_role`
- `content_cleanup_positive`

Legacy shuffled-role is intentionally excluded because Report 091 showed the
legacy permutation contains fixed points and is a weaker natural-source
control.

## Results

| condition | top1 | Wilson 95% CI | scene_tix | content_tix | LOO top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate | 0.9512 | [0.9449, 0.9567] | 0.9506 | 0.9512 | 0.9492-0.9555 |
| random_role | 0.0000 | [0.0000, 0.0007] | 0.9506 | 0.0000 | 0.0000-0.0000 |
| deranged_role | 0.0004 | [0.0001, 0.0014] | 0.1086 | 0.0004 | 0.0002-0.0004 |
| fixedpoint_free_shuffled_role | 0.0010 | [0.0004, 0.0023] | 0.1061 | 0.0010 | 0.0007-0.0011 |
| content_cleanup_positive | 1.0000 | [0.9993, 1.0000] | 1.0000 | 1.0000 | 1.0000-1.0000 |

Candidate per-seed top1:

| seed | top1 |
| ---: | ---: |
| 17 | 0.9492 |
| 11 | 0.9531 |
| 23 | 0.9609 |
| 1 | 0.9688 |
| 2 | 0.9355 |
| 3 | 0.9668 |
| 5 | 0.9590 |
| 7 | 0.9668 |
| 13 | 0.9121 |
| 29 | 0.9395 |

Control per-seed top1:

| seed | random_role | deranged_role | fixedpoint_free_shuffled_role |
| ---: | ---: | ---: | ---: |
| 17 | 0.0000 | 0.0000 | 0.0039 |
| 11 | 0.0000 | 0.0000 | 0.0039 |
| 23 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0020 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 | 0.0000 |
| 7 | 0.0000 | 0.0020 | 0.0000 |
| 13 | 0.0000 | 0.0000 | 0.0000 |
| 29 | 0.0000 | 0.0000 | 0.0020 |

## Interpretation

The cleaned natural-source gate passes the precommitted clean-recovery
criterion:

- candidate remains high: `0.9512`;
- random-role control is zero: `0.0000`;
- deranged-role control is below threshold: `0.0004`;
- fixed-point-free shuffled control is below threshold: `0.0010`;
- content cleanup is solved: `1.0000`;
- leave-one-seed-out candidate sensitivity is stable: `0.9492-0.9555`.

This resolves the Report 090 dirty-control caveat for the cleaned natural
source protocol. The Report 091 diagnosis was correct: source/control hygiene
removes the residual while preserving candidate recovery.

## Comparison

| report/source | protocol | candidate | random | deranged | shuffled/control |
| --- | --- | ---: | ---: | ---: | ---: |
| Report 088 controlled native | synthetic controlled provenance | 0.9779 | 0.0000 | 0.0000 | 0.0348 legacy shuffled |
| Report 090 natural native | original natural query plan | 0.9539 | 0.0129 | 0.0250 | 0.0441 legacy shuffled |
| Report 093 natural native | cleaned low-frequency unique targets | 0.9512 | 0.0000 | 0.0004 | 0.0010 fixed-point-free shuffled |

Report 093 is the cleanest non-synthetic native provenance result in this
series. It shows the context-source mechanism survives natural-source hygiene.

## Decision Read

This permits a new integration precommit, not Phase 5 graduation.

The next useful step is to decide how to integrate this cleaned natural-source
protocol into the Phase 5′ design surface:

- as a fixed diagnostic source for bundle-first structural memory;
- as a source-builder hygiene requirement for future natural traces;
- or as a bridge to downstream Phase 5′ integration checks.

Do not jump directly to a full matrix, M2, MQAR/bAbI headline, or Phase 5
graduation claim.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.
