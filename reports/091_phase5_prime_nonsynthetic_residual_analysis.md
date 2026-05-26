# Report 091: Phase 5' Non-Synthetic Native Residual Analysis

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-nonsynthetic-native-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Analysis-only localization of Report 090 dirty role-negative controls
**Status:** Diagnostic only. No new gate, no source change, no Phase 5
graduation, no Delta E headline, no full matrix, and no M2 commitment.

## Question

Report 090 showed high candidate recovery for
`trajectory_native_provenance_context_trace`:

`candidate=0.9539`, CI `[0.9478,0.9593]`

but the role-negative controls missed the precommitted clean threshold:

- `random_role=0.0129`;
- `deranged_role=0.0250`;
- `shuffled_role=0.0441`.

This report asks what produced those dirty-control hits without changing the
source, query plan, controls, or retrieval protocol.

## Artifact

Analysis script:

`scripts/phase5_prime_nonsynthetic_native_residual_analysis.py`

Raw analysis JSON:

`reports/phase5_prime_nonsynthetic_native_residual_analysis.json`

Analysis JSON SHA-256:

`02b8ca83a6a1e58fdf0db58911670fb5ece28845b9b6a5f403fd61f8522021ed`

Input gate JSON:

`reports/phase5_prime_nonsynthetic_native_gate.json`

Input gate JSON SHA-256:

`8ada5e4c01104f79545e9da156c3b870279fdff0da214d54d969f1d85a2ef280`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_nonsynthetic_native_residual_analysis.py
```

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_nonsynthetic_native_residual_analysis.py tests/test_phase5_prime_nonsynthetic_native_residual_analysis.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase5_prime_nonsynthetic_native_residual_analysis -v
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_nonsynthetic_native_residual_analysis.py
jq empty reports/phase5_prime_nonsynthetic_native_residual_analysis.json
shasum -a 256 reports/phase5_prime_nonsynthetic_native_residual_analysis.json
```

## Method

The analysis script deterministically replays the same Report 090 control
cells and records per-query facts:

- target atom and token;
- predicted atom and token;
- target scene and retrieved scene;
- query role and unbound role;
- whether the same-scene wrong-role atom equals the target;
- whether the retrieved-scene unbound-role atom equals the target;
- whether the target appears anywhere in the retrieved scene;
- source-frequency and vocabulary-rank diagnostics.

Every recomputed condition/seed count is cross-checked against Report 090.
All 30 cross-checks pass.

## Results

| control | top1 | hit retrieved-unbind match | hit same-scene unbind match | hit scene-hit rate | hit `<UNK>` fraction | hit source-frequency mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_role | 0.0129 | 1.0000 | 0.9394 | 0.8636 | 0.8182 | 661.0 |
| deranged_role | 0.0250 | 1.0000 | 0.2188 | 0.1406 | 0.7031 | 617.5 |
| shuffled_role | 0.0441 | 1.0000 | 0.5973 | 0.4956 | 0.5177 | 461.2 |

For comparison, all control queries have:

- `<UNK>` target fraction `0.0924`;
- mean target source frequency `116.3`;
- median target source frequency `13.0`.

The dirty hits are therefore strongly enriched for frequent atoms, especially
`<UNK>`.

## Exact-Match Localization

For all three controls, every dirty hit is explained by exact atom equality at
the retrieved scene's unbound role:

```text
hit_retrieved_unbind_match_rate = 1.0000
```

That means the content cleanup head is not inventing the target through a soft
mixture artifact. It is faithfully cleaning up the atom made available by the
scene-MHN plus wrong-role unbind path.

The residual comes from natural source aliasing:

- repeated atoms within the same source row;
- high-frequency atoms appearing in many retrieved scenes;
- `<UNK>` absorbing many rare repo tokens into one shared atom.

## Control-Specific Read

### Random Role

`random_role` uses `(query_role + 1) % K_roles` as the unbound role while the
scene retrieval remains mostly correct.

The residual is mostly same-row duplication:

- `hit_scene_hit_rate=0.8636`;
- `hit_same_scene_unbind_match_rate=0.9394`;
- `<UNK>` accounts for `54/66` hits.

So this control is dirty because adjacent wrong-role slots often contain the
same atom as the held-out target, especially `<UNK>`.

### Deranged Role

`deranged_role` has low scene recovery:

`hit_scene_hit_rate=0.1406`

Only `21.9%` of hits are explained by the same source row. The rest come from
retrieving another scene whose wrong unbound role contains the same target
atom. Again the hits are frequency dominated:

- `<UNK>` accounts for `90/128` hits;
- `the` accounts for `16/128` hits;
- hit target-frequency mean is `617.5` vs `116.3` for all queries.

This is high-frequency natural-corpus aliasing, not role-specific recovery.

### Shuffled Role

`shuffled_role` is the dirtiest control at `0.0441`.

Two effects combine:

1. The shuffle control is a permutation, not a derangement. Some query roles
   remain fixed. Role delta `0` accounts for `94/226` shuffled hits.
2. The remaining hits are still frequency/duplicate dominated:
   `<UNK>` accounts for `117/226` hits and `the` for `22/226`.

This makes shuffled-role a weaker control than deranged-role for this natural
source. The deranged control is the better readout of role-specific structure.

## Interpretation

Report 091 localizes the Report 090 dirty controls to source distribution and
control-design artifacts:

- the repo-sample source has a large shared `<UNK>` atom;
- high-frequency function tokens recur across many source rows;
- repeated atoms within a 16-token row can make wrong-role unbinding look
  correct;
- shuffled-role control allows fixed points, which inflates its residual.

This supports the Report 090 classification: high non-synthetic candidate
recovery, but not a clean solved source mechanism.

## Decision Read

Stop here for this source/gate cycle.

The next useful work is a new precommit if we want to change anything:

- an OOV/special-token handling preflight;
- an atom-frequency-stratified source/query preflight;
- a deranged-only or fixed-point-free shuffled control specification;
- or a natural-source source builder that reduces shared-token aliasing.

Any such change is a changed source/control protocol and must be precommitted
and preflighted before another candidate/control top1 run.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.
