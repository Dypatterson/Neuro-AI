# Report 092: Phase 5' Natural Source/Control Cleanup Preflight

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-nonsynthetic-native-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Preflight-only source/control hygiene discriminator after Report 091
**Status:** Diagnostic preflight only. No candidate/control retrieval, no new
top1 gate, no Phase 5 graduation, no Delta E headline, no full matrix, and no
M2 commitment.

## Question

Report 091 localized Report 090's dirty role-negative controls to exact target
atom equality at the retrieved scene's unbound wrong-role slot. The residual
was dominated by `<UNK>`, high-frequency tokens, same-row repeated atoms, and
legacy shuffled-control fixed points.

This report asks whether a stricter natural-source query/control protocol is
feasible before any new gate is run.

## Artifact

Precommit note:

`notes/notes/2026-05-25-phase5-prime-natural-source-control-cleanup-preflight.md`

Preflight script:

`scripts/phase5_prime_natural_source_control_cleanup_preflight.py`

Raw preflight JSON:

`reports/phase5_prime_natural_source_control_cleanup_preflight.json`

Preflight JSON SHA-256:

`3c3a5c9781c7cb2f2a052923703ad8d4d60a84fcae5a590e82a1c8f29cf40aaf`

Command:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_natural_source_control_cleanup_preflight.py
```

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_natural_source_control_cleanup_preflight.py tests/test_phase5_prime_natural_source_control_cleanup_preflight.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase5_prime_natural_source_control_cleanup_preflight -v
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_natural_source_control_cleanup_preflight.py
jq empty reports/phase5_prime_natural_source_control_cleanup_preflight.json
shasum -a 256 reports/phase5_prime_natural_source_control_cleanup_preflight.json
```

## Protocol

The preflight uses the existing Report 089 source rows and selected
observed-role plans. It does not run retrieval.

Eligible query triples must satisfy:

- target atom is not special (`<UNK>` or `<MASK>`);
- target atom appears only once in its 16-role source row;
- query role is held out from its own observed context;
- target source-row frequency is below the protocol cap.

Frequency caps tested:

`none`, `512`, `256`, `128`, `64`, `32`

For each selected query plan, same-scene exact-match opportunity is measured
for:

- `random_role`: `(query_role + 1) % K_roles`;
- `deranged_role`: fixed derangement;
- `fixedpoint_free_shuffled_role`: fixed derangement generated from the
  shuffled-control seed family.

The legacy Report 090 shuffled permutation is reported only as a comparison
because it can contain fixed points.

## Results

All tested protocols pass the preflight criteria.

| protocol | eligible triples min | selected queries | target freq mean | target freq median | distinct targets mean | legacy shuffle fixed-point rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no cap | 18620 | 512/seed | 35.97 | 8.95 | 328.3 | 0.0656 |
| freq <= 512 | 18620 | 512/seed | 35.37 | 8.85 | 328.5 | 0.0691 |
| freq <= 256 | 18092 | 512/seed | 22.52 | 8.60 | 336.7 | 0.0695 |
| freq <= 128 | 17340 | 512/seed | 17.52 | 7.95 | 341.2 | 0.0682 |
| freq <= 64 | 16112 | 512/seed | 13.15 | 6.90 | 355.2 | 0.0732 |
| freq <= 32 | 13808 | 512/seed | 8.40 | 5.50 | 372.3 | 0.0686 |

The strictest passing protocol is:

`non_special_unique_target_freq_le_32`

For that protocol:

- minimum eligible triples per seed: `13808`;
- selected queries per seed: `512`;
- selected target special fraction: `0.0000`;
- selected same-row duplicate fraction: `0.0000`;
- random-role same-scene exact opportunity: `0.0000`;
- deranged-role same-scene exact opportunity: `0.0000`;
- fixed-point-free shuffled same-scene exact opportunity: `0.0000`;
- mean selected-target source frequency: `8.40`;
- median selected-target source frequency: `5.50`;
- mean distinct target atoms per seed: `372.3`.

Legacy shuffled-role exact opportunity remains nonzero because fixed points
survive in the old permutation control:

`legacy_shuffle_fixed_point_rate_mean = 0.0686`

For unique target rows, that fixed-point rate equals the same-scene exact
opportunity rate.

## Interpretation

The Report 091 residual is not a hard blocker for a cleaner natural-source
test. There is ample support for a stricter query plan that removes the known
same-scene exact-match pathways:

- no `<UNK>` / `<MASK>` targets;
- no repeated target atom within the source row;
- low-frequency target cap;
- no fixed points in the shuffled-role control.

The strictest tested cap (`<=32`) still leaves far more than 512 eligible query
triples per seed, so a future fixed gate can test whether candidate recovery
survives source/control hygiene.

## Decision Read

Report 092 is preflight-only. It does not authorize a gate by itself.

The next step, if continuing this line, is a separate gate precommit for exactly
one protocol:

`non_special_unique_target_freq_le_32`

That future precommit should specify only:

- `candidate`;
- `random_role`;
- `deranged_role`;
- `fixedpoint_free_shuffled_role`;
- `content_cleanup_positive`;
- the selected query plans committed in the Report 092 JSON;
- same source rows and seeds as Report 089;
- no matrix, no M2, no headline, no graduation claim.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No adaptive source selection or metric-triggered fallback.
