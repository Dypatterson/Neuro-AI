# Report 085: Phase 5' Context-Source V2 Preflight

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-context-source-v2-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Preflight only for the precommitted matched 4-role context-source
v2 gate
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 084 localized the first replay-observed context-source mismatch to two
issues:

- the passive trace rows cover only roles `{0,1,2,3}`;
- learned snapshot pattern vectors do not align with the synthetic hard-cell
  context geometry.

The v2 gate precommit therefore required a matched 4-role preflight before any
candidate/control retrieval run.

This report answers only the preflight question: is the matched
`K_roles=4,context_roles=2` gate well-posed enough to run the precommitted
candidate/control discriminator?

## Artifact

Preflight script:

`scripts/phase5_prime_context_source_v2_preflight.py`

Raw preflight JSON:

`reports/phase5_prime_context_source_v2_preflight.json`

SHA-256:

`9c33210347cdf8878b5ecad57106e88a2294fb5be29acdcb65db83cd2c095702`

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_context_source_v2_preflight.py
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_context_source_v2_preflight.py
shasum -a 256 reports/phase5_prime_context_source_v2_preflight.json
```

Input snapshot:

`reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`

Fixed preflight cell:

```text
D=4096
K_roles=4
N=512
cue_noise=0.15
cooccurrence=skewed
scene_token_weight=0.25
context_roles=2
n_queries=512
n_seeds=10
C_codebook=2048
```

Seeds:

`17, 11, 23, 1, 2, 3, 5, 7, 13, 29`

## Support And Query Schedule

All support criteria pass across the 10 seeds:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| raw rows | 1064.0 | 1064.0 | 1064.0 |
| eligible rows | 1064.0 | 1064.0 | 1064.0 |
| required rows | 512.0 | 512.0 | 512.0 |
| used rows | 512.0 | 512.0 | 512.0 |
| rows invalid | 0.0 | 0.0 | 0.0 |
| rows too short | 0.0 | 0.0 | 0.0 |

The selected source rows are matched to the diagnostic role universe:

| selected-row role | count per seed |
| ---: | ---: |
| 0 | 512 |
| 1 | 512 |
| 2 | 512 |
| 3 | 512 |

Every selected row has role set `{0,1,2,3}`. The query planner then holds out
the target role from the same query's observed context while keeping all query
roles inside the source role universe:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| query fraction in source role support | 1.0000 | 1.0000 | 1.0000 |
| query fraction in own observed context | 0.0000 | 0.0000 | 0.0000 |
| query fraction in global observed role set | 1.0000 | 1.0000 | 1.0000 |

This fixes the Report 084 hard-cell failure mode where every query role was
outside passive trace support.

## Atom Distribution

The selected rows keep the same broad-but-skewed passive trace distribution.
Across seeds:

| metric | range |
| --- | ---: |
| distinct atoms | 576-610 |
| normalized atom entropy | 0.7019-0.7192 |
| atom fraction `>=1024` | 0.0747-0.0825 |

Representative seed 17 top atoms:

| atom | count |
| ---: | ---: |
| 0 | 555 |
| 2 | 148 |
| 3 | 80 |
| 5 | 60 |
| 4 | 52 |
| 6 | 39 |
| 7 | 39 |
| 10 | 21 |

So the matched 4-role gate is not balanced synthetic data. It remains a skewed
passive-source diagnostic, but it is now role-universe matched.

## Geometry

The preflight separates re-encoded passive context geometry from learned
snapshot pattern-token geometry:

| comparison | diag mean | offdiag mean | diag top1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| re-encoded partial context -> matched synthetic full context | 0.5945 | 0.0220 | 0.8668 | 1.34 |
| learned snapshot pattern -> matched synthetic full context | 0.0034 | 0.0018 | 0.0031 | 254.93 |
| learned snapshot pattern -> re-encoded partial context | 0.0048 | 0.0025 | 0.0021 | 253.65 |

Learned pattern tokens remain highly clustered:

| learned-token pairwise metric | value |
| --- | ---: |
| offdiag mean | 0.2732 |
| offdiag std | 0.1277 |
| nearest-neighbor mean | 0.6275 |
| nearest-neighbor max | 1.0000 |

The role-universe preflight passes, but the learned-vector geometry problem
does not disappear under the matched 4-role setup.

## Interpretation

The matched 4-role gate is well-posed for the precommitted candidate/control
run:

- source support is sufficient (`1064 >= 512`);
- selected rows cover the entire diagnostic role universe `{0,1,2,3}`;
- query roles are in source support;
- the queried target role is not included in that query's observed context.

The geometry read is mixed:

- re-encoded passive context should be informative, with same-scene top1
  `0.8668` against matched synthetic full context;
- learned snapshot patterns remain near chance against both matched synthetic
  full context and re-encoded partial context.

That means the next run is allowed by the precommit, but it should be read as a
diagnostic discriminator, not as a graduation path. The expected split is:
`replay_observed_context_trace` may recover from the Report 084 role-universe
mismatch, while `replay_observed_pattern_context_trace` is still likely to
remain weak unless retrieval benefits from anchor-like clustering.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Run exactly the precommitted matched 4-role candidate/control discriminator
from `2026-05-25-phase5-prime-context-source-v2-gate.md`:

1. `replay_observed_context_trace`
2. `replay_observed_pattern_context_trace`

Each source must use the same seeds, rows, query schedule, and matched controls:
`candidate`, `random_role`, `deranged_role`, `shuffled_role`, and
`content_cleanup_positive`.

Stop after that fixed gate. Do not expand to a matrix, M2, or headline framing.
