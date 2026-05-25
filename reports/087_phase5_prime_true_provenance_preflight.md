# Report 087: Phase 5' True Provenance-Source Preflight

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-true-provenance-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Preflight only for the precommitted true provenance-source gate
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 086 showed that matching the passive trace role universe to the
diagnostic role universe only partially recovers the replay-observed context
source:

- `replay_observed_context_trace`: candidate `0.7668`, CI `[0.7550,0.7782]`;
- `replay_observed_pattern_context_trace`: candidate `0.5219`, CI
  `[0.5082,0.5355]`.

The next precommitted step was a true provenance-source preflight: before any
new candidate/control retrieval run, build or recover a source whose trace rows
natively live in the same role universe as the diagnostic scene and verify the
support/query/geometry diagnostics.

This report runs that preflight only.

## Artifact

Preflight script:

`scripts/phase5_prime_true_provenance_preflight.py`

Native provenance source artifact:

`reports/phase5_prime_native_provenance_context_source.json`

Source artifact SHA-256:

`bce6c08de6558cc9d01df51ca5eb53c26b1f8fea9c239687a26fa5196379b7b8`

Raw preflight JSON:

`reports/phase5_prime_true_provenance_preflight.json`

Preflight JSON SHA-256:

`620d5d6d2d3cb6e9e0a7d2162aeeab6a99151c0a318aa8a423f2786626265b99`

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_true_provenance_preflight.py
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_true_provenance_preflight.py
jq empty reports/phase5_prime_true_provenance_preflight.json
jq empty reports/phase5_prime_native_provenance_context_source.json
shasum -a 256 reports/phase5_prime_true_provenance_preflight.json reports/phase5_prime_native_provenance_context_source.json
```

## Source Contract

Source name:

`native_provenance_context_trace`

Source kind:

`synthetic_controlled_native_provenance`

The source artifact contains, for each seed:

- full `K_roles=16` encoder-term rows generated before retrieval evaluation;
- selected observed context roles for each row;
- fixed query plan with `scene`, `known_role`, `query_role`, and
  `observed_roles`;
- row/query generation seed formulas.

This is a controlled synthetic provenance source, not a learned or natural
trace source. Its purpose is to test whether a source that natively matches the
diagnostic role universe clears the support and geometry gates. It must not be
reported as solved learned trace context.

## Fixed Preflight Cell

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

## Support And Query Schedule

All support and query criteria pass:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| raw rows | 512.0 | 512.0 | 512.0 |
| eligible rows | 512.0 | 512.0 | 512.0 |
| required rows | 512.0 | 512.0 | 512.0 |
| used rows | 512.0 | 512.0 | 512.0 |
| rows invalid | 0.0 | 0.0 | 0.0 |
| rows too short | 0.0 | 0.0 | 0.0 |
| query fraction in source role support | 1.0000 | 1.0000 | 1.0000 |
| query fraction in own observed context | 0.0000 | 0.0000 | 0.0000 |
| query fraction in global observed role set | 1.0000 | 1.0000 | 1.0000 |

The query target role is always held out from that query's observed context,
but all query roles are inside the source role universe.

## Atom Distribution

The source distribution is broad and much cleaner than the prior passive
snapshot rows:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| distinct atoms | 1072.3 | 1070.0 | 1078.0 |
| normalized atom entropy | 0.9884 | 0.9877 | 0.9891 |
| atom fraction `>=1024` | 0.0308 | 0.0288 | 0.0333 |

The distribution is still synthetic/skewed by construction, but it does not
have the severe low-token concentration seen in the passive provenance
snapshot rows.

## Geometry

The required source/context geometry is clean:

| comparison | diag mean | offdiag mean | diag top1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| source observed context -> matched full scene context | 0.4098 | 0.0041 | 1.0000 | 1.00 |

The same-row partial source context is the top match for every source row
across all seeds.

## Pass Criteria

All precommitted preflight criteria pass:

| criterion | result |
| --- | :---: |
| eligible rows >= N | pass |
| used rows == N | pass |
| no invalid rows | pass |
| no too-short rows | pass |
| query roles in source support | pass |
| query role held out from own observed context | pass |
| source artifact SHA recorded | pass |
| geometry reported | pass |

## Interpretation

This preflight establishes that the first true provenance-source gate is
well-posed at the support/query/geometry level.

It does not establish a learned or natural provenance mechanism. The source is
a controlled synthetic native-provenance artifact: rows are generated directly
in the `K_roles=16` diagnostic role universe before retrieval evaluation. That
makes it a valid next discriminator for source plumbing and role-universe
matching, not a final context-source solution.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Because the preflight passes, the precommitted candidate/control gate is now
allowed, but only after this preflight artifact is preserved:

- source: `native_provenance_context_trace`;
- conditions: `candidate`, `random_role`, `deranged_role`, `shuffled_role`,
  `content_cleanup_positive`;
- same source rows, selected observed roles, query schedule, seeds, and support
  set as the source artifact;
- no full matrix, no M2, and no headline framing.
