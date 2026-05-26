# Report 089: Phase 5' Non-Synthetic Native Provenance Preflight

**Date:** 2026-05-25
**Branch:** `codex/phase5-prime-nonsynthetic-native-preflight`
**Base stack:** `phase5-m1-role-energy-stack`
**Scope:** Preflight only for the precommitted non-synthetic native-provenance
source gate
**Status:** Diagnostic only. No Phase 5 graduation, no Delta E headline, no
full matrix, and no M2 commitment.

## Question

Report 088 showed that a controlled synthetic native-provenance source restores
the K=16 context-source gate (`candidate=0.9779`, CI `[0.9735,0.9816]`), but
that source was generated directly in the diagnostic role universe.

The next precommitted question was whether a non-synthetic native source can
provide K=16 encoder-term rows with enough support and clean enough geometry
before any candidate/control retrieval run.

This report runs that preflight only.

## Artifact

Preflight script:

`scripts/phase5_prime_nonsynthetic_native_preflight.py`

Source artifact:

`reports/phase5_prime_nonsynthetic_native_context_source.json`

Source artifact SHA-256:

`2200f3c1b1599c478ce6ecc6a3cb1e783c3bfd51ceda3f335642bb9158989398`

Raw preflight JSON:

`reports/phase5_prime_nonsynthetic_native_preflight.json`

Preflight JSON SHA-256:

`868ac20feb75169eaeb6b4440bc32aeb4d336a74af6c40dbd793aadebeca9bab`

Verification commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile scripts/phase5_prime_nonsynthetic_native_preflight.py tests/test_phase5_prime_nonsynthetic_native_preflight.py
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase5_prime_nonsynthetic_native_preflight -v
PYTHONPATH=src:. .venv/bin/python scripts/phase5_prime_nonsynthetic_native_preflight.py
jq empty reports/phase5_prime_nonsynthetic_native_preflight.json
jq empty reports/phase5_prime_nonsynthetic_native_context_source.json
shasum -a 256 reports/phase5_prime_nonsynthetic_native_preflight.json reports/phase5_prime_nonsynthetic_native_context_source.json
```

## Source Contract

Source name:

`trajectory_native_provenance_context_trace`

Source family:

`trajectory_derived_native`

Source kind:

`repo_sample_phase2_window_trace`

The source rows are sampled from repo-sample Phase 2 train windows, encoded with
`encode_window_with_provenance()`, wrapped as `TrajectoryTrace.encoder_terms`,
and placed in a `ReplayStore`. This is not the Report 087/088 controlled
source: no generated diagnostic scene rows are used as source rows.

The source co-occurrence is therefore `repo_sample_natural`, not the earlier
synthetic `skewed` generator. That makes this a source-validity preflight, not
a direct top1 comparison to the controlled-skew Report 088 gate.

## Fixed Preflight Cell

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
corpus_source=repo_sample
cooccurrence=repo_sample_natural
```

Seeds:

`17, 11, 23, 1, 2, 3, 5, 7, 13, 29`

## Support And Query Schedule

All support and query criteria pass:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| raw rows | 5352.0 | 5352.0 | 5352.0 |
| eligible rows | 5352.0 | 5352.0 | 5352.0 |
| required rows | 512.0 | 512.0 | 512.0 |
| used rows | 512.0 | 512.0 | 512.0 |
| replay store rows | 512.0 | 512.0 | 512.0 |
| rows invalid | 0.0 | 0.0 | 0.0 |
| rows too short | 0.0 | 0.0 | 0.0 |
| query fraction in source role support | 1.0000 | 1.0000 | 1.0000 |
| query fraction in own observed context | 0.0000 | 0.0000 | 0.0000 |
| query fraction in global observed role set | 1.0000 | 1.0000 | 1.0000 |

The query target role is always held out from that query's observed context,
but every query role is inside the source role universe.

## Atom Distribution

The repo-sample source distribution is broad but naturally skewed by document
frequency:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| distinct atoms | 1501.5 | 1484.0 | 1522.0 |
| normalized atom entropy | 0.8309 | 0.8276 | 0.8389 |
| atom fraction special tokens | 0.0953 | 0.0869 | 0.1041 |

The source covers many more distinct atoms than the synthetic controlled source,
but its entropy is lower because the repo text has high-frequency function
tokens.

## Geometry

The required source/context geometry is clean:

| comparison | diag mean | offdiag mean | diag top1 | mean rank |
| --- | ---: | ---: | ---: | ---: |
| source observed context -> matched full source context | 0.4090 | 0.0051 | 0.9998 | 1.0002 |

The same-row partial source context is the top match for essentially every row
across all seeds. The only miss rate is `1/5120` source rows.

## Pass Criteria

All precommitted preflight criteria pass:

| criterion | result |
| --- | :---: |
| eligible rows >= N | pass |
| used rows == N | pass |
| replay store rows == N | pass |
| no invalid rows | pass |
| no too-short rows | pass |
| query roles in source support | pass |
| query role held out from own observed context | pass |
| source artifact SHA recorded | pass |
| geometry reported | pass |

## Interpretation

This preflight establishes that a repo-sample, encoder-native provenance source
is well-posed at the support/query/geometry level for K=16.

It does not establish a solved retrieval mechanism. No candidate/control top1
condition was run. Because the source is natural repo co-occurrence rather than
the controlled skewed generator, the next gate should be framed as a
non-synthetic source discriminator, not as a direct replay of Report 088.

## Boundary

- No Phase 5 graduation claim.
- No Phase 5 `Delta E` headline run.
- No full all-controls matrix.
- No M2 commitment.
- No MQAR/bAbI headline pivot.
- No claim about continuous learning across multiple domains.
- No claim about an emergent self.

## Next Work

Because the preflight passes, the fixed candidate/control gate is now allowed
for this source only:

- source: `trajectory_native_provenance_context_trace`;
- conditions: `candidate`, `random_role`, `deranged_role`, `shuffled_role`,
  `content_cleanup_positive`;
- same source rows, selected observed roles, query schedule, seeds, and support
  set as the source artifact;
- report top1, Wilson CI, per-seed, leave-one-seed-out, `scene_tix`,
  `content_tix`, entropy, margin, and source SHA;
- no full matrix, no M2, and no headline framing.
