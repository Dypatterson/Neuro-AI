# Report 096: Phase 5' Natural-Source Mini-Matrix Preflight

**Date:** 2026-05-26
**Scope:** broader follow-up preflight only
**Status:** passed support and static-plan checks

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This preflight
does not measure that headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This preflight does
not execute that graduation-control matrix.

**Last verified result:** Report 095 proved byte-identical reusable-path parity
for the cleaned Report 093 natural-source gate.

**Why this experiment now:** Report 095 closed the cleaned-protocol drill-down
chain. The next safe step is a static mini-matrix preflight, not another local
parity run, M2, a full matrix, or a Phase 5 headline run.

## Command

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase5_prime_natural_source_mini_matrix_preflight.py
```

Output artifact:

```text
reports/phase5_prime_natural_source_mini_matrix_preflight.json
```

SHA-256:

```text
8486bfbd8092bcac375f01945defc5024b81f35277b4b470fc292c379d787500
```

## Result

The preflight passes all criteria:

- cleaned protocol: `non_special_unique_target_freq_le_32`;
- fixed hard cell: `D=4096`, `K_roles=16`, `N=512`, `context_roles=4`,
  `scene_token_weight=0.25`;
- seeds: `17, 11, 23, 1, 2, 3, 5, 7, 13, 29`;
- queries: `512` per seed;
- cue-noise sweep: `{0.0, 0.05, 0.10, 0.15}`;
- planned cells: `32`;
- conditions: `candidate`, `random_role`, `deranged_role`,
  `fixedpoint_free_shuffled_role`, `content_cleanup_positive`,
  `bundle_positive`, `perfect_cue`, and `no_scene_token_baseline`;
- role-negative same-scene exact opportunities are zero for random, deranged,
  and fixed-point-free shuffled controls.

## Interpretation

This is a planning/support artifact only. It freezes the selected cleaned query
plan and the broader candidate/control cell grid before any retrieval run.

It does not run scene-MHN retrieval, content cleanup, top1 scoring, M2, or the
Phase 5 `Delta E` headline. It does not claim graduation and does not authorize
a full matrix.

## Verification

Compile check passed:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_natural_source_mini_matrix_preflight.py \
  tests/test_phase5_prime_natural_source_mini_matrix_preflight.py
```

Focused tests passed:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_natural_source_protocol \
  tests.test_phase5_prime_natural_source_control_cleanup_preflight \
  tests.test_phase5_prime_natural_source_control_cleanup_gate \
  tests.test_phase5_prime_natural_source_mini_matrix_preflight -v
```

Result:

```text
Ran 14 tests in 0.315s
OK
```

Full suite passed:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

Result:

```text
Ran 382 tests in 1.248s
OK
```

## Next Step

The next allowed implementation step is a separate fixed gate precommit that
consumes this exact mini-matrix plan. That gate should report top1, Wilson CI,
leave-one-seed-out, `scene_tix`, `content_tix`, entropy, margins, and
scene-vs-content failure split for the planned cells only.
