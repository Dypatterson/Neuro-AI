# Report 108: Phase 5' Strict Discriminator Precommit

**Date:** 2026-05-26
**Scope:** static seed-17 query-subset precommit after Report 107
**Status:** completed; four strict-disagreement probes selected

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This precommit
does not run or replace the headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This precommit is
not a control matrix.

**Last verified result:** Report 107 localized the Report 106 zero-`Delta E`
failure to `min_branch_target_scene_attractor_collapse`.

**Why this precommit now:** The next allowed work was a stricter discriminator
or fixed query subset before any widening.

## Implementation

Added:

```text
scripts/phase5_prime_strict_discriminator_precommit.py
tests/test_phase5_prime_strict_discriminator_precommit.py
reports/phase5_prime_strict_discriminator_precommit.json
```

The script audits all 512 seed-17 cleaned-protocol queries by static prior
ranking only. It does not call `run_branched_retrieval()`.

## Selector

```text
selector_id = role_topk_target_content_topk_excludes_target_v1
```

A probe is selected only when:

```text
target scene is present in role-prior top-K schema set
target scene is absent from content-prior top-K schema set
K_main = 4
delta_redundant = 0.95
```

Selected probes:

```text
[29, 110, 220, 331]
```

These are the only four seed-17 probes where the strict top-K discriminator is
available under the fixed cleaned protocol.

## Artifact

```text
reports/phase5_prime_strict_discriminator_precommit.json
SHA-256: 4c0deadbe9b3c8ad3c9f7ebce0140a01781d17964038152f2019ea55ce907e70
size: 7.5K
```

Static audit summary:

```text
n_queries = 512
target_in_both_topK = 507
target_in_role_topK_not_content_topK = 4
target_in_content_topK_not_role_topK = 1
target_is_content_top1 = 497
target_is_role_top1 = 492
target_is_role_top1_not_content_top1 = 12
content_role_topK_equal = 0
```

This confirms Report 106 used an easy subset: most seed-17 queries expose the
target scene to both content and role top-K branch sets. The strict subset is
small but sufficient for a local bridge viability falsifier.

## Viability Plan

```text
plan_id = bundle_resettle_strict_discriminator_viability_v1
selected probes = [29, 110, 220, 331]
conditions = content, role, random
primary combiner = q_bundle
comparison combiner = q_greedy
run_combiners = true
include_surprise_branch = false
readout = cue_conditioned_scene_energy_v1
beta sweep = {10, 30}
gamma sweep = {0, 0.5, 1, 2, 4, 8}
magnitude floor = 5.5e-3
```

The primary combiner is the design-preferred energy-weighted bundle plus
unbiased re-settle, not the min-branch summary used in Reports 103-107.

Precommitted viability criteria:

```text
mean Delta E bundle >= 5.5e-3
role beats content on at least 3/4 probes
role bundle target-scene rate > content
role bundle target-scene rate > random
role energy lower than random on mean
```

## Boundary

This report authorizes only the fixed seed-17/four-probe strict-discriminator
viability diagnostic above. It does not authorize n=3, n=10, a candidate/control
gate, full matrix, M2, a Phase 5 headline run, or graduation.

## Anti-Homunculus Check

Pass. The selector is static and fixed before retrieval. The beta/gamma sweep is
fixed by this precommit and cannot choose a production mode from observed
performance.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_strict_discriminator_precommit.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_strict_discriminator_precommit -v
```

```text
Ran 1 test in 0.702s
OK
```
