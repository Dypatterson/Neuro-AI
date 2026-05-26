# Report 104: Phase 5' Raw Scene-Energy Degeneracy Analysis

**Date:** 2026-05-26
**Scope:** analysis-only follow-up to the Report 103 seed-17/four-probe smoke
**Status:** completed; `raw_scene_energy_v0` settled-state energy is degenerate
at smoke scale

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This analysis
does not run or replace the headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This analysis
reruns the exact Report 103 seed-17/four-probe smoke scope only.

**Last verified result:** Report 103 showed the nine required pilot cells execute
and produce finite branches, but all paired `Delta E` values were `0.0` and all
per-probe minimum energies were `-1.0`.

**Why this analysis now:** Report 103 made raw-energy degeneracy the next blocker
before any n=3, n=10, gate, full matrix, M2, headline, or graduation run.

## Implementation

Added:

```text
scripts/phase5_prime_raw_scene_energy_degeneracy_analysis.py
tests/test_phase5_prime_raw_scene_energy_degeneracy_analysis.py
reports/phase5_prime_raw_scene_energy_degeneracy_analysis.json
```

The analysis reruns the exact Report 103 scope and logs score-level evidence per
branch:

- final top scene index;
- top scene score;
- recomputed raw softmax energy;
- branch-recorded raw/step3 energies;
- `logsumexp(beta * scores) - beta * top_score`;
- saturation flags for top score and raw energy.

It does not add a mechanism, change the headline, select routes, or widen
evidence scale.

## Artifact

```text
reports/phase5_prime_raw_scene_energy_degeneracy_analysis.json
SHA-256: d7f4339f70a90868a7a654cb398bebe3e080ff91f1e93564abd3ba16d2c23c7e
size: 128K
```

Source/protocol anchors:

```text
source: reports/phase5_prime_nonsynthetic_native_context_source.json
source SHA-256: 2200f3c1b1599c478ce6ecc6a3cb1e783c3bfd51ceda3f335642bb9158989398
cleanup preflight: reports/phase5_prime_natural_source_control_cleanup_preflight.json
cleanup SHA-256: 3c3a5c9781c7cb2f2a052923703ad8d4d60a84fcae5a590e82a1c8f29cf40aaf
prior gate: reports/phase5_prime_nonsynthetic_native_gate.json
prior gate SHA-256: 8ada5e4c01104f79545e9da156c3b870279fdff0da214d54d969f1d85a2ef280
Report 103 smoke artifact SHA-256: 5fe527a9741e3a0751718747a76a98c978159ff3ebcc2ab08906c4e2ff656715
protocol: non_special_unique_target_freq_le_32
```

## Result

All analysis pass criteria are true:

```text
analysis fixed to seed 17 / <=4 probes = true
all nine cells rerun = true
all paired Delta E groups zero = true
all final top scores saturated = true
all energies saturated at -1 = true
all step3 energies equal raw = true
recomputed energy matches branch readback = true
top scene indices not all identical = true
```

Aggregate readback:

```text
total_branches = 120
unique_top_scene_indices = [0,1,394]
unique_top_scene_index_count = 3
top_score_min = 1.0
top_score_max = 1.0
max_gap_to_negative_one = 0.0
max_logsumexp_excess_over_top_logit = 0.0
target_scene_top_rate = 0.9833333333333333
```

Paired `Delta E` groups:

```text
main:            [0.0,0.0,0.0,0.0], mean=0.0
K=1:             [0.0,0.0,0.0,0.0], mean=0.0
no-prior:        [0.0,0.0,0.0,0.0], mean=0.0
no-schema-store: [0.0,0.0,0.0,0.0], mean=0.0
```

Per-cell final top scene indices:

```text
k1_content_g0.5:                         [0,1]
k1_role_g0.5:                            [0,1]
main_content_K4_g0.5:                    [0,1,394]
main_role_K4_g0.5:                       [0,1]
no_prior_content_K4_g0:                  [0,1]
no_prior_role_K4_g0:                     [0,1]
no_schema_store_content_K4_g0.5:         [0,1]
no_schema_store_role_K4_g0.5:            [0,1]
random_schema_K4_g0.5:                   [0,1]
```

## Diagnosis

`raw_scene_energy_v0` is degenerate in its current settled-state form.

Every branch settles exactly onto a normalized stored scene bundle. In this
substrate, a stored scene bundle has self-similarity `1.0`, and the raw softmax
energy

```text
E(q*) = -logsumexp(beta * sim(scene_store, q*)) / beta
```

saturates at `-1.0` when `q*` is a stored scene. That makes the energy readout
unable to distinguish content-prior, role-prior, random-schema, K=1, no-prior,
or no-schema-store branches once they have landed on any scene attractor.

The fact that `unique_top_scene_indices = [0,1,394]` while every branch still
has top score `1.0` and energy `-1.0` is the key finding: the energy readout is
measuring "landed on a stored scene" rather than "landed on the right or
structurally preferred scene."

## Boundary

This closes the Report 103 blocker but does not authorize a wider retrieval run.
Do not run n=3, n=10, a candidate/control gate, full matrix, M2, headline, or
graduation.

The next allowed work is a precommitted bridge-readout decision: define one
fixed, non-saturated energy readout for bundle-first scenes before collecting
new evidence. Candidate directions must be specified before running them; do not
select among alternatives from observed performance.

## Anti-Homunculus Check

Pass. This is passive score analysis over fixed cells. It adds no adaptive
routing, metric-triggered bias, route selection, sampler switching, or
best-condition selection.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_raw_scene_energy_degeneracy_analysis.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_raw_scene_energy_degeneracy_analysis -v
```

```text
Ran 1 test in 0.380s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 390 tests in 1.291s
OK
```

```bash
git diff --check
```

```text
passed
```
