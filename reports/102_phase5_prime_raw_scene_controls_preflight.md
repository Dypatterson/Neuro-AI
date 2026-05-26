# Report 102: Phase 5' Raw Scene-Energy Controls Preflight

**Date:** 2026-05-26
**Scope:** preflight-only controls planner for `raw_scene_energy_v0`
**Status:** static controls preflight passed; retrieval still not authorized

## Preamble

**Active phase:** Phase 5' precommit.

**Decision source:** Report 101 precommitted `raw_scene_energy_v0` as the first
bundle-first bridge baseline: `score_bias=None`, raw scene-MHN energy, and the
unchanged headline form `Delta E = E_content-prior - E_role-prior`.

**This report does not run retrieval.** It only proves that the required
controls can be represented against the same fixed source/query plan.

## Implementation

Added:

```text
scripts/phase5_prime_raw_scene_controls_preflight.py
tests/test_phase5_prime_raw_scene_controls_preflight.py
reports/phase5_prime_raw_scene_controls_preflight.json
```

The script:

- loads the committed non-synthetic native source and cleaned protocol;
- validates source/query-plan integrity for the fixed seed set;
- constructs one seed-17 bundle-first state for shape validation only;
- plans static control cells for `raw_scene_energy_v0`;
- records legacy-default mismatch flags before any retrieval run.

It does not call `run_branched_retrieval()`, settle branches, compute top1,
compute energies, run n=3/n=10, or make a headline claim.

## Artifact

```text
reports/phase5_prime_raw_scene_controls_preflight.json
SHA-256: 849e0f4382794b36d0af555a507b91f0dc8e1b1f8a89b2a4088a75d3c9aedb06
```

Source/protocol anchors:

```text
source: reports/phase5_prime_nonsynthetic_native_context_source.json
source SHA-256: 2200f3c1b1599c478ce6ecc6a3cb1e783c3bfd51ceda3f335642bb9158989398
cleanup preflight: reports/phase5_prime_natural_source_control_cleanup_preflight.json
cleanup SHA-256: 3c3a5c9781c7cb2f2a052923703ad8d4d60a84fcae5a590e82a1c8f29cf40aaf
prior gate: reports/phase5_prime_nonsynthetic_native_gate.json
prior gate SHA-256: 8ada5e4c01104f79545e9da156c3b870279fdff0da214d54d969f1d85a2ef280
protocol: non_special_unique_target_freq_le_32
```

## Planned Controls

The artifact plans nine static cells:

```text
main_content_K4_g0.5
main_role_K4_g0.5
random_schema_K4_g0.5
k1_content_g0.5
k1_role_g0.5
no_prior_content_K4_g0
no_prior_role_K4_g0
no_schema_store_content_K4_g0.5
no_schema_store_role_K4_g0.5
```

All cells use:

```text
baseline_id = raw_scene_energy_v0
headline_metric = Delta E = E_content-prior - E_role-prior
score_bias = None
energy_readout = raw_scene_mhn_energy
retrieval_executed = false
same query_plan_ref = source.query_plan_by_seed
```

The no-schema-store cells change only the schema/prior source:

```text
schema_source = content_codebook
memory_source = bundle_first_scene_store
headline_metric unchanged
```

## Static Checks

Pass criteria are all true:

```text
framing_preflight_only=true
raw_scene_energy_v0_selected=true
score_bias_none_all_cells=true
no_retrieval_executed=true
main_delta_pair_planned=true
random_schema_planned=true
k1_pair_planned=true
no_prior_pair_planned=true
no_schema_store_pair_planned=true
same_query_set_all_controls=true
source_manifest_sha_recorded=true
protocol_name_recorded=true
query_plan_integrity_ok=true
scene_store_shapes_validated=true
no_schema_store_shapes_planned_not_materialized=true
config_mismatch_flags_recorded=true
headline_metric_unchanged=true
stop_conditions_encoded=true
```

Shape validation:

```text
scene_matrix_shape = [512, 4096]
roles_shape = [16, 4096]
content_shape = [2048, 4096]
query_context_tokens_shape = [512, 4096]
```

The content-codebook no-schema-store bindings are planned as
`[2048,16,4096]` but not materialized.

## Config Mismatch Flags

The planner records the expected mismatch with legacy Phase 5 headline
defaults:

```text
beta: raw_scene_value=30.0, legacy_headline_default=10.0
k_main: raw_scene_value=4, legacy_headline_default=1
```

These are precommitted by Report 101 for the raw-scene bridge baseline, but they
must remain visible before any retrieval run.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_raw_scene_controls_preflight.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_raw_scene_controls_preflight -v
```

```text
Ran 2 tests in 0.382s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 388 tests in 1.259s
OK
```

```bash
git diff --check
```

```text
passed
```

```bash
rg -n "run_branched_retrieval|_run_probe_conditions|settle_branch|retrieve\(" \
  scripts/phase5_prime_raw_scene_controls_preflight.py \
  tests/test_phase5_prime_raw_scene_controls_preflight.py
```

```text
no matches
```

## Boundary

This report authorizes no evidence run. It does not claim:

- a candidate/control gate;
- retrieval/top1/energy results;
- n=3 or n=10 evidence;
- M1 escalation;
- M2 implementation or run;
- a full matrix;
- a Phase 5 `Delta E` headline run;
- Phase 5 graduation.

## Anti-Homunculus Check

Pass. The artifact is a static control plan. It introduces no adaptive routing,
metric-triggered bias, sampler switching, retrieval result feedback, or
best-of-N selection.

## Next Step

Before running even a pilot retrieval smoke, write a retrieval-smoke precommit
that fixes:

- exact seed and cue subset;
- exact nine planned cells above;
- whether the beta/K mismatch flags are accepted for the smoke;
- output artifact path and required readback fields;
- stop condition if any control cannot run against the same fixed query set.
