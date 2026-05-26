# Report 100: Phase 5' Bundle-First Delta E Bridge Preflight

**Date:** 2026-05-26
**Scope:** preflight-only bridge from fixed bundle-first scene states to the
existing Phase 5 content-prior-vs-role-prior energy comparison
**Status:** interface preflight passed; production headline not ready

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This preflight
does not change that metric and does not make a headline claim.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This preflight does
not execute that graduation-control matrix.

**Why this implementation now:** Report 099 made the next useful step a small
bridge preflight asking whether fixed bundle-first scene states can be wrapped
by the existing Phase 5 `Delta E` comparison without changing the headline.

## Implementation

Added:

```text
scripts/phase5_prime_delta_bridge_preflight.py
tests/test_phase5_prime_delta_bridge_preflight.py
reports/phase5_prime_delta_bridge_preflight.json
```

Extended:

```text
src/energy_memory/phase5/bundle_first_scene_memory.py
src/energy_memory/phase5/__init__.py
```

The reusable module now exposes `BundleFirstSeedState` and
`build_bundle_first_seed_state()` so downstream bridge code can use the same
fixed scene/content construction as the candidate/control gates.

The preflight script:

- loads the committed non-synthetic native source and cleaned protocol;
- builds seed-17 bundle-first scene states for the fixed cleaned query plan;
- stores scene states in `TorchHopfieldMemory`;
- builds schema bindings with the existing `experiments/40_phase5_branching.py`
  `compute_schema_bindings()` helper;
- calls the existing `run_branched_retrieval()` machinery for content, role,
  and random priors;
- records whether paired content-vs-role energy is computable.

It sets `score_bias=None`, so Step-3 energy equals raw scene-MHN energy. That is
intentional for the bridge check, and it is also the main unresolved production
decision.

## Artifact

```text
reports/phase5_prime_delta_bridge_preflight.json
SHA-256: 72033e8b8af19a6c3d0f5895e20d3b90598881acd2e872e9303d08c904bbbfee
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

## Result

Configuration:

```text
seed=17
D=4096
N=512
K_roles=16
C_codebook=2048
scene_token_weight=0.25
beta=30.0
gamma=0.5
k_main=4
n_probe_cues=4
device=cpu
```

Interface checks passed:

```text
scene_matrix_shape_ok=true
memory_scene_count_matches=true
schema_atom_idx_shape_ok=true
schema_bindings_shape_ok=true
cue_bindings_present_all_probes=true
content_prior_returned_branches=true
role_prior_returned_branches=true
random_prior_returned_branches=true
content_prior_finite_energy=true
role_prior_finite_energy=true
random_prior_finite_energy=true
delta_e_computable_all_probes=true
headline_metric_unchanged=true
preflight_fences_present=true
```

Probe-only diagnostic:

```text
delta_e_values = [0.0, 0.0, 0.0, 0.0]
mean_probe_delta_e = 0.0
step3_energy_equals_raw_because_score_bias_none = true
production_headline_ready = false
```

The zero probe deltas are not interpreted as evidence for or against Phase 5.
They only show that the paired content/role energy field can be computed through
the existing `run_branched_retrieval()` path when the bundle-first scene states
are used as the schema store.

## Production Blockers

Before any real headline run, the spec must resolve:

1. Scene-state `score_bias` mapping. The existing Phase 5 headline uses the
   Step-3-weighted landscape. This preflight uses `score_bias=None`, so it does
   not answer whether bundle-first scene states should use raw scene-MHN energy
   or a legitimate scene-level Step-3 bias.
2. Required controls. Random-schema, K=1, no-prior, and no-schema-store controls
   were not run here.
3. Evidence scale. This is seed 17 plus four probe cues only. Seed 17 is a
   pilot/reference seed, not representative evidence.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  src/energy_memory/phase5/bundle_first_scene_memory.py \
  src/energy_memory/phase5/__init__.py \
  scripts/phase5_prime_delta_bridge_preflight.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_delta_bridge_preflight \
  tests.test_phase5_bundle_first_scene_memory -v
```

```text
Ran 4 tests in 0.421s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 386 tests in 1.281s
OK
```

```bash
git diff --check
```

```text
passed
```

## Boundary

This report does not authorize or claim:

- a new candidate/control gate;
- M1 escalation;
- M2 implementation or run;
- a full matrix;
- a Phase 5 `Delta E` headline run;
- Phase 5 graduation;
- a new top1-based graduation metric.

## Anti-Homunculus Check

Pass. The bridge uses fixed scene states, fixed prior selectors, fixed
settling, and passive diagnostics. It does not add adaptive routing,
metric-triggered branch selection, sampler switching, or best-of-N condition
selection.

## Next Step

Write a narrow spec decision for the production bridge before running any
headline-scale experiment:

```text
Does the bundle-first Phase 5 headline use raw scene-MHN energy
(score_bias=None), or does it require a scene-level Step-3 score_bias derived
from an explicit, fixed, non-adaptive scene/consolidation mapping?
```

If the answer changes the headline landscape or required controls, stop and
update the Phase 5 design/checklist before running a production experiment.
