# Report 105: Phase 5' Cue-Conditioned Bridge Readout Precommit

**Date:** 2026-05-26
**Scope:** fixed non-saturated bridge-readout decision after Report 104
**Status:** precommitted; no retrieval or evidence run

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This precommit
keeps the `Delta E` sign convention but does not run or replace the headline.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. The same nine
Report 102 cells remain the required pilot surface.

**Last verified result:** Report 104 showed settled-state `raw_scene_energy_v0`
is degenerate: every final branch lands on a stored scene with top score `1.0`
and energy `-1.0`, so all paired `Delta E` groups are zero even when selected
scene indices differ.

**Why this decision now:** Report 104 permits only a fixed non-saturated
bridge-readout precommit before collecting any new evidence.

## Decision

The next bridge readout is:

```text
readout_id = cue_conditioned_scene_energy_v1
E_bridge(q_scene_final, cue) = -Re(<q_scene_final, cue>) / D
Delta E = E_content-prior - E_role-prior
```

Lower energy means the final settled scene is more compatible with the fixed
cue. Positive `Delta E` keeps the existing sign convention: the role-prior
branch has lower energy than the content-prior branch under the fixed readout.

## Rationale

Report 104 shows that scoring the final scene against the scene store reads out
self-similarity. Any exact stored scene has self-similarity `1.0`, so raw
scene-MHN energy saturates at `-1.0` regardless of whether the selected scene is
structurally useful.

`cue_conditioned_scene_energy_v1` scores the same final scene state against the
fixed cue instead. This preserves the bundle-first bridge path while avoiding
the self-similarity ceiling. It is still label-free and substrate-local: the
readout uses only the cue vector and final settled scene state.

## Allowed Inputs

The readout may use only:

```text
fixed cue vector
final settled scene state
```

Forbidden inputs:

```text
target atom label
target scene label
branch prior vector
content-cleanup top1
observed performance metrics
condition winner selection
```

The settling path, schema selection, source/query plan, nine controls,
`score_bias=None`, and `Delta E` sign convention remain unchanged.

## Implementation

Added:

```text
src/energy_memory/phase5/bridge_readouts.py
scripts/phase5_prime_bridge_readout_precommit.py
tests/test_phase5_bridge_readouts.py
tests/test_phase5_prime_bridge_readout_precommit.py
reports/phase5_prime_bridge_readout_precommit.json
```

The helper implements:

```text
cue_conditioned_scene_energy(cue, scene_state)
delta_e_content_minus_role(content_energy, role_energy)
```

The precommit script writes a static artifact only. It does not call
`run_branched_retrieval()`.

## Artifact

```text
reports/phase5_prime_bridge_readout_precommit.json
SHA-256: 3abcac6a6153ab50d90e55585db5c2df9812a35981573b2a9e0ed1e5b0ba5045
size: 12K
```

Readback:

```text
passes_all_criteria = true
readout_id = cue_conditioned_scene_energy_v1
planned_cells = 9
retrieval_executed = false
```

Toy sanity check:

```text
self_energy = -1.0
different_scene_energy = -0.0
different_scene_not_saturated = true
delta_e_content_minus_role = 1.0
positive_delta_when_role_energy_lower = true
```

## Boundary

This authorizes only implementation of the cue-conditioned readout in the
existing seed-17/four-probe pilot surface. It does not authorize n=3, n=10, a
candidate/control gate, full matrix, M2, a Phase 5 headline run, or graduation.

The next allowed work is a seed-17/four-probe smoke over the same nine cells,
using `cue_conditioned_scene_energy_v1`, with readback of paired `Delta E`,
same-query-subset checks, and explicit no-widening flags.

## Anti-Homunculus Check

Pass. The readout is a fixed cue-scene compatibility term. It uses no target
labels, branch-prior energy term, metric feedback, route switching, or
best-condition selection.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  src/energy_memory/phase5/bridge_readouts.py \
  scripts/phase5_prime_bridge_readout_precommit.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_bridge_readouts \
  tests.test_phase5_prime_bridge_readout_precommit -v
```

```text
Ran 3 tests in 0.376s
OK
```
