# Report 109: Phase 5' Strict Discriminator Bridge Viability

**Date:** 2026-05-26
**Scope:** fixed Report 108 seed-17/four-probe strict-discriminator diagnostic
**Status:** completed; current bridge path is not viable

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This is not a
headline run and does not claim Phase 5 graduation.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This diagnostic
includes the random-prior control but does not run the full control matrix.

**Last verified result:** Report 108 precommitted the strict query subset and
the exact viability criteria.

**Why this diagnostic now:** The user asked to proceed until the current bridge
path is classified as viable or not.

## Implementation

Added:

```text
scripts/phase5_prime_strict_discriminator_viability.py
tests/test_phase5_prime_strict_discriminator_viability.py
reports/phase5_prime_strict_discriminator_viability.json
```

The runner loads the Report 108 precommit artifact and runs exactly:

```text
seed = 17
probes = [29, 110, 220, 331]
conditions = content, role, random
primary combiner = q_bundle
beta sweep = {10, 30}
gamma sweep = {0, 0.5, 1, 2, 4, 8}
readout = cue_conditioned_scene_energy_v1
```

It does not run n=3, n=10, a gate, a full matrix, M2, headline, or graduation.

## Artifact

```text
reports/phase5_prime_strict_discriminator_viability.json
SHA-256: bc2ffa2d5c23dba340a2a7d6c40035e68617cf3646f82c0b9daf6965eae519aa
size: 724K
```

Precommit source:

```text
reports/phase5_prime_strict_discriminator_precommit.json
SHA-256: 4c0deadbe9b3c8ad3c9f7ebce0140a01781d17964038152f2019ea55ce907e70
```

## Decision

```text
decision_id = current_bridge_path_not_viable_v1
viability = not_viable_current_bridge
```

Path classified:

```text
current bundle-first Delta E bridge using scene-MHN branch dynamics,
preferred bundle-resettle combiner, and cue_conditioned_scene_energy_v1
```

This does **not** falsify bundle-first scene memory as a context-completion
architecture, range-shaped replay, M2, or a future user-approved headline
redesign. It closes the current bridge/readout path only.

## Result

No precommitted operating point passed the viability criteria:

```text
viable operating points = 0/12
max mean Delta E bundle = 0.0021672621369361877
magnitude floor = 0.0055
all operating points raw-energy saturated = true
all operating points uniform branch weights = true
```

Default legacy-smoke operating point:

```text
beta = 30
gamma = 0.5
Delta E bundle values = [0.00409446656703949, 0.0, 0.0, 0.004574581980705261]
mean Delta E bundle = 0.0021672621369361877
content target-scene hits = 0/4
role target-scene hits = 0/4
random target-scene hits = 0/4
role-random same-scene count = 4/4
viable = false
```

Headline-beta operating point:

```text
beta = 10
gamma = 0.5
Delta E bundle values =
  [-0.02223220467567444, -0.03241558372974396,
   -0.002141237258911133, -0.015152081847190857]
mean Delta E bundle = -0.017985276877880096
content target-scene hits = 0/4
role target-scene hits = 0/4
random target-scene hits = 0/4
role-random same-scene count = 3/4
viable = false
```

High-gamma operating points also fail. At `gamma=8`, role reaches the target
scene on only one of four probes and mean `Delta E` is strongly negative under
both beta values.

## Diagnosis

The current bridge path fails for three linked reasons:

1. **Scene-MHN raw energy remains saturated.** Every branch lands on a stored
   scene with raw/step3 energy `-1.0`, so the design-preferred branch combiner
   receives uniform branch weights instead of an energy signal.
2. **Role priors do not create clean target-scene settlement.** Even when the
   static role top-K contains the target scene and content top-K excludes it,
   the settled bundle state usually follows the same cue/content attractor as
   random-prior control.
3. **The cue-conditioned readout cannot rescue the bridge.** At the active
   `beta=30,gamma=0.5` operating point, mean `Delta E` is below the magnitude
   floor and role equals random on all four final bundle scenes. At `beta=10`,
   the sign is negative.

The issue is not just the earlier min-branch summary. Report 109 reruns the
preferred bundle-resettle combiner and the path still does not separate role
from content/random controls.

## Boundary

Stop widening this bridge path. Do not run it at n=3, n=10, gate scale, full
matrix, M2, headline, or graduation scale.

Any next path needs a fresh precommit that changes the bridge objective or
returns to another Phase 5' lane. Do not treat another bridge readout tweak as
routine continuation.

## Anti-Homunculus Check

Pass. The strict query selector and beta/gamma sweep were fixed in Report 108.
All diagnostics are passive; no observed metric changes routing inside a run.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_strict_discriminator_viability.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_strict_discriminator_viability -v
```

```text
Ran 2 tests in 0.709s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 398 tests in 1.318s
OK
```

```bash
git diff --check
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python - <<'PY'
import json, hashlib
from pathlib import Path
checks = {
    "reports/phase5_prime_strict_discriminator_precommit.json":
        "4c0deadbe9b3c8ad3c9f7ebce0140a01781d17964038152f2019ea55ce907e70",
    "reports/phase5_prime_strict_discriminator_viability.json":
        "bc2ffa2d5c23dba340a2a7d6c40035e68617cf3646f82c0b9daf6965eae519aa",
}
for name, expected in checks.items():
    path = Path(name)
    payload = json.loads(path.read_text())
    assert payload["passes_all_criteria"] is True
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected
viab = json.loads(Path("reports/phase5_prime_strict_discriminator_viability.json").read_text())
assert viab["decision"]["viability"] == "not_viable_current_bridge"
assert viab["aggregate"]["viable_operating_point_count"] == 0
assert viab["aggregate"]["max_mean_delta_e_bundle"] < viab["viability_plan"]["magnitude_floor"]
print("json readback passed")
PY
```

```text
json readback passed
```
