# Report 106: Phase 5' Cue-Conditioned Bridge Retrieval Smoke

**Date:** 2026-05-26
**Scope:** seed-17/four-probe pilot smoke over the Report 102 nine-cell surface
using `cue_conditioned_scene_energy_v1`
**Status:** completed; plumbing passed, but paired `Delta E` remains zero

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This smoke keeps
the sign convention but is not a headline run.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. The smoke executes
the same nine cells planned in Report 102.

**Last verified result:** Report 105 precommitted
`cue_conditioned_scene_energy_v1` and authorized only this seed-17/four-probe
pilot smoke.

**Why this experiment now:** `phase-5-prime-checklist.md` item I8 required a
cue-conditioned pilot readback before any widening.

## Implementation

Added:

```text
scripts/phase5_prime_cue_conditioned_bridge_smoke.py
tests/test_phase5_prime_cue_conditioned_bridge_smoke.py
reports/phase5_prime_cue_conditioned_bridge_smoke_seed17.json
```

The runner reuses the fixed source/query plan and same nine Report 102 cells.
It records:

- per-branch `cue_conditioned_scene_energy_v1`;
- per-branch raw scene energy for comparison;
- final top scene index and top scene score;
- paired `Delta E` for main, K=1, no-prior, and no-schema-store.

It does not run n=3, n=10, a gate, a full matrix, M2, headline, or graduation.

## Artifact

```text
reports/phase5_prime_cue_conditioned_bridge_smoke_seed17.json
SHA-256: 6b8c4b433643ed15c9880e9d25d745ea805db65d9bc7844aca400bef9d37f3ee
size: 116K
```

Readout:

```text
readout_id = cue_conditioned_scene_energy_v1
E_bridge(q_scene_final,cue) = -Re(<q_scene_final,cue>)/D
condition energy summary = minimum branch bridge energy within the fixed condition
Delta E = E_content-prior - E_role-prior
```

## Result

All smoke pass criteria are true:

```text
planned cells executed = 9/9
n_probe_cues = 4
device = cpu
all cells returned branches = true
all cue-conditioned bridge energies finite = true
bridge readout not all -1 = true
main/K=1/no-prior/no-schema-store Delta E computable = true
random-schema cell executed = true
no n=3/n=10/full-matrix/headline/graduation claim = true
```

Aggregate energy range:

```text
cue-conditioned bridge energy min = -0.25935444235801697
cue-conditioned bridge energy max = -0.23247389495372772
raw scene energy min = -1.0
raw scene energy max = -1.0
total branches = 120
```

Paired `Delta E` remains zero:

```text
main:            [0.0,0.0,0.0,0.0], mean=0.0
K=1:             [0.0,0.0,0.0,0.0], mean=0.0
no-prior:        [0.0,0.0,0.0,0.0], mean=0.0
no-schema-store: [0.0,0.0,0.0,0.0], mean=0.0
```

Per-probe minimum-energy top scene indices are identical across all nine cells:

```text
[0,1,1,1]
```

This includes content-prior, role-prior, no-prior, no-schema-store, and
random-schema cells.

## Interpretation

`cue_conditioned_scene_energy_v1` fixes the Report 104 saturation failure: the
bridge-energy range is no longer pinned at `-1.0`. However, this smoke is still
not positive bridge evidence. Under the current condition-energy summary
(`min` branch energy within a condition), every condition finds the same
cue-compatible scene per probe, so paired `Delta E` remains exactly zero.

The active blocker has moved from "raw energy saturates" to "condition summaries
collapse to the same final scene under this pilot query subset / branch
selection surface." Do not widen until that is diagnosed.

## Boundary

This report does not authorize n=3, n=10, a candidate/control gate, full matrix,
M2, a Phase 5 headline run, or graduation.

The next allowed work is a bounded condition-collapse analysis over this same
artifact/scope: inspect per-branch scene ranks, cue-conditioned energy ranks,
and whether the current `min` branch summary makes controls indistinguishable by
letting every condition select the same cue-compatible scene.

## Anti-Homunculus Check

Pass. The smoke uses a fixed readout and fixed cells. It logs passive
diagnostics only and does not use observed performance to route execution or
select a condition.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_cue_conditioned_bridge_smoke.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_cue_conditioned_bridge_smoke -v
```

```text
Ran 1 test in 0.382s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 394 tests in 1.290s
OK
```

```bash
git diff --check
```

```text
passed
```
