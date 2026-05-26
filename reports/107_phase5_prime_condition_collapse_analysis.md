# Report 107: Phase 5' Cue-Conditioned Condition-Collapse Analysis

**Date:** 2026-05-26
**Scope:** artifact-only analysis of Report 106 seed-17/four-probe smoke
**Status:** completed; collapse localized to same-target-scene minimum selection

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This analysis
does not run or replace the headline.

**Last verified result:** Report 106 showed `cue_conditioned_scene_energy_v1`
is finite and non-saturated, but all paired `Delta E` values remained zero.

**Why this analysis now:** `phase-5-prime-checklist.md` item I9 required a
bounded condition-collapse analysis before any widening.

## Implementation

Added:

```text
scripts/phase5_prime_condition_collapse_analysis.py
tests/test_phase5_prime_condition_collapse_analysis.py
reports/phase5_prime_condition_collapse_analysis.json
```

The script reads the Report 106 JSON only. It records per-cell/per-probe branch
energy ranks, top scene indices, minimum-energy scene selections, and paired
`Delta E` readback. It does not rerun retrieval, change cells, widen seeds, or
call Colab.

## Artifact

```text
reports/phase5_prime_condition_collapse_analysis.json
SHA-256: 3049865bea602cb07d8ae7a00daf8824b22f0f7e3dbd6332df8c142300b426f3
size: 75K
```

Source:

```text
reports/phase5_prime_cue_conditioned_bridge_smoke_seed17.json
SHA-256: 6b8c4b433643ed15c9880e9d25d745ea805db65d9bc7844aca400bef9d37f3ee
```

## Result

All analysis pass criteria are true:

```text
artifact_only_analysis = true
source_smoke_passed_all_criteria = true
seed17_only = true
pilot_probe_subset_only = true
planned_cell_count_is_nine = true
same_query_subset_all_cells = true
all_cells_returned_branches = true
cue_conditioned_readout_non_saturated = true
raw_scene_energy_still_saturated = true
all_paired_delta_values_zero = true
all_probe_minima_collapse_to_one_scene_set = true
all_minimum_selections_equal_probe_target_scene = true
no n=3/n=10/full-matrix/headline/graduation claim = true
```

Aggregate readback:

```text
total cells = 9
total per-cell probe summaries = 36
total branches = 120
branch target-scene hits = 118/120 = 0.9833333333333333
minimum-selection target-scene hits = 36/36 = 1.0
branch scene counts = {0: 30, 1: 88, 394: 2}
raw scene energy unique values = [-1.0]
cue-conditioned bridge energy unique values =
  [-0.259354442358017, -0.246921136975288,
   -0.237126141786575, -0.232473894953728]
paired Delta E values = sixteen zeros
```

Per-probe minimum scenes:

```text
probe 0: unique min scene set [[0]], min energy -0.246921136975288
probe 1: unique min scene set [[1]], min energy -0.259354442358017
probe 2: unique min scene set [[1]], min energy -0.237126141786575
probe 3: unique min scene set [[1]], min energy -0.237126141786575
```

The only off-target settled scene is scene `394`, appearing in two branches:
the `main_content_K4_g0.5` branches for probes 2 and 3. In both probes, the
target-scene branch has lower cue-conditioned energy, so the current
minimum-within-condition summary still selects scene `1`.

## Diagnosis

The collapse mode is:

```text
min_branch_target_scene_attractor_collapse
```

Report 106 was not blocked by bridge-readout saturation. The new readout has
four finite non-`-1` energy values. The collapse is instead a branch-set and
summary collapse: for each probe, every condition has access to the same
cue-compatible target-scene minimum. Taking the minimum branch energy inside
each condition therefore makes content-prior, role-prior, no-prior,
no-schema-store, and random-schema cells indistinguishable at the paired
`Delta E` level.

This explains why `Delta E` remains zero even though the cue-conditioned readout
itself is no longer saturated.

## Boundary

This report does not authorize n=3, n=10, a candidate/control gate, full matrix,
M2, a Phase 5 headline run, or graduation.

The next allowed work is a precommit for a stricter discriminator or fixed query
subset that cannot be solved by every condition selecting the same
target-scene minimum. That precommit must happen before any widening.

## Anti-Homunculus Check

Pass. This is passive artifact readback only. It adds no metric-triggered route
choice, condition selection, best-of-N execution rule, or retrieval rerun.

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_condition_collapse_analysis.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_condition_collapse_analysis -v
```

```text
Ran 1 test in 0.005s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 395 tests in 1.285s
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
path = Path("reports/phase5_prime_condition_collapse_analysis.json")
p = json.loads(path.read_text())
assert p["passes_all_criteria"] is True
assert p["framing"]["retrieval_executed"] is False
assert p["diagnosis"]["condition_collapse_confirmed"] is True
assert p["diagnosis"]["collapse_mode"] == "min_branch_target_scene_attractor_collapse"
assert p["aggregate"]["branch_target_scene_hits"] == 118
assert p["aggregate"]["total_branches"] == 120
assert p["aggregate"]["min_selection_target_hits"] == 36
assert p["aggregate"]["total_probes_by_cell"] == 36
assert p["aggregate"]["paired_delta_values"] == [0.0] * 16
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

```text
3049865bea602cb07d8ae7a00daf8824b22f0f7e3dbd6332df8c142300b426f3
```
