# Report 099: Phase 5' Bundle-First Scene-Memory Extraction

**Date:** 2026-05-26
**Scope:** reusable bundle-first scene-memory module extraction
**Status:** implementation complete; local parity passed

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This extraction
does not measure that headline and does not change it.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This extraction
does not execute that graduation-control matrix.

**Last verified result:** Report 098 precommitted a behavior-preserving
extraction target after the Report 097 natural-source mini-matrix gate.

**Why this implementation now:** Report 098 made module extraction, not another
gate, the next allowed step. The repeated scene-memory mechanics now live under
`src/energy_memory/phase5/` before any further experiment.

## Implementation

Added:

```text
src/energy_memory/phase5/bundle_first_scene_memory.py
tests/test_phase5_bundle_first_scene_memory.py
```

The module owns:

- `BundleFirstConfig` and `BundleFirstResult`;
- native role-vector construction;
- scene bundle construction with fixed scene-token weight;
- query-side observed-context token construction;
- seed-level execution for candidate, random-role, deranged-role,
  shuffled-role, fixed-point-free shuffled-role, content-cleanup-positive,
  bundle-positive, perfect-cue, and no-scene-token-as-candidate conditions;
- aggregation for top1, Wilson CI, leave-one-seed-out top1, `scene_tix`,
  `content_tix`, entropy, and margins.

Routed these scripts through the reusable module:

```text
scripts/phase5_prime_nonsynthetic_native_gate.py
scripts/phase5_prime_natural_source_control_cleanup_gate.py
scripts/phase5_prime_natural_source_mini_matrix_gate.py
```

Report scripts still own artifact loading, SHA/path validation, protocol
selection, CLI arguments, and report-specific payload writing.

## Parity Results

Report 097 first-cell CPU smoke remains stable:

```text
candidate|protocol=non_special_unique_target_freq_le_32|D=4096|K=16|N=512|noise=0.0|scene_token=1|token_weight=0.25|token_source=trajectory_native_provenance_context_trace|context_roles=4|cooc=repo_sample_natural
top1=0.9551 CI=[0.9335,0.9699] scene_tix=489/512 content_tix=489/512
```

Report 093/095 cleaned-gate parity remains byte-identical:

```text
54d95700df01a43bb3c5d28a5f72f3f80cf49f2dd9f848ba860da90bebea98b9  reports/phase5_prime_natural_source_control_cleanup_gate.json
54d95700df01a43bb3c5d28a5f72f3f80cf49f2dd9f848ba860da90bebea98b9  /private/tmp/phase5_prime_natural_source_control_cleanup_gate_after_extract.json
```

## Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  src/energy_memory/phase5/bundle_first_scene_memory.py \
  scripts/phase5_prime_nonsynthetic_native_gate.py \
  scripts/phase5_prime_natural_source_control_cleanup_gate.py \
  scripts/phase5_prime_natural_source_mini_matrix_gate.py \
  src/energy_memory/phase5/__init__.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_bundle_first_scene_memory \
  tests.test_phase5_prime_nonsynthetic_native_gate \
  tests.test_phase5_prime_natural_source_control_cleanup_gate \
  tests.test_phase5_prime_natural_source_mini_matrix_preflight -v
```

```text
Ran 11 tests in 0.198s
OK
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 384 tests in 1.252s
OK
```

```bash
git diff --check
```

```text
passed
```

## Boundary

This is behavior-preserving code extraction and local parity only. It does not
authorize or claim:

- a new candidate/control gate;
- M1 escalation;
- M2 implementation or run;
- a full matrix;
- a Phase 5 `Delta E` headline run;
- Phase 5 graduation;
- a new top1-based graduation metric.

## Anti-Homunculus Check

Pass. The reusable module executes fixed local dynamics:

- scene-MHN settles by energy;
- role unbinding uses the fixed query/control role;
- content cleanup settles by energy;
- role-negative controls use fixed role maps;
- diagnostics are passive readouts.

No adaptive source selection, metric-triggered routing, sampler switching, or
best-of-N condition selection was introduced.

## Next Step

The next useful step is a small bridge preflight that asks whether fixed
bundle-first scene states can be wrapped in the existing Phase 5
content-prior-vs-role-prior energy comparison without changing the current
`Delta E` headline. If the bridge requires changing the headline, stop for an
explicit spec decision before implementing.
