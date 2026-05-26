# Report 103: Phase 5' Raw Scene-Energy Retrieval Smoke Precommit

**Date:** 2026-05-26
**Scope:** precommit for the first pilot retrieval smoke over `raw_scene_energy_v0`
**Status:** completed; pilot retrieval smoke passed plumbing criteria

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This smoke keeps
that form but is not a headline run.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. Report 102 proved
these controls can be represented against the same fixed query plan; this smoke
executes a tiny seed-17 subset only.

**Last verified result:** Report 102, static controls preflight only. No
retrieval, top1, energy result, n=3/n=10 evidence, full matrix, headline, or
graduation claim.

**Why this experiment now:** `phase-5-prime-checklist.md` item I5 requires a
separate retrieval-smoke precommit before running even a pilot retrieval smoke.

## Fixed Smoke

The smoke is exactly:

```text
baseline_id = raw_scene_energy_v0
score_bias = None
energy_readout = raw_scene_mhn_energy
headline form = Delta E = E_content-prior - E_role-prior
seed = 17 only
max_probes = 4
device = auto unless overridden
beta = 30.0
gamma = 0.5
k_main = 4
include_surprise_branch = false
run_combiners = false
max_settling_iter = 10
```

Seed 17 is a pilot/reference seed, not representative evidence.

## Fixed Cells

Run exactly the nine cells planned in Report 102:

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

All cells must use the same fixed source/query plan from the cleaned natural
source protocol. The no-schema-store cells may change only the schema/prior
source to the content codebook; the retrieval memory remains the bundle-first
scene store and the headline form remains unchanged.

## Output Contract

The smoke writes:

```text
reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json
```

Required readback fields:

- artifact/source SHA anchors;
- planned cell count and executed cell IDs;
- per-cell/per-probe branch counts;
- finite raw and step3 energy checks;
- paired `Delta E` for main, K=1, no-prior, and no-schema-store;
- random-schema execution flag;
- same-query-subset flag;
- legacy-default mismatch flags for `beta=30` vs `10` and `k_main=4` vs `1`;
- explicit no-n3/no-n10/no-full-matrix/no-headline/no-graduation flags.

## Stop Conditions

Stop before interpreting or widening the smoke if:

1. Any required control cannot run against the same fixed query subset.
2. No-schema-store requires changing the headline metric instead of only the
   schema/prior source.
3. A scene-level Step-3 `score_bias` is reintroduced without a fixed mapping
   precommit.
4. `beta`, `gamma`, or `K` defaults drift without an explicit precommit.
5. Any result would be described as Phase 5 graduation, n=3/n=10 evidence, a
   full matrix, M1 escalation, M2, or a top1-based headline replacement.

## Anti-Homunculus Check

Pass. The smoke uses fixed scene states, fixed prior selectors, fixed raw
scene-MHN energy, and passive diagnostics. It introduces no adaptive routing,
metric-triggered bias, sampler switching, retrieval-result feedback, or
best-of-N condition selection.

## Verification Before Retrieval

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_raw_scene_retrieval_smoke.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest \
  tests.test_phase5_prime_raw_scene_retrieval_smoke -v
```

```text
Ran 1 test in 0.389s
OK
```

## Smoke Execution

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/phase5_prime_raw_scene_retrieval_smoke.py \
  --device cpu \
  --out reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json
```

```text
wrote reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json passes=True cells=9 n_probe_cues=4 device=cpu
```

## Artifact

```text
reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json
SHA-256: 5fe527a9741e3a0751718747a76a98c978159ff3ebcc2ab08906c4e2ff656715
size: 160K
```

Source/protocol anchors:

```text
source: reports/phase5_prime_nonsynthetic_native_context_source.json
source SHA-256: 2200f3c1b1599c478ce6ecc6a3cb1e783c3bfd51ceda3f335642bb9158989398
cleanup preflight: reports/phase5_prime_natural_source_control_cleanup_preflight.json
cleanup SHA-256: 3c3a5c9781c7cb2f2a052923703ad8d4d60a84fcae5a590e82a1c8f29cf40aaf
prior gate: reports/phase5_prime_nonsynthetic_native_gate.json
prior gate SHA-256: 8ada5e4c01104f79545e9da156c3b870279fdff0da214d54d969f1d85a2ef280
controls preflight: reports/phase5_prime_raw_scene_controls_preflight.json
controls preflight SHA-256: 849e0f4382794b36d0af555a507b91f0dc8e1b1f8a89b2a4088a75d3c9aedb06
protocol: non_special_unique_target_freq_le_32
```

## Result

All smoke pass criteria are true:

```text
planned cells executed = 9/9
n_probe_cues = 4
device = cpu
same query subset all cells = true
all cells returned branches = true
all branch energies finite = true
step3 energy equals raw because score_bias=None = true
random-schema cell executed = true
main Delta E computable = true
K=1 Delta E computable = true
no-prior Delta E computable = true
no-schema-store Delta E computable = true
```

Interface shapes:

```text
scene_matrix = [512,4096]
scene_schema_bindings = [512,16,4096]
content_codebook = [2048,4096]
content_schema_bindings = [2048,16,4096]
memory_stored_count = 512
```

Legacy-default mismatch flags remain recorded and accepted only for this smoke:

```text
beta: raw_scene_value=30.0, legacy_headline_default=10.0
k_main: raw_scene_value=4, legacy_headline_default=1
```

Paired probe deltas:

```text
main:            [0.0,0.0,0.0,0.0], mean=0.0
K=1:             [0.0,0.0,0.0,0.0], mean=0.0
no-prior:        [0.0,0.0,0.0,0.0], mean=0.0
no-schema-store: [0.0,0.0,0.0,0.0], mean=0.0
```

Per-cell branch counts were as expected:

```text
K=1 cells: 1 branch/probe
all K=4 cells, including random-schema and no-schema-store: 4 branches/probe
all per-probe min step3 energies: -1.0
```

## Interpretation

This is a successful retrieval-plumbing smoke only. It proves that the nine
Report 102 cells can execute through the existing `run_branched_retrieval()`
path against a fixed seed-17/four-probe subset, including the no-schema-store
content-codebook source.

It is not positive raw-scene bridge evidence. The pilot energy readout is
degenerate at this scale: every paired `Delta E` is exactly `0.0`, and every
per-probe minimum step3/raw energy readback is `-1.0`. Before widening to n=3,
n=10, a full matrix, or a headline run, analyze why `raw_scene_energy_v0`
collapses to identical minimum energies under these probes.

## Final Verification

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile \
  scripts/phase5_prime_raw_scene_retrieval_smoke.py \
  scripts/phase5_prime_raw_scene_controls_preflight.py \
  scripts/phase5_prime_delta_bridge_preflight.py
```

```text
passed
```

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

```text
Ran 389 tests in 1.284s
OK
```

```bash
git diff --check
```

```text
passed
```

## Boundary

This precommit authorizes only the seed-17/four-probe pilot retrieval smoke
above. It does not authorize or claim:

- n=3 or n=10 evidence;
- a candidate/control gate;
- a full matrix;
- M1 escalation;
- M2 implementation or run;
- a Phase 5 headline run;
- Phase 5 graduation.
