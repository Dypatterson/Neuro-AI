# Report 098: Phase 5' Bundle-First Integration Precommit

**Date:** 2026-05-26
**Scope:** precommit for reusable bundle-first scene-memory extraction
**Status:** implementation target locked; no new gate run

## Preamble

**Active phase:** Phase 5' precommit.

**Headline metric per `notes/emergent-codebook/phase-5-unified-design.md:282-297`:**
mean `Delta E = E_content-prior - E_role-prior` with 95% CI. This precommit
does not measure that headline and does not change it.

**Required controls per `notes/emergent-codebook/phase-5-unified-design.md:309-316`:**
random-schema branches, K=1, no-prior, and no-schema-store. This precommit does
not execute that graduation-control matrix.

**Last verified result:** Report 097 ran the fixed Report 096 natural-source
mini-matrix gate on Colab L4. Candidate stayed high across cue noise
`{0.0,0.05,0.10,0.15}` (`0.9518`, `0.9527`, `0.9529`, `0.9512`), role-negative
controls stayed clean, positives solved, and the no-scene-token baseline stayed
lower. Artifact SHA-256:
`c6220f7ab8e4011190a7798fca3d4afa9e8d977abb83786372d23b812a2a3eb5`.

**Why this experiment now:** Reports 096-097 completed the fixed broader
natural-source mini-matrix. The next safe step is not another gate; it is a
behavior-preserving integration precommit that moves the repeated bundle-first
scene-memory mechanics into reusable Phase 5' code before any further
experiment.

## Decision

Proceed with reusable module extraction next.

The next implementation should add a module under `src/energy_memory/phase5/`
named:

```text
bundle_first_scene_memory.py
```

The module should own the fixed bundle-first scene-memory mechanics currently
duplicated across report scripts:

- native role-vector construction from `build_position_vectors()`;
- scene bundle construction from fixed source rows and a fixed
  `scene_token_weight`;
- query-side observed-context token construction from the selected query plan;
- candidate, random-role, deranged-role, fixed-point-free shuffled-role,
  content-cleanup-positive, bundle-positive, perfect-cue, and no-scene-token
  seed execution;
- aggregation helpers for top1, Wilson CI, leave-one-seed-out top1,
  `scene_tix`, `content_tix`, entropy, and margins.

The report scripts should remain responsible for loading JSON artifacts,
validating SHA/path manifests, choosing the already precommitted protocol, and
writing report-specific payloads. The reusable module must not know report
numbers, default report paths, or which experiment is "next."

## Interface Contract

The reusable module should expose a small data/API surface:

- `BundleFirstConfig`: immutable config containing `D`, `N`, `K_roles`,
  `C_codebook`, `context_roles`, `n_queries`, `beta`, `max_iter`,
  `scene_token_weight`, `cooccurrence`, and `source_name`.
- `BundleFirstResult`: immutable per-seed result matching the current
  `GateResult` fields used by Reports 093 and 097.
- `build_native_roles(fhrr, k_roles)`.
- `build_scene_matrix(fhrr, roles, content, rows, scene_token_weight=...)`.
- `build_query_context_tokens(fhrr, roles, content, rows, query_plan)`.
- `run_bundle_first_seed_condition(condition=..., seed=..., source=...,
  source_path=..., source_sha=..., config=..., cue_noise=..., device=...)`.
- `aggregate_bundle_first_results(results)`.

The implementation should be a no-behavior-change extraction. It should reuse
the existing `TorchFHRR`, `experiments.44_phase5_prime_bundle_first` perturb and
batched-Hopfield helpers, and `natural_source_protocol.py` controls rather than
introducing new sampling, routing, or scoring rules.

## Required Parity Targets

The extraction is acceptable only if these behavior-preserving checks pass:

1. `phase5_prime_natural_source_control_cleanup_gate.py` can route through the
   reusable module and still reproduce the Report 093/095 cleaned gate behavior
   for the existing fixed conditions.
2. `phase5_prime_natural_source_mini_matrix_gate.py` can route through the
   reusable module and still run the Report 096 planned-cell surface.
3. A local CPU smoke for the first Report 097 candidate cell remains stable:

```text
condition=candidate
seed=17
cue_noise=0.0
top1=0.9551
scene_tix=489/512
content_tix=489/512
```

4. The module-level tests cover the no-scene-token baseline as a candidate run
   with `scene_token_weight=0.0`, not as a separate mechanism.

## Explicit Non-Actions

This precommit does not authorize:

- a new candidate/control gate;
- a full matrix;
- an M1 escalation;
- an M2 implementation or run;
- a Phase 5 `Delta E` headline run;
- changing the Phase 5 headline from `Delta E`;
- committing a best-of-N interpretation from Report 097.

The raw Report 097 Colab JSON remains a hash-anchored runtime artifact, not a
repo-committed dependency. Future extraction tests should use local deterministic
smokes and, only after local parity passes, a separate Colab parity run if the
full 32-cell CUDA artifact needs to be regenerated.

## Delta E Bridge Boundary

The bridge from bundle-first top1 diagnostics to the Phase 5 `Delta E` headline
is a later design task. The reusable module may expose scene/content states and
margins needed for future analysis, but it must not define a new graduation
metric.

The next bridge preflight, if needed, should answer only:

```text
Can fixed bundle-first scene states be wrapped in the existing Phase 5
content-prior vs role-prior energy comparison without changing the headline?
```

If the answer requires changing the headline, that must be an explicit
user-approved spec amendment, not an implementation side effect.

## Anti-Homunculus Check

Pass, with constraints.

The intended extraction is fixed local dynamics and fixed algebra:

- scene-MHN settles by energy;
- role unbinding is specified by the fixed query role;
- content cleanup settles by energy;
- role-negative controls use fixed role maps;
- diagnostics are passive readouts.

No metric-triggered control flow, adaptive source selection, sampler switching,
or condition selection may be introduced in the reusable module.

## Verification for This Precommit

This report is documentation/precommit only. Verification for this session is:

```bash
git diff --check
```

No tests or experiments are required because no code path changes in this
precommit.

## Next Step

Implement `src/energy_memory/phase5/bundle_first_scene_memory.py` plus focused
unit tests, then route the existing cleaned gate and mini-matrix gate through
the reusable module with local parity checks. Do not run a new Colab gate until
the local module extraction parity checks pass.
