# Report 069 — Phase 5′ Design and Range-Shaped Replay Integration

**Date:** 2026-05-25
**Active phase:** 5′ precommit
**Status:** Design and integration surface landed locally. **No Phase 5
graduation claim.** This report documents code and harness readiness, not
n>=10 downstream evidence.

## Decision Boundary

M1 was not continued. [Report 064](064_phase5_m1_retrieval_smoke_cross_seed_null.md)
closed the M1 retrieval-side rescue at smoke scale: `hit_role = 0.000` across
90 cues x 3 seeds and `Delta E ~= -0.32`, strongly negative and far below the
floor in the wrong direction. Escalating M1 to n>=10 would spend compute on a
known smoke-null path.

The leading positive signal is bundle-first structural memory from
[Reports 066](066_mqar_ghrr_bundle_first_discriminator.md) and
[067](067_mqar_multirole_bundle_first.md). These remain diagnostic gates only.
[Report 068](068_range_shaped_replay_sampler_spike.md) remains sampler-only
evidence: range-shaped replay rectangularizes pairs, but its downstream effect
on Phase 4 consolidation and Phase 5 Delta E is unmeasured.

## What Changed

- Updated [STATUS.md](../STATUS.md): Phase 5 not graduated; M1 retrieval-side
  null; bundle-first is the leading Phase 5′ candidate; range-shaped replay is
  algorithm-validated but not downstream-integrated evidence.
- Added [phase-5-prime-bundle-first-design.md](../notes/emergent-codebook/phase-5-prime-bundle-first-design.md):
  architectural claim, scene-MHN identification, role unbinding, content
  cleanup, headline/diagnostics, controls, n>=10 standard, and
  anti-homunculus check.
- Added [phase-5-prime-checklist.md](../notes/emergent-codebook/phase-5-prime-checklist.md):
  MQAR reproduction, cue noise, scene-token robustness, co-occurrence skew,
  controls, range-shaped integration items, and verification standard.
- Moved the reusable sampler into
  [range_shaped_replay.py](../src/energy_memory/phase4/range_shaped_replay.py).
- Wired [replay_loop.py](../src/energy_memory/phase4/replay_loop.py) with
  static config: `standard | range_shaped`, fallback `skip | closest | rebind`,
  smoothing alpha, and single-binding vs window-preserving rebind modes.
- Updated [43_range_shaped_replay_gate.py](../experiments/43_range_shaped_replay_gate.py)
  to import the reusable sampler from `src/`.
- Added [44_phase5_prime_bundle_first.py](../experiments/44_phase5_prime_bundle_first.py)
  with K_roles/N/cue-noise sweeps, optional scene-token and co-occurrence
  conditions, controls, and CI/diagnostic output.

## Implementation Summary

`RangeShapedReplaySampler` now supports:

- `sample_pairs()` returning `(role, atom, trace_idx)` with `trace_idx=None`
  for unbacked pairs
- deterministic output with a seeded `torch.Generator`
- priority-weighted backing-trace selection
- safe ignore of traces missing `encoder_terms`
- optional atom-support smoothing
- `marginal_diagnostics()`

Rebind synthesis now has two fixed modes:

- single-binding: `bind(position_vectors[role], codebook[atom])`
- window-preserving: sample a full factored window and call
  `encode_window_with_provenance()`

Both return `TrajectoryTrace` with `encoder_terms` populated.

Phase 4 integration is static config only. The code does not switch sampler,
fallback, smoothing, or rebind mode based on observed metrics.

## Smoke Diagnostics

Small range-shaped replay smoke:

```text
experiments/43_range_shaped_replay_gate.py --seeds 17 --n_traces 80 \
  --n_roles 4 --n_atoms 32 --window_size 4 --skew_concentration 2 \
  --n_sampled 100 --device cpu --out /private/tmp/range_shaped_smoke.json
```

Result: rectangularity `1.3863 -> 0.0225`; cell coverage `8/128 -> 32/128`;
missing pairs `0.770`. This is an algorithm smoke, not downstream evidence.

Tiny bundle-first harness smoke:

```text
experiments/44_phase5_prime_bundle_first.py --Ds 128 --Ns 8 --K_roles 2 \
  --cue_noise 0.0 --seeds 17 --n_queries 8 --C_codebook 32 \
  --conditions candidate random_role shuffled_role perfect_cue \
  bundle_positive content_cleanup_positive --scene_token 0 \
  --cooccurrence uniform --device cpu --out /private/tmp/phase5_prime_smoke.json
```

Result: candidate `top1=0.750` with `scene_tix=6/8` and `content_tix=6/8`;
random-role and shuffled-role controls `top1=0.000`; perfect-cue,
bundle-positive, and content-cleanup-positive controls `top1=1.000`.
This is a wiring smoke only.

## Colab L4 Diagnostic Offload

The larger diagnostics were offloaded through Safari/Colab on an L4 GPU with
high-RAM runtime (`torch 2.10.0+cu128`, `cuda=True`, `NVIDIA L4`). These are
diagnostic/offload results only; they are not downstream Phase 4 consolidation
or Phase 5 Delta E evidence.

Range-shaped replay n=10 sampler stress:

```text
experiments/43_range_shaped_replay_gate.py --seeds 17 11 23 1 2 3 5 7 13 29 \
  --n_traces 4096 --n_roles 16 --n_atoms 1024 --window_size 16 \
  --skew_concentration 16 --n_sampled 8192 --device cuda
```

Aggregate result: rectangularity `2.7725 -> 0.0147`; cell coverage
`256/16384 -> 4096/16384`; missing pairs `0.9378`. This confirms the
sampler-level rectangularization and support expansion at a larger diagnostic
cell; it still does not measure consolidation or Delta E.

Bundle-first hard-cell controls, n=10, `D=4096`, `N=512`, `K_roles=16`,
`cue_noise=0.15`, `scene_token=1`, skewed co-occurrence:

| condition | top1 | Wilson CI | scene_tix | content_tix |
| --- | ---: | ---: | ---: | ---: |
| candidate | 0.8812 | [0.8624, 0.8978] | 0.8812 | 0.8812 |
| random_role | 0.0000 | [0.0000, 0.0030] | 0.8797 | 0.0000 |
| shuffled_role | 0.0141 | [0.0089, 0.0221] | 0.1000 | 0.0141 |
| perfect_cue | 1.0000 | [0.9970, 1.0000] | 1.0000 | 1.0000 |
| bundle_positive | 1.0000 | [0.9970, 1.0000] | 1.0000 | 1.0000 |
| content_cleanup_positive | 1.0000 | [0.9970, 1.0000] | 1.0000 | 1.0000 |

Bundle-first scene-token/co-occurrence ablation, same hard cell with candidate
condition only:

| scene_token | cooccurrence | top1 | Wilson CI | scene_tix | content_tix |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | uniform | 0.7781 | [0.7545, 0.8000] | 0.7781 | 0.7781 |
| 0 | skewed | 0.2219 | [0.2000, 0.2455] | 0.2078 | 0.2219 |
| 1 | uniform | 0.9844 | [0.9760, 0.9899] | 0.9844 | 0.9844 |
| 1 | skewed | 0.8812 | [0.8624, 0.8978] | 0.8797 | 0.8812 |

Interpretation: the hard-cell run strengthens the bundle-first diagnostic
case and shows the scene token is important under skew, but it is still a
single diagnostic slice rather than the full n>=10 verification matrix.

## Tests Run

```text
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase4_range_shaped_replay -v
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase4_replay_loop -v
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase4_replay_store_upgrades -v
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_phase4_trajectory -v
PYTHONPATH=src:. .venv/bin/python -m py_compile src/energy_memory/phase4/range_shaped_replay.py src/energy_memory/phase4/replay_loop.py experiments/43_range_shaped_replay_gate.py experiments/44_phase5_prime_bundle_first.py
PYTHONPATH=src:. .venv/bin/python -m unittest discover tests
```

Full suite result: **360 tests OK**.

Colab L4 offload artifacts written in session storage:

```text
/content/phase5_prime_results_l4/range_shape_n10_l4.json
/content/phase5_prime_results_l4/bundle_hard_cell_controls_n10_l4.json
/content/phase5_prime_results_l4/bundle_candidate_scene_cooc_n10_l4.json
```

## What Remains Unverified

- No n>=10 Phase 5′ Delta E verification run has been executed.
- No downstream Phase 4 consolidation comparison has been run with
  `range_shaped` replay enabled.
- No Phase 5 Delta E movement has been measured from range-shaped replay.
- The full `K_roles in {2,4,8,16}` x `N in {16,32,64,128,256,512}` x
  cue-noise/control matrix has not been run.
- Scene-token and skewed co-occurrence were offloaded for one hard diagnostic
  cell only; natural co-occurrence and the full grid remain unrun.
- Reports 066/067 remain diagnostic MQAR evidence, not graduation evidence.
- Report 068 remains sampler-algorithm evidence, not Phase 4-integrated
  downstream evidence.

## Anti-Homunculus Check

Pass. The additions are fixed sampling, algebraic synthesis, and passive
diagnostics:

- sampler choice is static config
- fallback mode is static config
- smoothing alpha is static config
- rebind mode is static config
- scene-MHN and content-MHN settle by energy
- diagnostics do not route execution

No controller, no metric-triggered switching, and no best-of-N graduation
claim were added.

## Bottom Line

The branch has been converted from "continue M1 role-energy rescue" into a
Phase 5′ precommit surface around bundle-first structural memory plus
range-shaped replay integration. The code and harness are ready for downstream
experiments, but Phase 5 is still ungraduated until n>=10 controls and the
appropriate headline evidence actually pass.
