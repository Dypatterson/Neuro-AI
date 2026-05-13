# Report 014 — MPS vs CPU benchmark at D=4096 (Tier C item 1)

**Date:** 2026-05-13
**Experiment:** `experiments/30_mps_benchmark_d4096.py`
**Raw data:** `reports/mps_benchmark_d4096.json`
**Hardware:** macOS, M-series GPU. torch 2.11.0, MPS available.

## Motivation

PROJECT_PLAN Phase 1 exit criterion:
> MPS speedup is meaningful for scoring and retrieval.

This report measures it explicitly across the hot-path operations at
the project's working scale (D=4096), with landscape sizes spanning
the regimes the project actually uses (64 = unit tests, 256–1024 =
Phase 2 baseline, 4096 = Phase 4 stress).

## Setup

- D = 4096
- Landscapes L ∈ {64, 256, 1024, 4096}
- 3 warmup iterations + 20 timed reps per (op, L, device)
- Device sync (`torch.mps.synchronize()`) before and after each timed
  call so MPS kernel completion is included in the measurement
- Operations: substrate primitives (random_vectors, bind, unbind,
  bundle) + the Hopfield kernels (similarity_matrix, full retrieve
  with β=10, max_iter=8)

## Headline result

| operation         | L=64    | L=256   | L=1024 | L=4096 |
| ----------------- | ------- | ------- | ------ | ------ |
| random_vectors    | 0.65×   | 0.74×   | 0.91×  | 0.92×  |
| bind              | **0.01×** | **0.01×** | **0.01×** | **0.01×** |
| unbind            | **0.02×** | **0.02×** | **0.03×** | **0.02×** |
| bundle            | 0.30×   | 0.81×   | **1.62×** | **1.98×** |
| similarity_matrix | 0.35×   | 0.43×   | **1.55×** | **1.62×** |
| retrieve          | 0.30×   | 0.81×   | **1.53×** | **1.30×** |

(Speedup = CPU_mean / MPS_mean. Greater than 1.0 means MPS is faster.)

## Three findings

### 1. MPS speedup is meaningful at L ≥ 1024 for the hot path

For the operations that dominate Phase 2/3/4 retrieval cost —
similarity_matrix and the full retrieve loop — MPS gives **1.3–2.0×
speedup** at landscape sizes ≥ 1024. This is consistent with the
PROJECT_PLAN Phase 1 exit criterion, and the L=4096 regime (where
Phase 4 sits) shows clean wins: bundle 1.98×, similarity_matrix
1.62×, retrieve 1.30×. The retrieve speedup is smaller than the
underlying similarity_matrix speedup because the settling loop does
~8 launches per call and each launch has fixed MPS overhead.

### 2. MPS is a wash or worse at small landscapes (L ≤ 256)

For unit-test and smoke-test scales (L ≤ 64), CPU wins on every
operation, often by 3–4×. The unit test default of D=128 or D=512
with handful-sized landscapes is *not* a regime where MPS helps;
the existing test design choice to use CPU is correct.

### 3. Per-vector substrate primitives have a 100× MPS overhead penalty

The `bind` and `unbind` operations are ~100× slower on MPS than on
CPU at every landscape size. The absolute timing (~200μs per call
on MPS, ~2μs on CPU) reveals the cause: this is pure MPS kernel-
launch overhead, not arithmetic. Each call hits the GPU for a
single complex multiplication of D=4096 elements — far below the
break-even point for GPU dispatch.

This is the most actionable finding in this report. Code paths that
make many sequential single-vector `bind`/`unbind` calls (e.g., the
encoding loop in `encode_window`, the permutation-slot construction
in `PermutationSlotTemporalMemory.store_sequence`) take a 100×
multiplier on MPS that they would not take on CPU. If those loops
ever become hot, they should either (a) batch their bind calls into
a single elementwise multiply over stacked vectors, or (b) move to
CPU explicitly even when the rest of the pipeline is on MPS.

### `random_vectors` underperforms because phases are generated on CPU

The slight CPU win on `random_vectors` (0.65–0.92×) traces to the
substrate generating phase tensors via
`torch.rand(..., generator=cpu_gen, device="cpu")` and then `.to(mps)`.
That host-to-device transfer dominates the actual MPS arithmetic.
Not a high priority to fix — substrate initialization is one-shot —
but worth noting if a future workflow ever calls `random_vectors` in
a tight inner loop.

## Implication for project ordering

* **No change needed to current MPS use in Phase 4.** The 4096-scale
  retrieve and similarity_matrix calls already get the MPS win.
* **The Phase 2 matrix on MPS** ([reports/phase2_full_matrix/](phase2_full_matrix/))
  ran at L ∈ {64, 256, 1024}. The L=64 and L=256 conditions were
  probably *slower on MPS than CPU* in that matrix, which is fine for
  apples-to-apples consistency but informative — future Phase 2 sweeps
  could route small-L conditions to CPU and large-L conditions to MPS
  for faster turnaround.
* **Watch the bind-loop hotspots.** Any sequential single-vector
  `bind` call on MPS pays the 100× penalty. The biggest current
  offender is probably `encode_window` (one bind per token per window)
  if it's ever called on MPS in a tight loop. Cheap fix when needed:
  batch the binds into one elementwise complex multiply over a
  `[window_size, D]` tensor.

## Recommended next steps

1. **No further benchmark work this iteration.** The headline is
   clear: MPS is worth it for L ≥ 1024 on the bulk operations.
2. **Move to Tier C item 2** — port Phase 0 sweeps to the Torch hot
   path. (Audit + targeted port already on the todo list.)
3. **If a future hotspot emerges in `encode_window` or similar,**
   come back and add a `bind_batched()` substrate method that does
   `bundle(role_matrix * filler_matrix)` in one MPS launch.

## Anti-homunculus check / FEP audit

Pure measurement primitive. No mechanism, no dynamics, nothing reads
the output to trigger a code path. Passes trivially.
