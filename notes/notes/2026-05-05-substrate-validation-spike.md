---
date: 2026-05-05
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Substrate Validation Spike — FHRR at D=4096

Pre-Phase-1 empirical spike to validate that FHRR via `torchhd` behaves as the architecture assumes. Run by Claude Code in a throwaway directory (`~/projects/personal-ai-spike/`) on May 5, 2026. Results captured here as the empirical baseline for Phase 1 onwards.

## Setup

| Field | Value |
|---|---|
| Hardware | Apple M5 Pro, 24 GB unified memory |
| PyTorch | 2.11.0 |
| torchhd | 5.8.4 (PyPI: `torch-hd`) |
| Substrate | FHRR, D=4096 complex (8192 real DOF) |

## Bottom line

FHRR behaves exactly as the architecture assumes. All five experiments produced results within rounding error of theoretical predictions. One significant library gotcha discovered (torchhd.bundle is incorrect for FHRR). MPS works and is meaningfully faster for matrix operations. Phase 1 cleared to proceed.

## Empirical baselines

### Pairwise orthogonality (Experiment 1)

1000 random unit-modulus FHRR vectors. 499,500 pairwise cosine similarities computed.

| Metric | Measured | Theoretical |
|---|---|---|
| Mean cosine sim | +0.000020 | 0.0 |
| Std | 0.011053 | 1/√8192 = 0.011049 |
| Mean \|sim\| | 0.008822 | √(2/π) × 0.011049 = 0.008815 |
| Min / Max | −0.050 / +0.051 | ±4.5σ plausible tail |
| \|sim\| < 3σ | 99.7% | 99.7% (Gaussian) |

Textbook match. Random FHRR vectors are quasi-orthogonal at D=4096.

### Round-trip fidelity (Experiment 2)

1000 random pairs (a, b). Compute `bound = bind(a, b)`, then `recovered = unbind(bound, b)`. Measure cosine similarity to original `a`.

| Metric | Measured |
|---|---|
| Mean fidelity | 1.0000000000 |
| Std | 0.0 |
| Min (worst case) | 1.0000000000 |
| % > 0.9999 | 100% |

Algebraically exact at float32. Stronger than expected — there is literally no numerical error at this scale, not just "close to 1.0." This means the error-driven pathway in Phase 3 operates against a perfectly clean baseline; the error signal in failed retrievals is genuinely informative, not contaminated by substrate noise.

### Bundling capacity (Experiment 3)

For each K from 2 to 50, bundled K random vectors and measured component recovery via similarity vs noise across 200 trials × 100 noise vectors.

| K | Recovery | Signal sim | Margin (vs noise floor 0.011) |
|---|---|---|---|
| 2 | 100% | 0.6367 | 58× |
| 5 | 100% | 0.4015 | 36× |
| 10 | 100% | 0.2820 | 26× |
| 20 | 100% | 0.1987 | 18× |
| 30 | 100% | 0.1621 | 15× |
| 50 | 100% | 0.1255 | 11× |

No breakdown observed up to K=50. Signal follows ~1/√K as predicted. Empirical breakdown threshold (3σ over noise) extrapolates to K ≈ 450.

**Implication for Phase 3 design:** while the substrate supports much larger bundles, signal strength scales as 1/√K. Context bags should stay small (K≤10) by design — the architectural decision is "what's the right context window," not "what's the maximum the substrate handles." Larger context bags have weaker per-atom signal even if they're recoverable.

### MPS backend (Experiment 4)

Complex tensor operations on Apple Silicon MPS:

| Operation | MPS | CPU | Speedup |
|---|---|---|---|
| `bind` [1000×4096] | 394.8 µs | 627.2 µs | 1.59× |
| `cosine_matrix` [500×500×4096] | 2.7 ms | 9.7 ms | 3.60× |
| `bundle` [K=50, D=4096] | 26.5 µs | 112.7 µs | 4.25× |

All operations work natively on MPS — no fallback to CPU. cosine_matrix (the bottleneck for Hopfield retrieval) is 3.6× faster. For Phase 1 timing budgets, expect larger speedups (5-10×) at the batch sizes relevant for Hopfield settling over thousands of stored patterns.

### Structured graph encoding (Experiment 5)

5-node directed graph with labeled edges, encoded as bundled bindings using triple form `bind(bind(src, role), dst)`. Query: `unbind(graph, bind(src, role)) ≈ dst`.

All 5 queries returned correct destinations. Signal ≈ 0.40 vs theoretical 1/√5 = 0.447. SNR ≈ 36×. The graph correctly handles the ambiguous case where node A has two outgoing edges with different roles — querying by role retrieves the correct destination.

## Surprises and gotchas

### torchhd.bundle does NOT normalize to unit modulus

The most architecturally significant finding. `torchhd.bundle(a, b)` returns an element-wise sum without renormalizing each dimension to unit modulus. The output has mean |z| ≈ 1.29 for K=2 vectors (expected 1.0).

FHRR algebra requires unit-modulus vectors throughout. If torchhd's default bundle is used in downstream bindings or Hopfield patterns, vectors accumulate magnitude and drift away from FHRR's algebraic constraints. After enough bundlings, similarity calculations become meaningless.

**Fix:** per-dimension normalization. The substrate's `bundle` method must compute `z_j = sum_k(v_{k,j}) / |sum_k(v_{k,j})|`. The spike implements this in a wrapper called `fhrr_bundle()`. This must be part of any codebook bundle operation in the real implementation.

The failure mode this would have caused if undetected: subtle, slow, hard to diagnose. Phase 3 would have produced mysteriously degrading results with no clear indication that the substrate was at fault. Catching this in the spike is exactly why spikes exist.

### torchhd is published as `torch-hd` on PyPI

Packaging mismatch: `pip install torchhd` / `uv add torchhd` fails. The correct PyPI name is `torch-hd` (with hyphen). Import name remains `torchhd`. Document in `pyproject.toml`.

### Round-trip fidelity is algebraically exact, not "close to 1.0"

Expected fidelity near 1.0 with some floating-point error. Actual: mean = 1.0, std = 0.0, min = 1.0 for all 1000 pairs. FHRR binding/unbinding is algebraically closed at float32 — the conjugate truly cancels without accumulation. Better than budgeted; means the Hebbian and error-driven pathways both operate against a clean substrate.

## Carry-forwards for Phase 1

- Build `bundle` with per-dimension normalization as a first-class function behind the substrate interface. Do not use `torchhd.bundle` directly anywhere in the real project.
- Use `torch-hd` (not `torchhd`) in `pyproject.toml`.
- Keep all Hopfield operations on MPS throughout.
- Context bags in Phase 3 should be K≤10 by design, not driven by substrate capacity.
- The spike's empirical numbers are the baseline for Phase 1 timing measurements. Significant deviations would signal something has changed (PyTorch version, MPS driver, dimension changes, etc.).

## What's confirmed for the architecture

1. Orthogonality at D=4096 — random codebook atoms start genuinely quasi-orthogonal. Similar tokens must earn their similarity through consolidation.
2. Binding algebra is algebraically exact. Compositional structures (position-bound tokens, triple-encoded edges, nested bindings) are recoverable without floating-point accumulation.
3. Bundling capacity is far above architectural needs.
4. MPS works for all FHRR operations with meaningful speedups.
5. Bundle-of-bindings retrieval works cleanly at scale relevant to Phase 1 experiments.

The substrate is sound. The torchhd.bundle wrapper is the one mandatory implementation detail that propagates from this spike into Phase 1.
