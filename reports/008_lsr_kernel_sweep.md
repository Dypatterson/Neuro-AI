# Report 008 — LSR kernel β-sweep at D=4096 (first probe)

**Date:** 2026-05-13
**Experiment:** `experiments/25_lsr_kernel_sweep.py`
**Raw data:** `reports/lsr_kernel_sweep_d4096_n32.json`

## Motivation

The 2026-05-13 brainstorm ([brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md](../brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md), Idea 2) proposed that swapping the softmax separation function in `TorchHopfieldMemory` for log-sum-ReLU (Epanechnikov, arXiv:2506.10801) would expose an intermediate-β regime where stored patterns remain stable fixed points *and* centroid blend attractors coexist — dissolving the "blend vs. retrieve" disjoint without a controller.

## Setup

- D = 4096, 32 random FHRR patterns, CPU substrate
- 32 clean cues (single pattern + 0.05 phase noise) — measures sharp retrieval
- 32 ambiguous cues (centroid of 2 random patterns) — measures blend attractor
- β sweep: 25 log-spaced values from 1e-4 to 1e2
- Headline joint score: J(β) = clean_top1 · ambiguous_blend_fraction
- Drill-downs: entropy, similarity-to-true-centroid vs. similarity-to-nearest-pattern, settling iterations

## Result

**Softmax kernel:** narrow joint-score peak `J ≈ 0.84` at β ≈ 3.16. Below this band, all cues blend to a global mean (clean_top1 = 0.03). Above it, all cues commit to the nearest stored pattern (ambiguous_blend_fraction → 0 in the immediate post-peak zone, then partial recovery around β ≥ 18 where the cue happens to be the centroid of its two source patterns and therefore *is* an attractor basin).

**LSR kernel under normalized updates: completely β-invariant.** Every β in the sweep produces `clean_top1 = 1.000, blend = 0.438, sim_centroid = 0.789, sim_nearest = 0.819` — identical to four decimal places. Mean iterations are always at max_iter (8), suggesting no settling.

## Why LSR is β-invariant in this form

The retrieval update used throughout the codebase is the normalized convex combination

```
ξ_{t+1} = normalize( Σ_i w_i x_i ),   w_i = κ(β · s_i) / Σ_j κ(β · s_j)
```

For κ = softmax, the LogSumExp denominator gives β a genuine role: weights re-distribute across patterns as β scales. For κ = ReLU, weights become

```
w_i = relu(β s_i) / Σ_j relu(β s_j) = relu(s_i) / Σ_j relu(s_j)   ∀ β > 0
```

since β > 0 does not change which scores are positive, and the common β factor cancels in the ratio. Energy magnitude scales with β (the conjugate `-β/2 · Σ max(0,s_i)²` does), but the **dynamics are β-independent**. There is no phase diagram to map.

## Implication

The Epanechnikov-DAM paper's coexistence regime almost certainly assumes the **unnormalized gradient-flow update** ξ_{t+1} ← ξ_t − η · ∇E, where the ReLU sum *is* β-scaled. That form is not interchangeable with our normalized convex combination — it changes the substrate's manifold guarantee (no automatic renormalization onto the FHRR phase manifold).

## Next steps (not done in this session)

1. **Don't yet adopt LSR architecturally.** The unit tests in `tests/test_torch_hopfield_lsr.py` and the kernel option in `TorchHopfieldMemory` are correct as a sidebar, but the brainstorm's central claim about the intermediate regime is not reproducible under this update rule.
2. If we want to test the coexistence claim properly, implement an **unnormalized LSR variant** as a separate retrieval mode (`update="grad"` vs. `update="convex"`) and re-sweep. This is a more substantial change and should be its own decision after we confirm the upstream claim is interesting at our scale.
3. The softmax-kernel J(β) result is the most defensible artifact from this run: the joint metric did peak in a narrow band around β ≈ 3, which matches the project's existing crossover-zone observations and confirms the *softmax* regime disjoint is real and tunable.

## Anti-homunculus check

The kernel parameter is set once at construction and never read by any controller; the LSR weight formula and energy are pure functions of (state, patterns, β). Passes.

## 2026-05-13 follow-up decision

Considered three paths:

- **A.** Drop LSR entirely.
- **B.** Add `update="grad"` as a second retrieval mode alongside the current normalized convex combination, since the brainstorm's regime-coexistence claim requires the unnormalized gradient-flow update where β does not cancel.
- **C.** Defer. Keep the `kernel` parameter and tests as a planted flag; route experimental effort to higher-leverage items.

**Chose C.** Two prerequisite checks have to come back negative before option B earns its place:

1. Does the permutation-slot temporal upgrade ([report 009](009_permutation_slots_ablation.md)) close enough of the Phase 3 structural gap that the regime question stops being load-bearing?
2. Does the GIB synergy estimator (`energy_memory.diagnostics.synergy`) detect an actual regime-disjoint when run on real Phase 4 consolidated bindings? If `mean_synergy` is stably high on settled states, the project isn't actually suffering the generalization/memorization disjoint that LSR coexistence was meant to dissolve.

The kernel parameter and `tests/test_torch_hopfield_lsr.py` are kept (~20 lines) so the empirical β-invariance finding stays documented and a future reversal is cheap.
