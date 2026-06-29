# Research Brief B — Modern Hopfield Capacity, SNR Scaling, and Effective Interaction Order

**Date:** 2026-05-20
**Angle:** Does moving from n=1 (linear/dot-product) to higher-order interactions (n≥2, polynomial, exponential, sparsemax) attack the role-fidelity uniformity pathology at D=4096, or is it orthogonal? Honest answer up front: **mostly orthogonal**, with one specific exception worth prototyping.

---

## The pathology, restated against the literature

The substrate's role-fidelity diagnostic `f_i = mean(1 - |G_jk|)` aggregates pairwise crosstalk between unbinds. In FHRR at D=4096, **pairwise** unbind crosstalk is governed by the central-limit lawful structure of phasor inner products: every off-diagonal `|G_jk|` is an i.i.d. draw from a distribution with mean ≈ 1/√D = 0.0156 and small variance. The mean of any reasonably sized set of such draws is itself nearly constant. Hence `f_i` ≈ 1 − 1/√D ≈ 0.984 for every atom. **The metric is measuring the dimension, not the atoms.**

This is a *measurement-side* pathology before it is a *substrate-side* one. The substrate is faithfully delivering O(1/√D) crosstalk, exactly as FHRR theory predicts (Plate; Schlegel et al. 2021).

---

## Key findings (with URLs)

### 1. Krotov-Hopfield capacity formulas (foundation)

Krotov & Hopfield (2016, "Dense Associative Memory") generalize the energy from quadratic to polynomial:

- **Polynomial interaction** F(x) = x^n  →  K_max ∝ N^(n−1) (for n ≥ 2, superlinear capacity).
- **Exponential interaction** F(x) = exp(x) (Demircigil et al. 2017; this is what Ramsauer 2020 instantiates with softmax) → K_max ∝ exp(α N), i.e. exponential in pattern dimensionality.
- Error/SNR: at fixed load α = K/N^(n−1), pattern overlap dominates noise; the noise floor recedes as N^(n−1) grows.

Sources:
- [Dense Associative Memory for Pattern Recognition (NIPS 2016)](https://pdfs.semanticscholar.org/ed33/2c92664cd64843a7ba9373d992e9547230f6.pdf)
- [On a Model of Associative Memory with Huge Storage Capacity (1702.01929)](https://ar5iv.labs.arxiv.org/html/1702.01929)
- [A Biologically Plausible Dense Associative Memory with Exponential Capacity (2601.00984)](https://arxiv.org/html/2601.00984)

### 2. Ramsauer "Hopfield Networks Is All You Need" (the softmax instantiation)

- Energy: E(ξ) = −lse(β, X^T ξ) + ½ ξ^T ξ + β^(−1) log N + ½ M²
- Update: ξ_new = X · softmax(β X^T ξ)  (this is *exactly* the project's iterative settling kernel)
- **β is the lever:** high β → sharp, single-pattern attractor; low β → metastable mixtures.
- Effective inverse temperature: β_eff = β · ‖x‖² · (1 − cos θ) where θ is separation angle. Below a critical β_eff, the landscape has only one global attractor.
- Three fixed-point classes: (i) global average, (ii) metastable subset average, (iii) single stored pattern.

Sources:
- [Hopfield Networks Is All You Need (2008.02217)](https://arxiv.org/abs/2008.02217)
- [Hopfield-layers blog (ml-jku.github.io)](https://ml-jku.github.io/hopfield-layers/)

### 3. Hopfield-Fenchel-Young + structured/sparse Hopfield

A generalization beyond softmax: pick the separation transform from the Fenchel-Young family (softmax, sparsemax, α-entmax, γ-normmax, SparseMAP). Crucial point for this project:

- **Sparse variants give EXACT retrieval, not just exponentially-small error.** Margin property guarantees one-step convergence to the stored pattern when separation Δ ≥ m/β.
- **SparseMAP returns pattern associations rather than single patterns** — directly relevant to compositional/role retrieval.
- Storage capacity remains exponential (Ω((2/√3)^D)).

Sources:
- [Hopfield-Fenchel-Young Networks (2411.08590v4)](https://arxiv.org/html/2411.08590v4)
- [Sparse and Structured Hopfield Networks (2402.13725)](https://arxiv.org/html/2402.13725)

### 4. Universal Hopfield Networks (Millidge et al. 2022)

Unifies all single-shot models as: z = P · sep(sim(M, q)).
- `sim` and `sep` are independent design choices.
- Polynomial separation (degree 10) is competitive with softmax; max is unboundedly capacious in principle.
- Empirically Euclidean/Manhattan `sim` beats dot-product on several tasks.

Source:
- [Universal Hopfield Networks (2202.04557 / PMC7614148)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/)

### 5. Energy Transformer (Hoover et al., NeurIPS 2023)

A single recurrent block where attention and Hopfield memory are **co-minimizers of one energy**. Iteration = gradient flow on energy; convergence to fixed points instead of stacked feedforward.

Source:
- [Energy Transformer (2302.07253)](https://arxiv.org/pdf/2302.07253)
- [NeurIPS 2023 page](https://proceedings.neurips.cc/paper_files/paper/2023/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)

### 6. Recent survey: "Modern Methods in Associative Memory" (2507.06211)

A 2025 systematic survey of kernel choices, dynamics, and capacity tradeoffs. Worth a careful read in a follow-up.

Source:
- [Modern Methods in Associative Memory (2507.06211)](https://arxiv.org/pdf/2507.06211)

### 7. Resonator networks (Frady, Kent, Olshausen, Sommer) — direct VSA factorization

Resonator networks iteratively factor a composite FHRR vector into its constituent role-filler bindings. They interleave VSA multiplication with cleanup. The factorization problem they solve **is** the problem of inverting compositional binding cleanly — i.e., it is the problem that "role fidelity" tries to measure indirectly.

Sources:
- [Resonator Networks 1 (RCTN PDF)](https://rctn.org/bruno/papers/resonator1.pdf)
- [Recent Advances in Resonator Networks (OpenReview)](https://openreview.net/pdf?id=FNrZd3Ls1d)

### 8. Schlegel et al. comparison of VSAs

FHRR matches matrix-binding architectures (VTB, MBAT) in unbind accuracy at lower cost. HRR uses approximate (involutive) unbind for noise robustness over exact inverse. The fundamental crosstalk floor at high D is architecture-invariant within VSAs sharing the same i.i.d. element distribution.

Source:
- [A Comparison of Vector Symbolic Architectures (2001.11797)](https://arxiv.org/pdf/2001.11797)

---

## Does higher-order interaction fix the role-fidelity uniformity problem?

**Short answer: no, not directly. The pathology is a property of the diagnostic interacting with FHRR's `1/√D` law, and higher-order Hopfield kernels operate on *retrieval*, not on *unbind crosstalk*.** But there is one specific path where they help.

### Why "raise n" is *not* the fix
- `|G_jk|` is computed from the unbind operation in FHRR, which is the conjugate Hadamard product. That operation produces O(1/√D) crosstalk by construction, independent of how downstream retrieval/settling is done.
- Switching the settling kernel from softmax (n=∞) to polynomial (n=2,3,...) changes the *capacity* of the cleanup memory and the *shape* of its attractors, but doesn't change the magnitude of pairwise unbind correlations.
- The current substrate uses softmax-iterative settling, which is already the highest-order (exponential) interaction in the family. Going *down* to polynomial would *decrease* capacity, not increase it.

### Where higher-order interactions DO help (the one exception)
The Hopfield-Fenchel-Young / sparsemax / SparseMAP line offers a qualitatively different behavior: **exact retrieval with finite margin** rather than exponentially-small error.

- Softmax always returns a *convex combination* of all stored patterns with weights softmax(β · sim). Even at very high β, weights are never identically zero; the readout vector has a residual mix of off-target atoms with magnitude O(N · exp(−β Δ)). This produces measurable but small `|G_jk|` floors.
- Sparsemax (and SparseMAP) **set non-supporting weights to exactly zero** when the margin is met. A readout from sparsemax cleanup contains only the support set, so off-support `|G_jk|` reads can be measured against a true-zero baseline rather than a soft-floor baseline.

**This doesn't fix the FHRR-side `1/√D` law for raw unbinds, but it would give a *retrieval-side* variant of role-fidelity (e.g., "is the retrieved atom in the support?") that has real per-atom variance, because support membership is a discrete event, not a Gaussian sum.**

### What this means for Phase 5 specifically
The role-fidelity diagnostic, as currently defined, will never have variance at D=4096 because it is integrating over O(N²) i.i.d. small phases. Two routes:

1. **Re-define the diagnostic against retrieval-side state.** Measure `f_i` from the cleanup readout (post-Hopfield-settling) rather than from raw unbinds. This is sensitive to attractor structure and will have per-atom variance.
2. **Use a sparse-margin readout (sparsemax/SparseMAP) so the diagnostic is discrete-support-based**, which gives natural per-atom variance and a clear failure mode (atom drops out of support).

Both are anti-homunculus-clean because they measure local geometric facts of the energy landscape, not arbitrate over them.

---

## Promising leads

1. **Hopfield-Fenchel-Young / sparsemax cleanup** as a drop-in for softmax settling, where the cleanup result has discrete support. This is a *substrate change*, not a diagnostic change.
2. **Energy Transformer**: a candidate replacement for the iterative-softmax settling loop, with attention + Hopfield as joint energy minimizers. Promising for Phase 6+ but probably oversized for Phase 5 graduation.
3. **Resonator networks** for compositional factorization: directly addresses the problem the role-fidelity metric is trying to measure (can we invert a binding cleanly?). The resonator's per-factor convergence trace is a natural per-atom-variance diagnostic.
4. **Universal Hopfield decomposition** as an audit tool: re-express the substrate's settling as (sim, sep, projection) and inspect which component is responsible for the uniformity floor. Likely it's the dot-product `sim`; swapping for Euclidean might help, but again does not change FHRR unbind crosstalk.
5. **`Modern Methods in Associative Memory` (2025 survey)** — recommended deep read; it likely contains a kernel-vs-fidelity table directly relevant.

---

## Concrete ideas (each with anti-homunculus check)

### Idea B1: Sparsemax / α-entmax settling instead of softmax
- **What:** Replace `softmax(β X^T ξ)` in the Hopfield update with `sparsemax(β X^T ξ)` (or α-entmax with α ∈ (1, 2]). Implementations available in PyTorch via `entmax` package.
- **Predicted effect on role-fidelity uniformity:** Post-settling readouts now have discrete support. A new diagnostic `f_i^{readout} = 1 − (off-support mass)` will have per-atom variance because some atoms will be perfectly recovered (off-support mass = 0) and others will land in metastable mixtures (off-support mass > 0).
- **Cost:** Loses the "always-differentiable" property; sparsemax has a Jacobian with discrete support. Phase-5 is not training-time, so OK.
- **Anti-homunculus flag:** CLEAN. Sparsemax is a local geometric projection onto the probability simplex; no module decides anything. The "decision" of which atoms are in support is a face-of-polytope fact, not a controller.
- **Falsification:** If post-sparsemax `f_i^{readout}` is still uniform across atoms, the pathology is in the codebook (homogeneous similarity structure), not in the settling kernel.

### Idea B2: Re-define role fidelity against the *retrieved* state, not the raw unbind
- **What:** `f_i = sim(retrieve(query_i), target_i)` where `retrieve` is the full settling output, not just the unbind. Use angular similarity in FHRR phase space, not pairwise |G_jk|.
- **Predicted effect:** Per-atom variance returns because retrieval has discrete success/failure modes (right basin vs metastable vs global average), unlike `|G_jk|` which is a CLT sum.
- **Cost:** Zero — just a metric redefinition; uses existing substrate.
- **Anti-homunculus flag:** CLEAN. Pure measurement of a dynamical fixed point.
- **Falsification:** If even the post-retrieval similarity is uniform, the substrate is genuinely degenerate at this scale and needs a non-FHRR codebook.

### Idea B3: Resonator-network diagnostic
- **What:** Run a resonator network over the substrate's stored codebook with composite test bindings; record per-factor convergence time and final overlap. These per-atom traces *cannot* be uniform across atoms because resonator dynamics are heterogeneous (some factors converge, some oscillate).
- **Predicted effect:** Direct per-atom fidelity signal with natural variance.
- **Cost:** Implementation of resonator network on top of existing FHRR ops (a few hundred lines).
- **Anti-homunculus flag:** CLEAN. Resonator is iterative pattern completion through binding — no supervisor, purely local update rules.
- **Falsification:** Resonator fails to converge at all → codebook too dense / D too low for compositional retrieval. Either way, an answer.

### Idea B4: Polynomial-n settling for *low-capacity* regime
- **What:** Replace softmax with F(x) = x^n for n = 3 or 4 (polynomial Krotov form).
- **Predicted effect on role-fidelity:** Probably none, because the diagnostic is still pairwise-unbind-based. Polynomial settling has *lower* capacity than softmax, so this is a step backward unless paired with B1/B2.
- **Anti-homunculus flag:** CLEAN.
- **Verdict:** Not promising for Phase 5; mention only for completeness.

### Idea B5: Energy Transformer as the settling substrate
- **What:** Replace iterative-softmax cleanup loop with one Energy-Transformer block (joint attention + Hopfield gradient flow). Iterate to fixed point.
- **Predicted effect:** Different fixed-point structure; potentially richer per-atom signal. Significant architectural change.
- **Cost:** High — requires re-implementation and likely retraining the consolidation pipeline.
- **Anti-homunculus flag:** CLEAN by construction (energy descent).
- **Verdict:** Better for Phase 6+. Park as a future option.

---

## Surprises

1. **The Phase 5 substrate is *already* at the high end of the interaction-order family.** Softmax = exponential interaction = K_max ∝ exp(α D). Going polynomial would be a *downgrade* of cleanup capacity. The intuition "raise n" doesn't apply because we're already at n=∞.
2. **Krotov's noise-floor recession is a property of the *cleanup* memory's noise, not the *encoding* memory's crosstalk.** The user's `|G_jk|` is encoding-side. The two are different SNRs that get conflated easily.
3. **Sparsemax gives *exact* retrieval with finite margin** — this is qualitatively different from softmax's "exponentially small error" and is the closest thing in the modern Hopfield literature to a real per-atom variance source.
4. **The 2025 Hopfield-Fenchel-Young paper's SparseMAP variant *returns associations rather than single patterns*** — it is *designed* for the structured/role-filler retrieval problem the project cares about. This is the most directly load-bearing reference found.
5. **Resonator networks already solve the factorization problem** the role-fidelity metric is indirectly trying to measure. If the substrate cannot pass a resonator-network test, no diagnostic will save it; if it can, the resonator's trace *is* the per-atom diagnostic.

---

## Direct answer to the brief's question

> Would moving from n=1 to higher-order interactions actually fix the role-fidelity uniformity problem, or is it orthogonal?

**Orthogonal, with one caveat.** The uniformity pathology is *not* caused by low-order interactions in the cleanup memory — the substrate already uses softmax (highest order). It is caused by the metric integrating O(N²) i.i.d. `1/√D` phases on the *unbind* side, where higher-order Hopfield does not apply.

The caveat: switching the cleanup separation function from softmax (always-on convex combination) to **sparsemax / α-entmax / SparseMAP** (discrete-support projection) gives a *retrieval-side* diagnostic with natural per-atom variance. This is not "raise n" in the polynomial sense; it is *change the separation operator within the Fenchel-Young family*. It is the most surgical and lowest-risk substrate change suggested by the literature.

**Confidence level:** medium-high on the orthogonality claim (it follows from FHRR theory plus the fact that softmax = exponential interaction). Medium on the sparsemax recommendation (depends on whether the project's "atoms" admit clean margins at D=4096, which is an empirical question requiring a one-day prototype).

**Recommended next step:** Idea B2 first (free — just a metric redefinition that probably already gives variance), then Idea B1 (sparsemax cleanup) as the substrate-level intervention if B2 doesn't suffice.

---

## Sources

- [Krotov & Hopfield, Dense Associative Memory for Pattern Recognition](https://pdfs.semanticscholar.org/ed33/2c92664cd64843a7ba9373d992e9547230f6.pdf)
- [On a Model of Associative Memory with Huge Storage Capacity (Demircigil et al.)](https://ar5iv.labs.arxiv.org/html/1702.01929)
- [Hopfield Networks Is All You Need (Ramsauer et al. 2020)](https://arxiv.org/abs/2008.02217)
- [Hopfield-layers technical blog](https://ml-jku.github.io/hopfield-layers/)
- [Hopfield-Fenchel-Young Networks](https://arxiv.org/html/2411.08590v4)
- [Sparse and Structured Hopfield Networks](https://arxiv.org/html/2402.13725)
- [Universal Hopfield Networks (Millidge 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/)
- [Energy Transformer (Hoover et al. NeurIPS 2023)](https://arxiv.org/pdf/2302.07253)
- [Modern Methods in Associative Memory (survey, 2507.06211)](https://arxiv.org/pdf/2507.06211)
- [Dynamical Properties of Dense Associative Memory](https://arxiv.org/pdf/2506.00851)
- [Non-Linear Attention via Modern Hopfield Networks](https://arxiv.org/html/2506.11043v1)
- [Resonator Networks (Frady, Sommer et al.)](https://rctn.org/bruno/papers/resonator1.pdf)
- [Recent Advances in Resonator Networks (OpenReview)](https://openreview.net/pdf?id=FNrZd3Ls1d)
- [A Comparison of Vector Symbolic Architectures (Schlegel et al.)](https://arxiv.org/pdf/2001.11797)
- [A Biologically Plausible Dense Associative Memory with Exponential Capacity](https://arxiv.org/html/2601.00984)
