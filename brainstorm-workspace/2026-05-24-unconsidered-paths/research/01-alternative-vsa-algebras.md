# Alternative VSA / HDC Algebras — Beyond FHRR

**Date:** 2026-05-24
**Research thread:** unconsidered-paths
**Context:** Phase 5 (role/structural retrieval) is stuck. Four retrieval-mechanism families have all returned `hit_role = 0.000`. The diagnosis is "the substrate as currently shaped lacks role basins." This research investigates whether the substrate algebra choice itself (FHRR's circular-convolution-via-phase-multiplication) is the foreclosure.

---

## Angle

**Hypothesis under test.** FHRR's binding operator (Hadamard product on unit-magnitude phasors) is *commutative* and *self-inverse-by-conjugation*. Commutativity makes role-vs-filler asymmetry purely a labelling convention: there is no algebraic gradient pushing role-shaped quantities into a different geometric region than filler-shaped ones. If role basins are needed to break ties between candidate fillers, an algebra that makes role/filler *geometrically distinguishable* — by sparsity, by non-commutativity, by being an outer-product / tensor — might unlock what no retrieval-side mechanism on FHRR can.

The corollary, equally important to surface: if a substrate swap is *not* what's needed, the failure of every alternative algebra in the literature on the exact role-binding workload would be strong evidence the problem is upstream (codebook geometry, replay objective, energy mismatch), not in the algebra.

---

## Key findings

### 1. Sparse Block Codes (SBC) and Generalized SBC (GSBC) — IBM Research / Hersche / Rahimi

The most directly relevant alternative algebra. Block-structured sparse vectors with operationally tested role-filler factorization.

- **Binary SBC binding:** D-dim vector divided into B blocks of length L = D/B. Each block has exactly *one* nonzero element. Binding = block-wise modulo-L sum of one-hot offset representations. Preserves dimensionality and sparsity exactly.
- **GSBC:** relaxes to non-negative reals with unit ℓ₁-norm per block. Binding = block-wise circular convolution. Similarity is ℓ∞-based (`s∞(x,y) = 1 − ℓ∞(x − y)`), which induces *sparse activations* — most similarities are exactly 0. That's a structural sparsity that FHRR's cosine similarity simply cannot have.
- **Factorization capacity ("Block Codes Factorizer", BCF):** operational capacity >99% accuracy reaches **5×10⁶** at D=512 with B=4, F=2 factors. Binary SBC with dot-product caps at ~10³; FHRR resonator networks operate around 10⁴–10⁵.
- **Convergence:** ~11 iterations to factor a problem of size 10⁶ vs 68 for dense bipolar. Roughly 6× speedup at the same operational capacity.
- URLs:
  - https://arxiv.org/abs/2303.13957 (Hersche et al., "Factorizers for Distributed Sparse Block Codes")
  - https://arxiv.org/html/2303.13957v2 (HTML with equations)
  - https://research.ibm.com/publications/factorizers-for-distributed-sparse-block-codes
  - https://github.com/IBM/in-memory-factorizer (code)
  - https://arxiv.org/abs/2009.06734 (Frady-Sommer 2020, "Variable Binding for Sparse Distributed Representations" — the theoretical basis)

### 2. Generalized Holographic Reduced Representations (GHRR) — 2024

A direct FHRR descendant with **non-commutative** binding. This is the most surgical possible change to the current substrate.

- Extends FHRR with a flexible, non-commutative binding operation.
- Non-commutativity is essential for representing nested or ordered structures *without* having to introduce explicit permutations (which is what the current FHRR codebase has to do for position).
- Empirically shown to encode attention in transformer encoders, with a binding-based positional encoding.
- URL: https://arxiv.org/abs/2405.09689

### 3. Residue Hyperdimensional Computing (RHC) — Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer, Olshausen, 2023

- Unifies residue number systems with HDC over FHRR-style vectors.
- Represents and operates on numerical values over a large dynamic range with resources scaling **logarithmically** in the range.
- Combines naturally with resonator factorization — the factorizer becomes a residue decoder.
- *Most interesting feature for this project:* RHC effectively gives FHRR a separable algebraic structure on top of phase multiplication. Different "roles" (different moduli) become geometrically non-overlapping by construction.
- URLs:
  - https://arxiv.org/abs/2311.04872
  - https://direct.mit.edu/neco/article/37/1/1/125267 (Neural Computation 2025, journal version)
- Recent extension: **VSA Lisp with residue arithmetic** (Hanley, Tomkins-Flanagan, Kelly 2025) — proves Turing completeness of FHRR + RHC and shows arithmetic primitives over the substrate. https://arxiv.org/abs/2511.08767

### 4. Vector Function Architecture (VFA) — Frady, Kleyko, Kymn, Olshausen, Sommer 2021–2022

- Generalizes VSA to function spaces. Vectors represent functions in a reproducing kernel Hilbert space.
- Inner product of two encoded data points = a kernel similarity. Algebraic vector ops correspond to well-defined operations in function space.
- Provides an algebraic framework for large-scale kernel machines with random features.
- URL: https://arxiv.org/abs/2109.03429

### 5. Spatial Semantic Pointers (SSP) — Komer, Stewart, Voelker, Eliasmith

- FHRR-family algebra with **fractional binding** for continuous variables.
- Roles AND fillers can encode continuous quantities.
- Decoding: `M ⊛ P(x,y)^{-1}` then cleanup. Native primitive for continuous role retrieval.
- Already empirically deployed for spatial reasoning, dynamical-system prediction.
- URLs:
  - https://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf
  - https://compneuro.uwaterloo.ca/files/publications/lu.2019.pdf
  - https://direct.mit.edu/neco/article/33/8/2033/102625 ("Simulating and Predicting Dynamical Systems With SSPs", Voelker et al.)

### 6. MAP / MAP-C / MAP-I (Multiply-Add-Permute) — Gayler

- Bipolar {−1, +1} (MAP) or real [−1,1] (MAP-C) vectors. Binding = Hadamard product. **Every vector is its own multiplicative inverse.**
- Self-inverse simplifies unbinding but means binding is still commutative.
- Permutation operator is the typical companion for breaking symmetries and "quoting" information.
- The Schlegel-Neubert-Protzel comparison paper is the canonical empirical reference.
- URLs:
  - https://link.springer.com/article/10.1007/s10462-021-10110-3 (Schlegel et al., 2021)
  - https://arxiv.org/pdf/2001.11797

### 7. Tensor Product Representations (TPR) — Smolensky 1990

The "obvious" role-filler algebra. Outer product `r ⊗ f` gives a clean role-filler factorization. The project rejected it for capacity reasons.

- Modern revisits: **RNNs implicitly implement TPRs** (McCoy, Linzen, Dunbar, Smolensky, ICLR 2019) — Tensor Product Decomposition Networks (TPDNs) can recover role-filler structure from RNN hiddens. https://arxiv.org/abs/1812.08718
- Transformer studies (BERT/RoBERTa) show variable binding is *not* the primary internal mechanism; transformers rely on the input as external memory. That is direct evidence that TPR-style internal binding is *hard* to learn from data — but it's not evidence that an explicit TPR substrate underperforms.
- The dimensional explosion (D² for role × filler) is the standard rejection, but for D=4096 with a small role inventory (say 16 roles), a banked TPR is D × R = 65,536 — not catastrophic.

### 8. Sparse Distributed Memory (SDM) + K-winner MHN — NeurIPS 2023

- SDM-style address decoders combined with Modern Hopfield retrieval.
- "Sequential Learning and Retrieval in a Sparse Distributed Memory: The K-winner Modern Hopfield Network" (NeurIPS 2023) — K-winner MHNs retain old memories better than localist variants.
- Theoretical unification: attention ≈ Hopfield ≈ SDM under one framework.
- URL: https://openreview.net/forum?id=VOSrMFgWdL
- Companion: https://arxiv.org/abs/2208.09416 ("Kernel Memory Networks: A Unifying Framework")

### 9. Hopfield-Fenchel-Young Networks (HFYN) and Sparse and Structured Hopfield Networks (SSHN) — deep-spin, 2024

The retrieval side of the same idea — but with a structural twist that exactly matches Phase 5's "retrieve a pattern *association* not a single pattern."

- HFYN: energy = difference of two Fenchel-Young losses. Tsallis / norm entropies give sparse, end-to-end differentiable retrieval.
- **SparseMAP variant** retrieves *pattern associations* — multiple memory items selected jointly under structural constraints (e.g. "k contiguous items in a stored sequence"). This is *exactly* the shape of role-filler retrieval: not "the closest memory" but "the role-coherent subset."
- URLs:
  - https://arxiv.org/abs/2411.08590
  - https://arxiv.org/abs/2402.13725
  - https://github.com/deep-spin/HFYN
  - https://github.com/deep-spin/SSHN

### 10. "Attention as Binding" — VSA-lens on Transformers, Dec 2025

- Interprets QKV as VSA: queries+keys = role space, values = fillers, attention weights = soft unbinding, residual stream = superposition.
- Proposes explicit **binding/unbinding heads** and **hyperdimensional memory layers** with training objectives that promote role-filler separation.
- No empirical results yet, but the architectural recommendations align with what a substrate-level intervention here would look like.
- URL: https://arxiv.org/abs/2512.14709

### 11. Resonator Networks, FactorHD, Compositional Factorization (2024)

- Resonator networks (Frady-Sommer 2020) remain the workhorse FHRR factorizer; capacity ~10⁴-10⁵.
- **FactorHD** (2025, IEEE DAC) — explicit class-subclass relations via a "memorization clause" appended to bindings. Selectively eliminates redundant classes during factorization. 92.48% factorization accuracy on CIFAR-10 with ResNet-18; 5667× speedup at representation size 10⁹. https://arxiv.org/abs/2507.12366
- **Compositional Factorization of Visual Scenes with Convolutional Sparse Coding and Resonator Networks** (2024) — concrete demonstration of FHRR + resonator solving compositional scene factorization. https://arxiv.org/html/2404.19126

### 12. Histogram-Recovery VSA — Deng & Raviv, Nov 2025

- Coding-theoretic VSA built from concatenated Reed-Solomon + Hadamard codes.
- Recovery of compositional representations reduces to a histogram-recovery problem.
- Formal guarantees on encoding efficiency, quasi-orthogonality, and recovery *without* training or heuristics.
- This is the only VSA in the field with provable role-filler decoding guarantees as of the cutoff date.
- URL: https://arxiv.org/abs/2511.01838

---

## Concrete ideas for the project

Each idea is sized for a 1-3 day implementation spike. Anti-homunculus screen is per-idea.

### Idea A — Swap FHRR for GSBC at D=4096, B=64, L=64

- D=4096, B=64 blocks of L=64, one nonzero per block in binary mode (or unit-ℓ₁ per block in GSBC).
- Binding: block-wise circular convolution.
- Cleanup / role retrieval: ℓ∞-based similarity gives structural sparsity for free — most candidates evaluate to exact 0.
- This is the minimum-blast-radius "substrate swap" experiment. Existing codebook / replay / Hopfield pipeline ports because cap-coverage, meta-stable rate, and entropy all generalize.
- **Anti-homunculus screen:** PASSES. Binding and similarity are pure geometry. The ℓ∞ similarity zeros are an algebraic property of the metric, not a thresholding decision by a supervisor module.
- **Why this matters for role basins:** In GSBC, role and filler vectors that share no block-offset overlap have exact-zero similarity. The "no role basin" diagnosis may simply be that FHRR's cosine similarity puts a non-trivial baseline between any two random unit-magnitude vectors. GSBC has structural zeros.

### Idea B — Add a GHRR binding head to the existing FHRR substrate

- Keep the substrate FHRR. Add *one* non-commutative binding operator (GHRR-style) as a second binding op specifically for role-filler pairs.
- Role-filler is now algebraically distinguishable from filler-filler (which stays commutative FHRR).
- Implementation cost: low. GHRR's binding is a small modification of FHRR's circular convolution.
- **Anti-homunculus screen:** PASSES if and only if the choice of *which* binding to apply is a function of the codebook vectors themselves (e.g. a learned typing inferred from cluster geometry), NOT a hand-coded rule "if X is a role". If the binding choice is hand-coded, the screen fails — that's a supervisor.
- **Risk:** the hand-coded version is much easier and tempting. Watch for this.

### Idea C — Layer Residue Hyperdimensional Computing over FHRR

- Assign role indices to residue moduli. Different roles live in algebraically non-overlapping residue classes.
- Filler content lives in the unrestricted phase domain.
- The "role basin" problem reduces to "find the right residue modulus" — a discrete factorization with a known efficient algorithm.
- Compatible with existing FHRR substrate, codebook, MHN energy.
- **Anti-homunculus screen:** PASSES. Residue moduli are geometric — they're frequency-domain partitions, not arbitration decisions. Resonator-style factorization is local energy descent.
- **Highest leverage idea on this list.** Smallest blast radius, biggest algebraic upgrade.

### Idea D — Replace MHN energy with Hopfield-Fenchel-Young + SparseMAP for the retrieval step

- Keep FHRR substrate. Swap the retrieval energy from Ramsauer MHN to HFYN with SparseMAP transformation.
- SparseMAP retrieves pattern *associations* under structural constraints — natively the right primitive for role-filler.
- The constraint "the retrieved set must be role-coherent" is a structural constraint, not a supervisor decision.
- **Anti-homunculus screen:** PASSES if the structural constraint is encoded as an energy term (which HFYN supports) rather than as a post-hoc filter.
- **Implementation:** github.com/deep-spin/SSHN has reference code.

### Idea E — Banked TPR with D × R sparse representation

- D=4096, R=16 named roles. Store as a D × R sparse matrix where exactly one column is active per role-binding event.
- Outer-product binding `r ⊗ f` becomes column selection + filler vector.
- Memory cost: 16× FHRR's memory but role retrieval is *literally* a column index lookup.
- **Anti-homunculus screen:** PASSES if role identities emerge from the codebook (e.g. each role is one cluster's representative direction). FAILS if roles are hand-assigned labels.
- **Why this is worth revisiting:** the project rejected TPR for capacity reasons. With D=4096 already accepted, a 16-role bank is 65K dims — well within budget. The original rejection rationale needs to be revisited with current dim numbers.

### Idea F — Histogram-recovery VSA spike

- Use concatenated Reed-Solomon + Hadamard codes as the substrate.
- This is the only VSA with formal recovery guarantees on compositional decoding.
- If FHRR's "no role basin" failure is fundamentally a decodability failure under noise, this is the algebra most likely to surface a clean signal.
- **Anti-homunculus screen:** PASSES. Coding-theoretic decoding is pure geometry.
- **Risk:** least mature codebase. Likely no PyTorch implementation; would need to be built.

### Idea G — Add a non-commutative permutation as the role marker

- Cheapest possible intervention. Keep FHRR; permute each role's filler by a role-specific permutation matrix before binding.
- This is what the Gayler MAP literature recommends as the "minimal" non-commutativity fix.
- Already partially done in the codebase via position encoding — but position is not the same as role.
- **Anti-homunculus screen:** PASSES if permutations are tied to the *learned* role identities; FAILS if hand-assigned.

---

## Anti-homunculus screen — meta-observation

A common failure mode across all the alternative algebras: introducing role/filler asymmetry by **typing** vectors (this is a role, that is a filler). Typing is a supervisor decision unless the type is itself a *geometric attractor* of the codebook. The strongest ideas above (C residue, A GSBC, D HFYN-SparseMAP, F histogram-recovery) all have role asymmetry baked into the *algebra*, not into a tagging convention. The weaker ideas (B GHRR if hand-routed, E TPR if hand-labelled, G permutation if hand-assigned) all have a homunculus risk that has to be specifically designed away.

---

## Surprises

1. **GHRR exists and is from 2024.** A non-commutative FHRR variant is sitting there already, published a year before the project's Phase 5 push. It looks like exactly the surgical mod the substrate needs, and the project notes do not appear to reference it.
2. **HFYN/SSHN already does "retrieve a pattern association under structural constraints."** This is the *literal* primitive that every Phase 5 retrieval mechanism has been trying to approximate. The project notes already mention sparsemax / Hopfield-Fenchel-Young as covered — but the *SparseMAP structured variant* that retrieves k-item *associations* (not single patterns) appears to be a different beast and may not have been considered for role retrieval specifically.
3. **Residue HDC over FHRR is a drop-in upgrade.** Project keeps FHRR substrate; gains algebraically separable role indices via residue moduli. The 2023 paper and the 2025 VSA-Lisp follow-up suggest this is now a mature subsystem.
4. **Histogram-recovery VSA has *provable* compositional decoding** — the only VSA family with formal guarantees of this kind. If the Phase 5 failure is fundamentally a decodability failure, this is the strongest theoretical foundation.
5. **The "Attention as Binding" Dec-2025 paper** independently arrived at the architectural intervention the project might consider: explicit binding/unbinding heads + hyperdimensional memory layers + role-filler separation training objectives. That's three independent design moves the project could test.
6. **Transformer experiments (BERT/RoBERTa via TPDN) show that variable binding is *not* what large LMs do internally** — they use the input as external memory. This is a quiet vote in favour of giving the substrate an *explicit* binding mechanism rather than hoping it emerges.
7. **FactorHD's "memorization clause"** — an explicit appendage to bindings that gates factorization — is the kind of move the project might dismiss as a homunculus but is empirically getting 92% factorization on CIFAR-10 with a clean energy interpretation. Worth a careful anti-homunculus reading.

---

## Promising leads to dig deeper

1. **Resonator network applied to GSBC + FHRR composite** — combine the algebraic separation of GSBC with FHRR's continuous codebook. Hersche's "in-memory-factorizer" repo has the BCF reference.
2. **GHRR + Modern Hopfield energy** — does the GHRR binding operator compose with Ramsauer's exp(βx) attention as an energy? The GHRR paper doesn't address this. A 1-day spike could establish whether the two compose cleanly.
3. **SparseMAP retrieval as a Phase 5 retrieval-side experiment** — keep substrate, swap retrieval energy. This is the lowest-risk experiment on this list and directly answers "is the substrate or the retriever the problem?"
4. **Residue HDC role indexing** — Christopher Kymn (Berkeley Redwood) has been the most active author. Code likely lives in the Redwood/Sommer lab ecosystem.
5. **Histogram-recovery VSA** — Deng & Raviv at WashU. New (Nov 2025) and unlikely to have a PyTorch implementation yet, but coding-theoretic guarantees are too strong to ignore.
6. **K-winner MHN (NeurIPS 2023)** as a sparse alternative to the project's current Ramsauer MHN — sparsity gives geometric room for role basins that dense MHN can't.

---

## Sources

- [Variable Binding for Sparse Distributed Representations (Frady-Sommer 2020)](https://arxiv.org/abs/2009.06734)
- [Factorizers for Distributed Sparse Block Codes (Hersche et al. 2023/2025)](https://arxiv.org/abs/2303.13957)
- [IBM in-memory-factorizer code](https://github.com/IBM/in-memory-factorizer)
- [Generalized Holographic Reduced Representations (2024)](https://arxiv.org/abs/2405.09689)
- [Computing with Residue Numbers in High-Dimensional Representation (Kymn et al. 2023)](https://arxiv.org/abs/2311.04872)
- [Vector-Symbolic Lisp with Residue Arithmetic (Hanley et al. 2025)](https://arxiv.org/abs/2511.08767)
- [Computing on Functions Using Randomized Vector Representations / VFA (Frady-Kleyko 2021)](https://arxiv.org/abs/2109.03429)
- [Spatial Semantic Pointers — Komer 2019](https://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf)
- [Simulating Dynamical Systems with SSPs (Voelker et al. 2021)](https://direct.mit.edu/neco/article/33/8/2033/102625)
- [A Comparison of VSAs (Schlegel-Neubert-Protzel 2021)](https://link.springer.com/article/10.1007/s10462-021-10110-3)
- [RNNs Implicitly Implement Tensor Product Representations (McCoy et al. ICLR 2019)](https://arxiv.org/abs/1812.08718)
- [K-winner Modern Hopfield Network (NeurIPS 2023)](https://openreview.net/forum?id=VOSrMFgWdL)
- [Hopfield-Fenchel-Young Networks (2024)](https://arxiv.org/abs/2411.08590)
- [Sparse and Structured Hopfield Networks (2024)](https://arxiv.org/abs/2402.13725)
- [deep-spin HFYN code](https://github.com/deep-spin/HFYN)
- [deep-spin SSHN code](https://github.com/deep-spin/SSHN)
- [Attention as Binding: VSA on Transformer Reasoning (Dec 2025)](https://arxiv.org/abs/2512.14709)
- [FactorHD (IEEE DAC 2025)](https://arxiv.org/abs/2507.12366)
- [Compositional Factorization of Visual Scenes with Resonator Networks (2024)](https://arxiv.org/html/2404.19126)
- [Efficient VSAs from Histogram Recovery (Deng-Raviv Nov 2025)](https://arxiv.org/abs/2511.01838)
- [Kernel Memory Networks: A Unifying Framework (2022)](https://arxiv.org/abs/2208.09416)
- [HD-Computing FAQ portal](https://www.hd-computing.com/faq)
- [HDC/VSA Survey Part II (Kleyko et al. ACM CSUR 2023)](https://dl.acm.org/doi/abs/10.1145/3558000)
