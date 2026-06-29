---
date: 2026-05-13
angle: hopfield-temperature-dynamics
session: 2026-05-13-neuro-personal-ai
---

# Adaptive Temperature / Inverse Beta Dynamics in Modern Hopfield Networks: A Research Brief

## Angle

**What was investigated:** Whether the "regime disjoint failure" observed at D=4096 — where soft-blend (β ≤ 0.001) and sharp-retrieval (β ≥ 0.01) regimes are mutually exclusive — is a known theoretical result, whether the crossover zone at β ∈ (0.001, 0.01) can be bridged with dynamic temperature control, and what architectural alternatives can give both graded blending and sharp pattern completion from the same Hopfield landscape.

**Why it's relevant:** The project needs its retrieval substrate to produce two qualitatively different signals from the same memory bank: (1) a graded, K-weighted blend when a cue is genuinely ambiguous across K stored patterns, and (2) a committed single-pattern retrieval when the cue is unambiguous. At D=4096 with the standard exponential energy function, these two behaviors live in non-overlapping β regimes. If this is a fundamental geometric impossibility, the architecture needs a different mechanism. If it is solvable by scheduling, kernel choice, or learned adaptive temperature, that changes the design space significantly.

---

## Key Findings

### 1. The regime disjoint is a known theoretical result, not a tuning problem

The phase transition is formally characterized in Demircigil et al. (2022) and extended in multiple 2023–2026 works. The key result: the exponential Modern Hopfield Network (softmax energy) undergoes a **sharp phase transition at a critical inverse temperature β_c**. Below β_c the energy has a single global attractor (the centroid of all stored patterns); above it, attractors individuate onto stored patterns.

The critical β_c is not a fixed constant — it depends on the **effective inverse temperature β_eff**, which couples the nominal β to the distribution and norms of stored patterns: β_eff accounts for the "size" of the stored memory set as experienced by the update rule. This is the key insight from the 2023 NeurIPS workshop paper by Ramsauer's group: the gap between "blending" and "sharp retrieval" at any given β may be entirely caused by mismatch between β and β_eff, not just the raw β value.

The geometrically-derived reason: on the N-sphere, the free energy decomposes into a kernel-dependent energy term (alignment gain) and a **purely geometric entropy term s(ϕ) = ½ ln(1 − ϕ²)** that depends only on the angular alignment ϕ and the spherical geometry. This entropic pressure grows with dimensionality and penalizes committed (high-ϕ) states. At D=4096, this entropic penalty is very large, which pushes the required β for sharp retrieval much higher than at lower D — consistent with observing the crossover at β ∈ (0.001, 0.01) rather than at lower values.

**Source:** [Exploring the Temperature-Dependent Phase Transition in Modern Hopfield Networks (arXiv:2311.18434)](https://arxiv.org/abs/2311.18434); [Geometric Entropy and Retrieval Phase Transitions in Continuous Dense Associative Memory (arXiv:2604.07401)](https://arxiv.org/html/2604.07401)

### 2. The disjoint is not symmetric across kernel choices — LSR solves part of the problem

The Gaussian kernel (Log-Sum-Exp, i.e., standard softmax) has infinite support: every stored pattern always interferes with every other, at every temperature. This creates a critical line at all memory loads α > 0. There is no temperature at which sharp retrieval is guaranteed regardless of load.

The Epanechnikov kernel (Log-Sum-ReLU, compact support) has a qualitatively different phase structure: **below a support threshold α_th, no spurious patterns fall within kernel support, and retrieval is perfect at ANY temperature** — the critical line vanishes in the low-load regime. Both kernels share the same zero-temperature capacity limit α_c(0) = 0.5 (confirming capacity is a geometric property of the sphere, not the kernel). But the LSR kernel achieves temperature-independent sharp retrieval when the memory load is below α_th.

Critically, a 2026 paper also shows that at **intermediate β**, the LSR energy does something the LSE cannot: it generates **emergent memories** — novel local minima at weighted centroids of overlapping patterns — while simultaneously preserving perfect memorization of all original patterns. The LSR energy can be in a "global emergence" regime (β intermediate) where the original M patterns are all fixed points AND exponentially many novel interpolation states exist as additional attractors.

**This is the closest existing result to "both blending and retrieval from the same landscape."**

**Source:** [Thermal Robustness of Retrieval in Dense Associative Memories: LSE vs LSR Kernels (arXiv:2603.13350)](https://arxiv.org/html/2603.13350); [Dense Associative Memory with Epanechnikov Energy (arXiv:2506.10801)](https://arxiv.org/html/2506.10801)

### 3. Dynamic β scheduling (annealing) is theoretically principled but practically crude

Classical simulated annealing uses high-temperature starts (β low → global blend) and gradual β increases to settle into attractors. This is the original Boltzmann machine insight and still works in the MHN setting. However, annealing schedules are query-agnostic: they reduce temperature uniformly for all queries regardless of whether the cue is ambiguous or unambiguous. The 2023 NeurIPS workshop paper notes empirically that **decaying β over training epochs improves generalization** in Modern Hopfield Classifiers — suggesting the schedule should match the phase of learning (blend early, sharpen late), not be dynamically adapted to the query.

For the project's use case, schedule-based annealing solves the temporal problem (blend then commit during a single retrieval trajectory) but does not solve the per-query problem (ambiguous cues need different treatment than unambiguous ones at the same β).

**Source:** [Exploring the Temperature-Dependent Phase Transition in Modern Hopfield Networks (arXiv:2311.18434)](https://arxiv.org/abs/2311.18434)

### 4. Input-driven plasticity provides the cleanest known mechanism for per-cue temperature-equivalent dynamics

The Input-Driven Plasticity (IDP) Hopfield model (Science Advances, 2025) achieves the closest neuroscience-grounded answer to the project's problem. Its mechanism: external inputs directly modulate the **synaptic weight matrix** rather than just setting the initial state. Each memory's "saliency weight" is computed from alignment between the cue and stored patterns, and the energy landscape is continuously reshaped — deepening basins for dominant matches, flattening ones for weak matches.

The result: for an **unambiguous cue** (strong alignment to one pattern), the basin for that pattern deepens fast and the network sharpens its commitment. For an **ambiguous cue** (distributed alignment across multiple patterns), multiple basins remain simultaneously viable, and the network maintains a graded state between them. There is **no explicit β parameter being adjusted** — the effective temperature of the retrieval emerges from the dynamics of the saliency-weighted landscape itself.

This is exactly the anti-homunculus shape the project demands: no module reads β and adjusts it; the "apparent decision" between blend and commit is local geometry — the depth of wells relative to each other.

**Source:** [Input-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks (arXiv:2411.05849)](https://arxiv.org/html/2411.05849v1)

### 5. Sparse Hopfield (SparseMAP) enables structured multi-pattern retrieval, not just soft blend

Two separate NeurIPS/ICML 2023–2024 lines of work show that replacing the softmax separation function with a sparse transformation (sparsemax, α-entmax, SparseMAP) fundamentally changes the regime structure. The sparse Hopfield energy has a **margin property**: when the score gap between the top-k patterns and the rest exceeds the margin, those top-k patterns are retrieved exactly (sharp, zero error), with zero probability on all others. For ambiguous cues below the margin, the output is a soft combination over the top-k patterns.

SparseMAP extends this to **structured k-pattern retrieval**: the network can retrieve an association of k patterns simultaneously rather than a winner-take-all single pattern, with structural constraints ensuring which combinations are valid. This is different from the blend/retrieve tradeoff: it is a committed retrieval of a *set* of patterns rather than a soft average.

**Source:** [On Sparse Modern Hopfield Model (NeurIPS 2023)](https://arxiv.org/html/2309.12673); [Sparse and Structured Hopfield Networks (ICML 2024)](https://arxiv.org/html/2402.13725); [Hopfield-Fenchel-Young Networks (arXiv:2411.08590)](https://arxiv.org/html/2411.08590)

### 6. Learnable per-head temperature in attention solves a related but weaker problem

Work on learnable temperature in self-attention (NeurIPS 2024 Selective Attention; blogpost from 2024) shows that adding a per-head or per-layer scalar temperature to the softmax in attention dramatically improves performance. Different heads learn different temperature values — some sharpen to near-sparsemax behavior, some remain soft. However, this is a per-head-per-layer fixed parameter, not a per-query dynamic. It differentiates memory *roles* (some heads always sharp, some always soft) but does not adapt within a single query based on cue geometry.

The 2025 paper on optimal attention temperature (arXiv:2511.01292) derives a closed-form optimal τ that depends on input covariance shifts and noise — suggesting the theoretically correct β for sharp retrieval is a function of **the query's distributional distance from the stored patterns**, not just a fixed global constant. This provides theoretical grounding for per-query adaptive β.

**Source:** [Selective Attention: Enhancing Transformer through Principled Context Control (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/14fc4a68da97a3d31eb11c642b0b10fc-Paper-Conference.pdf); [Optimal Attention Temperature (arXiv:2511.01292)](https://arxiv.org/html/2511.01292)

### 7. The memorization-to-generalization transition shows that the blend/retrieve gap IS the generalization gap

Multiple 2024–2025 papers (NeurIPS 2024, ICLR 2025) analyze the Hopfield energy landscape through the lens of the transition from memorization to generalization. The finding: **spurious states (mixtures of stored patterns) are NOT retrieval failures — they ARE the onset of generalization**. As load α increases past the memorization threshold, new attractors form at pattern centroids. These spurious states represent the network's generalization capacity.

This reframing is important for the project: the "blending signal" at low β is not a degraded sharp-retrieval signal — it is a distinct, meaningful output: a prototype or centroid over the K most-relevant patterns. The architectural question is not "how do we get blending to also do sharp retrieval" but "how do we get the system to output whichever one is appropriate for this cue."

**Source:** [Memorization to Generalization: Emergence of Diffusion Models from Associative Memory (NeurIPS 2024 / arXiv:2505.21777)](https://arxiv.org/html/2505.21777)

---

## Promising Leads

**1. LSR (Epanechnikov) kernel at intermediate β for this project.**
The "global emergence" regime of the LSR energy — intermediate β, all M originals preserved AND emergent blend attractors coexist — is a direct match for what the project needs. The phase diagram for the LSR energy should be mapped at D=4096 to see where the emergence regime falls. This is a concrete experiment: replace the exponential energy with LSR and scan β. Papers: arXiv:2506.10801, arXiv:2603.13350.

**2. Input-Driven Plasticity as an architectural primitive.**
The IDP model's saliency-weighted synaptic modulation is implementable as a Hopfield variant where the memory matrix M is modulated by a diagonal saliency gate W_s = diag(sim(cue, memories)). This changes the effective β per-memory rather than globally, achieving the per-cue effect without any supervisor. This is anti-homunculus-clean: the saliency is a local dot product, the gate is a function of local geometry. Full paper: arXiv:2411.05849, Science Advances 2025.

**3. SparseMAP for structured K-pattern retrieval.**
For the K-ambiguous signal use case specifically, SparseMAP provides an output that is not a soft blend over all M patterns (diffuse, useless) but a hard commitment to the top-K patterns with zero weight on all others. This gives the K-graded signal as a *sparse structured* output rather than a continuous blend. Papers: arXiv:2402.13725, arXiv:2411.08590.

**4. Per-query effective β via query norm estimation.**
The β_eff formulation from arXiv:2311.18434 suggests that if pattern norms are fixed, the effective temperature can be adjusted by scaling the **query** rather than by changing β globally. A pre-retrieval step that normalizes the query to sit at a target similarity distribution to the memory bank could push the retrieval into the correct regime for that specific cue. This is a learnable or heuristic pre-processing step, not a supervisor.

**5. Dynamic Manifold Hopfield Networks (arXiv:2506.01303).**
Context-dependent reshaping of the attractor manifold through learned network interactions — achieves 64% retrieval accuracy at 2N patterns in N neurons vs 1%/13% for classical/modern variants. This could be the right shape for the project's contextual-completion target. Investigate the mechanism: if the deformation is local and geometric (no supervisor), it passes the anti-homunculus filter.

**6. EDEN (multi-timescale Hopfield) for blend-then-commit dynamics.**
EDEN (arXiv:2510.24965) separates fast convergence to current attractors from slow asymmetric interactions that reshape the landscape over time. The α_s/α_c ratio controls whether the network stays sharp (static mode) or allows sequential transitions (dynamic mode). This could be adapted as: start in dynamic mode (effective high-T, blend) → shift ratio → converge to sharp attractor. The mechanism is local timescale separation, not a supervisor.

---

## Concrete Ideas

**Idea 1: Two-kernel cascade (blending pass → sharpening pass)**
Use an LSR kernel at β_intermediate for the blending pass to identify the top-K candidate patterns as a sparse emergent attractor, then run a second pass with β_high focused only on those K patterns. This is a two-stage retrieval that separates the blend and commit operations into separate Hopfield steps over different subsets. The switch between stages is triggered by convergence of the first stage (energy settled), not by any metric-reading supervisor.

**Idea 2: Saliency-gated IDP as a drop-in Hopfield layer**
Implement the IDP saliency weighting as a modulated memory matrix: for each retrieval, compute sim_k = exp(β_low * q · m_k) for all k, then scale the memory matrix column k by sim_k^γ for some γ > 0 before running standard Hopfield dynamics at β_high. This creates a landscape where only the most-relevant memories form deep wells, with depth proportional to relevance. The γ parameter controls how aggressively ambiguous vs. clear cues differ. No supervisor: γ is fixed, the depth differential emerges from geometry.

**Idea 3: Learnable α in sparse Hopfield for automatic regime selection**
The generalized sparse Hopfield with learnable α (the MAGICS-LAB implementation has this mode) allows the network to learn per-layer or even per-query α that interpolates between softmax (α=1, full blend) and sparsemax (α=2, top-K commitment). Train on both blend and commit examples. Let the network learn that ambiguous-cue inputs should have low α, clear-cue inputs should have high α. The adaptation is local to the scoring function, not a global controller.

**Idea 4: Query-norm-based β_eff normalization**
Before each retrieval, compute the mean cosine similarity of the query to all stored patterns: μ = (1/M) Σ_k cos(q, m_k). Use this as a proxy for "how much in the blend zone is this query." Scale the effective β as β_eff = β_base / (1 + γ * σ²_similarity), where σ² is the variance of similarities. When the query has high-variance similarity distribution (spread across K patterns), β_eff decreases, producing blending. When it has low-variance (concentrated on one pattern), β_eff stays high, producing sharp retrieval. No supervisor — the computation is a local geometric measurement.

**Idea 5: Cascade retrieval using K-winner Hopfield as the first pass**
Use the K-winner modern Hopfield Network (NeurIPS 2023 workshop paper) as a coarse selector: run one retrieval step with β_low to identify which K patterns the cue is nearest to, then route to a β_high Hopfield over only those K patterns. The K-winner step acts as a geometric projection, not a decision-maker — it reports which K memories are within the cue's basin of attraction. The subsequent sharp step is just standard Hopfield on a smaller M.

---

## Surprises

**Surprise 1: The blend/retrieve gap is not a bug — it is the generalization/memorization gap, and it is fundamental to the geometry.**
Multiple 2024–2026 papers converge on this: the blend regime (low β, single global attractor) and the retrieval regime (high β, individual pattern attractors) are not imperfect versions of each other. They are categorically different uses of the same landscape. The project framing of "regime disjoint failure" is accurate as an engineering characterization, but the theoretical framing is that these are two distinct computational modes of the same energy function. The question is not "can we tune β to be in both regimes simultaneously" — it is "can we architect a system that selectively operates in each regime depending on cue geometry."

**Surprise 2: The Epanechnikov/LSR kernel achieves what the exponential kernel cannot: coexistence of memorized originals and emergent blends.**
This was unexpected. The standard assumption (from Ramsauer 2020 onward) is that you pick one: either you have a diffuse energy that blends, or a peaked energy that retrieves. The LSR paper (arXiv:2506.10801, 2026) shows that with a compactly-supported kernel at intermediate β, you can simultaneously have all M original patterns as stable fixed points AND exponentially many emergent blend attractors. The system navigates to originals from unambiguous cues and to emergent centroids from ambiguous ones — automatically, via energy minimization. This is exactly the architecture the project wants, and it is a new result not present in the mainstream MHN literature.

**Surprise 3: Adaptive temperature in the attention mechanism sense (per-head learnable scalar) is already the state of practice in 2024–2025 transformers, with measurable benefits.**
The project is not proposing something exotic — it is proposing something that the transformer community has already converged on empirically (NeurIPS 2024 Selective Attention, focal attention 2025, learnable per-head temperature blog 2024). The difference is that transformer work treats different heads as having different fixed temperatures (roles), not dynamically adapting per query. The theoretical paper arXiv:2511.01292 shows there is a principled optimal temperature as a function of the query's distributional shift from the training distribution — this is the theoretical grounding for per-query adaptive β that does not yet exist as a practical implementation in Hopfield networks.

**Surprise 4: Dynamic Manifold Hopfield achieves 64% retrieval accuracy at 2N patterns in N neurons vs. 1% for standard MHN — without changing β at all.**
The DMHN result (arXiv:2506.01303) suggests that the constraint being hit at D=4096 is not purely about β — it is about the rigidity of the learned attractor manifold. If the interactions are allowed to be context-modulated (learned to deform the manifold per cue), the capacity and retrieval accuracy improve dramatically even at fixed β. This frames the D=4096 regime disjoint not just as a temperature problem but as a **manifold rigidity problem**: the softmax landscape has a fixed geometry, and at high D, that geometry is too rigid to support both blend and sharp attractors at the same β.

**Surprise 5: The stochastic Hopfield network has a critical noise regime (p_c ~ 0.23–0.3) where temporal memory becomes long-range correlated.**
The critical zone in the stochastic MHN (arXiv:2509.17152) is not between blend and sharp retrieval — it is a dynamical criticality regime with long-range temporal correlations. This is related but distinct from the β-phase transition. The project's crossover zone at β ∈ (0.001, 0.01) may actually be overlapping with this dynamical criticality zone — which would explain why neither pure blend nor pure retrieval is achieved there. The system may be in a critical state with persistent temporal memory, not a broken retrieval state.

---

## Summary Table: Approaches to Bridging the Regime Gap

| Approach | Mechanism | Anti-Homunculus? | Directly Applicable? |
|---|---|---|---|
| β annealing during retrieval trajectory | Schedule-based global β decrease → increase | Yes (no reader) | Partial — agnostic to cue |
| β_eff normalization via query norm | Geometric pre-scaling of query | Yes | Yes, implementable now |
| LSR / Epanechnikov kernel | Compact support creates emergent memories at intermediate β | Yes | High priority experiment |
| Input-Driven Plasticity (IDP) | Saliency-gated synaptic modulation | Yes | Architectural change required |
| Sparse Hopfield / SparseMAP | Margin-based k-pattern commit | Yes | Yes, drop-in for softmax |
| Learnable α (generalized sparse) | Trained sparsity per layer | Yes | Requires training |
| Dynamic Manifold Hopfield | Learned context-dependent attractor deformation | Needs verification | Medium-term |
| Two-kernel cascade | Blend pass (LSR, β_mid) → commit pass (LSE, β_high) | Yes | Composable, try this |

---

*Research conducted 2026-05-13. All sources accessible as of this date.*
