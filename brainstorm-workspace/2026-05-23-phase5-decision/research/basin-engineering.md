# Modern Hopfield Basin Engineering — survey for Neuro-AI Phase 5

**Date:** 2026-05-23
**Angle:** Modern Hopfield basin engineering: how to preserve capacity while
keeping basin geometry differentiable enough for prior-traversal to discriminate.
**Author:** research agent
**Audience:** Phase 5 brainstorm — the K-branch / prior-traversal foreclosure
problem in the Neuro-AI substrate.

---

## 0. Angle in one paragraph

Phase 5's K-branch mechanism requires that different priors land in *different*
attractors of the substrate's Modern Hopfield retrieval. After Phase 4's binary
mass-death + Phase 5's A+B+A1' refinement, the substrate has been engineered for
"clean sharp self-retrieving basins": all 1064 surviving atoms have f_i = 0.9858
± 0.0015, effective dimensionality ~5 of 4096, and K=4 priors all converge to
the same min-energy attractor (state divergence at FP-precision floor 6.7e-6 to
2.7e-3). This is the textbook failure mode of a softmax-over-inner-products
energy at high β: when the energy landscape is dominated by deep, well-separated,
near-identical basins, *every* query within any reasonable cone of a basin gets
pulled into the same fixed point regardless of the prior weighting. The
literature 2023-2026 has been actively wrestling with the sharp-vs-soft
basin tradeoff under different names (kernel choice, sparsity, input-driven
synapses, context-dependent manifolds, stochastic settling). The good news is
that there are *several* concrete, published, mathematically clean mechanisms
that decouple "capacity at scale" from "prior-discriminable basin geometry."
The best fit for Neuro-AI Phase 5 is likely a hybrid of (a) prior-as-attention-
logit-bias (Varner 2026), (b) Langevin stochastic settling at intermediate β
(Alswaidan & Varner 2026), and (c) input-driven synapses (Betteti et al. 2024)
— all of which are *additive* to the existing Phase 4 substrate.

---

## 1. The sharp-vs-soft basin tradeoff in modern Hopfield (2024-2026 state)

### 1.1 The textbook tension

Ramsauer et al. 2020's "Hopfield Networks is All You Need" exposed the
parameter that controls the tradeoff: inverse temperature β.
- β → ∞: deep, sharp basins; one minimum per stored pattern; winner-take-all
  retrieval; minimal cross-talk; *but* identical input + nearly identical
  basins ⇒ no prior-discriminability.
- β → 0: shallow, blurred energy with a single global minimum at the origin;
  prior-traversal is geometrically possible but retrieval is meaningless.
- Phase 5's choice (β=10) sits in the upper-middle of this range but, combined
  with N=1064 atoms ≈ ε·D and Phase 4's basin-sharpening, has saturated into
  the "every prior wins the same attractor" regime.

Krotov & Hopfield's higher interaction-order construction (n=2,3,20,30) showed
that the *shape* of the separation function controls how memories transition
from "features" to "prototypes" at n≈20
([arXiv:1606.01164](https://arxiv.org/abs/1606.01164)). β-tuning is one knob;
*functional form of the energy* is the other.

### 1.2 The 2024-2026 state-of-art reframing

The recent literature has converged on three insights:

1. **Kernel choice matters more than β.** LSE (Gaussian, infinite support)
   vs LSR (Epanechnikov, finite support) give *qualitatively* different
   phase diagrams — LSR has a finite-support threshold below which retrieval
   is perfect at any temperature, LSE does not
   ([Hoover et al. 2025 — arXiv:2506.10801](https://arxiv.org/abs/2506.10801),
   [Petrova 2026 — arXiv:2603.13350](https://arxiv.org/abs/2603.13350),
   [Petrova 2026 — arXiv:2604.07401](https://arxiv.org/abs/2604.07401)).
2. **The synaptic matrix itself can be input-dependent.** Betteti et al.'s
   IDP Hopfield ([arXiv:2411.05849](https://arxiv.org/abs/2411.05849),
   Sci. Adv. 2025) makes W = W(u(t)) so each input *reshapes* basin depths.
3. **The settling dynamics, not the energy, do the work.** Langevin
   sampling at intermediate β yields multi-basin coverage that deterministic
   gradient descent cannot
   ([Alswaidan & Varner 2026 — arXiv:2603.06875](https://arxiv.org/abs/2603.06875)).

The unifying frame is: **don't fix the energy and then settle deterministically
— make the energy a function of a context/prior signal, or settle stochastically,
or both.** The architectural premise that "the substrate has one energy and
priors are query perturbations on top" is the part that breaks.

---

## 2. Approaches surveyed

### 2.1 Hopfield-Fenchel-Young Networks (HFYN)
- **Citation:** Santos, Niculae, McNamee, Martins — ICML 2024 + arXiv 2024/11
  ([arXiv:2411.08590](https://arxiv.org/abs/2411.08590);
  prequel "Sparse and Structured Hopfield" — [arXiv:2402.13725](https://arxiv.org/abs/2402.13725); code at [deep-spin/HFYN](https://github.com/deep-spin/HFYN), [deep-spin/SSHN](https://github.com/deep-spin/SSHN)).
- **Mechanism:** Replaces the softmax in modern Hopfield update with a
  *family* of Fenchel-Young losses, each parameterized by a generalized
  (Tsallis, norm) entropy. Softmax is one instance; sparsemax, α-entmax,
  and SparseMAP are others. The energy is `E = FY_Ω(score) − FY_Ω'(output)`.
- **Sharp at scale?** Yes — the paper proves a connection between *loss
  margin* and *exact retrieval*: sparser losses with bigger margins give
  exact single-pattern retrieval. Sparsemax in particular gives exact
  retrieval with finite-support attention weights, no asymptotic tail.
- **Differentiable basin shaping under priors?** Indirectly — different
  Ω choices give different basin shapes, but Ω is *fixed* per layer in the
  paper, not query-conditioned. However, the framework is "end-to-end
  differentiable sparse transformations," so a learned, prior-dependent Ω
  is a clean extension.
- **Settling/convergence:** Single-step retrieval guarantees under margin
  conditions; otherwise iterative CCCP-style updates with energy descent.
- **K-parallel branches?** Yes via SparseMAP — it "retrieves pattern
  *associations* instead of a single pattern" (i.e., a sparse subset of
  patterns simultaneously). This is the closest published thing to a
  built-in top-K mechanism.

### 2.2 Sparse Modern Hopfield (sparsemax/entmax)
- **Citation:** "On Sparse Modern Hopfield Model" NeurIPS 2023
  ([MAGICS-LAB/SparseModernHopfield](https://github.com/MAGICS-LAB/SparseModernHopfield)).
- **Mechanism:** Same idea as HFYN's sparse instance — sparsemax or 1.5-entmax
  in place of softmax. Output of the update rule has exact zeros on irrelevant
  memories.
- **Sharp at scale?** Yes — proves exponential capacity with sparser-than-LSE
  sparsity.
- **Per-atom discriminability under priors?** Partly. Sparsemax has a
  *piecewise-linear* update with hard thresholds, so small prior perturbations
  can flip an atom in/out of the support — this is the kind of
  discontinuity that *could* make priors discriminate, but it also makes the
  retrieval non-smooth (potentially hostile to Phase 5's "differentiable"
  desideratum).

### 2.3 LSR / Epanechnikov-energy Dense Associative Memory
- **Citation:** Hoover, Pham, Liang, Krotov, Strobelt, Chau, Zaki — June 2025
  ([arXiv:2506.10801](https://arxiv.org/abs/2506.10801)). Follow-up
  thermodynamic analysis: Petrova 2026 ([arXiv:2603.13350](https://arxiv.org/abs/2603.13350) and [arXiv:2604.07401](https://arxiv.org/abs/2604.07401)).
- **Mechanism:** Replace LSE (log-sum-exp, Gaussian kernel, infinite support)
  with LSR (log-sum-ReLU, Epanechnikov kernel, finite support):
  - `E_β^LSR(x; Ξ) = -(1/β) log(ε + Σ_μ ReLU(1 - (β/2)‖x - ξ_μ‖²))`
  - Each pattern contributes only within radius `√(2/β)`.
- **Sharp at scale?** Yes — matches LSE exponential capacity, *and* gives
  *exact* zero gradients at minima (LSE is only approximate). Theorem 1:
  single-step retrieval to ξ_μ when `β = 2/(r-Δ)²`.
- **Differentiable basin shaping under priors?** YES — and this is the
  most exciting finding. The LSR energy admits "abundant additional emergent
  local minima" that are *centroids of overlapping basin subsets*:
  `x* = (1/|B(x*)|) Σ_μ∈B(x*) ξ_μ` where `B(x*) = {μ: ‖x* - ξ_μ‖ ≤ √(2/β)}`.
  These emerge at critical β values where multiple support regions overlap.
  **Unlike LSE's spurious memories, these are *desirable*: they form a
  discrete combinatorial menu of attractors that a prior could select among.**
  Count scales exponentially with M (number of patterns), or polynomially
  `Θ((M^(1/d) - λ^(1/d) + 1)^d)` on a uniform grid.
- **Settling/convergence:** Single-step deterministic retrieval. Authors
  provide enumeration algorithms (~15 min for VAE-latent experiments).
- **K-parallel branches?** The paper doesn't explicitly use K-branch language,
  but the emergent-minima menu is directly compatible: K different priors
  could each select a different overlapping-subset centroid.
- **Watch out:** Gradient vanishes *exactly* outside the support of all
  patterns — distant queries get no gradient. Authors propose hybrid LSE-LSR
  or query-conditioned β(x) as future work.

### 2.4 Energy Transformer (ET)
- **Citation:** Hoover, Liang, Pham, Panda, Strobelt, Chau, Zaki, Krotov —
  NeurIPS 2023 ([arXiv:2302.07253](https://arxiv.org/abs/2302.07253)).
- **Mechanism:** Replace the sequence of feedforward transformer blocks with
  a *single* associative-memory layer whose attention layers are interpreted
  as gradient descent on an engineered energy `E_attn + E_HN`. The
  representation *settles* via several gradient steps to a low-energy
  configuration of all tokens jointly.
- **Sharp at scale?** Mostly — it inherits modern Hopfield's exponential
  capacity, but optimized for *joint* token-configuration basins ("context
  wells"), not per-pattern basins.
- **Differentiable basin shaping?** Yes — input tokens act as boundary
  conditions on the energy. Different inputs ⇒ different settling
  trajectories ⇒ different attractors. This is exactly the desired
  prior-traversal property, *at the configuration level*.
- **Settling iterations:** Typically 6-12 iterations of energy gradient
  descent per block.
- **K-parallel branches?** Not natively, but the architecture is well-suited
  to it: spin up K parallel settling chains from K initial conditions / K
  prior-biased starts, let each find its own context well.
- **Related work:** Hyperspherical Energy Transformer / Hyper-SET
  ([arXiv:2502.11646](https://arxiv.org/abs/2502.11646)) extends this to
  spherical geometry with recurrent depth and is currently the
  state-of-art energy-transformer.

### 2.5 Universal Hopfield Networks (Millidge et al. 2022) + follow-ups
- **Citation:** Millidge, Salvatori, Song, Lukasiewicz, Bogacz —
  ICML 2022 ([PMLR v162](https://proceedings.mlr.press/v162/millidge22a.html);
  also [PMC 7614148](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/)).
- **Mechanism:** Decomposes any single-shot associative memory model into
  three operations: **similarity → separation → projection**. This is the
  abstract framework that makes the "kernel choice" insight precise — both
  the similarity and the separation function are *independently choosable*.
- **Why it matters here:** This is the lens through which to understand
  the LSE→LSR shift, the softmax→sparsemax shift, etc. Phase 5's substrate
  is (cosine similarity) → (softmax separation) → (linear projection). The
  three knobs can be turned independently. A particularly under-explored
  combination for Phase 5 would be: (cosine similarity) → (top-K separation
  with prior bias) → (linear projection).

### 2.6 Krotov Hierarchical Associative Memory (HAM)
- **Citation:** Krotov 2021 ([arXiv:2107.06446](https://arxiv.org/abs/2107.06446)),
  extension Hoover et al. 2024 ([arXiv:2406.12220](https://arxiv.org/html/2406.12220v1)).
- **Mechanism:** Multi-layer recurrent associative memory with a *global*
  energy bounded from below. Bottom-up and top-down information flow.
- **Why it matters:** Provides the formal architecture for layered
  energy-based settling. Phase 5's K-branch could be implemented as a HAM
  with a "prior layer" whose activations bias the bottom layer's settling.
- **Caveat:** Trains as a single energy network; doesn't natively address
  basin-discriminability under priors.

### 2.7 Hopfield Encoding Networks (HEN)
- **Citation:** Kashyap, D'Souza, Shi, Wong, Wang, Syeda-Mahmood —
  NeurIPS UniReps workshop 2024 ([arXiv:2409.16408](https://arxiv.org/abs/2409.16408)).
- **Mechanism:** Encode raw content into a *learned latent space* (via a
  pre-trained autoencoder) before storing in the Modern Hopfield Network.
- **Why it matters for Phase 5:** This is the closest published address of
  the meta-stable / collapse problem that Phase 4 mass-death also addressed.
  The lesson: *the geometry of the input space dictates the basin geometry
  of the Hopfield layer.* If the substrate's atom geometry has effective
  dimensionality 5, any softmax-Hopfield on top will collapse to those 5
  directions, regardless of K, prior, or β. The fix is upstream: shape the
  representations such that the basins are well-separated *in directions
  Phase 5 cares about*.
- **Hetero-association:** HEN's cross-domain retrieval (text → image) shows
  that a *prior in one space* can retrieve a basin *in another* if the
  encoder maps them to a shared latent. This is the cleanest published
  precedent for prior-traversal.

### 2.8 Stochastic Attention via Langevin Dynamics
- **Citation:** Alswaidan & Varner — May 2026 ([arXiv:2603.06875](https://arxiv.org/abs/2603.06875)).
- **Mechanism:** Add isotropic noise to the modern Hopfield update:
  - `ξ_{t+1} = (1-α)ξ_t + α·X·softmax(β X^T ξ_t + b) + √(2α/β)·ε_t`
  - Three terms: contraction toward origin, softmax attention pull, Gaussian
    perturbation. Temperature β controls the tradeoff continuously.
- **Sharp at scale?** Yes — at high β it reduces to deterministic retrieval.
- **Differentiable basin shaping under priors?**
  - **Bias term b is the prior knob.** Adding `b_k = -∞` masks memories out
    of the prior's support; finite `b_k = log r_k` gives soft prior weighting.
  - On Olivetti faces: 96.0% subject recovery with hard masking vs 10.4%
    unconditional. This is essentially a textbook demonstration of
    prior-discrimination via attention-logit bias.
- **Multi-basin coverage:** YES, this is the key insight. At intermediate β
  (β=200 for d=64), single-chain diversity 0.796 *exceeds* multi-chain
  β=2000 diversity 0.600. Lower β = better inter-basin mixing.
- **Settling/sampling cost:** Per-step O(dK), same as one attention head;
  typical protocol T=5000 with burn-in 2000, thinning every 100, giving
  150 samples from 30 chains. Training-free.
- **K-parallel branches?** Natively — "30 independent chains, each initialized
  near a different stored pattern." This is *exactly* the K-branch architecture
  Phase 5 needs, generalized to K=30+.
- **Critical β formula:** `β* ∼ √d` (∼√4096 = 64 for Phase 5's substrate)
  for the entropy-inflection transition. Worth measuring this on the actual
  Phase 5 substrate.

### 2.9 Conditioning via Hopfield Pattern Multiplicity
- **Citation:** Varner — March 2026 ([arXiv:2603.20115](https://arxiv.org/abs/2603.20115)).
- **Mechanism:** A *single scalar parameter* added as a bias to softmax
  attention logits:
  - `b_k = log r_k`, where r_k is per-pattern multiplicity weight.
  - Binary partition: K_des "designated" patterns get weight ρ; K_bg
    "background" patterns get weight 1.
  - **Multiplicity ratio:** `ρ(f) = f·K_bg / [K_des·(1-f)]` controls the
    effective designated-fraction f at the retrieval energy level.
- **What this is, in our terms:** **Log-prior absorbed into softmax logits.**
  This is the simplest possible "prior steering" of a modern Hopfield network,
  requires no architectural change, no retraining, no different energy.
- **Calibration gap:** Δ = f_eff − f_obs decomposes as Δ_attn + Δ_PCA +
  Δ_argmax. Fisher separation index S predicts gap: S>0.3 ⇒ Δ≈0; S<0.2 ⇒
  Δ≈0.27-0.64. Linear: Δ ≈ 0.72 − 1.8·S, R²=0.80.
- **Why this matters for Phase 5:** The *substrate-pure* fix to the K-branch
  problem may be no more than adding `b_prior = log p_prior(atoms)` to the
  softmax logits. If the substrate has Fisher separation > 0.3, this will
  work cleanly. If <0.2, the problem is upstream (basin geometry) and adding
  log-prior bias won't help — which would be *itself* a useful diagnostic
  for Phase 5.

### 2.10 Input-Driven Plasticity (IDP) Hopfield
- **Citation:** Betteti, Baggio, Bullo, Zampieri — Sci. Adv. 2025
  ([arXiv:2411.05849](https://arxiv.org/abs/2411.05849);
  [Science Advances](https://www.science.org/doi/10.1126/sciadv.adu6991)).
- **Mechanism:** The synaptic matrix W(u(t)) becomes input-dependent:
  - `W(u(t)) = (1/N) Σ_μ α_μ(t) ξ_μ ξ_μ^T`
  - `α_μ(t) = ξ_μ^T u(t)` (saliency = input-pattern alignment).
  - Energy: `E(x; W(u)) = -½ Ψ(x)^T W(u) Ψ(x) + x^T Ψ(x) - Σ_i ∫_0^x_i ψ(z)dz`
- **The key result:** Memories with α_μ > α_existence=1 *exist*; with
  α_μ > α_stability > 1 are *stable*. Stability ordering follows the saliency
  ordering: `α_μ > α_ν ⟹ x_μ has deeper well`.
- **Why this is exactly what Phase 5 needs:** **Different inputs produce
  entirely different basin structures.** "A memory stable under one input
  may become a saddle point under another." This is the formal statement
  of prior-discriminable basin geometry, with a clean closed-form mechanism.
- **Sharp at scale?** Not directly addressed; classical-style capacity
  (~0.14N) noted but the IDP improvement is about *robustness under mixed
  inputs* rather than capacity. Combining IDP with modern Hopfield's
  exponential capacity is "future work" but mechanically straightforward
  (`W(u)` carries over to LSE/LSR).
- **Per-atom discriminability:** Direct from `α_μ > 1` existence threshold
  + ordering of basin depths. This is *exactly* the per-atom discriminability
  Phase 5 needs (f_i differentiated across atoms within a single retrieval).

### 2.11 Dynamic Manifold Hopfield Networks (DMHN)
- **Citation:** Li, Zeng, Xue, Feng — June 2025, rev March 2026
  ([arXiv:2506.01303](https://arxiv.org/abs/2506.01303)).
- **Mechanism:** "Contextual modulation dynamically reshapes attractor
  geometry, transforming a static attractor manifold into a *context-dependent
  family of neural manifolds*." Network interactions learned data-drivenly
  to "intrinsically deform the geometry of its attractor manifold across
  cues without explicit context-specific parameterization."
- **Capacity claim:** 64% accuracy storing 2N patterns in N neurons, vs
  1% (classical) and 13% (modern). This is a *2× over modern Hopfield*
  capacity gain via dynamic context-dependent geometry.
- **Why this matters for Phase 5:** This is the closest published precedent
  for "the substrate's attractor manifold itself changes with the prior."
  It's the architectural opposite of Phase 4's static post-graduation
  substrate. The trade-off — capacity goes *up*, basins are *family-of*
  rather than fixed — is the trade-off Phase 5 is implicitly asking for.
- **Caveat:** The exact mechanism is data-driven, not closed-form. Requires
  end-to-end training of the modulation. Phase 5's substrate-pure
  graduation requirement may not accommodate a learned modulation.

### 2.12 Provably Optimal Capacity — Spherical Codes (KHM)
- **Citation:** Hu, Wu, Liu — NeurIPS 2024
  ([arXiv:2410.23126](https://arxiv.org/abs/2410.23126)).
- **Mechanism:** Cast memory storage as a *spherical code* problem. Optimal
  capacity = optimal point arrangement on the unit sphere. Tight exponential
  scaling with d.
- **U-Hop / U-Hop+:** Two-stage retrieval — minimize separation loss to spread
  memories uniformly, then standard energy minimization. Provably optimal
  capacity in sublinear time.
- **Why this matters for Phase 5:** Phase 4's effective-dim-5-of-4096
  collapse is the *opposite* of optimal sphere-packing — it's a code with
  exponentially poor angular separation. Quantifying Phase 4 surviving
  atoms' sphere-packing quality (Fisher index / minimum pairwise distance)
  would tell you whether the K-branch foreclosure is a capacity-engineering
  failure (atoms not well-separated on sphere) or a basin-engineering
  failure (atoms well-separated but softmax-LSE energy doesn't expose the
  separation under priors).

### 2.13 Modern Hopfield with Continuous-Time Memories
- **Citation:** Santos, Farinhas, McNamee, Martins — Feb 2025
  ([arXiv:2502.10122](https://arxiv.org/abs/2502.10122), ICLR 2025).
- **Mechanism:** Compress L discrete memories into N basis-function
  coefficients, get a continuous probability density `p(t)` over a
  continuous memory domain. Attended output = ∫p(t)v(t)dt.
- **Why it's relevant:** Smoother basin geometry (single-mode Gibbs density)
  but loses multi-modality. Probably *not* the right direction for Phase 5
  — opposite of what's wanted.
- **Why I mention it anyway:** Shows that "smoothing the basin" by basis
  compression is *not* automatically good. The Phase 5 K-branch problem is
  about adding *modes*, not smoothing.

### 2.14 Associative Memory and Dead Neurons
- **Citation:** Fanaskov & Oseledets — ICLR 2025
  ([arXiv:2410.13866](https://arxiv.org/abs/2410.13866)).
- **Mechanism:** Identifies that ReLU-style activations in
  Krotov-Hopfield ODEs produce "dead neurons" — non-compact regions of
  flat energy where the energy alone cannot resolve the state. Proposes a
  modified system without flat directions but with the same steady-state
  structure.
- **Why it's *directly* relevant:** Phase 4 mass-death may have produced
  the exact pathology this paper formalizes. Effective dim 5/4096 with
  identical f_i across all surviving atoms is the symptom of a flat-direction
  basin pathology. The fix proposed (modified Lyapunov, same steady states)
  is a way to *get the basin sharpness back without the flat direction*.
- **Action item:** Read this paper carefully. If Phase 4's mass-death
  protocol can be re-derived as their "non-flat" modification, the K-branch
  problem may be solvable without giving up basin sharpness — by recovering
  the directions the dead-neuron problem ablated.

### 2.15 Liquid Hopfield (multi-mode retrieval)
- **Citation:** PNAS 2024 ([arXiv:2310.18853](https://arxiv.org/abs/2310.18853);
  [PNAS DOI](https://www.pnas.org/doi/10.1073/pnas.2320504121)).
- **Mechanism:** Multicomponent liquid-mixture analog of Hopfield. Stores
  *multiple pairs* of stationary retrieval and anti-retrieval phases
  simultaneously, retrievable by nucleation. Capacity *increases linearly*
  with number of components.
- **Why it matters:** Existence proof that a Hopfield-style energy can
  support multiple stably-retrievable phases, accessed by initial-condition
  "seeding." Generalizes the K-branch idea to "K-phase coexistence."
- **Caveat:** Physical analog, not directly a neural architecture.

### 2.16 Adaptive Hopfield Network (similarity-adaptive)
- **Citation:** Wang, Pan, Shen, Zhang, Wang, Li — Nov 2025
  ([arXiv:2511.20609](https://arxiv.org/abs/2511.20609)).
- **Mechanism:** Replaces fixed similarity (cosine / dot product) with a
  *learned multi-scale similarity footprint* `w^T ftpt_sim(ξ, x)`. Multiple
  base similarities weighted by learnable β_k.
- **Why it matters:** Shows that the *similarity function*, not just the
  separation, can be query-conditioned. Underexplored design knob for
  Phase 5.

### 2.17 Attractor-Keyed Memory (physical-key selector)
- **Citation:** Berloff — March 2026 ([arXiv:2603.17049](https://arxiv.org/abs/2603.17049)).
- **Mechanism:** Physical selectors (Ising machines, lasers, condensates)
  generate stereotyped high-dim signatures used as memory keys. Decoding
  fidelity vs routing reliability error decomposition.
- **Why it matters as inspiration:** Architecturally separates *selection*
  (basin choice) from *retrieval* (basin contents). Suggests that Phase 5
  could think of the prior-traversal mechanism as a separate selector
  pre-stage, with the Modern Hopfield as the retrieval back-end. This is
  similar to MoE routing.

### 2.18 Other notable references
- **Yet another exponential Hopfield model** —
  [arXiv:2509.06905](https://arxiv.org/abs/2509.06905). Another energy
  variant in the exponential-capacity family.
- **Generalised Hopfield with mismatched patterns** —
  [arXiv:2204.04520](https://arxiv.org/abs/2204.04520). Treats "fidelity"
  vs basin radius tradeoff for non-uniform patterns.
- **Hyper-SET (Hyperspherical Energy Transformer)** —
  [arXiv:2502.11646](https://arxiv.org/abs/2502.11646). Recurrent-depth
  ET on a sphere.
- **Non-linear Attention via MHN (Farooq, May 2025)** —
  [arXiv:2506.11043](https://arxiv.org/abs/2506.11043). Unifies attention
  and MHN; introduces "context wells" as multi-token basin objects.
- **NRGPT (energy-based GPT alternative)** —
  [arXiv:2512.16762](https://arxiv.org/pdf/2512.16762). Builds GPT-scale
  models on energy-minimization principles.
- **Transformers as Intrinsic Optimizers** —
  [arXiv:2511.00907](https://arxiv.org/html/2511.00907v2). Forward
  inference framed as energy descent.
- **Synaptic noise & accuracy/capacity of MHN** —
  [arXiv:2503.00241](https://arxiv.org/html/2503.00241). Empirical
  capacity under noise.
- **Geometric entropy & retrieval phase transitions in continuous DAM** —
  Petrova [arXiv:2604.07401](https://arxiv.org/abs/2604.07401). LSE vs LSR
  thermal phase boundaries.
- **Thermal Robustness LSE vs LSR** —
  [arXiv:2603.13350](https://arxiv.org/abs/2603.13350). Finite-temperature
  retrieval comparison.

---

## 3. Specific mechanisms for prior-traversal: which approaches expose "different priors → different basins"

This is the bottleneck question. Ranked by mechanism strength + closeness of fit
to Neuro-AI Phase 5's "substrate-pure D1 graduation + prior-discriminable"
requirement:

| Rank | Approach | Prior mechanism | Substrate-pure? | Closeness of fit |
| --- | --- | --- | --- | --- |
| 1 | **IDP Hopfield (2411.05849)** | `W(u)` synaptic modulation by input | Yes (no learning) | Direct — basin depths reorder per-input |
| 2 | **Pattern Multiplicity (2603.20115)** | `b_k = log r_k` added to softmax logits | Yes (no learning) | Direct — single scalar per pattern, log-prior |
| 3 | **Langevin Stochastic Attention (2603.06875)** | Bias `b` + intermediate β + multi-chain init | Yes (no learning) | Direct — K chains explore K basins |
| 4 | **Sparsemax / Sparse Hopfield (2402.13725)** | Hard masking via prior-shifted logits, support changes | Yes | Indirect — non-smooth |
| 5 | **LSR Emergent Minima (2506.10801)** | Different priors land in different overlapping-subset centroids | Yes | Indirect — depends on basin-overlap structure |
| 6 | **HEN (2409.16408)** | Prior is a query in a shared latent encoder space | No (learned encoder) | Indirect — works for cross-modal |
| 7 | **DMHN (2506.01303)** | Learned manifold modulation by context vector | No (learned modulation) | Direct mechanism but requires retraining |
| 8 | **Energy Transformer (2302.07253)** | Initial token configuration acts as prior; settling diverges | Partly (trained ET) | Configurations, not single patterns |
| 9 | **Attractor-Keyed Memory (2603.17049)** | Physical selector signature is the prior key | No (separate hardware) | Architectural inspiration only |

**Key insight:** The substrate-pure options (#1, #2, #3) are *all additive*
to Phase 5's existing FHRR + Modern Hopfield substrate, *none* require
retraining or new learned components, and they are *composable*: a Langevin
sampler at intermediate β with log-prior-biased logits and input-dependent
synapses is mathematically well-defined and would compound the three
mechanisms.

---

## 4. Settling dynamics literature

### 4.1 Temperature annealing / scheduling
Classical insight: at large β, minima are near single stored patterns; at low β,
minima are linear combinations. The natural prior-traversal protocol is
β-annealing: *start low (basins blurred, prior dominates), end high (basins
sharp, content dominates)*. This is the inverse of the Phase 5 protocol that
fixes β=10.

- The "Temperature-Dependent Phase Transition in Modern Hopfield Networks"
  paper ([arXiv:2311.18434](https://arxiv.org/pdf/2311.18434)) maps the
  phase diagram explicitly. Critical β where retrieval becomes reliable
  scales with √d. For d=4096 this is ~β=64; Phase 5's β=10 is *below* this
  threshold, but the basin sharpness at N=1064 makes it behave as if above it.

### 4.2 Mixed-time stochastic dynamics
- Adjusting Hopfield via time-variant stimulus
  ([arXiv:2402.18584](https://arxiv.org/pdf/2402.18584)) shows time-variant
  external stimuli induce multi-scroll / grid-multi-scroll attractor
  patterns. The relevance: Phase 5 settling could include a *time-varying
  prior* (anneal the prior in / out during settling) to expose basin
  geometry to the prior at the right phase.

### 4.3 Langevin / Kramers escape
- Kramers theory: escape rate from a basin scales as `exp(-ΔE/T)`. For
  Phase 5's near-equipotent basins (ΔE ~ FP precision floor), thermal
  noise of any non-trivial amplitude will randomize the chain — but at
  the right amplitude, it exposes basin geometry. This is the
  mechanism Langevin attention exploits.

### 4.4 Predictive coding inference
- Inference in PC networks
  ([arXiv:2407.04117](https://arxiv.org/html/2407.04117v1) survey;
  [arXiv:2408.11979](https://arxiv.org/html/2408.11979v1) energy landscape
  analysis NeurIPS 2024) is itself iterative energy minimization with
  top-down priors injected at each layer. "Internal representations
  in energy models differ depending on the prior when the input is
  ambiguous." This is the same machinery, applied at the architecture
  level rather than the layer level.

### 4.5 Parallel tempering / replica exchange
- CREPE ([arXiv:2509.23265](https://arxiv.org/abs/2509.23265)) and
  Generalised Parallel Tempering ([arXiv:2502.10328](https://arxiv.org/abs/2502.10328))
  use multiple temperature replicas with swap moves to explore multimodal
  energy landscapes. Could be ported to Hopfield settling: K replicas at
  different β with swap moves, each settling under its own prior.

---

## 5. Key papers (consolidated table)

| Year | Authors | Title (short) | arXiv | Relevance |
| --- | --- | --- | --- | --- |
| 2016 | Krotov & Hopfield | Dense Associative Memory | [1606.01164](https://arxiv.org/abs/1606.01164) | Interaction-order foundation |
| 2020 | Ramsauer et al. | Hopfield Networks is All You Need | [2008.02217](https://arxiv.org/abs/2008.02217) | Modern Hopfield baseline |
| 2021 | Krotov | Hierarchical Associative Memory | [2107.06446](https://arxiv.org/abs/2107.06446) | Multi-layer energy |
| 2022 | Millidge et al. | Universal Hopfield Networks | [PMLR v162](https://proceedings.mlr.press/v162/millidge22a.html) | Similarity-separation-projection frame |
| 2023 | Hoover et al. | Energy Transformer | [2302.07253](https://arxiv.org/abs/2302.07253) | Settling-based attention |
| 2024 | Santos et al. | Sparse and Structured Hopfield | [2402.13725](https://arxiv.org/abs/2402.13725) | Sparsemax retrieval |
| 2024 | Hu, Wu, Liu | Optimal capacity (Spherical Codes) | [2410.23126](https://arxiv.org/abs/2410.23126) | Capacity = sphere-packing |
| 2024 | Wu et al. | U-Hop ICML | [2404.03827](https://arxiv.org/abs/2404.03827) | Two-stage uniform retrieval |
| 2024 | Kashyap et al. | HEN | [2409.16408](https://arxiv.org/abs/2409.16408) | Encoder reshapes basin geom |
| 2024 | Santos et al. | Hopfield-Fenchel-Young | [2411.08590](https://arxiv.org/abs/2411.08590) | Unified sparse framework |
| 2024 | Betteti et al. | Input-Driven Plasticity Hopfield | [2411.05849](https://arxiv.org/abs/2411.05849) | W(u) reshapes basins per input |
| 2024 | Fanaskov & Oseledets | Associative Memory & Dead Neurons | [2410.13866](https://arxiv.org/abs/2410.13866) | Flat-direction pathology fix |
| 2025 | Santos et al. | Continuous-Time Memories | [2502.10122](https://arxiv.org/abs/2502.10122) | Smoothing (counterexample) |
| 2025 | Hoover et al. | LSR Epanechnikov | [2506.10801](https://arxiv.org/abs/2506.10801) | Finite-support + emergent minima |
| 2025 | Farooq | Non-linear Attention via MHN | [2506.11043](https://arxiv.org/abs/2506.11043) | Context wells |
| 2025 | Li et al. | Dynamic Manifold Hopfield (DMHN) | [2506.01303](https://arxiv.org/abs/2506.01303) | Context-dependent manifold family |
| 2025 | Wang et al. | Adaptive Hopfield | [2511.20609](https://arxiv.org/abs/2511.20609) | Learned similarity footprint |
| 2026 | Alswaidan & Varner | Stochastic Attention Langevin | [2603.06875](https://arxiv.org/abs/2603.06875) | Multi-chain prior-biased settling |
| 2026 | Varner | Pattern Multiplicity Conditioning | [2603.20115](https://arxiv.org/abs/2603.20115) | log r_k bias on softmax |
| 2026 | Berloff | Attractor-Keyed Memory | [2603.17049](https://arxiv.org/abs/2603.17049) | Selector / retrieval decomposition |
| 2026 | Petrova | Thermal Robustness LSE vs LSR | [2603.13350](https://arxiv.org/abs/2603.13350) | Finite-T phase diagram |
| 2026 | Petrova | Geometric Entropy & Phase Transitions | [2604.07401](https://arxiv.org/abs/2604.07401) | Geometric phase analysis |

---

## 6. Concrete ideas for Neuro-AI Phase 5

The user's CLAUDE.md requires (a) anti-homunculus filter — no `if X then do Y`
arbitration; (b) substrate-pure D1 graduation; (c) headline metric per
phase-5-unified-design.md:256-281 (ΔE between role-prior and content-prior). All
ideas below are *additive* to the existing Phase 4 substrate and stay within the
anti-homunculus filter.

### Idea A — Log-prior softmax bias (Varner 2026 port)
**Mechanism:** Replace `softmax(β X^T q)` with `softmax(β X^T q + log p_prior)`,
where `p_prior ∈ R^N` is a per-atom prior weight from the K-branch prior signal.
This is mathematically equivalent to absorbing the prior as `b_k = log p_prior,k`
in the energy `E = -lse_β(X^T q + log p) + ½‖q‖²`.
- **Cost:** O(N) per retrieval; trivial.
- **Substrate-pure?** Yes — no learning, no new components, no homunculus.
- **What it tests:** Whether Phase 5's K-branch foreclosure is a *softmax-
  insensitivity* problem (in which case this trivially solves it) or a
  *basin-geometry* problem (in which case it won't help and that's a key
  diagnostic). Either way, this should be the first experiment.
- **Diagnostic to compute:** Fisher separation index S on the substrate's
  current basins. If S>0.3 (Varner threshold), prior-bias will work. If <0.2,
  the substrate basin geometry itself needs surgery.
- **Anti-homunculus check:** The prior signal is an *additive bias* on the
  retrieval energy, not a control signal that decides among retrieval outputs.
  Passes.

### Idea B — Langevin K-chain settling at intermediate β
**Mechanism:** Run K=4 Langevin chains, each initialized from a different
prior-weighted starting point, settling under the Hopfield energy with isotropic
Gaussian noise at intermediate β (e.g. β≈10× lower than current). Capture
each chain's settled state.
- **Cost:** K× the per-step cost × T sampling steps; ~30× expensive but
  embarrassingly parallel.
- **Substrate-pure?** Yes — adds Langevin noise to existing retrieval, no
  new learned parameters.
- **What it tests:** Whether multi-basin geometry exists *at all* in the
  substrate but is being suppressed by deterministic high-β gradient descent.
  If chains converge to distinct attractors, K-branch is achievable; if all
  K chains converge to the same attractor, the basin geometry truly is
  collapsed.
- **Critical β estimate:** β* ~ √d = √4096 = 64 for entropy inflection;
  current β=10 is well below this, but Phase 4 sharpening has effectively
  raised the operational β. Worth measuring.
- **Anti-homunculus check:** Stochasticity is local geometric dynamics
  (Brownian motion on energy landscape), not an arbiter. Passes.

### Idea C — Input-driven plasticity port (IDP Hopfield)
**Mechanism:** Make the synaptic matrix `W(u)` depend on the prior signal:
- `W(prior) = Σ_μ α_μ(prior) ξ_μ ξ_μ^T`, with `α_μ(prior) = ⟨ξ_μ, prior⟩`.
- Different priors *reshape the energy landscape itself*, not just the query.
- **Cost:** O(N·D) per prior change; non-trivial but cacheable.
- **Substrate-pure?** Yes — closed-form mechanism, no learning.
- **What it tests:** The strongest form of the K-branch hypothesis — that
  the substrate's basins themselves should be reshapeable by the prior.
- **Key advantage:** Provably gives different basin depth orderings under
  different priors. Stability ordering `α_μ > α_ν ⟹ x_μ deeper`.
- **Anti-homunculus check:** Synaptic modulation is local geometric
  dynamics. The prior reshapes the landscape; the settling decides where
  the state goes. No arbiter. Passes.

### Idea D — Hybrid LSE-LSR energy with prior-conditioned β(x)
**Mechanism:** Replace softmax-LSE energy with a hybrid: LSR contributions
within local prior-conditioned support radii, LSE backbone elsewhere. The
prior selects which Epanechnikov support regions are "active" by widening
or narrowing `√(2/β(x))` per pattern.
- **Cost:** O(N·D) plus a per-pattern β; moderate.
- **Substrate-pure?** Yes — closed-form energy modification.
- **What it tests:** Whether *emergent local minima* (Hoover 2025
  Proposition 2) — the centroids of overlapping basin subsets — can serve
  as the K basins of Phase 5's K-branch.
- **Anti-homunculus check:** The hybrid energy is a geometric object; the
  prior is a parameter of the geometry, not a decider. Passes.

### Idea E — Decode-time top-K via Sparsemap / structured Hopfield
**Mechanism:** Use SparseMAP (from HFYN [arXiv:2411.08590](https://arxiv.org/abs/2411.08590)) as the
separation function instead of softmax. SparseMAP natively retrieves a
*combinatorial structure* (k-subset of patterns) rather than a single
pattern. The prior weight is absorbed as a structured-output bias.
- **Cost:** SparseMAP inference is more expensive than softmax (O(N log N)
  or O(N·K) depending on structure) but tractable for N=1064.
- **Substrate-pure?** Yes if SparseMAP is treated as an inference rule,
  not a learned module.
- **What it tests:** Whether the K-branch "candidate set" can be made the
  native output of the retrieval rather than something K-branch has to
  build by hand.

### Idea F — Pre-stage selector + Hopfield back-end (Attractor-Keyed analog)
**Mechanism:** Decompose the K-branch operation into (1) a prior-key selector
that produces K different *initial conditions* in the Hopfield basin, and
(2) the Hopfield retrieval as the back-end. The selector is a deterministic
function of the prior; the Hopfield is the existing substrate.
- **Substrate-pure?** Yes if the selector is a closed-form transformation
  (not a learned router).
- **Anti-homunculus check:** The selector is a *mapping*, not a decider.
  Each K-branch input gets its own initial condition; settling decides where
  it lands. Passes.
- **Connection to existing project:** This is conceptually close to "K-branch
  uses K different priors to construct K different queries before retrieval"
  — but more architecturally explicit about what's prior-side vs
  retrieval-side.

### Idea G — Energy-Transformer-style settling on the substrate
**Mechanism:** Treat the Phase 5 substrate as the storage layer of an
Energy Transformer-like settling loop. Multiple priors initialize multiple
joint configurations; each settles to a context well. This generalizes
K-branch from "K queries" to "K joint-state initializations + parallel
settling."
- **Cost:** T (settling iterations) × K (branches) × per-step cost; non-trivial
  but well-studied.
- **Substrate-pure?** Yes if no trainable components added.

### Idea H — Diagnostic: measure Fisher separation index + sphere-packing
**Not a mechanism, a measurement.** Before any of the above:
- Measure Phase 4 surviving atoms' (1) Fisher separation index per Varner 2026,
  (2) sphere-packing optimality per Hu et al. 2024 spherical-codes framing,
  (3) minimum pairwise angular distance.
- This will *empirically distinguish* between:
  - "Basins are well-separated on the sphere but softmax-LSE energy maps
    them to identical attractors under priors" (in which case Ideas A, B
    fix it without touching basin geometry).
  - "Basins are collapsed to a 5-dim subspace and there is no separation
    to expose" (in which case Ideas C-G are needed; the basin geometry
    itself must change).
- **This is the cheapest informative experiment in this list.** Run it first.

### Composing the ideas
The most ambitious-but-still-substrate-pure direction would be:
1. Add log-prior bias (A).
2. Anneal β down from current 10 to ~1 during settling (temperature schedule).
3. Run K=4 Langevin chains at the low-β phase (B).
4. Re-sharpen β to 10 at the end (deterministic snap).
5. Capture each chain's attractor; ΔE between role-prior and content-prior
   becomes the Phase 5 headline.

This compound protocol uses *only* mechanisms from published papers (Varner
2026 + Alswaidan-Varner 2026 + classical β-annealing), introduces *no* learned
components, and *no* homunculus. The Phase 4 substrate is unchanged. The only
new code is: log-prior bias, Langevin noise, β schedule, multi-chain bookkeeping.

---

## 7. Surprises

1. **The "prior as log-bias on softmax logits" trick is dead simple, recently
   formalized (Varner 2026), and has a closed-form calibration-gap predictor.**
   This should have been the first thing tried for K-branch. If it works,
   Phase 5 is mostly done. If it doesn't, the calibration-gap formula tells
   you which of (attention, PCA, argmax) is the failure mode. The geometric
   diagnostic — Fisher separation index — is a 30-line computation on the
   Phase 4 substrate.

2. **LSR / Epanechnikov energy has *emergent local minima as a desirable
   feature*, not a spurious-memory bug.** Modern Hopfield treats spurious
   minima as a pathology; LSR's Proposition 2 reframes them as a *menu of
   combinatorial-subset attractors*. This inverts the standard wisdom and is
   exactly the "discrete combinatorial menu" K-branch needs.

3. **Langevin attention at intermediate β gives *more* basin diversity from a
   single chain than deterministic attention does from many chains at
   high β.** This argues that Phase 5's "K=4 chains all converge identically"
   is partly an artifact of β being too high for the chains to *explore* —
   not (only) of the basins being collapsed.

4. **Input-driven synaptic plasticity has been a serious neuroscience-grade
   research result for 18 months** ([Science Advances 2025](https://www.science.org/doi/10.1126/sciadv.adu6991)),
   formally proves that different inputs produce different basin
   stability orderings, and *requires no learning* — but appears nowhere in
   the Phase 5 design discussion based on the CLAUDE.md / STATUS.md context
   provided. This is exactly the substrate-pure mechanism for prior-traversal
   that the project has been searching for.

5. **The 2024-2026 modern Hopfield literature has effectively decomposed
   into two camps:** capacity-engineering (spherical codes, U-Hop, LSR
   exponential capacity, HFYN sparsity) and basin-shape-engineering (IDP,
   DMHN, HEN, Langevin, multiplicity-bias). Phase 5's problem lives entirely
   in the second camp; Phase 4's mass-death lived entirely in the first.
   The conflict between the two camps is the project's central architectural
   tension, and it has been *separately* recognized in the literature.

6. **The "Dead Neurons" paper (Fanaskov-Oseledets ICLR 2025) is alarmingly
   on-the-nose for Phase 4's mass-death + Phase 5's effective-dim-5
   problem.** It identifies the *exact pathology* — flat non-compact energy
   regions where activations have saturated — and proposes a closed-form
   fix that *preserves steady-state structure*. This deserves a careful
   read for whether Phase 4 can be re-derived as a "non-flat" variant that
   doesn't ablate the basin geometry K-branch needs.

7. **DMHN's 2N-patterns-in-N-neurons at 64% accuracy vs Modern Hopfield's
   13%** is a 5× capacity gain *and* a context-dependent manifold geometry —
   the two desiderata Phase 5 is trying to reconcile. The cost is that the
   manifold is *learned*, not closed-form, so it's not substrate-pure. But
   it shows the two desiderata are *jointly achievable* in principle, which
   is itself a finding for Phase 5's design space.

8. **The "context wells" framing of Non-Linear Attention via MHN
   (Farooq 2025)** reframes basins as *multi-token joint configurations*
   rather than single-pattern states. This is the right abstraction for
   Phase 5 if K-branch is fundamentally about *joint role-content
   configurations* being the things that need different basins, not
   individual atoms.

---

## 8. Open research questions

These are unresolved in the literature and would each be a meaningful
experimental contribution if Phase 5 generated data on them.

1. **What is the Fisher separation index of Phase 4's surviving atom set?**
   This single number determines whether the substrate's K-branch foreclosure
   is fixable by log-prior bias alone (S>0.3 ⇒ yes) or requires basin-geometry
   surgery (S<0.2 ⇒ no). Cheap to compute. Should be the first measurement.

2. **Is the Phase 4 substrate sphere-packing-optimal (Hu et al. 2024)?**
   Effective-dim-5-of-4096 sounds like it isn't, but the metric depends on
   what "patterns" means (atom embeddings vs role-bound atoms). If atoms
   are near-optimal but their *images under role binding* are degenerate,
   that suggests the FHRR binding is collapsing the geometry, not the
   Hopfield substrate.

3. **Does β-annealing during settling expose K-branch discrimination on the
   current Phase 4 substrate?** The Langevin paper's β*∼√d=64 prediction
   for d=4096 says current β=10 is in a regime where multi-chain mixing
   should be *easy*, suggesting the chains' convergence to identical
   attractors is *not* from being trapped in deep basins but from something
   else (degenerate landscape).

4. **What does LSR's emergent-minima count formula predict for Phase 4's
   N=1064, D=4096?** Hoover 2025 Proposition 3 gives count ~Θ((M^(1/d) -
   λ^(1/d) + 1)^d) on a uniform grid. Plugging M=1064, d=4096, λ=appropriate
   support overlap, gives a *concrete prediction* for how many distinct
   attractors a LSR-port would expose. If this >> K=4, LSR-port has plenty
   of basins for K-branch; if it ~= 1, the substrate geometry is the
   problem, not the energy.

5. **Can the IDP `W(u)` construction be combined with LSE-style exponential
   capacity?** Betteti et al. note this as future work. Mechanically it's
   `W(u) = Σ_μ α_μ(u) ξ_μ ξ_μ^T` substituted into the LSE energy. The
   theoretical capacity vs prior-discriminability tradeoff for this hybrid
   is unsolved.

6. **Does the Dead-Neurons modified Lyapunov function (Fanaskov 2025) give
   back the directions Phase 4 mass-death ablated?** If yes, this would be
   the clean theoretical underpinning for an alternate Phase 4 graduation
   protocol that doesn't collapse to effective dim 5. If no, the
   modification's "same steady states" guarantee is a red herring for
   Phase 5's purpose.

7. **What is the Phase 5 substrate's *operational* β?** Phase 5 reports β=10
   but the basin sharpness suggests the *effective* β (in terms of where the
   substrate sits on Ramsauer's phase diagram) is much higher, post-Phase-4
   mass-death. A simple diagnostic: measure softmax entropy under random
   queries; map back to "what β on a fresh substrate would give this entropy."
   This diagnostic alone may resolve whether the K-branch foreclosure is a
   β-regime problem or a basin-geometry problem.

8. **Is the Phase 5 K-branch architecture better thought of as multi-token
   "context wells" (Farooq 2025) rather than per-pattern K attractors?**
   This is a framing question rather than a math question, but if reframed,
   then Energy Transformer's settling-of-joint-configurations machinery
   becomes the natural substrate, and the K=4 branches are different
   *initial joint configurations* rather than different *priors over a
   single retrieval.*

9. **Composability of mechanisms.** None of the surveyed papers compose
   log-prior bias + Langevin + IDP simultaneously. The hybrid is
   mathematically well-defined but its phase diagram is uncharted. This is
   a meaningful research opportunity if Phase 5 attempts it.

10. **Where in the basin-engineering literature should the
    headline-vs-drill-down distinction be enforced?** From CLAUDE.md:
    "each phase has one headline metric that defines whether the phase
    crossed its viability threshold." For Phase 5's prior-traversal, the
    headline is ΔE(role-prior vs content-prior). The literature offers
    several drill-down candidates: Fisher separation index, multi-chain
    diversity (Alswaidan), saliency-stability ordering (IDP), emergent-
    minimum count (LSR), context-manifold deformation magnitude (DMHN).
    *Which* of these best *explains* movements in ΔE is an empirical
    question Phase 5's experimental design should plan to answer.

---

## 9. Suggested reading order for the user

If time-limited, read in this order:

1. **Varner 2026 ([arXiv:2603.20115](https://arxiv.org/abs/2603.20115))** —
   the log-prior bias trick + calibration gap. 30-minute read. Highest
   chance of immediately suggesting a Phase 5 experiment.
2. **Betteti et al. 2024 ([arXiv:2411.05849](https://arxiv.org/abs/2411.05849))**
   — IDP Hopfield, W(u). 1-hour read. The mechanism Phase 5 has been
   looking for, with closed-form math.
3. **Alswaidan & Varner 2026 ([arXiv:2603.06875](https://arxiv.org/abs/2603.06875))**
   — Langevin attention. 1-hour read. Multi-chain prior-conditioned settling,
   no training required.
4. **Hoover et al. 2025 ([arXiv:2506.10801](https://arxiv.org/abs/2506.10801))**
   — LSR energy and emergent minima. 1-2 hour read. The most theoretically
   developed treatment of "engineered multi-basin geometry."
5. **Fanaskov & Oseledets 2024 ([arXiv:2410.13866](https://arxiv.org/abs/2410.13866))**
   — Dead Neurons. 1-hour read. May explain Phase 4's effective-dim-5
   pathology and offer a clean re-derivation.
6. **Li et al. 2025 ([arXiv:2506.01303](https://arxiv.org/abs/2506.01303))**
   — Dynamic Manifold Hopfield. 1-2 hour read. The most ambitious
   architectural reframing of the sharp-vs-soft tradeoff, but requires
   learned modulation.
7. **Santos et al. 2024 ([arXiv:2411.08590](https://arxiv.org/abs/2411.08590))**
   — Hopfield-Fenchel-Young unified framework. 2-3 hour read. The
   theoretical umbrella that explains *why* the kernel/separation/projection
   knobs are independent.

If only one hour available: read just Varner 2026, then run the
Fisher-separation-index diagnostic on Phase 4's surviving atoms.

---

## 10. Sources

- [arXiv:2008.02217 — Hopfield Networks is All You Need (Ramsauer et al. 2020)](https://arxiv.org/abs/2008.02217)
- [arXiv:1606.01164 — Dense Associative Memory (Krotov & Hopfield 2016)](https://arxiv.org/abs/1606.01164)
- [arXiv:2107.06446 — Hierarchical Associative Memory (Krotov 2021)](https://arxiv.org/abs/2107.06446)
- [PMLR v162 — Universal Hopfield Networks (Millidge et al. 2022)](https://proceedings.mlr.press/v162/millidge22a.html); [PMC 7614148](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/)
- [arXiv:2302.07253 — Energy Transformer (Hoover et al. 2023)](https://arxiv.org/abs/2302.07253)
- [arXiv:2402.13725 — Sparse and Structured Hopfield (Santos et al. ICML 2024)](https://arxiv.org/abs/2402.13725)
- [arXiv:2411.08590 — Hopfield-Fenchel-Young Networks (Santos et al. 2024)](https://arxiv.org/abs/2411.08590)
- [arXiv:2404.03827 — U-Hop Uniform Memory Retrieval (Wu et al. ICML 2024)](https://arxiv.org/abs/2404.03827)
- [arXiv:2410.23126 — Optimal Memory Capacity Spherical Codes (Hu et al. NeurIPS 2024)](https://arxiv.org/abs/2410.23126)
- [arXiv:2409.16408 — Hopfield Encoding Networks HEN (Kashyap et al. 2024)](https://arxiv.org/abs/2409.16408)
- [arXiv:2411.05849 — Input-Driven Dynamics for Robust Retrieval IDP Hopfield (Betteti et al. 2024)](https://arxiv.org/abs/2411.05849); [Science Advances 2025](https://www.science.org/doi/10.1126/sciadv.adu6991)
- [arXiv:2410.13866 — Associative Memory and Dead Neurons (Fanaskov & Oseledets ICLR 2025)](https://arxiv.org/abs/2410.13866)
- [arXiv:2502.10122 — Continuous-Time Memories (Santos et al. ICLR 2025)](https://arxiv.org/abs/2502.10122)
- [arXiv:2502.11646 — Hyper-SET Hyperspherical Energy Transformer (2025)](https://arxiv.org/abs/2502.11646)
- [arXiv:2506.10801 — Dense Associative Memory with Epanechnikov Energy (Hoover et al. 2025)](https://arxiv.org/abs/2506.10801)
- [arXiv:2506.11043 — Non-Linear Attention via MHN (Farooq 2025)](https://arxiv.org/abs/2506.11043)
- [arXiv:2506.01303 — Dynamic Manifold Hopfield Networks (Li et al. 2025)](https://arxiv.org/abs/2506.01303)
- [arXiv:2511.20609 — Adaptive Hopfield Network (Wang et al. 2025)](https://arxiv.org/abs/2511.20609)
- [arXiv:2603.06875 — Stochastic Attention via Langevin Dynamics (Alswaidan & Varner 2026)](https://arxiv.org/abs/2603.06875)
- [arXiv:2603.13350 — Thermal Robustness LSE vs LSR (Petrova 2026)](https://arxiv.org/abs/2603.13350)
- [arXiv:2604.07401 — Geometric Entropy & Retrieval Phase Transitions (Petrova 2026)](https://arxiv.org/abs/2604.07401)
- [arXiv:2603.17049 — Attractor-Keyed Memory (Berloff 2026)](https://arxiv.org/abs/2603.17049)
- [arXiv:2603.20115 — Conditioning via Hopfield Pattern Multiplicity (Varner 2026)](https://arxiv.org/abs/2603.20115)
- [arXiv:2407.04117 — Predictive Coding Survey (2024)](https://arxiv.org/html/2407.04117v1)
- [arXiv:2408.11979 — Energy Landscape of Predictive Coding Networks (NeurIPS 2024)](https://arxiv.org/html/2408.11979v1)
- [arXiv:2311.18434 — Temperature-Dependent Phase Transition in Modern Hopfield](https://arxiv.org/pdf/2311.18434)
- [arXiv:2402.18584 — Adjusting Hopfield via Time-variant Stimulus (2024)](https://arxiv.org/html/2402.18584v1)
- [arXiv:2310.18853 — Liquid Hopfield (PNAS 2024)](https://arxiv.org/html/2310.18853v5); [PNAS DOI](https://www.pnas.org/doi/10.1073/pnas.2320504121)
- [arXiv:2509.06905 — Yet another exponential Hopfield model (2025)](https://arxiv.org/abs/2509.06905)
- [arXiv:2503.00241 — Synaptic noise & capacity of MHN (2025)](https://arxiv.org/html/2503.00241)
- [arXiv:2509.23265 — CREPE Replica Exchange Diffusion (2025)](https://arxiv.org/abs/2509.23265)
- [arXiv:2502.10328 — Generalised Parallel Tempering (2025)](https://arxiv.org/abs/2502.10328)
- [arXiv:2406.12220 — Hierarchical Associative Memory + MetaFormer (Hoover et al. 2024)](https://arxiv.org/html/2406.12220v1)
- [github.com/deep-spin/HFYN — HFYN code](https://github.com/deep-spin/HFYN)
- [github.com/deep-spin/SSHN — Sparse Hopfield code](https://github.com/deep-spin/SSHN)
- [github.com/MAGICS-LAB/UHop — U-Hop code](https://github.com/MAGICS-LAB/UHop)
- [github.com/MAGICS-LAB/SparseModernHopfield — Sparse Modern Hopfield code](https://github.com/MAGICS-LAB/SparseModernHopfield)
