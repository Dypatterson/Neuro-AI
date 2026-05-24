# Binding Operators Beyond FHRR: A Phase 5 Brainstorm Brief

> **Angle:** Alternative binding operators with better signal-to-noise at
> moderate D (512–4096) for role-binding tasks. Surveying the operator
> zoology (TPR, HRR, FHRR, BSC, BSDC, MAP, VTB, MBAT, permutation, GHRR,
> HLB, MCR, sparse-block-codes, residue HD, attention-as-binding,
> Kuramoto-phase-binding) with attention to (a) per-atom variance / noise
> floor scaling with D, (b) capacity scaling, (c) compatibility with
> iterative settling / Hopfield-style cleanup, and (d) whether they have
> been tested under role-prior vs. content-prior bias dynamics like
> Neuro-AI's Phase 5 design.
>
> **Project context:** Neuro-AI runs FHRR at D=4096 and has hit a
> saturation finding: the role-prior vs. content-prior energy gap is 4.2×
> below the substrate's noise floor `(1/β)·log(1 + (N−1)·exp(−β·(1−1/√D)))`,
> driven by FHRR crosstalk `1/√D ≈ 0.0156`. The question for this brief
> is: what gives us a *better noise floor* (smaller crosstalk per atom,
> sharper role-filler discrimination, lower interference at moderate D)
> without throwing the whole substrate away?

---

## 1. Operator zoology — what's actually out there

The most authoritative recent taxonomy is Schlegel, Neubert & Protzel
(2021/2022), *A comparison of Vector Symbolic Architectures*, which
compares **eleven** VSA implementations along a shared experimental
protocol.

- arXiv: <https://arxiv.org/abs/2001.11797>
- Springer (Artif Intell Rev): <https://link.springer.com/article/10.1007/s10462-021-10110-3>

The eleven families they compare:

| # | Family | Vector space | Binding op | Self-inverse? | Commutative? |
|---|--------|--------------|------------|---------------|--------------|
| 1 | **TPR** (Smolensky 1990) | ℝᴰ → ℝᴰˣᴰ | outer product | exact (via inverse role) | no |
| 2 | **HRR** (Plate 1995) | ℝᴰ Gaussian | circular convolution | approximate (involution) | yes |
| 3 | **FHRR** (Plate, Fourier domain) | 𝕋ᴰ (unit complex phasors) | element-wise complex × | exact (conjugate inverse) | yes |
| 4 | **BSC** (Kanerva 1996, "Binary Spatter Code") | {0,1}ᴰ | XOR | yes (XOR is self-inverse) | yes |
| 5 | **MAP-C** (Gayler 1998 "Multiply-Add-Permute, Continuous") | ℝᴰ | element-wise × | approximate | yes |
| 6 | **MAP-B** (bipolar {-1,+1}ᴰ) | {-1,+1}ᴰ | element-wise × | yes | yes |
| 7 | **MAP-I** (integer mod p) | ℤₚᴰ | modular sum/product | depends on p | yes |
| 8 | **BSDC-CDT** (sparse binary, Rachkovskij 2001) | sparse binary | Context-Dependent Thinning | no exact inverse | yes |
| 9 | **BSDC-SHIFT** (Laiho 2015) | sparse binary, block-wise | block shift | yes (shift inverse) | no |
|10 | **VTB** (Vector-Derived Binding, Gosmann & Eliasmith 2019) | ℝᴰ | matrix-vector with derived matrix | approximate | no |
|11 | **MBAT** (Tissera & McDonnell 2014 / Gallant) | ℝᴰ | matrix-vector with random fixed matrix | yes if orthogonal | no |

Plus more recent additions outside Schlegel's 2022 cutoff:

| Family | Year | Refs |
|--------|------|------|
| **Sparse block-code binding** (block-wise circular convolution, Frady/Kleyko/Sommer) | 2021 | <https://arxiv.org/abs/2009.06734> |
| **Vector Function Architecture (VFA / FPE)** — Frady, Kleyko, Kymn, Olshausen, Sommer | 2021 | <https://arxiv.org/abs/2109.03429> |
| **Residue HD Computing** — Kymn et al. (Neural Computation 2025) | 2023 | <https://arxiv.org/abs/2311.04872> |
| **GHRR** (Generalized HRR; non-commutative, matrix-phase FHRR) — Yeung, Zou, Imani | 2024 | <https://arxiv.org/abs/2405.09689> |
| **HLB** (Walsh-Hadamard Linear Binding) — Alam, Oberle, Raff et al. | NeurIPS 2024 | <https://arxiv.org/abs/2410.22669> |
| **MCR** (Modular Composite Representation, full eval) | 2025 | <https://arxiv.org/abs/2511.09708> |
| **Kuramoto / phase-binding (KomplexNet)** | Feb 2025 | <https://arxiv.org/abs/2502.21077> |
| **PRISM** (phasor language model) | Dec 2025 | <https://arxiv.org/abs/2512.01208> |
| **Attention-as-binding** — Vector-symbolic perspective on transformers | Dec 2025 | <https://arxiv.org/abs/2512.14709> |
| **Category-theoretic VSA foundation** | Jan 2025 | <https://arxiv.org/abs/2501.05368> |

---

## 2. The noise-floor story across operator families

### 2.1 The FHRR baseline you're already running

FHRR vectors live on 𝕋ᴰ (componentwise unit-modulus complex phasors).
For two independent random phasor vectors **r₁**, **r₂** ∈ 𝕋ᴰ, the
expected cosine similarity is 0 with standard deviation `1/√D`:
this is the *crosstalk* that bounds capacity. At D=4096, σ ≈ 0.0156.

Plate's original HRR analysis (Plate 1995, *IEEE Trans. Neural
Networks*) shows that retrieval of a bound pair x*y under superposition
of N pairs gives expected cosine similarity to the correct filler ≈
**1/√N** with variance ~ **N/D** — capacity scales as O(D/log N) for a
fixed error rate.

- Plate 1995 PDF: <https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf>

Crucially, in **complex-valued FHRR**, retrieval inverse is exact
(elementwise conjugate), so all the noise after unbinding is interference
noise from the other bound pairs, not approximation noise. This puts
FHRR at the *theoretical Plate-Frady noise floor* — you cannot beat it
within the FHRR family at fixed D without changing the binding operator.

The Frady/Sommer information-theoretic analysis quantifies this:
*A theory of sequence indexing and working memory in RNNs* (Frady &
Sommer, 2018) derives optimal capacity bounds for VSA superposition,
showing crosstalk noise σ² ≈ K/D for K stored items at dimension D.

- Frady & Sommer 2018: <https://arxiv.org/abs/1803.00412>

**Implication for Neuro-AI Phase 5:** the `1/√D ≈ 0.0156` crosstalk
you've named as the substrate noise floor is the *fundamental*
Plate/Frady bound for FHRR. You are not blocked by a sloppy implementation
detail. You are blocked by the operator family. That is the most
important thing this brief has to say.

### 2.2 Operators that change the per-atom variance

There are four ways to actually move the noise floor:

1. **Reduce the variance per atom** — operators with lower σ at fixed D.
2. **Increase effective D for free** — sparse codes that pack more
   information per bit/component.
3. **Make the binding non-commutative** — so role-prior vs filler-prior
   asymmetry is *intrinsic* to the operator rather than something the
   substrate has to manufacture from settling dynamics.
4. **Switch to a fundamentally different binding regime** — TPR
   (exact, dimension-blowing) or attention-binding (learned, soft).

### 2.3 Per-operator SNR comparison (what we actually know)

#### TPR (Tensor Product Representation) — the gold standard for SNR

- Smolensky 1990: <https://www.sciencedirect.com/science/article/abs/pii/000437029090007M>
- *RNNs Implicitly Implement Tensor Product Representations* (McCoy et al. 2019): <https://arxiv.org/abs/1812.08718>

TPR binds with the outer product `r ⊗ f`. Unbinding is exact via the
inverse role vector, with **zero crosstalk** when role vectors are
orthogonal. The pathology is that representation dimension grows from
D to D² per bind level — this is what HRR/FHRR were invented to
compress away. But for a *fixed two-level role-filler structure*, TPR
has the best SNR of any operator. Recent neural-network work
(Attention-based Iterative Decomposition for TPR, 2024) shows TPR is
still actively used.

- Attention-based TPR Decomposition (2024): <https://arxiv.org/html/2406.01012v1>

#### Permutation binding (Recchia 2015 — the operator Neuro-AI shelved)

Recchia, Sahlgren, Kanerva & Jones (2015) — *Encoding Sequential
Information in Semantic Space Models: Comparing Holographic Reduced
Representation and Random Permutation*:

- <https://onlinelibrary.wiley.com/doi/10.1155/2015/986574>
- PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4405220/>

Specific numbers (D=2048, paired-associate retrieval):
- Random permutations (RP): **457 m-correct, SD=86**
- Circular convolution (HRR): **381 m-correct, SD=145**
- F(1,48)=11.85, **p=0.001**

The interaction between operator and dimensionality was not significant
(p=0.06) but the *trend* is that "RP's capacity drops off more slowly
than convolution's as dimensionality is reduced." Translation: **the
RP advantage holds across D=256..2048 but is most useful at moderate D**.

Why does permutation beat convolution? Two reasons:
1. RP is a *unary* operator: P_role(filler). There is no role-filler
   interference at all — the role is encoded in *which permutation*
   you apply, not in a vector being convolved.
2. RP preserves all the filler's variance exactly — no smearing across
   D positions.

The catch is also why Neuro-AI shelved it: RP makes role retrieval
deterministic given the permutation, but doesn't naturally support
*similarity-based* role queries (the role is a permutation, not a
vector you can interpolate). Most VSA work uses permutation alongside
HRR/FHRR for *sequence position* binding while keeping convolution for
content binding (e.g., random-indexing models).

#### HLB (Walsh-Hadamard Linear Binding) — the 2024 deep-learning-friendly winner

Alam, Oberle, Raff, Biderman, Oates, Holt — *A Walsh Hadamard Derived
Linear Vector Symbolic Architecture* (NeurIPS 2024):

- arXiv: <https://arxiv.org/abs/2410.22669>
- HTML: <https://arxiv.org/html/2410.22669v1>
- NeurIPS PDF: <https://proceedings.neurips.cc/paper_files/paper/2024/file/0525fa17a8dbea687359116d01732e12-Paper-Conference.pdf>
- Code: <https://github.com/FutureComputing4AI/Hadamard-derived-Linear-Binding>

Operator:
- **Binding:** `B(x, y) = x ⊙ y` (element-wise product on ℝᴰ)
- **Unbinding:** `B*(z, y) = z ⊘ y` (element-wise division)
- **Self-inverse?** No — but **exact recovery** when ρ=1:
  `B*(B(x,y), y) = x` exactly, no approximation.
- **Initialization (critical!):** MiND — Mixture of Normal Distributions:
  half from N(−μ, 1/D), half from N(+μ, 1/D). Keeps E[x]=0 but
  E[|x|]=μ to avoid div-by-zero.

Why it matters for Neuro-AI:
- **O(D)** binding/unbinding (vs FHRR's O(D log D) FFT).
- **Stable across binding depth**: norm `||B*(b_{t+1}, x_t)||₂ = √D`
  per iteration regardless of ρ (Figure 3). Other VSAs blow up or
  collapse. This is the most relevant property for iterative
  settling — and Neuro-AI's Hopfield retrieval loop is exactly an
  iterative-settling regime where binding stability matters.
- Cosine sim after ρ binds: φ ≈ 1/√ρ (same Plate scaling), but
  paper explicitly applies √ρ correction.
- Benchmarks on Mini-ImageNet privacy task: HLB **59.48%** vs HRR
  **40.99%**, VTB **45.81%**. Geometric mean across tasks: HLB
  **77.17%** vs MAP-C 70.89%, HRR 67.14%.
- Designed for differentiable systems — drops into autograd without
  the FHRR phasor-projection hack from Ganesan et al. 2021
  (<https://arxiv.org/abs/2109.02157>).

The **practical headline for Phase 5**: HLB has the same Plate noise
floor (1/√ρ) but doesn't suffer FHRR's complex-arithmetic numerical
instability and is much faster on autograd-friendly hardware. It is
*not a fundamentally lower noise floor*. But it is *cleaner under
iterative settling*.

#### GHRR (Generalized HRR) — non-commutative binding, the under-appreciated 2024 result

Yeung, Zou, Imani — *Generalized Holographic Reduced Representations*
(May 2024):

- arXiv: <https://arxiv.org/abs/2405.09689>
- HTML: <https://arxiv.org/html/2405.09689v1>

GHRR generalizes FHRR's element-wise scalar complex multiplication
(U(1) phases) to element-wise **matrix** multiplication on U(m) (m×m
unitary matrices). Setting m=1 recovers FHRR. With m>1, binding
becomes **non-commutative**: `r ∗ f ≠ f ∗ r` when the underlying
unitaries don't commute.

Why this is *exactly* relevant to Neuro-AI Phase 5:
- The role-prior vs. content-prior asymmetry that you need is
  *intrinsic to non-commutative binding*. With FHRR (commutative),
  the substrate has to manufacture asymmetry from prior strength /
  settling order. With GHRR, "this is a role, that is a content"
  is built into the operator.
- The paper proves (Corollaries 4.1.1, 4.1.2) that GHRR maintains
  quasi-orthogonality and the kernel structure of FHRR. So your
  Hopfield cleanup compatibility is preserved.
- Memorization capacity (Figure 10): higher with larger m — but the
  paper is unfortunately mostly qualitative, no head-to-head ΔE-style
  numbers vs FHRR.
- Trade-off: vectors are now D×m×m sized. At D=512, m=4 you spend
  the same memory as FHRR at D=8192 but get richer per-atom geometry.

#### MCR (Modular Composite Representation) — the 2025 BSC successor

Efficient HDC with MCR (Nov 2025): <https://arxiv.org/abs/2511.09708>
HTML: <https://arxiv.org/html/2511.09708v1>

Binding: `c_i = mod_r(h_i + u_i)` (component-wise modular sum, r is
the modulus, typically r=4 or 16).

Specific numbers:
- vs BSC: **+25.5%** decoding accuracy across sequence lengths 10–400.
- MCR-4 vs BSC at *identical dimensionality* (123 classification
  datasets): **+4.84%** mean accuracy.
- vs FHRR: approaches FHRR accuracy at **~1/4 the memory footprint**.
- MCR-4 (4 bits/component) at D=256 matches BSC at D=1024 in memory,
  beats it by +3.94%.

Implication: **at fixed memory budget**, MCR beats both BSC and FHRR
for capacity. The catch: it lacks an exact analytic inverse in the
FHRR sense — unbinding by modular subtraction works only if you know
the modulus structure exactly. For Hopfield cleanup it's fine but
for the role-prior energy gap analysis, FHRR's complex-phase geometry
is more directly compatible with energy-based settling.

#### Sparse block codes (Frady, Kleyko, Sommer 2021)

- arXiv: <https://arxiv.org/abs/2009.06734>
- PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12180425/>

Splits D into B blocks of size D/B, each block one-hot (sparse). Two
binding ops:
- **Block-wise circular convolution** — sparsity-preserving, has
  ideal capacity properties (the paper proves this).
- **General sparse binding** — lossy but compatible with non-block
  sparse codes.

Key claim: sparse block codes match HRR capacity at the same D but
with **much lower per-bit cost** — fewer active components means
fewer neurons firing in neuromorphic implementations. Implemented on
Intel Loihi (Renner & Sandamirskaya 2022): <https://www.sandamirskaya.eu/resources/ICONS_2022_Renner_VSA_Loihi_binding.pdf>

For Neuro-AI specifically: sparse block codes give you a way to push
effective D higher *without* increasing memory. At D=4096 dense, you
have 4096 phasors. At D=4096 block-sparse with B=64 blocks of 64
positions, you have 64 active components — the *information* capacity
is the same but the *crosstalk* depends on the active-set overlap,
not on 1/√D. This is one of the most promising routes to actually
moving the noise floor in your Phase 5 context.

#### VFA (Vector Function Architecture) + Fractional Power Encoding

- Frady, Kleyko, Kymn, Olshausen, Sommer 2021 — *Computing on Functions
  Using Randomized Vector Representations*: <https://arxiv.org/abs/2109.03429>

FPE encodes continuous values by raising a random base vector to a
fractional power: `E(x) = z^x` (in the Fourier domain, just phase
rotation by x·angle(z)). This induces a kernel over continuous data
and is what makes resonator networks and grid-cell models work.

For Neuro-AI: relevant if the role-prior vs content-prior question
extends to *continuous-valued* roles (positions in a sequence, time
indices, spatial coordinates). FPE gives you a smooth role-vector
manifold where the role-prior energy varies continuously, which might
make the ΔE gap more readable.

#### Residue HD Computing (Kymn et al., Neural Computation 2025)

- arXiv: <https://arxiv.org/abs/2311.04872>

Represents integers via residues modulo a set of co-prime bases, each
residue encoded as a separate hypervector. Algebraic operations
(addition, multiplication) become element-wise vector ops. Logarithmic
scaling: representing 10^6 distinct values uses ~6× the resources of
representing 10^3.

Probably orthogonal to Phase 5 directly, but a strong example of a
**factorizable** binding family — if your role-space is small (a few
hundred roles, say) but combined with many content fillers, residue
HD gives you compact storage with exact factorization via resonator
networks.

#### Kuramoto / phase-binding (KomplexNet, Feb 2025)

- arXiv: <https://arxiv.org/abs/2502.21077>

Each neuron has amplitude (feature identity) + phase (object
membership). Neurons representing the same object synchronize their
phases via Kuramoto dynamics. Binding = phase synchrony; unbinding =
phase clustering.

The paper proves Kuramoto dynamics minimize an energy function
`E(θ) = −Σ_{ij} w_ij cos(θ_i − θ_j)` — this is *literally* a Hopfield
energy with phase-based coupling. **Directly compatible with
energy-based settling.**

Why this is interesting for Phase 5: phase-binding gives you a
binding mechanism where "what binds together stays together" is a
property of the dynamics itself, not a metric you read off afterward.
The role-prior vs content-prior asymmetry could be encoded as a
priori different coupling strengths in the Kuramoto network. The
paper reports 10–15% gains over real-valued baselines on overlapping
digits with 5–10 Kuramoto iterations to convergence.

Caveat: this is built for visual feature binding, not arbitrary
role-filler structures. Translating to your substrate would require
significant work — but the *anti-homunculus check passes cleanly*
because there's no arbiter, just local oscillator dynamics.

#### Attention-as-binding (Dec 2025)

- arXiv: <https://arxiv.org/abs/2512.14709>

Reframes transformer attention as a *learned, soft* VSA: queries =
roles, keys = role-cues, values = fillers, attention weights = soft
unbinding. Architecturally recommends "explicit binding/unbinding
heads" with multiplicative role-filler interactions.

For Neuro-AI: not a drop-in replacement (it's a learning architecture,
not a substrate), but the analytical framing is useful — it gives you
a way to read off Hopfield retrieval as a soft VSA unbinding step,
which might surface ΔE-readable structure that pure FHRR analysis
misses.

#### Hrrformer (ICML 2023) — for context

- arXiv: <https://arxiv.org/abs/2305.19534>
- Code: <https://github.com/FutureComputing4AI/Hrrformer>

Replaces transformer attention with HRR binding/superposition.
O(TH log H) time, O(TH) space, 28× faster than Luna-256, 10× faster
convergence. The architecture *forces* role-filler separation
because positions are bound to tokens via HRR convolution, not
via positional embeddings + dot-product attention.

Not directly relevant to Phase 5 substrate work but a strong
existence proof that HRR-style binding scales to large sequence-
prediction tasks (which Neuro-AI explicitly is *not* aiming at) —
useful as a sanity check that VSAs work at meaningful scale.

---

## 3. Specific numbers and capacity scaling

Compiling the per-operator capacity headline numbers from the
literature:

| Operator | D=2048 paired-associate capacity | Self-inverse | Noise floor type | Hopfield-compatible? |
|----------|----------------------------------|--------------|------------------|----------------------|
| FHRR | ~381 (Recchia 2015) | exact (conj) | 1/√D per atom | yes (current) |
| HRR (real) | ~381 (Recchia 2015) | approx (involution) | 1/√D per atom | yes |
| Random Permutation | ~457 (Recchia 2015) | exact (P⁻¹) | none for unary | partially (role is non-vector) |
| HLB | comparable to HRR/VTB AUC | exact when ρ=1 | 1/√ρ but stable norm | yes, stable norm |
| TPR | exact (1.0 with orthog roles) | exact (inverse role) | 0 with orthog roles | yes if D² fits |
| BSC | ~Hamming 0.5 noise floor | yes (XOR) | 1/2 − 1/(2√D) | partially |
| MCR-4 | +25% over BSC at same D | inverse via mod-subtraction | depends on modulus r | yes |
| BSDC sparse block | matches HRR at lower active count | yes for SHIFT | sparsity-dependent | yes |
| GHRR (m=4) | improves on FHRR per Fig 10 | exact (matrix conj) | similar to FHRR but matrix-richer | yes |

### What this table says about the Phase 5 saturation finding

Your headline blocker is the FHRR `1/√D` crosstalk → `0.0156` noise
floor at D=4096, and the role/content ΔE gap sitting 4.2× below it.
The honest options to *move* that noise floor are:

1. **Permutation for role-binding** (Recchia +20% capacity at D=2048).
   Roles become permutations rather than vectors. Compatible with
   keeping FHRR for content binding.
2. **Sparse block codes** at the same D — moves you off the dense
   `1/√D` regime to a sparsity-dependent crosstalk that can be
   lower for the same memory budget.
3. **GHRR matrix-phase binding** — same D but per-atom
   discriminability scales with m (matrix dim). Crosstalk is no
   longer scalar `1/√D` but a matrix-norm equivalent that empirically
   memorizes more bound pairs.
4. **HLB Hadamard binding** — same fundamental Plate noise floor as
   FHRR but *stable norm under iterative settling*, which is the
   regime your Hopfield retrieval actually lives in. Fix the
   *iterative noise compounding*, not the per-step crosstalk.
5. **TPR for the role-filler core** — at moderate D (e.g., 256),
   D²=65536 stays manageable. Zero crosstalk for orthogonal roles.

Options 1, 3, 4 each have a specific argument for being the right
shelved-but-readable next step. Option 2 needs more substrate work.
Option 5 reopens the dimensionality-explosion debate that HRR
originally closed but is *the* right answer if you literally need
zero crosstalk.

---

## 4. Concrete ideas for Neuro-AI

### Idea A — Permutation for role-binding, FHRR for content (hybrid substrate)

Use random permutations as role operators (P_role₁, P_role₂, ...)
and FHRR convolution as content binding. This is the original
"random indexing" trick and works because permutations *don't
interfere with FHRR phases* — they just reshuffle positions.

- Direct test of the Recchia 2015 +20% finding in your substrate.
- Substrate retrain is *less* than full operator swap because you
  keep FHRR for half the binding logic.
- Role-prior energy becomes the energy of "which permutation
  family is active" which is naturally piecewise-discrete — could
  give a much sharper ΔE.

### Idea B — Reuse FHRR substrate at D=4096 but switch to GHRR with m=4

- GHRR has FHRR as the m=1 special case. The substrate code change
  is local to the binding routine.
- Non-commutativity means role-prior vs filler-prior asymmetry is
  *built into* the operator. The Phase 5 ΔE question changes from
  "can the substrate manufacture asymmetry through settling?" to
  "what does the matrix-binding asymmetry give us out of the box?"
- Memory cost: 16× per vector (m=4 → 4×4 = 16 matrix entries per
  position). At D=512 with m=4 you spend the same memory as FHRR
  at D=8192.

### Idea C — Switch to HLB for the iterative-settling regime

If your saturation is partly *iterative noise compounding* in the
Hopfield loop (each settling step adds a fraction of `1/√D` noise),
HLB's constant `||·||₂ = √D` under repeated unbinding directly
addresses that. Same Plate noise floor per step but stable across
iterations, which is the regime Hopfield retrieval lives in.

- O(D) binding/unbinding vs FHRR O(D log D) — also a speedup.
- Drops cleanly into autograd, removes the Ganesan 2021 projection
  hack.
- Numerical stability in 32-bit float (FHRR phasors accumulate
  irrational arithmetic error).

### Idea D — Sparse block codes at the same memory budget

D=4096 dense → D=4096 block-sparse with B=64 blocks. 64 active
components per vector instead of 4096. Crosstalk depends on active-
set overlap (binomial), not on `1/√D`. At very low activity ratio
the noise floor drops below `1/√D`.

- Compatibility with current Hopfield retrieval needs checking —
  modern Hopfield is energy-based and works on dense vectors. You'd
  need a sparse-Hopfield variant.
- Renner & Sandamirskaya 2022 show this on Loihi hardware.

### Idea E — TPR for the *test* of "does zero crosstalk fix it?"

A clean experiment: build a TPR variant of the role-prior/content-
prior test at D=64 (so D²=4096, same memory as FHRR at D=4096).
Run the same ΔE measurement. If the ΔE gap closes, you've proven
the saturation is *operator crosstalk*, not architectural. If it
doesn't, you've eliminated the operator hypothesis entirely.

This is the cheapest discriminating experiment in this brief —
it doesn't commit to a substrate change, it tests whether the
substrate-side question has any answer at all.

### Idea F — Kuramoto phase-binding as a *substrate-native* binding op

If you're willing to consider a structurally different substrate
(probably Phase 6+ work), Kuramoto phase synchronization is the
binding mechanism whose anti-homunculus check passes most cleanly
of anything in this survey. The "decision" of what binds with
what is *literally* a local oscillator coupling dynamic. No
arbitration module. Direct energy compatibility with your existing
Hopfield framework.

The translation cost from FHRR is high — phase-binding doesn't
factorize into FHRR primitives — but for the long-term
contextual-completion target it's worth knowing this exists.

---

## 5. Surprises

1. **The Plate `1/√D` noise floor is a *fundamental* property of
   dense complex-phasor binding, not an implementation detail.** Most
   of the engineering improvements (GHRR, HLB) preserve the same
   `1/√ρ` Plate scaling — they fix other things (non-commutativity,
   stability, speed) but not the per-atom crosstalk. This is the
   most important load-bearing finding for Phase 5.

2. **The only operators that *actually* break the `1/√D` regime are
   permutation (unary, no crosstalk), TPR (exact, dimension blows
   up), and sparse codes (crosstalk shifts to overlap-based).** GHRR
   gives more *capacity* but the same noise floor. MCR gives more
   capacity per bit but the same conceptual interference model.

3. **HLB has the best stability under iterative settling of any
   modern VSA.** This is the property Hopfield retrieval cares
   about, not raw single-shot SNR. The Neuro-AI substrate's
   iterative-settling regime makes HLB more relevant than HLB
   "comparable AUC to HRR" benchmark numbers suggest.

4. **GHRR's non-commutativity is exactly the architectural feature
   your Phase 5 design needs.** Role-prior vs content-prior asymmetry
   becomes structural rather than dynamical. This is under-cited in
   the broader VSA literature — most people think of GHRR as "FHRR
   with more capacity" rather than "FHRR with intrinsic role-filler
   asymmetry."

5. **Recchia 2015's RP+20% finding is the most concrete, replicable
   capacity-gain number in the entire VSA literature** and was
   correctly flagged in the 2026-05-20 brainstorm. It deserves to
   be the *first* drill-down on the Phase 5 saturation finding, not
   shelved.

6. **Kuramoto phase-binding is the only operator family whose
   anti-homunculus check passes by construction.** It is a local
   dynamic that emergently performs binding. Worth treating as a
   long-term substrate option even if Phase 5 doesn't commit to it.

7. **The "Attention as Binding" framing (Dec 2025) is more useful
   as a diagnostic lens than as an architectural recommendation** —
   it gives you a way to *interpret* what your existing Hopfield
   loop is computing in VSA terms, which might reveal whether
   role-prior and content-prior are doing separable work.

8. **None of the surveyed operators have been tested under explicit
   "role-prior vs content-prior bias" experiments.** Phase 5 is
   probing a regime the VSA literature has not directly mapped. The
   noise-floor papers are about *retrieval* SNR, not *prior-bias*
   SNR. This means there's no off-the-shelf comparator and Neuro-AI
   has to set the standard — that's a publishable contribution if
   the experiment is done cleanly.

---

## 6. Open questions worth more research

1. **What is the role-prior vs content-prior energy gap *structurally
   predicted* to be for each operator family?** The Plate noise floor
   gives single-shot retrieval SNR; deriving the analogous bound for
   prior-bias asymmetry would tell us *a priori* which operators
   stand a chance of beating Neuro-AI's 4.2× sub-noise gap.

2. **For GHRR specifically: does the non-commutativity give an
   intrinsic ΔE gap?** Quick paper-and-pencil derivation would tell
   us if GHRR has *any* hope of meeting the Phase 5 spec without
   substrate-side tricks. Worth doing before committing to a substrate
   change.

3. **What does "iterative settling under HLB" look like in
   simulation?** The HLB paper shows stable norm across binding depth
   in a feedforward sense. Whether the same stability holds in
   recurrent Hopfield cleanup is not directly tested. A 100-line
   simulation would answer this.

4. **Does the BSDC-SHIFT (block-shift) binding give an exact
   role-filler operator that's also self-inverse?** This is the
   "best of both worlds" candidate that Schlegel 2022 mentions but
   doesn't deep-dive. Frady/Kleyko/Sommer 2021 prove block-wise
   circular convolution has ideal capacity but the shift variant
   might be even better for role-binding specifically.

5. **Has anyone benchmarked "active drift" (the term from your
   PROJECT_PLAN) for these alternative operators?** Active drift is
   a Neuro-AI-specific stress test. Translating it to a generic VSA
   benchmark and running each operator family through it would be
   high-leverage if Phase 5 graduates and you want to publish.

6. **Is there a way to get TPR's zero-crosstalk at FHRR's storage
   cost?** Low-rank TPR (Smolensky's original "compressed TPR" via
   PCA or random projection back to ℝᴰ) might give an intermediate
   point with much better SNR than HRR/FHRR but without the full D²
   blowup. This direction has been under-explored since the late 90s.

7. **The category-theoretic VSA foundation (Schlegel et al. Jan 2025,
   <https://arxiv.org/abs/2501.05368>) proves element-wise binding
   is optimal under Kan extensions.** What does this say about
   GHRR's matrix binding? GHRR is *also* element-wise (just with
   matrix entries instead of scalars). Does the category-theoretic
   framework already give us a "best" binding within the matrix-
   entry generalization? This is a literature-search rabbit hole
   that might collapse the operator-design space significantly.

---

## 7. Key papers — full reference list

### Foundational / survey

- Plate, T. A. (1995). *Holographic Reduced Representations*. IEEE
  Trans. Neural Networks 6(3). PDF: <https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf>
- Smolensky, P. (1990). *Tensor product variable binding and the
  representation of symbolic structures in connectionist systems*.
  Artificial Intelligence. <https://www.sciencedirect.com/science/article/abs/pii/000437029090007M>
- Schlegel, K., Neubert, P., & Protzel, P. (2021). *A comparison of
  Vector Symbolic Architectures*. Artificial Intelligence Review.
  arXiv: <https://arxiv.org/abs/2001.11797>
- Frady, E. P., & Sommer, F. T. (2018). *A theory of sequence indexing
  and working memory in RNNs*. arXiv: <https://arxiv.org/abs/1803.00412>
- Kleyko et al. (2021). *A Survey on Hyperdimensional Computing aka
  Vector Symbolic Architectures*. ACM Computing Surveys. arXiv: <https://arxiv.org/pdf/2111.06077>
- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Greff, K. et al. (2020). *On the Binding Problem in Artificial
  Neural Networks*. arXiv: <https://arxiv.org/abs/2012.05208>

### Specific operator papers

- Recchia, G., Sahlgren, M., Kanerva, P., & Jones, M. N. (2015).
  *Encoding Sequential Information in Semantic Space Models:
  Comparing Holographic Reduced Representation and Random
  Permutation*. <https://onlinelibrary.wiley.com/doi/10.1155/2015/986574>
- Gosmann, J., & Eliasmith, C. (2019). *Vector-Derived Transformation
  Binding*. <https://compneuro.uwaterloo.ca/files/publications/gosmann.2019b.pdf>
- Frady, E. P., Kleyko, D., & Sommer, F. T. (2021). *Variable Binding
  for Sparse Distributed Representations: Theory and Applications*.
  arXiv: <https://arxiv.org/abs/2009.06734>
- Frady, E. P., Kleyko, D., Kymn, C. J., Olshausen, B. A., & Sommer,
  F. T. (2021). *Computing on Functions Using Randomized Vector
  Representations*. arXiv: <https://arxiv.org/abs/2109.03429>
- Ganesan, A. et al. (2021). *Learning with Holographic Reduced
  Representations*. NeurIPS. arXiv: <https://arxiv.org/abs/2109.02157>

### 2023–2025 operator advances

- Alam, M. M., Raff, E., et al. (2023). *Recasting Self-Attention with
  Holographic Reduced Representations* (Hrrformer). ICML. arXiv:
  <https://arxiv.org/abs/2305.19534>
- Kymn, C. J. et al. (2023, Neural Computation 2025). *Computing With
  Residue Numbers in High-Dimensional Representation*. arXiv:
  <https://arxiv.org/abs/2311.04872>
- Yeung, C., Zou, Z., & Imani, M. (2024). *Generalized Holographic
  Reduced Representations*. arXiv: <https://arxiv.org/abs/2405.09689>
- Alam, M. M., Oberle, A., Raff, E., Biderman, S., Oates, T., & Holt,
  J. (2024). *A Walsh Hadamard Derived Linear Vector Symbolic
  Architecture*. NeurIPS 2024. arXiv: <https://arxiv.org/abs/2410.22669>
- Clarkson, K. L., Ubaru, S., & Yang, E. (2024). *Capacity Analysis of
  Vector Symbolic Architectures*. arXiv: <https://arxiv.org/abs/2301.10352>
- *Efficient Hyperdimensional Computing with Modular Composite
  Representations* (Nov 2025). arXiv: <https://arxiv.org/abs/2511.09708>
- *Enhancing deep neural networks through complex-valued representations
  and Kuramoto synchronization dynamics* (KomplexNet, Feb 2025). arXiv:
  <https://arxiv.org/abs/2502.21077>
- *Language as a Wave Phenomenon: Semantic Phase Locking and
  Interference in Neural Networks* (PRISM, Dec 2025). arXiv:
  <https://arxiv.org/abs/2512.01208>
- *Attention as Binding: A Vector-Symbolic Perspective on Transformer
  Reasoning* (Dec 2025). arXiv: <https://arxiv.org/abs/2512.14709>
- *Developing a Foundation of Vector Symbolic Architectures Using
  Category Theory* (Jan 2025). arXiv: <https://arxiv.org/abs/2501.05368>
- *Holographic Global Convolutional Networks for Long-Range Prediction
  Tasks in Malware Detection* (Mar 2024). arXiv: <https://arxiv.org/abs/2403.17978>

### Hardware / neuromorphic implementations

- Renner, A., Sandamirskaya, Y. et al. (2022). *Sparse Vector Binding
  on Spiking Neuromorphic Hardware Using Synaptic Delays*. ICONS. PDF:
  <https://www.sandamirskaya.eu/resources/ICONS_2022_Renner_VSA_Loihi_binding.pdf>
- Renner, A. et al. (2024). *Neuromorphic Visual Scene Understanding
  with Resonator Networks*. Nature Machine Intelligence. arXiv:
  <https://arxiv.org/abs/2208.12880>
- Kleyko, D. et al. (2022). *Vector Symbolic Architectures as a
  Computing Framework for Nanoscale Hardware*. arXiv:
  <https://arxiv.org/abs/2106.05268>

### Code repositories

- Hadamard Linear Binding (HLB): <https://github.com/FutureComputing4AI/Hadamard-derived-Linear-Binding>
- Hrrformer: <https://github.com/FutureComputing4AI/Hrrformer>
- Learning with HRR: <https://github.com/FutureComputing4AI/Learning-with-Holographic-Reduced-Representations>
- HRR utilities (Alam): <https://github.com/MahmudulAlam/Holographic-Reduced-Representations>

---

## 8. Bottom line for Phase 5

The 4.2× sub-noise-floor finding is the *Plate fundamental limit* of
dense complex-phasor binding at D=4096. You cannot beat it within
the FHRR family. There are four honest routes off the limit:

1. **Permutation for role-binding** (Recchia 2015, +20% at D=2048,
   easiest substrate-side experiment).
2. **GHRR matrix-phase binding** (Yeung/Zou/Imani 2024, builds the
   role/content asymmetry into the operator, FHRR is the m=1 case).
3. **HLB Hadamard binding** (Alam et al. NeurIPS 2024, same Plate
   floor per step but stable under iterative settling — directly
   addresses Hopfield-loop noise compounding).
4. **TPR at D=64 as a discriminating test** (zero crosstalk at the
   same memory budget — tells you if operator crosstalk is the real
   blocker, before committing to a substrate change).

The *strongest* recommendation: run option (4) first as a 1-day
discriminating experiment. It costs almost nothing and tells you
whether you are operator-limited (→ pursue 1, 2, or 3) or whether
the saturation is architectural (→ Phase 5 needs to look elsewhere
entirely, and this brief has saved you from a Phase 6 detour into
operator-swap work that wouldn't have helped).
