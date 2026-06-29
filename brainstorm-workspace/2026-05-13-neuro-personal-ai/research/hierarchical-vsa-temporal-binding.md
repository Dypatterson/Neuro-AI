---
date: 2026-05-13
angle: hierarchical-vsa-temporal-binding
project: neuro-personal-ai
brainstorm-session: 2026-05-13-neuro-personal-ai
---

# Research Brief: Hierarchical VSA, Temporal Binding Primitives, and Structure Emergence

## Angle

This thread investigates whether temporal offsets are a viable VSA binding *primitive* —
specifically: has anyone proven that binding atoms with "before/during/after" roles (rather
than unordered co-occurrence bundles) preserves causal structure in a VSA substrate, and
what does the recent literature say about how hierarchical structure emerges from
consolidation in VSA systems?

The project currently uses temporal context *bags* at layer-2 discovery: unordered bundles
of atoms that co-occurred within a window. The hypothesis under investigation is that
replacing bag-of-atoms bundling with directed temporal-offset binding would preserve causal
ordering, not just co-occurrence statistics, and would be sufficient to bridge temporal
association and structural reasoning without adding a supervisor module.

Searches spanned: fractional power encoding for temporal axes, GC-VSA / grid-cell-inspired
VSAs, GHRR non-commutative binding, resonator networks, the NeurIPS 2024 hippocampal
compositionality paper, the "Attention as Binding" (2512.14709) transformer-VSA synthesis,
TiMem temporal hierarchy, SRMU streaming memory, and the Komer/Eliasmith SSP lineage.

---

## Key Findings

### 1. Fractional Power Encoding (FPE) on a Temporal Axis is Established — But Not Proven for Causal Structure

The Komer/Stewart/Eliasmith group (UWaterloo) established that FHRR fractional binding
extends naturally from spatial coordinates to any continuous dimension, including time.
The identity g(x) ⊙ g(y) = g(x+y) — carry-free additive group law in the phase domain —
means temporal offsets compose without decode/recode cycles. A base temporal vector **Vt**
raised to power *t* gives a distinct phase-encoded position; querying with **Vt^(ta - tb)**
is equivalent to asking "what was at offset *ta - tb* from here?"

The 2024 paper "Improved Cleanup and Decoding of Fractional Power Encodings"
(arxiv.org/abs/2412.00488) solved the degradation problem: without a cleanup mechanism,
bundled FPE vectors become unusable after two or three queries. The paper introduces a
coupled least circular distance (LCD) gradient ascent that converges in ~10 iterations
without any neural network. This is a practical unblocking result — FPE on a temporal axis
is now a viable substrate operation, not just a theoretical one.

**GC-VSA (arxiv.org/abs/2503.08608)** from 2025 implements exactly this: temporal
position is bound into scene representations using FPE on a dedicated temporal generator
vector, and multi-scale temporal structure is handled via grid-cell-inspired modules
with exponentially spaced scales (factor ~1.42). This is the closest analogue to
"temporal offsets as binding primitives" that exists in the recent literature — but it
encodes *when* an object was present, not causal directionality. It treats time as a
dimension, not as a relation.

**Sources:**
- FPE cleanup: https://arxiv.org/abs/2412.00488
- GC-VSA spatio-temporal: https://arxiv.org/html/2503.08608v1
- SSP foundations: https://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf

---

### 2. Non-Commutative Binding (GHRR) Addresses Ordered Relationships — But Not Temporal Semantics

Standard FHRR binding is commutative: A ⊙ B = B ⊙ A. This means co-occurrence bags are
naturally representable, but "A before B" and "B before A" produce identical vectors.
This is the core algebraic reason temporal offsets require either:
  (a) permutation/positional encoding (asymmetric by construction), or
  (b) a non-commutative binding operation.

**GHRR (arxiv.org/abs/2405.09689)** introduces flexible non-commutative binding while
preserving HDC desiderata (robustness, transparency, algebraic closure). It demonstrates
improved decoding accuracy and memorization capacity over FHRR for compositional
structures where order matters. The practical implication: if temporal direction matters
for an atom pair (A-caused-B is different from B-caused-A), GHRR binding is the correct
algebraic substrate. The paper does not test causal/temporal reasoning explicitly, but the
theoretical properties are the right ones.

**qFHRR (arxiv.org/abs/2604.25939)** shows that FHRR (including fractional binding for
continuous temporal positions) can be quantized to 3-4 bits per dimension with
integer-only modular arithmetic, matching full-precision performance. This is primarily
an efficiency result, but it confirms that the algebraic properties of FPE — including
fractional binding — survive aggressive quantization.

**Sources:**
- GHRR: https://arxiv.org/abs/2405.09689
- qFHRR: https://arxiv.org/abs/2604.25939

---

### 3. NeurIPS 2024: Hippocampal Compositionality via Residue Number System + Conjunctive Binding

**"Binding in hippocampal-entorhinal circuits enables compositionality in cognitive maps"**
(Kymn, Mazelet, Thomas, Kleyko, Frady, Sommer, Olshausen — NeurIPS 2024,
arxiv.org/abs/2406.18808) is the most directly relevant recent paper to the project's
architecture.

Key claims:
- Spatial position is encoded in a **residue number system (RNS)** across K modules with
  coprime moduli. Each modulus corresponds to a grid cell module in mEC.
- Binding is **component-wise complex-valued multiplication** (standard FHRR binding) on
  complex-valued high-dimensional vectors.
- The carry-free identity g(x) ⊙ g(y) = g(x+y) enables **path integration without
  decode/recode** — the attractor network enforces self-consistency between the composite
  position vector and its residue components.
- A **resonator network** factorizes composite representations back into residue
  components for recall.
- The model achieves **superlinear scaling** of coding patterns with dimension (exponential
  in K, linear storage) and noise robustness.
- Context binding: environment identity or event context is bound *into* the position
  representation via a separate context vector — this is how the same atoms (spatial
  positions) can mean different things in different contexts.

For the project, this paper is the biological/computational grounding for why the
FHRR binding + Hopfield retrieval combination is the right substrate: the hippocampal
formation literally uses this algebra. The RNS + multi-scale grid cell structure also
suggests that layer-2 hierarchy should use **exponentially spaced temporal scales** (like
GC-VSA) rather than fixed windows, because biological multi-scale encoding uses this
geometry.

**Sources:**
- Paper: https://arxiv.org/abs/2406.18808
- PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230348/
- Code: https://github.com/smazelet/Hippocampal_enthorinal_circuit

---

### 4. "Attention as Binding" (Dec 2025): Transformers Are Approximate VSAs

**arxiv.org/abs/2512.14709** argues that transformer attention implements approximate
VSA role-filler binding: queries/keys define role spaces, values are fillers, attention
weights perform soft unbinding, residual connections implement superposition.

Critical for the project:
- Sequential relationships in transformers are encoded via **permutation operators on
  positional roles**: `s = Σ π^i(pos) ⊗ x_i`. RoPE's rotations are identified as
  differentiable permutation operations compatible with VSA binding. This is a worked-out
  answer to "how do you encode temporal order in FHRR?" — use a cyclic permutation of the
  role vector at each offset step.
- CoT text is reinterpreted as externalization of a **trajectory through VSA-structured
  internal space**: `s^(t+1) = s^(t) ⊕ Σ_k r_k^(t) ⊗ f_k^(t)`. Each reasoning step is
  a superposition of new role-filler pairs.
- Proposed architectural enhancements include explicit binding/unbinding heads and
  hyperdimensional memory layers — directly analogous to what the project's Hopfield +
  FHRR substrate already does.

Implications for temporal offset binding: the paper does *not* use explicit "before/after"
role types. Sequence is handled via permutation. For the project's use case (binding
word-level atoms with temporal offset roles), permutation-as-position is the more
principled choice over named role vectors, because it scales to arbitrary offset depths
without adding role vocabulary.

**Source:** https://arxiv.org/abs/2512.14709

---

### 5. Resonator Networks: Hierarchical Factorization is Solved, Codebook Emergence is Active Research

Resonator networks (Frady, Kent, Olshausen, Sommer) solve the factorization problem: given
a composite VSA vector, recover its constituent atoms from a codebook via iterative
parallel search. The 2024 advance ("Recent Advances in Resonator Networks,"
openreview.net/forum?id=FNrZd3Ls1d) adds hierarchical factorization for non-commutative
transformations (visual scene understanding with translation + rotation handled in
separate partitioned sub-vectors), neuromorphic hardware implementation, and connections
to hippocampal-entorhinal cognitive map formation.

The December 2024 paper on noise in factorizers (arxiv.org/abs/2412.00354) addresses
**limit cycle failures** in resonator networks — a key practical issue — showing that
controlled noise perturbations in the codebook resolve stuck factorizations. This is
relevant: the project's Hebbian codebook updater creates an emergent codebook without a
pre-specified vocabulary. Resonator factorization assumes a known codebook; the project
must bootstrap the codebook *and* do factorization from the same dynamics.

No paper in the literature directly combines emergent (self-organizing) codebook formation
with resonator-based factorization. This appears to be an open problem.

**Sources:**
- Recent advances: https://openreview.net/forum?id=FNrZd3Ls1d
- Noise in factorizers: https://arxiv.org/abs/2412.00354

---

### 6. SRMU: Relevance-Gated Streaming Memory — Temporal Decay Without Temporal Structure

**SRMU (arxiv.org/html/2604.15121)** provides a single-level streaming update rule for
VSA associative memories combining temporal decay (γ·M) with novelty-gated writes
(w = 1 − cosine_similarity). It does not encode temporal offset binding or causal
structure — it is orthogonal to the question. Its relevance: SRMU is the cleanest
published formulation of how to handle streaming non-stationarity in a VSA memory
without a supervisor, which is directly compatible with the project's anti-homunculus
requirement. The update M_t = γ·M_{t-1} + w·(k_t ⊗ v_t) is a candidate primitive for
how consolidation-gating could work at layer-1 without introducing a decision module.

**Source:** https://arxiv.org/html/2604.15121

---

### 7. TiMem Temporal Hierarchy: LLM-Based, But the Tree Geometry Is VSA-Compatible

**TiMem (arxiv.org/html/2601.02845v1)** implements a five-level Temporal Memory Tree
(L1: dialog turns → L2: sessions → L3: daily → L4: weekly → L5: profiles) with the
formal property that parent intervals strictly contain child intervals. The consolidation
mechanism (LLM-prompted summarization up the tree) is not applicable to the project's
substrate, but the **temporal containment tree geometry** is independently useful:

- Temporal hierarchy as *containment* (not just timescale separation) gives a lattice
  structure that VSA RNS encoding naturally implements — each level's modulus defines
  a coarser temporal grain.
- The failure modes TiMem identifies ("temporal inaccuracy" and "temporal fragmentation")
  map precisely to what an unordered co-occurrence bag fails at: bags lose relative
  ordering (fragmentation) and collapse simultaneous/sequential (inaccuracy).

This is an indirect argument for why temporal-offset binding is worth the added complexity:
unordered bags are architecturally committed to both failure modes.

**Source:** https://arxiv.org/html/2601.02845v1

---

### 8. Human Hippocampal Time Cells (Nature 2024): Biological Temporal Offset Binding Exists

The 2024 Nature paper on human hippocampal/entorhinal temporal structure
(s41586-024-07973-1, reported in ScienceDaily/UCLA Health) found time cell populations
in the medial temporal lobe that fire at specific moments within a task, with the
*stability* of the time signal during encoding predicting temporal ordering accuracy at
retrieval. A separate "ramping cell" population in lateral entorhinal cortex provides
a slowly varying scalar temporal context across longer intervals.

Architectural implication: the brain appears to use two separate temporal signals —
discrete time cells (high-resolution, event-indexed) and ramping cells (slow,
continuous) — bound together with event content. This is a functional analog of:
binding atoms with a discrete temporal offset role (which event slot), and separately
with a slow consolidation signal (elapsed context since episode start). The project's
existing trajectory trace + Benna-Fusi consolidation may already capture the slow
signal; the fast discrete time-cell signal has no current analog.

**Source:** https://pubmed.ncbi.nlm.nih.gov/39322671/
(ScienceDaily coverage: https://www.sciencedaily.com/releases/2024/09/240925122844.htm)

---

## Promising Leads

1. **GC-VSA (2503.08608) full paper**: The abstract confirms multi-scale temporal
   representation and spatio-temporal querying. The full paper likely specifies exactly
   what equations are used for temporal fractional binding and how temporal query
   works (e.g., "what atom was at offset −3 from now?"). Worth reading in full before
   designing the temporal-offset binding layer.

2. **GHRR + temporal role combination**: GHRR gives non-commutative binding; applying it
   to directed temporal roles (BEFORE-role ⊙ atom_A, distinct from AFTER-role ⊙ atom_A)
   is not yet done in the literature. This is an open experiment.

3. **Resonator networks + emergent codebooks**: No paper has done resonator-style
   factorization on a self-organized (Hebbian-grown) codebook. If the project's layer-2
   binds atoms with temporal offset roles, resonator recovery requires knowing the
   temporal-role codebook entries. How does a learned codebook expose factorizable
   temporal structure to a resonator?

4. **Phase precession as temporal offset encoding**: The biological literature on phase
   precession in hippocampal place cells (theta phase varies monotonically with distance
   traversed) is a direct analog of FPE: the temporal axis is encoded as a phase angle.
   The Kymn et al. (2024) paper makes this connection explicit. This suggests that the
   project's FHRR phase encoding already has the right algebraic structure for temporal
   offsets — the question is whether to use the continuous (FPE scalar) or discrete
   (permutation-indexed) formulation.

5. **"Attention as Binding" architectural proposals**: The paper proposes explicit
   binding/unbinding heads as transformer add-ons (2512.14709 §5). For the project's
   Hopfield + FHRR substrate, the analogous proposal would be dedicated temporal-unbinding
   queries: given a settled state, query the temporal context bundle for the vector at
   offset k. This is a concrete architectural primitive worth specifying.

6. **Hebbian reservoir reshaping (Nature Communications 2025)**:
   nature.com/articles/s41467-025-67137-1 reports Hebbian Architecture Generation (HAG),
   an unsupervised rule that grows reservoir connections between co-active neurons. This is
   a closer functional analog to the project's Hebbian codebook than standard HDC.
   Worth checking whether HAG's connectivity emergence maps to the codebook-as-associative
   weight structure.

---

## Concrete Ideas

### Idea 1: Replace Temporal Context Bags with FPE-Encoded Offset Bundles

Instead of bundling co-occurred atoms into an unordered set, encode each co-occurrence
pair as: `atom_A ⊗ (Vt^Δt ⊗ atom_B)` where Δt is the signed offset (positive = B
comes after A, negative = B came before A). Bundle across the temporal window. This
preserves directional information while still operating with a single FHRR binding +
bundling operation. Querying: given atom_A, apply `atom_A*` and search for temporal
offsets using FPE decoding (now tractable via the LCD cleanup from 2412.00488).

Anti-homunculus check: the binding is purely local — each co-occurrence event writes
one role-filler pair. The directionality comes from the sign of Δt (a geometric property
of the temporal axis), not from a supervisor deciding what counts as causal. Passes.

### Idea 2: Discrete Permutation-as-Temporal-Position (Following "Attention as Binding")

Use the permutation encoding `π^k(pos) ⊗ atom` where k is the discrete slot offset
(−N, ..., 0, ..., +N) from the current anchor. This is the scheme the "Attention as
Binding" paper identifies in transformer positional encoding (RoPE). For the project's
context: each atom in the temporal window is bound with a permuted position role vector
for its slot. Bundling across all atoms gives a sequence-encoded hypervector that
supports unbinding by applying the inverse permutation π^{-k}.

This has a practical advantage over FPE: no continuous decoder needed, and the permutation
is exact (no approximation error). Disadvantage: only discrete offsets; doesn't naturally
generalize to arbitrary timescales.

### Idea 3: Two-Level Temporal Encoding (Time Cells + Ramp) Inspired by Biology

Following the human hippocampal data: bind each atom with two separate temporal
components:
- **Discrete slot role** (permutation at offset k within an episode): fine-grained,
  captures event order within a window.
- **Slow ramp context vector** (FPE on elapsed consolidation time): coarse, captures
  which consolidation epoch the atom belongs to.

Bundle these as: `π^k(slot) ⊗ ramp_context^t ⊗ atom`. The slow ramp is already
approximated by the Benna-Fusi slow variable u_m — the project could use the u_m
value as the exponent for a temporal ramp FPE vector at consolidation time.

Anti-homunculus check: both components are geometric measurements of dynamics (slot
index = local sequence counter, ramp = consolidation variable value). No supervisor
reads these and decides anything. Passes.

### Idea 4: RNS Multi-Scale Temporal Hierarchy (Following Kymn et al.)

Model the temporal hierarchy explicitly as a residue number system:
- Level 1 (layer-1 codebook): atoms at token/word grain — modulus m_1 = small
- Level 2 (phrase patterns): spans of ~5-15 tokens — modulus m_2
- Level 3 (episode/session): spans of minutes — modulus m_3

Each level's atoms are bound with the residue encoding at that level, and the composite
temporal address is the component-wise product of all three level encodings. Resonator
recovery can then factorize composite temporal addresses back into individual levels.

This requires pre-specifying the moduli — but biological grid cell modules use
exponentially spaced periods (~1.42× scale factor between adjacent modules), so
m_k = round(1.42^k · m_0) is a principled prior.

### Idea 5: SRMU as a Layer-1 Streaming Gate

Apply SRMU's update rule M_t = γ·M_{t-1} + w·(k_t ⊗ v_t) as the write rule for the
layer-1 Hopfield landscape, where:
- k = atom hypervector, v = temporal-context bundle
- w = 1 − cosine_similarity(M·k, v) — suppresses redundant temporal context updates
- γ = slow decay (e.g., 0.999) preserves history

This gives temporal recency weighting and novelty gating without any decision module —
both are pure geometric operations on the current memory state. The decay handles the
"stale information" failure mode; the novelty gate handles redundancy. Both are needed
for a streaming personal AI substrate.

---

## Surprises

### Surprise 1: No Paper Has Used Temporal Offsets as VSA Binding Roles Directly

After searching extensively, there is no published paper that explicitly creates VSA role
vectors labeled "before-2", "before-1", "during", "after-1", "after-2" and binds atoms
with these temporal semantic roles. The closest approaches are:
- FPE on a temporal axis (treats time as a continuous coordinate, not a named role)
- Permutation-indexed positions (treats temporal slot as an unnamed index)
- GHRR non-commutative binding (orders matter algebraically, but roles are not named)

This is both a gap and an opportunity. The project is designing something that has not
been explicitly validated in the VSA literature. The FPE and permutation results provide
the algebraic foundation, but the specific claim that named temporal roles ("BEFORE-role"
vs. "AFTER-role") are a better binding primitive than anonymous permutations is untested.

### Surprise 2: NeurIPS 2024 Hippocampal Binding Paper (Kymn et al.) Uses the Exact Algebra the Project Already Has

The most technically relevant 2024 paper turned out to model the biological system the
project is drawing architectural inspiration from (hippocampal-entorhinal circuits) and
uses component-wise complex multiplication as binding, a resonator network for
factorization, and RNS multi-scale encoding — all of which are either already in the
project's substrate or are natural next steps. The paper is essentially a normative
computational model of what the Hopfield + FHRR combination is doing at the systems
level. This is strong evidence that the architectural direction is correct.

### Surprise 3: Emergent Codebook + Resonator Factorization is an Open Problem

The resonator network literature assumes a known codebook. The project's Hebbian codebook
grows through self-organization. These two assumptions are in direct tension: resonator
factorization needs codebook vectors to search over; a Hebbian codebook does not
pre-commit to a fixed vocabulary. No paper has bridged this gap. The implication: if the
project wants resonator-style hierarchical factorization at layer-2, it may need to
periodically snapshot the emergent codebook as the resonator's search space, and
update that snapshot on a slow timescale. This is a clean Phase 5/6 design question.

### Surprise 4: "Attention as Binding" Reframes CoT as VSA Trajectory — Directly Relevant to Personal AI

The paper's interpretation of chain-of-thought as externalization of a trajectory through
VSA-structured internal space suggests a design connection: if the project stores temporal
sequences as FPE-encoded role-filler bundles, then a settling trajectory in the Hopfield
landscape is computing the same thing as a CoT reasoning trace — just internally. This
bridges the "personal AI companion" use case with the formal substrate: the memory system
is not a static lookup but a compositional reasoning substrate that externalizes its
trajectories as language when queried.

### Surprise 5: Temporal Binding Literature Is Split Between "Time as Coordinate" and "Time as Role"

The SSP / GC-VSA / fractional binding literature treats time as a continuous coordinate
(a dimension like x, y, z). The cognitive architecture / semantic role labeling literature
treats temporal relations as named roles (BEFORE, AFTER, DURING, WHILE). These two
traditions have not been bridged in the VSA context. Coordinate-time gives smooth
generalization (nearby times are similar vectors) but loses semantic content
(there's no distinction between "happening simultaneously" and "causing"). Role-time gives
semantic clarity but requires learning or pre-specifying the role vocabulary.

For the project's use case (personal AI that needs to reason about "A caused B" vs.
"A happened before B"), a hybrid may be needed: an anonymous continuous temporal axis
for encoding and retrieval, plus a small set of learned relational roles for reasoning.
The atomic-level layer-1 should use the continuous axis; the layer-2 discovery should
learn whether consistent temporal co-occurrence patterns warrant promoting to a named
relational role.

---

## Source Index

- GC-VSA grid cell structured vector algebra: https://arxiv.org/abs/2503.08608
- FPE cleanup and decoding: https://arxiv.org/abs/2412.00488
- Generalized HRR (GHRR) non-commutative binding: https://arxiv.org/abs/2405.09689
- qFHRR quantized FHRR: https://arxiv.org/abs/2604.25939
- NeurIPS 2024 hippocampal compositionality (Kymn et al.): https://arxiv.org/abs/2406.18808
- NeurIPS 2024 hippocampal compositionality PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11230348/
- Resonator network recent advances: https://openreview.net/forum?id=FNrZd3Ls1d
- Noise in factorizers: https://arxiv.org/abs/2412.00354
- Attention as Binding (VSA + Transformer): https://arxiv.org/abs/2512.14709
- SRMU streaming hyperdimensional memory: https://arxiv.org/html/2604.15121
- TiMem temporal-hierarchical memory consolidation: https://arxiv.org/html/2601.02845v1
- LARS-VSA abstract rule learning: https://arxiv.org/abs/2405.14436
- Assembling hierarchical cognitive map learners: https://arxiv.org/abs/2404.19051
- SSP fractional binding foundations (Komer et al.): https://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf
- Human hippocampal temporal structure (Nature 2024): https://pubmed.ncbi.nlm.nih.gov/39322671/
- VSA survey Part I: https://arxiv.org/abs/2111.06077
- VSA survey Part II (applications, cognitive models): https://dl.acm.org/doi/10.1145/3558000
- Hebbian reservoir reshaping: https://www.nature.com/articles/s41467-025-67137-1
- Modern Hopfield continuous-time memories: https://arxiv.org/abs/2502.10122
- Hopfield + encoded representations (HEN): https://arxiv.org/abs/2409.16408
