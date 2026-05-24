# 05 — Tolman-Eichenbaum Machine & Hippocampal Relational Memory

## Angle

Phase 5's headline failure is structural: the substrate does not contain
role-target attractors, so the role prior never has a basin to fall into.
TEM (Whittington et al., 2020 Cell) is the canonical computational model
that *explicitly* separates structural / relational representations
(medial entorhinal `g_t`) from sensory / content representations
(lateral entorhinal `x_t`) and binds them in the hippocampus. The
research question for this angle: does TEM (and its successors) tell us
*mechanistically* what is missing from the emergent FHRR codebook such
that role-cued retrieval cannot find a distinct basin — and is there a
bridge from "TEM is a generative model trained end-to-end with BPTT"
to "this is a local Hebbian update in an FHRR codebook"?

The short answer is *yes, with one major caveat*. TEM's binding scheme
is mathematically equivalent (modulo compression) to what the project
is already doing with FHRR. What TEM has that the project lacks is
(a) a *learned generative prior over `g`* that is updated by a
prediction-error signal across episodes, and (b) explicit *path-
integration / transition dynamics* on `g` that make the structural
embedding manifold separable from the sensory one without needing
"role basins" to be carved by Hebbian consolidation alone.

---

## Key findings

### 1. TEM's factorization is *learned*, not assumed.

Per Whittington et al. 2020 (Cell), TEM has two streams:

- `g_t` — abstract location / role, updated by an action-conditioned
  recurrent transition: `g_t = f(g_{t-1}, a_t)` where `f` uses
  action-specific learned weights `W_a`. This is path integration on
  arbitrary graphs.
- `x_t` — sensory / content, passed through a learned LEC filter
  (approximate Laplace transform).
- `p_t` — hippocampal conjunction, formed as the **outer product**
  (= tensor product) of `g_t` and `x_t`.

The loss is sensory prediction: maximize `p_θ(x_{≤T})` over
trajectories, optimized via an amortized variational ELBO with BPTT.
The KL term in the ELBO regularizes the *inferred* posterior over
`g_t` toward a *generative* prior over `g_t` derived from the path-
integrated transition. **This KL is what enforces the factorization**:
the model is penalized for needing sensory information to predict
`g_t` when path integration already explains it.

### 2. Hippocampal memory `M` is a Hebbian Hopfield store of conjunctions.

Once `p_t` is formed, it is written into a Hebbian weight matrix `M`
with the classical outer-product update `ΔM ∝ p_t p_tᵀ`. Retrieval
indexes `M` with a partial cue (sensory `x` or path-integrated `g`),
runs attractor dynamics, and recovers the full `p`. The attractor
network *is* a Hopfield network. This is the closest TEM analog to
the project's MHN.

Crucially: TEM's "structural retrieval" — using a `g`-cue to recover
`p` and then read out a predicted `x` — works *because `g` was
optimized end-to-end to be a useful key into `M`*. The role-target
basins exist because the upstream variational training shaped `g`
so they would exist.

### 3. The FHRR ↔ TEM mapping is closer than it looks.

The literature confirms (Generalized HRR, arxiv 2405.09689 and
references) that FHRR binding is **a holographic projection of the
tensor product**. TEM uses the full outer product because it can
afford the dimensionality. FHRR uses element-wise multiplication on
complex unit vectors as a compressed surrogate. Algebraically:

- TEM hippocampus: `p = vec(g xᵀ)` ∈ ℝ^{N_g · N_x}
- FHRR codebook atom: `p = g ⊛ x` ∈ ℂ^D

Both are bilinear in `(g, x)`. Both unbind by an approximate inverse
(`g⁻¹ ⊛ p ≈ x` in FHRR; pseudo-inverse / cleanup in TPR). The
algebra is *not* the bottleneck.

### 4. The 2022 paper "Associative memory of structured knowledge"
(Kymn / Stewart et al., Sci Reports) is the most direct precedent.

It stores role-filler structures in a Hopfield via **HRR-style
circular convolution** binding, then uses **pseudo-inverse**
(not Hebb) learning to write fixed-point attractors. It reports
SNR ∼ N/L and storage capacity α ≈ 0.1–0.3. Partial role-only cues
recover the full structure — explicitly, "extracted items come from
pairs that are not part of the cueing structure." This is exactly
the property Phase 5's role-prior branches *fail* to demonstrate.

The Phase 5 substrate uses **Hebb-style accumulation, not pseudo-
inverse**, and the MHN softmax retrieval is closer to dense Hopfield
than to the classical Hebb-Hopfield analyzed by Kymn. This may be
the load-bearing gap — see Idea 4 below.

### 5. Successor representations (Stachenfeld, Botvinick, Gershman
2017, Nature Neuro) give a third angle.

The SR encodes each state by its time-discounted future occupancy.
Importantly, SR can be learned by a **purely local TD rule**, and
neural-learning-rule papers (Bono et al. eLife 2022) show that
spike-timing-dependent plasticity with theta phase precession produces
SR-like firing fields. The SR is *intrinsically relational* — it
encodes structure of transitions, not content. If atom occupancy
under replay were used as the substrate's structural feature, the
relational basin would emerge from transition statistics rather than
needing to be carved by binding alone.

### 6. The 2025 Bakermans/Whittington/Behrens Nature Neuro paper
("Constructing future behaviour…") shows the modern TEM frontier:
hippocampal compositional memories where new state spaces are
*composed* from learned primitives, and *replay strengthens the
remote fields of those compositions*. Replay is no longer just
consolidation — it is the mechanism by which new compositions
acquire spatial / relational embedding.

### 7. TEM failure modes when factorization breaks.

Reported (or implied) failure modes:

- **Insufficient action diversity** — if `g` is never updated
  by a meaningful transition, it collapses onto `x` because
  the only signal shaping it is sensory prediction.
- **No KL regularizer** — without the prior matching, `g` and
  `x` representationally merge.
- **Single-frequency / single-module entorhinal** — without
  diverse `g` scales, remapping degrades and basin separability
  collapses.
- **No replay** — Bakermans 2025 shows replay is needed to
  carve remote fields for novel compositions.

### 8. Anti-homunculus audit of TEM.

TEM does have a "generative model vs inference model" split, which
*looks* like arbitration. But inference at runtime is not an
arbiter — it is the forward pass through the inference network,
plus attractor settling in `M`. There is no `if` rule. The
"decision" of whether `g` or `x` dominates is a balance of two
local gradients (KL on `g`, reconstruction on `x`) baked into the
weights at training time. **This passes the anti-homunculus filter.**

---

## Promising leads

- **Replay-driven update of `g` (Bakermans 2025).** Replay is the
  mechanism that carves remote relational fields. The project's
  consolidation pass currently reinforces atoms by frequency, not by
  participation in *transitions*. Replay as transition-conditioned
  reinforcement is unexplored.
- **Pseudo-inverse / Storkey-style storage (Kymn/Stewart 2022).**
  Hebb-only storage gives the wrong basin structure for role-filler
  retrieval. Modern Hopfield with softmax may be *too smooth* — the
  retrieval logit landscape is shaped by content similarity and
  effectively masks role similarity.
- **Action / transition variables in the codebook.** TEM's `g` is
  updated by `W_a g + b`. Phase 5 has no analog. Without a transition
  generator, there is no path-integration signal that would shape
  `g` away from `x`.
- **Two-stream codebook with KL coupling.** A `g`-codebook and an
  `x`-codebook with explicit KL pressure that `g` should be
  predictable from previous `g` *without* needing `x`. This is the
  TEM ELBO term, and it is the load-bearing inductive bias.
- **Successor-feature-style replay.** Aggregate replay co-occurrence
  matrix to define a structural feature per atom; bind that with
  content to form the conjunction.

---

## Concrete ideas for the project

Each carries an anti-homunculus check.

### Idea A — Add a transition operator `T_a` to the codebook.

**Mechanism.** Maintain a small bank of FHRR transition vectors
`{T_a}` (e.g. for "next-token", "same-window-different-position",
"role-shift"). Update an internal `g_t` slot by `g_t = T_a ⊛ g_{t-1}`.
Bind `p = g_t ⊛ x_t` as the consolidated atom. Store `p` in MHN.

**Why it should produce role basins.** `g_t` now lives on a manifold
parametrized by composed transitions; role-cued retrieval uses `g_t`
directly (or via a different `T_a` chain). The structural embedding
manifold is *geometrically distinct* from the content manifold
because it is generated by a different orbit.

**Anti-homunculus check.** No supervisor decides whether to apply
`T_a`. The transition is applied unconditionally on every step; the
codebook update is Hebbian on `p`. Retrieval is energy-only. Pass.

**FHRR + MHN + Hebbian fit.** All three are preserved. Costs one
extra FHRR multiply per step, no new modules.

### Idea B — Replace pure-Hebb consolidation with a Storkey-style
local correction.

**Mechanism.** When writing atom `p_i` into MHN weights, use the
local Storkey rule `ΔW_ij ∝ p_i p_j − p_i h_j − h_i p_j` where
`h = W p`. This is still local and Hebbian-in-form, but it carves
*orthogonal* basins instead of correlated ones.

**Why it should help.** Kymn et al. show that role-filler retrieval
needs basins that are *separated* in the energy landscape; Hebbian
storage gives basins with correlation = inner product, which is
exactly what makes role-cued retrieval collapse to the most
content-similar attractor (the Phase 5 failure mode).

**Anti-homunculus check.** Storkey is a local synaptic update, not
an arbitration. Pass.

**FHRR + MHN + Hebbian fit.** Storkey is widely characterized as
"Hebbian with a self-correction term"; it counts as Hebbian for the
purposes of the project's "no online error-driven" rule. Worth
explicit user agreement.

### Idea C — Successor-feature replay channel.

**Mechanism.** During post-death consolidation, build a co-occurrence
matrix `C` of atoms across replayed sequences. For each surviving
atom, compute a successor feature `s_i = Σ_j γ^{k(i,j)} C_{ij} c_j`
where `c_j` is the atom's content vector. Bind `s_i` with the atom
to form the consolidated form `p_i = s_i ⊛ c_i`.

**Why it should produce role basins.** `s_i` encodes the *relational
context* the atom appears in across replay. Role-cued retrieval
uses a similarity to `s`, which is decorrelated from content
similarity because replay sequences mix contents at the same role
positions.

**Anti-homunculus check.** Co-occurrence accumulation is statistical;
the conjunction is a fixed binding operation; retrieval is energy-
only. Pass.

**FHRR + MHN + Hebbian fit.** Replay is already in the architecture.
Co-occurrence is the simplest possible local statistic. Adds one
matrix `C`.

### Idea D — KL-style two-stream pressure (the TEM-of-FHRR).

**Mechanism.** Split the codebook into a small `g`-bank
(structural atoms) and the existing content atoms. During Phase 5
batch consolidation, regress each atom's content prediction on its
previous `g` and minimize a KL-like term that penalizes `g`-atoms
that need content to be predictable. This is an offline batch pass,
which the project's rules permit.

**Why it should work.** This is the TEM ELBO compressed into one
batch term. It is exactly the inductive bias that creates the role/
content geometric separation.

**Anti-homunculus check.** Offline gradient pass with explicit
permission from the design rules; no runtime arbitration. Pass.

**FHRR + MHN + Hebbian fit.** Compatible with the project's
"error-driven only in offline batch" rule. Requires a small loss
function and a separate `g`-bank.

### Idea E — Diagnostic-first probe: re-run Phase 5 with pseudo-
inverse MHN storage.

**Mechanism.** Without changing anything else, swap the Hebbian
write into MHN for a pseudo-inverse computation on the consolidated
atoms (one-shot, post-consolidation). Re-measure `hit_role` and
`rank_role`.

**Why this is the cheapest diagnostic.** If role basins suddenly
become retrievable, the bottleneck is *storage geometry*, not
*atom geometry*. This isolates whether the codebook itself has
role information that MHN softmax is currently masking.

**Anti-homunculus check.** Storage operation only. Pass.

---

## Surprises

- **TEM does not have role basins by Hebbian magic.** It has them
  because a KL term in a variational loss was carving them across
  millions of BPTT steps. The project's expectation that Hebbian
  consolidation alone would produce equivalent basins is, in
  hindsight, optimistic — there was no analog of TEM's prior-matching
  pressure.
- **FHRR binding *is* the TEM conjunction, compressed.** This is
  the bridge: the algebra of role-content binding is the same in
  both systems. The Phase 5 failure is not at the algebra layer; it
  is at the *training signal* layer (TEM had one; the project does
  not have a structural-prior-matching one).
- **Resonator networks already do role-cued retrieval in VSA** by
  factoring a bound vector into its components via iterative
  attractor dynamics. This is a candidate retrieval algorithm that
  is *not* MHN softmax and may be more sensitive to role-cue
  structure (see Research Brief 01).
- **2025 Bakermans/Whittington explicitly identifies replay as the
  carver of remote relational fields.** The project's replay
  currently does not condition on transitions / compositions; making
  it do so is a small architectural change with a large theoretical
  payoff.
- **The Hopfield-VSA paper (Kymn 2022) chose pseudo-inverse over
  Hebb for exactly the storage-geometry reason Phase 5 is hitting.**
  This is a smoking gun that the storage rule, not the binding
  algebra, is the load-bearing parameter.

---

## Sources

- Whittington, Muller, Mark, Chen, Barry, Burgess, Behrens (2020).
  "The Tolman-Eichenbaum Machine: Unifying Space and Relational
  Memory through Generalization in the Hippocampal Formation."
  *Cell*. [PMC7707106](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/)
- Whittington, Warren, Behrens (2022). "Relating transformers to
  models and neural representations of the hippocampal formation."
  ICLR 2022. [OpenReview](https://openreview.net/pdf?id=B8DVo9B1YE0)
- Stachenfeld, Botvinick, Gershman (2017). "The hippocampus as a
  predictive map." *Nature Neuroscience*.
  [PDF](https://gershmanlab.com/pubs/Stachenfeld17.pdf)
- Behrens, Muller, Whittington, Mark, Baram, Stachenfeld, Kurth-Nelson
  (2018). "What is a cognitive map? Organizing knowledge for flexible
  behavior." *Neuron*.
  [Cell](https://www.cell.com/neuron/fulltext/S0896-6273(18)30856-0)
- Bakermans, Warren, Whittington, Behrens (2025). "Constructing
  future behavior in the hippocampal formation through composition
  and replay." *Nature Neuroscience* 28(5):1061–1072.
  [PMC12081289](https://pmc.ncbi.nlm.nih.gov/articles/PMC12081289/)
- Sun, Advani, Spruston, Saxe, Fitzgerald (2023). "Organizing
  memories for generalization in complementary learning systems."
  *Nature Neuroscience*.
  [PMC10400413](https://pmc.ncbi.nlm.nih.gov/articles/PMC10400413/)
- Kymn / Stewart et al. (2022). "Associative memory of structured
  knowledge." *Scientific Reports*.
  [PMC9759586](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759586/)
- Recanatesi, Farrell, Lajoie, Deneve, Rigotti, Shea-Brown (2021).
  "Predictive learning as a network mechanism for extracting
  low-dimensional latent space representations." *Nature
  Communications*.
  [PMC7930246](https://pmc.ncbi.nlm.nih.gov/articles/PMC7930246/)
- Frady, Kent, Olshausen, Sommer (2020). "Resonator networks for
  factoring distributed representations of data structures."
  [arxiv 2007.03748](https://arxiv.org/abs/2007.03748)
- "Recent Advances in Resonator Networks for Neurosymbolic
  Computing." [OpenReview](https://openreview.net/pdf?id=FNrZd3Ls1d)
- Generalized HRR. [arxiv 2405.09689](https://arxiv.org/html/2405.09689v1)
- Bono et al. (2023). "Neural learning rules for generating flexible
  predictions and computing the successor representation." *eLife*.
  [eLife 80680](https://elifesciences.org/articles/80680)
- "A Biologically Interpretable Cognitive Architecture for Online
  Structuring of Episodic Memories into Cognitive Maps." (2025).
  [arxiv 2510.03286](https://arxiv.org/pdf/2510.03286)
- "The Spiking Tolman-Eichenbaum Machine." (2025).
  [bioRxiv 2025.10.16.682754](https://www.biorxiv.org/content/10.1101/2025.10.16.682754v1)
- Schaeffer et al. "Disentangling Fact from Grid Cell Fiction in
  Trained Deep Path Integrators."
  [PMC10723537](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723537/)
