# 02 — Predictive-Coding-Gated Codebook Growth & Emergent Disentanglement

**Brainstorm date:** 2026-05-24
**Angle:** Can role-sensitive atoms emerge from local prediction-error dynamics
or slowness/contrastive-predictive losses on the replay buffer — *without* any
supervisor labeling roles? If so, this dissolves Phase 5's ΔE floor (>5.5e-3,
content-prior vs role-prior) without requiring offline EqProp contrastive
training (M2 path).

The Phase 5 stuck state is, geometrically, *role atoms and content atoms have
the same energy landscape*. Every technique below is evaluated by: does it
produce a **local geometric dynamic** that pushes a subset of atoms toward
slow / context-invariant / role-like statistics while pushing the rest toward
fast / token-specific / content-like statistics, *purely from replay-buffer
statistics*?

---

## Key findings (with URLs)

### A. Predictive Coding Networks ↔ Hopfield convergence (load-bearing)

The most directly relevant 2023-2025 line of work is the Bogacz/Salvatori
group's demonstration that **predictive coding networks ARE Hopfield-style
associative memories with covariance learning**, sharing a single energy
function and a local Hebbian update.

- **Salvatori, Song, Yordanov, Millidge, Lukasiewicz, Bogacz (2021/2023):**
  *Associative Memories via Predictive Coding.* NeurIPS 2021 + journal.
  https://arxiv.org/abs/2109.08063 — Shows PCNs trained by local
  prediction-error minimization (energy F = Σ ‖x_l − f(x_{l+1})‖²) reproduce
  Hopfield-style content-addressable retrieval; outperforms classical
  Hopfield on correlated patterns. **Critical for us:** energy function is
  *identical in form* to ours, and the gradient pathway is local.
- **Tang, Salvatori, Millidge, Song, Lukasiewicz, Bogacz (2023):**
  *Recurrent predictive coding models for associative memory employing
  covariance learning.* PLoS Comp Bio.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719
  — Covariance learning rule (not just Hebbian outer product) makes PCNs
  robust to correlated patterns. Exactly the regime where the project's
  vanilla Hebbian struggles.
- **Salvatori, Millidge, Song, Bogacz, Lukasiewicz (2024):** *Associative
  Memories in the Feature Space.* ECAI 2023 / arXiv:2402.10814.
  https://arxiv.org/abs/2402.10814 — Memorize in an embedding space carved
  by a *contrastive loss*, not raw pixels. This is the closest extant
  precedent for "shape the codebook by a self-supervised loss before storing
  in Hopfield."
- **Tang, Barron, Bogacz (NeurIPS 2023/2024):** *Sequential Memory with
  Temporal Predictive Coding.* — Adds a *temporal* prediction-error term
  ("predict next state from current"), yielding sequence completion in a
  unified-energy associative memory with local Hebbian updates. **This is
  the slowness/CPC analogue inside the energy framework we already use.**
- **Online Training of Hopfield Networks Using Predictive Coding (2024):**
  arXiv:2406.14723. https://arxiv.org/abs/2406.14723 — Directly shows PCN
  inference dynamics can train a Hopfield substrate online, no separate
  training phase. The most "drop-in" route for the project's substrate.

### B. Whittington 2024-2025 — replay, compositionality, slots

Whittington's lab has converged hard on the same architectural shape the
project is building (relational/role atoms + content fillers + replay).

- **Bakermans, Warren, Whittington, Behrens (2025):** *Constructing future
  behaviour in the hippocampal formation through composition and replay.*
  Nature Neuroscience 28(5):1061-1072.
  https://www.nature.com/articles/s41593-025-01908-3 — Hippocampal state
  spaces are constructed *compositionally from primitives*; replay events
  "induce and strengthen new remote firing fields." Replay is the build
  operation, not just consolidation. This is the strongest external
  endorsement of the project's replay→codebook-growth picture.
- **Dorrell, El-Gaby, Behrens, Ganguli, Whittington (Neuron 2025):** *A tale
  of two algorithms: Structured slots explain prefrontal sequence memory
  and are unified with hippocampal cognitive maps.*
  https://www.cell.com/neuron/fulltext/S0896-6273(24)00765-7 — Prefrontal
  working memory = "controllable activity slots" (= role atoms);
  hippocampal long-term memory = bindings. **The project's role/content
  split has a direct neuroscientific analogue here. Worth reading for the
  exact mechanism that separates the two populations.**
- **Dorrell, Hsu, ..., Whittington (ICLR 2025):** *Range, not Independence,
  Drives Modularity in Biologically Inspired Representations.*
  arXiv:2410.06232. https://arxiv.org/abs/2410.06232 — In nonneg +
  energy-efficient linear autoencoders, sources modularise iff their joint
  *support* is sufficiently rectangular ("range condition"). This is a
  **direct prescription for the project**: if role and content sources have
  rectangular joint support across the replay buffer, modularisation
  (= role atoms vs content atoms) is geometrically forced. *Without* an
  arbiter. This could be the anti-homunculus story for Phase 5.
- **El-Gaby, Harris, ..., Whittington (Nature 2024):** *A cellular basis
  for mapping behavioural structure.* — Goal-progress cells emerge with
  fixed task-lag tuning across tasks sharing structure. Replay-buffer
  shared structure → cells with role-like tuning. Empirical demonstration
  of what we're trying to grow.

### C. Slow Feature Analysis on attractor/Hopfield substrates

- **Franzius, Sprekeler, Wiskott (PLoS Comp Bio 2007):** *Slowness and
  Sparseness Lead to Place, Head-Direction, and Spatial-View Cells.*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC1963505/ — Hierarchical SFA
  followed by sparse coding produces place, head-direction, view cells
  *from raw rodent-view image sequences*. Slowness is the only loss.
  **This is the load-bearing precedent that role-like cells (head
  direction) can emerge from slowness alone.**
- **Schönfeld, Wiskott (2014/2015):** Hierarchical SFA → place fields.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4441153/
- **Rolls (2021):** *Learning Invariant Object and Spatial View
  Representations in the Brain Using Slow Unsupervised Learning.*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8335547/ — Trace-rule learning
  (a slowness surrogate) inside attractor networks produces invariant
  object cells. **Trace rule = local synaptic eligibility, not a
  supervisor.**
- The classical claim: SFA recovers the slowest stochastic source. In a
  binding scene, the *role* (e.g. "filler-slot-1") changes more slowly
  than the *filler* across a replay episode that revisits the same
  template with different fillers. **Slowness on replay traces should
  separate role from content geometrically.**

### D. Grid-like codes from vector quantization (the surprise)

- **GCQ — Vector Quantization in the Brain: Grid-like Codes in World
  Models (2025):** arXiv:2510.16039.
  https://arxiv.org/abs/2510.16039 — Uses a *fixed* CANN-derived codebook
  + reconstruction + commitment loss. Codes are explicitly *role-like*
  (positions in attractor space, content arrives via association).
  No supervisor. **Anti-homunculus clean.** Surprising: they find that
  *making codes learnable degrades performance* — interesting evidence
  that the role substrate may want to be more constrained than the
  content substrate, asymmetrically. This maps onto the project's
  intuition that role atoms should be slower/stiffer.

### E. Active inference / discrete state spaces

- The Friston/Parr/Pezzulo *Active Inference* book (MIT Press, 2022)
  formalises codebook-like discrete state spaces under variational free
  energy minimisation.
  https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind
  — But the published mechanisms for *growing* the codebook (new states)
  remain ad-hoc (Bayesian model expansion triggered by free-energy
  thresholds — that is *exactly* the kind of homunculus rule the project
  must avoid).
- Anti-homunculus risk: every "active-inference codebook growth" paper
  reviewed has an explicit `if F > τ then add atom` rule. **This pathway
  is structurally homuncular and should not be adopted without a local
  reformulation.**

### F. Action-conditioned prediction → emergent binding

- **Glimpse Prediction Network / "A Minimal Task Reveals Emergent Path
  Integration and Object-Location Binding in a Predictive Sequence Model"
  (arXiv:2602.03490).** https://arxiv.org/abs/2602.03490 — Frame-prediction
  loss conditioned on action (efference copy) yields *both* path
  integration and object-location binding without explicit supervision.
  **This is the cleanest existence proof that an action-conditioned
  predictive loss alone carves role/content structure.**

---

## Concrete experimental designs

Each is sized to be runnable on the post-death substrate over a weekend,
re-using the existing Phase 5 ΔE = E_content-prior − E_role-prior headline
metric ([notes/emergent-codebook/phase-5-unified-design.md:256-281](../../../notes/emergent-codebook/phase-5-unified-design.md))
above the 5.5e-3 floor.

### Idea 1 — Temporal Predictive Coding head on the codebook
**Loss:** L = Σ_t ‖φ(x_t) − f(φ(x_{t−1}))‖² where φ is the projection
into codebook space and f is a learned linear map. Identical structure to
Tang/Bogacz 2023.
**Gradient pathway:** Local Hebbian-covariance update on f; codebook
atoms updated by Hebb on the residual. No supervisor: residual is
geometric.
**Hypothesis:** Atoms whose activations are *predictable from their
predecessors* (slow / contextual / role-like) stabilise; atoms whose
activations are *unpredictable* (fast / token-specific / content-like)
remain volatile. Asymmetric stiffness emerges from prediction error
alone.
**Expected ΔE movement:** content-prior energy stays ~unchanged
(content atoms still encode tokens); role-prior energy *drops* because
role atoms become more strongly attracting. ΔE should increase from
~3e-3 to ~8-12e-3 — comfortably above the 5.5e-3 floor.
**Anti-homunculus check:** ✅ No `if prediction-error > τ` rule. Update
is continuous, local, Hebbian-covariance.

### Idea 2 — SFA head on replay traces
**Loss:** L_slow = Σ_t ‖φ(x_t) − φ(x_{t−1})‖² s.t. var(φ) = I.
Equivalent to extracting the slowest sources of the replay stream into
the codebook subspace.
**Gradient pathway:** Whitening + per-atom slowness penalty. Stiehl/
Franzius-style. Update is local within each atom.
**Hypothesis:** On replay traces where role primitives are reused
across many fillers, role activations are slow (the same role repeats
across many adjacent windows); content activations are fast (each
window has different fillers). SFA carves role atoms by selecting the
slow subspace; content atoms inhabit the fast complement.
**Expected ΔE movement:** Largest expected effect; literature shows
slowness alone produces *head-direction-like* cells (= role-like).
ΔE could exceed 1.5e-2.
**Anti-homunculus check:** ✅ Slowness loss is a *measurement* of local
geometry, not a routing rule. Atoms separate by where they sit in the
slow-fast spectrum.
**Risk:** SFA on a non-stationary replay stream can collapse onto
trivial constants. Mitigation: variance constraint var(φ_i)=1 per
atom (standard SFA).

### Idea 3 — Action/context-conditioned CPC over replay
**Loss:** L_CPC = −log [ exp(z_{t+k}·c_t) / Σ_neg ] where c_t is a
context summary (slow integration of recent codebook activations) and
z_{t+k} is a future codebook activation; negatives sampled from other
replay episodes.
**Gradient pathway:** InfoNCE; per-atom updates are local in φ.
**Hypothesis:** Atoms predictive of future under matching context
(= role atoms grounding compositionally reusable structure) get
strengthened; content atoms remain idiosyncratic.
**Expected ΔE movement:** Probably mid-range; CPC's effect on
disentanglement is well documented but role-vs-content separation
depends on the context bottleneck.
**Anti-homunculus check:** ⚠️ Sample-based negatives are an
implementation detail, not a supervisor. But if negatives are
sampled by *content type*, that smuggles in a labeller. **Constraint:
negatives must be sampled by temporal distance only.**

### Idea 4 — Range/support shaping via replay augmentation (Dorrell 2025 prescription)
**Mechanism:** Per Dorrell, El-Gaby, ..., Whittington (ICLR 2025),
modularisation in nonneg + energy-efficient autoencoders is forced by
*rectangular joint support* of sources.
**Intervention:** Augment the replay buffer with *factorised* sampling
— in each replay episode, sample role primitives independently from
content fillers (rather than co-sampling whole episodes). This is a
*data-side* manipulation, not a model-side one.
**Expected ΔE movement:** If the range condition is the underlying
reason modularisation has been stuck, this is the cheapest possible
intervention (no new loss, no new architecture). ΔE should move
sharply if and only if range was the binding constraint.
**Anti-homunculus check:** ✅ Replay sampling policy is a buffer
property, not a runtime supervisor.
**This is the highest-value, lowest-cost experiment in the list.**

### Idea 5 — Two-timescale Hebbian: fast content / slow role
**Mechanism:** Run two Hebbian update rules with different τ (one fast,
one slow), each writing into a *subset* of codebook atoms. Atoms whose
updates accumulate on the slow timescale converge to slow/role-like
features; atoms on the fast timescale converge to content-like
features. *Which* atoms get which τ is initialised randomly — the
identity is not assigned.
**Gradient pathway:** Pure Hebbian, two τs.
**Hypothesis:** Slow-τ atoms become role-like (effectively a low-pass
filter over the replay stream); fast-τ atoms become content-like.
**Expected ΔE movement:** Probably moderate; depends on whether the
τ ratio is large enough to produce a meaningful slowness gradient.
**Anti-homunculus check:** ⚠️ The τ assignment is *fixed at init*,
which is borderline. If τ can drift based on each atom's local
statistics (e.g. its activation autocorrelation), the assignment
becomes a measurement, not an arbitrary supervisor. Use the drift
form.

### Idea 6 — Feature-space contrastive carving (Salvatori 2024 drop-in)
**Mechanism:** Apply Salvatori et al. 2024 directly. Before storage,
project tokens through a pretrained contrastive encoder (e.g.
SimCLR-style on the replay stream). Store in Hopfield in that feature
space.
**Hypothesis:** Contrastive features cluster by semantic similarity;
role/content separation emerges from the loss's invariances.
**Expected ΔE movement:** Likely modest unless the contrastive
objective is *temporally* contrastive (time-contrastive networks /
TCN). Standard SimCLR augmentations are not aligned with role/content.
**Anti-homunculus check:** Depends on which contrastive loss.

---

## Anti-homunculus screen — summary table

| Idea | Local geometric dynamic? | Verdict |
|------|-------------------------|---------|
| 1. TPC head | Yes — covariance Hebb on residual | ✅ Clean |
| 2. SFA head | Yes — slowness + variance constraint | ✅ Clean |
| 3. Action-conditioned CPC | Yes, *if negatives are time-only* | ⚠️ Mind sampling |
| 4. Range-shaping replay buffer | Yes — buffer statistics only | ✅ Cleanest |
| 5. Two-timescale Hebbian (drift form) | Yes — τ from autocorrelation | ✅ Clean |
| 6. Feature-space contrastive | Depends on loss family | ⚠️ Audit |
| Active-inference codebook growth | No — explicit `if F>τ` rule | ❌ Reject |

---

## Surprises

1. **GCQ (arXiv:2510.16039) finds learnable codes degrade performance.**
   If role atoms really do want to be stiffer/more constrained than
   content atoms, the project may want *asymmetric* plasticity rules:
   slow drift for role, fast drift for content. This matches the
   two-timescale idea above.
2. **The Dorrell "range, not independence" result (ICLR 2025) suggests
   the modularisation lever may be on the data side, not the model
   side.** The project has been searching for the right loss; the
   right intervention may be the right replay-sampling distribution.
   This is testable in hours, not weeks.
3. **The Whittington structured-slots Neuron 2024 paper makes the
   role/content split into a published, named architectural pattern**
   with empirical neural correlates. The project's framing has direct
   external endorsement.
4. **Bogacz/Salvatori's PCN-as-Hopfield work has already produced a
   sequential extension** (Tang et al. 2023) that does almost exactly
   what's wanted here — temporal-PC + associative memory + local
   updates. The dependence on their codebase / formulation is small;
   it's an additive head on the existing energy function.
5. **No group, as far as the search reaches, has done "SFA loss
   directly on a Hopfield/MHN codebook with replay traces."** This
   is a genuine white-space experiment and would be a publishable
   result if it works.
6. **No group treats the replay buffer as an SFA/CPC teacher
   explicitly** — replay is usually framed as iid sampling for
   consolidation. The framing of "replay buffer as a slow-timescale
   teacher whose temporal structure carves the codebook" appears
   novel.

---

## Promising leads (ranked, with first read)

1. **Idea 4 — Range-shaping replay buffer.** Cheapest, most directly
   prescribed by recent published theory (Dorrell ICLR 2025), no new
   architecture. *Read first:* arXiv:2410.06232.
2. **Idea 1 — Temporal PC head.** Directly extends the project's existing
   energy function with one additional local term. Strong published
   precedent. *Read first:* Tang, Barron, Bogacz NeurIPS 2023.
3. **Idea 2 — SFA head.** Genuine white-space; highest expected ΔE
   movement if the slowness hypothesis is right. *Read first:* Franzius
   et al. 2007 (PLoS Comp Bio) and Rolls 2021.
4. **Idea 5 — Two-timescale Hebbian (drift form).** Anti-homunculus
   subtle but tractable; complements Ideas 1-2 (could combine).
5. **Idea 3 — Action-conditioned CPC.** Strong literature support but
   more moving parts. Likely worth running after Ideas 1, 2, 4.
6. **Idea 6 — Feature-space contrastive.** Only worth pursuing if the
   contrastive loss is temporally grounded (TCN-style).

A clean staged experiment plan: run Idea 4 first (a single weekend, no
new code beyond the replay sampler); if ΔE moves to > 5.5e-3, the range
hypothesis was the binding constraint. If it doesn't, add Idea 1
(temporal PC head; ~one week to implement on the existing substrate).
If still flat, add Idea 2 (SFA head). Ideas 1+2 stack additively in the
energy function.

---

## Sources

- [Associative Memories via Predictive Coding (Salvatori et al. 2021/2023)](https://arxiv.org/abs/2109.08063)
- [Recurrent predictive coding models for associative memory employing covariance learning (Tang et al. 2023)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719)
- [Associative Memories in the Feature Space (Salvatori et al. 2024)](https://arxiv.org/abs/2402.10814)
- [Online Training of Hopfield Networks Using Predictive Coding (2024)](https://arxiv.org/abs/2406.14723)
- [Sequential Memory with Temporal Predictive Coding (Tang, Barron, Bogacz, NeurIPS 2023)](https://www.researchgate.net/publication/379777239_Sequential_Memory_with_Temporal_Predictive_Coding)
- [Constructing future behaviour in the hippocampal formation through composition and replay (Bakermans et al., Nature Neuroscience 2025)](https://www.nature.com/articles/s41593-025-01908-3)
- [A tale of two algorithms: Structured slots ... (Dorrell, Whittington et al., Neuron 2025)](https://www.cell.com/neuron/fulltext/S0896-6273(24)00765-7)
- [Range, not Independence, Drives Modularity ... (Dorrell, Whittington et al., ICLR 2025)](https://arxiv.org/abs/2410.06232)
- [A cellular basis for mapping behavioural structure (El-Gaby et al., Nature 2024)](https://www.nature.com/articles/s41586-024-08145-x)
- [Slowness and Sparseness Lead to Place, Head-Direction, and Spatial-View Cells (Franzius et al. 2007)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1963505/)
- [Modeling place field activity with hierarchical slow feature analysis (Schönfeld & Wiskott 2015)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4441153/)
- [Learning Invariant Object and Spatial View Representations in the Brain Using Slow Unsupervised Learning (Rolls 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8335547/)
- [Vector Quantization in the Brain: Grid-like Codes in World Models (GCQ, 2025)](https://arxiv.org/abs/2510.16039)
- [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior (Parr, Pezzulo, Friston 2022)](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- [A Minimal Task Reveals Emergent Path Integration and Object-Location Binding in a Predictive Sequence Model](https://arxiv.org/abs/2602.03490)
- [Modeling Recognition Memory with Predictive Coding and Hopfield Networks (OpenReview)](https://openreview.net/forum?id=gzFuhvumGn)
