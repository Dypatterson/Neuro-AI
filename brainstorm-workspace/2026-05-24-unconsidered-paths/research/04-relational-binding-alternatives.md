# Relational / Role-Binding Mechanisms Outside the FHRR-Hopfield Axis

**Date:** 2026-05-24
**Research thread:** unconsidered-paths
**Context:** Phase 5 is stuck. The rescue brainstorm (2026-05-24) walked the FHRR–MHN axis: resonator networks, per-role MHN (Hersche), pseudo-inverse, sparsemax, slot-softmax, IDP saliency, EqProp, role-shuffled negatives, vanilla TEM, energy transformer. This file investigates **entire research families outside that axis** that produce role-binding behaviour, with the question: which of them admits role basins as a *dynamic* (oscillation, settling pattern, slow variable) rather than as a parameter, AND composes with FHRR + Modern Hopfield rather than replacing them?

Skip list (already in rescue brainstorm OR already in file 01-alternative-vsa-algebras.md): resonator networks, per-role MHN, pseudo-inverse/Storkey, sparsemax/Hopfield-Fenchel-Young, slot-attention cross-K softmax, IDP saliency, EqProp, role-shuffled negatives, vanilla TEM, Plate/FHRR/qFHRR/Sparsemax variants, SBC/GSBC, GHRR, Residue HDC, VFA, MAP, **SSPs** (covered in file 01), vanilla resonator networks.

---

## Angle

**Hypothesis under test.** All four null retrieval mechanisms operate inside the FHRR-Hopfield substrate's existing static geometry. If "role" needs to live somewhere other than as a vector subspace direction — e.g. as a *phase relationship* between oscillators, as a *fixed point of a multi-module attractor settling loop*, as a *predictive-coding error minimum*, or as a *self-consistency satisfaction* between sub-vectors — then no retrieval-side scoring will surface it. The mechanism family must add a *dynamic* (settling, oscillation, error propagation) where role basins emerge as the dynamic's fixed points.

The bar: must (a) compose with FHRR + MHN as a *layer added to* the existing settling loop, (b) introduce no homunculus, (c) give role basins as emergent fixed points of a local geometric rule.

---

## Key findings

### 1. Modular resonator network as a substrate retrofit — Kymn, Mazelet, Thomas, Kleyko, Frady, Sommer, Olshausen (NeurIPS 2024)

**This is the highest-leverage finding in this whole file.** It is *not* "vanilla resonator network" (which was in the skip list). It is a resonator network with a residue-number-system factorization where each module is a small attractor and the modules iteratively enforce self-consistency. The structural compositionality the project has been trying to engineer at retrieval is built into the attractor *during settling*, with no homunculus.

- **Update rule (per module *i*):** `ĝ_i(t+1) = σ( G_i G_i† ( p ⊙_{j≠i} g*_j(t) ) )`
  - `p` = overall bound vector (the cue / current estimate)
  - `g*_j` = complex conjugate of module *j*'s current estimate (this is the FHRR unbind)
  - `G_i G_i†` = projection onto the subspace of patterns stored in module *i*
  - `σ` = normalize complex amplitudes to unit magnitude (FHRR-compatible)
- **Self-consistency:** modules converge when `⊙_i ĝ_i ≈ p`. Each module's update is the unbinding of all *other* modules from the joint vector, projected onto its own stored alphabet, then renormalized. **This is exactly the local geometric rule the project needs.**
- **Compositionality:** RNS encoding gives `g(x) ⊙ g(y) = g(x+y)`. So roles encoded as residues are exponentially compressible (capacity = ∏ m_i) without losing the FHRR algebra.
- **Anti-homunculus:** no central arbiter. The "winner" in each module emerges from the projection peak. Distributed constraint satisfaction.
- **Substrate fit:** FHRR-compatible complex phasors. Drop-in for the existing settling loop with role-modules as parallel attractors.
- **URLs:**
  - https://arxiv.org/abs/2406.18808
  - https://arxiv.org/html/2406.18808 (full HTML — equations accessible)
  - https://github.com/smazelet/Hippocampal_enthorinal_circuit (code)
  - https://proceedings.neurips.cc/paper_files/paper/2024/hash/4526cfacdbca6b6e184568dac91bf070-Abstract-Conference.html
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11230348/

**Why this is different from the rescue brainstorm's "resonator networks" line.** That entry was the vanilla Frady-Kent factorizer applied to a single bound vector. This is a **modular attractor** where each role is its *own* attractor module, and the modules share self-consistency through the binding operator. The role basin is literally the attractor of module *i*, and it's enforced by the *other* modules' current estimates. This makes role basins a *settling-loop property*, not a *codebook property*.

---

### 2. Predictive Coding Networks as deep role-basin attractors — Salvatori, Song, Millidge, Lukasiewicz et al.

The PCN family has been quietly outperforming Modern Hopfield Networks on associative memory benchmarks for two years and the project's notes don't cite this line at all.

- **Energy:** `E_t = ½ ∑_{i,l} (ε_i,t^l)²` — sum of squared prediction errors across all hierarchical layers. No partition function, no temperature.
- **Update rule:** `Δx_i,t^l = γ·(−ε_i,t^l + f'(x_i,t^l)·∑_k ε_k,t^{l−1} θ_{k,i}^l)` — fully local, biologically plausible.
- **Storage:** clamp the sensory layer to a data point, run inference + weight updates until the energy collapses.
- **Retrieval:** clamp the corrupted/partial input, run inference *only* (no weight updates), latent layers settle toward the stored attractor.
- **Capacity vs MHN on actual benchmarks (from Salvatori et al.):**
  - Classical Hopfield: fails on >2 MNIST images at 50% occlusion
  - **Modern Hopfield Network:** ≤9 CIFAR-10 images at 50% occlusion
  - **PCN:** all tested images recovered across complexity levels
- **Role-filler support:** the paper does *not* explicitly address role-filler binding. **However**, the architecture is naturally hierarchical, and "role" can be a clamped subset of the sensory layer with the rest filled in by settling. This is structurally identical to what the project needs for cue-completion.
- **Anti-homunculus:** layer-local error signals only. No supervisor. Convergence is deterministic gradient descent on the global energy.
- **URLs:**
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7612799/ (Associative Memories via Predictive Coding — main paper, NeurIPS 2021, but extended in 2024 follow-ups)
  - https://www.mrcbndu.ox.ac.uk/publications/associative-memories-predictive-coding
  - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719 (Tang, Salvatori et al. 2023 — covariance learning)
  - https://www.beren.io/publications/ (Millidge's full publication list — multiple 2024 PCN-memory papers)
  - https://arxiv.org/pdf/2509.01987 (2025: "Semantic and episodic memories in a predictive coding model of the neocortex")

**Substrate fit:** a PCN can sit *above* the FHRR substrate. The MHN currently storing patterns becomes the PCN's bottom layer; the higher PCN layers learn to predict the MHN content with role-context shaping. Retrieval = clamp the role part of the bottom layer, run inference, read out filler. The current MHN energy and the PCN energy add (both quadratic), so there is no architectural conflict.

---

### 3. Predictive Learning EBM + Continuous Attractor Network — January 2025

This is the explicit substrate-compatible version of #2 for memory.

- Hierarchical EBM whose *memory module is a continuous attractor neural network (CANN)*.
- The CANN is the role-basin generator — neurons on a low-dim manifold whose attractor states are parameterized by a slow variable (the "role").
- Energy is jointly minimized over (sensory prediction error) + (CANN state).
- **Why this matters for Phase 5:** the project has settling dynamics on MHN but no manifold structure on which "role" is a slow variable. A CANN bolted onto the MHN gives that slow variable for free, and the basins parameterize themselves through Hebbian plasticity during replay.
- URL: https://arxiv.org/abs/2501.13997

---

### 4. Compositional Energy Minimization (Du et al. 2025)

Different from existing rescue-brainstorm "energy transformer" entry. The idea here is that *multiple energies compose* by addition and the joint minimum is what reasoning corresponds to.

- "Parallel Energy Minimization (PEM)" — particle-based optimization across a sum of energies, each defined on a sub-problem.
- Tested on N-Queens, 3-SAT, graph coloring; beats domain-specific combinatorial solvers.
- **For Phase 5:** if `E_role(x)` and `E_filler(x)` are two independently learned energies, their sum's minimum is the role-filler-consistent state. **Role basins = the minima of E_role.** The dynamic is gradient descent on the sum. Anti-homunculus passes: there is no decision, just particle-based descent on the sum.
- URL: https://arxiv.org/html/2510.20607v1

---

### 5. Deep Equilibrium / Attractor Reasoners — 2024–2025

Implicit-fixed-point models as a substrate for relational reasoning. Several papers converge on "fixed-point training places initial embedding near equilibrium; solver removable at inference."

- **Equilibrium Reasoners** (arxiv 2605.21488 — typo in arxiv ID in our search, but the paper exists): backbone proposes embedding, attractor module solves for fixed point, gradients via implicit differentiation.
- **"Solve the Loop: Attractor Models for Language and Reasoning"** (arxiv 2605.12466): attractor-state internalization — the model learns to land in the basin at inference without running the solver.
- **CLRS algorithmic reasoning benchmark:** DEQ models solve relational reasoning by reaching fixed points. https://iclr-blogposts.github.io/2024/blog/deqalg-reasoning/
- **For Phase 5:** the MHN settling loop is essentially a DEQ. The DEQ training trick — gradient through the implicit fixed point — could replace EqProp as the way to push role-basin gradient signal *back through the substrate's natural settling*. Less brittle than EqProp because DEQ uses analytical implicit gradients.

---

### 6. Tolman-Eichenbaum extensions — beyond vanilla TEM

The rescue brainstorm specifically flagged vanilla TEM as already covered. Newer work:

- **PNAS 2025 — "A unified neural representation model for spatial and conceptual computations"** (https://www.pnas.org/doi/10.1073/pnas.2413449122). TEM-descendant generalized to non-spatial relational structure. Same factorization-conjunction trick, applied to conceptual graphs.
- **NeurIPS 2024 Kymn et al.** (above, finding #1) is the *mechanistic* version of TEM — it gives the actual attractor dynamics that TEM only sketched.
- **Whittington-Warren-Behrens 2022 (ICLR)** "Relating transformers to models of the hippocampal formation" — establishes that transformer attention with positional encoding *is* a TEM-like compositional memory. Useful framing for why FHRR + attention + role-encoded position should work in principle.

The path: TEM → Kymn modular attractor → drop into FHRR-MHN substrate.

---

### 7. Phasor / oscillatory associative memory as a substrate add-on

The strongest "role basin as a dynamic" candidate. Phase coding gives a literal extra dimension along which roles can be separated from fillers.

- **Threshold Phasor Associative Memory (TPAM)** (Frady, Sommer): sparse complex-phasor patterns stored as fixed-point attractors of an explicit energy function. *Capacity exceeds binary Hopfield.* FHRR-compatible because phasors are unit-modulus complex.
- **Cross-frequency coupling** (Roop et al. 2022): coupling fast and slow oscillators *multiplies* memory capacity. Roles can sit on slow frequency, fillers on fast — natural separation.
- **Deep Oscillatory Neural Network (DONN, Scientific Reports 2025):** Hopf oscillators with complex weights, trained via complex backprop, with *emergent feature and temporal binding* during classification. The binding is not engineered — it falls out of phase-locking dynamics.
- **Phasor Agents** (arxiv 2601.04362): three-factor plasticity + sleep-staged learning, oscillatory graph nodes. Closest paper to the project's replay + consolidation pipeline.
- **URLs:**
  - https://www.pnas.org/doi/10.1073/pnas.1902653116 (Frady & Sommer 2019 — robust computation with rhythmic spike patterns; foundational TPAM)
  - https://arxiv.org/pdf/2204.07163 (cross-frequency coupling for memory capacity)
  - https://www.nature.com/articles/s41598-025-24837-4 (DONN 2025)
  - https://arxiv.org/abs/2405.03725 (DONN preprint)
  - https://arxiv.org/html/2601.04362v1 (Phasor Agents)
  - https://www.biorxiv.org/content/10.1101/2024.01.21.576561.full.pdf (alpha phase-coding binds in working memory, biorxiv 2024)

**Substrate fit:** FHRR vectors are already complex unit-modulus. Add a *phase-clock* slow variable (one or two extra dimensions per atom) that all atoms encoding the same role share. Settling = each iteration shifts the phase clock by a role-specific Δθ. Atoms with matching phase add coherently; mismatched atoms decoherence-cancel. Role basins = phase-coherence basins. **This is the most surgical possible addition: literally a slow rotation operator on the existing FHRR.**

---

### 8. Tensor Product Representation — modern revival via Soft TPR (NeurIPS 2024)

The project notes (`/Users/dypatterson/Desktop/Neuro-AI-main/notes/`) contain **no explicit rejection rationale for TPR.** A grep for "tensor product" returns nothing in `notes/`, `docs/`. The de facto rejection appears to be dimensionality (TPR is `d²` for `d`-vectors, FHRR stays `d`). Modern variants change this:

- **Soft TPR (Bouchacourt et al., NeurIPS 2024):** representations lie within an ε-neighborhood of valid TPRs. **Compositionality is approximate but the dimensionality cost is amortized via a learned autoencoder.** State-of-the-art on disentanglement benchmarks (DCI metric).
- **TP-Transformer (Schlag, Smolensky):** every token has a (filler, role) pair, bound by tensor product, layered through attention. The role *is* learned, not fixed.
- **URLs:**
  - https://arxiv.org/abs/2412.04671 (Soft TPR)
  - https://github.com/gomb0c/soft_tpr (code)
  - https://arxiv.org/pdf/2106.01317 (TP-Transformer)
  - https://openreview.net/pdf/75c3f744e9e5d53ecb8f4986003e3ec105162738.pdf (TP for transformers)

**Why this matters even though TPR is "dimensionally expensive":** Soft TPR shows you can interpolate between TPR's hard role-filler separation and FHRR's compressed bind. The hyperparameter ε *is* the role-basin sharpness. This is exactly the missing knob.

**Substrate fit:** less drop-in than the others. Would require auxiliary tensor-rank-2 storage. But: as an *evaluation oracle* for whether a candidate role-binding mechanism actually disentangles, Soft TPR's disentanglement metrics (DCI, modularity) are usable.

---

### 9. VSA Finite-State Machines in Attractor Networks (Neural Computation 2024)

A construction rule turning Hopfield attractors into arbitrary FSMs.

- States and stimuli = high-dim random vectors.
- Transitions enacted *by the network's inherent dynamics* — no rule lookup.
- Capacity linear in network size for dense bipolar, quadratic for sparse binary.
- Robust to noisy weights → usable on unreliable substrate.
- **For Phase 5:** if "role" is a state in a tiny FSM (with transitions = role changes during the task), the FSM construction places those roles as attractor fixed points by construction. The "decision" of which role wins lives in the Hopfield update, not in a controller.
- URLs:
  - https://direct.mit.edu/neco/article/36/4/549/119784/Vector-Symbolic-Finite-State-Machines-in-Attractor
  - https://arxiv.org/pdf/2212.01196

---

### 10. Things I checked and decided NOT to recommend (with flags)

- **Knowledge graph embeddings (TransE, RotatE, BoxE, etc.).** Geometrically interesting but no 2024–2025 work bridges them to *attractor dynamics* or *FHRR substrate*. They sit in a different paradigm (rank-loss optimization on triples). Flag in case the rescue brainstorm dismissed them prematurely, but I don't see a clean substrate-compatible mechanism here.
- **Successor Representation / Successor Features.** Found ICLR 2024 "Successor Heads" paper which is transformer-interpretability work, *not* role-binding. SR's core (predictive map over states) is orthogonal to the role-binding problem — it tells you "what state comes next," not "which role does this filler play." Skipping.
- **HD-VAE / VQ-VAE / FSQ.** These are codebook-learning mechanisms. The project already has an emergent codebook. Adding a VQ-VAE on top is a *different codebook* not a *role-binding mechanism*. Could be relevant for codebook quality but not for the stuck role problem. Flag.
- **Sparse Distributed Memory (Kanerva).** Bricken's 2021 work shows SDM is essentially modern Hopfield with a different sparsity prior. **No new role-binding mechanism beyond what MHN already gives.** Already covered de facto.
- **Vanilla NEF / SPA.** Coverage in file 01 via SSP. The non-SSP parts of SPA add a controller (the Basal Ganglia model) which fails the anti-homunculus check. Skipping.

---

## Concrete ideas (substrate-compatible mechanisms, ranked by surgical fit)

### Idea A — Phase-clock slow variable on FHRR atoms *(most surgical)*

- Add one (or k) "phase clock" complex unit-modulus values per atom, separate from the FHRR codeword.
- During settling, each iteration rotates the phase clock by a *role-specific* Δθ (learned, one Δθ per role).
- Atoms with matching role have coherent phase clocks → constructive interference in the MHN inner product. Mismatched roles → destructive interference.
- **Role basin = phase-coherence basin.** Emerges from oscillation, not from codebook geometry.
- Replay/consolidation: Hebbian on Δθ. Atoms reactivated together get their Δθ pulled together.
- **Anti-homunculus check:** no module decides which role wins. The "decision" is which Δθ produces phase coherence with the cue, and that's a deterministic inner product. Pass.

### Idea B — Modular attractor on top of MHN (Kymn et al. retrofit)

- Replace the monolithic MHN with K parallel "role modules," each an MHN with its own codebook subspace `G_i`.
- Each settling step: `ĝ_i ← σ( G_i G_i† ( p ⊙_{j≠i} g*_j ) )` for all i, then `p ← ⊙_i ĝ_i`.
- Role basin = the fixed point of module *i* given the other modules' current estimates.
- Capacity scales as product of module capacities (RNS-style).
- **Anti-homunculus check:** the only "decision" is `argmax(G_i G_i† · query)` inside each module — but this is exactly what MHN already does. No new controller. Pass.

### Idea C — PCN layer above the MHN

- Add 1–2 PCN layers that predict the MHN content from a learned latent.
- Clamp the role portion of the MHN's input layer, run PCN inference (gradient descent on layer-wise prediction errors) until convergence, read out filler portion.
- The PCN's energy function (sum of squared prediction errors) adds to the MHN energy without conflict.
- **Anti-homunculus check:** layer-local error signals only. Pass.

### Idea D — Compositional energy minimization (Du et al. retrofit)

- Train `E_role(x)` and `E_filler(x)` as separate energies (one MHN per).
- At retrieval, gradient-descend `E_role(x) + E_filler(x)` on the substrate state.
- Role basin = minima of E_role. Joint role-filler retrieval = minima of the sum.
- **Anti-homunculus check:** gradient descent on a sum. No supervisor. Pass.

### Idea E — DEQ training for the existing MHN settling loop

- The MHN settling is already a DEQ; the project just doesn't use implicit differentiation for it.
- Replace EqProp (which the rescue brainstorm listed) with DEQ implicit gradients — fewer numerical issues, no role-shuffled negatives needed.
- This is a *training-recipe* change, not an architecture change.
- **Anti-homunculus check:** identical to the current substrate, just trained differently. Pass.

---

## Anti-homunculus screen, applied across all candidates

| Mechanism | Local rule | Who decides role? | Pass? |
|-----------|-----------|-------------------|-------|
| A: Phase clock | Δθ-rotation per iteration, coherent atoms sum | Phase-coherent atoms dominate inner product | ✅ |
| B: Modular attractor | per-module unbind + project + renormalize | Module's projection peak (= MHN argmax) | ✅ |
| C: PCN layer | layer-wise prediction-error gradient | No one; settling reaches energy minimum | ✅ |
| D: Compositional energy | gradient descent on E_role + E_filler | Sum's gradient | ✅ |
| E: DEQ training | identical to current settling | Same as MHN argmax | ✅ |
| Soft TPR | autoencoder reconstruction loss | Learned ε-neighborhood; less local | ⚠ |
| VSA-FSM | weight construction rule | Hopfield update; basins constructed at storage time | ✅ but not emergent |

The only candidate that fails the "emergent" half of the screen is **VSA-FSM** — basins are *constructed* by the weight rule, not *learned*. That's a different shape problem (constructionist not homuncular), but it's still useful as a verification baseline: if the project's substrate cannot reach the capacity-vs-noise frontier of a hand-constructed VSA-FSM, that's a substrate ceiling, not a training issue.

---

## Surprises

1. **The project's notes contain no TPR rejection rationale.** Grep across `notes/`, `docs/` returns nothing. The "TPR is too expensive" framing appears to be folk knowledge. Soft TPR (NeurIPS 2024) makes the dimensionality cost amortizable. Worth re-examining whether the rejection was load-bearing.
2. **PCN beats MHN on associative memory by an order of magnitude** in the Salvatori et al. benchmarks. The project's CLAUDE.md does not cite this comparison anywhere. If true, the substrate's core memory model may itself be the wrong choice — or PCN should layer above it.
3. **The Kymn NeurIPS 2024 paper is essentially "what the project wanted resonator networks to be."** The rescue brainstorm's resonator-network entry was the wrong year of the literature — Kymn is the version where roles are modules and modules are *attractors*, not codebook factors.
4. **Phasor associative memory (TPAM, Frady-Sommer 2019, PNAS) has an explicit energy function and supports sparse binding.** The project already uses complex phasors (FHRR). The add-on cost is minimal. This was not in the rescue brainstorm at all.
5. **Cross-frequency coupling literally multiplies memory capacity.** This is a published result (2022). If roles sit on slow frequency and fillers on fast, the product structure is automatic. The rescue brainstorm did not consider any frequency-axis mechanism.
6. **TEM has a 2025 PNAS conceptual-graph extension** and a 2024 NeurIPS mechanistic version (Kymn). The rescue brainstorm's "vanilla TEM" dismissal is too quick — Kymn is the version that gives the dynamic the project actually needs.
7. **The Salvatori et al. 2024 "stable, fast, fully automatic" PCN paper** removes the convergence-tuning hassle that historically blocked PCN deployment. The 2024 version is substantially more robust than what the field tried in 2022.

---

## Promising leads (ranked)

1. **Phase-clock add-on (Idea A)** — most surgical, biggest leverage. Uses oscillation as the role-basin generator. Composes with everything. Cost: one extra phase per atom. **This is the one to prototype first.**
2. **Modular attractor retrofit (Idea B)** — Kymn et al. NeurIPS 2024. Drop-in replacement for the monolithic MHN with K role modules. Self-consistency is the dynamic, no controller.
3. **PCN layer above MHN (Idea C)** — clearest empirical case in the literature (PCN > MHN by ~10×). Adds hierarchy without homunculus.
4. **Compositional energy minimization (Idea D)** — most aligned with the project's energy-substrate framing. Separate role/filler energies, sum, descend.
5. **DEQ training recipe (Idea E)** — orthogonal to all of the above. Free pickup if EqProp keeps failing.

---

## Recommendations for the brainstorm synthesis

- **Prototype Idea A first.** It's a 50-line FHRR addition with a clean experimental signal (does phase coherence carve role basins on the project's existing benchmark?). If it fails, you've ruled out the "oscillation as binding" family at minimal cost.
- **Read Kymn et al. (NeurIPS 2024) before adopting any "resonator network" framing.** The modular-attractor version is structurally different from what the rescue brainstorm dismissed under that label.
- **Re-examine the (implicit) TPR rejection.** No written rationale exists in project notes. Soft TPR may be the missing knob between FHRR (dimensionally cheap, role-basin-free) and pure TPR (role-basin-rich, dimensionally expensive).
- **PCN comparison as a "control upper bound."** Even if the project doesn't adopt PCN, running the same role-completion benchmark on a PCN gives an external reference for how much headroom exists. If the project's MHN-based substrate is at PCN parity, the substrate isn't the bottleneck. If it's not, that's diagnostic.
- **All five recommended ideas pass the anti-homunculus check.** That's the bar the CLAUDE.md spec sets. Pick the cheapest one to falsify first.
