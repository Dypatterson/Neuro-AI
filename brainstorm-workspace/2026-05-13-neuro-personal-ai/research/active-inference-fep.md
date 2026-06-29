---
date: 2026-05-13
angle: active-inference-fep
session: 2026-05-13-neuro-personal-ai
---

# Active Inference and the Free Energy Principle: A Research Brief

## Angle

**What was investigated:** Whether Karl Friston's Free Energy Principle (FEP) and active inference framework offers a rigorous foundation for the project's unresolved "diagnostics-to-actuators" problem — specifically, how geometric observations (surprise, prediction error, free energy) can become slow-timescale regulatory responses WITHOUT adding a supervisory controller.

**Why it's relevant:** The 2026-05-09 note identifies the next major architectural threshold as crossing from "geometry-as-observation" to "geometry-as-endogenous-regulation." It articulates the exact risk: the standard reading of every diagnostic-actuator pair ("high drift causes replay pressure") imports a hidden controller. The FEP is a mature framework in which apparent decisions ARE local energy minimization dynamics — which is exactly the grammar the project needs.

The core question: Does FEP actually resolve the anti-homunculus constraint, or does it just give the same controller a different name?

---

## Key Findings

### 1. FEP provides a principled answer to the diagnostic-actuator grammar problem

The central FEP claim is that any system with a Markov blanket must, on average, minimize its variational free energy. Crucially, this is not a decision made by a supervisor — it is the thermodynamic consequence of the system's existence as a bounded entity. The "response" to high free energy (surprise, prediction error) is not triggered — it *is* the forward dynamics of the system.

This maps directly onto the 2026-05-09 note's right-hand column ("the consolidation gate threshold rises with d̄, so spread atoms naturally get smaller updates without anyone reading d̄"). FEP gives this intuition a formal grounding: the consolidation gate threshold IS the free energy gradient. Nothing reads d̄ and decides; d̄ contributes to the free energy that drives the dynamics.

**Source:** [Free Energy Principle — Wikipedia](https://en.wikipedia.org/wiki/Free_energy_principle); [The Free Energy Principle Made Simpler — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S037015732300203X)

### 2. Markov blankets of Markov blankets: subsystems couple without an arbiter

An ensemble of Markov-blanketed subsystems can self-organize into a global system that itself has a Markov blanket. Each component minimizes its own free energy in a way that is *consistent with* the ensemble's free energy minimum — not because a coordinator ensures this, but because the generative models are shared and the free energy landscape is jointly defined.

**This directly addresses the coupling-without-arbiter question.** When two subsystems (e.g., the Hopfield retrieval dynamics and the Benna-Fusi consolidation chain) share a generative model, their free energy minima coincide. The "decision" about how much replay pressure to apply is not made by either subsystem — it is the joint free energy landscape evolving.

**Source:** [The Markov Blankets of Life — Royal Society Interface](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792); [A Free Energy Principle for a Particular Physics — arXiv](https://arxiv.org/pdf/1906.10184)

### 3. Timescale separation is native to FEP — not an add-on

In active inference, state inference operates on a fast timescale (real-time settling), policy inference on an intermediate timescale (behavioral), and parameter learning (Dirichlet hyperparameter accumulation) on a slow timescale. This is a mathematical consequence of the factored posterior, not an architectural choice.

The multi-timescale structure maps directly:
- Fast timescale: Hopfield settling (posterior over current hidden state)
- Intermediate timescale: replay policy selection (posterior over trajectory)  
- Slow timescale: codebook/consolidation parameter updates (posterior over model parameters, i.e., Dirichlet pseudo-counts)

The Benna-Fusi chain of slow variables is already structurally equivalent to the slow-learning layer of the FEP hierarchy. FEP provides the mathematical reason *why* this structure produces near-linear capacity scaling: the mean-field approximation between fast and slow timescales is justified precisely by the separation.

**Source:** [Active Inference and Learning — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167251/); [Deep Temporal Models and Active Inference — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5998386/)

### 4. Variational free energy minimization yields Hopfield dynamics as a special case

This is the most direct technical finding. Minimizing variational free energy with respect to the internal states of a system yields a Boltzmann-machine-like stochastic update mechanism — with continuous-state stochastic Hopfield networks as a special case. The Hopfield energy landscape IS a variational free energy landscape when the generative model is a Gaussian.

The implication: the Hopfield retrieval dynamics (which the project already has) are already performing approximate variational inference. The project does not need to add FEP on top of Hopfield — Hopfield IS a restricted form of FEP inference. The question is whether extending to the full FEP framework (adding the generative model, priors, and slow learning dynamics) gives the project anything it doesn't already have.

**Source:** [Self-Orthogonalizing Attractor Networks Emerging from the Free Energy Principle — arXiv 2505.22749](https://arxiv.org/abs/2505.22749); [Free Energy Minimization: A Unified Framework — arXiv](https://arxiv.org/pdf/2011.14963)

### 5. The precision mechanism: FEP's answer to gating without a gate

In active inference, "precision" (inverse variance / uncertainty) weights prediction errors. High-precision channels dominate the update; low-precision channels are attenuated. This is implemented neurobiologically via neuromodulatory gain control — not a supervisor deciding which channel wins, but a local multiplicative weighting that emerges from the generative model's uncertainty estimate.

This is exactly the gating structure the project needs for the diagnostic-actuator pairs. For example: "metastability → replay prioritization" becomes "metastable trajectories have higher prediction error → higher precision weight → larger contribution to the free energy gradient driving replay." The arbitration is not supervisory; it is a consequence of the precision structure of the generative model.

**Source:** [Active Inference, Attention, and Motor Preparation — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3177296/); [The Many Roles of Precision in Action — MDPI](https://www.mdpi.com/1099-4300/26/9/790)

### 6. Hippocampal replay as inverted active inference during consolidation

A 2025 paper in Cerebral Cortex proposes that NREM sleep implements a continuation (specifically, an inversion) of waking active inference — refining representations through long-term depression. REM sleep then updates the generative model itself through long-term potentiation. These are not two separate systems arbitrated by a controller; they are dual phases of the same free energy minimization process operating at different timescales and using complementary plasticity rules.

This maps cleanly to the project's replay/consolidation design: the Hopfield retrieval dynamics (NREM-equivalent, inhibitory, stabilizing) and the codebook refinement (REM-equivalent, excitatory, updating the generative model) are the same free energy process viewed at two temporal resolutions.

**Source:** [Adaptive Consolidation of Active Inference — Cerebral Cortex / PubMed](https://pubmed.ncbi.nlm.nih.gov/40422982/)

### 7. Self-orthogonalizing attractor networks from FEP (Spisak & Friston, May 2025)

A paper just published on arXiv in May 2025 (co-authored by Friston) derives attractor networks directly from the FEP applied to random dynamical systems. The key result: FEP-derived attractor networks spontaneously develop approximately orthogonalized attractor representations — not because of an explicit orthogonalization rule, but because simultaneously optimizing predictive accuracy and model complexity (the two terms in variational free energy) penalizes redundant attractors.

**This is directly relevant to the codebook.** The project's emergent codebook is trying to maintain a diverse, non-collapsed set of attractor basins (the NC1 reformulation in the 2026-05-09 note). FEP provides a principled reason why this should happen automatically if the system is minimizing free energy: redundant attractors have higher complexity cost without accuracy gain, so the free energy landscape pushes toward orthogonalization.

The paper also shows that when data is presented in fixed sequences (rather than random order), the learning rule produces asymmetric weights implementing temporal predictive coding — directly relevant to the temporal co-occurrence work from the 2026-05-11 PAM note.

**Source:** [Self-Orthogonalizing Attractor Neural Networks Emerging from the Free Energy Principle — arXiv 2505.22749](https://arxiv.org/abs/2505.22749), [GitHub](https://github.com/pni-lab/fep-attractor-network)

### 8. Generative model of memory construction and consolidation (Nature Human Behaviour, 2023)

This paper models memory consolidation as a two-stage process: (1) one-shot memorization in a modern Hopfield network (hippocampal encoding), followed by (2) variational autoencoder training driven by hippocampal replay (neocortical consolidation). The generative networks take over from the Hopfield network, producing memories that are more abstract, more generalizable, and more prone to schema-based distortions.

This is the architecture the project is independently converging toward — but the paper provides empirical validation (simulates memory age effects, hippocampal lesion effects, semantic memory, imagination, relational inference, boundary extension). It also frames replay as training a generative model, which aligns the project's Benna-Fusi slow-variable chain with a FEP generative model parameter update.

**Source:** [A Generative Model of Memory Construction and Consolidation — Nature Human Behaviour](https://www.nature.com/articles/s41562-023-01799-z); [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.01.19.524711v2.full)

---

## Promising Leads

### A. pymdp — concrete implementation reference

pymdp is an open-source Python library for active inference agents in discrete state spaces. It implements exactly the timescale-separated update equations: fast state inference (`infer_states()`), intermediate policy inference (`infer_policies()`), and slow Dirichlet parameter learning. This is a reference implementation the project could read to understand the concrete update equations before deciding what to adapt.

The key notebook is the free-energy calculation notebook in the docs.

**Source:** [pymdp — GitHub](https://github.com/infer-actively/pymdp); [pymdp paper — arXiv 2201.03904](https://arxiv.org/abs/2201.03904)

### B. Expected Free Energy decomposition: epistemic + pragmatic value

Expected free energy (EFE) — the quantity minimized for policy selection — decomposes into an epistemic term (information gain, uncertainty reduction) and a pragmatic term (outcome utility, goal achievement). Minimizing EFE automatically balances exploration and exploitation without any meta-level decision.

**Relevance:** the codebook refinement problem may be naturally framed as EFE minimization. Updating a codebook entry has epistemic value (reduces surprise about future inputs) and pragmatic cost (disrupts existing attractor basins). EFE balances these without a controller deciding when refinement is "worth it."

**Source:** [Generalised Free Energy and Active Inference — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/); [Active Inference and Epistemic Value — UCL FIL](https://www.fil.ion.ucl.ac.uk/~karl/Active%20inference%20and%20epistemic%20value.pdf)

### C. Predictive Coding + Hopfield unified framework

Several recent papers derive both predictive coding networks and Hopfield networks from the same energy-based framework. Both perform metric learning in recognition memory tasks; PCNs outperform Hopfield networks on correlated patterns. The unified framework suggests the project's Hopfield substrate could be extended with predictive coding dynamics (top-down prediction + bottom-up error) without abandoning the Hopfield energy formulation.

**Source:** [Modeling Recognition Memory with Predictive Coding and Hopfield Networks — OpenReview](https://openreview.net/forum?id=gzFuhvumGn); [Associative Memories via Predictive Coding — OpenReview](https://openreview.net/forum?id=VuzPO_TZHPc)

### D. Scale-free active inference and hierarchical planning

Recent work (2024-2025) builds deep active inference systems with hierarchical world models at multiple timescales, where slow latent variables contextualize fast latent transitions. The MTRSSM (Multiple Timescale Recurrent State Space Model) is the computational instantiation. These architectures do long-horizon planning entirely through latent imagination — no symbolic planner, no supervisor.

**Relevance:** this is what "Phase 4+ multi-scale retrieval" might look like once the consolidation dynamics are in place. The slow latent states (Benna-Fusi slow variables in the project's language) contextualize which attractor basins are active at the fast timescale.

**Source:** [From Pixels to Planning: Scale-Free Active Inference — arXiv 2407.20292](https://arxiv.org/abs/2407.20292); [Deep Active Inference with Diffusion Policy and Multiple Timescale World Model — arXiv 2510.23258](https://arxiv.org/abs/2510.23258)

---

## Concrete Ideas

### Idea 1: Reframe each diagnostic-actuator pair as a free energy gradient

The 2026-05-09 note's right-hand column already has the right shape; FEP gives it a name. Concretely, for each diagnostic-actuator pair, define a quantity F_component that contributes to the total variational free energy, and show that the "actuator response" is the gradient descent direction on F.

| Diagnostic | FEP quantity | Gradient response |
|---|---|---|
| High drift | KL divergence between current attractor and prior model | Gradient descent → replay, which reduces KL |
| High spread (d̄) | Precision of the consolidation likelihood | Low precision → smaller parameter update (not a gating rule, but lower weight on gradient) |
| Bimodality | Complexity cost of redundant attractors | Gradient penalizes redundancy → splitting |
| Metastability | Prediction error on settling trajectories | Higher error → higher precision weight → larger replay contribution |
| Low cap-coverage | Surprise at reconstruction from attractor | Surprise gradient drives consolidation restructuring |

This is not a new mechanism — it reframes existing mechanisms in FEP language. The value is that the FEP language is provably anti-homuncular: gradient descent on free energy has no arbiter.

### Idea 2: Use the FEP orthogonalization result to replace explicit repulsion in codebook dynamics

The Spisak & Friston (2505.22749) result shows that minimizing free energy naturally orthogonalizes attractor representations without an explicit repulsion rule. The project's current Hebbian codebook updater (post the f4475f9 commit) is local and biologically plausible. The FEP framing suggests it is already minimizing something like free energy, and the orthogonalization property should be emergent.

**Concrete test:** Run the Hopfield + Hebbian update substrate and measure pairwise cosine similarity between codebook entries over time. If they are drifting toward orthogonality without explicit repulsion, the FEP prediction is confirmed and the project has evidence that its substrate is already operating in the FEP regime.

### Idea 3: Treat the Benna-Fusi slow variable chain as the FEP slow-learning layer

The Benna-Fusi chain (fast synapse → cascade → slow consolidation) has the same structure as the FEP slow-learning layer (fast state inference → intermediate policy inference → slow parameter learning via Dirichlet accumulation). The mathematical correspondence:

- Fast synaptic variable ↔ variational posterior over current hidden state
- Chain variables u₁...uₙ ↔ accumulated pseudo-counts at nested timescales
- Slow variable leak ↔ Dirichlet prior concentration pulling pseudo-counts back toward prior

If this correspondence is formalized, the Benna-Fusi capacity scaling result (near-linear vs. √N) gets a FEP interpretation: near-linear scaling arises because the hierarchical factored posterior allows mean-field approximation between timescales.

### Idea 4: Precision as a substrate-native attention mechanism (no separate attention module)

Active inference's precision mechanism implements attention through local multiplicative weighting of prediction errors — no separate attention module, no softmax over attended channels. In the FHRR substrate, precision could be implemented as a scalar weight on each codebook entry's gradient contribution, derived from the entropy of its retrieval distribution (low entropy = high precision = high weight).

This is consistent with the anti-homunculus filter: the weight is a function of local geometric state (retrieval entropy), not a decision made by a controller. It also connects to the project's existing softmax-entropy-as-feature/prototype-mode-classifier (from 2026-05-09 note) — that entropy is already being computed, and could be repurposed as a precision weight.

### Idea 5: Sleep-replay phases as dual free energy dynamics

The 2025 adaptive consolidation paper provides a concrete architectural proposal: NREM sleep implements inhibitory consolidation (stabilizing new representations via LTD), REM sleep implements excitatory generative model update (updating prior beliefs via LTP). These are the same free energy dynamic at different phases, not two systems arbitrated by a controller.

For the project: define two replay modes — stabilization replay (reduces surprise at current attractor geometry, equivalent to NREM/LTD) and generative replay (updates codebook parameters, equivalent to REM/LTP). The fraction of time spent in each mode is determined by the relative magnitudes of the accuracy term and the complexity term in variational free energy. High surprise → more stabilization; high complexity → more generative. No controller picks the mode; the free energy composition determines it.

---

## Surprises

### Surprise 1: The self-orthogonalization paper is from May 2025 and directly bridges FEP and Hopfield

This paper (arXiv 2505.22749, Spisak & Friston) was published three weeks ago and is precisely the bridge between FEP and the project's specific substrate concerns. It was not in any of the project's reading lists. The result — that attractor orthogonalization is a free energy consequence, not a design choice — is directly relevant to the NC1 reformulation and the "avoid codebook collapse" concern. This paper should be bookmarked immediately and read before Phase 3.

The GitHub repository (pni-lab/fep-attractor-network) includes the full manuscript source and simulation code.

### Surprise 2: The project's architecture is already structurally equivalent to a generative memory model described in Nature Human Behaviour

The 2023 Nature Human Behaviour paper (Kumaran group) independently arrived at: Hopfield network for rapid episodic encoding → replay drives VAE training → VAE becomes the slow-timescale generative model. This is architecturally isomorphic to the project's: Hopfield retrieval → replay-driven Benna-Fusi consolidation → slow variable chain as the compressed world model. The paper has empirical validation across memory phenomena that the project has not yet tested against. The distortion effects (boundary extension, schema-based gist) are direct predictions the project's system should also produce — and could validate against.

### Surprise 3: FEP does NOT resolve the diagnostic-actuator problem by magic — it dissolves it by definition

The temptation is to read FEP as "just use free energy minimization and the controller problem goes away." It is more subtle. FEP resolves the problem by providing a language in which the diagnostic and the actuator are the same object: the free energy landscape, viewed at two temporal resolutions. This does not mean the architecture becomes trivially easy to build. It means the project now has a formal criterion for when a proposed mechanism passes the anti-homunculus filter: *does this mechanism have a natural description as gradient descent on a free energy functional?* If yes, pass. If not (i.e., if the mechanism requires inspecting a metric and triggering a response), fail.

This is the real contribution of FEP to the project: not a mechanism to add, but a formal test to apply to every candidate mechanism.

### Surprise 4: The VSA/FHRR community and the FEP community have not met

There is essentially no literature at the intersection of FHRR / hyperdimensional computing and active inference. This is a gap, not a finding against the approach — but it means any FEP-FHRR integration would be novel rather than drawing on established prior work. The closest bridge is through the Hopfield network (which both communities work with), but the FHRR binding structure has no direct FEP counterpart in the literature. This could be a novel contribution or a warning sign that the connection is less tight than it appears.

### Surprise 5: Expected Free Energy has an "epistemic foraging" term the project might need earlier than Phase 5

The EFE decomposition has an epistemic value term that drives the agent to actively seek information-rich states — even when they are not goal-directed. In the context of the project's memory system, this means the retrieval dynamics should naturally seek out high-surprise inputs (novel or ambiguous patterns) rather than only operating on presented stimuli. This is a form of active attention or curiosity that emerges from free energy minimization without any explicit curiosity module. The project may want to track whether retrieval dynamics already show this property, or whether it needs to be engineered in.

---

## Additional Sources

- [Active Inference: A Process Theory — activeinference.github.io](https://activeinference.github.io/papers/process_theory.pdf)
- [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior — MIT Press](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)
- [A Step-by-Step Tutorial on Active Inference — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022249621000973)
- [Canonical Neural Networks Perform Active Inference — Communications Biology / Nature](https://www.nature.com/articles/s42003-021-02994-2)
- [In Vitro Neural Networks Minimise Variational Free Energy — Scientific Reports](https://www.nature.com/articles/s41598-018-35221-w)
- [The Criticality of Consciousness: E/I Balance and Dual Memory Systems in Active Inference — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385856/)
- [Self-Evidencing Through Hierarchical Gradient Decomposition — arXiv 2510.17916](https://arxiv.org/html/2510.17916)
- [Dynamic Planning in Hierarchical Active Inference — arXiv 2402.11658](https://arxiv.org/html/2402.11658v3)
- [Hopfield-Fenchel-Young Networks: A Unified Framework for Associative Memory Retrieval — arXiv 2411.08590](https://arxiv.org/abs/2411.08590)
- [Predictive Attractor Models — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/5df4313ecd4875931fbdacc486cc1fcf-Paper-Conference.pdf)
