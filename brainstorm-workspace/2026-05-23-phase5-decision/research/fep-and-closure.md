# FEP-grounded reformulation, and closure-as-deliverable

Deep-research notes for the Phase 5 decision brainstorm. Two sub-angles:
A) whether the Free Energy Principle (FEP) / active-inference family offers a
principled alternative to K-branch energy-margin discrimination that does NOT
require the substrate to expose energy gaps between priors; and
B) what "closing the phase as graduation-unattained, with the substrate-
saturation finding as the architectural deliverable" looks like in the
literature — who has done this and how they framed it.

The driving observation from the project: D=4096 sharp-basin self-retrieving
substrates do not stratify K-branch settling by raw energy margin; K=1 is
empirically >= K=4; the entire premise of multi-branch settling-by-energy
may be incompatible with the substrate's geometry. The question is whether
(A) a different score function (expected free energy rather than raw energy)
moves the problem out of the saturation regime, or whether (B) the cleanest
move is to write up the constraint and stop.

---

## Sub-angle A: Active inference as a reformulation of "structural branching"

### A.1 What is hypothesis selection under expected free energy?

Active inference (Friston and collaborators, ~2015–present) replaces "score
each hypothesis by its raw energy / likelihood" with "score each hypothesis
by its expected free energy (EFE)." EFE decomposes (canonically) into:

- **Pragmatic value** — expected log-utility, i.e. how well the hypothesis
  is expected to satisfy preferences / priors over outcomes.
- **Epistemic value** — expected information gain about latent states under
  that hypothesis (how much the hypothesis is expected to *resolve
  uncertainty*).

This is a fundamentally different scalar than "energy of the converged
attractor." A branch can win on EFE because it is expected to disambiguate
the latent state, even if its raw energy is no lower than a competitor.
The framework is documented in:

- Friston et al., **"Active Inference: A Process Theory"** (Neural
  Computation 2017) — the canonical formulation;
  https://activeinference.github.io/papers/process_theory.pdf
- **"Whence the Expected Free Energy?"** (Millidge, Tschantz, Buckley,
  Neural Computation 2021) — derives EFE from first principles;
  https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy
- Champion, Bowman, Marković, Grześ, **"Reframing the Expected Free
  Energy: Four Formulations and a Unification"** (arXiv:2402.14460,
  Feb 2024) — formalizes the derivation problem and shows the standard
  decompositions are equivalent or bounded; this is the
  state-of-the-art on what EFE *is* mathematically;
  https://arxiv.org/abs/2402.14460

### A.2 The relevant "structural branching" architectures (2023–2026)

Several lines of work directly implement multi-hypothesis settling scored
by EFE rather than raw energy / likelihood. The ones most analogous to
the Neuro-AI K-branch setup:

1. **Sophisticated inference** (Friston et al. 2021;
   https://arxiv.org/abs/2006.04120). A recursive form of EFE that
   implements a *tree search over belief states* (not states per se).
   At each branch point the agent expands hypotheses about what its
   posterior beliefs *would be* under each candidate action/policy, and
   scores branches by expected free energy under the resulting belief
   trajectory. This is conceptually the closest analog to "K-branch
   settling" — except the discriminator is EFE, and the branching is
   over belief trajectories, not over fixed-point attractors.

2. **Active Inference Tree Search (AcT)** — Maisto, Gregoretti, Friston,
   Pezzulo, Neurocomputing 2024 (originally arXiv:2103.13860, updated
   Dec 2024). Combines active inference's normative scoring with MCTS-
   style branch expansion for large POMDPs.
   https://www.sciencedirect.com/science/article/pii/S0925231224020903

3. **Dynamic planning in hierarchical active inference** — Neural
   Networks 2024 (ScienceDirect). Hierarchical active inference where
   higher levels propose abstract plans and lower levels propose
   refinements, all scored by level-appropriate EFE.
   https://www.sciencedirect.com/science/article/pii/S0893608024010049

4. **Deep Active Inference Agents for Delayed and Long-Horizon
   Environments** (arXiv:2505.19867, May 2025). VAE-style state-space
   models with Recurrent State-Space Models for memory; uses EFE for
   action selection and demonstrates scaling beyond toy domains.
   https://arxiv.org/html/2505.19867v1

5. **Expected Free Energy-based Planning as Variational Inference**
   (arXiv:2504.14898, April 2025). Reformulates EFE planning explicitly
   as variational inference over plans — extends the "planning as
   inference" line and clarifies that branch selection is variational
   posterior contraction, not a hard argmax over scalar scores.
   https://arxiv.org/pdf/2504.14898

6. **"Reframing the Expected Free Energy"** (Champion et al. 2024,
   above) — most relevant to a project deciding *which* EFE
   formulation to use. The paper enumerates four root definitions and
   shows their equivalences; this matters because different EFE
   decompositions induce different gradients during branch scoring and
   thus different discriminative behavior.

7. **ActiveInference.jl** (Heins, Klein, Demekas, et al., Entropy 2025,
   27/1/62) — open-source Julia library implementing the canonical
   POMDP-based active inference loop including EFE scoring. Useful as
   reference implementation; pymdp (Python) is the older counterpart.
   https://www.mdpi.com/1099-4300/27/1/62

### A.3 Why this *might* avoid the "branches need to stratify by raw energy" failure mode

The Neuro-AI substrate-saturation finding is: at D=4096 with sharp self-
retrieving basins, branches converge to nearly identical raw energies
because the substrate's basin geometry collapses them onto the same
attractor manifold. The K-branch architecture relied on raw energy
*margins* between branches as the discriminative signal; those margins
vanish.

EFE-scored branches change the discriminative signal in a way that should
not collapse for the same reason:

- **Epistemic value is computed against a posterior over latent
  variables**, not against the attractor's converged energy. Two
  branches with nearly identical attractor energy can still differ
  sharply in *how much they would update beliefs about the role
  bindings*. The discriminator lives in the KL divergence between prior
  and posterior over latent states, not in the energy of the converged
  fixed point.

- **Pragmatic value is computed against a prior over preferred
  outcomes** (effectively a goal prior). For Neuro-AI this maps cleanly
  onto the "role-prior vs content-prior" axis from
  [phase-5-unified-design.md:256-281](../../notes/emergent-codebook/phase-5-unified-design.md) —
  the ΔE that was the original headline metric is closely analogous
  to a pragmatic-value differential, but EFE additionally weights it
  by an epistemic term that may break the substrate-saturation tie.

- **EFE is computed *before* settling, not after.** The classical
  K-branch architecture lets each branch settle and then scores
  converged energies. EFE scoring scores the *expected* result of a
  branch — this is computed from the prior/posterior structure of the
  generative model and need not depend on the substrate's basin
  geometry at all. (In Friston's process-theory framing, EFE is
  evaluated under the variational posterior, which is independent of
  the Hopfield-style energy landscape.)

The honest caveat: EFE scoring requires an explicit generative model of
the latent state space (typically a POMDP or hierarchical state-space
model). The Neuro-AI substrate is implicit — the "latents" are FHRR
bindings inside the attractor, not declared random variables with priors
and likelihoods. Bolting EFE onto the existing substrate requires
*either* (a) declaring the role/content split as the latent factorization
explicitly, with a proper generative model over each, or (b) using
amortized inference (a recognition network) to approximate the posterior
needed for EFE.

This is non-trivial. Concretely it means Phase 5 stops being "K-branch
settling on the FHRR/Hopfield substrate" and becomes "active-inference
agent with FHRR+Hopfield as its perceptual front end and a small
declared generative model on top." That is a larger architectural move
than just changing the branch score.

### A.4 Concrete idea for Neuro-AI

The minimal experiment that would test whether EFE discriminates where
energy-margin doesn't:

1. Keep the K=4 branch architecture from Phase 5 exactly as-is.
2. For each branch, compute three scalars at the *end* of settling:
   - raw converged energy (current headline)
   - posterior entropy over the role-binding slot (cheap proxy for
     epistemic value — the branch that produces lower entropy over
     roles has higher information gain)
   - log-likelihood of the converged content under a learned content-
     prior (proxy for pragmatic value)
3. Define EFE_proxy = pragmatic_value + epistemic_value, and re-score
   branches under EFE_proxy.
4. Measure: does EFE_proxy stratify branches when raw energy doesn't?
   Specifically, on the same evaluation set where the substrate-
   saturation finding showed K=1 >= K=4 under raw-energy weighting,
   does EFE-weighted bundling recover a K=4 > K=1 result?

This is a one-day diagnostic, not a phase replan. It would tell you
whether the saturation finding is *substrate-level* (no signal exists)
or *score-function-level* (signal exists but raw energy is the wrong
read-out). If it's substrate-level, that's a stronger architectural
constraint — and the closure framing in sub-angle B applies directly.
If it's score-function-level, Phase 5 has a path forward, but the path
involves declaring a generative model and is no longer "settling on the
existing substrate."

Anti-homunculus check on EFE-scored branching:
- Who decides which branch wins? A softmax over EFE values, which is
  the same shape as the current energy-weighted softmax — no new
  arbiter, just a different score.
- The "decision" still lives in the variational posterior contraction,
  which is a local geometric dynamic over belief states. EFE just
  *defines what gradient that posterior is descending*; it is not a
  rule that reads metrics and triggers responses.
- This passes the anti-homunculus filter cleanly. The mechanism is
  expressible as "the posterior over branches contracts to the branch
  whose expected log model evidence is highest."

### A.5 Adjacent: predictive-coding associative memory

Worth flagging because it sits in the same FEP family but with a
different commitment:

- **Associative Memories via Predictive Coding** (Salvatori et al.,
  NeurIPS 2021; PMC7612799) — predictive-coding networks as
  associative memories that *outperform* modern Hopfield networks on
  retrieval accuracy and robustness, and degrade more gracefully under
  load. "Classical Hopfield networks recovered original images only
  when trained on two images but failed when trained on more than
  two." PC-AM keeps working at much higher capacity.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7612799/

- **BayesPCN** (Yoo & Wood, NeurIPS 2022; arXiv:2205.09930) — Bayesian
  generative predictive-coding network with continual one-shot writes,
  no meta-learning. Recalls high-dimensional data observed hundreds-
  to-thousands of timesteps ago without large recall drop. Implements
  graceful forgetting via posterior-uncertainty growth, not via fixed
  capacity limit.
  https://arxiv.org/abs/2205.09930

- **Semantic and episodic memories in a predictive coding model of the
  neocortex** (Tang et al., arXiv:2509.01987, Sept 2025) — recent
  unification of semantic + episodic memory in one PC network. Worth
  reading because it makes the FEP/PC architectural commitments
  explicit for memory systems.
  https://arxiv.org/pdf/2509.01987

- **Recurrent predictive coding models for associative memory
  employing covariance learning** (Tang et al., PLOS Comp Bio 2023;
  PMC10132551) — directly addresses the criticism that fully
  hierarchical PC models lack CA3-style recurrence. Adds recurrent
  connections and shows the model still works as an AM.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10132551/

The implication for Neuro-AI: the FEP-aligned alternative to modern
Hopfield substrate is *predictive-coding-based associative memory*,
not active-inference-bolted-on-Hopfield. If Phase 5 graduates into a
fundamental substrate change, switching the memory substrate from
Modern Hopfield + FHRR to a PC-AM (or BayesPCN-style continual PC-AM)
that already has explicit latents and posterior beliefs would make
EFE-scored branching natural rather than bolted-on.

This is a Phase 6+ direction, not a Phase 5 patch. But it answers the
question "if active inference is the right framing, what substrate is
it the right framing *for*?"

---

## Sub-angle B: Closure as architectural contribution

The user has explicitly raised the option of closing Phase 5 as
graduation-unattained and treating the substrate-saturation finding as
the architectural deliverable. This sub-angle asks: is that a
recognized contribution pattern in the field, what does it look like
done well, and where does it get published?

### B.1 The publication norm has shifted toward accepting negative results

The single most directly relevant paper:

- **Karl, Kemeter, Dax, Sierak, "Position: Embracing Negative Results
  in Machine Learning"** (ICML 2024 oral, arXiv:2406.03980). The
  paper's thesis: predictive performance alone is a bad indicator of
  publication worth; the community should normalize publishing
  negative results, both to reduce wasted re-derivation and to align
  researcher incentives with truth-seeking. The fact that this
  appeared as an *oral* at ICML 2024 is itself a signal — the
  conference selected it as one of the highest-visibility position
  papers of the year.
  https://arxiv.org/abs/2406.03980
  https://proceedings.mlr.press/v235/karl24a.html
  https://icml.cc/virtual/2024/poster/35063

Concrete recommendations from that paper relevant to Neuro-AI closure:
- Frame negative results as *what was learned about the problem
  structure*, not as "we tried X and it didn't work."
- Include all the rigor of a positive result (controls, CIs, multi-
  seed). The substrate-saturation finding already meets this bar by
  the project's own standards.
- Acknowledge what *would* be required for the approach to work;
  this is what makes it a contribution and not just a complaint.

NeurIPS 2025 also added a **Position Paper Track** for the first time
specifically to host perspective pieces, including "this architecture
has this constraint" arguments. This is a venue.
https://intuitionlabs.ai/pdfs/neurips-2025-a-guide-to-key-papers-trends-stats.pdf

### B.2 Examples of substrate-saturation / architectural-constraint findings published as primary contributions

These are the closest reference points for what a Neuro-AI closure
paper would look like:

1. **"Modern Hopfield Networks Require Chain-of-Thought to Solve
   NC^1-Hard Problems"** (arXiv:2412.05562, Dec 2024). The headline
   *is* a limitation: constant-layer modern Hopfield networks with
   O(n) hidden dimension *cannot* solve NC^1-hard problems like
   undirected graph connectivity. This is exactly the shape of
   contribution Neuro-AI could make: "substrate X cannot do task Y
   without addition Z, here is the mechanism." The paper is in the
   complexity-theoretic tradition (expressivity bounds), which is
   the most prestigious framing for an architectural limitation.
   https://arxiv.org/pdf/2412.05562

2. **"On Computational Limits of Modern Hopfield Models: A Fine-Grained
   Complexity Analysis"** (Hu, Lin, Song, Liu — referenced in the
   Awesome Modern Hopfield Networks repo; ICML 2024). Another
   expressivity-bound paper for modern Hopfield networks. Pattern:
   identify the architectural family, prove what it cannot represent
   efficiently, document the cost of evading the limit.
   https://github.com/Event-AHU/Awesome_Modern_Hopfield_Networks

3. **"The Capacity of Modern Hopfield Networks under the Data Manifold
   Hypothesis"** (arXiv:2503.09518, 2025). Generalizes capacity
   computation for exponential Hopfield to realistic pattern
   ensembles. The framing is "here is the actual capacity once you
   stop assuming i.i.d. uniform patterns" — a substrate-constraint
   finding presented as a positive contribution by sharpening the
   theory.
   https://arxiv.org/abs/2503.09518

4. **"Accuracy and capacity of Modern Hopfield networks with synaptic
   noise"** (arXiv:2503.00241, 2025). Documents how capacity scales
   under noise — another architectural constraint as primary
   contribution.
   https://arxiv.org/pdf/2503.00241

5. **"Practical Lessons on Vector-Symbolic Architectures in Deep
   Learning-Inspired Environments"** (Carzaniga et al., NeSy 2025;
   PMLR 284:218-236). Four "lessons" from comparing VSA families —
   some are positive (HLB ≈ HRR for retrieval; linear readout beats
   similarity search), some are negative (HRR's FFT convolutions are
   slower than MAP/HLB despite optimization). This is *directly* the
   contribution shape Neuro-AI could adopt: "after running these
   architectures hard, here is what we learned about which choices
   actually matter and which are constrained." Published at a
   workshop-track conference, which is the appropriate venue for
   empirical lessons that don't form a single theorem.
   https://openreview.net/forum?id=5ZmvZkqyoy
   https://proceedings.mlr.press/v284/carzaniga25a.html

6. **"Inverse Scaling in Test-Time Compute"** (arXiv:2507.14417, 2025).
   Shows that extending Large Reasoning Models' reasoning processes
   amplifies flawed heuristics — i.e., more compute is *worse* on
   specific task families. The contribution is the characterization
   of which task structures break the scaling assumption. The frame
   is "we expected X, we measured not-X, here is why" — a high-
   profile negative-result paper.
   https://arxiv.org/pdf/2507.14417

7. **"Scaling Laws Are Unreliable for Downstream Tasks: A Reality
   Check"** (arXiv:2507.00885, 2025). Same shape: an established
   assumption (training-loss scaling predicts downstream accuracy)
   shown to fail; the contribution is *the failure characterization*.
   https://arxiv.org/html/2507.00885v1

8. **"Paradoxical increase of capacity due to spurious overlaps in
   attractor networks"** (arXiv:2510.17593, 2025). The surprise IS
   the contribution — spurious overlaps, traditionally treated as a
   defect, increase capacity through a sparsening mechanism. This is
   a model where the "negative" finding (spurious overlaps exist) is
   reframed as a positive mechanism (they sparsen activity), and the
   paper is published on that reframe alone.
   https://arxiv.org/pdf/2510.17593

9. **"Self-Organization and Spectral Mechanism of Attractor Landscapes
   in High-Capacity Kernel Hopfield Networks"** (arXiv:2511.13053,
   2025). Introduces "Pinnacle Sharpness" as a metric for attractor
   stability; shows the network self-organizes into a critical state
   trading global stability against rank for multi-pattern storage.
   This is highly relevant to the Neuro-AI sharp-basin finding —
   they have *named the same trade-off* and made characterizing it
   the contribution.
   https://arxiv.org/pdf/2511.13053

### B.3 How are these typically framed?

Three framing modes, ordered by prestige:

**(a) Theorem-style — "here is the bound."** Examples: the NC^1
hardness paper, the fine-grained complexity paper, the capacity-
under-manifold paper. The contribution is a mathematical statement
about what the architecture *cannot* do (or can do only at cost X).
This requires a clean formalization and is highest-prestige but also
hardest to write. The Neuro-AI substrate-saturation finding probably
does not have a tight theorem statement yet, but might.

**(b) Mechanism-explanation — "here is why."** Examples: the spurious-
overlaps paper, the Pinnacle Sharpness paper, the inverse-scaling-in-
test-time-compute paper. The contribution is the *mechanism* that
explains why an expected behavior fails to materialize. This is the
most likely framing for the Neuro-AI finding: the substrate saturates
because [specific geometric reason about D=4096 sharp basins and role-
binding interference]; here is the controlled experiment that isolates
the mechanism.

**(c) Empirical-lessons — "here is what we learned."** Examples: the
VSA Practical Lessons paper, position papers like Embracing Negative
Results. The contribution is the *catalog* of empirical results from
running an architecture family hard. Lower prestige per paper but
lower barrier to entry; appropriate when no single mechanism explains
all observations. NeSy, NeurIPS workshops, and the new NeurIPS
Position Paper Track all accept this shape.

For Neuro-AI, the cleanest path is probably **(b) — mechanism-
explanation** if a single geometric mechanism explains the K=1≥K=4
finding, ΔE not stratifying, and the 6-instance constraint at D=4096.
Fall back to **(c) — empirical lessons** if the explanation is
"several mechanisms compound."

### B.4 Venues that accept this shape of contribution

- **NeurIPS Position Paper Track** (new in 2025). Designed for
  exactly this kind of perspective contribution.
- **ICML Position Track** (since 2024; "Embracing Negative Results"
  was an oral here). Same shape.
- **NeSy** (International Conference on Neurosymbolic Learning and
  Reasoning). Smaller venue, friendly to empirical-lessons papers
  like the VSA one.
- **Neural Computation** (journal). Friendly to mechanism-explanation
  papers in computational neuroscience; "Whence the EFE" was here.
- **Entropy** (MDPI). Friendly to FEP-adjacent / information-
  theoretic framings; ActiveInference.jl was here.
- **eLife / PLOS Computational Biology**. Friendly to negative-
  result computational neuroscience if framed as a constraint on
  the space of plausible biological models.
- **arXiv preprint with a workshop submission**. Lowest barrier;
  appropriate as a first move while the framing solidifies.

### B.5 What it looks like to document the Neuro-AI substrate-saturation finding well

Drawing on the patterns above, a Neuro-AI closure paper would have
this shape:

**Title (mechanism-explanation style):** "Sharp-basin self-retrieving
substrates resist role-binding-prior traversal: a capacity-geometry
constraint on FHRR + Modern Hopfield + emergent codebook
architectures."

**Section 1 — what the architecture was supposed to do.** State the
target: contextual completion via K-branch energy-weighted bundling
where branches stratify by ΔE between role-prior and content-prior.
Cite the Phase 5 design as the formal target.

**Section 2 — what we measured.** Headline metric and CIs. The
substrate-saturation finding: at D=4096 with sharp basins, branches
do not stratify by raw energy; K=1 ≥ K=4 across N seeds; the 6-
instance constraint. Include the controls: random codebook,
shuffled tokens, no-replay. Multi-seed with bootstrap or Wilson CIs.
This is the rigor bar from the Embracing Negative Results paper.

**Section 3 — the mechanism.** This is the most important section.
Why does the substrate saturate at D=4096? The hypothesis is that
sharp self-retrieving basins make role-binding-prior traversal
*incompatible*: traversal requires the substrate to expose
intermediate states that the basins collapse away. Express this as
a local geometric property of the substrate, with a small
diagnostic experiment that isolates the mechanism (e.g. measure
basin depth vs traversal distance at increasing D).

**Section 4 — what would be required.** Two paths:
(a) keep the substrate, change the score function (EFE rather than
raw energy — sub-angle A above);
(b) change the substrate (e.g. predictive-coding-based AM with
explicit latents, ref. Salvatori 2021, BayesPCN, Tang 2025).
Be explicit that the finding constrains a *family* of architectures
(sharp-basin attractor + raw-energy-scored branching), not just the
specific Neuro-AI implementation.

**Section 5 — discussion.** Connect to the existing literature on
attractor capacity-vs-basin trade-offs (arXiv:2511.13053, the
Pinnacle Sharpness paper, is the closest analog and should be cited
explicitly). Note the relationship to the modern-Hopfield-NC^1-
hardness result: both are constraints on what fixed-substrate
attractor models can do without additional machinery.

**Appendix — full experimental record.** All seeds, all controls,
the full Phase 5 design doc as supplementary material. The bar from
the Practical Lessons VSA paper: when the contribution is "we ran
this architecture hard, here is what we learned," the running-it-
hard is the data and needs to be reproducible.

This is publishable. The closest reference points (the NC^1 paper,
the Pinnacle Sharpness paper, the Practical Lessons VSA paper, the
inverse-scaling paper) are all 2024–2025 publications at top
venues. The contribution shape is recognized and the venues exist.

### B.6 The "shelved work" failure mode to avoid

Negative results get shelved (not published) when:

- The framing is "we tried X, it didn't work" with no mechanism.
- Controls are missing or single-seed.
- The constraint is presented as a deficiency of the *project*
  rather than of the *architecture family*.
- The author keeps trying patches instead of writing up. (The
  Embracing Negative Results paper specifically calls this out as
  the wasted-effort failure mode.)

The Neuro-AI finding does not have these failure modes by default —
the project's STATUS-and-checklist discipline already produces the
rigor needed. The risk is timing: if Phase 5 keeps spawning
diagnostics, the writeup never happens. The closure decision
benefits from being made *before* the next drill-down.

---

## Key papers (URLs, arXiv IDs)

**Sub-angle A: active inference and EFE**

- Friston et al., "Active Inference: A Process Theory," Neural
  Computation 2017. https://activeinference.github.io/papers/process_theory.pdf
- Millidge, Tschantz, Buckley, "Whence the Expected Free Energy?"
  Neural Computation 2021 (arXiv:2004.08128).
  https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy
- Friston et al., "Sophisticated Inference," 2021. arXiv:2006.04120.
  https://arxiv.org/abs/2006.04120
- Champion, Bowman, Marković, Grześ, "Reframing the Expected Free
  Energy: Four Formulations and a Unification," Feb 2024.
  arXiv:2402.14460. https://arxiv.org/abs/2402.14460
- Maisto, Gregoretti, Friston, Pezzulo, "Active Inference Tree
  Search in Large POMDPs," Neurocomputing 2024 (arXiv:2103.13860,
  updated Dec 2024).
  https://www.sciencedirect.com/science/article/pii/S0925231224020903
- "Dynamic planning in hierarchical active inference," Neural
  Networks 2024.
  https://www.sciencedirect.com/science/article/pii/S0893608024010049
- "Deep Active Inference Agents for Delayed and Long-Horizon
  Environments," May 2025. arXiv:2505.19867.
  https://arxiv.org/html/2505.19867v1
- "Expected Free Energy-based Planning as Variational Inference,"
  April 2025. arXiv:2504.14898. https://arxiv.org/pdf/2504.14898
- Heins, Klein, Demekas, et al., "Introducing ActiveInference.jl,"
  Entropy 2025, 27(1):62. https://www.mdpi.com/1099-4300/27/1/62
- Friston interview, "Bayesian brain computing and the free-energy
  principle," National Science Review May 2024 (notes Friston's
  view that AI should move toward "in-memory processing read as
  variational message passing on a graph").
  https://academic.oup.com/nsr/article/11/5/nwae025/7571549
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11060478/

**Predictive-coding associative memory (FEP-family AM substrate)**

- Salvatori, Song, Hong, Sha, Frieder, Xu, Bogacz, Lukasiewicz,
  "Associative Memories via Predictive Coding," NeurIPS 2021.
  arXiv:2109.08063. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7612799/
- Yoo & Wood, "BayesPCN: A Continually Learnable Predictive Coding
  Associative Memory," NeurIPS 2022. arXiv:2205.09930.
  https://arxiv.org/abs/2205.09930
- Tang et al., "Recurrent predictive coding models for associative
  memory employing covariance learning," PLOS Comp Bio 2023.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10132551/
- Tang et al., "Semantic and episodic memories in a predictive
  coding model of the neocortex," Sept 2025. arXiv:2509.01987.
  https://arxiv.org/pdf/2509.01987

**Sub-angle B: closure / architectural-constraint contributions**

- Karl, Kemeter, Dax, Sierak, "Position: Embracing Negative Results
  in Machine Learning," ICML 2024 oral. arXiv:2406.03980.
  https://arxiv.org/abs/2406.03980
  https://proceedings.mlr.press/v235/karl24a.html
- "Modern Hopfield Networks Require Chain-of-Thought to Solve
  NC^1-Hard Problems," Dec 2024. arXiv:2412.05562.
  https://arxiv.org/pdf/2412.05562
- Hu, Lin, Song, Liu, "On Computational Limits of Modern Hopfield
  Models," ICML 2024 (referenced in Awesome MHN repo).
  https://github.com/Event-AHU/Awesome_Modern_Hopfield_Networks
- "The Capacity of Modern Hopfield Networks under the Data Manifold
  Hypothesis," 2025. arXiv:2503.09518.
  https://arxiv.org/abs/2503.09518
- "Accuracy and capacity of Modern Hopfield networks with synaptic
  noise," 2025. arXiv:2503.00241.
  https://arxiv.org/pdf/2503.00241
- Carzaniga et al., "Practical Lessons on Vector-Symbolic
  Architectures in Deep Learning-Inspired Environments," NeSy 2025
  / PMLR 284:218-236.
  https://openreview.net/forum?id=5ZmvZkqyoy
  https://proceedings.mlr.press/v284/carzaniga25a.html
- "Inverse Scaling in Test-Time Compute," 2025. arXiv:2507.14417.
  https://arxiv.org/pdf/2507.14417
- "Scaling Laws Are Unreliable for Downstream Tasks: A Reality
  Check," 2025. arXiv:2507.00885.
  https://arxiv.org/html/2507.00885v1
- "Paradoxical increase of capacity due to spurious overlaps in
  attractor networks," 2025. arXiv:2510.17593.
  https://arxiv.org/pdf/2510.17593
- "Self-Organization and Spectral Mechanism of Attractor Landscapes
  in High-Capacity Kernel Hopfield Networks," 2025.
  arXiv:2511.13053. https://arxiv.org/pdf/2511.13053
- NeurIPS 2025 introduction of the Position Paper Track (general
  reference; track listing in NeurIPS 2025 trends report).
  https://intuitionlabs.ai/pdfs/neurips-2025-a-guide-to-key-papers-trends-stats.pdf

---

## Concrete ideas / recommendations

**Decision tree for the Phase 5 closure question:**

1. **Run the EFE-proxy diagnostic in sub-angle A.4 first.** It is a
   one-day experiment that distinguishes "substrate-level saturation"
   (no signal exists, raw energy isn't the issue) from "score-
   function-level saturation" (signal exists, raw energy is the wrong
   read-out). The diagnostic is *cheap* and *informative* and changes
   the framing of any closure document. If EFE-proxy stratifies
   branches, the closure becomes "we found the substrate is fine, the
   score function was wrong, here is the fix" — which is closer to a
   positive result. If EFE-proxy also fails to stratify, the closure
   becomes the stronger architectural claim "sharp-basin substrates
   resist this class of branching regardless of score function."

2. **Regardless of outcome, write the closure document in
   mechanism-explanation style (B.3 mode b).** The substrate-
   saturation finding has a single coherent geometric mechanism
   candidate (sharp basins collapse intermediate states needed for
   role-binding-prior traversal). Articulate that mechanism. The
   Pinnacle Sharpness paper (arXiv:2511.13053) is the closest
   prior-art analog and should be cited explicitly — it both
   validates the project's framing (sharp-basin trade-offs are a
   recognized object of study) and gives the writeup a clean
   intellectual lineage.

3. **Target NeurIPS 2026 Position Paper Track or NeSy 2026 as the
   submission venue.** Both accept this shape; NeSy is friendlier to
   empirical-lessons framings, NeurIPS Position Track to perspective
   framings. Neural Computation is the slower but higher-prestige
   journal option.

4. **If the EFE-proxy diagnostic succeeds, treat Phase 5 as
   *continued with a different score function*, not closed.** The
   reformulation would be "active-inference K-branch settling on
   FHRR+Hopfield substrate," with the EFE decomposition as the
   discriminator. This requires declaring an explicit generative
   model over role/content latents, which is a real architectural
   move but smaller than substrate replacement.

5. **If both raw-energy and EFE-proxy fail to stratify, treat the
   finding as a substrate constraint that *also* generalizes against
   PC-AM substrates and document the comparison.** This is the
   strongest version of the closure: not just "this substrate
   doesn't do this," but "this *family* of substrates doesn't do
   this." That framing maps cleanly onto the NC^1-hardness paper's
   style.

**Lower-priority but worth noting:**

- The "Reframing the Expected Free Energy" paper (Champion et al.
  2024, arXiv:2402.14460) shows that the canonical EFE
  decompositions are equivalent or bounded — which means the
  EFE-proxy diagnostic does *not* need to pick the "right"
  decomposition; reasonable proxies will be in the same equivalence
  class up to bounded reweighting. This lowers the design cost of
  the diagnostic.

- The Friston 2024 interview (PMC11060478) explicitly endorses
  "in-memory processing read as variational message passing on a
  graph" as the AI direction he wants the field to move in. The
  Neuro-AI substrate is in-memory processing on a graph (the
  FHRR+Hopfield+codebook composition). If a reframe in that
  language is natural, it positions the project well in the FEP
  community regardless of the Phase 5 outcome.

---

## Surprises

- **NeurIPS 2025 created a Position Paper Track.** This is new in
  2025 and is *exactly* the venue for the kind of contribution under
  discussion. Worth knowing about: it lowers the venue-uncertainty
  cost of the closure path significantly.

- **"Embracing Negative Results in Machine Learning" was an ICML 2024
  *oral*.** Not a workshop paper, not a position paper buried in a
  side track — an oral. The field has visibly shifted toward
  accepting this contribution shape in the last 18 months.

- **arXiv:2510.17593 ("Paradoxical increase of capacity due to
  spurious overlaps")** is a paper whose entire contribution is
  reframing a known "defect" as a useful mechanism. This is the
  *exact* rhetorical move available to Neuro-AI: substrate
  saturation at D=4096 with sharp basins is a defect under the K-
  branch energy-margin frame, but might be a *signature* of a useful
  property (e.g., the substrate's commitment to self-retrieval as a
  capacity-amplifying mechanism) under a different frame. Worth
  reading carefully if writing the closure document.

- **Sophisticated Inference (Friston 2021, arXiv:2006.04120) is
  conceptually closer to K-branch settling than anything else found.**
  It implements a tree search over *belief trajectories*, scored by
  recursive EFE. The Neuro-AI K-branch architecture, modulo
  substrate, is almost a 1-step sophisticated-inference variant.
  This is the cleanest active-inference lineage to claim if Phase 5
  is reformulated.

- **The VSA Practical Lessons paper (NeSy 2025) is a very direct
  precedent for "we ran this family of architectures hard, here are
  the empirical lessons."** Four lessons, three of which are
  essentially negative results about VSA family choices that don't
  matter as much as people thought (HRR vs MAP/HLB). If Neuro-AI's
  closure document is empirical-lessons style, this is the template.

- **"Modern Hopfield Networks Require Chain-of-Thought to Solve
  NC^1-Hard Problems" (arXiv:2412.05562)** prove an architectural
  limit of *exactly the substrate family Neuro-AI uses*. That paper
  exists. It is two months newer than the project's substrate
  decisions. A Neuro-AI closure document can cite it directly: "the
  fixed-substrate modern Hopfield family is provably limited; we
  document an additional limit specific to sharp-basin self-
  retrieving instantiations of this family." This is a clean
  positioning.

---

## Open questions

1. **Is the substrate-saturation finding *expressible* as a
   geometric statement?** Theorem-style framings (the most prestigious
   B.3 mode) require this. If the finding can be stated as "for
   self-retrieving Modern Hopfield + FHRR substrates with basin
   depth > τ, role-binding-prior traversal at distance d requires
   intermediate states that the substrate cannot expose," that
   would be a near-theorem. Worth attempting to write down before
   committing to mechanism-explanation framing.

2. **Does the EFE-proxy diagnostic actually need the substrate to
   expose posterior beliefs, or can it be computed from converged
   states alone?** The diagnostic proposed in A.4 uses posterior
   entropy over role-binding slots as the epistemic-value proxy. If
   the substrate doesn't naturally expose this posterior, the
   diagnostic is harder than one day. Worth scoping before
   committing.

3. **Is the K=1 ≥ K=4 finding actually about saturation, or about
   the bundling weights being mis-calibrated?** EFE-weighted
   bundling uses softmax(EFE), and EFE can be on a very different
   scale than raw energy. If the temperature on the current energy-
   weighted softmax is wrong, K=1 ≥ K=4 might be a temperature
   artifact rather than a saturation finding. The diagnostic should
   include a temperature sweep.

4. **Should the closure document explicitly recommend PC-AM as the
   alternative substrate, or stay neutral?** Recommending introduces
   a Phase 6+ commitment; staying neutral preserves optionality.
   The position-paper genre allows either move; the mechanism-
   explanation genre prefers neutral.

5. **What is the right relationship between the closure document and
   the project's STATUS.md?** A published closure paper would
   typically reference an open-source repo with the experimental
   evidence. The project already has this discipline. Worth
   deciding whether the writeup is a project artifact (lives in
   /reports/ + STATUS.md update) or an external artifact (arXiv
   preprint + workshop submission) — these are not mutually
   exclusive but the framing differs.

6. **Are there active-inference researchers who would co-author or
   pre-review a Neuro-AI reformulation paper?** Beren Millidge
   (Oxford / Conjecture) sits at the exact intersection of FEP +
   predictive-coding-AM + active-inference and has authored
   foundational papers on each (Whence the EFE, Universal Hopfield
   Networks, PC-AM). He is a plausible pre-reviewer for either the
   reformulation (A) or the closure (B) document.
   https://www.beren.io/

7. **Does the user want the "Phase 5 closure as architectural
   deliverable" framing to be *terminal* for the project, or
   *transitional* into a Phase 6 with a different substrate?** This
   determines whether the closure document recommends a successor
   (PC-AM, BayesPCN, active-inference-on-explicit-latents) or
   simply documents the constraint. The project's PROJECT_PLAN
   currently does not appear to specify Phase 6, so this is a
   genuine open question.
