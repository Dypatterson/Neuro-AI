# 02 — Active Inference / Predictive Coding as the Substrate for Branching

## Angle

Phase 5's K-branch design seeds each branch with an external prior (role
vs content) and asks which lands lower in energy. The branches collapse
to the same basin, and the log-prior fix manufactures the energy margin
by reshaping logits rather than by changing what is *retrieved*
(hit_role ≈ 0, rank_role ≈ random; report 061). The angle here:
**treat each branch as a generative hypothesis and let local prediction
error (PE) / free energy drive its evolution.** The branch "decision"
becomes the geometry of PE-modulated settling and PE-prioritised replay,
not an external pick. This preserves the FHRR + MHN substrate and the
anti-homunculus filter, because the prioritisation is expressed as
softmax / Boltzmann dynamics over local scalars, not as an `if PE >
threshold then replay` rule.

## Key findings

1. **Predictive coding on Hopfield substrates is already a real
   research line, with strictly local update rules.** The Whittington–
   Bogacz (2017) tutorial defines value nodes φ and error nodes ε with
   updates `τ dε/dt = φ − W σ(φ_above) − b − ζε` and weight rule
   `γ dW/dt = ε σ(φ_above)`. The error node is what's local: ε_i lives
   at neuron i and is computed from the gap between i's value and the
   top-down prediction. Salvatori & Pinchetti et al. (2021,
   *Associative Memories via Predictive Coding*, arXiv:2109.08063) and
   the more recent *Online Training of Hopfield Networks using
   Predictive Coding* (Yoo et al., arXiv:2406.14723, 2024) explicitly
   port this to Hopfield-style retrieval and demonstrate **strictly
   better recall of correlated patterns than Modern Hopfield**, which is
   the exact regime the Neuro-AI substrate is in (post-death survivors
   are correlated by design). The 2024 paper *explicitly flags* that
   adapting the rule to MHN softmax retrieval is open.

2. **The Mattar–Daw EVB formula has a clean dynamic form.** EVB(s_k,a_k)
   = Gain(s_k,a_k) × Need(s_k). Gain ≈ TD/PE-style improvement from
   updating that memory; Need is the expected future occupancy
   (successor representation column). Yuan & Mattar 2021 (Improving
   Experience Replay with SR, arXiv:2111.14331) and Antonov & Dayan 2023
   (Exploring Replay, bioRxiv) both express the prioritisation as a
   **softmax / sampling distribution** rather than argmax — i.e. it is
   already in the right shape for the anti-homunculus filter. The
   homunculus form is "pick the highest-EVB transition"; the dynamic
   form is "replay budget per atom is p_i ∝ exp(β · gain_i · need_i)".

3. **Input-driven plasticity (IDP) for Hopfield (Betteti et al.,
   *Science Advances* 11, eadu6991, Apr 2025; arXiv:2411.05849)
   *reshapes the energy landscape from the cue itself.*** Their
   reformulation in modern Hopfield form is
   `τ_x ẋ = −x + M_x Ψ_x(y ⊙ α)`, `τ_y ẏ = −y + M_y Ψ_y(x)`,
   `τ_α α̇ = −α + M_α Ψ_α(u)`. The saliency α_μ = ⟨ξ^μ, u⟩ is *exactly*
   the kind of local prediction-error / alignment scalar we want, and
   the Hadamard `y ⊙ α` is a precision-weighted retrieval gate. This is
   *not* the same as the project's log-prior spike: α modulates the
   *gradient* (so the basin geometry changes), not just the read-out
   logits. This is also the closest result to a published mechanism for
   "the prior is the energy-landscape modulator, not the basin picker."

4. **Friston-style precision weighting maps cleanly onto MHN's
   inverse-temperature β.** Precision = inverse variance of an error
   channel; in MHN, β controls how sharply softmax separates basins.
   The active-inference frame says: precision is **learned per channel
   and per cue**, not global. A learnable, per-atom or per-role-band
   β_role(c) is the FHRR/MHN-native form of "attention as precision."

5. **No 2024–2026 paper directly combines FHRR with active inference**
   (the closest, Bricken et al. *Geometric Priors for Generalizable
   World Models via VSA*, arXiv:2602.21467, uses FHRR for transition
   modelling but not free energy). This is a *gap*, not a contradiction
   — the project is in unclaimed territory if it goes here.

## Promising leads

- **PCN-trained codebook** (Salvatori et al. 2021; Yoo et al. 2024) for
  the *consolidation* step (allowed offline by project rules), with PE-
  driven settling preserved at runtime.
- **EVB-shaped replay budget**, where the Need term comes from MHN
  retrieval co-activation statistics (a substrate-native successor-
  representation analogue: row j of the co-activation matrix
  approximates "how often atom j follows atom i in retrieval").
- **IDP for the K-branch substrate**: make each branch's saliency
  vector α^(k) computed from a different feature of the cue (its role-
  unbound part vs its content part). The branches then settle in
  *different reshaped landscapes*, not the same one.
- **Precision-weighted MHN**: learn β_role(c) per cue from a running PE
  estimate, allowed in the consolidation pass.

## Concrete ideas for the project

Each idea below comes with an explicit anti-homunculus check.

### Idea A — Per-atom PE as the saliency that reshapes the MHN landscape (IDP-style)

*Mechanism.* For cue q and atom ξ_i, define
`PE_i = 1 − cos(ξ_i, q_predicted)`,
where `q_predicted = Σ_j p_j(t) ξ_j` is the current settling-state
prediction. The MHN retrieval becomes
`p_i(t+1) ∝ exp(β · ⟨ξ_i, q⟩ · α_i(t))` with
`α_i(t+1) = (1−η)α_i(t) + η · PE_i(t)` (low-pass filtered PE).
PE rises when an atom's content disagrees with the current consensus,
and that atom gets gated *up* — precisely the precision-weighted error
unit. The Hadamard `⟨ξ_i,q⟩ · α_i` reshapes the energy landscape per
cue, not per supervisor.
*Anti-homunculus check.* α_i is per-atom local; PE_i is computed only
from atom-i's content and the current consensus vector (which is
already on-substrate). No supervisor reads any global quantity to
trigger a rule. Branches differ because they start with different α(0)
seeds — e.g. α^(role)(0) = role-binding alignment, α^(content)(0) =
content alignment — but evolution is purely the local IDP dynamic.
This **replaces the project's log-prior spike with a landscape-shaping
prior**, which is exactly what the report-061 diagnosis called for.

### Idea B — Branch budget as softmax over EVB-style scores (no argmax pick)

*Mechanism.* Replace "run K=4 equal-budget branches" with a continuous
budget vector b(t) ∈ Δ^K satisfying
`b_k(t+1) ∝ exp(β_b · gain_k(t) · need_k(t))`,
where `gain_k = ΔE_k(t) − ΔE_k(t−1)` (how much the branch's energy
descent is still doing useful work) and `need_k =
Σ_j p_j(t)^(k) · freq_j` (Phase-4-style frequency, which is the
substrate's native successor-representation analogue). Each branch
takes `b_k(t)` iterations per outer step. Branches that have plateaued
(low gain) or that are settling into rare-atom basins (low need)
naturally lose budget; branches that are still moving fast through
common-atom regions get more.
*Anti-homunculus check.* No `if X then Y` rule. The budget is a
softmax over substrate-native scalars (energy delta, frequency-weighted
state mass). No branch is "chosen as the winner"; budget reallocation
is the dynamic itself. The "decision" lives in `b(t)` as a Boltzmann
distribution.

### Idea C — PE-modulated Hebbian consolidation (PE as learning rate scalar)

*Mechanism.* During the consolidation pass, the Hebbian update for
atom ξ_i from co-activation with cue q becomes
`Δξ_i = η · PE_i · (q − ⟨ξ_i,q⟩ ξ_i)`,
i.e. atoms that *failed to predict* the cue get pulled harder toward
it. This is allowed by project rule (error-driven only in batch offline
passes). Atoms that already predict the cue well (PE_i small) get
almost no update — they're already "explaining" that part of input
space, so they stop being reinforced for it. This pushes the
substrate toward **error-distributed atoms**, the regime where
distinct basins form.
*Anti-homunculus check.* PE_i is per-atom local. No supervisor decides
which atoms get updated. The update is Hebbian-shaped, modulated by a
local scalar. This is the canonical Bogacz 2017 local rule, just lifted
to atoms rather than synapses.

### Idea D — Replay sampling as softmax over PE × frequency

*Mechanism.* Instead of replaying all atoms uniformly, sample the
replay batch from `p_i ∝ exp(γ · PE_i · freq_i^α)`. This is exactly
the Mattar–Daw EVB form (gain × need → PE × frequency), in its
softmax (not argmax) form. The branches that benefit are those whose
basins contain atoms with high PE_i: replay actively fixes the parts
of the substrate that are mispredicting, which directly addresses
"role-target basins don't exist" — replay would preferentially
strengthen role-bundled patterns precisely when they are not yet
basins.
*Anti-homunculus check.* The sampling distribution *is* the
mechanism; there is no thresholded routing. p_i is a continuous local
function. No global comparator decides what to replay.

### Idea E — Branches as parallel hypothesis particles, not parallel basin probes

*Mechanism.* Reframe K branches as a population of generative
hypotheses x^(k) settling under the IDP dynamics from Idea A. Compute
per-branch *free energy* F_k = E(x^(k)) + KL(p^(k) ‖ prior^(k)) and
let branches **die / clone** by a continuous birth-death process with
rate proportional to softmax(-β_F · F_k) — like a particle filter
over hypotheses. After T steps, the *ensemble distribution* over
branches is the "answer", not any single branch's basin.
*Anti-homunculus check.* Birth-death is a local stochastic rule; the
softmax is a distribution, not a pick; the ensemble is the answer.
This is the cleanest active-inference framing — but it is also the
biggest engineering jump from the current K-branch code.

## Surprises

- **The Bogacz local rules are already shaped exactly like Hebbian
  with a per-unit gain.** This means "PE as Hebbian learning rate
  scalar" (Idea C) is not a new mechanism — it is the canonical
  predictive-coding update written in atom-substrate language. The
  project rule "error-driven only in batch offline passes" was framed
  defensively against catastrophic forgetting; PCN literature suggests
  it is also the *right* shape for the runtime/consolidation split.
- **The 2025 IDP paper has independently invented exactly the
  mechanism the project's report 061 said it needed**: a prior that
  reshapes the landscape rather than the logits. The math is published.
- **EVB has a softmax form in the literature already** (Yuan & Mattar
  2021; Antonov & Dayan 2023). The project's anti-homunculus filter
  reads at first like a strong constraint on EVB-style mechanisms;
  actually the constraint is already standard in the field.
- **No paper combines FHRR with active inference.** This is a real
  gap. The project is closer to the frontier than the framing implies.

## Sources

- [Whittington & Bogacz 2017 — *An Approximation of the Error
  Backpropagation Algorithm in a Predictive Coding Network with Local
  Hebbian Synaptic Plasticity*, Neural Computation](https://direct.mit.edu/neco/article/29/5/1229/8261/An-Approximation-of-the-Error-Backpropagation)
- [Bogacz 2017 — *A tutorial on the free-energy framework for modelling
  perception and learning*, J. Math. Psychol.](https://pubmed.ncbi.nlm.nih.gov/28298703/)
- [Salvatori, Song, Hong, Sha, Frieder, Xu, Bogacz, Lukasiewicz 2021 —
  *Associative Memories via Predictive Coding*,
  arXiv:2109.08063](https://arxiv.org/abs/2109.08063)
- [Yoo et al. 2024 — *Online Training of Hopfield Networks using
  Predictive Coding*, arXiv:2406.14723](https://arxiv.org/html/2406.14723v1)
- [Mattar & Daw 2018 — *Prioritized memory access explains planning
  and hippocampal replay*, Nature Neuroscience](https://www.nature.com/articles/s41593-018-0232-z)
- [Yuan & Mattar 2021 — *Improving Experience Replay with Successor
  Representation*, arXiv:2111.14331](https://arxiv.org/abs/2111.14331)
- [Antonov & Dayan 2023 — *Exploring Replay*, bioRxiv
  2023.01.27.525847](https://www.biorxiv.org/content/10.1101/2023.01.27.525847v1)
- [Betteti, Baggio, Bullo, Zampieri 2025 — *Input-driven dynamics for
  robust memory retrieval in Hopfield networks*, Science Advances 11,
  eadu6991 (preprint arXiv:2411.05849)](https://www.science.org/doi/10.1126/sciadv.adu6991)
- [Millidge, Tschantz, Buckley 2022 — *Universal Hopfield Networks: A
  General Framework for Single-Shot Associative Memory Models*, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614148/)
- [Alonso & Krichmar 2024 — *A Sparse Quantized Hopfield Network for
  Online-Continual Memory*, Nature Communications (arXiv:2307.15040)](https://arxiv.org/abs/2307.15040)
- [Pathak, Agrawal, Efros, Darrell 2017 — *Curiosity-driven Exploration
  by Self-supervised Prediction*, arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
- [Friston (review on precision-weighting / attention as gain)
  via Frontiers 2011 / Feldman & Friston 2010](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2011.00218/full)
- [Bricken et al. 2026 — *Geometric Priors for Generalizable World
  Models via VSA*, arXiv:2602.21467](https://arxiv.org/html/2602.21467)
- [Champion et al. 2022 — *Branching Time Active Inference: Empirical
  Study and Complexity Class Analysis*, Neural Networks](https://www.sciencedirect.com/science/article/pii/S0893608022001824)
