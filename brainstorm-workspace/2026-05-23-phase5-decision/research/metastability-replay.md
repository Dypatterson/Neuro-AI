# Metastability as a Replay-Prioritization Signal — 2023-2026 Literature Scan

**Research angle (Phase 5 brainstorm, pair #4):** Should the Neuro-AI
substrate carry per-atom retrieval metastability `c_i = w_i · (1 − max_j w_j)`
(or a trajectory analog) as an EMA `m_i`, and use it to modulate ReplayStore
priority via `(1 + κ·m_i)`? Headline metric: Δ meta-stable-rate at W=3.

**Why this scan:** the 1-seed smoke for the trajectory-c_i reformulation
returned m_max ≈ 0.0082, below the 0.05 pre-committed gate. Before we either
spend more seeds chasing it or drop the pair, we want fresh grounding from
the 2024-2026 literature on whether (a) the signal shape we picked has
precedent and analytic justification, (b) the diagnostic-actuator coupling
pattern is shared by other people doing this for real, and (c) there are
alternative substrate-pure signals that dominate metastability on this kind
of substrate.

---

## TL;DR for the decision

1. **The general pattern (use a per-item uncertainty/surprise/novelty
   statistic to modulate replay priority continuously) is exactly what
   the 2024-2026 field is converging on.** UPER (Sutton et al. RLC 2025,
   arXiv 2506.09270), ReaPER (Pleiss et al. 2025, arXiv 2506.18482),
   and SuRe (Surprise-Driven Prioritised Replay for Continual LLM Learning,
   arXiv 2511.22367) all do this, and all of them frame it explicitly as
   *modulating the sampling distribution* rather than *branching on
   thresholds* — anti-homunculus-compatible.

2. **The biological side has gotten dramatically firmer in 2024-2025.**
   Yang/Sun/Buzsáki (Science 2024) showed that awake sharp-wave ripples
   *select* which experiences get consolidated, and Mattar/Daw-style
   "gain × need" prioritization has been extended to incorporate
   reward-prediction error (Nature Comms 2025) and information gain /
   epistemic foraging (Németh et al. bioRxiv 2025). The biological
   prioritization signal is closer to "surprise × need" than to
   "metastability of the retrieve weights."

3. **The trajectory-c_i shape (max-over-time minus final) does have
   precedent**, but the closest match is **hindsight-TD-error / regret
   minimization** (Liu et al. ReMERN/ReMERT, NeurIPS 2021). It is also
   structurally similar to **lateral-inhibition-resolved competition
   traces** in WTA dynamics. So the shape itself is not heterodox — it
   just needs to be expressed as "how much did this atom compete for the
   slot before losing" rather than "metastability."

4. **The substrate-noise-floor risk is real and not specific to
   metastability.** On a substrate where all atoms are tied near the
   noise floor of similarity, *any* per-atom statistic computed from
   softmax weights at the chosen β will be degenerate. The literature's
   answer is usually one of: (a) raise β at retrieval time to amplify
   the margin, (b) use **input-driven settling dynamics** (Betteti et
   al. Science Advances 2025) so the energy landscape is shaped by the
   query, not by the stored patterns alone, (c) compute the signal at
   a different stage (e.g. on the *gradient* of E w.r.t. each atom, not
   on the softmax output). Pair #4 should at least pre-commit to which
   of these it would adopt if the signal is at the floor.

5. **Concrete strongest alternatives surfaced by the scan:**
   - **Hindsight regret** `r_i = max_t w_i(t) − w_i(T)` (the literal
     reformulation, but renamed and rederived from Liu et al. 2021),
   - **Epistemic-uncertainty replay** in the UPER style: ensemble two
     Hopfield heads with different β or different orderings, use
     disagreement as `m_i`,
   - **Need × gain** in the Mattar/Daw sense: `m_i = p(retrieve_i) ×
     E[ΔE_consolidate | replay_i]`, both expressible as local geometry,
   - **Recall-gated plasticity** (Tyulmankov et al. eLife 2024): only
     consolidate if the recall is high-margin, which is *the inverse* of
     metastability and is actually closer to what the Neuro-AI substrate
     already does implicitly.

---

## What is metastability as a memory signal?

### Definition lineage

The expression `c_i = w_i · (1 − max_j w_j)` is the per-item "edge of
decision" magnitude: it peaks for items that *would have won if not for one
slightly stronger competitor*. It is closely related to:

- **Margin-based confidence** in classification (`p_top1 − p_top2`),
  but with the asymmetry that it tracks the *runner-up* mass rather than
  the gap itself.
- **Entropy on a single coordinate** of the softmax. Note that
  `c_i` peaks at `w_i ≈ 0.5` when `max ≈ 0.5` (the two-way tie), so it
  has the same "max at maximum confusion" shape as bit entropy.
- **Polarization-of-attention** in transformer-interpretability work;
  several recent papers use the entropy of attention weights as an
  interpretability signal but to my knowledge none use the per-token
  product `w · (1 − max)` as a downstream learning signal.

In the dynamical-systems / hippocampus literature, "metastability" means
something different and important: it refers to **transient occupancy of
near-attractor states that the system leaves before fully settling**.
The two senses are related (a high-c_i atom is sitting in a
near-but-losing basin) but the second sense is older and load-bearing.
The 2024-2025 hippocampus-replay literature uses the dynamical-systems
sense almost exclusively.

### Dynamical-systems metastability and consolidation (2024-2025)

- **Co-existence of synaptic plasticity and metastable dynamics in a
  spiking model of cortical circuits** (Recanatesi et al., PMC10723399,
  2024) shows that metastable transitions between cluster attractors can
  themselves drive synaptic consolidation in a recurrent network. The
  signal there is dwell-time on each attractor, integrated over plasticity
  time-scales.
- **A Quasi-Stationary Approach to Metastability in a System of Spiking
  Neurons with Synaptic Plasticity** (arXiv 2403.09678, 2024) gives the
  formal framework: metastable states are quasi-stationary distributions
  of the neural dynamics, and their escape rates set the consolidation
  rate. Conceptually this is the cleanest mapping to "use metastability
  as a rate-modulating signal" and it is rate-modulating by construction.
- **Mesoscopic description of hippocampal replay and metastability in
  spiking neural networks with short-term plasticity** (PMC9822116, 2023)
  treats replay events themselves as metastable transitions; replay
  prioritization in their model is determined by which attractors are
  energetically closest, with no explicit "decider" — anti-homunculus
  clean.

**Implication for Neuro-AI:** the dynamical-systems sense maps cleanly
onto MHN settling dynamics. A per-atom dwell-time signal — "how many
settling steps was atom i the soft argmax before being displaced" —
would (a) be a per-atom local quantity, (b) not require an external
threshold, (c) directly express the trajectory information we are
already losing by reading softmax weights only at the final step.
**This is essentially what the trajectory-c_i formulation is trying to
compute**, but expressing it as dwell-time-over-iterations rather than
max-minus-final makes its connection to the biology cleaner and avoids
the optical mismatch with the static-c_i formulation.

### Engram / Hopfield-side use of "metastable" (2024-2025)

- Wu et al. *Provably Optimal Memory Capacity for Modern Hopfield Models*
  (arXiv 2410.23126) — formal analysis of the basin structure; explicitly
  treats metastable basins as the failure mode of dense associative
  memory and shows how spherical-code arrangements eliminate them at
  high capacity.
- *Input-driven dynamics for robust memory retrieval in Hopfield
  networks* (Betteti, Baggio, Bullo, Zampieri, Science Advances 2025,
  arXiv 2411.05849) — proposes an external input that shapes the energy
  landscape during settling, reducing misclassification under noise.
  This is *exactly* the right tool to handle the "all atoms at noise
  floor" failure mode the smoke surfaced: if energy gaps are tiny, drive
  them larger from the input side instead of trying to extract signal
  from the small gaps.
- *Hopfield Encoding Networks* (ICLR 2024) — explicitly motivated by
  the observation that metastable states are the dominant failure mode
  in raw modern Hopfield retrieval on real-world image collections.
  Their fix is to learn an encoder before the Hopfield step so the
  stored basins are well-separated. Conceptually adjacent to what
  Phase 4 codebook is trying to do.

---

## Replay prioritization 2024-2026 — state of the art beyond PER

The state of the art has fragmented into ~4 streams:

### 1. Epistemic-vs-aleatoric uncertainty (most directly relevant)

**Uncertainty Prioritized Experience Replay (UPER)** — Sutton, Liu, et al.,
RLC 2025 (arXiv 2506.09270, RLJ paper at
https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_45.pdf). They argue
that vanilla PER (Schaul 2015) prioritizes on absolute TD error, which
treats irreducible noise (aleatoric uncertainty) the same as
reducible-by-learning noise (epistemic uncertainty). They estimate
both via an ensemble of QR-DQN agents and prioritize on an
information-gain functional that's roughly `epistemic^2 / (epistemic +
aleatoric)`. **Crucially: the priority is *modulated* by the signal,
not branched on it.** Atari-57 sample efficiency beats PER+QR-DQN
baselines.

**Reliability-Adjusted Prioritized Experience Replay (ReaPER)** —
Pleiss, Sutter, Schiffer (arXiv 2506.18482, 2025). Same flavor:
estimate the reliability of the TD-error estimate itself, downweight
priorities when the error estimate is unreliable. Beats PER on Atari-10.

**Take for pair #4:** Both papers explicitly reject "TD error is enough"
in favor of a more decomposed signal. Our c_i (or hindsight regret r_i)
sits naturally in the same conceptual slot: an estimator of *epistemic-
like* uncertainty about which atom is the right one. The argument
"vanilla PER over-prioritizes noisy stored atoms" is the closest
analog in our setting to "raw retrieval winners over-prioritize
already-confident memories that don't need consolidating," which is
the original motivation for pair #4. **The literature now backs the
overall framing.**

### 2. Surprise / prediction-error driven (closest to biology)

**SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning**
(arXiv 2511.22367, late 2025). For continual-learning of LLMs, replay
priority = per-token prediction loss at insertion time. Maps onto the
2024-2025 hippocampus literature where prediction error is the dominant
prioritization signal (see §"Biological precedent" below). Anti-
homunculus PASS (the loss is a per-token local statistic).

**Take for pair #4:** Surprise on the *content prior* side (how much
the consolidated reconstruction disagrees with the encoded item) is a
candidate alternative `m_i`. In our substrate this would be ΔE between
the role-prior energy and the content-prior energy at consolidation
time — which is literally what Phase 5's design-spec headline metric
already measures (notes/emergent-codebook/phase-5-unified-design.md:
256-281). **If pair #4 were reframed as "use the Phase-5-headline ΔE
itself as the per-atom replay signal" instead of c_i, it would
collapse the new mechanism and the headline metric into one, which is
both cleaner and more honest.**

### 3. Maximally-interfered retrieval (Aljundi line)

The original MIR paper (Aljundi et al., NeurIPS 2019, arXiv 1908.04742)
selects for replay the samples whose loss would *grow most* under the
next gradient step. Aljundi's recent (2024) work is less about the
replay-prioritization mechanism and more about (a) selective parameter
update for big foundation models (CVPR 2024) and (b) "Continual
Learning: Applications and the Road Forward" (TMLR 2024), a synthesis.
The MIR-style approach has not been deprecated — it survives as one
strong baseline — but the field has broadly moved to uncertainty
estimators as the prioritization signal, both because MIR is expensive
(requires a forward simulation per candidate) and because the
information-theoretic justifications for UPER are cleaner.

**Take for pair #4:** MIR-style "which atom would lose most from the
next consolidation step" is the closest classical analog to our
metastability signal. It already has anti-homunculus PASS in its
original derivation (Aljundi explicitly argues the prioritization is
*derived from the loss landscape*, not chosen by a supervisor). Worth
re-reading.

### 4. Diffusion / generative replay & regularization-based

These streams (DERpp, RAR, the diffusion-replay line from Hu et al.
arXiv 2411.10809) prioritize replay implicitly through how the
generative model is sampled. Less directly relevant to pair #4 because
we have an explicit replay store and the question is how to *sample
from it*, not how to *regenerate items*.

### Biological precedent — how does the hippocampus actually prioritize?

This is where 2024-2025 has been most informative:

- **Yang, Sun, Huszár, Hainmueller, Kiselev, Buzsáki — *Selection of
  experience for memory by hippocampal sharp wave ripples* (Science
  2024, vol 383 pp 1478-1483).** Awake SWRs *tag* events; tagged events
  get repeatedly replayed during sleep SWRs and become long-term
  memories. The tagging is driven by salience signals (reward,
  novelty), not by post-hoc decoding. This is the strongest biological
  argument in years that prioritization is *built in to the replay
  generator*, not applied to a buffer.
- **Van der Meer & Bendor — *Awake replay: off the clock but on the
  job* (Trends in Neurosciences 48(4):257-267, 2025).** Review arguing
  that awake replay is for *offline tagging and prioritized
  consolidation*, not online behavior. Reinforces "prioritization
  signal is computed at the awake-replay event, not later."
- **Mattar-Daw 2018 follow-up — Némedy et al., *Between planning and
  map building: Prioritizing replay when future goals are uncertain*
  (Neuron 2025).** Extends gain × need to include goal uncertainty;
  replay priority increases with goal entropy.
- **Németh, Chartouny, Jedlovszky, Freire, Khamassi — *The hippocampus
  as an epistemic forager* (bioRxiv 2025.10.31.685837).** Combines
  reward and information gain; introduces the **Epistemic Replay
  Algorithm (ERA)**. Replay priority is the *sum* of expected reward
  and expected information gain — both are continuous local signals,
  no arbitration.
- **Post-learning replay of hippocampal-striatal activity is biased by
  reward-prediction signals** (Nature Comms 2025, s41467-025-65354-2).
  Empirically confirms that replay frequency for a state scales with
  its reward-prediction error.
- **How prediction error drives memory updating: role of locus
  coeruleus–hippocampal interactions** (Trends in Neurosciences 2025,
  S0166-2236(25)00189-4). Small prediction errors → editing of
  existing memories. Large prediction errors → formation of new
  memories. The *signal magnitude itself sets the regime* — there is
  no decision module that picks "edit" vs "form new." **This is the
  best biological example I've seen of a diagnostic-actuator coupling
  done the way Neuro-AI requires it.**

**Synthesis for pair #4:** the biological consensus signal is
**surprise / prediction error** (instantiated through LC-noradrenergic
or LC-dopaminergic neuromodulation). Metastability of retrieve weights
is a *poor proxy* for this, because metastability is high when the
system *almost-knew* something, while surprise is high when the system
*didn't know* something. These differ in sign on the most informative
cases.

This is a meaningful tension. The Neuro-AI substrate could legitimately
chase either, but if "biological precedent" is one of the load-bearing
arguments for pair #4, **surprise is more defensible than metastability**.
The current "metastability EMA" framing should be challenged with
"why not prediction-error EMA instead?"

---

## Diagnostic-actuator coupling examples

How have other people coupled a measured signal to a process rate
dynamically, without an if/then arbitration? A few clean examples:

1. **LC-noradrenaline / LC-dopamine in the hippocampus** (How prediction
   error drives memory updating, 2025). Prediction error magnitude
   sets the rate of LC firing; LC firing modulates the gain of
   hippocampal plasticity. Small PE → small LC → small gain change →
   small edit. Large PE → large LC → large gain change → larger
   modification. The "decision" between edit and form-new is *the
   shape of the gain curve*, not a branch.

2. **Recall-gated plasticity** (Tyulmankov, Litwin-Kumar et al., eLife
   2024 reviewed preprint, PMC11257680). Synaptic consolidation rate
   is multiplied by recall quality at the synapse: bad recall → low
   consolidation, good recall → high consolidation. **This is the
   closest formal analog to what pair #4 is trying to do**, and it has
   the right anti-homunculus shape: the gating is local and
   continuous, not a routing decision. Tyulmankov has a closed-form
   analysis showing the recall-gating selectively consolidates
   reliable signals.

3. **Engram dynamics with inhibitory plasticity** (Tomé, Lassagne,
   Sanyal et al., Nature Neuroscience Jan 2024, s41593-023-01551-w).
   Engrams transition from unselective to selective as inhibitory
   plasticity sculpts the engram boundary. The "decision" that some
   neurons stay in the engram and others drop out is a result of
   competing inhibition, not a supervisor.

4. **Diffusion of neuromodulators for temporal credit assignment**
   (arXiv 2603.08949). Credit signal is a literal diffusing chemical
   in a recurrent network. The "decision" of which synapses get more
   credit is the chemical's diffusion kernel, period.

5. **Selective consolidation via recall-gated plasticity** (Tyulmankov,
   et al., already cited). Worth listing twice because the analytic
   treatment is the clearest.

**The clean recipe these examples share:**
- A locally-computable signal (LC firing rate, recall margin, attention
  weight magnitude),
- Multiplicatively gates a rate constant (plasticity rate,
  consolidation probability, replay sampling probability),
- The rate constant feeds into a dynamic that would happen anyway —
  the modulation just changes its speed.

Pair #4's `(1 + κ·m_i)` priority modulation has the right *form* in this
framework. The only question is whether `m_i = trajectory-c_i` is the
right *signal*, given that prediction error / recall margin / hindsight
regret are equally well-formed and have stronger biological and
RL-literature precedent.

---

## Trajectory-c_i precedent — does "max minus final" have a name?

Yes, three reasonably clear precedents:

### 1. Hindsight TD error / regret (closest match)

Liu, Zhang, Hu, Yang, *Regret Minimization Experience Replay in Off-Policy
Reinforcement Learning* (NeurIPS 2021, arXiv 2105.07253; introduces
**ReMERN** and **ReMERT**). The prioritization weight is constructed
from the regret a sample carries relative to the optimal action:
roughly "how much value did the agent leave on the table by *not*
picking what was actually the best action with hindsight."

The structural analogy with trajectory-c_i `= max_t w_i(t) − w_i(T)` is
exact: *what was the maximum confidence we had in atom i over the
settling trajectory, minus the final confidence*. An atom that peaks
high and then loses is exactly a "near-miss with hindsight."

**Recommendation:** rename the signal. Call it `regret_i` or
`hindsight_w_i`. Both are clearer than "trajectory metastability" and
both have a clean RL provenance.

### 2. Competition-trace / dwell-time in WTA / spiking dynamics

The mesoscopic-replay paper (PMC9822116, 2023) and the recanatesi
"co-existence" paper (PMC10723399, 2024) both compute essentially
"how many time bins did neuron i lead a transient cluster" as their
metastability/dwell signal. This is the spiking-network equivalent
of integrating w_i(t) over settling iterations and seeing where atom i
contributed.

A natural variant: `dwell_i = (1/T) Σ_t [argmax_j w_j(t) == i]`. This is
fully on-trajectory and avoids the "max minus final" subtraction.
Possibly more robust at the noise floor than max-minus-final because
it doesn't depend on a single value of w at one timestep.

### 3. Max-over-time pooling in attention literature

Several attention-mechanism papers use max-over-time pooling of
attention weights as a salience signal (e.g. in document QA). This is
the "max" half of trajectory-c_i. Less directly relevant because they
don't subtract the final weight, but worth mentioning that the "max
over the settling trajectory" notion has independent existence.

---

## Key papers (with URLs / arXiv IDs)

### Algorithmic — replay prioritization
- **UPER**: Sutton et al., *Uncertainty Prioritized Experience Replay*,
  RLC 2025 / arXiv 2506.09270.
  https://arxiv.org/abs/2506.09270 ·
  https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_45.pdf
- **ReaPER**: Pleiss, Sutter, Schiffer, *Reliability-Adjusted Prioritized
  Experience Replay*, arXiv 2506.18482 (2025).
  https://arxiv.org/abs/2506.18482
- **SuRe**: *Surprise-Driven Prioritised Replay for Continual LLM
  Learning*, arXiv 2511.22367 (late 2025).
  https://www.arxiv.org/pdf/2511.22367
- **ReMERN/ReMERT**: Liu et al., *Regret Minimization Experience Replay
  in Off-Policy Reinforcement Learning*, NeurIPS 2021 / arXiv 2105.07253.
  https://arxiv.org/abs/2105.07253
- **MIR**: Aljundi et al., *Online Continual Learning with Maximally
  Interfered Retrieval*, NeurIPS 2019 / arXiv 1908.04742.
  https://arxiv.org/abs/1908.04742
- **MERS**: *Leveraging Complementary Embeddings for Replay Selection
  in Continual Learning with Small Buffers*, arXiv 2604.08336.
- Original **PER**: Schaul et al., arXiv 1511.05952.

### Algorithmic — Hopfield / memory
- **Input-driven dynamics for robust memory retrieval in Hopfield
  networks** — Betteti, Baggio, Bullo, Zampieri. Science Advances 2025;
  arXiv 2411.05849.
  https://www.science.org/doi/10.1126/sciadv.adu6991 ·
  https://arxiv.org/abs/2411.05849
- **Exploring the Temperature-Dependent Phase Transition in Modern
  Hopfield Networks** — arXiv 2311.18434 (showed β phase transition;
  important for noise-floor regime selection).
- **Provably Optimal Memory Capacity for Modern Hopfield Models** —
  Wu et al., arXiv 2410.23126.
- **Hopfield Encoding Networks** — ICLR 2024, OpenReview
  d582d629de2930168548e6650daa61da1a3cfe35.
- **Autonomous retrieval for continuous learning in associative memory
  networks** — PMC12418250 (2025).

### Algorithmic — continual learning structure
- *Continual Learning: Applications and the Road Forward* — Verwimp,
  Aljundi et al., TMLR 2024.
- *Overcoming the Stability Gap in Continual Learning* — Hess et al.
  arXiv 2306.01904 / OpenReview sSyytcewxe at ICLR 2024.
- *Flashbacks to Harmonize Stability and Plasticity* — arXiv 2506.00477
  (2025).
- *Active Dendrites Enable Efficient Continual Learning in
  Time-To-First-Spike Neural Networks* — arXiv 2404.19419 (2024).
- *Theories of synaptic memory consolidation and intelligent plasticity
  for continual learning* — arXiv 2405.16922 (2024 review).

### Biological — replay & consolidation
- **Yang, Sun, Huszár, Hainmueller, Kiselev, Buzsáki**, *Selection of
  experience for memory by hippocampal sharp wave ripples*. Science 383,
  1478-1483 (2024). DOI 10.1126/science.adk8261.
- **Van der Meer & Bendor**, *Awake replay: off the clock but on the
  job*. Trends in Neurosciences 48(4):257-267 (2025).
  https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(25)00037-2
- **Némedy et al.** (Mattar group), *Between planning and map building:
  Prioritizing replay when future goals are uncertain*. Neuron (2025).
  https://www.cell.com/neuron/abstract/S0896-6273(25)00709-3
- **Németh, Chartouny, Jedlovszky, Freire, Khamassi**, *The hippocampus
  as an epistemic forager: When curiosity and reward jointly steer
  exploration and hippocampal replay*. bioRxiv 2025.10.31.685837.
- **Post-learning replay of hippocampal-striatal activity is biased by
  reward-prediction signals**. Nature Communications (2025)
  s41467-025-65354-2.
- **How prediction error drives memory updating: role of locus
  coeruleus–hippocampal interactions**. Trends in Neurosciences (2025)
  S0166-2236(25)00189-4.
- **Large sharp-wave ripples promote hippocampo-cortical memory
  reactivation and consolidation during sleep**. Neuron (2025)
  S0896-6273(25)00756-1.
- **Replay and Ripples in Humans** — Annual Reviews 2024,
  10.1146/annurev-neuro-112723-024516.

### Biological — engram / synaptic consolidation
- **Tomé, Lassagne, Sanyal et al.**, *Dynamic and selective engrams emerge
  with memory consolidation*. Nature Neuroscience (Jan 2024).
  DOI 10.1038/s41593-023-01551-w.
- **Tyulmankov, Litwin-Kumar et al.**, *Selective consolidation of
  learning and memory via recall-gated plasticity*. eLife 2024.
  PMC11257680.
- **Organizing memories for generalization in complementary learning
  systems** — Nature Neuroscience (2023) s41593-023-01382-9.
- **Mesoscopic description of hippocampal replay and metastability
  in spiking neural networks with short-term plasticity** —
  PMC9822116 (2023).
- **Co-existence of synaptic plasticity and metastable dynamics in a
  spiking model of cortical circuits** — PMC10723399 (2024).
- **Neuronal competition shapes the encoding, consolidation, and
  retrieval of precise spatial memories in mice** — Curr Biology (2025).
- **Diffusion of Neuromodulators for Temporal Credit Assignment** —
  arXiv 2603.08949.

---

## Concrete ideas for Neuro-AI pair #4

These are ordered roughly by "how much they preserve the original
proposal" vs "how much they restructure it."

### Refinement #1 — rename the signal as **regret** and rederive it

The smoke result (m_max = 0.0082) is partly a *framing* problem. With
the static `c_i = w_i (1 − max_j w_j)`, the noise-floor regime makes
the signal degenerate because w_i values are all small. With the
trajectory reformulation `r_i = max_t w_i(t) − w_i(T)`, the signal is
non-zero whenever some atom *briefly led* during settling.

Renaming `r_i` to **hindsight regret** has two effects:
- The signal has a 4+ year RL provenance (Liu et al. 2021 ReMERN)
  with well-understood properties.
- It frees the analysis from needing to justify metastability — we
  can argue directly that "an atom that briefly led during settling
  but was displaced is exactly the atom we want to consolidate, because
  consolidation should reinforce near-misses that the system had
  evidence for."

**Action:** rewrite the design note to use `r_i = max_t w_i(t) − w_i(T)`
as primary, cite Liu et al. (2021), drop the metastability framing
entirely. Headline metric stays the same (Δ meta-stable-rate at W=3 is
about *substrate behavior*, not about what the signal is called).

### Refinement #2 — replace c_i with **dwell-fraction**

Variant of #1: `dwell_i = (1/T) Σ_t [argmax_j w_j(t) == i]`. This is
fully on-trajectory, doesn't depend on a single timestep, and is the
spiking-replay literature's exact convention. Likely more robust at the
noise floor because it integrates over T iterations.

**Action:** in the same smoke harness, log `dwell_i` alongside `r_i` and
see which has higher m_max on the same substrate. Cheap (one extra
tensor add per iteration). Use whichever is larger.

### Refinement #3 — collapse the new signal into the Phase 5 headline ΔE

The Phase 5 design-spec headline is ΔE between role-prior and content-
prior at consolidation time (phase-5-unified-design.md:256-281). If
ΔE itself is what's *available* and what we *care about*, then making
pair #4 use ΔE per-atom — `m_i = |E_role(i) − E_content(i)|` averaged
over recent retrieves — would:
- Collapse "what we modulate replay by" and "what we report as the
  headline" into one quantity, removing a degree of freedom.
- Be derivable directly from the existing instrumentation (no new
  hot-loop signal to add).
- Match the SuRe biological precedent exactly (surprise-driven replay).

**Anti-homunculus check:** ΔE is a local geometric quantity per atom.
Used multiplicatively in priority. PASS.

**Recommendation:** this is the strongest variant if Phase 5 graduates
on ΔE anyway. The case for keeping metastability/regret as a *separate*
signal is mostly that it would be informative *if* it carried
orthogonal information to ΔE; otherwise it's a second knob doing the
same job.

### Refinement #4 — substrate-side fix for the noise floor (input-driven energy)

If all atoms are tied at the substrate noise floor, no static signal
extracted from softmax weights will help. The fix from Betteti et al.
(Science Advances 2025) is to add an **input-driven term** to the
energy that depends on the query: `E(s, q) = E_Hopfield(s) − λ ⟨s, q⟩`
so the basins around atoms matching the query *deepen* during settling.

Applied here: at retrieve time, modulate the per-iteration energy with
the encoded query bundle so the substrate noise floor doesn't dominate.
This is independent of pair #4 but might be the *enabling* fix that
makes pair #4 testable in the first place.

**Recommendation:** flag this to the user as a possible Phase-4-side
substrate fix. If `c_i` is at the noise floor because *no signal* is
extractable from the current settling dynamics, we should consider
whether the dynamics themselves need the Betteti-style input drive
*before* spending more iterations on pair #4 instrumentation.

### Refinement #5 — ensemble disagreement as `m_i` (UPER analog)

Run two Hopfield retrieves with different β (or different temperature
schedules) and use the disagreement on each atom's weight as `m_i`.
This is the UPER recipe ported to associative memory: signal = epistemic
uncertainty estimated by ensemble disagreement, modulates priority.

Costs: ~2× retrieval cost, but settlement should be cheap because both
retrieves see the same query and stored atoms.

Benefits: directly imports the UPER theoretical justification
(information-gain functional), with anti-homunculus PASS.

**Risks:** the two retrieves at different β might just *both* be at the
noise floor and *both* agree.

### Refinement #6 — make the m_max gate pass

If the user wants to keep the existing trajectory-c_i formulation as-is
and just get the m_max above 0.05, the cheapest interventions are:

(a) **Raise β at retrieve time** to amplify margins. This is exactly
    the Hopfield phase-transition lever; the 2311.18434 paper has the
    relevant phase diagram.
(b) **Normalize the per-iteration c_i by the per-iteration max c_j**
    so the signal is relative-not-absolute. Trades off interpretability
    (no longer in units of probability) for guaranteed non-degeneracy.
(c) **Lower the κ used in the EMA** so the signal accumulates slowly
    but consistently across many retrieves. Combined with longer
    burn-in, m_max grows linearly with the number of retrieves it
    integrates over.
(d) **Drop the gate** — argue from the literature that even small `m_i`
    perturbations of priority give continuous-modulation benefits and
    don't need a magnitude threshold. The UPER / ReaPER papers don't
    pre-commit to a magnitude floor on their priority signal.

The user pre-committed to 0.05 for a reason; (d) is the easiest to
write down but the costliest in terms of pre-commitment integrity.
(a)-(c) are mechanical and worth trying in the same smoke.

### Refinement #7 — recall-gated consolidation framing

If pair #4 is truly about consolidation rate, the Tyulmankov et al.
(eLife 2024) framing is more direct: gate the *consolidation rate*
(not the replay sampling rate) by recall margin. In Neuro-AI terms:
when an atom is recalled with high margin (low metastability), give
its consolidation step a *larger* weight, not a smaller one. This is
the *inverse* sign of the current proposal.

Worth thinking about because the eLife paper's headline result is that
recall-gated consolidation specifically prevents spurious
consolidations of noisy patterns, which is one of the failure modes
our substrate has been chasing for several phases.

**Caveat:** this flips the sign. If pair #4 is "prioritize uncertain
atoms for replay," recall-gated consolidation is "prioritize confident
atoms for consolidation." They are not the same mechanism even though
they share a vocabulary. Worth making explicit which one is intended.

---

## Risks

### Risk 1 — metastability is *also* at the substrate noise floor

This is the dominant risk and the 1-seed smoke already half-confirmed
it. The trajectory reformulation moves the needle from "structurally
inert" to "0.0082" but still under the 0.05 gate. On a substrate where
all atom energies are within ε of each other, no per-atom softmax-
derived signal will be informative. The recommended mitigations are
(in order):
1. Try the input-driven energy modification (Refinement #4) first to
   see if the substrate dynamics themselves can be made non-degenerate.
2. Try dwell-fraction (Refinement #2) which is more robust to
   per-timestep noise.
3. Try ensemble disagreement (Refinement #5) which doesn't depend on
   the absolute magnitude of any one retrieve.

If all three fail, the substrate is genuinely flat at retrieve time
and pair #4 should be dropped in favor of pursuing a signal computed
*not* from softmax weights — e.g. from the gradient of E w.r.t. each
atom, which is non-zero whenever any atom is non-orthogonal to the
query, regardless of softmax temperature.

### Risk 2 — the signal moves but the headline doesn't

Even if we get m_max > 0.05, the Δ meta-stable-rate at W=3 might not
improve. The literature is consistent that priority modulation *speeds
up learning* (sample efficiency) more than it *changes the final
fixed point* (steady-state quality). If the Phase 4 substrate is at
its steady-state quality limit, no replay-prioritization scheme will
help. The fix in this case is on the substrate side (Phase 4
modifications), not the replay side.

**Mitigation:** include the *time-to-convergence* of meta-stable-rate
in the drill-downs, not just the steady-state value. If with-priority
gets to the same steady state but faster, that's a real but
qualitatively different win than what pair #4 currently promises.

### Risk 3 — over-fitting to one substrate config

A1' substrate with W=3 is a single configuration. The Yang et al.
(Science 2024) result is that prioritization regimes are *task-
dependent*: novelty matters more for some tasks, reward for others.
A signal that fails on A1' might succeed on a slightly different
substrate config (different N, different α, different β schedule).

**Mitigation:** if pair #4 survives smoke and shows promise on A1',
the next step should be a substrate-config sweep before declaring
graduation, not a multi-seed deepening on the single config.

### Risk 4 — the metastability framing is itself misleading

Pointed out in §"Trajectory-c_i precedent": metastability of softmax
weights is a poor proxy for the things the biology actually uses
(surprise, novelty, recall margin). Continuing to call this signal
"metastability" risks anchoring future analysis on the wrong question.

**Mitigation:** rename to "hindsight regret" before any further
experiments. Cheap.

### Risk 5 — interaction with Phase 4 codebook dynamics

The emergent codebook is itself non-stationary. Adding a replay
prioritization signal that depends on retrieve weights creates a
feedback loop: high-m_i atoms get replayed more → their geometry gets
sharpened → m_i for those atoms changes → priority changes. This loop
might converge or might oscillate. Anti-homunculus check: the
oscillation is itself a dynamical fact, not an arbitration, so the
check passes — but the *dynamics* of the loop need analysis before we
trust the steady-state metric.

**Mitigation:** explicit logging of (m_i, replay count, atom geometry)
over training time, plus a control where κ is varied across seeds to
trace the bifurcation diagram of the loop.

---

## Surprises

1. **The biological consensus signal is prediction error, not
   metastability.** The 2024-2025 LC-hippocampus literature is
   remarkably consistent: surprise (signed or unsigned) drives both
   replay priority and consolidation gain. Going in I expected
   "metastability" to have stronger biological backing than this. It
   doesn't. This pushes pair #4 toward Refinement #3 (use ΔE) or
   #6 (recall margin) as the signal.

2. **Mattar/Daw is still the dominant theoretical frame in 2025**, with
   the most recent extension being Németh et al.'s ERA paper (Oct 2025).
   The "gain × need" decomposition has held up across seven years of
   neural recordings and gets steadily refined rather than replaced.
   The biological prioritization framework is not in flux — it's
   converging.

3. **There is no clean precedent for `w_i · (1 − max_j w_j)`** as a
   downstream learning signal in the 2024-2026 literature I searched.
   Margin (top1 − top2) is well-known; entropy is well-known; the
   specific product form appears to be original to the Neuro-AI design
   note. This is a small surprise — it might be worth ablating against
   the simpler `1 − max_j w_j` (entropy-of-best-channel) or
   `w_top2/w_top1` (relative-runner-up) to see if the product form
   buys anything.

4. **Input-driven Hopfield dynamics (Betteti et al. 2025) is a much
   better fit for the noise-floor problem than I expected.** I went in
   looking for replay-prioritization papers and found a Hopfield
   substrate paper that addresses the exact failure mode that's
   blocking the replay-prioritization work. If the substrate is the
   real bottleneck, this paper is probably the most actionable single
   reference of the scan.

5. **Recall-gated plasticity (Tyulmankov 2024) is the structural
   inverse of pair #4** but has a more mature analytic treatment.
   Worth at least reading the paper before committing to which sign
   the modulation should be.

6. **SuRe (arXiv 2511.22367, late 2025) just did the surprise-driven
   replay approach for LLM continual learning.** Pair #4's
   "metastability-driven replay" lines up structurally with their
   "surprise-driven replay," but they used per-token loss directly as
   the signal, which is cleaner than our c_i. Concurrent-work
   precedent.

---

## Open questions

1. **Is the substrate-noise-floor problem fundamental to A1', or is it
   a tuning issue with β?** The 2311.18434 phase-transition paper
   suggests this is a tunable knob. Has anyone in the project run a
   β-sweep on A1' to see where the substrate has the largest dynamic
   range in `max_j w_j`? If not, this is the cheapest experiment to run
   *before* committing to pair #4.

2. **Is the m_max=0.05 gate the right threshold?** The threshold was
   pre-committed but no rationale is in the design note we read.
   Should it be (a) absolute m_max, (b) m_max / m_median, (c) m_max
   variance across seeds? Any of these has a more principled
   justification than absolute m_max.

3. **Would `r_i = max_t w_i(t) − w_i(T)` and `dwell_i = (1/T) Σ_t
   [argmax == i]` and `m_i = ΔE_role-vs-content` give similar
   rankings of atoms?** If they do, the choice is mostly cosmetic.
   If they don't, the disagreement itself is informative — it tells us
   which signal is picking up substrate vs. content vs. dynamics.

4. **Is the project actually replay-bottlenecked or substrate-
   bottlenecked?** If the headline metric (meta-stable-rate at W=3)
   isn't moving even when we hand-construct an "ideal" replay priority,
   then pair #4 can't help no matter how clever the signal is.
   Worth running an oracle control: priority = ground-truth Δ
   meta-stable rate per atom (i.e. cheat with hindsight). If oracle
   priority moves the headline, pair #4 has headroom; if it doesn't,
   pair #4 is the wrong pair.

5. **What is the relationship between meta-stable-rate at W=3 and the
   Phase 5 design-spec headline ΔE?** If they correlate strongly, then
   measuring one suffices. If they don't, then *which one is the
   headline?* The CLAUDE.md preamble rule says it's whatever
   phase-5-unified-design.md:256-281 says, which is ΔE. The pair #4
   proposal optimizes for meta-stable-rate (a Phase 4 metric). This is
   a documented case of the kind of drift that CLAUDE.md warns against
   — pair #4 might be solving the *previous* phase's headline rather
   than the current one.

6. **Is there a Saighi-line neuromorphic-implementation argument?**
   I didn't find recent Saighi follow-ups specifically, but the
   neuromorphic-memristor literature broadly (PMC12899887 review,
   2026) treats short-term plasticity as the natural substrate for
   rate-modulated consolidation. If pair #4 has a neuromorphic
   implementation story, the rate-modulation pattern (Refinement #3
   or #7) is the easiest to realize physically. This is not a
   load-bearing argument for the current decision but it's worth
   noting as a forward-compatibility property.

7. **Does the trajectory-c_i signal interact well with the
   diffusion-based replay schedules in the broader literature?**
   E.g. Hu et al. (arXiv 2411.10809) use diffusion processes for
   trajectory replay. Our `r_i` could in principle parameterize the
   diffusion's drift term, giving a continuous-time analog of
   priority. Out of scope for now but worth a future note.

---

## Bottom-line recommendation

If the goal is "decide whether to pursue pair #4," the literature scan
gives the following structured answer:

- **The pattern is right.** Using a per-item local signal to
  modulate replay priority continuously is exactly what the 2024-2026
  field is doing (UPER, ReaPER, SuRe), and it's anti-homunculus-clean
  in all of those instantiations.

- **The signal might be wrong.** "Metastability of softmax weights" has
  weak biological precedent (the biology uses prediction error / recall
  margin). Better candidates in order of how much they preserve the
  current design:
  1. **Hindsight regret** (rename of trajectory-c_i with RL provenance).
  2. **Dwell-fraction** (closer to spiking-network metastability).
  3. **ΔE role-vs-content** (collapses with Phase 5 headline; most
     parsimonious).
  4. **Recall margin** (inverse sign; Tyulmankov eLife 2024 precedent).

- **The substrate might be the real bottleneck.** The 1-seed smoke
  m_max = 0.0082 is consistent with "all atoms at noise floor," which
  the Betteti et al. (Science Advances 2025) input-driven-energy fix
  addresses directly. If the substrate is degenerate at retrieve time,
  no choice of `m_i` will save pair #4.

- **The cheapest next experiment is not "more seeds on pair #4."** It's
  one of:
  - β-sweep on A1' to characterize the noise-floor regime,
  - Oracle-priority control to test whether replay-prioritization can
    move the headline at all on this substrate,
  - Log `r_i`, `dwell_i`, `ΔE_i` all in one smoke pass and compare
    their ranges and rankings.

Either of these is < 1 session of work and would let pair #4 be
decided with calibrated evidence rather than the current single-seed
smoke + literature-prior judgment call.
