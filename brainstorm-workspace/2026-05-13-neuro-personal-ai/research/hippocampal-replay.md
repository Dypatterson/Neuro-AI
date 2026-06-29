---
date: 2026-05-13
angle: Hippocampal replay and systems consolidation — biological mechanisms for the Benna-Fusi design
---

# Research Brief: Hippocampal Replay and Systems Consolidation

## Angle

This brief investigates what biological and computational neuroscience knows about
hippocampal sharp-wave ripple (SWR) driven replay and systems consolidation, specifically
to inform design choices for:

1. The **replay gating function** — currently `engagement × (1 - resolution)` — and
   whether the biology suggests refinements or alternative formulations.
2. The **Benna-Fusi consolidation variable coupling** — specifically, what drives the
   u_k ↔ u_{k+1} transfer and whether there are biologically motivated alternatives.
3. The **timing and structure of replay** — when replay occurs, how it sequences,
   whether forward/reverse matter, and how the landscape (neocortex) changes as a result.
4. Whether recent work challenges or refines the Benna-Fusi framework's assumptions.

The project already has a detailed Phase 4 design (notes/emergent-codebook/phase-4-unified-design.md)
that defers sleep/wake cycles and cross-scale replay interactions. This brief
specifically targets those deferred questions plus latent risks in the existing design.


## Key Findings

### 1. SWRs do not sample randomly — content is selected by a two-stage tagging mechanism

**Source:** Joo & Frank (2023), "Selection of experience for memory by hippocampal sharp wave
ripples," *Science* 382. PMC open-access version:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10659301/

The most important finding for this project: awake SWRs function as a **tagging event**,
not a consolidation event. When the brain transitions from theta (exploration) to SWR
(rest/reward consummation), ripples encode trial-specific spike content. The distribution
of decoded trial identities during post-experience **sleep** SWRs is directly predicted
by which trials were tagged during **awake** SWRs. The mechanism is:

- Awake SWR: fires during brief behavioral pauses or reward consummation; its spike
  content tags specific experience episodes (their cell-assembly patterns).
- Sleep SWR: preferentially replays whatever awake SWRs tagged most frequently.

This means the "what to consolidate" decision is made **twice**: once by the awake SWR
(which fires at pauses and reward) and once by accumulated tag frequency (which patterns
were tagged most gets replayed most during sleep).

**Design implication:** The project's current single-pass gate (`engagement × (1 -
resolution)`) corresponds roughly to the awake-SWR stage. But the biology has a second
stage: **frequency of tagging predicts sleep replay priority**. A pattern that passes the
gate repeatedly is more consolidation-worthy than one that passes once. This maps
naturally onto the replay-store `age` counter and the Benna-Fusi u_1 accumulation — both
already in Phase 4. Good news: the architecture is structurally aligned. But the
*operationalization* of "tagged frequently" in the biology is tag count, whereas the
current design uses age (time since first entry). These are different: a trace that sits
unresolved for a long time is old, but a trace that keeps re-generating a gate signal
across *different* cues is frequently tagged. This distinction matters.

---

### 2. SWR amplitude (size) is a graded consolidation signal, not a binary gate

**Source:** Neuron (2025, accepted), "Large sharp-wave ripples promote hippocampo-cortical
memory reactivation and consolidation during sleep."
https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1

Closed-loop optogenetic boosting of SWRs during post-task sleep enhanced ensemble
reactivation and improved memory performance. Large SWRs are associated with stronger
hippocampo-cortical coordination during the event. This is a **graded** signal: larger
ripples → stronger reactivation → more consolidation.

**Design implication:** The project's `gate_signal = engagement × (1 - resolution)` is
already a scalar, not a binary. The scalar value should modulate consolidation strength
(how hard u_1 gets hit) proportionally — not just determine in/out. The existing Phase 4
spec writes `u_state.initialize(new_idx, u_1=novelty_strength)` with a fixed
`novelty_strength`. This is where the gate signal magnitude should feed in: higher gate
signal → larger u_1 initialization. This is a specific, well-motivated refinement.

---

### 3. Replay content is biased by reward prediction error — not reward per se

**Source:** Mattar et al. (2018), "Prioritized memory access explains planning and
hippocampal replay," *Nature Neuroscience*.
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620/

**Source:** Post-learning replay biased by reward-prediction signals, *Nature
Communications* 2025. https://www.nature.com/articles/s41467-025-65354-2

Priority of replay is the product of two terms:
- **Gain**: how much a memory update would change behavior (= policy improvement
  potential). Asymmetric: learning that a bad action is actually worse-than-expected
  carries low gain; learning it's better-than-expected carries high gain.
- **Need**: how often the agent expects to revisit the state (= Successor
  Representation / predictive map).

Empirically (2025 paper), replay is biased by **reward prediction error** (RPE), not
reward magnitude. Post-task rest preferentially reactivated RPE signals rather than raw
reward signals. Dopamine inactivation in novel environments caused aberrant, spatially
non-selective SWR increases (eLife 2024: https://elifesciences.org/articles/99678), but
familiar environments maintained RPE-tracking replay through VTA-independent pathways.

**Design implication:** The current `engagement × (1 - resolution)` gate captures a
geometric analog of "gain" — unresolved trajectories represent places where the system's
model is uncertain (high landscape force, no commitment). This is a legitimate
information-theoretic proxy for prediction error. But the "need" term (how often will
this state be revisited?) is **absent** in the current architecture. Biology weights
replay priority jointly by gain × need. A replay-worthy trace that will never be
encountered again is lower priority than one from a frequently re-entered attractor
neighborhood.

This opens a concrete addition: track how many times a query arrives in the neighborhood
of a given replay-store entry's original query. Replay-store entries in frequently
visited regions get a multiplier. Still local geometry — no supervisor required.

---

### 4. Forward vs. reverse replay have distinct biological roles

**Source:** Buzsaki (2015, NRN), "The hippocampal sharp wave–ripple in memory retrieval
for immediate use and consolidation."
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/

**Source:** PNAS Nexus (2024), "Hippocampal replay sequence governed by spontaneous
brain-wide dynamics." https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348

- **Forward replay**: occurs during awake rest before/after experience; supports
  decision-making and planning; correlates with brain-wide "microcascades" of one
  temporal order.
- **Reverse replay**: occurs during reward consummation; supports credit assignment
  (linking outcomes backward to prior actions); modulated by reward magnitude
  independently of forward replay.

The 2024 PNAS Nexus paper found that replay sequences are **embedded in pre-existing
brain-wide spiking cascades** — approximately 70% of forebrain neurons participate in
coarse-scale cascades (5-15 sec) and subsecond microcascades. Forward and reverse replay
events align with two types of microcascades of opposite temporal order. No external
supervisor — the cascades are intrinsic, self-organized dynamics.

**Design implication:** The project's trajectory traces are necessarily forward (they
follow the settling sequence as written). But the biological substrate distinguishes
forward (planning) from reverse (credit assignment). For a personal-AI context, reverse
replay would correspond to re-settling a trajectory backwards — starting from the
resolved endpoint and tracing back to the original cue. This could strengthen which
earlier steps in the trajectory receive u_k reinforcement. Currently, only the final
resolved state gets a u_1 hit. Reverse-replay semantics would allow intermediate
attractors in the settling path to also receive partial credit.

---

### 5. The three-stage autonomous consolidation model — time constants and mechanisms

**Source:** Wörgötter et al. (2014), "Memory consolidation from seconds to weeks:
a three-stage neural network model with autonomous reinstatement dynamics,"
*Frontiers in Computational Neuroscience*.
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4077014/

The most computationally explicit model of multi-timescale consolidation found in
this search. Three stages:

| Stage | Analog | Plasticity time constant |
|-------|--------|--------------------------|
| PFC | Working memory (50 units) | ~3 minutes |
| HIP | Intermediate storage (250 units) | ~3 hours |
| CTX | Long-term memory (500 units) | ~6 days |

Autonomous reactivation occurs at ~6.5 Hz in hippocampus and ~6 Hz in cortex without
external noise injection. The mechanism: cellular adaptation and synaptic depression
create quasi-stable attractor states that naturally transition — depression at one
pattern's local synapses shifts the network toward the next stored pattern. This is
a **local dynamics** explanation for spontaneous replay.

The time constants span 6 orders of magnitude (minutes to days). The Benna-Fusi model
achieves similar coverage through exponential scaling of chain variables.

**Design implication:** The synaptic depression → natural transition mechanism suggests
a specific local mechanism for why the replay store samples across trajectories rather
than getting stuck replaying the same one. The project currently uses gate-signal-weighted
sampling from the store plus an age counter. A synaptic-depression analog would be: after
a trace is replayed, its gate signal is transiently suppressed (mimicking depression),
causing the next sample to draw from a different trace. This is closer to the biological
mechanism than pure age-counting and prevents replay monopolization by the highest-gate
entry.

---

### 6. Recall-gated plasticity — a systems-level alternative to pure Benna-Fusi

**Source:** Aitchison et al. (2024), "Selective consolidation of learning and memory via
recall-gated plasticity," *eLife*.
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11257680/

**Source:** bioRxiv preprint version with theory:
https://www.biorxiv.org/content/10.1101/2022.12.08.519638v1.full

This is a systems-level model where long-term plasticity is **gated by whether the short-
term system can already recall the pattern**. The core rule: LTM consolidation occurs
only when a pattern is reinforced while it is currently recalled (retrieved) by STM.
Recall strength = overlap of the pattern with current synaptic state in STM.

This is explicitly different from Benna-Fusi: B-F is a synaptic cascade that passively
flows energy from fast to slow variables when stimulated. Recall-gated consolidation is
selective: only patterns that the STM can reliably recall get written to LTM. This acts
as a quality filter — unreliable, noisy patterns that only weakly activate STM don't
consolidate.

Key advantages demonstrated in the paper:
- Explains spaced learning effects (spacing boosts recall probability at time of
  re-exposure → more consolidation per exposure)
- Explains task-dependent consolidation rates
- Filters noise without global supervisor (the recall check is local population activity)

**Design implication:** The project currently uses a resolution threshold at replay time
to determine whether a re-settled trace becomes a candidate pattern. This is structurally
similar to recall-gated consolidation: the "recall check" is whether the re-settle
produces a clean resolution. The eLife paper formalizes exactly why this filter improves
long-term memory fidelity. But there is a subtle difference: the project's gate fires on
*high* engagement × *low* resolution (putting unresolved things into the store), then
requires *high* resolution at replay time to consolidate. The recall-gated paper would
suggest an additional check: the pattern being replayed should already have *some*
existing representation strength (not be being written for the first time) to justify
consolidation. This maps to: only consolidate (increase u_k chain) if u_1 is already
non-trivially positive. First-time patterns get u_1 initialized but should clear a
"minimum visits" bar before propagating to u_2, u_3, etc.

---

### 7. Synaptic tagging and capture — the local mechanism behind the replay gate

**Source:** Royalsociety 2024 review, "Synapses tagged, memories kept: synaptic tagging
and capture hypothesis in brain health and disease."
https://royalsocietypublishing.org/rstb/article/379/1906/20230237/42846/Synapses-tagged-memories-kept-synaptic-tagging-and

**Source:** PMC: Neuromodulator-dependent synaptic tagging and capture retroactively
controls neural coding in spiking neural networks.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9588040/

Synaptic tagging and capture (STC) is the biological local mechanism that bridges LTP
induction (fast, minutes) and long-term memory (hours-days):
1. Pre+postsynaptic co-activation sets a **synaptic tag** that decays over ~1 hour.
2. A neuromodulatory signal (dopamine, norepinephrine) triggers **protein synthesis** (the
   "capture product") — but only if a tag is present.
3. Tags that don't encounter capture products within their lifetime → no long-term change.

This is a three-factor rule: Hebbian co-activation × neuromodulator × time-window.
The tag is the local eligibility trace; the neuromodulator is the "third factor."

**Design implication:** The Benna-Fusi cascade is a continuous version of this. The
project's u_1 initialization at pattern creation is the "tag," and repeated replay is
the "capture." The 1-hour tag decay maps to the u_1 decay timescale. The key biological
parameter missing from the current design: the tag needs a neuromodulatory "capture
signal" to persist. In the architecture, this could map to the gate signal itself: a
high-gate trace that **keeps re-generating** gate signal on subsequent queries acts as
its own capture trigger (the system keeps "noticing" it). A trace that gets tagged once
but never generates gate signal again → tag decays → no capture → no consolidation.
This is already partially implemented via the age counter, but formalizing it as a
"tag strength" that decays between replay cycles (and is renewed by each replay attempt)
would bring the architecture closer to STC biology.

---

### 8. Sleep oscillation coordination — triple phase-locking

**Source:** Systems memory consolidation during sleep review, 2026.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/

During NREM sleep, three oscillations exhibit "triple phase-locking":
- **Slow oscillations (0.1-4 Hz)**: cortical up/down states creating synchronization windows
- **Sleep spindles (10-15 Hz)**: thalamic bursts nested in slow-oscillation up-states;
  artificially inducing them during up-states enhances memory
- **SWRs (150-250 Hz)**: hippocampal high-frequency events nested within spindles

The SWR fires during the trough of a spindle, which fires during the up-state of a slow
oscillation. This hierarchical nesting is what enables the hippocampal replay to arrive
at the cortex in windows of maximal cortical excitability.

Norepinephrine oscillates at ~0.02 Hz during NREM (extremely slow) and its timing is
critical for memory stability. Dopamine surges at NREM→REM transitions, signaling a
shift from stabilization to generalization/abstraction.

**Design implication:** The project defers sleep/wake cycles. But this research suggests
that the "replay cycle" cadence (every K cues) should not be uniform — there is a natural
analogy to the slow oscillation as the "frame" that batches replay events. A batch
of replay cycles (hippocampal-analog settling events) followed by a consolidation step
(u-variable chain propagation) followed by a quiescence period mimics the SWR→spindle→
slow-oscillation hierarchy. This could inform how the replay batch size and the Benna-
Fusi step frequency should be coupled.

---

### 9. SFMA model — explicit replay priority formula from the field

**Source:** Biderman et al. (2023), "A model of hippocampal replay driven by experience
and environmental structure facilitates spatial learning," *eLife*.
https://elifesciences.org/articles/82301

The SFMA model gives the most explicit computational formula for replay priority found
in this search:

```
R(e | e_t) = C(e) × D(e | e_t) × [1 − I(e)]
```

Where:
- `C(e)`: experience strength (frequency × reward history)
- `D(e | e_t)`: experience similarity to current state (using Default Representation /
  environmental structure)
- `I(e)`: inhibition of return (suppresses recently replayed experiences)

The inhibition of return component prevents monopolization of replay by the highest-
priority entry — exactly the concern raised in finding #5 above about synaptic
depression.

**Design implication:** The SFMA formula maps almost directly onto the project's design
with one gap: the `D(e | e_t)` term — similarity of a replay candidate to the *current*
query — is absent. The current replay store samples by gate signal × age, not by
proximity to the current cue. Adding a similarity modulation (replay entries whose
stored query is cosine-near to the current active query get a priority boost) would make
the replay system more responsive to context, matching the biological "recency-sensitive"
replay pattern. This is geometrically local: it's just cosine similarity between the
current query and the stored trace's original query.

---

### 10. Intelligent plasticity — limitations of pure cascade Benna-Fusi

**Source:** Review "Theories of synaptic memory consolidation and intelligent plasticity
for continual learning," arXiv 2405.16922.
https://arxiv.org/html/2405.16922v1

The Benna-Fusi model's cascade is **passive**: energy flows down the chain according to
fixed dynamics regardless of whether the stored pattern is behaviorally important. The
critique: Lahiri and Ganguli showed existing cascade models are not theoretically optimal.

"Intelligent plasticity" approaches (Synaptic Intelligence, EWC) compute **online
importance scores** per parameter, then scale plasticity inversely with importance:
important synapses are protected from overwriting.

The cascade model's specific weakness: a limitation due to **saturation of fast
synaptic variables** — the timescale of internal variables determines optimal spacing
effects, and intervening stimuli can block the effect by preventing saturation. This is
a known failure mode when new experiences arrive at irregular intervals.

**Design implication:** The project uses Benna-Fusi as the consolidation substrate. The
saturation/irregular-arrival critique is directly relevant since the project's replay
store will emit candidates at irregular intervals (whenever re-settling resolves). A
hybrid could be used: Benna-Fusi dynamics for the cascade structure, but with an
importance-weighted u_k coupling coefficient. Patterns that have been retrieved many
times (high `retrieval_count`) get a higher coupling coefficient (faster transfer to
slow variables) rather than fixed α. This is architecturally local: u_1's coupling to
u_2 is scaled by the pattern's retrieval frequency, computable without supervision.


## Promising Leads

1. **The "Dynamical modulation of hippocampal replay through firing rate adaptation"
   (Nature Communications 2025)** — found in search results but paywalled. This paper
   specifically addresses local mechanisms (firing rate adaptation, short-term
   depression, acetylcholine) that control replay dynamics. Directly relevant to the
   local-dynamics requirement. Source:
   https://www.nature.com/articles/s41467-025-68042-3

2. **Replay without sharp wave ripples (Nature Communications 2025)** — challenges the
   assumption that SWRs are necessary for replay. If replay can occur without SWRs,
   then the SWR is not the gate but a correlate. Relevant to whether the project's gate
   needs to model a "sharp-wave" analog or just the replay dynamics.
   https://www.nature.com/articles/s41467-025-65181-5

3. **The recurrent network model of planning explaining hippocampal replay (Nature
   Neuroscience 2024)** — a mechanistic model of how internal planning drives replay.
   If planning is the driver, then the project's meta-loop (the landscape "thinking
   about its own contents") is the right biological analog.
   https://www.nature.com/articles/s41593-024-01675-7

4. **Human hippocampal ripples prioritize model-based learning (bioRxiv 2025)** —
   human fMRI/iEEG evidence that ripple priority signals are value-weighted. Longer
   ripples carry stronger priority signals. Directly maps to gate signal magnitude.
   https://www.biorxiv.org/content/10.1101/2025.07.31.667862v1.full

5. **The MÖBIUS model (Communications Biology 2026)** — a probabilistic model of when
   REM sleep fails to contain internally generated content, causing it to be mis-encoded
   as episodic memory. If the project's replay buffer generates content that gets
   consolidated without the "novel vs. experienced" distinction, there's an architectural
   analog of this failure mode. Worth reading to understand what a "containment failure"
   looks like.
   https://www.nature.com/articles/s42003-026-09781-x

6. **Input-driven plasticity in Hopfield networks (Science Advances 2025)** — a new
   mechanism where external inputs reshape the Hopfield energy landscape through
   saliency weights. This could inform how the replay-store's re-settling step modifies
   the landscape (vs. just reading from it). The project currently uses re-settle as
   read-only; IDP suggests write-back during replay could be architecturally motivated.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12017325/


## Concrete Ideas

### Idea 1: Two-stage gate with frequency accumulator (maps to awake→sleep tagging)

The current gate is single-pass: `engagement × (1 - resolution)` → in/out store.
Biology uses two stages: awake SWR tags → sleep SWR replays most-tagged.

**Concretely:** Add a `tag_count` field to each replay store entry. Each time a new
query arrives and its gate signal is high *and* the new query's trajectory overlaps
significantly with an already-stored trace (cosine of queries > overlap_threshold),
increment `tag_count` rather than creating a duplicate entry. Sample replay probability
proportional to `gate_signal × tag_count`. This captures "frequently tagged = higher
sleep replay priority."

Anti-homunculus check: `tag_count` increments on a geometric overlap condition. No
supervisor computes it. Pass.

---

### Idea 2: Gate signal magnitude → u_1 initialization (graded SWR amplitude)

Currently `u_state.initialize(new_idx, u_1=novelty_strength)` uses a fixed value.

**Concretely:** `u_1_init = base_novelty_strength × gate_signal_at_resolution`. A
trajectory that resolves after accumulating high gate signal initializes with a stronger
u_1 than one that barely cleared the resolve_threshold. Biologically motivated by the
"large SWRs → stronger consolidation" finding. Locally computable. No supervisor.

---

### Idea 3: Inhibition of return in replay sampling (synaptic depression analog)

Currently sampling is weighted by gate signal × age. The highest-gate trace is
preferentially sampled, which can monopolize replay cycles.

**Concretely:** After each replay attempt on a trace, apply a transient suppression
multiplier (e.g., 0.5 decay per replay, recovering toward 1.0 over time with recovery
rate r). Next sampling weights become `gate_signal × suppression_multiplier × age`.
This is the SFMA inhibition-of-return term, biologically grounded in synaptic
depression. Ensures the replay store explores its full inventory rather than getting
stuck.

---

### Idea 4: Context-sensitive replay priority (SFMA's D(e|e_t) term)

Add similarity weighting to replay store sampling: `priority = gate_signal × cos_sim(current_query, stored_query) × suppression`.

When the active query is in a neighborhood near a stored trace, that trace's replay
priority increases. This is the biological "recency and context" effect. Geometrically
local: just cosine similarity between current FHRR vector and stored query vector.

---

### Idea 5: Retrieval-frequency weighted coupling coefficient (intelligent plasticity hybrid)

In Benna-Fusi, the coupling α between u_k and u_{k+1} is fixed. Replace with:

```
α_eff(pattern) = α_base × (1 + λ × normalized_retrieval_count(pattern))
```

Patterns retrieved frequently get faster transfer through the consolidation chain.
Patterns never retrieved stay in fast variables until they decay. This is the
"intelligent plasticity" correction to the pure cascade model. Locally computable:
each pattern has its own retrieval counter.

---

### Idea 6: Partial credit for intermediate attractors (reverse replay semantics)

The current design gives u_1 credit only to the resolved final pattern. Biology uses
reverse replay for credit assignment — the outcome propagates backward to prior states.

**Concretely:** When a trajectory resolves, trace back through `trace.snapshots` in
reverse order. For each snapshot's `top_k_indices`, apply a decaying credit signal:
`u_1_credit[step] = base_credit × discount^(n_steps - step)`. Attractors that appeared
near the end of settling get more credit; those that appeared early get less. This turns
consolidation into a temporal-difference update along the settling path.

Anti-homunculus check: credit is a mathematical function of the trajectory itself.
No supervisor chooses which attractors matter. The decay rate is an architectural
constant. Pass.

---

### Idea 7: Tag-decay formalization (STC analog)

Each replay store entry has a "tag strength" that decays between replay cycles at rate
`τ_decay`. Each time the entry accumulates tag signal (from Idea 1 — new overlapping
query) or is successfully replayed (even if unresolved), tag strength is renewed.
Entries whose tag strength → 0 are pruned before their age counter expires. This is the
synaptic tag decay from STC biology (~1 hour biological timescale, mapped to replay
cycles). Prevents the store from holding traces indefinitely when the relevant context
is no longer being encountered.


## Surprises

### Surprise 1: Replay can happen without SWRs

A 2025 Nature Communications paper found that hippocampal replay in a spatial memory
task occurred without sharp-wave ripples. If replay is not SWR-dependent, then the SWR
is not the gate — it may be a correlate or amplifier of a more fundamental dynamics.
This is an open challenge to the field's standard model and suggests the "true" replay
trigger may be something more continuous (like attractor competition / settling
dynamics), which is actually more aligned with the Hopfield architecture than the
SWR-centric story.

### Surprise 2: Cortex may drive hippocampal replay, not the reverse

The Nature Reviews Neuroscience (2018/2019) finding that cortical activity can precede
hippocampal SWRs by ~200ms in sensory areas challenges the standard story where the
hippocampus initiates replay. If the cortex cues the hippocampus, then systems
consolidation may be bidirectionally coordinated rather than hippocampus-pushing-to-
cortex. For the project, this suggests that the landscape (neocortex analog) may need
to send "request signals" to the replay store rather than the replay store autonomously
pushing candidates. However, this can still be anti-homunculus: the cortical signal is
the landscape's own current settling difficulty, which is geometrically expressible.

### Surprise 3: REM sleep has a distinct generalization/abstraction function

The systems consolidation review separates NREM (precise reactivation → stabilization)
from REM (theta-driven → generalization, abstraction, emotional tagging). The project's
replay mechanism is modeled on NREM-style precise reactivation (re-settling specific
trajectories). The REM analog — which would generate *novel* variants of stored patterns
rather than re-settling them exactly — is architecturally missing. This may be the
mechanism behind "creative insight" in biological systems. For a personal-AI system, this
maps to generative replay (hallucinating novel trajectories near stored attractors) vs.
reconstructive replay (re-settling actual stored queries). The distinction may matter for
discovering genuinely new relationships vs. confirming existing ones.

### Surprise 4: Dopamine's role is novelty-dependent, not universal

The eLife 2024 dopamine paper found that VTA dopamine inactivation only broke replay
selectivity in **novel** environments. In familiar environments, replay was VTA-
independent (maintained through other pathways, likely LC dopamine). This means the
"neuromodulatory capture signal" in STC biology is context-dependent: novelty requires
dopaminergic gating; familiarity does not. For the project, this suggests the gate
signal function may need to be non-stationary — a novelty-sensitive boost for new
patterns entering the landscape for the first time, relaxing to structural (geometric)
criteria once the landscape is established. The Benna-Fusi model doesn't naturally
distinguish these two regimes; this is a gap worth tracking.

### Surprise 5: Brain-wide spiking cascades govern replay order autonomously

The PNAS Nexus 2024 paper found that ~70% of forebrain neurons participate in
spontaneous self-organized spiking cascades during rest/sleep, and these cascades
determine the temporal order of replay (forward vs. reverse). The hippocampus does not
independently decide replay sequence — it is embedded in brain-wide dynamics. For the
project, this suggests the replay store's sampling order is not a local hippocampal
decision but is governed by something more like the global "landscape tension" across
all stored traces simultaneously. The current design samples from the store one trace at
a time; the biology suggests the whole replay sequence should be governed by a coherent
dynamical trajectory through the full store, not independent draws. This is a more
ambitious redesign and probably belongs in a later phase, but it's worth noting.
