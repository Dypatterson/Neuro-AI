# Research note 03 — Sleep-state replay dynamics and learning-from-remembering loops

Date: 2026-05-24
Angle: Replay primitives that *reshape* attractor topology (not merely
rehearse weights). Looking for substrate-level mechanisms that
(a) carve role-like / directional / relational basins, (b) remain local
dynamics (no controller), (c) move a measurable Phase-5 headline in
< 2 weeks of work.

This note is a synthesis pass: each primitive is screened against the
project's anti-homunculus rule and translated into a concrete
substrate change tied to existing code paths
(`torch_hopfield.retrieve`, the replay/consolidation channel, the
emergent codebook).

---

## Why this angle is load-bearing right now

Phase 5's open commitment in `STATUS.md` is frequency-weighted
Benna-Fusi α — i.e. "the more a pattern is recalled, the slower its
fast pool decays, so compression becomes abstraction." That is a
**rate-weighted rehearsal** mechanism. It does not change *which*
patterns get rehearsed, only *how fast* each one's fast pool drains.

Biological replay is doing something stronger: the burst structure,
direction, phase, and selection of replayed events all encode
*topological* information that the current substrate is silently
discarding. The hypothesis below is that the missing Phase-5 role
geometry can be carved by replay topology even without ever
introducing an explicit role variable — provided we replace
uniform-rate replay with structured replay that respects burst
compression, reverse ordering, schema-driven coupling, and
cue-biased reactivation.

---

## Key findings (URLs)

### Sharp-wave-ripple computational models — burst-structured compressed replay
- *Large SWRs promote hippocampo-cortical reactivation, 2025* —
  identifies a subset of LARGE SWRs that selectively increase after
  learning and drive hippocampal → cortical transfer. Implication:
  not all replay events are equal; amplitude/burst size matters.
  https://www.biorxiv.org/content/10.1101/2025.06.27.662061v1.full and
  Neuron version https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1
- *Selection of experience for memory by SWRs, Science 2024* —
  SWRs select WHICH waking experiences get replayed; selection is a
  property of the network state, not of an external controller.
  https://www.science.org/doi/10.1126/science.adk8261
- *Drift-diffusion dynamics of hippocampal replay, bioRxiv 2025* —
  state-space + drift-diffusion model of replay; replay events have
  internal momentum/inertia, not just sampling.
  https://www.biorxiv.org/content/10.1101/2025.10.14.682470v2
- *SWRs and replay emerge from structured synaptic interactions in
  CA3, eLife (Ecker et al.)* — the chain structure of recurrent
  excitation is sufficient AND necessary for SWR generation. SWRs
  are an *epiphenomenon of chain structure*, not a separate process.
  https://elifesciences.org/articles/71850
- *Dynamical modulation of hippocampal replay through firing-rate
  adaptation, Nature Communications 2025* — firing-rate adaptation
  (a strictly local cellular dynamic) is sufficient to produce the
  spectrum of forward/reverse/diffusive replay modes from one
  continuous attractor.
  https://www.nature.com/articles/s41467-025-68042-3

### Theta-gamma cross-frequency coupling — slot-based binding
- *Nonlinearity as universal CFC mechanism, Frontiers 2025* —
  derives gamma amplitude/frequency modulation by theta from
  nonlinear coupling; provides an analytical mapping that could be
  ported to a dynamical-systems substrate.
  https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2025.1553000/full
- *Feedforward/feedback inhibition modulating theta-gamma CFC, PLOS
  Comp Bio 2025* — ING/PING dichotomy: feedforward inhibition gives
  fast→slow direction, feedback inhibition gives slow→fast.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013363
- *Theta-gamma + communication-through-coherence reconciled, PLOS
  Comp Bio* — gamma cycles within a theta cycle act as discrete
  binding slots; ~7±2 items per theta cycle, the original Lisman
  result.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005162

### Reverse replay and direction-of-replay
- *Recurrent network model for goal-directed sequences through reverse
  replay (Haga & Fukai 2018, still load-bearing)* — symmetric STDP +
  short-term depression / afterdepolarization biases plasticity
  *opposite* to propagation direction. This is the only fully-local
  mechanism for reverse replay I found.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6059768/
- *Time course and organization of hippocampal replay, Science 2024
  (Sainsbury Wellcome)* — symmetry-breaking attractor model
  recapitulates forward + reverse organization; MEC excitatory input
  biases the direction.
  https://www.science.org/doi/10.1126/science.ads4760
  Sainsbury press: https://www.sainsburywellcome.org/web/qa/replaying-past-predicting-future-new-model-hippocampus

### Targeted Memory Reactivation (TMR)
- *Update on TMR during sleep, npj Science of Learning 2024* —
  reviews how a cue presented during SWS preferentially reactivates
  the associated trace via the spontaneous SO/spindle/ripple cascade.
  https://www.nature.com/articles/s41539-024-00244-8

### Mattar/Daw prioritized replay — and its successors
- *Mattar & Daw 2018 Nature Neuroscience* — utility = gain × need.
  This is the controller-smell paper. https://www.nature.com/articles/s41593-018-0232-z
- *Exploring Replay, Nature Communications 2025 (Antonov & Dayan)* —
  extends Mattar/Daw to exploration; importantly, derives the
  ordering from first principles rather than imposing it.
  https://www.nature.com/articles/s41467-025-56731-y
- *Between planning and map-building: prioritizing replay when
  future goals are uncertain, PMC 2025* — shows gain/need can be
  approximated by local quantities under uncertainty.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12633763/

### Schemas (Tse–Morris–Bethus)
- Foundational: Tse et al. 2007 Science
  https://www.science.org/doi/abs/10.1126/science.1135935
- Schemas + myelination (Benna-Fusi adjacent): PMC6902718.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6902718/
- Neural network model of schemas: bioRxiv 434696.
  https://www.biorxiv.org/content/10.1101/434696v1.full

### Engram allocation and competition
- *Writing the engram: epigenetic mechanisms of memory allocation,
  J. Neurochem 2025* — review of allocation as competition over
  intrinsic excitability + CREB.
  https://onlinelibrary.wiley.com/doi/10.1111/jnc.70328
- *Intrinsic neural excitability biases allocation and overlap of
  memory engrams, J. Neurosci 2024 (Delamare, Feitosa Tomé,
  Clopath)* — computational; co-temporal events overlap engrams
  ⇒ supports memory linking.
  https://www.jneurosci.org/content/44/21/e0846232024
- *The cost of remembering: engram competition as a flexible
  mechanism of forgetting, Trends Neurosci 2025* — directly
  connects "act of remembering" to "prunes competitors."
  https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(25)00153-5

### Reactivation-induced forgetting / adaptive forgetting
- *Retrieval induces adaptive forgetting via cortical pattern
  suppression, Nat Neurosci (Wimber et al.)* — direct neuroimaging
  evidence that remembering selectively suppresses competing
  cortical patterns.
  https://www.nature.com/articles/nn.3973

### Sleep-modulated disinhibition, two-factor consolidation
- *Sleep-modulated disinhibition enables replay for memory
  consolidation, accelerated by ripples, bioRxiv 2025* —
  disinhibition (not active driving) is sufficient to release replay.
  This is a beautifully anti-homunculus framing.
  https://www.biorxiv.org/content/10.64898/2025.12.09.693276v2.full
- *Hippocampal indexing alters the stability landscape of synaptic
  weight space, bioRxiv 2025* — frames replay as a stability-landscape
  reshaping operation, not weight-refresh. Directly supports my
  "carve the topology" thesis.
  https://www.biorxiv.org/content/10.1101/2025.11.10.687549.full.pdf
- *Two-factor synaptic consolidation reconciles robust memory with
  pruning and homeostatic scaling, bioRxiv 2024* — STRONG
  candidate complement to Benna-Fusi: synapses survive iff
  reactivated *and* scale-down survives.
  https://www.biorxiv.org/content/10.1101/2024.07.23.604787.full.pdf
- *SO-spindle coupling Bayesian meta-analysis, eLife 2024* —
  precision of SO-spindle phase coupling (not amplitude) predicts
  consolidation. Argues that the substrate needs a phase-coupling
  primitive at all.
  https://elifesciences.org/reviewed-preprints/101992

### Langevin / stochastic consolidation
- *Stochastic attention via Langevin dynamics on Modern Hopfield
  energy, arXiv 2026* — explicitly converts Modern Hopfield
  retrieve into a Langevin sampler.
  https://arxiv.org/pdf/2603.06875
- *Long-term memory stabilized by noise-induced rehearsal, PMC* —
  unstructured neural noise carries the imprint of all stored
  patterns in its temporal correlations.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3704454/
- *Stochastic consolidation of lifelong memory, bioRxiv* —
  random noise + Hebbian rule sufficient for autonomous
  consolidation.
  https://www.biorxiv.org/content/10.1101/2021.08.24.457446.full.pdf

### Reconsolidation (act-of-remembering changes the memory)
- *Windows of change: temporal and molecular dynamics of memory
  reconsolidation and persistence, Neurosci Biobehav Rev 2025* —
  retrieval opens a labile window in which prediction error gates
  re-encoding. The labile-window concept is the load-bearing piece
  for a substrate analog.
  https://www.sciencedirect.com/science/article/abs/pii/S0149763425001988

### Hopfield basin / landscape reshaping under repeated stimuli
- *Sudden restructuring of memory representations in recurrent
  neural networks with repeated stimulus presentations, PMC 2025
  (Howlett)* — repeated presentations cause abrupt jumps in basin
  size that follow a lognormal distribution. This is the
  "replay carves topology" effect in its purest empirical form.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12585986/
- *Memorisation and forgetting in a learning Hopfield NN: bifurcations,
  attractors, basins, arXiv 2025* —
  https://arxiv.org/html/2508.10765v1

---

## Concrete ideas — substrate-level mechanisms

Each idea names: (a) **substrate change**, (b) **expected landscape
effect**, (c) **headline metric**, (d) **<2-week test plan**,
(e) **anti-homunculus screen**.

---

### Idea 1 — Reverse-replay carving via symmetric-STDP + short-term depression

**Source:** Haga & Fukai 2018; Sainsbury 2024.

**Substrate change.** During the existing replay phase, replace the
forward-only trajectory replay with a *symmetric* update rule plus
short-term depression on the most-recently-active atom. With STD,
the trajectory replays "rebound" backward: A→B→C presented forward
during awake fires C→B→A backward during the consolidation pass.
Apply Hopfield-style outer-product updates from these reverse
sequences into the codebook outer product.

**Expected landscape effect.** Forward replay reinforces basin C
(the goal/reward terminus). Reverse replay seeds basins B and A
*from C* — i.e. the predecessor states acquire a directional pull
toward the terminus. This is the substrate analog of a value
function being smeared backward from reward. Crucially it carves
**directional** basin geometry without ever instantiating a "value"
variable.

**Headline metric.** Δ Recall@K under cued retrieval where the cue
is an *intermediate* position in a known trajectory. If reverse
replay has carved predecessor→successor coupling, cueing position
B should retrieve C (the terminus) with substantially higher
energy gap than the random-replay control.

**<2-week test plan.**
1. (3 days) Add a `replay_direction` mode to the existing replay
   driver: `forward` (status quo), `reverse` (Haga-Fukai-style with
   adaptation on most-recent atom), `bidirectional` (alternating).
2. (2 days) Run Phase-4 trajectory benchmark for all three modes
   under identical seeds.
3. (2 days) Measure (i) basin asymmetry — for a stored A→B→C,
   does the energy gradient at B point toward C more strongly
   than at A points toward B? (ii) Recall@K from intermediate cue.
4. (3 days) Multi-seed (≥5), Wilson CIs, write up.

**Anti-homunculus screen.** PASS. The reversal is a property of the
local depression dynamic + symmetric STDP, not an external decision.
The mechanism is "the most recent atom is briefly suppressed, so the
next-most-active wins, and so on backward." No supervisor.

---

### Idea 2 — Large-SWR-style burst replay: amplitude-gated outer-product update

**Source:** *Large SWRs promote hippocampo-cortical reactivation*
(2025); Ecker et al. eLife.

**Substrate change.** Today's replay applies a uniform-strength
update per replay event. Instead: compute the *energy depth* of the
replayed trajectory at retrieval — this is a free quantity from the
Modern Hopfield `retrieve()` already on hot path. Gate the codebook
outer-product update by a monotone function of energy depth:
shallow replays (low energy, ambiguous attractor) apply weak
update; deep replays (strongly converged attractor) apply strong
update. The "large SWR" is the substrate's deepest-energy replay
events.

**Expected landscape effect.** Strong basins get stronger; weak,
half-formed basins do not get amplified. Functionally equivalent to
the biology's selective amplification of the post-learning subset
of SWRs. Should help cap-coverage by preventing junk replays from
broadening the basin set.

**Headline metric.** Δ cap-coverage with active drift + Δ
meta-stable rate. Specifically, does amplitude-gated replay
*reduce* meta-stable rate (collapse half-formed basins) without
hurting Recall@K?

**<2-week test plan.**
1. (1 day) Add `energy_depth` capture to `torch_hopfield.retrieve()`
   trace output (project memory says this was refactored recently —
   the per-iteration states are already captured).
2. (2 days) Modify replay update: `Δ W = sigmoid((depth - μ)/σ) ·
   outer(pattern, pattern)`.
3. (2 days) Compare against uniform-update control on Phase-4
   benchmark.
4. (3 days) Sweep μ, σ; multi-seed.

**Anti-homunculus screen.** PASS *if* μ, σ are computed from the
network's own running statistics, not set externally. Use an EMA
over recent retrieve depths. If you set μ, σ as hyperparameters,
this is just a tuned threshold — borderline.

---

### Idea 3 — Theta-gamma slot binding as a discrete sequence index

**Source:** Lisman / Jensen; Frontiers 2025 nonlinearity paper;
PLOS Comp Bio 2025 ING/PING paper.

**Substrate change.** The substrate currently uses continuous
FHRR position vectors to bind items to positions in a window.
Replace (or augment) the continuous position vectors with a
**phase-clocked discrete slot index**: a slow theta cycle gates a
fast gamma cycle, where each gamma cycle = one slot, and item i
gets bound to the i-th gamma cycle within a theta cycle. The
"binding" is implemented as a periodic phase term added to the
FHRR position vector. During replay, you sweep the same theta-gamma
clock, which means replayed items emerge in the same gamma-slot
order they were encoded.

**Expected landscape effect.** Item-order information that is
currently carried by continuous position vector projections becomes
*phase-locked*. This makes the order signal more discrete and
should improve cap-coverage of long sequences where continuous
position vectors run out of headroom.

**Headline metric.** Δ Recall@K on long-window encoded sequences;
Δ basin-overlap between adjacent-position items (should *decrease*
under phase locking — different phases ⇒ orthogonal binding).

**<2-week test plan.**
1. (3 days) Add `position_clock.py` implementing theta-modulated
   gamma slot index → FHRR phase vector.
2. (2 days) Plug into `encoding.py` as an alternative position
   scheme; A/B against current continuous positions.
3. (3 days) Run Phase-4 benchmark on long windows (W=16, 32, 64);
   measure cap-coverage and position-confusion matrix.

**Anti-homunculus screen.** PASS. Theta-gamma clock is a free-running
oscillator, no decision about what to bind to which slot — items
just take the next available slot. The slot itself is a periodic
function of time.

---

### Idea 4 — Cue-biased replay (substrate-level TMR)

**Source:** TMR review (Antony / Schreiner, npj Science of Learning
2024).

**Substrate change.** Today's replay samples from the codebook
uniformly (or near it). Instead: maintain a small **"recent cue"
buffer** of the last K retrieved patterns. During the replay
pass, bias the replay sampler so that the seed state for each
replay event is drawn from `0.7 · uniform + 0.3 · cue_buffer`. The
"cue" is implicit — whatever the system has just been recalling
gets preferentially replayed during consolidation, which is the
substrate analog of the daytime trace being reactivated by a
nighttime cue.

**Expected landscape effect.** Recently-active basins get
preferentially deepened — but crucially, *the act of recalling
biases what gets consolidated*, which is the learning-from-
remembering loop. Combined with Idea 2 (amplitude gating), this
should target consolidation on the patterns that the system is
actively using.

**Headline metric.** Δ Recall@K on a *workload-skewed* test:
construct a query distribution that revisits a small subset of
stored patterns 10× more often than the rest. Measure whether
recall on the frequently-queried subset *improves* over uniform
replay (it should), and whether the rest *degrades* significantly
(it should not, much).

**<2-week test plan.**
1. (1 day) Add `cue_buffer` (deque) populated by every `retrieve()`
   call.
2. (1 day) Modify replay seed sampler.
3. (2 days) Build workload-skewed eval.
4. (3 days) Multi-seed; ablate buffer size K ∈ {0, 4, 16, 64};
   compare to uniform-replay baseline.

**Anti-homunculus screen.** PASS — the buffer is filled by the
system's own retrieval activity. No external scheduler decides what
to reactivate. BUT: if you weight the buffer by recency
*explicitly*, that's a parameter. Best version: use the
amplitude-gating from Idea 2 to weight buffer entries
(depth-weighted FIFO).

---

### Idea 5 — Engram-allocation excitability gate

**Source:** Delamare/Clopath J Neurosci 2024; Tarulli 2025 review;
Trends Neurosci 2025 engram-competition piece.

**Substrate change.** Each codebook atom gets a slow-decaying
**excitability** scalar e_i, initialized at zero, updated by:
`e_i ← (1-λ) e_i + λ · 1[atom_i was active in this retrieve]`.
At encoding time of a new pattern, the choice of which atoms
participate is biased by `softmax(β · e)` — the most-excited atoms
preferentially absorb the new pattern.

**Expected landscape effect.** Patterns encoded close in time
share atoms (memory linking, exactly as Delamare/Clopath predict),
which carves *associative* basin overlap. Patterns encoded far
apart use disjoint atoms (separation). This is "free" schema
formation: temporally-clustered events form blobs in atom space.

**Headline metric.** Δ basin overlap between co-temporal pattern
pairs vs. distant pattern pairs. Δ Recall@K for queries that
exploit co-temporal cues.

**<2-week test plan.**
1. (2 days) Add e vector + update + softmax-biased allocation.
2. (2 days) Build a co-temporal vs distant pattern benchmark
   (this may already exist for memory-linking studies).
3. (3 days) Run, multi-seed, write up.

**Anti-homunculus screen.** PASS. The excitability variable is
purely local and per-atom; the allocation is a softmax (energy
function), not an if/then. Note that this is similar to *winner-
take-all* but driven by a slow trace, not instantaneous activation.

---

### Idea 6 — Frequency-weighted Benna-Fusi with a *reactivation-depth*
weight instead of count

**Source:** `STATUS.md` open item + the bioRxiv 2025 large-SWR
selectivity paper + two-factor consolidation 2024.

**Substrate change.** STATUS.md proposes
`α_eff = α_base · (1 + λ · normalized_retrieval_count)`. Replace
the **count** with the **summed-energy-depth** over retrievals.
i.e. `α_eff = α_base · (1 + λ · normalized_summed_depth)`. A
pattern retrieved 10× with shallow energy gets a smaller α
adjustment than a pattern retrieved 5× with deep energy.

**Expected landscape effect.** This is the test for the
"compression → abstraction" claim, but with the biologically
correct signal. Counts are noisy; depths reflect how reliably the
pattern is functioning as an attractor. Should produce sharper
separation between "true memories" (deep, often-retrieved) and
"noise patterns" (count high but shallow).

**Headline metric.** Δ retention of true patterns vs. distractor
patterns under aggressive forgetting pressure. ΔE between
role-prior and content-prior (the load-bearing Phase 5 headline at
phase-5-unified-design.md:256-281) — because depth-weighted
consolidation should preferentially preserve patterns that align
with content-prior basins.

**<2-week test plan.**
1. (1 day) Modify `α_eff` computation to use EMA of depth instead
   of EMA of count.
2. (2 days) Add the count-based version as a control (this is the
   committed STATUS.md experiment, so build BOTH).
3. (5 days) Run side-by-side: vanilla / count-weighted /
   depth-weighted. Multi-seed. Headline = ΔE in design spec.

**Anti-homunculus screen.** PASS. Per-atom local scalar updated
by local activity. No supervisor.

---

### Idea 7 — Langevin replay (committed but never built)

**Source:** `STATUS.md` open item + arXiv 2026 stochastic-attention
paper.

**Substrate change.** During replay, after each Hopfield retrieve,
inject calibrated Gaussian noise into the recovered pattern before
using it for the outer-product update. Noise scale follows the
classical Langevin formula `sqrt(2 · T · dt)` where T is a slow
"sleep temperature" that anneals over the replay pass.

**Expected landscape effect.** Replays no longer settle into the
exact stored pattern — they settle into *neighborhoods* around it.
Outer-product updates therefore broaden basins rather than
sharpening them. Tunable: high T = broad basins = generalization;
low T = sharp basins = fidelity. The annealing schedule provides
"sleep-stage-like" coarse-to-fine consolidation.

**Headline metric.** Δ generalization on held-out variants of
trained patterns. Δ basin width measured by perturbation radius
that still retrieves the correct attractor.

**<2-week test plan.**
1. (1 day) Add `langevin_replay(T_schedule)` to replay pass.
2. (1 day) Sanity check: T=0 reproduces existing behavior bit-exact.
3. (5 days) Sweep T_schedule (constant low, constant high, linear
   anneal, exponential anneal) on Phase-4 benchmark with
   generalization eval.
4. (3 days) Combine with Idea 1 (reverse replay) — these compose:
   reverse + Langevin = broad directional basins.

**Anti-homunculus screen.** PASS. Langevin noise is a stochastic
differential equation, not a decision.

---

### Idea 8 — Schema-channel: dual-rate consolidation with cross-channel coupling

**Source:** Tse-Morris 2007; BrainCog-inspired 2026-05-02 note;
PMC6902718 myelination-schema paper.

**Substrate change.** Add a **second, slower consolidation channel**
parallel to the existing Benna-Fusi cascade. The slow channel
accumulates the *running average* of recent Hopfield outer-products
(not individual events) — i.e. it stores the *centroid* of recent
basin geometry. When a new pattern arrives, its encoding is
shifted toward the nearest existing schema centroid by a small
amount before being consolidated. This is the substrate analog of
"new patterns are assimilated to existing schemas, accelerating
their consolidation."

**Expected landscape effect.** New patterns that "fit" an existing
schema consolidate fast (schema centroid attracts them, so they
deepen an already-deep basin region). New patterns that don't fit
consolidate slowly via the standard cascade. Tests the
Tse-Morris-Bethus accelerated-consolidation claim directly.

**Headline metric.** Δ time-to-stable-recall for schema-consistent
vs schema-inconsistent new patterns (the original Tse-Morris
prediction).

**<2-week test plan.**
1. (3 days) Add schema-centroid store (a small set of FHRR
   centroids updated by running average).
2. (2 days) Add "assimilation shift" to encoder.
3. (5 days) Build the schema vs non-schema benchmark (pre-train on
   a structured family of patterns, then introduce new members
   that either fit or violate the family).

**Anti-homunculus screen.** BORDERLINE. The assimilation shift is
local (distance to centroid), but "which centroid is nearest" is a
nearest-neighbor lookup. As long as the nearest-neighbor is
implemented via energy minimization (Hopfield retrieve on the
centroid set), this stays a local geometric dynamic. If you wire
it as an explicit argmin over a list, that's a controller.

---

### Idea 9 — Reactivation-induced competitor suppression (adaptive forgetting)

**Source:** Wimber et al. Nat Neurosci; Trends Neurosci 2025
engram-competition; Howlett 2025 basin-restructuring paper.

**Substrate change.** When pattern P is retrieved, apply a small
*anti-Hebbian* update to atoms that were partially active (above
threshold but not strongly enough to win) during the retrieve.
Concretely: `Δ W_ij = -ε · partial_activity_i · partial_activity_j`
for the runners-up only, while doing the normal Hebbian update for
the winner.

**Expected landscape effect.** Competing basins get *suppressed
by the act of remembering*, not by an external pruner. Over many
retrievals, the basin set becomes more orthogonal — competitors
that were close to the retrieved pattern get pushed away.
Directly addresses cap-coverage by clearing out near-degenerate
basins.

**Headline metric.** Δ basin separation (mean inter-basin
distance) over consolidation pass. Δ cap-coverage. Δ false-recall
rate.

**<2-week test plan.**
1. (2 days) Add anti-Hebbian "runners-up" term to retrieve-time
   plasticity (Phase 5 already does some retrieve-time updates).
2. (3 days) Tune ε; verify it doesn't destroy stored patterns.
3. (5 days) Multi-seed cap-coverage + false-recall benchmark.

**Anti-homunculus screen.** PASS. The "runners-up" set is defined
by the energy function (atoms above threshold but not the winner) —
no decision. The anti-Hebbian update is a local plasticity rule.

---

### Idea 10 — Reconsolidation-window plasticity gate

**Source:** Nader; Windows of Change 2025 review; Hopfield indexing
paper 2025.

**Substrate change.** When a pattern is retrieved, it enters a
brief "labile window" of N replay events. During this window,
plasticity at the retrieved atoms is *elevated* (e.g. 3× normal).
After the window, plasticity returns to baseline. A pattern that
is retrieved without further context within the window gets
re-stabilized as-is; a pattern that is retrieved *and then*
followed by a similar-but-not-identical pattern within the window
gets the new info *integrated* into the old trace (not stored
separately).

**Expected landscape effect.** Memory updating becomes possible
without storing a new pattern for every minor variant. This
should reduce capacity pressure and let basins absorb small
deformations, which is exactly the "compression → abstraction"
behavior STATUS.md is chasing.

**Headline metric.** Δ stored-pattern count after N variants of a
base pattern are streamed (should plateau at ~1 + small overhead
under reconsolidation, vs. ~N under naive encoding). Δ Recall@K
on variant queries.

**<2-week test plan.**
1. (2 days) Add `labile_window` state per atom (counter + elevated
   plasticity flag).
2. (2 days) Modify encoding path to check labile window before
   allocating new atoms vs. updating existing.
3. (3 days) Build a variant-streaming benchmark.
4. (3 days) Multi-seed; ablate window length.

**Anti-homunculus screen.** PASS. The window is a per-atom timer;
the integration vs new-allocation decision is driven by similarity
energy (already in the substrate), not by a controller.

---

## Anti-homunculus screen — full table

| Idea | Local dynamic? | Decision-free? | Headline metric? | Risk |
|------|---------------|----------------|------------------|------|
| 1 Reverse replay (STD + symmetric STDP) | YES | YES | YES (Recall@K from midpoint) | low |
| 2 Amplitude-gated replay | YES (EMA stats) | YES | YES (cap-coverage) | low if μ,σ are running stats |
| 3 Theta-gamma slot binding | YES (oscillator) | YES | YES (cap-coverage on long seqs) | low |
| 4 Cue-biased replay (TMR) | YES (FIFO) | YES | YES (workload-skewed Recall@K) | low |
| 5 Engram allocation excitability | YES | YES (softmax) | YES (basin overlap) | low |
| 6 Depth-weighted Benna-Fusi α | YES | YES | YES (ΔE design spec) | LOW — directly closes a STATUS.md item |
| 7 Langevin replay | YES (SDE) | YES | YES (generalization) | LOW — closes STATUS.md item |
| 8 Schema dual-channel | borderline | borderline | YES (acc.-consolidation) | MEDIUM — must implement schema lookup as energy minimization, not list-argmin |
| 9 Anti-Hebbian competitor suppression | YES | YES | YES (basin separation) | low |
| 10 Reconsolidation labile window | YES | YES | YES (variant absorption) | low |

The highest-leverage moves (highest gain × lowest controller risk × directly closes a committed STATUS.md item) are:
**Idea 6** (depth-weighted α — closes the frequency-weighted Benna-Fusi commitment with the *correct* signal),
**Idea 7** (Langevin replay — closes the named-but-never-built commitment),
**Idea 1** (reverse replay — directly addresses Phase 5's missing directional/role-like basin geometry).

---

## Surprises

1. **Firing-rate adaptation alone produces the full forward/reverse/diffusive replay spectrum** (Nat Comms 2025). I had assumed reverse replay needed a dedicated mechanism. It doesn't. The substrate could get reverse replay essentially for free by adding adaptation to whatever's driving its replay pass. This may obsolete much of Idea 1's specific STD+sym-STDP machinery — try adaptation first as the simpler intervention.

2. **Large SWRs are the load-bearing subset** (bioRxiv 2025, Neuron 2025). Most replay events do NOT consolidate; only the large-amplitude subset does. This is huge for the substrate — it says current uniform-rate consolidation is wasteful and possibly destructive of useful structure. Idea 2 is the cleanest fix.

3. **Disinhibition (not active drive) is sufficient to release replay** (bioRxiv 2025 sleep-disinhibition paper). This is the most anti-homunculus framing I've found. The substrate could implement replay as "remove the inhibitory cap and let basins fire spontaneously" rather than as "actively reactivate stored patterns." Worth a Phase-5 design-doc footnote even if not directly tested.

4. **SO-spindle coupling *precision* (not amplitude) predicts consolidation** (eLife 2024 Bayesian meta-analysis). The substrate currently has no phase concept — Idea 3 (theta-gamma) would partly address this, but a deeper architectural question is whether the substrate needs a phase-coupling primitive at the consolidation channel level, not just at the encoding level.

5. **Replay events have drift-diffusion dynamics, with internal momentum** (bioRxiv 2025). Replays are not Markov samples; they have inertia. This is consistent with continuous-attractor sweeps but suggests that the substrate's replay pass should be a *trajectory* on the energy manifold, not independent samples.

6. **Repeated stimulus presentations produce LOGNORMAL JUMPS in basin size** (Howlett 2025). This is the most direct empirical evidence that replay reshapes topology in a structured, non-uniform way. It also means our Phase-5 instrumentation should be looking at basin-size distributions, not just averages — the lognormal tail is the signal.

7. **Engram allocation is "just" excitability competition** (J Neurosci 2024 Delamare/Clopath). The biology of memory linking — co-temporal events sharing atoms, distant events using disjoint atoms — falls out of a single scalar (intrinsic excitability) with no decision logic at all. This is one of the cleanest anti-homunculus mechanisms in the entire literature.

---

## Promising leads (ranked by leverage)

1. **Idea 6: Depth-weighted Benna-Fusi α.** Closes the STATUS.md commitment with a strictly better signal (energy depth) than the committed proposal (retrieval count). Directly targets Phase 5's design-spec ΔE headline. Build BOTH (count and depth) for clean ablation. *Build first.*

2. **Idea 1 / firing-rate-adaptation variant: Reverse replay carves directional basins.** Strongest candidate for unlocking Phase 5's missing role geometry. The Nat Comms 2025 result says adaptation alone is sufficient — start there, before the STD+sym-STDP machinery. *Build second.*

3. **Idea 2: Amplitude-gated replay (large-SWR analog).** Cheap (energy depth already computable), high leverage on cap-coverage, composes with Ideas 1 and 7. *Build third.*

4. **Idea 9: Anti-Hebbian competitor suppression.** Implements adaptive forgetting as a local plasticity rule. Directly addresses cap-coverage and false-recall. Cheap to build. *Build fourth.*

5. **Idea 7: Langevin replay.** Closes the named-but-never-built commitment. Composes cleanly with everything else (temperature schedule is orthogonal). *Build any time — could be the noise term added to Ideas 1, 2, 4 simultaneously.*

6. **Ideas 4, 5, 10 (cue-biased replay, engram allocation, reconsolidation window).** Lower priority but each is a clean local dynamic with a clear test. The engram-allocation excitability scalar (Idea 5) is the most beautiful mechanism in the set but the hardest to test in isolation.

7. **Idea 8: Schema dual-channel.** Highest architectural ambition, highest controller risk. Defer until the smaller wins above are in.

8. **Idea 3: Theta-gamma slot binding.** Architecturally large (touches encoding, not just consolidation). Defer unless the easier ideas underperform on long-sequence cap-coverage.

---

## What to put in a brainstorm-output one-liner

> "Frequency-weighted Benna-Fusi α — with retrieval **energy depth**
> (not count) as the weight — is the right next experiment for Phase 5.
> The depth signal is exactly the biology's large-SWR selectivity, it
> closes the committed STATUS.md item with a strictly stronger control
> condition, and it composes with reverse-replay (adaptation-driven)
> and Langevin replay to carve the directional/role-like basin
> geometry Phase 5 is currently chasing in vain."
