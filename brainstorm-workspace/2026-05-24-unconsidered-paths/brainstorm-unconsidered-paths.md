# Brainstorm: Unconsidered Paths Across Every Phase

> **⚠ Correction posted 2026-05-24 after external audit.** This document inherited a stale claim from STATUS.md that the frequency-weighted Benna-Fusi α experiment was "never built." That is wrong: [Report 040](../../reports/040_freq_weighted_alpha_sweep.md) ran it at n=10×40 production scale on 2026-05-17 and found it redundant (λ ∈ {0, 0.5, 1.0} statistically indistinguishable on every metric; λ=2.0 supercritical). The [Phase 5 design doc:97-107](../../notes/emergent-codebook/phase-5-unified-design.md) records the resolution. All "Idea P4.A — Depth-weighted α" and "Theme A — closes a documented commitment" framings below assume freq-α is unresolved; **they should be read with that walk-back in mind.** The two findings that survive intact: (a) range-shaped replay (Dorrell-Whittington ICLR 2025) is genuinely unbuilt, and (c) MQAR-style associative recall is genuinely the missing unit test — synthetic smoke confirms Hopfield-key-only recall at 0% top-1 from N=128.

> Generated 2026-05-24 from STATUS.md, PROJECT_PLAN.md, the Phase 5 design doc,
> all phase checklists, the 64-report empirical record, the dated synthesis notes
> (especially 2026-05-09 anti-homunculus, 2026-05-16 substrate-vs-readout, the
> 2026-05-20 dynamic-form variants, and 2026-05-24 session-close), and six
> parallel deep-research passes covering: alternative VSA algebras; predictive-
> coding codebook growth; replay/sleep landscape dynamics; relational binding
> beyond the rescue brainstorm; Phase 6/7 world-model + LLM-as-voice; continual-
> learning external evaluation.

---

## Project understanding

The project is building a **memory-first cognitive substrate**: FHRR (D=4096
complex unit-magnitude vectors) + Modern Hopfield retrieval + emergent codebook
+ multi-timescale replay/consolidation. Phases 0–4 graduated. Phase 5
(role/structural retrieval on the post-death substrate) is **stuck after four
retrieval-mechanism families returned null** (D1 storage rule swap, D3 cross-K
slot softmax, E1 centered log-prior landscape reshape, M1 role-energy stack).
The diagnosis the project has converged on is that **the FHRR + plain-Hebb
Modern Hopfield substrate, as currently shaped, does not contain role-target
attractors**. The recent rescue brainstorm covered resonator networks,
sparsemax MHN, EqProp / role-shuffled negatives, IDP saliency, slot attention
cross-K softmax, and per-role MHN.

The core commitments are non-negotiable. **Anti-homunculus**: every addition
must be local geometry / energy / settling / tension / consolidation, never
an arbitration. **Headline metric**: one number per phase, drill-downs explain
movement. **Memory is the self**: identity lives in the landscape, not the LLM.
**Local-first** and **contextual completion over token prediction**.

What this brainstorm targets: paths the project has *not yet considered*, that
**span all phases (not just 5)** and **pass the anti-homunculus filter on
inspection**. Three categories of finding emerged:

1. **Documented-but-never-built**. STATUS.md lists open commitments (regime
   classifier, θ′(β) calibration, frequency-weighted Benna-Fusi α, the
   2026-05-02 trajectory-trace meta-loop, Langevin replay noise). These are
   the cheapest expected-value moves in the entire ideation space.
2. **External-endorsement gaps**. The Whittington 2024-2025 cluster
   (Bakermans/Behrens Nat Neurosci 2025, Dorrell/Whittington Neuron 2025,
   Dorrell ICLR 2025) has independently converged on *exactly* the role-atom
   / content-atom / replay-driven-composition architecture this project is
   building. The project's notes do not cite this. Similarly: SQHN (Nature
   Comms 2024) and VAE+MHN-CLS (2024) report Split-MNIST CIL numbers the
   project has never compared against. McAlister et al. 2025 (Neural
   Computation) is prior art on phase transitions in DAM continual learning.
3. **Substrate-algebra alternatives the rescue brainstorm missed**. GHRR
   (non-commutative FHRR variant), GSBC (block-sparse with structural zeros
   between non-overlapping bindings), Residue HDC (role indices as residue
   moduli — keeps the FHRR substrate), the Kymn et al. NeurIPS 2024 *modular*
   resonator (structurally different from what the rescue brainstorm called
   "resonator networks"), and the Hoover/Krotov May 2025 unification
   (diffusion = MHN = attention = energy descent at different temperatures).
   This last one collapses Phase 5 → Phase 6 into one mathematical object.

The single highest-leverage observation: **the project's external
intellectual neighborhood is much larger and more validating than the notes
reflect**. Multiple 2024–2026 published architectures already implement the
project's own architectural intuitions; comparing against them is overdue.

---

## Top recommendations (ranked, ordered by cost × expected ΔE × architectural
fit)

| # | Idea | Phase | Cost | Why now |
|---|------|-------|------|---------|
| 1 | **Range-shape the replay buffer** (factorise role/content sampling, no architecture change) | 3/4/5 | <1 day | Dorrell/Whittington ICLR 2025 theorem says rectangular joint support *forces* modularisation. If data is the bottleneck, no Phase 5 architecture change is needed. |
| 2 | **Depth-weighted Benna-Fusi α** (replace count with `max(energy_depth)` in α_eff) | 4 | 3-4 days | Strictly stronger than the never-built count-weighted α already in STATUS.md. Matches large-SWR biological selectivity (Neuron 2025). Closes a flagged open commitment. |
| 3 | **MQAR + bAbI 1/4/5/11/12 + CLUTRR as Phase 5 sanity check** | 5 | 1-2 weeks | The project's current ΔE headline cannot distinguish "role-binding works" from "content-binding works in role-like cues." MQAR and CLUTRR can. If they fail, the Phase 5 diagnosis is correct; if they unexpectedly pass, the substrate has been doing role-binding all along and the ΔE metric is wrong-shape. |
| 4 | **Residue HDC over the existing FHRR substrate** | 1/5 | 1-2 weeks | Smallest-blast-radius substrate-algebra upgrade: keeps FHRR phase representation, MHN energy, codebook, replay; only role indices change to residue moduli. Makes different roles algebraically non-overlapping by construction. Kymn-Kleyko-Frady-Olshausen-Sommer 2023. |
| 5 | **Phase-clock slow variable on FHRR atoms** | 5 | <1 week (50 LOC prototype) | Most surgical Phase-5 addition. Per-atom phase clock rotates by role-specific Δθ during settling; role basins emerge as phase-coherence basins, not as parameters. |
| 6 | **Modular resonator network (Kymn et al. NeurIPS 2024)** | 5 | 2-3 weeks | Structurally different from the rescue brainstorm's "resonator" entry. Each role is its own attractor module; modules enforce self-consistency through FHRR-compatible binding. Role basins are settling-dynamic, not codebook geometry. |
| 7 | **Reverse-replay carving via firing-rate adaptation + symmetric STDP** | 4/5 | 1-2 weeks | Nat Commun 2025 result: adaptation alone produces the full forward/reverse/diffusive replay spectrum. Reverse replay carves directional basins — the exact missing role geometry Phase 5 needs. |
| 8 | **PCN-above-MHN (Salvatori et al. predictive coding networks)** | 3/5 | 2-3 weeks | Published ~10× better than MHN on associative memory benchmarks. Local layer-wise gradient, no controller. Project's notes do not cite this comparison. |
| 9 | **Temporal Predictive Coding head as an additive energy term** (Tang/Bogacz NeurIPS 2023) | 3/5 | 1-2 weeks | Drops into the existing energy function as one extra local term. Bogacz line has shown PCN = Hopfield-with-covariance-learning, so this is architecturally additive, not replacement. |
| 10 | **Phase 6 architecture as Hoover/Krotov unification** | 6 | Months (paradigm-level) | Diffusion = MHN = attention = energy descent. Phase 5 → Phase 6 is the same object viewed at different temperatures, not a new module. Reframes Phase 6 from "build a transition model" to "anneal the existing memory landscape." |

---

## Ideas and approaches, organised by phase

### Phase 0/1 — Substrate

The substrate is treated as load-bearing settled physics. But the recent Phase
5 nulls suggest the substrate may itself be the problem, not the retrieval
mechanism layered on top.

#### Idea P1.A — Residue HDC over the existing FHRR substrate
**What.** Keep FHRR (complex unit-magnitude D=4096), keep MHN energy, keep
codebook, keep replay. Assign role indices to *residue moduli* (Kymn et al.
2023). Different roles are algebraically non-overlapping by construction:
unbind(cue, role_i) lives in a residue subspace disjoint from unbind(cue,
role_j). The four-family Phase 5 null suggests roles don't separate in the
current shared subspace; residue HDC forces the separation at the algebra
layer.
**Why now.** Smallest-blast-radius substrate-algebra change in the entire
search space. Resonator factorization over residues is a well-studied
problem with capacity guarantees.
**Anti-homunculus.** ✅ Algebraic. The "decision" of which role to look up
is gradient descent on per-residue energy; no controller.
**How to test.** Use existing Phase 5 ΔE protocol, swap FHRR position
vectors for residue-modulus roles. Sweep moduli 2-8.
**Source.** Kymn-Kleyko-Frady-Olshausen-Sommer 2023; VSA-Lisp 2025
(arXiv:2511.08767).

#### Idea P1.B — GSBC (Generalized Sparse Block Codes) as alternate substrate
**What.** Split D=4096 into B=64 blocks of D/B each, sparse one-hot or
top-k within each block. ℓ∞-similarity gives **structural zeros** between
non-overlapping bindings — geometrically impossible in FHRR cosine.
Operational factorisation capacity ~5×10⁶ vs ~10⁴-10⁵ for FHRR resonators.
**Why now.** The Phase 5 null is consistent with "all role-cue overlaps
look similar in cosine"; GSBC's structural zeros remove that confusion.
**Anti-homunculus.** ✅ Sparse blocks are a substrate property, not a
routing decision.
**How to test.** Re-run the Phase 5 ΔE protocol on a GSBC substrate at
matched D. Reference code: IBM/in-memory-factorizer.
**Source.** Hersche et al. 2023/2025.

#### Idea P1.C — GHRR (Generalized HRR, non-commutative variant)
**What.** Algebraic upgrade: drop the commutativity of FHRR binding,
giving the algebra a notion of role/filler asymmetry at the operator
level. Drop-in substrate change at D=4096.
**Why now.** The current substrate treats `bind(role, filler) =
bind(filler, role)`; this is part of what makes the role-basin absent.
**Anti-homunculus.** ✅ Algebraic.
**How to test.** Swap binding operator. Cost: ~50 LOC.
**Source.** arXiv:2405.09689 (2024).

#### Idea P1.D — TPAM (Threshold Phasor Associative Memory) on existing FHRR
**What.** Frady-Sommer 2019 PNAS construction: a complete energy-function-
based phasor memory the project's FHRR substrate can host directly. Sits
*between* current MHN and a future Resonator stack.
**Why now.** The rescue brainstorm did not catalogue this. It is the
closest pre-built energy memory designed for the project's exact vector
type.
**Anti-homunculus.** ✅ Energy memory.
**Source.** Frady & Sommer, PNAS 2019.

---

### Phase 2 — Static contextual completion

Phase 2 has minimal coverage beyond the existing WikiText-2 runner. The gap
that matters: there is no **external** evaluation that distinguishes
content-matching from role-binding.

#### Idea P2.A — Adopt MQAR as the substrate's missing unit test
**What.** Multi-Query Associative Recall (Stanford Hazy Research). Synthetic
KV recall over capacity curves. Attention is flat; Mamba and many memory-
augmented baselines degrade. The Phase 2 design has nothing that produces
a capacity curve.
**Why now.** Phase 2's headline (Recall@K on masked-token completion) can
be high while role-binding capacity is zero. MQAR sees through that.
**Anti-homunculus.** ✅ Cue-driven, no policy.
**How to test.** Drop existing substrate into the Hazy MQAR harness. One
weekend.

#### Idea P2.B — Linear probing as Goodhart guard
**What.** Davari CVPR 2022 convention: at every phase checkpoint, train a
linear probe on the latent state for a task the substrate wasn't optimised
for. If cap-coverage and ΔE move but the linear probe doesn't, the headline
is Goodharting.
**Why now.** STATUS.md flags the "substrate-vs-readout discipline" tension
(2026-05-16); linear probing is the canonical operational answer.
**Anti-homunculus.** ✅ The probe is the test, not the system.

---

### Phase 3 — Growing codebook

#### Idea P3.A — Range-shape the replay buffer (Dorrell-Whittington 2025) ★
**What.** *Data-side* intervention with no architecture change. Sample role-
context and content-context independently in the replay buffer (rectangular
joint support) rather than co-sampling whole episodes. ICLR 2025 theorem
"Range, not Independence, Drives Modularity" says rectangular support
*forces* modularisation in nonneg + energy-efficient autoencoders.
**Why now.** Cheapest possible intervention. If this moves ΔE above the
5.5e-3 floor, the project was data-bound, not architecture-bound — and the
Phase 5 stuck state dissolves without M2/EqProp.
**Anti-homunculus.** ✅ Sampling discipline at the buffer level. No
runtime arbitration.
**How to test.** Modify replay sampler; re-run Phase 5 headline at
n=10×30. <1 day implementation; ~3 days including the n=10 confirmation.
**Source.** Dorrell, Whittington et al., ICLR 2025
(arXiv:2410.06232).

#### Idea P3.B — SFA (Slow Feature Analysis) head on the replay buffer ★
**What.** Add a slowness loss on the latent state across replay traces:
atoms that change slowly over the trace become "role atoms," fast-changing
become "content atoms." Genuinely white-space — no group has done this on
a Hopfield/MHN codebook. Franzius-Sprekeler-Wiskott 2007 showed slowness
alone produces head-direction-like (= role-like) cells from raw input.
**Why now.** This is the substrate-level analog of "what stays slow under
replay is a role." Phase 5's missing role basins are precisely what an
SFA loss would carve.
**Anti-homunculus.** ✅ Loss function on a local quantity (state
derivative). No supervisor.
**How to test.** Add SFA loss to consolidation pass; smoke at n=3 seeds
× 30 cues.

#### Idea P3.C — Temporal Predictive Coding head (Tang/Bogacz NeurIPS 2023)
**What.** Adds one local energy term: prediction of next state given
current state, with local layer-wise gradient. Salvatori line has shown
PCN = Hopfield-with-covariance-learning, so this is architecturally
additive, not replacement.
**Why now.** Drops directly into the existing energy function.
**Anti-homunculus.** ✅ Local layer-wise PE gradient. The "decision" is
energy descent.
**Source.** Tang, Barron, Bogacz NeurIPS 2023.

#### Idea P3.D — Consolidation-geometry regime classifier (open STATUS.md commitment)
**What.** STATUS.md flags this as a never-built pre-Phase-3 commitment.
Compute `d̄` and `d_eff` per atom; classify tight vs. spread regime per
the Geometry of Consolidation protocol.
**Why now.** Phase 3 graduation is technically incomplete without it. The
classifier is the stratification axis for the Phase 3 headline. Two days
of work.
**Anti-homunculus.** ✅ Diagnostic measurement.

#### Idea P3.E — Empirical θ′(β) calibration spike (open STATUS.md commitment)
**What.** Replace the θ′ ≈ 1/β approximation with the calibrated mapping
via the Geometry of Consolidation E1 protocol. Two days. Recommended
pre-Phase-3 in the 2026-05-09 note; never done.
**Anti-homunculus.** ✅ Calibration.

---

### Phase 4 — Replay & consolidation

#### Idea P4.A — Depth-weighted Benna-Fusi α (replaces the never-built freq-weighted α) ★
**What.** STATUS.md flags `α_eff = α_base × (1 + λ · normalized_retrieval_count)`
as the unbuilt "compression → abstraction" experiment. **Stronger version:**
replace count with `max(energy_depth)` across retrievals. Neuron 2025: only
LARGE sharp-wave-ripples consolidate; uniform-rate replay is wasteful.
Energy depth is the substrate's analog of large-SWR amplitude. So weight α
by depth, not count.
**Why now.** Closes the STATUS.md commitment with a strictly stronger
biological signal. ~3 LOC change to the replay loop.
**Anti-homunculus.** ✅ Per-atom plasticity gain modulated by a local
geometric quantity (basin depth).

#### Idea P4.B — Reverse-replay carving via firing-rate adaptation + symmetric STDP ★
**What.** Add firing-rate adaptation (short-term depression) to atoms during
replay. Nat Commun 2025 result: adaptation alone produces the full forward
+ reverse + diffusive replay spectrum from a single homogeneous network.
Reverse replay traverses trajectories backwards, which carves **directional
basins** — the exact missing role geometry Phase 5 needs.
**Why now.** Mechanism is local (adaptation = per-atom STD scalar).
Carves landscape topology rather than rehearsing existing patterns.
**Anti-homunculus.** ✅ Local adaptation variable; no scheduler decides
which traces to reverse.
**Source.** Pang & Maoz, Nat Commun 2025 (search for "adaptation reverse
replay 2025").

#### Idea P4.C — Langevin noise during replay settling (open STATUS.md item)
**What.** Add temperature-controlled noise during replay re-settling so
trajectories can escape shallow local minima. Named-but-never-built in
the 2026-05-03 Kona/EBRM note.
**Why now.** Compounds with P4.B: adaptation generates the diversity,
Langevin lets the diversity actually move. Closes a documented gap.
**Anti-homunculus.** ✅ Temperature schedule, not a controller.

#### Idea P4.D — Trajectory-trace meta-loop (BrainCog 2026-05-02 design, never built)
**What.** Full design exists. Three components: (1) trajectory trace
observer (~30 LOC), (2) fast-replay-store gating on high-engagement +
low-resolution traces, (3) second consolidation channel for discovered
patterns. The 2026-05-02 note identifies this as the meta-loop instantiation.
**Why now.** Three weeks of work to fully cash the architectural premise
"the system learns from the act of remembering." Currently sitting designed
and unbuilt.
**Anti-homunculus.** ✅ Passive observer + density-gated consolidation.

#### Idea P4.E — Schema dual-channel consolidation (Tse-Morris)
**What.** Second consolidation channel for traces that match a previously
consolidated schema — these consolidate faster, in line with the Tse-Morris
finding that schemas accelerate hippocampal-cortical transfer.
**Why now.** Composes with P4.D's "discovered-pattern" channel. Could be
the same channel.
**Anti-homunculus.** ✅ Density gate, not a decision module.

#### Idea P4.F — Engram allocation via per-atom excitability scalar (Delamare/Clopath J Neurosci 2024)
**What.** Single per-atom excitability scalar `e_i` modulates encoding
probability. Delamare-Clopath 2024 show engram-linking falls out of this
with zero decision logic.
**Anti-homunculus.** ✅ Local scalar plasticity.

---

### Phase 5 — Structure & abstraction (beyond the rescue brainstorm)

The rescue brainstorm already covered: resonator networks (the *original*
formulation), per-role MHN, pseudo-inverse storage, sparsemax / Hopfield-
Fenchel-Young, cross-K slot softmax, IDP saliency, EqProp + role-shuffled
negatives, KL two-stream pressure, Energy Transformer. These are not
re-suggested.

#### Idea P5.A — Phase-clock slow variable on FHRR atoms ★
**What.** Most surgical Phase-5 addition. FHRR atoms are already complex
unit-modulus. Add a per-atom *phase clock* that rotates by a role-specific
Δθ during settling. Role basins emerge as **phase-coherence basins**, not
as parameters. Atoms whose phases align under the role-specific rotation
settle into a coherent role-basin; misaligned atoms cannot bind.
**Why now.** ~50-line prototype. Uses existing FHRR substrate. Role basin
is a dynamic property of settling, not a designed-in parameter.
**Anti-homunculus.** ✅ Phase rotation is a settling-dynamic primitive.

#### Idea P5.B — Modular resonator network (Kymn et al. NeurIPS 2024) ★
**What.** Structurally different from the "resonator network" entry in the
rescue brainstorm. Each role is its own attractor module; modules enforce
self-consistency through FHRR-compatible binding. Update rule:
`ĝ_i ← σ(G_i G_i† (p ⊙_{j≠i} g*_j))`. Role basins emerge from settling
dynamics, not codebook geometry.
**Why now.** The rescue brainstorm's "resonator" reading was the 2020
single-codebook formulation. The modular 2024 formulation is what the
Phase 5 design actually wants. Re-read before any further architectural
decision.
**Anti-homunculus.** ✅ Each module is an attractor; cross-module
coupling is by binding/unbinding, not arbitration.
**Source.** Kymn et al. NeurIPS 2024 (arXiv:2406.18808).

#### Idea P5.C — PCN-above-MHN (Salvatori, Song, Millidge, Lukasiewicz)
**What.** Add a Predictive Coding Network layer above the MHN substrate.
Published result: PCN outperforms Modern Hopfield by ~10× on associative
memory benchmarks (e.g., ≤9 vs ~all on CIFAR-10 at 50% occlusion).
Local layer-wise gradient, no controller.
**Why now.** Not cited in any project note. If the published 10× holds on
the FHRR substrate, Phase 5 dissolves.
**Anti-homunculus.** ✅ Local PE gradient.
**Source.** Salvatori et al., "Associative Memories via Predictive Coding."

#### Idea P5.D — Compositional energy minimisation (Du et al. 2025)
**What.** Gradient descent on `E_role + E_filler`. Particle-based. Beats
domain-specific solvers on N-Queens / SAT / coloring. The composition is
algebraic at the energy level — no controller decides which energy to
minimise.
**Source.** arXiv:2510.20607 (Oct 2025).

#### Idea P5.E — HFYN + SparseMAP retrieval
**What.** Hopfield-Fenchel-Young Network with SparseMAP retrieval (deep-spin
2024). SparseMAP retrieves pattern associations *under structural
constraints* — literally the primitive the four failed Phase 5 mechanisms
were trying to approximate.
**Anti-homunculus.** ✅ Structural constraint is in the loss/energy, not
a routing rule.
**Source.** github.com/deep-spin/SSHN.

#### Idea P5.F — DEQ implicit-gradient training (replaces EqProp)
**What.** Deep Equilibrium training: analytical implicit differentiation
through the existing MHN settling loop. Replaces EqProp's two-phase nudge
protocol with a one-pass implicit gradient. Same training objective; better
numerical conditioning; doesn't require role-shuffled negatives.
**Anti-homunculus.** ✅ Recipe change only.

#### Idea P5.G — Theta-gamma discrete slot binding
**What.** Cross-frequency coupling published 2022: role on slow rhythm
(theta), filler on fast rhythm (gamma). Multiplicative capacity gain.
Discretises slots in time rather than in algebra. Not in the rescue
brainstorm.
**Anti-homunculus.** ✅ Oscillation is a dynamic; nesting is a phase-
locking property.

#### Idea P5.H — Soft TPR (NeurIPS 2024 revival)
**What.** Tensor Product Representations with amortised dimensionality.
The TPR rejection in the project's folk knowledge predates this paper.
No explicit TPR-rejection rationale exists in `notes/` (grep returns
nothing). Worth re-evaluating in light of NeurIPS 2024.
**Anti-homunculus.** ✅ Outer-product is algebraic.

---

### Phase 6 — Predictive world model & latent rollouts

#### Idea P6.A — Hoover/Krotov unification reframes Phase 6 as one object ★
**What.** May 2025 result (arXiv:2505.21777 + arXiv:2506.11043): diffusion
models, Modern Hopfield Networks, and transformer attention are the **same
operation** — energy descent at different temperatures. The Phase 5
landscape *is* the Phase 6 transition model viewed at higher temperature
with a step-rollout interpretation.
**Why now.** Reframes Phase 6 from "build a new module" to "anneal the
existing memory landscape." Compresses two phases of work into one
architectural object. Most paradigm-level finding in this brainstorm.
**Anti-homunculus.** ✅ Temperature schedule, not a controller.

#### Idea P6.B — Planning as Descent (arXiv:2512.17846, Dec 2025)
**What.** Planning = gradient descent on a goal-conditioned energy
landscape. No policy network. No tree search. 95% on OGBench vs 68% prior.
**Why now.** This is what Phase 6 latent rollouts should look like under
the project's anti-homunculus rule. Cite-and-implement template.
**Anti-homunculus.** ✅ Pure energy descent. Goal is a clamp on the
energy, not a chooser.

#### Idea P6.C — EFE-as-Variational-Inference (Apr 2025)
**What.** Friston/Parr line; recent operationalisation (arXiv:2504.14898).
Planning IS variational inference — no separate optimiser. Composes
naturally with energy memory.
**Claimable gap.** No 2026 paper combines active inference with FHRR /
Modern Hopfield substrates. The Phase 6 design proposed here would be
that paper.
**Anti-homunculus.** ✅ Free-energy minimisation is the dynamic.

#### Idea P6.D — EBWM / EBT (Energy-Based World Model / Transformer)
**What.** NeurIPS 2024 energy-based transformer for autoregressive-style
transition prediction with System-2 settling. Drop-in for the Phase 6
latent transition T(z_t → z_{t+1}).
**Anti-homunculus.** ✅ Energy descent.
**Source.** arXiv:2406.08862.

---

### Phase 7 — LLM interface

#### Idea P7.A — Anti-MemGPT architecture: LLM as frozen voice ★
**What.** Counter to MemGPT/Letta (which make the LLM the self-editor —
the homunculus). Architecture: **LLM body frozen**; substrate is identity;
replay updates only small projection matrices P_enc (text → workspace cue)
and P_dec (settled state → tokens). Swapping the LLM swaps only the
projections; the substrate is untouched.
**Why now.** The MemGPT/Letta architectural pattern is the exact pattern
PROJECT_PLAN.md says not to build ("LLM as source of identity"). Phase 7
needs an explicit anti-template to compare against.
**Anti-homunculus.** ✅ LLM is decoder, not decider. Tool-use becomes
"latent commitment from settling," not "LLM decides to call."

#### Idea P7.B — Sparse memory finetuning for the projections
**What.** arXiv:2510.15103 (Oct 2025): sparse memory finetuning + replay.
Composes with P7.A: replay drives small-scale projection updates without
touching the LLM body.

---

## Cross-cutting themes

### Theme A — Documented commitments still outrank novel ideas

STATUS.md flags four "designed but never built" items: consolidation-geometry
regime classifier, θ′(β) calibration, frequency-weighted Benna-Fusi α, and
the 2026-05-02 trajectory-trace meta-loop. All four are low-cost (<2 weeks
each), pass anti-homunculus cleanly, and address known gaps. Before adding
new ideas to the queue, close these. The depth-weighted-α reformulation
(Idea P4.A) is a *strictly stronger* version of one of them.

### Theme B — External-endorsement gaps are larger than the project realises

The Whittington cluster (Bakermans/Behrens Nat Neurosci 2025, Dorrell/
Whittington Neuron 2025, Dorrell ICLR 2025) has converged on the project's
own architectural intuitions. SQHN (Nature Comms 2024) and VAE+MHN-CLS
(2024) report Split-MNIST CIL numbers the project has never compared
against. McAlister 2025 (Neural Computation) is prior art on phase
transitions in DAM continual learning. PCN-vs-MHN comparisons are
published but uncited. The project is closer to the published frontier
than the notes suggest, and the right next session may be **literature
review, not implementation**.

### Theme C — The substrate-vs-readout discipline applies to the algebra too

The 2026-05-16 substrate-vs-readout note demoted ΔR@10 from headline to
drill-down because the readout was confounding the substrate signal. The
same logic applies to **the binding algebra itself**: FHRR is the readout
of the project's structural intent, and four retrieval-mechanism families
on top of it have failed. The next discipline step is to test alternative
algebras (Residue HDC, GSBC, GHRR) on the same retrieval mechanisms,
inverting the recent experimental design.

### Theme D — Phase 5 → Phase 6 may be one object, not two

Hoover/Krotov May 2025 collapses the diffusion / MHN / attention trio into
one energy descent. The Phase 5 design has been treated as "build a static
role-binding memory"; Phase 6 has been treated as "build a transition
model on top." If the Hoover/Krotov framing holds, both phases are facets
of one annealed energy landscape, and the right Phase 5 metric is the
landscape's *temperature schedule*, not its static `hit_role`.

### Theme E — Eval is currently biased toward content-matching

WikiText rewards content-matching. The Phase 5 ΔE headline rewards
energy-margin between role-prior and content-prior. Neither distinguishes
"role-binding works" from "content-binding works on role-like cues."
MQAR, bAbI 1/4/5/11/12, and CLUTRR-by-path-length do. Adopting these as
external sanity checks is overdue.

---

## Challenges and counterarguments

- **Substrate change risk.** Residue HDC / GSBC / GHRR all require
  re-running the whole post-death substrate pipeline. The current substrate
  is a hard-won artifact; substrate changes break every existing report's
  baseline. Mitigation: run substrate swaps in parallel branches with the
  existing Phase 5 protocol, not in main.
- **PCN-above-MHN may not transfer.** The published 10× gain over MHN is
  on image associative memory, not FHRR + WikiText cue regimes. The gain
  could be substrate-dependent. Mitigation: smoke-test on a small slice
  before committing.
- **Range-shaping is sampling discipline, not architecture.** It might
  move ΔE only marginally if the buffer's joint support is already mostly
  rectangular. Mitigation: measure joint support shape first (~1 hour);
  decide whether to commit.
- **Modular resonator is heavier than the rescue brainstorm covered.**
  Re-introducing it requires reading the Kymn 2024 paper carefully and
  re-doing the anti-homunculus audit; the per-role attractor modules
  could become controller-shaped under pressure.
- **Phase 7 anti-MemGPT pattern is the right shape but unvalidated.**
  No published system implements "LLM as frozen voice + replay-trained
  projections + substrate-as-identity." The project would be claiming
  this architectural pattern.

---

## Rabbit holes worth following

1. **"Attention as Binding"** (arXiv:2512.14709, Dec 2025): independent
   proposal for explicit binding/unbinding heads + hyperdimensional memory
   layers + role-filler separation training. May converge with the project's
   own architecture. Worth one full read.
2. **Histogram-recovery VSA** (Deng-Raviv, Nov 2025): Reed-Solomon +
   Hadamard concatenation; only VSA with formal recovery guarantees on
   compositional decoding. Theoretically strongest but least mature.
3. **Reconsolidation labile window** (Nader, Dudai): traces re-enter a
   labile state when retrieved. Substrate analog: a plasticity gate that
   opens on retrieval and closes on time. Could be the cleanest way to
   close the "the system learns from the act of remembering" loop.
4. **Successor representation over atoms** (Stachenfeld-Botvinick-Gershman):
   could encode role-binding implicitly as a predictive map; under-
   investigated in the Hopfield literature.
5. **Mattar-Daw prioritised replay** as a *local* dynamic: the published
   form is gain × need (controller-shaped), but the gain-need product
   could be reformulated as a local energy term. Worth one design pass.
6. **NEF / Spaun-style binding** (Eliasmith): full population-coded
   binding system that already exists end-to-end. Not in the rescue
   brainstorm. Substrate-compatibility unclear but worth a half-day check.

---

## Sources

### Substrate algebras (research/01)
- GHRR, arXiv:2405.09689
- GSBC, Hersche et al. 2023/2025; github.com/IBM/in-memory-factorizer
- Residue HDC: Kymn-Kleyko-Frady-Olshausen-Sommer 2023; VSA-Lisp arXiv:2511.08767
- HFYN + SparseMAP: github.com/deep-spin/SSHN
- Histogram-recovery VSA: Deng-Raviv 2025
- Attention as Binding: arXiv:2512.14709

### Predictive coding + codebook (research/02)
- Tang, Barron, Bogacz NeurIPS 2023 — Temporal Predictive Coding for Sequential Memory
- Dorrell, Whittington et al. ICLR 2025 — "Range, not Independence, Drives Modularity" (arXiv:2410.06232)
- Bakermans, Whittington, Behrens Nat Neurosci 2025 — Composition & Replay
- Dorrell, Whittington Neuron 2025 — Structured Slots
- Franzius, Sprekeler, Wiskott 2007 — Slowness → place/head-direction cells
- Salvatori et al. 2024 — Associative Memories in the Feature Space
- GCQ 2025, arXiv:2510.16039

### Replay & sleep dynamics (research/03)
- Pang & Maoz Nat Commun 2025 — adaptation produces forward/reverse/diffusive replay
- Neuron 2025 — only large SWRs consolidate
- Howlett 2025 — repeated presentations produce lognormal basin-size jumps
- Delamare & Clopath J Neurosci 2024 — engram linking from excitability scalar
- Tse, Morris et al. — schemas accelerate consolidation
- Mattar & Daw 2018 — prioritised replay

### Relational binding alternatives (research/04)
- Kymn et al. NeurIPS 2024 — modular resonator (arXiv:2406.18808)
- Salvatori, Song, Millidge, Lukasiewicz — PCN > MHN on associative memory
- Du et al. 2025 — compositional energy minimisation (arXiv:2510.20607)
- Soft TPR — NeurIPS 2024
- Frady & Sommer PNAS 2019 — TPAM

### Phase 6/7 (research/05)
- Hoover & Krotov May 2025 — diffusion = MHN = attention (arXiv:2505.21777, arXiv:2506.11043)
- Planning as Descent, arXiv:2512.17846 (Dec 2025)
- EBWM / EBT, arXiv:2406.08862 (NeurIPS 2024)
- EFE-as-Variational-Inference, arXiv:2504.14898 (Apr 2025)
- Sparse memory finetuning, arXiv:2510.15103 (Oct 2025)
- V-JEPA 2 (Meta 2024-2025)

### Continual learning + eval (research/06)
- MQAR — Stanford Hazy Research
- bAbI / CLUTRR / Long Range Arena
- Avalanche / Mammoth continual-learning suites
- SQHN — Alonso & Krichmar, Nature Comms 2024
- VAE + MHN-CLS 2024
- McAlister et al. 2025 — phase transitions in DAM, Neural Computation 37(10)
- Lopez-Paz GEM 2017 — ACC, BWT, FWT triad
- Davari CVPR 2022 — linear probing

---

*Per-angle research briefs in `research/01-*.md` through `research/06-*.md`.*
