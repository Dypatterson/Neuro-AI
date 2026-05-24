# Prior Phases Context: What Cannot Be Un-Decided (Phases 2–4)

**Date:** 2026-05-23
**Purpose:** Structured context for Phase 5 brainstorm, synthesizing Phases 2–4 architectural decisions, substrate constraints, and empirical findings that constrain Phase 5 design space.
**Audience:** Phase 5 design session; anti-homunculus audit.

---

## Executive Summary

Phases 2–4 built a retrieval-native substrate (FHRR + learnable codebook + Benna-Fusi consolidation) and validated that it can produce *structurally clean settling* under online Hebbian learning and replay-driven consolidation. Phase 5 inherits:

1. A substrate dimensionality ~40–45 (out of 4096) after Phase 4 mass-death filters to ~5–30 atoms per scale.
2. Non-negotiable anti-homunculus discipline: all decisions must distribute into local geometric dynamics, not supervisor arbitration.
3. Three critical empirical findings: (a) online error-driven contrastive updates collapse retrieval and are OUT OF SCOPE; (b) metric discipline: readout is Phase 6/7's job, substrate is Phase 4's; (c) binary death works for D1 but breaks K-branch settling by collapsing effective dimensionality.
4. A settled design for Phase 5: **energy-guided structural branching (HAM × top-K schema priors from post-death substrate)** as the unified mechanism for binding, splitting, and hierarchical retrieval.

Phase 5 cannot change the substrate's intrinsic capacity without re-opening Phases 2–4. Phase 5 cannot add supervisory routing. Phase 5 must produce atoms or schemas that either unambiguously split (two distinct settled patterns) or unambiguously co-occur (bundle), with no homunculus reading a metric and deciding.

---

## Phase 2: Substrate + FHRR

**Built:**
- Holographic Reduced Representations (FHRR) in PyTorch/MPS, D=4096.
- Hopfield-style associative retrieval with modern softmax energy (parameter β controls feature-to-prototype regime).
- Static random codebooks (lexicon-sized: ~5000 atoms).
- Masked-token contextual-completion evaluation (more architecturally appropriate than sequence prediction for an associative memory).

**Dimensionality & invariants:**
- FHRR atoms are unit-magnitude per dimension (complex unit circle per coordinate).
- Binding is multiplication; unbinding is conjugate multiplication; bundling is circular convolution.
- The substrate operates at **effective dimensionality d_eff = 4096**, but stored patterns (codebook atoms) occupy a much lower subspace due to structure.
- Hopfield retrieval uses softmax energy: energy(pattern) ∝ β · Re(⟨settled_state, pattern⟩). Low β → multiple patterns active (feature mode, blended); high β → single pattern dominates (prototype mode, sharp).

**Non-negotiable rules that all downstream phases inherit:**
- The pure-Python FHRR reference implementation cannot be removed; it is the ground truth for all optimizations.
- No module that decides which subsystem wins. Settling is *coupled*: all energy terms (content, temporal, structural, value, etc.) act on the same latent state simultaneously.
- Contextual completion (pattern completion given partial cue) is the native question, not sequence prediction. Tries to make it predict-like fight the substrate.

**Validation (Phase 2 exit criteria):**
- Static random codebooks support retrieval better than content-only baselines (memorization case).
- Generalization failure modes are characterized (cap-coverage, metastable-state rate introduced here).
- Masked-token setup beats next-token setup by a clear margin.
- MPS acceleration is functional.

---

## Phase 3: Emergent Codebook

**Built:**
- **Atom entry:** Lexical allocation (new token → new random atom on first encounter). Not pattern-based or error-driven; the latter two are deferred.
- **Two-pathway refinement (hybrid Hebbian + error-driven):**
  - **Hebbian pathway (success-reinforced):** when retrieval quality is high, atoms drift toward their co-occurrence context bag (other atoms in the cue). Small magnitude, continuous.
  - **Error-driven pathway (failure-driven):** when retrieval quality is low, atoms drift toward making failed cues look more like successful patterns. Batched, larger magnitude, fires on consolidation events.
  - Both pathways coexist, gated per-experience by retrieval quality q ∈ [0, 1]. High q → Hebbian-dominant. Low q → error-driven-dominant.
- **Stability tracking:** moving variance of each atom's vector. Gates consolidation depth (stable atoms get smaller updates) and decay protection.
- **Decay:** time-based AND quality-based (both conditions required). Prevents pure time-based aggression (rare-but-important atoms die) and pure quality-leniency (low-utility atoms persist).
- **Repulsion:** when atoms drift too close, they are pushed apart slightly. Prevents collapse.
- **Bimodality tracking (signal only; splitting deferred to Phase 5):** persistent multimodality in context-bag distribution flagged for later splitting.

**Consolidation geometry diagnostic (from Vangara & Gopinath 2026):**
- **Tight regime:** mean within-cluster cosine distance d̄ < θ' (retrieval slack). Hebbian/centroid drift is near-optimal. Error-driven adds nothing.
- **Spread regime:** d̄ ≥ θ'. Atoms' contexts are too diverse; consolidation destroys retrievability unless the error-driven pathway pulls atoms toward correct patterns.
- The regime boundary is hard: an atom in spread regime will lose identity under consolidation regardless of the consolidator, unless error-driven provides correction.

**Critical empirical finding (reports 028–039):**
- **Online error-driven contrastive updates collapse retrieval.** Tested on random init and pretrained codebooks. Top-1 accuracy → chance by ~3000 cues.
- **Consequence:** Error-driven consolidation is out of scope for streaming / online use. It belongs in batch retraining passes (`ReconstructionLearner`, `ErrorDrivenLearner`), not in the hot path.
- **Runtime implication:** At inference/conversation, the codebook uses only Hebbian reinforcement (small, light updates on successful retrievals). The codebook's significant learning happens during dedicated offline passes.

**Metrics (Phase 3 headline vs. drill-downs, per 2026-05-09 discipline):**
- **Headline:** Recall@K on masked-token contextual completion, stratified by tight/spread regime, vs. shuffled-token control.
  - The shuffled control rules out corpus-statistical artefacts.
  - Tight-regime success vs. spread-regime success are different stories; stratification distinguishes them.
- **Drill-downs (Tier 1 sanity):**
  - No collapse: NC1 (within-basin variability) stays bounded and non-zero; does not trend toward zero.
  - No explosion: codebook size stays within budget.
  - Meta-stable-state rate stays low (fraction of retrievals with max-pattern cosine < 0.95).
  - Cap-coverage error tracked at θ ∈ {0.3, 0.5, 0.7}. Distinguishes "wrong pattern retrieved" from "no pattern reached."
- **Drill-down (Tier 2 distributional):** Semantic relationships (synonyms, collocations) show increased similarity over training; unrelated pairs stable.

**Atoms' properties at end of Phase 3:**
- Codebook is ~5000 atoms (vocabulary-sized).
- Atoms are clustered into tight and spread regimes; both are stable and retrievable.
- High-frequency atoms are predominantly tight-regime (contexts overlap, centroid-drift works).
- Low-frequency atoms show more bimodal context distributions (polysemy candidates for Phase 5 splitting).
- Atoms have metadata: usage count, last-used step, stability score, utility score, bimodality flag.

**The anti-homunculus filter (Project-level commitment from 2026-05-09):**
- Every metric, diagnostic, or actuator added to any phase must be expressible as an *intrinsic geometric state* — not as an arbitration over states.
- A metric that requires a supervisor to interpret it (e.g., "if retrieval quality drops, increase repulsion strength") does the work of a controller at evaluation time, not at architecture time.
- This filter is load-bearing for Phase 5 design.

---

## Phase 4: Consolidation + Replay

**Built:**
- **Trajectory trace:** Hopfield settling is wrapped with a recorder that captures per-step co-activation snapshots (top-K pattern indices and weights, entropy at each step).
- **Engagement and resolution metrics:**
  - **Resolution** = max cosine similarity of settled state to any stored pattern. High = clean basin lock-in. Low = metastable or feature mode.
  - **Engagement** = mean entropy across settling trajectory. High = many patterns co-active = strong constraint from landscape. Low = single basin dominated from the start.
- **Gate signal** = engagement × (1 − resolution). Trajectories with gate > threshold are "unresolved but engaged" — candidates for replay.
- **Replay store:** Bounded buffer of traces. Evicts lowest-gate entries when full. Traces decay if they don't resolve after re-settling through the current landscape.
- **Benna-Fusi consolidation cascade:** Each stored pattern has u_1, ..., u_m variables. New candidates enter at u_1 (fast, weak). Replay activity drives bidirectional coupling u_k ↔ u_{k+1}. Patterns that consistently re-emerge propagate to slow variables (u_m) and become durable. Patterns that stop being retrieved decay below threshold and are garbage-collected (binary death).
- **Re-encoding:** After codebook drift (from Hebbian reinforcement or batch retraining), stored patterns are re-settled through the current landscape and their vectors re-aligned.

**Headline metric for Phase 4 (per 2026-05-16 discipline note):**
- **D1: meta-stable-state rate (fraction of retrievals with max-pattern cosine < 0.95).** Decreased D1 means fewer meta-stable states; basins are more separable.
- **Verified at integration regime (n=10, Phase 3+4 online):** ΔD1 (with replay vs. no replay) = −0.79, CI [−0.94, −0.65], 10/10 seeds negative. CI-disjoint from zero. **Phase 4 graduates on D1.**

**Drill-downs (per substrate-vs-readout discipline):**
- **R@K (substrate readout):** ΔR@10 = +0.0089, CI [−0.002, +0.020]. Small positive signal, present across all reports but with CI hair-thin or crossing zero at small-N runs. Replicated at integration regime.
- **Cap-coverage (substrate readout):** ΔCap_t05 = −0.052, CI [−0.106, +0.002]. Variance-bound; depends on binary-death survival outcomes rather than pure replay effect. Not a graduation criterion per 2026-05-16.
- **Top1 (readout, not substrate):** ΔTop1 = −0.048, CI [−0.083, −0.013]. Regression is real but is a *drill-down*, not a headline. It tells us the substrate is being reshaped in a way that breaks naive argmax, but Phase 4 is not responsible for building Phase 6/7's readout. Per the anti-homunculus discipline: don't measure a subsystem against outputs it isn't architecturally responsible for producing.

**Pattern death mechanism (binary form that graduated Phase 4 but breaks Phase 5):**
- When effective_strength(pattern) < threshold for `death_window` consecutive steps, the pattern is deleted.
- This is a step function on a local metric. Shape diagnosis: a controller reads effective_strength, checks the threshold, and deletes. The action is discrete, not continuous.
- **Anti-homunculus filter: FAILS.** The mechanism graduated Phase 4 (D1 works) because D1 only cares about basin separability, not substrate capacity. At Phase 5, where K-branch settling needs diverse schemas, binary death **collapses substrate effective dimensionality from d_eff ≈ 40–45 to d_eff ≈ 2.5–6.4** in a single event. With K=4 branches and d_eff ≈ 5, the architecture's "distinct schemas produce distinct settled states" claim fails geometrically.
- **Consequence:** Phase 5 MUST replace binary death with a continuous local dynamic that preserves effective dimensionality. Three candidates specified in 2026-05-20 note: (A) coverage-weighted reinforcement rate; (B) dimensionality-preserving repulsion field; (C) per-atom inhibition with redundancy-coupled decay.

**Substrate effective dimensionality after mass-death (from report 044):**
- Pre-death: d_eff ≈ 40–45 (out of 4096).
- Post-death: d_eff ≈ 2.5–6.4 (n=5 seeds × pre/post).
- Post-death atom count: ~5–30 atoms per scale (W=2 has fewest; W=3, W=4 have more).

**Tight coupling between Phase 3 and Phase 4 (proven at integration):**
- Online Hebbian (Phase 3) + replay/consolidation (Phase 4) run concurrently.
- Hebbian fired at meaningful rate: consolidations 52–571 per seed (report 038).
- Phase 4 discovered candidates: 63–108 per seed.
- The two mechanisms' interplay (Hebbian shaping the codebook, replay discovering stable patterns) produces the D1 effect. Cannot separate them cleanly.

---

## Architectural Carry-Forward into Phase 5

**Phase 5 inherits and cannot change without re-opening Phases 2–4:**

1. **Substrate intrinsic capacity:** d_eff ~40–45 before death, ~2.5–6.4 after. Phase 5 cannot ask for higher capacity without redesigning the consolidation cascade or codebook structure.

2. **Post-death substrate is the schema source:** The ~5–30 surviving atoms per scale after Phase 4 mass-death form the Prior bank for Phase 5 branching. This is not a design choice; it is the only candidate population (phase 2–4 generate no other "higher-level" patterns).

3. **Online Hebbian is the only sanctioned online learning rule.** Error-driven is out; it collapses retrieval. This is empirically final (reports 028–039).

4. **Metric discipline:** Phase 5 is responsible for producing *richer attractors* (through binding, splitting, hierarchical structure), not for picking literal words. Top1 accuracy is Phase 6/7's problem. Phase 5's headline should be substrate-native (e.g., K-branch settling produces K distinct states with comparable energies; atom-splitting diagnostic fires on persistent bimodal branches).

5. **Anti-homunculus filter is non-negotiable.** Every Phase 5 mechanism must pass the test: does it distribute into local geometry, or does it require a supervisor to read a metric and decide? Branching, combining, and splitting must all be geometric dynamics.

6. **No dynamic mode switching.** The pure-Python reference backend is the ground truth. If a mechanism requires a mode controller ("in mode X use rule A, in mode Y use rule B"), it fails the filter.

---

## The Anti-Homunculus Filter in Concrete Terms

### Mechanisms that PASSED:

1. **Trajectory trace + engagement × (1 − resolution) gate (Phase 4):**
   - Traces are passive observations of every settling path.
   - Gate signal is a geometric property of the trace (entropy and basin sharpness).
   - Replay store threshold is a fixed number, not a rule read by a supervisor.
   - Replaying happens on a fixed cadence (every K cues) or when the store fills.
   - **Passes:** All decisions are properties of the settling landscape; no supervisor reads them.

2. **Benna-Fusi u-variable consolidation (Phase 4):**
   - Each pattern has m variables; they evolve under a coupled ODE (Eq. 10–11 in design spec).
   - New patterns enter with strong u_1 perturbation.
   - Retrieval success reinforces u_1 mildly (mild additive term).
   - u-variables couple bidirectionally; slow variables accumulate over repeated replay.
   - No supervisor reads "this pattern is good, keep it" or "this pattern is bad, kill it."
   - **Passes:** The dynamics themselves determine consolidation depth; the response is integral to the dynamic, not triggered by it.

3. **Regime classification for Hebbian vs. error-driven pathway (Phase 3):**
   - Tight-regime atoms get Hebbian-only updates (nearest-optimal per Geometry of Consolidation paper).
   - Spread-regime atoms get both pathways, error-driven providing correction.
   - The regime boundary is computed per atom from the atom's own context-bag geometry.
   - No supervisor reads "this atom is in spread regime" and *decides to apply error-driven*.
   - **Passes:** The regime is a property of the atom's neighborhood; the pathway is applied because the problem structure changes, not because a rule fires.

### Mechanisms that FAILED the filter:

1. **Binary death (Phase 4, breaks Phase 5):**
   - A controller reads `effective_strength(pattern)` for each pattern.
   - A threshold check turns the metric into a binary signal.
   - A discrete `delete` operation is triggered.
   - The controller sees the population state (all effective_strength values) and makes a population-level decision (delete the weakest).
   - **Fails:** The deletion is an arbitration over the population, not a continuous evolution of local geometry.

2. **Hypothetical "if NC1 starts dropping, increase repulsion strength" rule:**
   - A metric (NC1 trend) is read by a supervisor.
   - The supervisor decides whether to increase a parameter.
   - Repulsion then fires at a different magnitude than before.
   - **Fails:** The decision is arbitration, not geometry. The mechanism's response to a condition is rule-based, not intrinsic.

3. **Hypothetical "use mode A if meta-stable-state rate > 0.2, else mode B" routing:**
   - A metric (meta-stable-state rate) is read by a supervisor.
   - The supervisor routes to one of two modes.
   - Different dynamics run in different modes.
   - **Fails:** This is a explicit supervisor making a mode decision. Even if the modes are themselves geometric, the router is a controller.

---

## Headline vs. Drill-Down Discipline

**Principle (from 2026-05-09 and 2026-05-16):**
- One headline metric per phase defines whether the phase crossed its viability threshold.
- Drill-downs explain why the headline moved; they do NOT replace it.
- When a metric surprises (e.g., Δtop1 regression), investigate the mechanism geometrically. Do NOT promote it to headline status.

**Phase 2 headline:** Recall@K with static random codebooks beats bigram baselines.
**Phase 3 headline:** Recall@K on masked-token contextual completion, regime-stratified, vs. shuffled-token control shows improvement.
**Phase 4 headline:** Δ meta-stable-state rate (D1) under active drift, CI-disjoint from zero, n=10, Phase 3+4 integration.

**Why top1 is not Phase 4's headline:**
- Phase 4 produces a *settled attractor* in the latent space.
- Phase 6/7 decodes that attractor to a word.
- Literal-word accuracy depends on both Phase 4 (substrate quality) AND Phase 6/7 (readout design).
- Phase 4 is measured on substrate quality (D1: are basins separable?), not on readout accuracy.
- Top1 regression (ΔTop1 = −0.048) is real but is a drill-down: it tells us the substrate is being reshaped, but not whether the reshape is *wrong*. The question "is it a problem?" is Phase 6/7's to answer.

**Applying this to Phase 5:**
- Phase 5's headline should be substrate-native: e.g., "K-branch settling produces K distinct states with comparable low energies" or "atom-splitting diagnostic fires predictably on persistent bimodal branches."
- Readout metrics (e.g., "does binding help top1 accuracy?") are drill-downs, not headlines.
- If K-branch settling fails to produce K distinct states, Phase 5 has failed. Investigate the geometry, not the readout.

---

## Already-Falsified Mechanisms (That Are Out of Scope)

| Mechanism | Report(s) | Outcome | Lesson |
|-----------|-----------|---------|--------|
| Online error-driven contrastive updates | 028, 029, 030, 039 | Top-1 accuracy collapses by ~3000 cues. | Error-driven is out of the streaming loop. Batch retraining only. |
| A_k (Saighi-style per-attractor inhibition with time-decay) | 034, 035 | Top1 +0.009 on seed 1 (promising) but Δtop1 −0.051 (CI strictly negative) at n=10. ΔR@10 +0.009 (same as baseline). | Not producing substrate benefit beyond passive replay. Redundancy-coupled variant not yet tested; still risky. |
| Freq-weighted α (codebook budget squeezing via frequency-weighted decay) | 040 (040_freq_weighted_alpha_sweep.md) | No improvement in substrate capacity or D1. Binary death already filters to sub-capacity regime. | Redundant mechanism; mass death does the filtering. |
| K-branch settling without continuous death replacement | 042, 043, 044 | Substrate d_eff collapses from ~40 to ~2.5–6.4 in mass-death event. K=4 branches produce K=1 distinct settled state (geometric degenerate). | Binary death breaks K-branching. MUST replace with continuous local dynamic before Phase 5 branches. |

---

## The Death Mechanism Impasse and Phase 5's Structural Constraint

**The problem:** Binary death graduated Phase 4 (D1 = basin separability) but breaks Phase 5 (K-branch settling needs diverse schemas).

**The impasse:** Continuous local dynamics that preserve d_eff all require computing per-atom redundancy or per-dimension repulsion at each step. Three candidates exist (2026-05-20 note):

| Candidate | Pros | Cons |
|-----------|------|------|
| **A: Coverage-weighted reinforcement rate** | Highly local; local per-atom gradient; death is the limit of zero reinforcement, not a discrete action. | O(N³) projection cost per update. Needs low-rank approximation. |
| **B: Dimensionality-preserving repulsion field** | Clean energy formulation; d_eff is exactly the quantity being conserved; gradient is per-atom and continuous. | Does NOT kill atoms; needs a separate under-capacity mechanism if under-capacity is necessary. |
| **C: Redundancy-coupled inhibition** | Variant of the Saighi A_k that couples decay to redundancy (high-r_i atoms suppress easily; low-r_i atoms persist). Similar redundancy-cost as A. | A_k was falsified on top1 at n=10; high bar to use variants without strong upfront commitment about what would falsify this one. |

**Phase 5's dependency:** One of these MUST be chosen and proven before Phase 5 branches. Branching cannot work with binary death. This is not optional.

**Implementation precondition (per anti-homunculus audit 2026-05-20):**
- For A: the per-atom redundancy estimate MUST be a continuous local dynamic, not a periodic global recompute broadcast to atoms. O(N³) is infeasible; a streaming SVD or EMA of projection contributions is required.
- For B: H_anti is part of the substrate energy at EVERY call site, not a consolidation-time-only term. α is fixed at training start; NOT adapted from observed d_eff trajectories.
- For C: needs a test falsification criterion decided upfront, not hindsight after n=10 repeats the A_k story.

---

## Summary Table: What Phases 2–4 Fixed and What's Deferred

| Component | Phase | Status | Fixed | Deferred |
|-----------|-------|--------|-------|----------|
| FHRR substrate (D=4096) | 2 | Proven | Yes | — |
| Hopfield retrieval (softmax energy) | 2 | Proven | Yes | Multi-scale coupling (per-scale today) |
| Static codebook capacity | 2 | Proven | Sufficient | — |
| Atom entry (lexical) | 3 | Proven | Yes | Pattern-based or error-driven allocation |
| Hebbian refinement (success-driven) | 3 | Proven | Yes | — |
| Error-driven consolidation (batch-only) | 3 | Proven | Online is OUT | Batch retraining works; use it |
| Bimodality tracking | 3 | Tracking only | Yes | Splitting deferred to Phase 5 |
| Trajectory trace + engagement gate | 4 | Proven | Yes | — |
| Benna-Fusi u-cascade | 4 | Proven | Yes | Adaptive m (start m=6, fixed) |
| Re-encoding | 4 | Proven | Yes | — |
| Binary death mechanism | 4 | Proven for D1 | Phase 4 only | MUST replace for Phase 5 |
| Metric discipline (substrate vs. readout) | 4/5 | Discipline set | Yes | — |
| Post-death substrate as schema source | 5 | Design choice | Yes (immovable given Phases 2–4) | — |
| Schema-prior branching | 5 | Specified | Design locked | — |
| K-branch combiner + splitting diagnostic | 5 | Design specified | Will be built | — |

---

## Key Non-Negotiable Constraints for Phase 5

1. **Substrate is fixed.** d_eff ~40–45 pre-death, ~2.5–6.4 post-death. Cannot increase without redesigning consolidation (Phase 4 re-opened).

2. **No supervisor.** All branching, combining, and splitting must distribute into local geometry.

3. **Death mechanism must be continuous and local.** Binary death is gone for Phase 5. Choose one of the three candidates and prove it preserves d_eff.

4. **Schemas come from post-death substrate.** These are the only "higher-level" patterns the architecture generates. Phase 5 cannot invent other schema sources without reimporting the controller pattern (who decides what counts as a schema?).

5. **Online learning is Hebbian only.** Error-driven is batch-only. This is empirical and final.

6. **Headline metric is substrate-native.** K-branch settling producing K distinct states with comparable energies. Splitting diagnostic firing on persistent bimodal branches. Literal-word readout is Phase 6/7's problem.

7. **The anti-homunculus filter applies unchanged.** Every new mechanism is geometry or measurement, never arbitration.

---

## References

- **PROJECT_PLAN.md:** Phase decomposition, non-negotiable rules.
- **phase-4-unified-design.md:** Trajectory, consolidation, replay, and anti-homunculus checks.
- **phase-4-checklist.md:** Component status and verification evidence.
- **phase-3-deep-dive.md:** Codebook dynamics, failure modes, and Tier 1–3 evaluation structure.
- **consolidation-geometry-diagnostic.md:** Regime classification (tight vs. spread).
- **2026-05-09-papers-diagnostics-and-actuator-dynamics.md:** Anti-homunculus filter, diagnostic stack, headline-vs-drill-down discipline.
- **2026-05-16-substrate-vs-readout-metric-discipline.md:** Readout is not Phase 4's responsibility.
- **2026-05-20-diagnostic-actuator-death-dynamic-form.md:** Three candidates for continuous death replacement.
- **038_phase4_d1_graduation.md:** D1 verified at integration (n=10).
- **044_consolidation_geometry_diagnostic.md:** d_eff collapse evidence (pre/post-death).
- **Reports 028–039:** Online error-driven contrastive update falsification chain.
- **Report 040:** Freq-weighted α redundancy.
- **Reports 042–043:** K-branch settling failure under binary death.

