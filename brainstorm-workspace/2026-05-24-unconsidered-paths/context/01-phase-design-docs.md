---
date: 2026-05-24
title: Phase Design Documents — Extracted Metrics, Criteria, Built vs Untested
project: neuro-ai
---

# Phase Design Documents: Built vs. Designed-But-Untested

A comprehensive extraction of headline metrics, exit criteria, and mechanism status across phases 0-5. This document is structured by phase with explicit sections for "built/tested" vs "designed-but-untested" — the latter is the highest-leverage section for downstream work.

**Key observation from STATUS.md (2026-05-24):** Phase 5 is not graduated; active blockers are listed as Path D split and architectural commit decision pending. Four retrieval-mechanism families have returned null. This document captures what was *designed* vs what was *actually built and validated*.

---

## Phase 0: Pre-Phase Commitments & Prerequisites

### Headline Metric
None (this is pre-phase). The commitment is to resolve open architectural decisions before Phase 1 begins.

### Exit/Graduation Criteria
Not gating Phase 1. But documented as "pre-phase commitments still open" in STATUS.md:

1. **Consolidation-geometry regime classifier** — compute $\bar{d}$ (mean within-cluster cosine distance) and $d_{eff}$ (effective dimension of covariance spectrum) per atom. Classify atoms as "tight regime" ($\bar{d} < \theta'$) vs "spread regime" ($\bar{d} \geq \theta'$).
   - **File:** consolidation-geometry-diagnostic.md
   - **Status:** DESIGNED, NOT BUILT
   - **Why not built:** documented as "not blocking" in consolidation-geometry-diagnostic.md:169. "Not built" is different from "not needed" — it's listed in STATUS.md as a pre-phase commitment still open.

2. **Empirical θ′(β) calibration spike** — Recommended pre-Phase-3 (2026-05-09, from consolidation-geometry-diagnostic.md). Run the Geometry of Consolidation E1 protocol on FHRR substrate at β ∈ {0.01, 0.1, 1.0}, generate synthetic clusters, store in Hopfield, measure empirical retrieval-success boundary, back out calibrated θ'(β) mapping.
   - **Status:** DESIGNED, NOT DONE
   - **Estimated effort:** ~2 days
   - **Why deferred:** "should land before Phase 3 starts" but didn't

3. **High-leverage brainstorm idea 5** — frequency-weighted Benna-Fusi α as the "compression → abstraction" mechanism. Explicitly named as "the key experiment for the architecture's compression → abstraction claim."
   - **File:** brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md
   - **Status:** NEVER BUILT
   - **Citation:** STATUS.md "Pre-phase commitments still open"

### Mechanisms Built/Tested
- Nothing; this is pre-phase.

### Mechanisms Designed But Not Built/Tested
- Consolidation-geometry regime classifier (full implementation with per-atom tracking)
- θ'(β) calibration protocol
- Frequency-weighted α dynamics

---

## Phase 1: Substrate Validation

### Headline Metric
**"Clean recovery of bound structures, no surprises in the algebra at 4096 dimensions."** (overview.md + experimental-progression.md:33-41)

Concretely: bind two random vectors, unbind, recover the original within tolerance. Encode a structured pattern (small graph, tree, labeled record) as bundled bindings, retrieve specific roles by unbinding, confirm role-filler recovery.

**File:** experimental-progression.md:33-41
**Status:** INFERRED (not explicitly named in phase-1-specific doc; Phase 1 has no dedicated design doc)

### Exit/Graduation Criteria
- Clean recovery of bound structures
- No surprises in algebra at 4096 dimensions
- Performance characteristics fit compute budget

**File:** experimental-progression.md:37

### Mechanisms Built/Tested
Per mvp-architecture.md:

- **FHRR substrate** (4096 complex dimensions)
  - Binding: element-wise complex multiplication ✓
  - Unbinding: multiplication by complex conjugate ✓
  - **Bundling: per-dimension normalized element-wise sum** — note: torchhd's default is broken; requires wrapper (empirically validated in 2026-05-05 substrate-validation spike) ✓
  - Similarity: cosine ✓
  - **Empirical baselines from spike** (mvp-architecture.md:36-41):
    - Pairwise orthogonality: mean |sim| = 0.0088, std = 0.011 ✓
    - Round-trip fidelity: exactly 1.0 across 1000 trials ✓
    - Bundling capacity: 100% recovery at K=50 with 11× noise floor margin ✓
    - MPS speedup: 1.6× on bind, 3.6× on cosine_matrix, 4.3× on bundle ✓

- **Input layer** — BPE tokenizer (GPT-2 baseline) providing permanent input segmentation ✓

- **Codebook data structure**
  - token_id → hypervector + metadata ✓
  - Metadata: usage_count, last_used_step, stability_score, utility_score, context_bag_history ✓

- **Sequence encoding**
  - Bundled position-token bindings: `bundle(bind(pos_i, token_i))` ✓

- **Hopfield layer** (Modern Hopfield with log-sum-exp energy) ✓

### Mechanisms Designed But Not Built/Tested
- None. Phase 1 is substrate math — everything specified has been built and tested.

---

## Phase 2: Static Codebook Baseline (Dual Objective)

### Headline Metric
**"Retrieval accuracy on test population (masked-token Recall@1 and next-token Recall@1, separately) compared against bigram baseline with non-overlapping 95% confidence intervals across majority of experimental matrix conditions."** (experimental-progression.md:58)

**File:** experimental-progression.md:58
**Line:** 58

Drill-down metrics added 2026-05-09:
- Cap-coverage error at θ ∈ {0.3, 0.5, 0.7}
- Meta-stable-state rate at θ = 0.95
- Both derived from per-retrieval max stored-pattern cosine at convergence

**Files:** experimental-progression.md:60, phase-3-deep-dive.md (NC1 reformulation)

### Exit/Graduation Criteria
- Retrieval quality (Recall@K) meaningfully above random for at least one objective
- Both objectives > chance (if both at chance, there's a bug)

**File:** experimental-progression.md:62

### Mechanisms Built/Tested
- Static random codebook ✓
- Hopfield on sequences with fixed codebook ✓
- Next-token prediction objective ✓
- Masked-token prediction objective ✓
- Per-retrieval max stored-pattern cosine measurement ✓
- Cap-coverage error at multiple thresholds ✓
- Meta-stable-state rate computation ✓

### Mechanisms Designed But Not Built/Tested
- None for Phase 2 core; the comparison objective is the learning experiment itself.

---

## Phase 3: Codebook Growth

### Headline Metric
**"Recall@K on masked-token contextual completion, stratified by regime classification (from consolidation-geometry-diagnostic.md), evaluated against the shuffled-token control."** (phase-3-deep-dive.md:180-189)

**File:** phase-3-deep-dive.md
**Lines:** 180-189

One number, one stratification axis, one controlled comparison.

### Exit/Graduation Criteria
- Codebook stabilizes within finite training budget
- Headline metric improves meaningfully over Phase 2 baseline
- Distributional structure appears in codebook geometry (similar tokens → similar hypervectors)
- Shuffled-token control fails to produce same semantic organization

**File:** experimental-progression.md:74

### Mechanisms Built/Tested
- **Two-pathway hybrid update rule** — Hebbian on success, error-driven on failure, gated per-experience by retrieval quality (phase-3-deep-dive.md:47-55) ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:47-55
  - **Status:** Per STATUS.md 2026-05-24, Phase 3 is implemented. The per-phase notes call out "updated 2026-05-09" with new diagnostics.

- **Context bag** (clean bundle of co-occurring atoms, no position bindings) ✓ IMPLEMENTED
  - **Citation:** overview.md:40

- **Error-driven update** — atoms in failed cues drift toward corresponding atoms in correct retrieval targets ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:111-127

- **Bimodality tracking** — persistent bimodal distribution of context bags signals polysemy (Phase 3 hook for Phase 5 splitting)
  - **Citation:** phase-3-deep-dive.md:147-171
  - **Status:** IMPLEMENTED (diagnostic-only; splitting deferred to Phase 5)

- **Atom allocation** — lexical (MVP recommendation) ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:35-45

- **Atom stability tracking** — moving variance over recent consolidation events ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:57-64

- **Atom decay** — time-based + quality-based gating ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:66-70

- **Repulsion term** — prevents collapse by pushing nearest neighbors apart ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:72-78

- **Soft codebook cap** — weakest-decay when budget exceeded ✓ IMPLEMENTED
  - **Citation:** phase-3-deep-dive.md:79

- **Tier 1 sanity checks** — codebook drift stabilization, atom-level drift rate, NC1 within-basin variability, no explosion, meta-stable-state rate, cap-coverage error
  - **Citation:** phase-3-deep-dive.md:191-209
  - **Status:** MOSTLY IMPLEMENTED
  - **Caveat:** NC1 reformulation done 2026-05-09 to fix supervised-case transfer problem. New formulation: maintain bounded non-zero within-basin variability while preserving inter-basin separability (joint metric: NC1 + separability).

- **Tier 2 distributional structure** — synonym/antonym/collocation similarity changes over training, shuffled-token control comparison
  - **Citation:** phase-3-deep-dive.md:211-217
  - **Status:** IMPLEMENTED

- **Tier 3 retrieval-shaped analogical structure** — partial structural pattern → completions (lower bar for Phase 3)
  - **Citation:** phase-3-deep-dive.md:219-221
  - **Status:** IMPLEMENTED

- **Per-retrieval interpretive layer** (2026-05-09 additions):
  - Softmax entropy as feature/prototype mode classifier
  - Regime classifier distribution (tight vs spread)
  - Bimodality flag rate
  - **Citation:** phase-3-deep-dive.md:223-229

### Mechanisms Designed But Not Built/Tested
- **Pattern-based allocation** — "when pattern of co-occurrence exceeds novelty threshold" (middle complexity gradient, not MVP)
  - **Citation:** phase-3-deep-dive.md:40-41
  - **Status:** DESIGNED, NOT TESTED
  - **Why:** MVP uses lexical allocation; pattern-based is the next step if lexical is too coarse

- **Error-driven allocation** — "when retrieval failures reveal current atoms can't explain a pattern" (most thesis-aligned)
  - **Citation:** phase-3-deep-dive.md:42-43
  - **Status:** DESIGNED, NOT TESTED
  - **Why:** Most sophisticated; deferred until lexical + pattern-based are validated

- **Consolidation-geometry regime classifier as a *gating mechanism*** — only fire error-driven updates on spread-regime atoms, skip on tight-regime
  - **Citation:** consolidation-geometry-diagnostic.md:84-92
  - **Status:** PARTIALLY IMPLEMENTED
  - **Details:** The regime classifier itself is designed but not built (see Phase 0). The *use case* of gating error-driven by regime is designed but untested because the classifier doesn't exist.

- **Online error-driven contrastive updates** — EXPLICITLY BANNED after empirical failure (reports phase34_stable_v2, phase34_hebbian)
  - **Citation:** phase-3-deep-dive.md:245 (failure mode), phase-4-unified-design.md:296-309 (runtime vs training asymmetry)
  - **Status:** TESTED, FOUND BROKEN
  - **Details:** "Online error-driven contrastive updates collapse retrieval" — top-1 degrades from 0.107 → 0.020 by ~3500 cues. Runtimes use Hebbian only; error-driven belongs in batch passes.

---

## Phase 4: Hierarchical Compression (Replay + Benna-Fusi Consolidation)

### Headline Metric
**"Δ meta_stable_w3 (meta-stable rate at W=3) under active drift at Phase 3+4 integration regime, n=10 seeds, CI disjoint from 0."** (phase-4-checklist.md:38, upgraded from R@K and cap-coverage per 2026-05-16 substrate-vs-readout discipline note)

**Previous headline:** "Recall@K and cap-coverage on masked-token contextual completion, measured before vs. after N consolidation cycles, with active codebook drift between cycles." (phase-4-unified-design.md:276-288)

**File:** phase-4-checklist.md
**Line:** 38

**Status:** ✅ VERIFIED
- **Evidence:** [Report 038](../../reports/038_phase4_d1_graduation.md)
- **Result:** Δ = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds at floor
- **Citation:** phase-4-checklist.md:38 (B0 row)

### Exit/Graduation Criteria
Per project plan:
- E1: Replayed trajectories improve future retrieval ✓ VERIFIED
- E2: Stale atom drift can be corrected without global rewrites (reencode_every=100 implemented but contribution not ablated) 🟨 PARTIAL
- E3: Repeated solution paths become faster and lower entropy ❌ NOT PASSED (entropy flat at W=2; rises above baseline by step 2000 at W=4)

**File:** phase-4-checklist.md:84-91

### Mechanisms Built/Tested

- **Trajectory trace** (TrajectorySnapshot, TrajectoryTrace, TracedHopfieldMemory wrapper)
  - **Citation:** phase-4-unified-design.md:84-102, phase-4-checklist.md:A1
  - **Status:** ✅ VERIFIED
  - **Evidence:** src/energy_memory/phase4/trajectory.py exists

- **Engagement metric** (entropy-weighted across settling steps)
  - **Citation:** phase-4-unified-design.md:109-115
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 023 gate audit

- **Resolution metric** (max cosine of settled state to any stored pattern)
  - **Citation:** phase-4-unified-design.md:106-108
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 023

- **Gate signal = engagement × (1 − resolution)**
  - **Citation:** phase-4-unified-design.md:123-133
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 023 (gate fires 59% at threshold 0.05)

- **Replay store** (bounded buffer with gate-ranked eviction)
  - **Citation:** phase-4-unified-design.md:136-142
  - **Status:** ✅ VERIFIED
  - **Evidence:** src/energy_memory/phase4/replay_loop.py

- **Replay re-settle through current landscape**
  - **Citation:** phase-4-unified-design.md:140-152
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 026

- **Candidate emission (resolve_threshold)**
  - **Citation:** phase-4-unified-design.md:145
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 024 (threshold sweep)

- **Trace age + decay**
  - **Citation:** phase-4-unified-design.md:150-152
  - **Status:** 🟨 PARTIAL
  - **Details:** Mechanism present; age distribution never aggregated
  - **Citation:** phase-4-checklist.md:A8

- **Benna-Fusi u-chain (Eq. 10/11)**
  - **Citation:** phase-4-unified-design.md:159-170
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 026 §u_k drill-down

- **u_1 novelty input + retrieval reinforcement**
  - **Citation:** phase-4-unified-design.md:172-175
  - **Status:** ✅ VERIFIED
  - **Evidence:** Report 026

- **Effective strength weighted sum**
  - **Citation:** phase-4-unified-design.md:182-189
  - **Status:** ✅ VERIFIED
  - **Evidence:** Mean strength reported per checkpoint

- **Pattern death (strength < threshold for window)**
  - **Citation:** phase-4-unified-design.md:191-193
  - **Status:** ⚠️ BROKEN/REFRAMED
  - **Details:** Mechanism FEP-clean; BUT cannot fire in n_cues=1500 at canonical config. >99.9% of vocab atoms are never reinforced. Naive tuning produces mass-death of unreached initial atoms, not architecturally-intended stale-discovered-pattern purge.
  - **Citation:** phase-4-checklist.md:A12
  - **Resolution:** STATUS blocker #2 reshaped scope decision.

- **SQ-HN sparse-update principle**
  - **Citation:** phase-4-unified-design.md:195-202
  - **Status:** 🟨 PARTIAL
  - **Details:** Code respects sparsity; no ablation showing it matters
  - **Citation:** phase-4-checklist.md:A13

- **Re-encode stored patterns through current codebook**
  - **Citation:** phase-4-unified-design.md:259, phase-4-checklist.md:A14
  - **Status:** 🟨 PARTIAL
  - **Details:** reencode_every=100 knob enabled; contribution never ablated

### Mechanisms Designed But Not Built/Tested

- **Cross-pattern coupling** — when pattern A is consolidated, do related patterns' u variables also update?
  - **Citation:** phase-4-unified-design.md:324-326
  - **Status:** EXPLICITLY DEFERRED
  - **Details:** "Defer to Phase 5 or later"

- **Adaptive m (chain length)** — Benna-Fusi proves m ≈ log(T). Start with fixed m=6.
  - **Citation:** phase-4-unified-design.md:327-329
  - **Status:** EXPLICITLY DEFERRED

- **Adaptive store_threshold**
  - **Citation:** phase-4-unified-design.md:330-331
  - **Status:** EXPLICITLY DEFERRED

- **Multi-scale interaction** — do three scales (W=2,3,4) share trajectory traces?
  - **Citation:** phase-4-unified-design.md:332-334
  - **Status:** EXPLICITLY DEFERRED
  - **Details:** "Start with per-scale replay and consolidation"

- **Sleep/wake cycles** — when replay runs vs interleaved with retrieval
  - **Citation:** phase-4-unified-design.md:335
  - **Status:** EXPLICITLY DEFERRED

- **Pattern death mechanism refinement** — the existing mechanism cannot actually purge stale discovered patterns in realistic regimes
  - **Citation:** phase-4-checklist.md:A12, STATUS blocker #2
  - **Status:** DESIGNED FOR PHASE 5+
  - **Details:** The A+B note (2026-05-20) specifies a replacement: continuous coverage-weighted reinforcement (A) + −α log(d_eff) repulsion in substrate energy (B) replacing binary death. Pre-committed falsification criteria documented.
  - **Citation:** phase-5-checklist.md:J (last line)

---

## Phase 5: Binding Discovery and Atom Splitting (HAM × Energy-Guided Structural Branching)

### Headline Metric
**"Δ final-state energy E_A − E_B (content-prior minus role-prior) > 0 with 95% CI disjoint from zero, on held-out cue set designed for structural retrieval, n_seeds ≥ 10, AND mean ΔE ≥ 5.5e-3 per the magnitude-floor pre-commit."** (phase-5-unified-design.md:281-299, phase-5-checklist.md:A1)

**File:** phase-5-unified-design.md and phase-5-checklist.md
**Lines:** phase-5-unified-design.md:281-299, phase-5-checklist.md:A1:39

**Status:** ⚠️ GRADUATION-UNATTAINED — Directional but sub-noise
- **Evidence:** [Report 053](../../reports/053_phase5_headline_n10_directional_subnoise.md)
- **Result:** n=10 on A+B+A1' substrate: ΔE = +0.00130, CI [+0.00071, +0.00193]; 10/10 seeds positive in mean. CI-half PASSES (lower > 0); magnitude-half FAILS (4.2× below floor).
- **Controls:** γ=0 = 0.000000 exact; random-schema sits between content and role.
- **Citation:** phase-5-checklist.md:A1:40

### Exit/Graduation Criteria (Full Matrix)

Per phase-5-checklist.md, Phase 5 graduates cleanly only if ALL of the following hold:

1. **A1** passes — ΔE CI-disjoint from zero, n ≥ 10. ❌ FAILED (sub-noise magnitude)
2. **All of B1–B4** behave as predicted (controls remove or shrink the effect).
   - B1 Random-schema branches ✅ VERIFIED
   - B2 K=1 single-branch ⚠️ FIRES AT N=5 (branching is gratuitous/destructive; K1 ΔE ≥ K4)
   - B3 No-prior (γ=0) ✅ VERIFIED (exact zero across all seeds × cues)
   - B4 No-schema-store ❌ NOT YET RUN
3. **C1 AND at least one of {C2, C4}** — effect survives ≥1 schema source not depending on late-run death
   - C1 Post-death (design default) 🟨 PARTIAL (n=5; doesn't pass)
   - C2 Pre-death top-k ❌ NOT YET RUN
   - C3 Pre-death random-k ❌ NOT YET RUN
   - C4 Step-1500 top-k ❌ NOT YET RUN
   - C5 Step-1500 random-k ❌ NOT YET RUN
4. **D1 + D2** — n ≥ 10 with LOSO CI excludes zero for every leave-one-out subset.
   - D1 N ≥ 10 ✅ ATTAINED
   - D2 LOSO CI sensitivity 🟨 REPORTED AT N=5 (no leave-one-out subset excludes 0)
5. **G1** — W=3 meta-stable rate does not regress under role-prior branching. ❌ NOT YET CHECKED

**File:** phase-5-checklist.md:I (Interpretation rule)

### Mechanisms Built/Tested

- **Schema selector** (top-K schemas by similarity + diversity constraint + surprise branch)
  - **Citation:** phase-5-unified-design.md:126-141
  - **Status:** 🟨 IMPLEMENTED AND PARTIALLY TESTED
  - **Details:** Schema store resolved as post-death substrate (report 040). Top-K selection works; surprise branch selection works (log-ratio novelty score defined).
  - **Caveat:** Post-death substrate is fragile (5–30 atoms per seed at W=2 per report 037)

- **Per-branch settling with prior bias (HAM layer 1)**
  - **Citation:** phase-5-unified-design.md:152-168
  - **Status:** 🟨 IMPLEMENTED
  - **Details:** Energy formulation has two variants: per-pattern (CHOSEN per decision 5 spike result) vs global-pull. Per-pattern selected after 2026-05-20 decision spike (report phase5_decision5_local).
  - **Decision 5 result:** Per-pattern ΔE = +2.6e-5 (98% positive at K=1); global-pull = −0.027 (alignment 0.967 vs 0.998).

- **Branch scoring (unbiased energy)**
  - **Citation:** phase-5-unified-design.md:177-200
  - **Status:** ✅ IMPLEMENTED
  - **Details:** Softmax weights computed over branches using unbiased (γ-omitted) energy

- **Branch combination** (energy-weighted bundle + re-settle)
  - **Citation:** phase-5-unified-design.md:204-222
  - **Status:** 🟨 IMPLEMENTED
  - **Details:** Bundling works algebraically; re-settle convergence not explicitly measured (listed as decision #4)

- **Atom-splitting diagnostic** (joint criterion: energy similarity + state divergence)
  - **Citation:** phase-5-unified-design.md:225-246
  - **Status:** 🟨 IMPLEMENTED AS MEASUREMENT-ONLY
  - **Details:** Diagnostic logged; actual split *action* deferred to Phase 5 sub-component
  - **Caveat:** Metrics show uniform branch-energy dispersion (branches are equi-energetic to FP precision) — splitting criterion fires weakly if at all

### Mechanisms Designed But Not Built/Tested

- **Atom-splitting *action* mechanism** — once an atom is marked split-eligible, what actually splits?
  - **Citation:** phase-5-unified-design.md:443-450
  - **Status:** DEFERRED
  - **Details:** Two candidate mechanisms sketched (independent-attractor creation, schema-store partitioning) but not specified

- **Cross-cue learning / concept promotion** — if branch energies are stable across many cues for a given schema, schema becomes a *concept*
  - **Citation:** phase-5-unified-design.md:450-452
  - **Status:** DEFERRED TO PHASE 5+

- **Sleep/wake interleaving** — when branching runs (during retrieval) vs when bundling propagates back (during replay)
  - **Citation:** phase-5-unified-design.md:453-455
  - **Status:** DEFERRED

- **Phase 6 reuse** — this mechanism becomes Phase 6's temporal rollout substrate
  - **Citation:** phase-5-unified-design.md:456-457
  - **Status:** DEFERRED

- **Path D training-time intervention (M2)** — EqProp + role-shuffled-negatives + DSM warm-start (the only retrieval-mechanism family not yet smoked at Phase 5)
  - **Citation:** STATUS.md 2026-05-24 (active blocker #1)
  - **Status:** NOT YET SMOKED
  - **Details:** Four retrieval-mechanism families (D1 storage, D3 branch-coupling, E1 landscape-reshape, M1 role-energy stack) all returned null. M2 is the remaining path.

- **Path B′ framing** — "accept Phase 5 closure / pivot per 2026-05-23 SNR walk-back"
  - **Citation:** STATUS.md 2026-05-24 (active blocker #3)
  - **Status:** DESIGN DECISION PENDING
  - **Details:** Three options: (a) commit to M2 implementation on new branch; (b) accept Phase 5 closure / pivot; (c) try dual-code GHRR P1 variant before M2

---

## Phase 6: Integration (Not Yet Designed)

### Headline Metric
Not specified. Phase 6 design has not been written.

### Exit/Graduation Criteria
**File:** experimental-progression.md:104
- SONAR-replacement is non-regressive on retrieval tasks
- Structural reasoning capabilities are present and measurable
- Consolidation channel responds to structural content as designed

### Mechanisms Built/Tested
- None (Phase 6 not started)

### Mechanisms Designed But Not Built/Tested
- **Replay-and-re-encode mitigation** for atom drift staleness
  - **Citation:** experimental-progression.md:106-116, llm-integration.md:102-110
  - **Status:** DESIGNED, NOT BUILT
  - **Design questions:**
    1. Frequency: how often does refresh run?
    2. Prioritization: which patterns get refreshed first?
    3. Granularity: whole patterns or only drifted atoms?

- **Workspace decoder** — takes settled HD states and produces LLM-consumable context
  - **Citation:** llm-integration.md:52-77
  - **MVP version:** top-K retrieval (unbind by positions, find K nearest atoms in codebook, convert to tokens)
  - **Status:** DESIGNED, NOT BUILT
  - **Future versions (deferred):** learned HD-to-text generator, cross-attention bridge, hybrid

- **Cue construction in the workspace** — turn "current activity" into Hopfield cue
  - **Citation:** llm-integration.md:119-120
  - **Status:** OPEN DESIGN QUESTION
  - **Details:** "What's the exact mechanism?" — not yet specified

- **Decoder fidelity question** — is top-K enough, or does decoder need structured context (relationships, temporal context)?
  - **Citation:** llm-integration.md:121-122
  - **Status:** OPEN DESIGN QUESTION

- **Workspace query frequency** — per-turn or continuous?
  - **Citation:** llm-integration.md:123-124
  - **Status:** OPEN DESIGN QUESTION

- **LLM model choice for v1**
  - **Citation:** llm-integration.md:125-126
  - **Status:** OPEN DECISION
  - **Options:** Llama 3.2 1B-3B, Qwen 2.5 3B, Liquid Foundation Model

---

## Cross-Phase Anti-Homunculus Filter Status

**Project commitment:** Every metric, diagnostic, or actuator must measure/express intrinsic geometric state — never as arbitration over states. (phase-3-deep-dive.md:23-31, experimental-progression.md:27-29)

### Phases 1-4: Filter Status

**PASSING cleanly:**
- Substrate operations (binding, bundling, similarity) — physical laws ✓
- Codebook growth via experience — local dynamics ✓
- Hebbian refinement (drift toward context-bag centroid) — local dynamics ✓
- Trajectory trace capture — passive observation ✓
- Engagement × resolution gate — geometric property filter ✓
- Benna-Fusi u-chain — well-known FEP-clean dynamic ✓
- Pattern effective strength — weighted sum ✓
- Pattern death by strength decay — natural endpoint of dynamics ✓

**BORDERLINE / REQUIRED OVERSIGHT:**
- Quality threshold for consolidation buffer entry (phase-3-deep-dive.md:102) — if this becomes a tuned control value, it's an arbitration. Currently ~0.5; presented as "retrieve quality score q ∈ [0,1]" gating. **Status:** Treated as geometric property (soft threshold on continuous signal) ✓

- Repulsion strength (phase-3-deep-dive.md:138) — defined as hyperparameter; if it becomes a "knob the system reads to decide" it's a homunculus. Currently: fixed per-dimensional constraint (FHRR unit-modulus) + soft repulsion term. **Status:** Geometric repulsion, not control ✓

- Cap → codebook soft budget (phase-3-deep-dive.md:124) — accelerated decay on weakest atoms. **Status:** Decay is a local dynamic; weakness (stability + utility) is a measurement, not a decision ✓

### Phase 5: Filter Status

**FLAGGED AS REQUIRING EXPLICIT DISCIPLINE (phase-5-checklist.md:H):**

- H1: **Schema-source ablation diagnostic-only** — the system MUST NOT switch schema sources based on robustness results. Adaptive rerouting is textbook homunculus. **Status:** Constraint documented; Phase 5 does not have adaptive switching logic ✓

- H2: **Atom-splitting diagnostic** is logged; actual split action deferred. **Status:** Current scope is measurement only; split action needs its own anti-homunculus check when deferred phase implements it

- H3: **Branch selection is energy-based only**, never metric-based. Per-branch diagnostics logged, not used for selection. **Status:** Code enforces energy-only selection ✓

- H4: **No "if ΔE fails on C1 but passes on C2, declare graduation"** — graduation is structural, not best-of-N. A robustness sweep that becomes a search is a homunculus. **Status:** Explicit prohibition documented; graduation rule in phase-5-checklist.md:I requires all conditions ✓

**VIOLATED / CAUSING PHASE 5 FAILURE:**

Phase 5's core mechanism (energy-guided branching) passes the anti-homunculus check. But the *failure-mode response* has not:

- **Pattern death mechanism** (Phase 4, carrying to Phase 5) — the binary death step is not a pure geometric dynamic; it's a threshold-triggered action (pattern strength < τ for window → delete). The *response* (deletion) is externally imposed, not a natural endpoint. This was documented as STATUS blocker #2.
  - **Resolution:** 2026-05-20 A+B note proposes replacement (continuous reinforcement + energy-based repulsion). Anti-homunculus reviewer PASS after 4 fixes.
  - **Citation:** phase-5-checklist.md:J (Carried Phase-3/4 items)

---

## Consolidated Critical Gaps (Built vs. Untested)

### Highest-Leverage Untested Mechanisms (by impact on architecture viability)

1. **Consolidation-geometry regime classifier** — core to Phase 3 gating logic, explicitly deferred
   - **Impact:** Tier 1 sanity checks depend on this for stratified headline metric
   - **Status:** DESIGNED, NOT BUILT
   - **Est. effort:** ~1 day (regime computation) + ~2 days (calibration spike)

2. **Error-driven allocation (most thesis-aligned allocation strategy)** — Phase 3 deepens understanding of when atoms are *needed*
   - **Impact:** Directional signal for codebook emergence from retrieval failures
   - **Status:** DESIGNED, NOT TESTED
   - **Est. effort:** Phase 3 follow-up; deferred

3. **Atom-splitting action mechanism** — Phase 5 depends on this to convert bimodality signal to structural abstraction
   - **Impact:** Enables multi-atom polysemous representation
   - **Status:** DESIGNED AS MEASUREMENT; ACTION DEFERRED
   - **Est. effort:** Phase 5 sub-component; complexity TBD

4. **Pattern death mechanism redesign** — Phase 4 status quo cannot actually purge stale patterns
   - **Impact:** Memory bloat, non-convergent dynamics
   - **Status:** REPLACEMENT DESIGNED (A+B), AWAITING IMPLEMENTATION
   - **Est. effort:** ~30 LOC + parameter tuning (per 2026-05-20 note)

5. **Cross-pattern u-variable coupling** — Phase 4 designed but deferred
   - **Impact:** Whether related patterns consolidate together
   - **Status:** DESIGNED, NOT TESTED
   - **Est. effort:** Phase 5 or later

6. **Structural-retrieval mechanism (Phase 5 core)** — four retrieval-mechanism families all returned null
   - **Impact:** Whether the system can do role-based binding discovery
   - **Status:** DESIGNED EXTENSIVELY; EVERY TESTED PATH NULL; ONE PATH REMAINING (M2 training-time)
   - **Est. effort:** Depends on M2 outcome; if M2 fails, Phase 5 closure decision

### Explicitly Open Design Questions (Not Yet Specified)

- **Adaptive m in Benna-Fusi** — theoretical prediction is m ≈ log(T); currently fixed m=6
- **Adaptive store_threshold** — currently fixed
- **Multi-scale interaction** — scales share replay traces or separate?
- **Sleep/wake interleaving** — when replay couples back to substrate
- **Cue construction in workspace** — what bundle represents "current thought"?
- **Decoder fidelity** — what structure must workspace decoder output?
- **Atom-splitting into which attractors** — two candidate mechanisms unsolved
- **Concept promotion mechanism** — when does schema become concept?
- **Phase 6 temporal rollout** — how does Phase 5 machinery extend to time?

---

## STATUS.md Cross-References

**Current status (2026-05-24):**
- **Active phase:** 5 (not graduated)
- **Last verified result:** Report 064 — M1 retrieval-side cross-seed smoke null
- **Active blockers:**
  1. Path D split, retrieval-side ruled out; M2 (training-time) only path not smoked
  2. STATUS blocker #3 retired (rerun gate no longer makes sense)
  3. Architectural commit decision pending (M2 vs closure vs GHRR P1 variant)

**Pre-phase commitments still open:**
- Consolidation-geometry regime classifier
- Empirical θ'(β) calibration spike
- High-leverage brainstorm idea 5 (frequency-weighted α)

---

## Summary Table: Phase Status at a Glance

| Phase | Headline Metric | Status | Citation | Notes |
|-------|---|---|---|---|
| **1** | Clean algebra at scale | ✅ VERIFIED | exper-prog:37-41 | Substrate validated 2026-05-05 spike |
| **2** | Recall@1 vs bigram | ✅ VERIFIED | exper-prog:58 | Both objectives tested; masked-token confirmed superior |
| **3** | Recall@K regime-stratified vs shuffle | 🟨 PARTIAL | phase-3-deep:180-189 | Implementation exists; regime classifier not built |
| **4** | Δ meta_stable_w3 integration regime | ✅ VERIFIED | phase-4-check:38 | Report 038; n=10; D1 graduated |
| **5** | ΔE (role − content) > 5.5e-3 CI-disj | ⚠️ UNATTAINED | phase-5-check:A1 | Report 053; directional but sub-noise; 4 retrieval paths null |
| **6** | (Not yet designed) | ❌ NOT STARTED | exper-prog:104 | Open questions listed; Phase 5 must graduate first |

