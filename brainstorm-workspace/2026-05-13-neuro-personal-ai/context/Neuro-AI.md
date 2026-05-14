---
date: 2026-05-13
project: personal-ai / neuro-ai
type: brainstorm-context-summary
---

# Neuro-AI: Context Summary for Brainstorming Session

## What This Is

Neuro-AI (formerly "personal-ai") is a **memory-first cognitive substrate** project building a persistent, locally-running AI companion that learns from lived experience through an energy-memory landscape. The core thesis is that personality and understanding emerge from consolidated memory, not from LLM weights. The system is structured around FHRR (Holographic Reduced Representations), Modern Hopfield Networks, and a growing codebook of learned atomic patterns that consolidate over time through replay and multi-timescale dynamics.

The project rejects sequence prediction as the organizing principle and embraces **contextual completion** — given a partial cue, retrieve and blend relevant experiences from the landscape. The philosophical commitment: build a companion that knows *me* (specific, persistent, learning from our shared history), not a god that knows *everything* (general-purpose, frozen, broadcast).

The repository is actively under development with substantial phase-based architecture implemented, reaching Phase 5 in some areas (hierarchical abstraction via higher-order associative memory) while consolidating Phase 3 (emergent codebook growth) and Phase 4 (replay-driven consolidation) empirical validation.

## Key Themes and Concepts

### Core Architectural Commitments

1. **Memory is the self.** System personality lives in the landscape topology and consolidation history, not in LLM weights. Two instances of the same frozen LM, paired with different users for months, should diverge into distinct companions through memory consolidation alone.

2. **No homunculus controller.** Every architectural decision (which subsystem wins, when to switch strategies, what to consolidate) is expressed as local geometry, energy dynamics, settling behavior, or tension metrics — never as a supervisor module that reads metrics and executes rules.

3. **Contextual completion, not next-token prediction.** The native question is "what does this remind me of and what fills this unresolved gap?" not "what token comes next?" This reframing aligns Hopfield retrieval (pattern completion via settling) with the system's core capability. Masked-token training is the natural objective; retrieval-quality metrics are the natural evaluation frame.

4. **Continuous learning through lived experience.** The system has fast episodic storage (recent experience, overwritten constantly) and slow stable storage (learned patterns that earn their way in through consolidation). Novel experiences write to fast storage; surprise + user relevance signals gate which experiences consolidate into durable landscape modifications.

5. **Latent reasoning before language.** Reasoning happens in vector/energy states via settling and trajectory dynamics. Language generation comes afterward, via an LLM that reads settled states from the workspace. The LLM is the *voice*, not the *mind*.

### Substrate Technologies

- **FHRR (Holographic Reduced Representations):** Vectors of D=4096 random complex numbers (Phase 1 validation range). Binding, unbinding, bundling, cosine similarity, and cleanup operations are universal and fixed. Operates via pure Python reference backend (no external dependencies) or Torch/MPS for scaling.
- **Modern Hopfield Networks:** Iterative softmax settling that retrieves and blends stored patterns from an energy landscape. Can produce sharp prototypes (high β, high resolution) or soft feature blends (low β, multi-pattern contribution). Built on Ramsauer et al. (2020) architecture.
- **Temporal Association Memory:** Each stored pattern is bundled with a "temporal context bag" of nearby co-experienced patterns, enabling retrieval by temporal co-occurrence rather than similarity alone. This is load-bearing evidence that temporal association carries information independent of content geometry.
- **Codebook as Emergent Atoms:** Replaces random codebooks with learned atomic patterns that allocate, drift, stabilize, split, and decay through consolidation. Atoms are persistent but plastic — unlike frozen embeddings or static vocabularies.
- **Trajectory Tracing + Replay:** Hopfield settling steps are captured as a trajectory (per-step co-activations, entropy, convergence path). Unresolved trajectories enter a replay buffer gated by engagement × low-resolution. Replayed trajectories re-settle in the current landscape and enter consolidation if resolved.
- **Benna-Fusi Multi-Timescale Consolidation:** Retrieved patterns move through a chain of m consolidation variables (u₁, ..., uₘ) with exponentially increasing time constants. Bidirectional coupling bridges fast settling and slow stable storage. Only patterns that consistently re-emerge through replay consolidate to slow variables and become durable.

### Emergent Properties

- **Memory and exploration are the same operation.** Retrieving from a blended Hopfield landscape can produce states that were never explicitly stored — two ideas combining into a novel blend. This is what the system calls "creative recombination" and it emerges from retrieval dynamics, not from bolted-on exploration modules.
- **Temporal shuffle ablation is the key control.** The cleanest evidence that the system uses temporal structure (not just similarity) is: shuffle temporal order, preserve content, measure the gap in retrieval quality. This empirically tests the core thesis across any substrate.
- **Compression regime determines transferability.** When training a predictor under-capacity constraints (forced compression), it extracts recurrent patterns that transfer to held-out data. Over-fitted predictors memorize contingent co-occurrences that don't generalize. This suggests Phase 4/5 consolidation should create under-capacity bottlenecks in the replay pathway.
- **Cap-coverage error is the true retrieval metric.** The neighborhood around a retrieved state matters more than exact-token recovery. Cap-coverage = "does a ball around the representative contain the correct pattern?" This is architecturally aligned with the claim that *meaning lives in the retrieval neighborhood*.

## Current State

### What Exists

**Phase 0 – Energy Memory Kernel (Complete)**
- Pure-Python FHRR with binding, unbinding, bundling, cosine similarity, cleanup
- Iterative Hopfield retrieval with softmax settling, entropy tracing, energy diagnostics
- Temporal association memory with shuffle controls
- Synthetic temporal recall and similarity-distractor experiments
- 30+ experiment drivers and validation tests
- MPS backend available outside sandbox

**Phase 1 – Scaled Substrate Validation (Complete)**
- Torch/MPS batched FHRR operations at D=4096 (validated empirically)
- Batched Hopfield retrieval
- Larger synthetic memories with benchmarking
- Runtime and memory footprint reports
- Deterministic persistence for codebooks and patterns

**Phase 2 – Static Contextual-Completion Baseline (Mostly Complete)**
- Full experimental matrix: 540 conditions across window size, mask count, mask position, landscape size, β, retrieval condition, objective
- WikiText-2 pipeline (word-level tokenization, ~9K vocabulary)
- Baseline comparisons (bigram, unigram, random)
- Cap-coverage, metastable-rate, entropy, energy, confidence-gap metrics
- Per-retrieval trajectory diagnostics (settled state modulus, softmax entropy, top-2 margin)
- Reports: Phase 2 static baseline performance vs masked-token/next-token/control conditions
- Key finding: **masked-token outperforms next-token** consistently, validating contextual-completion framing

**Phase 3 – Growing Codebook (Actively Developed)**
- Lexical atom allocation (one per token type initially)
- Two-pathway hybrid update: Hebbian reinforcement on high-quality retrieval, error-driven update on failure
- Context-bag tracking for atomic drift
- Stability metrics, decay, and budget pressure
- Online codebook learner with per-experience Hebbian pathways
- Error-driven learner triggered on retrieval failures
- Reconstruction learner for backpropagating failed retrievals into atom updates
- Multiple test variants: `phase3a_hebbian`, `phase3b_error_driven`, `phase3c_reconstruction`
- Validation on multiscale performance (Recall@K by landscape size)
- Key finding: **codebook does not collapse**, atoms stabilize around meaningful neighborhoods

**Phase 4 – Replay and Trajectory Consolidation (Actively Developed)**
- TrajectoryTrace data structure capturing per-step settling dynamics
- Engagement metric (mean entropy across steps, weighted by iteration)
- Resolution metric (max cosine of settled state to any stored pattern)
- Replay store gated by engagement × (1 - resolution) threshold
- Replay scheduler with age decay and prioritization
- Benna-Fusi consolidation chain implemented with configurable time constants
- Sparse-update principle (only update consolidation variables for affected patterns)
- Validation runs showing replay improves future retrieval and stabilizes drift
- Key finding: **replayed trajectories consolidate correctly** and improve subsequent retrieval

**Phase 5 – Structure and Abstraction (Early Exploration)**
- Higher-order associative memory (HAM) for composing learned patterns into structures
- Layer-2 discovery (binding relationships between atoms across multiple timescales)
- Experiments with hierarchical compression and structural matching
- Integration with Phase 3/4 learning dynamics
- Early results suggest structural patterns emerge from co-consolidation

### What's in Progress

- **Phase 2 Session 3+:** Multi-session experimental driver consolidation with new metrics (cap-coverage, NC1 variability, metastable-rate); full matrix aggregation with Wilson CIs; frequency stratification; visualization
- **Phase 3 Diagnostic Integration:** Cap-coverage error, NC1 within-cluster variability (unsupervised reformulation), meta-stable-state rate, softmax entropy as feature/prototype classifier, empirical θ′(β) calibration
- **Phase 4 Full Integration:** Multi-timescale replay loop with all consolidation stages; validation against shuffled-token and random-codebook controls
- **Phase 5 Scaling:** Hierarchical composition of atoms into schemas; cross-timescale binding discovery

### What's Incomplete or Uncertain

- **Spatial composition beyond temporal context bags.** Temporal association is load-bearing; structural role/filler binding from experience is sketched but not empirically validated
- **Atom splitting for polysemy.** Bimodality tracking is implemented; splitting mechanics and its interaction with consolidation are not
- **World model and predictive rollouts (Phase 6).** Designed conceptually; not built
- **LLM integration (Phase 7).** Workspace architecture sketched; concrete wiring to a small local LLM not yet attempted
- **Diagnostics-to-actuators threshold.** Identified as the next conceptual blocker; the project can measure geometric states but hasn't established principled rules for slow-timescale response

## Open Questions and Tensions

### Architecture-Level

1. **Temporal co-occurrence + structural binding coexistence.** Dury's PAM papers (added 2026-05-11) show that temporal co-occurrence is a separate information channel from content similarity, with empirical evidence (90% gap collapse under temporal shuffle). How does this integrate into the FHRR + Hopfield substrate without adding separately-trained predictors? The current sketch (temporal context bundle at consolidation time) works but may not be load-bearing vs. simpler alternatives.

2. **Query generation from "the idea."** The workspace is where ideas live separately from token surfaces. The mechanism to extract an idea from current dynamics and turn it into a cue for retrieval is unspecified. "Extract and embed the idea itself" is the phrase; the operation is not concrete.

3. **Writing into the landscape.** Retrieval (settling) is well-specified. What does writing a new experience into the landscape actually do — carve a new valley, deform an existing one, merge with a nearby pattern? The suspicion is reading and writing are more entangled than they appear, but this wasn't worked through.

4. **Compression regime vs bottleneck design.** Dury's concept-discovery paper shows that under-capacity training extracts transferable patterns while over-fitted training extracts memorized specifics. Should Phase 4/5 consolidation be designed with explicit bottlenecks in the replay pathway to force pattern extraction? Or does the Benna-Fusi multi-timescale mechanism already produce this for free?

5. **Soft-blend vs prototype regimes.** At D=4096, the soft-blend regime (β ≤ 0.001) produces multi-pattern blending but also fails to retrieve entirely on memorization tasks. Sharp-retrieval (β ≥ 0.01) succeeds but commits to single patterns. This is a regime disjoint failure — can it be bridged dynamically, or does it suggest the architecture needs a different temperature control mechanism?

### Evaluation-Level

6. **Headline metric specification.** Phases 2+ are committed to explicit headline metrics with drill-down diagnostics, not multi-metric panels. Phase 3 headline: **Recall@K on masked-token contextual completion, stratified by regime classification, vs shuffled-token control.** The choice to stratify by regime (feature vs prototype mode) is principled but requires empirical θ′(β) calibration. Is the calibration worth the 2-day spike before Phase 3 Session 1?

7. **When does diagnosis become intervention?** The anti-homunculus filter says all responses to geometric observations must be local dynamics, not rules. But the boundary between "measuring a slow dynamic" (metastable-state decay over replay cycles) and "implementing a controller" (decay threshold that gates pattern eviction) is fuzzy in practice. The diagnostics-vs-actuators threshold is identified but not worked out.

8. **Control condition sufficiency.** Temporal shuffle ablation is the project's standing control for "is the system using temporal structure?" But the four other control types (random codebook, random atoms, no-replay, shuffled-token labels) haven't all been run on all Phase 3+ conditions simultaneously. Should Phase 3 Session 1 include a comprehensive control matrix, or is it better to be selective?

### Implementation-Level

9. **Sparse vs dense atom updates.** Phase 4 uses sparse updates (only modify consolidation variables for affected patterns) per SQ-HN's anti-forgetting results. But it's unclear whether sparsity applies only to consolidation variables or also to atom vector drifts in Phase 3. The interaction between sparse updates and the two-pathway learning (Hebbian + error-driven) is not fully specified.

10. **Replay buffer management under changing codebooks.** A trajectory was recorded with the current atom set; replay runs it through a later landscape where atoms may have drifted, split, or been garbage-collected. What is the correct semantics for re-settling a trajectory in a changed landscape? Should the trajectory bind to the nearest-neighbor atoms, or does it fail to resolve and re-enter the buffer?

## Interesting Threads

### Emerging Possibilities

1. **Predictive associative memory as orthogonal channel.** Dury's work suggests that temporal co-occurrence is empirically separate from content similarity. A dual-channel architecture (similarity + association) for "specificity = similarity × association" may be loadbearing. The PAM papers explicitly call this out; the personal-ai project has temporal association but hasn't explored the similarity × association product for retrieval decisions.

2. **Temporal context as structural binding.** Current implementation uses temporal context bags (unordered bundles of co-occurred atoms). Could time itself be a binding dimension? Atoms bound with temporal offsets (before, during, after) would preserve causal structure while remaining generic. This may be the bridge between temporal association and structural reasoning that Phase 5 needs.

3. **Bimodality detection → atom splitting → polysemy preservation.** Bimodality is tracked in Phase 3; splitting is designed in Phase 5. The empirical observation that some atoms cluster into bimodal distributions (polysemous words) suggests the system could detect *when* to split. Timing the split to high-consolidation periods (not every retrieval failure) may be the rule that preserves ambiguity strategically.

4. **Multi-scale codebook discovery.** Early Phase 5 results show layer-2 patterns (relationships between atoms) emerge at different timescales. This may be evidence that the codebook itself is hierarchically organized — word-level atoms at fast scale, concept-level atoms at slow scale, schema atoms at even slower scale. A unified consolidation mechanism operating at multiple scales simultaneously could discover this hierarchy without explicit specification.

5. **Engagement as a learning-rate signal.** Replay store gating uses engagement (mean entropy across settling steps) as a measure of "which trajectories had unresolved tension." What if engagement is also used to modulate the magnitude of consolidation updates? High engagement → larger weight on error-driven pathway (the landscape was pulling hard in multiple directions). Low engagement → larger weight on Hebbian pathway (the landscape was settled and consistent). This would couple learning rate to the landscape's own signal about its stability.

6. **Generalization through under-capacity bottlenecks in replay.** If Dury is right that compression extracts transferable patterns, perhaps Phase 4's replay should be designed with explicit dimensional bottlenecks — the re-settling must produce a lower-dimensional summary before it enters consolidation. This would be analogous to autoencoders forcing information through a narrow channel. Worth testing as a Phase 4 variant.

### Threads from Recent Reading

1. **Actuator-dynamics threshold (2026-05-09 insight).** The project can now measure geometric states (cap-coverage, NC1, metastable-rate, entropy). The next level is implementing slow-timescale *responses* to these measurements without adding supervisory logic. Example: "if cap-coverage drops, increase spatial separation" is a rule (homunculus). "Replay drives coupling between consolidation variables u_k and u_{k+1}, and this coupling naturally increases separation under coverage pressure" would be the dynamic. This is the difference between actuators that read metrics and dynamics that *express* metrics. Identifying this threshold in Phase 4/5 work is likely to surface new design insights.

2. **Compression regime as a first-principles handle on generalization.** Dury's papers show that the same architecture (same training signal, same loss, same hyperparameters) produces memorizers or generalizers depending on capacity constraints. This suggests the project's consolidation design should *actively impose* under-capacity conditions in the slow-variable chain, not hope they emerge. Phase 4 should include a variant with explicit bottlenecking in the Benna-Fusi chain (e.g., m=2 vs m=4) and measure transfer learning as a function of bottleneck width.

3. **Entity persistence as a missing precondition for creative bridging.** Dury explicitly notes that cross-trajectory association requires entity identity persistence. Personal-ai will eventually need "object permanence" — recognizing that the same entity appears in multiple episodes — to enable creative bridging across separate trajectories. This is likely a Phase 5/6 feature, but it should be on the long-term roadmap. Early experiments could use synthetic worlds with tagged entities to validate the principle.

## Technical Stack/Tools

- **Language:** Python 3.x
- **Core backends:** 
  - Pure Python (no external dependencies) for reference substrate
  - PyTorch + MPS (Metal Performance Shaders) for scaling
- **Dependencies:** NumPy, Torch, HuggingFace datasets (WikiText-2), scipy (CIs)
- **Development environment:** M5 MacBook Pro with 24GB unified memory; MPS available outside sandbox
- **Persistence:** Pickle/torch.save for codebooks, patterns, checkpoints
- **Testing:** Python unittest (no pytest), 50+ validation tests
- **Experimentation:** Modular driver architecture (one driver per phase/variant) producing markdown reports and CSV results
- **Data:** WikiText-2 corpus for contextual-completion validation; synthetic temporal-sequence worlds for Phase 0 proofs
- **Version control:** Git; research papers in `/research`, extracted text in `/tmp/pdf_text/`

## Navigation and Continuation

### For the Next Brainstorming Session

Key documents to review before diving deep:
- `docs/PROJECT_PLAN.md` — phased roadmap and non-negotiable design rules
- `notes/briefing.md` — original architectural framing and core dissatisfaction
- `notes/emergent-codebook/phase-3-deep-dive.md` — the highest-risk phase with open design questions
- `notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md` — the diagnostic stack, headline metric principle, anti-homunculus filter
- `notes/notes/2026-05-11-pam-dury-papers.md` — recent literature synthesis on temporal association

### What Would Be Valuable to Explore

1. **Diagnostics-to-actuators bridging (highest impact).** The threshold identified in 2026-05-09 is the next conceptual bottleneck. Working through how a measured state becomes a local dynamic response (without a homunculus reading metrics) would unlock Phase 4/5 design clarity.

2. **Temporal co-occurrence + structural binding coexistence.** Dury's papers are fresh. How do they reshape Phase 5's binding design? Could temporal offsets be the primitive binding dimension, with roles/fillers emerging later?

3. **Multi-scale codebook emergence.** Early Phase 5 results hint at hierarchical structure. Is there a unified consolidation mechanism that discovers hierarchy automatically, or does it need to be specified?

4. **Compression via bottleneck in replay.** Test whether explicit dimensional constraints in the Benna-Fusi slow chain produce better transfer learning (fewer episodes needed to generalize). Variance across bottleneck widths could inform optimal replay design.

5. **Soft-blend regime bridging.** The current soft-blend failure at memorization may be solvable through dynamic temperature control or regime-switching, not regime replacement. Understanding whether this bridge exists is critical for Phase 5+ expressivity (feature blending is essential for abstraction).

---

**Last updated:** 2026-05-13  
**Status:** Ready for brainstorming exploration  
**Session focus:** Strategic architecture decisions for Phases 4–6  
