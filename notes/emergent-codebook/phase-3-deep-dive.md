---
date: 2026-05-04
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Phase 3 Deep Dive — Codebook Growth

Phase 3 is where most of the design risk lives. Phase 1 is validating math, Phase 2 is validating Hopfield mechanics — both have well-known answers. Phase 3 is where specific *learning dynamics* must work, and the literature is much thinner here.

> **Updated 2026-05-09** with the headline metric, diagnostic stack additions (cap-coverage, meta-stable-state rate, NC1 reformulation, softmax entropy interpretation), and the project-level anti-homunculus filter as a constraint on future evaluation additions. See `notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md` for the synthesis these came out of.

## Architectural framing

Before the mechanics: the system being built is a **contextual-completion architecture, not a sequence-prediction architecture**. The codebook's job is to support retrieval — given a partial cue (current context), surface relevant content from accumulated experience. Modern Hopfield networks are associative memories; pattern completion is what they do natively. Trying to make them act like RNNs or transformers (sequence prediction) would fight the substrate. Trying to make them surface contextually-relevant content (pattern completion) plays to it.

This framing flows through everything in Phase 3: the prediction objective is masked-token (pattern completion), the evaluation metrics are retrieval-relevance, and the consolidation gate routes per-experience between reinforcement and error-driven update pathways based on retrieval quality.

## Project-level commitment: the anti-homunculus filter

*(Added 2026-05-09.)*

Every metric, diagnostic, or actuator added to Phase 3 must measure or be expressible as an *intrinsic geometric state* — not as an arbitration over states. The five core questions and the two-pathway hybrid update were designed under this constraint; new evaluation additions must respect it too.

Practical consequence for evaluation: a metric that requires a supervisor to interpret it (e.g., "if retrieval quality drops, increase repulsion strength") is doing the same work as a controller, just at evaluation time. Diagnostics surface geometric states. Responses to those states should be local dynamics, not rule-based interventions on the dynamics. This filter is the standing reason to reject ad-hoc fixes when something doesn't work in early Phase 3 iterations.

The diagnostics-vs-actuators distinction (the next conceptual threshold for the project, identified in the 2026-05-09 note) is where this filter is most likely to be tested. When a Phase 3 metric surfaces a geometric problem, the response should be a slow-timescale dynamic that the metric is a fast-timescale snapshot of — not a rule that reads the metric and triggers an intervention.

## The five core questions

### 1. When does a new atom get allocated?

Three options on a complexity gradient.

**Simplest — lexical:** on first encounter of a new token type. Easy to implement, but gives a vocabulary-sized codebook from day one with no actual emergence.

**Middle — pattern-based:** when a pattern of co-occurrence exceeds a novelty threshold. More aligned with the thesis but adds a "what counts as novel" parameter to tune.

**Most thesis-aligned — error-driven:** when retrieval failures reveal that current atoms can't explain a pattern. The system grows representational capacity in response to its own failures. Active-inference framing.

**MVP recommendation:** start with lexical. Get refinement dynamics working before adding allocation complexity.

### 2. How do existing atoms refine?

**Two-pathway hybrid update**, gated per-experience by retrieval quality. The two pathways compute structurally different things, not the same operation at different magnitudes.

**Reinforcement pathway (Hebbian-shaped):** atoms drift toward their shared context bag. Signal: "you co-occurred with these atoms in a context where retrieval worked; this neighborhood is correct, settle into it more firmly." Continuous, small magnitude. Fires when retrieval quality is high.

**Error-driven pathway (consolidation event):** atoms drift toward making the cue more similar to stored patterns that would have produced the correct retrieval. Signal: "the cue retrieved the wrong thing; the right thing was X; restructure so cues like this would land closer to X-containing patterns next time." Batched, larger magnitude. Fires when retrieval quality is low.

Both pathways can fire on a single experience with proportional weighting based on retrieval quality score q ∈ [0, 1]. High q → Hebbian-dominant. Low q → consolidation-buffer-dominant. Very low information → skip both.

### 3. How do you measure stability?

An atom's stability score tracks how much it's been changing recently. Concretely: a moving variance of the atom's vector over the last N consolidation events. Atoms with low variance are stable; atoms with high variance are still settling in.

Stability gates two things:

- Consolidation depth — stable atoms get smaller updates per event, so they don't get destabilized by single experiences.
- Decay protection — stable atoms decay slowly even if temporarily unused.

### 4. When and how do atoms decay?

Combination of time-based and quality-based: an atom decays if it hasn't been activated in N steps AND its prediction-utility score is below threshold. The vector drifts toward zero (or random) over decay events; once below a magnitude threshold, the atom is pruned.

Two-condition gate matters. Pure time-based decay is too aggressive — rare-but-important atoms die. Pure quality-based is too lenient — low-utility but frequently-used atoms persist.

### 5. How do you prevent collapse and explosion?

**Collapse** (all atoms drift toward same point) is the worst failure mode. Two safeguards:

- Repulsion term in refinement dynamics that pushes atoms away from nearest neighbors when they get too close.
- Unit-modulus normalization per dimension (FHRR's natural constraint) prevents collapse to zero.

**Explosion** (codebook grows unbounded) is less catastrophic but inefficient. Soft cap with weakest-decay: when codebook hits N atoms, weakest atoms (lowest stability + utility) get marked for accelerated decay until size returns to budget.

## Concrete MVP design for Phase 3

### Data structure

Codebook is a dictionary of (token_id → hypervector + metadata). Metadata per atom:

- usage_count
- last_used_step
- stability_score (moving variance)
- utility_score (moving avg of contribution to correct retrievals)
- context_bag_history (rolling window of recent context bags for bimodality tracking)

### Per-experience flow (continuous, online during training)

For each masked-token training experience:

1. Encode the cue (sequence with masked position) as bundled hypervector.
2. Hopfield settles, retrieves completion.
3. Score retrieval quality q ∈ [0, 1]: cosine similarity between retrieved completion and the actual masked token's hypervector, possibly normalized against a baseline.
4. **If q is high (above success threshold):** apply Hebbian update with magnitude proportional to q. For each atom that participated in the cue, drift slightly toward the *clean context bag* (bundle of other atoms in the cue, no position bindings).
5. **If q is low (below success threshold):** add experience to consolidation buffer for error-driven processing at the next consolidation event.
6. **If retrieval was trivial** (very high q on a low-information cue, or high entropy retrieval that's effectively random): skip both pathways.

The Hebbian pathway is what runs continuously. It reinforces the geometry that's already working.

### Consolidation event (every K steps, start K=100)

Processes accumulated failed-retrieval experiences from the buffer. This is the error-driven pathway.

1. Pull experiences from the buffer.
2. For each experience, the cue retrieved the wrong stored pattern. Identify the *correct* stored pattern (the one that was supposed to be retrieved — known because we know the actual masked token, and the stored pattern containing it is the target).
3. For each atom t in any failed cue:
   - Identify t's position p in that cue
   - Identify x_p — the atom at position p in the correct retrieval target
   - x_p is what t "should have looked like" for that experience to retrieve correctly
4. Aggregate across all failed experiences involving t: target = average of x_p values across experiences (where each experience contributes its specific x_p to t's update target).
5. Apply update: `t_new = (1 - α) × t_old + α × target`
6. Renormalize per-dimension to unit modulus (preserves FHRR algebra).
7. Update stability_score (variance from previous version).
8. Update context_bag_history with new context bags from this batch.
9. Apply repulsion: if any atom has a nearest neighbor closer than threshold, push them apart slightly.
10. Decay step: atoms with last_used_step > threshold AND utility_score < threshold drift toward random by small amount.
11. Allocation step: any new tokens encountered get fresh random hypervectors.
12. Soft cap: if codebook size exceeds budget, accelerate decay on weakest atoms.

The error-driven update IS gradient-shaped feedback through HD operations — flow the error signal from the bundle level down to the constituent atoms. Atoms in failed cues get pulled toward the atoms that should have been there for retrieval to succeed.

### Starting hyperparameters

| Parameter | Starting value |
|-----------|----------------|
| Consolidation interval K | 100 |
| Hebbian learning rate (success) | 0.01 |
| Error-driven learning rate (failure) | 0.1 |
| Quality threshold for buffer entry | 0.5 |
| Trivial-skip thresholds | q > 0.95 or retrieval entropy > 0.9 |
| Decay threshold (steps unused) | 1000 |
| Repulsion threshold (cosine sim) | 0.7 |
| Codebook budget | vocab size + 20% buffer |

### Runtime vs training asymmetry

The error-driven pathway requires a known correct retrieval target — the actual masked token. This is available during training (over corpus or replay buffer) but not at runtime (live conversation has no ground truth).

**Implication:** the codebook's significant learning happens during dedicated training passes. Runtime use produces only Hebbian adjustments — small reinforcement of patterns that successfully retrieve. This matches the existing project architecture: replay drives most consolidation, live use is mostly retrieval with light reinforcement.

## Bimodality tracking (Phase 3 hooks for future atom-splitting)

A Phase 3 atom that consistently appears in two distinct context clusters (different "meanings") is a candidate for splitting. The signal: persistent **bimodality** in the atom's context bag distribution across consolidation events.

Crucially, *bimodality* — not raw variance. Function words like "the" have high context-bag variance because they appear with everything, but the distribution is unimodal (uniformly spread). Polysemous words like "bank" have bimodal distributions (clustered into financial vs river contexts). Splitting on raw variance would explode the codebook on function words; splitting on detected bimodality keeps "the" intact while separating polysemous atoms.

### Mechanism

For each atom, maintain context_bag_history as a rolling window over recent consolidation events. After each consolidation event, run a lightweight statistical test for multimodality:

- Hartigan's dip test on context bag similarity distribution, OR
- 2-component Gaussian mixture fit with BIC comparison against single-component

If the atom shows persistent bimodal distribution across N consecutive consolidation events (start N=5), flag it as a splitting candidate.

### When does splitting actually fire?

Phase 3 only tracks the signal — splitting itself fires in Phase 5 alongside binding discovery. But the diagnostic value during Phase 3 is real:

- High persistent bimodality on common atoms = system signaling polysemy that the codebook can't currently express
- Useful for tuning consolidation parameters; if everything is going bimodal, the codebook is being asked to do too much per atom

### Why not split in Phase 3?

Splitting is structurally similar to allocation but with extra mechanics: identifying the two cluster centroids, deciding which atoms inherit the original token ID, handling stored Hopfield patterns that contain the original atom. It's complex enough to deserve its own phase; lumping it into Phase 3 risks destabilizing the foundational dynamics.

## Evaluation: headline metric + drill-downs

*(Restructured 2026-05-09. The previous "Evaluation tiers" section is preserved as the drill-down structure below.)*

Phase 3 has one headline metric that defines whether the phase crossed its viability threshold, and a panel of drill-down metrics that explain why the headline moved. Drill-downs are debugging tools, not competing definitions of success. See `experimental-progression.md` for the project-wide commitment to this structure.

### Headline metric

**Recall@K on masked-token contextual completion, stratified by regime classification, evaluated against the shuffled-token control.**

One number, one stratification axis, one controlled comparison. Aligns directly with the architecture's core claim: retrieval quality emerges from geometry-conditioned consolidation.

- **Recall@K on masked-token completion** — the primary measurement. Pattern completion is the operation Modern Hopfield networks do natively; this metric is the most architecturally appropriate scoring.
- **Stratified by regime classification** — uses the consolidation-geometry diagnostic (`consolidation-geometry-diagnostic.md`) to split atoms into tight-regime and spread-regime sets. Strong performance only in tight-regime atoms is a different story from strong performance across both regimes — the stratification is what lets the headline distinguish them.
- **Vs. shuffled-token control** — the standard run vs. the run with token-to-initial-hypervector assignments randomly permuted. If the structure appears in both, it's a corpus-statistical artefact rather than learned representation. Real signal in the standard run should be absent or substantially attenuated under shuffled assignments.

The phase's viability decision is made on this number. Drill-downs explain the number; they do not replace it.

### Drill-down: Tier 1 — Sanity checks

These are minimum bars. Failing any indicates a bug or fundamental design problem.

- Codebook stops drifting (running variance of codebook decreases over training)
- Atom-level drift rate stabilizes (per-atom moving variance should decrease over training for the majority of atoms; atoms that remain high-drift after extended training are candidates for investigation — possible credit-assignment noise, underpowered repulsion, or genuine polysemy that Phase 5 splitting will resolve)
- **No collapse (added 2026-05-09):** NC1 within-basin variability remains bounded and non-zero — does not trend toward zero across consolidation events. **The supervised-case framing of neural collapse does not transfer directly: in supervised classification NC1 → 0 is a desirable property; in this unsupervised setting NC1 → 0 means basins have collapsed to points and the blended-retrieval property the architecture relies on is gone.** The reformulated diagnostic: *maintain bounded non-zero within-basin variability while preserving inter-basin separability.* Track NC1 jointly with an inter-basin separability metric (minimum pairwise basin distance, or generalized NC2 — distance from uniform separation). The pair tells the full story:

  | NC1 trajectory | Separability trajectory | Interpretation |
  |---|---|---|
  | Stable, non-zero | Growing | Healthy — basins forming with shape, getting more distinguishable |
  | Trending to zero | Growing | Failure — basins collapsing to points, classification-style |
  | Stable | Stable | Frozen codebook, refinement not firing |
  | Stable | Shrinking | Codebook converging to single point — catastrophic collapse |

- **No explosion** (codebook size stays bounded)
- **Meta-stable-state rate stays low (added 2026-05-09):** fraction of retrievals whose settled state is far from any stored pattern (max stored-pattern cosine below 0.95 at convergence). High meta-stable-state rate indicates stored patterns are not sufficiently separable — landscape is producing saddles rather than clean attractors. Direct lift from HEN methodology (Kashyap et al. 2024).
- **Cap-coverage error tracked (added 2026-05-09):** at θ ∈ {0.3, 0.5, 0.7}, the fraction of retrievals whose max stored-pattern cosine falls below θ. Distinguishes "retrieved the wrong pattern" (top-1 incorrect, max cosine still high) from "didn't reach any stored pattern's basin" (max cosine low). For an architecture where meaning lives in the retrieval *neighborhood* rather than in exact identity, these failure modes have different implications. Per the Geometry of Consolidation paper, identity and cap-coverage can diverge by 10× on the same data.
- Headline metric (Recall@K masked-token, regime-stratified, vs. shuffled control) shows meaningful improvement over Phase 2 baseline.

### Drill-down: Tier 2 — Distributional structure

Take 100-200 token pairs with known semantic relationships (synonyms, antonyms, common collocations — pull from WordNet or hand-curate). Compute cosine similarity between hypervectors at start and end of training. Synonyms and collocations should show *increased* similarity over training; unrelated pairs should show no significant change.

Distributional structure is the mechanism that *enables* good retrieval, so testing it directly remains valuable. The framing changes from "did the codebook learn to predict" to "did the codebook learn to support retrieval." This is the clearest "is the codebook actually learning the right structure" test.

The shuffled-token control referenced in the headline metric is computed at this tier: run the identical training pipeline with token-to-initial-hypervector assignments randomly permuted. If Tier 2 distributional structure still emerges under shuffled assignments, the structure is an artefact of corpus statistics (frequency, co-occurrence patterns baked into the training procedure itself) rather than genuine learned representation. Any real signal in the standard run should be absent or significantly attenuated in the shuffled control.

### Drill-down: Tier 3 — Retrieval-shaped analogical structure

Given a partial structural pattern, does the system retrieve completions that match? This is the analogical test reframed for retrieval. Don't expect to crush this at Phase 3 — analogical structure usually emerges more strongly at Phase 5 with binding discovery. But seeing *some* signal here would be encouraging.

### Drill-down: Per-retrieval interpretive layer (added 2026-05-09)

These don't have their own thresholds — they're interpretive overlays on the headline metric and Tier 1/2/3 drill-downs.

- **Softmax entropy as feature/prototype mode classifier.** From Krotov & Hopfield 2016. The interaction-order parameter (project's β) controls a regime spectrum from feature-matching (low β, blended retrieval, multiple memories contribute) to prototype-matching (high β, single memory dominates). Softmax entropy on the retrieval distribution at convergence directly classifies which mode each retrieval is operating in. Already computed as part of the per-retrieval metric set; this is the interpretive frame for what it means.
- **Regime classifier distribution.** From `consolidation-geometry-diagnostic.md`. What fraction of atoms are in tight regime vs. spread regime, and how does that distribution shift over training? Used to stratify the headline metric.
- **Bimodality flag rate.** From the bimodality tracking section above. What fraction of atoms are flagged as splitting candidates? Universal bimodality is a Phase 3 failure mode (codebook is being asked to do too much per atom).

## Phase 3 specific failure modes

| Failure | Diagnosis | Fix |
|---------|-----------|-----|
| Codebook collapse | NC1 within-basin variability trends toward zero AND/OR pairwise similarity histogram narrows toward 1.0 *(diagnosis updated 2026-05-09)* | Increase repulsion strength, decrease learning rates, check normalization step |
| Chaotic instability | Stability scores stay high indefinitely | Decrease learning rates, increase consolidation interval K, smaller repulsion magnitude |
| Frozen codebook | Post-training similarity for related pairs not different from pre-training | Increase learning rates, check that consolidation step is actually firing, verify retrieval-quality signal is being computed correctly |
| Long-tail starvation | Stability score correlated with usage count, rare tokens still random after extensive training | Usage-frequency-normalized learning rate (rare tokens get larger updates per encounter), or sampling strategy that oversamples rare tokens during consolidation |
| Allocation runaway | Codebook keeps growing because near-duplicate atoms keep getting allocated; budget keeps getting hit | Pre-allocation similarity check (don't allocate if similar atom already exists), or post-allocation merge step |
| Universal bimodality | Bimodality detector flags most atoms as candidates | Codebook is being asked to do too much per atom — too few atoms total, or context bags are too noisy. Check K (consolidation interval) and codebook budget |
| Pathway imbalance | Hebbian dominates entirely (most retrievals "succeed" trivially) or error-driven dominates entirely (almost nothing succeeds) | Adjust quality threshold; rebalance trivial-skip; verify masked-token training is producing meaningful retrieval challenges |
| Shuffled control matches standard | Distributional structure appears in both shuffled and standard runs | Structure is corpus-statistical artefact, not learned representation. Codebook refinement dynamics are not adding value beyond what the training procedure imposes. Investigate whether the update rules are actually modifying atoms meaningfully or just reflecting input statistics. |
| **Meta-stable-state rate climbs** *(added 2026-05-09)* | Fraction of retrievals with max stored-pattern cosine below 0.95 at convergence increases over training | Stored patterns are losing separability. Check landscape size vs. capacity; check whether atoms are drifting in a way that pulls multiple stored patterns toward each other. Consider whether β needs adjustment to recover sharper retrieval. |
| **Cap-coverage error climbs while top-1 holds** *(added 2026-05-09)* | Top-1 accuracy stays stable but cap-coverage at θ = 0.5 (or higher) increases over training | System still retrieves the right pattern as top-1, but is doing so from less-deep basins — the right pattern is winning by a thinner margin. Indicates the landscape is becoming flatter overall. Not always bad (could be increased ambiguity tolerance), but worth investigating whether the trend continues. |
| **Online error-driven contrastive updates collapse retrieval** *(added 2026-05-12)* | top-1 degrades within a few hundred cues when error-driven pull/push runs inside a streaming cue loop. Tested on random init (top-1 → chance by ~3000 cues) and on the pretrained Phase 3c codebook (top-1 0.107 → 0.020 by 3500 cues). | Online error-driven contrastive updates are out of scope. The deep-dive's runtime-vs-training asymmetry (line 140-145) is now empirically confirmed: error-driven belongs in batch passes (`ErrorDrivenLearner`, `ReconstructionLearner`); runtime uses Hebbian reinforcement (`HebbianOnlineCodebookUpdater`). See `reports/phase34_stable_v2/findings.md` and `reports/phase34_hebbian/findings.md` for the diagnostic chain. |

## The biggest open uncertainty

The biggest open question for Phase 3 isn't any of those mechanisms — it's whether the **retrieval signal is rich enough** to drive useful refinement at all.

The masked-token reframing helps significantly (pattern completion plays to Hopfield's strengths). But there's still a question whether the Hopfield-retrieval-quality signal in HD space produces enough gradient to shape atoms meaningfully. If the signal is too weak, no amount of clever consolidation rules will help.

If Tier 1 sanity checks pass but Tier 2 distributional structure doesn't appear, the retrieval signal is the most likely culprit. Possible fixes:

- Increase the difficulty of masked-token training (mask multiple positions, force longer-range structure)
- Add auxiliary contrastive loss between bundles that should be similar
- Add reconstruction loss (encode → bundle → Hopfield store → retrieve → decode → compare to original)

If sweeping these doesn't yield Tier 2 signal, the problem is structural. The substrate or Hopfield setup may need fundamental rework.

## Hyperparameter sweep priorities

If Tier 2 fails initially, sweep these in priority order:

1. Hebbian learning rate (try 0.001, 0.01, 0.05)
2. Error-driven learning rate (try 0.05, 0.1, 0.3)
3. Consolidation interval K (try 50, 100, 250, 500)
4. Quality threshold for buffer entry (try 0.3, 0.5, 0.7)
5. Repulsion strength (try 0.5x, 1x, 2x default)

If sweeping these doesn't yield Tier 2 signal, the problem is structural (likely retrieval signal richness), not parametric.

## Why this phase is the gate

If Phase 3 works, the rest of the architecture is engineering. The hierarchy in Phase 4, the binding discovery and atom splitting in Phase 5, the integration in Phase 6 — all build on Phase 3's load-bearing mechanism. If codebook growth dynamics produce a stable, useful, structured codebook, the architecture has crossed from speculative to viable.

If Phase 3 fails, none of the downstream phases can save it. A broken codebook produces broken hierarchies and broken binding discovery. Better to find out at Phase 3 than to commit to Phases 4-5 on a foundation that doesn't hold.

This is why disproportionate design attention belongs here.
