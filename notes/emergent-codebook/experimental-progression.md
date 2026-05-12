---
date: 2026-05-04
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Experimental Progression

Six phases, each with a clear success gate before moving to the next. Designed so the most uncertain mechanisms get tested early — failures inform redesign, not wasted effort.

> **Updated 2026-05-09** with the project-level evaluation commitment, the Phase 3 headline metric, and the Phase 2 metric additions (cap-coverage, meta-stable-state rate). See `notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md` for the synthesis these came out of.

## Architectural commitment

The system is a **contextual-completion architecture** (pattern completion via Hopfield retrieval), not a sequence-prediction architecture. The progression's evaluation metrics are framed accordingly: retrieval relevance is the primary success signal across phases. "What does this remind the system of?" is the load-bearing question, not "what comes next?"

## Project-level evaluation commitments

*(Added 2026-05-09.)*

Two commitments shape every phase's evaluation, regardless of which metrics it adds:

**Anti-homunculus filter.** Every diagnostic, metric, or actuator in this project must measure or be expressible as an *intrinsic geometric state* — never as an arbitration over states. If a candidate mechanism requires a controller to decide *when* it applies or *what its result means*, it is the wrong shape for this project. This filter rejects controller-creep before it can compound. The filter applies to evaluation as much as to architecture: a metric that requires a supervisor to interpret is doing the same work as a controller, just at evaluation time.

**Headline-vs-drill-down metric structure.** Each phase has one headline metric that defines whether the phase crossed its viability threshold, plus a panel of drill-down metrics that explain *why* the headline moved in unexpected ways. Drill-downs are debugging tools, not competing definitions of success. Multi-metric panels with no headline let projects rationalize any outcome as success — this commitment prevents that.

The headline metric for each phase is named in that phase's section below. Drill-downs vary by phase but live in the per-phase metric specs (`PHASE_N_SPEC.md` in the repo, or the relevant deep-dive note in this folder).

## Phase 1 — Substrate validation

Build the FHRR layer. Verify the math at scale: bind two random vectors, unbind, recover the original within tolerance. Encode a structured pattern (small graph, tree, labeled record) as bundled bindings, retrieve specific roles by unbinding, confirm role-filler recovery.

**Success looks like:** clean recovery of bound structures, no surprises in the algebra at 4096 dimensions, performance characteristics fit the compute budget.

**Why this phase:** small-scale HDC papers don't always tell you what happens when you scale up. Confirm nothing surprising happens at the scale you'll be operating at before building anything on top.

**Estimated time:** 1 week.

## Phase 2 — Static codebook baseline (dual objective)

Use a fixed random codebook (no growth yet). Train the Hopfield on sequences from a small corpus (Brown Corpus or WikiText-2). Test **both** prediction objectives as baselines:

1. **Next-token prediction:** given a sequence prefix, retrieve the next token. The traditional sequence-prediction framing.
2. **Masked-token prediction:** given a sequence with a masked position, retrieve the masked token. The contextual-completion framing.

**Why both:** modern Hopfield networks are associative memories, not transition models. Masked-token is more associative-friendly (pattern completion) while next-token requires sequential generalization (which Hopfield doesn't do natively). Comparing the two objectives is a critical design test, not a hyperparameter sweep item.

**What the comparison tells us:**

- If masked-token >> next-token: confirms the architecture supports pattern-completion-shaped prediction. Phase 3 should use masked-token as primary objective; next-token can stay as a secondary metric.
- If both are comparable: gap was smaller than expected; either objective can work for Phase 3.
- If both are bad: substrate or Hopfield setup has a fundamental problem; design needs revisiting before Phase 3.

**Headline metric:** retrieval accuracy on the test population (masked-token Recall@1 and next-token Recall@1, separately) compared against bigram baseline with non-overlapping 95% confidence intervals across the majority of the experimental matrix conditions.

**Drill-down metrics (added 2026-05-09):** cap-coverage error at θ ∈ {0.3, 0.5, 0.7} and meta-stable-state rate at θ = 0.95. Both derived from the per-retrieval max stored-pattern cosine at convergence, which is now part of the per-retrieval metric set. Cap-coverage distinguishes "wrong basin" from "no basin" — the architecture's contextual-completion thesis predicts these are different failure modes with different implications. Meta-stable-state rate is the literature-standard measurement for whether stored patterns are sufficiently separable (HEN, Kashyap et al. 2024). Full spec in `PHASE_2_SPEC.md`.

**Success looks like:** retrieval quality (Recall@K) meaningfully above random for at least one objective. Random codebooks won't carry semantic information so it won't be great, but it should be clearly above chance. If both objectives are at chance, there's a bug.

**Estimated time:** 1-2 weeks.

## Phase 3 — Codebook growth

Replace fixed codebook with the growth mechanism. Atoms refine through the **two-pathway hybrid update** — Hebbian reinforcement on retrieval success, error-driven update on retrieval failure, gated per-experience by retrieval quality. Bimodality tracking added as Phase 3 hook for future atom-splitting in Phase 5. See `phase-3-deep-dive.md` for full mechanics.

**Headline metric (added 2026-05-09):** Recall@K on masked-token contextual completion, **stratified by regime classification** from the consolidation-geometry diagnostic (`consolidation-geometry-diagnostic.md`), evaluated against the **shuffled-token control**. This is one number plus a controlled comparison plus a stratification axis. It aligns directly with the architecture's core claim: retrieval quality emerges from geometry-conditioned consolidation. The shuffled control rules out corpus-statistical artefacts; the regime stratification distinguishes "the system works in the tight regime where consolidation should be safe" from "the system works generally, including spread-regime atoms."

**Drill-down metrics:** cap-coverage error, meta-stable-state rate, NC1 within-basin variability + separability pair, softmax entropy as feature/prototype mode classifier, bimodality flag rate, regime-classifier distribution. These explain *why* the headline moves; they are not competing definitions of success. Full diagnostic stack and the NC1 reformulation are in `phase-3-deep-dive.md`.

**Success looks like:** codebook stabilizes within a finite training budget, headline metric improves meaningfully over the Phase 2 baseline, distributional structure appears in codebook geometry (similar tokens have similar hypervectors), shuffled-token control fails to produce the same semantic organization.

**Why this phase matters most:** this is where most of the design risk lives. Codebook dynamics are the load-bearing mechanism for the entire architecture. If they don't work, nothing downstream works.

**Estimated time:** 4-6 weeks of iteration.

## Phase 4 — Hierarchical compression

Add a second Hopfield layer. Frequent stable bundles from layer 1 (n-grams, phrases) become atoms at layer 2. Layer 2 does retrieval at higher abstraction — completing whole-phrase patterns, not just token-level patterns.

**Success looks like:** layer 2 atoms emerge that correspond to interpretable units (common phrases, sentence patterns), longer-range retrieval improves, retrieval at layer 2 is faster and more abstract than at layer 1.

**Estimated time:** 1 month.

## Phase 5 — Binding discovery and atom splitting

Two new mechanisms in this phase:

**Bind-versus-bundle discovery.** Replace fixed positional binding with learned bind-versus-bundle. Track positional consistency for atom pairs. Pairs with stable relative position get bound through a learned positional role; pairs with co-occurrence but variable position get bundled.

**Atom splitting.** Atoms flagged as persistently bimodal during Phases 3-4 get split into two atoms initialized from cluster centroids. Stored Hopfield patterns that contained the original atom either get re-encoded or accept transient noise as the new atoms stabilize.

**Success looks like:** the bind-versus-bundle distinction emerges from data rather than being declared. Analogical retrieval starts to work — give the system patterns like "X did Y to Z" and see if it surfaces stored patterns with similar structural shape but different content. Polysemous tokens get represented by appropriately-distinct atoms.

Hardest and least well-specified phase. Could be 2 months, could be 6, depending on how the dynamics shake out.

## Phase 6 — Integration

Plug into existing project architecture. Hybrid hypervector becomes the substrate for existing Hopfield+SDM+consolidation pipeline. LLM integration via Option A (BPE-as-segmentation, codebook-as-meaning, LLM-in-workspace pathway) — see `llm-integration.md`.

**Success looks like:** SONAR-replacement is non-regressive on retrieval tasks, structural reasoning capabilities are present and measurable, the consolidation channel responds to structural content as designed.

### Phase 6 issue: atom drift vs stored patterns

SDM stores patterns that include atom hypervectors at the time of storage. If atoms have drifted, stored patterns become stale relative to the current codebook. Mitigation: **replay-and-re-encode pass** — periodically replay stored patterns through the current codebook to refresh them. Replay was already in the architecture (per the briefing), so this is just applying replay to an additional purpose.

Phase 6 design questions for replay-and-re-encode:

- Frequency: how often does the refresh pass run? Tied to wake/sleep cycles or independent?
- Prioritization: which patterns get refreshed first when refresh capacity is limited? Recency-weighted? Importance-weighted? Drift-magnitude-weighted (patterns whose constituent atoms have drifted most)?
- Granularity: refresh whole patterns or only re-bind specific drifted atoms?

Doesn't need answers for Phase 1; needs to be on the map.

**Estimated time:** 1 month.

## What to test against

- **Headline metric per phase:** named in the phase section above. The single number that decides whether a phase crossed its viability threshold.
- **Drill-down metrics:** the rest of the panel. Used to explain why the headline moved, not to define what success means.
- **Distributional structure:** cosine similarity between hypervectors of token pairs that should be related (synonyms, antonyms, collocations). Pulled from WordNet or hand-curated.
- **Compositional generalization:** SCAN or COGS subsets — explicit tests for novel combinations of seen primitives. Don't expect to crush these; expect meaningful signal that increases with phases 3-5.
- **Structural reasoning:** retrieval-shaped analogical queries (given partial structural pattern, retrieve completions that match). Less load-bearing than retrieval quality; nice-to-have.
- **Emergence:** codebook interpretability — for each atom, what tokens does it cover? Does the structure look learned or random?

## How to know it's actually working

Three signals matter most across phases:

1. The codebook stabilizes without collapsing — atoms converge to consistent hypervectors over training without all merging into one. Tracked as NC1 within-basin variability staying bounded and non-zero (not trending toward zero), paired with growing inter-basin separability. See `phase-3-deep-dive.md` for the reformulation.
2. Similar items develop similar hypervectors without explicit supervision — if "king" and "queen" don't end up geometrically close after Phase 3, something's wrong. Cross-checked against the shuffled-token control: structure should appear in the standard run and be absent (or substantially attenuated) in the shuffled run. If structure appears in both, it's a corpus-statistical artefact, not learned representation.
3. Structural retrieval starts working at Phase 5 — retrieval-shaped analogical queries being above-chance is the clearest "the structural part is doing real work" signal.

## Cross-phase failure modes

- **Codebook collapse:** all atoms drift toward each other, become indistinguishable. Diagnosis (revised 2026-05-09): NC1 within-basin variability trends toward zero AND/OR pairwise similarity histogram narrows toward 1.0. Catastrophic.
- **Codebook explosion:** atoms keep getting allocated without consolidation, codebook grows unbounded. Diagnosis: budget keeps getting hit.
- **Failure to discover binding:** system bundles everything, loses structure. Phase 5 specific.
- **Hopfield interference:** too many stored patterns relative to capacity, retrieval degrades. Modern Hopfield helps but isn't infinite. Cross-checked via meta-stable-state rate spiking (added 2026-05-09).
- **Atom drift staleness:** stored patterns use outdated atom versions, retrieval quality degrades subtly over time. Phase 6 specific; replay-and-re-encode mitigation.

Phase-specific failure modes for the highest-risk phase are detailed in `phase-3-deep-dive.md`.

## Realistic timeline

| Phase | Duration |
|-------|----------|
| 1-2 (combined) | 2-3 weeks (mostly setup and validation) |
| 3 | 4-6 weeks (first real research phase) |
| 4 | 1 month |
| 5 | 2-6 months (depending on bind-discovery and splitting dynamics) |
| 6 | 1 month |

Total: ~6 months of focused work, could be longer. Each phase has an early-fail criterion, which means viability becomes clear within weeks rather than after committing to the full build.

The structure exists to find out fast if something fundamental doesn't work. Phase 2's dual-objective comparison and Phase 3's retrieval-quality dynamics are the most likely places to discover this; the rest of the progression is contingent on those results.
