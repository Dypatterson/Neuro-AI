---
date: 2026-05-09
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Research Papers, Diagnostic Stack, and the Actuator-Dynamics Threshold

## Entry Point

Twelfth working session. Dylan surfaced thirteen research papers found between sessions and asked for a synthesis against the project. The session ran a literature review pass, was cross-checked against an external review by GPT, then narrowed to two conceptual moves the cross-talk surfaced: anti-homunculus as a practical design filter, and the diagnostics-to-actuators threshold. Both moves emerged from the literature review but are project-level commitments that outlast it.

The papers themselves are catalogued in this note rather than getting individual notes; the conceptual moves are the load-bearing output.

## Papers Reviewed

Thirteen papers, ranked by current load-bearing relevance. One could not be extracted (Spin glass NN — scanned PDF, OCR not yet attempted).

| Paper | Role in the project |
|---|---|
| Vangara & Gopinath, *Geometry of Consolidation* (2026) | Already operationalized via `emergent-codebook/consolidation-geometry-diagnostic.md`. Phase 3+ regime classification. |
| Kashyap et al., *Modern Hopfield Networks meet Encoded Neural Representations* (HEN, 2024) | Validates the codebook-as-encoder thesis architecturally. Direct ablation template for the Phase 2→3 comparison. Meta-stable-state diagnostic methodology. |
| Alonso & Krichmar, *Sparse Quantized Hopfield Network* (SQHN, Nature Communications 2024) | Closest extant architecture. Online-continual associative memory with neuro-genesis and local learning rules. Two task formats (noisy encoding, episodic memory) directly applicable to Phase 3+ evaluation. |
| Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition* (2016) | Names the feature/prototype regime spectrum. Provides theoretical interpretation of the β temperature parameter and the softmax-entropy diagnostic. |
| Papyan, Han & Donoho, *Neural Collapse* (2020) | Names the codebook-collapse failure mode. Provides NC1 within-cluster variability as a quantitative diagnostic. Reinterpreted for the unsupervised case (see NC1 Reformulation below). |
| Krotov, *Hierarchical Associative Memory* (2021) | Worked-out math for Phase 4. Lagrangian formulation that handles softmax (no inverse-activation requirement). Bottom-up + top-down energy convergence. |
| Sharma, Chandra & Fiete, *MESH* (2022) | Independent convergence on a tripartite scaffold + heteroassociation + content structure. Demonstrates a CAM continuum without memory cliff. Phase 6 benchmark target rather than a corollary, since their scaffold is predefined and the project's grows from experience. |
| Benna & Fusi, *Computational principles of synaptic memory consolidation* (2016) | Theoretical anchor for Phase 6 multi-timescale dynamics. Bidirectional fast↔slow coupling produces near-linear (not √N) capacity scaling. Predicts that slow→fast feedback is necessary, not just fast→slow consolidation. |
| Aljundi et al., *Online Continual Learning with Maximally Interfered Retrieval* (OCL-MIR, 2019) | Phase 6 replay buffer selection. Maximally-interfered retrieval is provably better than random sampling for continual learning. |
| Dawid & LeCun, *Latent Variable Energy-Based Models* (2023) | Background. EBM lens; mostly already in the reading queue. |
| LeCun et al., *A Tutorial on Energy-Based Learning* (2006) | Background. Foundational EBM material. |
| Hopfield 1982 | Background. Foundational. Nothing new architecturally. |
| Spin glass NN | Could not extract — scanned PDF. OCR pending if Dylan wants it covered. |

The Geometry of Consolidation paper had been priority and is already integrated. The remaining papers add value primarily as diagnostics — none of them produced a substrate change or a new architectural component. The stress test of "what would actually change in the build because of these papers?" yielded a stack of measurements, not a stack of mechanisms. That stack is below.

## The Phase 3 Diagnostic Stack

Five additions to Phase 2/3 evaluation, each measuring an intrinsic geometric state and each promotable from "passive observation" to "endogenous regulation" once the actuator-dynamics threshold is crossed (see below).

### 1. Meta-stable-state rate

From HEN methodology. Cue with stored patterns, settle, count the fraction that don't converge to any stored pattern. This is a literature-standard measurement that the project's current per-retrieval metric set does not include directly. The Session 2 finding that the K=2 symmetric mixed state is "unstable to float32 numerical asymmetry" is the discrete version of this paper's continuous-space observation — consistent with literature, not anomalous.

Promote to Phase 2/3 core metric.

### 2. Cap-coverage error

From Geometry of Consolidation, Section 8. Cap-coverage = "does the cap around any representative cover the source?" Identity = "does the right representative fire?" These can differ by 10× on the same data. Identity collapses the retrieval neighborhood to a single point and reports yes/no on it; cap-coverage measures whether the neighborhood has the right shape. The architecture's premise is that meaning lives in the retrieval *neighborhood*, not in exact-token recovery, so cap-coverage is the more architecturally appropriate scoring.

Promote to Phase 2/3 core metric. This is the metric most aligned with the architecture's actual claim about where meaning lives.

### 3. NC1-style within-cluster variability

From Papyan et al. Tracks how much variability remains in settled states when retrieving the same stored pattern. NC1 → 0 means basins have collapsed to points and the blended-retrieval property the architecture relies on is gone. See NC1 Reformulation section below for the unsupervised-case interpretation.

Add as a collapse diagnostic. Track jointly with a separability metric — they're a pair, not two independent metrics.

### 4. Softmax entropy as feature/prototype mode classifier

From Krotov & Hopfield 2016. The interaction-order parameter (project's β) controls a regime spectrum from feature-matching (low β, blended retrieval, multiple memories contribute) to prototype-matching (high β, single memory dominates). Softmax entropy on the retrieval distribution at convergence directly classifies which mode each retrieval is operating in. Already computed as part of the per-retrieval metric set; this paper supplies the theoretical interpretation.

Already present; the contribution here is the principled reading, not a new measurement.

### 5. Empirical θ′(β) calibration

From Geometry of Consolidation E1 protocol. The `consolidation-geometry-diagnostic.md` file uses θ′ ≈ 1/β as a starting approximation for the regime classification. Running the paper's controlled (d_eff, d̄, θ) synthetic-grid protocol on the project's FHRR substrate would replace the approximation with a calibrated mapping. Two days of work, not blocking, but worth doing before relying on the regime classifier for any decision.

Pre-Phase-3 calibration. Not blocking.

## The Headline Metric Principle

Five diagnostics is the right number for Phase 3 development; it is the wrong number for Phase 3 *judgment*. Multi-metric panels permit narrative flexibility — when one metric improves and another worsens, any outcome can be rationalized as success. Pick one metric as the headline signal that defines whether Phase 3 crossed the viability threshold; treat the others as conditional drill-downs that explain why the headline moved.

Headline metric for Phase 3:

> **Recall@K on masked-token contextual completion, stratified by regime classification, evaluated against the shuffled-token control.**

This is one number plus a controlled comparison. Aligns directly with the architecture's core claim: retrieval quality emerges from geometry-conditioned consolidation. The shuffled-token control rules out corpus-statistical artefacts; the regime stratification distinguishes between "the system works in the easy regime where it geometrically should" and "the system works generally."

Cap-coverage, NC1, meta-stable-state rate, softmax entropy, and bimodality flag rate become explanatory tools that get inspected when the headline moves unexpectedly. They are not competing definitions of success.

This is a discipline commitment, not a science commitment. The project is complex enough now that without it, results from later phases will get talked into ambiguity at the end of each phase.

## Anti-Homunculus as a Practical Design Filter

The anti-homunculus principle from earlier sessions has been a description of how the architecture's pieces fit together. The cross-review surfaced its sharper form: a hard rule for the project.

> Every proposed addition to the architecture must either be a local geometric dynamic or be expressible as a measurement of one — never an arbitration over them. If a candidate mechanism cannot pass that test, it gets rejected or reframed.

The failure mode this filter prevents is the standard one in long-running architectural projects: complexity drift toward supervisory routing under pressure of instability. The easiest move when something doesn't work is "let's add a thing that decides what to do." Each individual addition is small; the cumulative drift is large enough that the architecture eventually becomes someone else's project.

The diagnostic stack passes this filter cleanly. None of the metrics requires a controller deciding *when* to apply or *what they mean*. They are properties of settled landscapes that anyone (or no one) could read. This is exactly "decisions distribute into geometry" applied at the diagnostic layer.

This filter is now a project commitment, not a session insight. Future sessions should reject candidate mechanisms that fail it without further consideration.

## NC1 Reformulation

The supervised-case framing of neural collapse in Papyan et al. does not transfer directly to the unsupervised codebook setting, but the measurement does. The reformulation:

> **Maintain bounded non-zero within-basin variability while preserving inter-basin separability.**

The basin should have shape, not collapse to a point. Some compression toward a basin's centre is exactly what the system wants — that's the Hebbian pathway working as designed. Total collapse to a point removes the blended-retrieval property the architecture's "creativity through retrieval" claim depends on. Soft overlap between basins is part of what enables associative blending, ambiguity retention, and low-β deliberation dynamics.

The diagnostic flags collapse when NC1 trends *toward* zero across consolidation events. Small stable values are correct, not a problem. NC1 should be tracked jointly with an inter-basin separability metric (minimum pairwise basin distance, or generalized NC2 — distance from uniform separation). The pair tells the full story:

| NC1 trajectory | Separability trajectory | Interpretation |
|---|---|---|
| Stable, non-zero | Growing | Healthy — basins forming with shape, getting more distinguishable |
| Trending to zero | Growing | Failure — basins collapsing to points, classification-style |
| Stable | Stable | Frozen codebook, refinement not firing |
| Stable | Shrinking | Codebook converging to single point — catastrophic collapse |

Drop the looser "avoid NC1" framing. Use the bounded-non-zero reformulation.

## Diagnostics vs. Actuators: The Conceptual Threshold

The sharpest thing surfaced by the cross-review was the next conceptual threshold for the project: the move from geometry-as-observation to geometry-as-endogenous-regulation. Several diagnostics naturally pair with architectural responses:

- high drift ~ replay pressure
- high spread ~ reduced consolidation
- bimodality ~ splitting pressure
- metastability ~ replay prioritization
- low cap coverage ~ restructuring pressure

This is real and important. It is also the place where controller-creep would actually re-enter the architecture under cover of a framing change. The grammar matters.

### Two readings of the same arrows

The standard reading: condition X *causes* response Y. This grammar requires a thing that detects X and triggers Y. That thing is a controller in disguise — even a tiny rule-based one. The arrows look local but the agent that runs the rules isn't.

The non-controller reading: condition X and response Y are the same physical event, viewed at different timescales. The "diagnostic" measures where the process is right now; the "actuator" is the process running. There is no transition from one to the other — there is a re-description of the same dynamic at different temporal resolutions.

Concretely, for each pair:

| Standard reading | Non-controller reading |
|---|---|
| high drift → replay pressure | drift contributes to a replay-tension energy quantity that drives replay when it crosses threshold |
| high spread → reduced consolidation | the consolidation gate threshold rises with d̄, so spread atoms naturally get smaller updates without anyone reading d̄ |
| bimodality → splitting pressure | persistent bimodality contributes to a splitting-tension quantity; splitting fires when energy exceeds threshold |
| metastability → replay prioritization | replay buffer is energy-ranked; metastable trajectories carry higher energy by construction |
| low cap-coverage → restructuring pressure | cap-coverage failure raises the local error gradient on consolidation; restructuring is the response of the same gradient |

The right-hand column has nothing that "decides." The left-hand column reads as if there's a thing watching the metric and triggering a response. Subtle but load-bearing — the difference between "the architecture has a controller" and "the architecture doesn't" lives in this distinction.

### The general principle

An actuator is a slow-timescale dynamic that some diagnostic happens to be a fast-timescale snapshot of. Diagnostic and actuator are the same physical process viewed at different temporal resolutions.

This generalizes the temperature-as-loop-mode resolution from session 7. There, "who sets temperature?" dissolved into "temperature is the geometric property of the query, not a parameter set by a controller." Here, "who triggers the response?" dissolves into "the response is the dynamic the metric snapshots." Same shape, different surface.

### What this implies for next session

If the project crosses this threshold the way the standard reading suggests — adding rules that read metrics and trigger responses — it imports a small controller into the architecture under cover of a framing change. The anti-homunculus filter would catch this if applied consistently, but the temptation will be strongest exactly here, because the standard reading is much easier to write down and code.

The session that crosses this threshold safely needs to do two things:

1. For each diagnostic-actuator pair, identify the slow-timescale dynamic it is a snapshot of.
2. Specify the energy-like quantity each diagnostic contributes to, such that the response is the natural evolution of that quantity, not a triggered action on it.

This is a phenomenology question first, an engineering question second. It is adjacent to but distinct from the meta-loop mechanics work — that work was about the dynamics of *retrieval*; this would be about the dynamics of *consolidation regulation*. The two are connected but not the same.

Importantly, this session happens *before* Phase 3 implementation, not during it. If Phase 3 begins measuring diagnostics and the dynamic-form is unspecified, the gravitational pull when something goes wrong will be toward rule-based fixes ("if NC1 starts dropping, increase repulsion strength"). The dynamic form needs to already exist so the response is automatic and local rather than ad-hoc and supervisory.

## What This Session Resolves

- **Phase 3 diagnostic stack**: five concrete additions identified, two promoted to core (cap-coverage, meta-stable-state rate), one calibration item flagged pre-Phase-3, two interpretive frames for existing measurements (softmax entropy, NC1 reformulation).
- **Headline metric**: Recall@K on masked-token contextual completion, stratified by regime classification, against shuffled-token control. The other diagnostics are drill-downs.
- **Anti-homunculus principle**: upgraded from architectural description to practical design filter. Hard rule for evaluating future additions.
- **NC1 reformulation**: bounded non-zero within-basin variability + preserved inter-basin separability. Tracked as a pair, not independently.

## What Remains Open

1. **Diagnostic-actuator dynamic forms** for each of the five pairs. This is the next major architectural threshold. Highest priority for next session if Dylan wants to take it on.
2. **Empirical θ′(β) calibration** via the Geometry of Consolidation E1 protocol on FHRR substrate. Not blocking, two days of work, refines the regime classifier.
3. **Anisotropic refinement of the Geometry-of-Consolidation bound** as a possible later contribution. The trajectory-trace mechanism (2026-05-02 note) records the path information their isotropic bound discards. Park as "if this ever becomes a paper, here's the hook." Not load-bearing.
4. **Spin glass NN paper coverage** if Dylan wants it — requires OCR. Probably foundational stat-mech material that adds nothing actionable beyond Hopfield 1982 + Krotov-Hopfield 2016.

## Next Session Candidates

1. **Diagnostic-actuator dynamic form**, working through two or three of the most architecturally consequential pairs (probably replay tension and consolidation gating, since those touch the most components). Same shape as the temperature-as-loop-mode session — work each pair until the dynamic is specified rather than the rule.
2. **Empirical θ′(β) calibration** as a Phase-2.5 spike. Concrete, scoped, refines an existing component. Lower architectural ambition but high practical leverage.
3. **Phase 3 evaluation spine** consolidation. Take the headline metric + drill-down structure decided here and write it into `experimental-progression.md` as a single coherent section. Lowest ambition, highest tidiness.

Recommendation: option 1. It is the natural continuation of today's work and the place where the anti-homunculus filter is most likely to be needed. Options 2 and 3 are useful but can wait or be done quickly outside a working session.
