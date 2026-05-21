# Bookmarked Literature — Phase 5 Graduation Brainstorm

Scan date: 2026-05-20. Anchor doc: `notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md`. Twenty-one PDFs in `research/`, twenty-one extracted texts in `tmp/pdf_text/` (one paper, Spin Glass NN, not extractable).

The Phase 5 blocker the brainstorm needs to break: a role-fidelity measure built as pairwise distance between unbound role vectors goes structurally noise-dominated at D=4096 — the discriminability signal is swamped by HRR cross-talk variance that grows with both binding count and dimension. The flagged-paper sections at the end of this document address that pathology directly.

---

## Group A: Load-bearing for Phase 5 (already operationalized or directly addresses the blocker)

### Ganesan et al., *Learning with Holographic Reduced Representations* (NeurIPS 2021)
Core claim. Naive HRR back-propagation fails because circular-convolution binding and its pseudo-inverse are numerically unstable; the variance of the unbinding response grows with binding count and overwhelms the present/absent gap. They introduce a *complex unit-magnitude projection* π that normalizes every component of F(x) to lie on the complex unit circle. With π, the pseudo-inverse coincides with the true inverse, and binding capacity scales linearly with d as Plate's original theory predicts but the naive scheme does not realize. They bind 1024 vectors in d=256 (4× theoretical capacity ratio) while keeping the unbind response cleanly near 1 (present) or 0 (absent).
Project draw. This is the load-bearing paper for the Phase 5 noise-floor issue. The project's FHRR substrate ostensibly already operates with unit-magnitude phasors in Fourier domain (FHRR = HRR but phase-only), so the projection in the strict Ganesan sense should be free. **The unexploited angle:** their loss formulation is itself a role-fidelity proxy that does not use pairwise distance. They evaluate "is x⊗y present in S?" by checking `cos(x, π(S ⊗ y*)) ≈ 1` and "is x⊗y absent?" by checking the same ≈ 0. The positive and negative responses live on a bounded calibrated scale by construction; the absent baseline is a property of the projection, not an empirical noise estimate. The project is computing pairwise distance of unbinds, which is a noisy way to derive what the cosine response gives in one cheap step. The full eq. (6)-(7) loss (Jp + Jn) also gives a *gradient signal on the codebook* that pushes role separation in the right direction without ever computing pairwise unbind distances.
Underexploited angles.
- The XML loss structure (Jp + Jn) is a recipe for end-to-end role-codebook learning that the project has not adopted. Even if not used as training loss, **Jp - Jn read as a fidelity metric is dimensionless, scale-free, and has a structural ceiling of 1 and floor of -1 — it does not noise-floor with D.**
- They report a *binding-capacity vs dimension* curve in Appendix D that scales linearly. The project's per-pattern Phase 5 experiment should be run against this curve to know whether it sits in the regime Ganesan's projection promises is well-behaved.
- The ablation showing that "skipping the projection step caused degradation to random-guessing performance" in all cases is a control condition the project has not run for its own substrate — i.e., whether the FHRR phasor-renorm is truly equivalent to Ganesan-π under composition with W=4 binding stacks.

### Vangara & Gopinath, *Geometry of Consolidation* (2026)
Core claim. For any consolidator mapping n unit-norm cluster items to m representatives, the identity-retrieval error is bounded by `ε_id ≥ 1 - c₁·m·(θ'/d̄)^(d_eff/2)`, where d_eff is the participation ratio of the cluster's covariance spectrum and d̄ is mean within-cluster cosine distance. The bound predicts a phase boundary at d̄ = θ': below it any consolidator works; above it, errors diverge in an order set by d_eff. Validated on 16,000 synthetic cells and six real corpora.
Project draw. Already operationalized via `emergent-codebook/consolidation-geometry-diagnostic.md`. Provides cap-coverage error as Phase 3 headline metric and θ'(β) calibration protocol.
Underexploited angle. **The same `(θ'/d̄)^(d_eff/2)` form is a candidate alternate role-fidelity metric.** Interpret each role vector's "cluster" as the set of unbinds of that role across W bindings; the role is "intact" iff cap-coverage of the unbind cluster around the true role vector exceeds the cap threshold. This converts role-fidelity from a pairwise-distance question (noise-floors) into a cap-coverage question (regime-aware, scales with d_eff, has the published bound for free). The unbind set is exactly the kind of "few representatives, asked whether they identify the source" the bound is designed for. Worth noting: real-text clusters live in the *tight regime* where centroid wins — the FHRR unbind cluster for W=4 may also be tight, and a centroid-of-unbinds (which is what the soft Hopfield retrieve already computes) is then provably near-optimal.

### Kashyap et al., *Modern Hopfield Networks meet Encoded Neural Representations* (HEN, 2024)
Core claim. MHNs hit "metastable states" — convergence to non-stored mixtures — when stored patterns have weak separability in high-dimensional input space. HEN solves this by encoding inputs through a pre-trained encoder-decoder before storage and decoding on retrieval; pattern separability in latent space is the controllable knob. They demonstrate substantial reduction in metastable rate plus increased usable capacity, and validate hetero-association from text queries to image patterns.
Project draw. Already cited as the architectural validation for codebook-as-encoder. Provides metastable-state-rate as a Phase 3 diagnostic.
Underexploited angle. **HEN's separability framing is the architectural antidote to role-fidelity noise floor.** If the role codebook is treated as the "encoded representation" and the FHRR substrate is the "MHN slab," then driving role-pair cosine separability *toward Ganesan-π's expected baseline* (and measuring how close it is to that structural baseline) is exactly the HEN move. The metastable-state diagnostic transfers to the role-binding case: cue with a true binding, settle, count fraction that converge to a stored role (vs. a mixed mode). This is a *settling-based* role-fidelity measurement that does not use pairwise distance of unbinds.

### Krotov & Hopfield, *Dense Associative Memory for Pattern Recognition* (2016)
Core claim. Higher-order energy functions (`E = -ΣF(ξ·σ)` with F polynomial of degree n) push capacity from `0.14N` to `N^(n-1)/log(N)` because cross-talk variance Σ² grows as `(K-1)N^(n-1)` while the mean energy gap grows as `N^(n-1)`. Error probability is a Gaussian tail with the SNR `mean_gap/σ ∝ √(N^(n-1)/K)`. Beta (the interaction order) controls the feature/prototype regime spectrum, and softmax entropy directly classifies the regime.
Project draw. Names the β temperature regime. Provides interpretive frame for softmax entropy as feature/prototype classifier.
Underexploited angle. **Equation 4-6 in the paper is literally the analytic noise-floor calculation the project needs to run for FHRR role-binding.** Replace "stored patterns ξ" with "unbinds of a stored binding," "K" with "number of co-bound roles W," "N" with effective dimension D, and the same `P_error ≈ √((2n-3)!!/2π) · (K/N^(n-1)) · exp(-N^(n-1)/(2K(2n-3)!!))` form predicts the role-discriminability noise floor as a function of W and D. The pathology the user is hitting (D=4096, role-fidelity goes noise-dominated) should map cleanly onto a specific (n, K, N) cell of this expression. **If the project's effective n is 1 (linear softmax retrieval at low β), the noise floor is structural; if effective n ≥ 2 (β large or kernelized retrieval), the noise floor recedes super-linearly.** This is testable.

### Krotov, *Hierarchical Associative Memory* (HAM, 2021)
Core claim. Lagrangian formulation of multi-layer MHNs with softmax-friendly activations and bottom-up + top-down energy convergence. No need to invert the activation; energy is bounded; the network is fully recurrent with feedback weights equal to feedforward transposes.
Project draw. Worked-out math for Phase 4 hierarchical extension. Audit-fix candidate ("HAM deferred-sync" in recent commits).
Underexploited angle. **Top-down feedback from a higher layer onto the role-binding layer is a natural role-fidelity *actuator* in the diagnostics-as-actuators sense.** If a higher layer encodes "expected binding shape" and the lower layer's energy is the FHRR binding energy, then a role that fails to settle into the higher-layer attractor is by definition low-fidelity — without ever computing pairwise distances. This is a Phase-5-graduates-to-Phase-4 path the project may have under-explored: instead of measuring role fidelity, *fold it into the energy* and read it off the convergence trajectory.

---

## Group B: Adjacent — actuator / consolidation / replay

### Saighi & Rozenberg, *Autonomous Retrieval for Continuous Learning in Associative Memory Networks* (Front. Comput. Neurosci. 2025)
Core claim. Continuous Hopfield Networks plus *self-inhibition* (adaptation term A_i that builds up on visited attractors) enables autonomous, sequential retrieval of all stored patterns without an external memory list. Each retrieved pattern's basin is shrunk via local plastic inhibition; resetting to the neutral state then drives convergence toward an as-yet-unretrieved attractor. The "biased phase" guides the trajectory; an optional "free phase" with inhibition removed completes a clean settle. Spike-frequency adaptation as the biological analogue.
Project draw. Project already running `phase34_saighi_*` experiments and `phase34_integrated_hebbian_*` series.
Underexploited angle. **Self-inhibition as a role-fidelity probe.** In the project's W=4 binding case, the analogue is: cue with one role-binding, settle, then inhibit *that role* in the codebook and re-cue with the same binding. If role fidelity is high, the second settling should land on a different role; if role fidelity is low (role vectors not well-separated in the substrate), the second settling collapses or repeats. This is a *dynamics-based* role-fidelity test, not a distance-based one, and it inherits the project's own anti-homunculus framing because the "measurement" is just the natural evolution under adaptation. Also: the paper emphasizes that the autonomous-retrieval process is robust at low β values — small adaptation steps. This maps onto a slow-timescale replay-pressure actuator the May-9 note explicitly calls out as the next architectural threshold.

### Benna & Fusi, *Computational Principles of Synaptic Memory Consolidation* (Nature Neuroscience 2016)
Core claim. Memory lifetime scales nearly linearly with synapse count N (vs. √N for previous bounded-synapse models) when synapses are *cascades of bidirectionally coupled* slow and fast variables. The bidirectional fast↔slow coupling is the key — slow→fast feedback is not just consolidation, it's what protects initial memory strength. Metaplasticity, spacing effects, and delayed expression of synaptic modifications all follow naturally.
Project draw. Theoretical anchor for Phase 6 multi-timescale dynamics.
Underexploited angles.
- **Role-fidelity is itself multi-timescale.** A role vector that has consolidated to a slow variable should be more separable than one that lives only in fast variables. The "role-fidelity at high D" issue may be that the project is measuring fidelity over the fast (recent-Hebbian) component only, where cross-talk dominates, when the consolidated slow component would show a much cleaner signal. Measure role-fidelity stratified by *which timescale variable* the role lives in.
- **Selection by bidirectional flow, not by frequency.** Most replay/consolidation selectors in the project pipeline use frequency-of-occurrence or recent-error magnitude. Benna-Fusi's selector is implicit in the bidirectional coupling itself — items that survive in the slow variable are those whose fast-variable signal had time to influence slow and whose slow variable hadn't been overwritten. This is selection by *consolidation dynamics*, not selection by counter. Directly addresses the "memory consolidation that uses something other than strength-by-frequency" flag.

### Aljundi et al., *Online Continual Learning with Maximally Interfered Retrieval* (OCL-MIR, NeurIPS 2019)
Core claim. Replay efficacy improves when the replay buffer is sampled by *maximum-interference* — pick the items whose loss will most increase given the foreseen parameter update — rather than uniformly. Works in both stored-replay and generative-replay regimes.
Project draw. Phase 6 replay-buffer selection.
Underexploited angles.
- **MIR as a role-fidelity selector.** When constructing the Phase 5 evaluation set, sample bindings/roles that are *currently maximally interfered* by recent consolidation. If role-fidelity holds under MIR sampling, it holds in general; if it collapses, the noise floor was hiding behind random sampling. This is also a stronger control condition than shuffled-token.
- The MIR signal is *self-supervised* — it does not require labels or task identity. Aligns with the codebook's unsupervised nature.

### Sun et al., *Information-Theoretic Online Memory Selection* (InfoRS, ICLR 2022)
Core claim. Online memory selection should pick points by a combination of *surprise* (negative log conditional probability given current memory) and *learnability* (how much the model improves on the point itself after absorbing it). Stochastic InfoRS modifies reservoir sampling to retain only points above an information threshold, robust to data imbalance. Bayesian linear model with rank-one updates makes the criterion cheap.
Project draw. Direct candidate for an information-theoretic alternative to strength-by-frequency consolidation selection.
Underexploited angles.
- **Surprise + learnability is exactly the alternative consolidation selector the brainstorm asks for.** Frequency selectors prefer items already represented; surprise prefers novel; learnability rejects pure noise. The 2-knob combination is the right shape for the project's anti-homunculus filter — neither knob *decides*; they're both local geometric properties of the cluster.
- The rank-one Bayesian update is GPU-cheap and would slot into the consolidation pass without breaking the FHRR substrate.

### Hayes et al. (Kanan), *Replay in Deep Learning: Current Approaches and Missing Biological Elements* (2021)
Core claim. A survey contrasting biological replay (selective, partial, multi-region, reward-modulated, NREM vs REM, spontaneous, temporally structured) against deep-learning replay (mostly veridical, single-layer, random-sampled). Identifies seven distinct biological replay properties largely missing from artificial systems.
Project draw. Phase 6 replay design vocabulary.
Underexploited angles.
- **Partial-experience replay.** Biology replays *parts* of episodes, enabling abstraction and integration. The project replays bindings whole. A natural Phase 5/6 design: replay only the *role* component or only the *filler* component of a stored binding, and measure whether role-fidelity is improved by role-only replay specifically. This is testable and theoretically motivated by the paper.
- **NREM vs REM separation.** NREM consolidates known patterns; REM recombines them. The Phase 5 role-fidelity vs Phase 6 abstraction-from-replay split may map cleanly onto this dichotomy, with consequences for what kind of replay each phase uses.
- **Reward modulation.** A natural anti-homunculus way to weight replay: tension/error gradient on each binding *is* the reward signal. Already on the road map under the actuator-dynamic-form open question.

### Alonso & Krichmar, *Sparse Quantized Hopfield Network* (Nature Communications 2024)
Core claim. SQHN combines sparse quantized neural codes with local MAP-learning and neuro-genesis (new units grown on demand) for online-continual associative memory. Outperforms SoTA on noisy-encoding and episodic-memory tasks while learning purely with local rules.
Project draw. Closest extant architecture; informs Phase 3+ evaluation task formats.
Underexploited angles.
- **Neuro-genesis as a role-fidelity actuator.** If role discriminability falls below threshold during operation, grow a new role unit rather than re-tuning existing ones. Anti-homunculus check: who decides? — the local cap-coverage gradient does (Vangara-Gopinath: low cap-coverage → restructuring pressure). Neuro-genesis is the *dynamic form* of "restructuring pressure" the May-9 note flags as an open question.
- **Quantized codes for roles.** The project's roles live in continuous FHRR phasor space; SQHN argues quantization is what makes pattern completion robust with local rules. Worth testing whether *quantizing the role codebook* (while leaving the filler / substrate continuous) improves role-fidelity at high D.

---

## Group C: Foundations / interpretive

### Sharma, Chandra & Fiete, *MESH* (ICML 2022)
Core claim. A CAM continuum with no memory cliff is achievable by factorizing memory into (a) a fixed pre-stabilized scaffold network and (b) heteroassociation between scaffold states and arbitrary external patterns. Saturates the O(N²) information bound at every operating point.
Project draw. Phase 6 benchmark target (project's scaffold grows from experience; MESH's is fixed).
Underexploited angle. **Pre-stabilized scaffold as a role-codebook bootstrap.** Phase 5's role-fidelity problem may be partly that roles are co-trained with everything else. MESH's recipe says: predefine a well-separated dictionary first, *then* heteroassociate downstream patterns. The project could test a Phase 5 variant where roles are initialized as a fixed MESH-style scaffold (random binary or simplex ETF) and not learned — does role-fidelity at D=4096 become a non-issue if roles are *pre-orthogonalized*? This is a clean ablation against the current emergent-roles approach.

### Papyan et al., *Neural Collapse* (PNAS 2020)
Core claim. In the terminal phase of training, last-layer activations collapse to class means (NC1), class means collapse to a simplex equiangular tight frame (NC2), classifier weights match the means (NC3 self-duality), and decision is nearest-class-center (NC4).
Project draw. Names codebook-collapse failure mode. NC1 reformulated to bounded-non-zero in May-9 note.
Underexploited angle. **NC2 (simplex equiangular tight frame) is the maximally-separated role codebook the project's Phase 5 needs.** If roles are initialized at an ETF and held fixed during binding-substrate training, role pairwise cosine is exactly `-1/(K-1)` — a structural constant, not a learned-and-noisy quantity. Role fidelity at high D is then a question about whether unbinds preserve the ETF, which is a *deterministic geometric* test, not a noisy distance test. (Compare to MESH scaffold above; ETF is the specific geometric form the project could use.)

### Hopfield 1982
Core claim. Founding paper for associative-memory-as-attractor-dynamics; the energy descent interpretation; the `0.14N` capacity result for random patterns under outer-product learning.
Project draw. Foundational.
Underexploited angle. None obvious for Phase 5; cited for completeness.

### Dawid & LeCun, *Latent Variable Energy-Based Models* (Les Houches 2023)
Core claim. EBM lens on autonomous machine intelligence; H-JEPA building block; contrastive vs architectural/regularized training.
Project draw. Background.
Underexploited angle. **The role-binding operator as a latent variable.** In an LV-EBM framing, the role is a latent that gets optimized jointly with the energy. "Role-fidelity" reduces to "how peaked is the posterior over roles given the binding" — an entropy/sharpness measurement on the posterior, not a pairwise distance. Differentiable and gradient-trainable. Worth sketching this framing to see if it gives the project a metric that does not depend on D.

### LeCun et al., *A Tutorial on Energy-Based Learning* (2006)
Core claim. Foundational EBM. Loss functionals; contrastive vs non-contrastive; the role of margins.
Project draw. Background.
Underexploited angle. The *margin loss* framing (push down on data, push up on contrastive points) is a direct prescription for role-codebook training: push down on `cos(unbind(S,role_true), role_true)`, push up on `cos(unbind(S,role_true), role_other)`. Exactly Ganesan's Jp + Jn at a higher level of abstraction.

### LeCun, *Autonomous Machine Intelligence* (2022 position paper)
Core claim. Architecture for autonomous intelligence around H-JEPA + world model + cost module + actor + perception, with the world model trained by JEPA-style embedding-space prediction.
Project draw. Background; influences the brainstorm's cognitive-architecture framing.
Underexploited angle. The "configurator" in the AMI proposal is essentially an anti-homunculus failure point the May-9 note already warns about. Useful as a *negative* example: the project's actuator-dynamic-form work is precisely the move that prevents the architecture from drifting toward an AMI-style configurator.

---

## Group D: New since 2026-05-09 (PAM family)

### Dury, *Predictive Associative Memory* (2026, arXiv 2602.11322)
Core claim. Replaces similarity-based retrieval with *temporal-co-occurrence-based* retrieval via a JEPA-style "Inward" predictor trained on temporal pairs from a continuous experience stream. The predictor's forward pass *is* memory retrieval: given state s(t), return the region of embedding space containing states from the temporal neighborhood of s(t). Validated on a synthetic benchmark: Association P@1 = 0.97, cross-boundary Recall@20 = 0.42 where cosine scores zero, AUC = 0.916 vs cosine 0.789. Specificity controls (temporal shuffle) collapse cross-boundary recall by 90%.
Project draw. Not yet integrated. New since May 9.
Underexploited angles.
- **PAM's "association ≠ similarity" is the closest thing in the bookmarked literature to a non-distance fidelity measure.** A role is "faithful" not because its unbind is *close to* the true role, but because it *reliably predicts* the surrounding bindings — the same shape as PAM's temporal-association criterion. Concretely: given a stored W=4 binding, treat the W roles as a "temporal window"; train (or simply measure) a predictor that, from any one unbound role, predicts the rest. Predictor accuracy is a role-fidelity score that does *not* go through pairwise distance. This is potentially the most direct answer to the brainstorm's "alternative ways to measure binding fidelity besides pairwise distance of unbinds" prompt.
- **Inward + Outward channel split** maps onto the project's actuator/diagnostic distinction at a higher level. The Outward channel is similarity-based (similarity = local geometry diagnostic). The Inward channel is association-based (association = consolidation actuator). Cross-pollinates with the diagnostic-as-actuator threshold work.
- The temporal-shuffle control is exactly the kind of control the project's headline-metric principle requires.

### Dury, *From Topic to Transition Structure* (2026, arXiv 2603.18420)
Core claim. Same PAM machinery scaled to corpus level (29.4M params, 373M co-occurrence pairs from 9,766 Gutenberg texts). Under capacity constraint (only 42.75% training accuracy), the model cannot memorize the full set and is forced to compress across recurring transition-structure patterns. Result: concept clusters that capture *narrative function* ("direct confrontation," "lyrical meditation," scene templates) rather than topical content, validated against BGE embedding clustering and unseen-novel transfer.
Project draw. Not yet integrated.
Underexploited angles.
- **Capacity-constrained contrastive training as a consolidation mechanism.** Dury explicitly calls this hippocampal-replay-like consolidation. The 42.75% training-accuracy ceiling functions as a *forced abstraction pressure* — exactly the kind of dynamic-form actuator the May-9 note asks for ("how does abstraction fire without a controller deciding to abstract?"). Answer: by capacity bottleneck.
- **Concept discovery from co-occurrence is an alternative role-codebook acquisition path.** Roles in the project are currently learned from binding-task gradients. The Dury route is: extract roles from the *co-occurrence structure* of substrate states across many bindings. Bypasses the binding gradient entirely. May or may not work, but is the most divergent codebook-acquisition path in the bookmarked literature.

### LLM-JEPA (Huang, LeCun, Balestriero 2025)
Core claim. Adds a JEPA-style embedding-space prediction objective on top of LLM training. Outperforms baseline on NL-RX, GSM8K, Spider, RottenTomatoes across Llama3, Gemma2, OpenELM, Olmo families. Robust to overfitting.
Project draw. Background — JEPA validated for language.
Underexploited angle. **JEPA as a substrate-level training signal for roles.** If two bindings share a role, their unbinds along that role should be predictable from each other (JEPA-style). Use that predictability as the training loss for the role codebook — bypasses the noisy unbind-distance signal entirely. Conceptually parallel to PAM but at the role level rather than the temporal level.

---

## Group E: Could not be extracted

### Spin Glass NN
PDF is image-only, OCR not yet run. The May-9 note guesses it adds nothing actionable beyond Hopfield 1982 + Krotov-Hopfield 2016. Probably correct — defer.

---

# Flagged Findings (Direct Answers to Brainstorm Prompts)

### A. Papers that explicitly address the high-D noise floor in VSA / HRR

1. **Ganesan et al. 2021** is *the* paper. The complex unit-magnitude projection π is engineered specifically to keep the unbind-response variance constant as binding count scales. Their Fig. 1 shows the failure mode the user is hitting (response distributions blow past ±1 with naive HRR even at d=256). FHRR should already include this; verify it does for the project's substrate. The bound-capacity-vs-dimension curve in Appendix D is the calibration the project needs to know whether D=4096 is in the linear-scaling regime.
2. **Krotov & Hopfield 2016** gives the analytic SNR formula for *any* Hopfield-like retrieval (which includes role-unbind in FHRR). The `P_error ≈ √(K/N^(n-1)) · exp(-N^(n-1)/(2K(2n-3)!!))` form predicts exactly when role-fidelity goes noise-dominated as a function of co-bound role count K, dimension N, and interaction order n. The user's D=4096 noise floor is a specific cell of this expression; n=1 (linear retrieval) is the worst case and probably what's happening.
3. **Vangara & Gopinath 2026** gives a *consolidation*-side bound: the `(θ'/d̄)^(d_eff/2)` form predicts cap-coverage error as a function of effective dimension. If the role-unbind cluster's d_eff is much smaller than D=4096, the bound says the cluster is in the tight regime regardless of nominal dimension. Calibrate d_eff for the unbind cluster — that's the dimension that actually matters, not D.
4. **HEN (Kashyap 2024)** addresses the noise floor architecturally rather than analytically: improve pattern separability in latent space and the metastable-state rate collapses. The recipe for the project's role codebook follows directly.

### B. Alternative ways to measure binding fidelity besides pairwise distance of unbinds

1. **Ganesan-style cosine-against-projection-baseline** (`cos(role_unbind, role_true)` evaluated against the structural π baseline of 0 for absent and 1 for present, *not* against an empirical noise estimate). Bounded by construction; doesn't noise-floor.
2. **Cap-coverage of unbind cluster** (Vangara-Gopinath). Does the cap around the true role contain the unbind cluster's center? Regime-aware; has a published bound.
3. **Metastable-state rate under settling** (HEN, Krotov-HAM). Cue with the binding, settle in the substrate, count fraction that converge to the true role (vs. mixed mode or wrong role). Dynamics-based, not distance-based.
4. **Self-inhibition autonomous retrieval** (Saighi). Inhibit one role, re-cue, check whether the second settling lands on a different and correct role. Tests separability via dynamics; no distance computation.
5. **PAM-style prediction-from-co-roles** (Dury 2602). From one unbound role, predict the other three roles in the binding. Predictor accuracy is the fidelity score. Bypasses the unbind-distance pathway entirely.
6. **JEPA-on-roles** (LLM-JEPA pattern). Predictability between unbinds of the same role across different bindings. Bounded by construction; trains the codebook as a side-effect.
7. **LV-EBM posterior sharpness** (Dawid-LeCun). Entropy of the role-posterior given the binding. Differentiable; D-independent in principle.
8. **HAM top-down agreement** (Krotov 2021). Does the higher layer's expected binding-shape settle? If yes, all roles are faithful by construction.

### C. Memory consolidation selectors other than strength-by-frequency

1. **Surprise + learnability** (Sun et al., InfoRS 2022). Two-knob, anti-homunculus-clean, GPU-cheap. Direct drop-in.
2. **Maximally interfered retrieval** (Aljundi et al., MIR 2019). Pick items whose loss will most rise under the foreseen update. Self-supervised; aligns with codebook geometry.
3. **Bidirectional fast↔slow cascade** (Benna-Fusi 2016). Selection is implicit in coupling dynamics; no explicit selector. Nearly-linear capacity scaling vs √N for selector-based approaches.
4. **Self-inhibition / spike-frequency adaptation** (Saighi 2025). Recently visited patterns are inhibited; selection emerges as "what's left." Anti-homunculus by construction.
5. **Cap-coverage failure as gradient pressure** (Vangara-Gopinath, May-9 note actuator pair). Low cap-coverage raises consolidation gradient locally; restructuring is the response of the same gradient. The May-9 note already names this; not yet implemented.
6. **Capacity-constrained contrastive training** (Dury 2603). Consolidation as forced abstraction under a representation-capacity ceiling. The bottleneck *is* the selector.
7. **Reward-modulated replay** (Kanan replay survey). Use binding-error or tension as the implicit reward weight on each replayed item. Same shape as MIR but via a different signal.

---

# Quick Reference: Which Papers Speak to Which Phase-5 Knob

| Knob | Best 2-3 papers |
|---|---|
| Why D=4096 noise-floors | Ganesan, Krotov-Hopfield 2016, Vangara-Gopinath |
| Alternative role-fidelity metric | Dury 2602 (PAM), Ganesan (Jp/Jn), Vangara-Gopinath (cap-coverage of unbinds) |
| Architectural fix for noise floor | HEN (encode/decode), MESH (pre-orthogonal scaffold), Papyan NC2 (ETF init) |
| Consolidation actuator (not freq) | InfoRS, MIR, Benna-Fusi, Saighi |
| Diagnostic-as-actuator dynamic form | Saighi (self-inhibition), Krotov HAM (top-down), Benna-Fusi (bidirectional cascade) |
| Codebook acquisition without binding-gradient | Dury 2603 (co-occurrence + capacity bottleneck), LLM-JEPA (JEPA-on-roles), SQHN (neuro-genesis) |
