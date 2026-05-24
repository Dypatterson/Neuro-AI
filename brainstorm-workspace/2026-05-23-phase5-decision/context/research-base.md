# Research Base Catalogue — Phase 5 Decision Point (D=4096 Substrate Saturation)

**Compiled:** 2026-05-23  
**Context:** Phase 5 substrate saturation finding (role-fidelity measurements noise-floor at D=4096). This document catalogues the papers the project has read, the architectural themes extracted from cross-paper synthesis, and the open questions that synthesis has flagged but not closed.

---

## I. Papers Catalogued

### Core Load-Bearing Papers (Already Operationalized)

| Paper | Authors | Year | Role in Project |
|---|---|---|---|
| **Geometry of Consolidation** | Vangara & Gopinath | 2026 | Operationalized via `consolidation-geometry-diagnostic.md`. Provides cap-coverage error metric and θ′(β) calibration protocol. Bounds identity-retrieval error as `ε_id ≥ 1 - c₁·m·(θ'/d̄)^(d_eff/2)` — regime-dependent phase boundary. *Underexploited:* same bound form is candidate role-fidelity metric (cap-coverage of unbind cluster). |
| **Modern Hopfield Networks meet Encoded Neural Representations (HEN)** | Kashyap et al. | 2024 | Validates codebook-as-encoder thesis. Direct ablation template for Phase 2→3 comparison. Provides metastable-state diagnostic. *Underexploited:* HEN's separability framing is the architectural antidote to role-fidelity noise floor — treat role codebook as encoded representation, drive role-pair cosine separability toward structural baseline. |
| **Dense Associative Memory for Pattern Recognition** | Krotov & Hopfield | 2016 | Names the β regime spectrum (feature vs. prototype). Softmax entropy classifier. Provides analytic noise-floor formula: `P_error ≈ √((2n-3)!!/2π) · (K/N^(n-1)) · exp(-N^(n-1)/(2K(2n-3)!!))` — *directly applicable to role-binding noise floor*. At n=1 (linear retrieval, low β), noise floor is structural; at n≥2, recedes super-linearly. |
| **Learning with Holographic Reduced Representations** | Ganesan et al. | 2021 | Addresses high-D noise floor in VSA/HRR. Complex unit-magnitude projection π keeps unbind-response variance constant as binding count scales. Introduces Jp + Jn loss (dimensionless, scale-free role-fidelity metric with structural ceiling 1, floor -1 — *does not noise-floor with D*). Binding-capacity-vs-dimension curve in Appendix D is the calibration curve Phase 5 needs. |
| **Hierarchical Associative Memory (HAM)** | Krotov | 2021 | Worked-out Lagrangian math for Phase 4. Bottom-up + top-down energy convergence. *Underexploited:* top-down feedback as a role-fidelity actuator — higher layer's expected binding shape folds role fidelity into energy without computing pairwise distances. |
| **Neural Collapse** | Papyan et al. | 2020 | Names codebook-collapse failure mode. NC1 within-cluster variability as quantitative diagnostic. Reformulated for unsupervised case (bounded non-zero with preserved inter-basin separability). *Underexploited:* NC2 (simplex equiangular tight frame) is the maximally-separated role codebook Phase 5 needs — if roles initialized at ETF and held fixed, pairwise cosine is structural constant `-1/(K-1)`, not noisy learned quantity. |
| **Computational Principles of Synaptic Memory Consolidation** | Benna & Fusi | 2016 | Theoretical anchor for Phase 6 multi-timescale dynamics. Bidirectional fast↔slow coupling produces near-linear (not √N) capacity scaling. Slow→fast feedback is necessary, not just consolidation. *Underexploited:* role-fidelity is multi-timescale; fidelity stratified by which timescale variable yields cleaner signal. Selection by bidirectional-coupling dynamics (implicit) rather than frequency-counter (explicit). |
| **Sparse Quantized Hopfield Network (SQHN)** | Alonso & Krichmar | 2024 | Closest extant architecture. Online-continual associative memory with neurogenesis and local learning rules. Two task formats (noisy encoding, episodic memory) applicable to Phase 3+ evaluation. *Underexploited:* quantized codes as role-fidelity structural fix; neurogenesis as dynamic-form actuator for cap-coverage failure. |

### Adjacent Papers (Actuator / Consolidation / Replay / Architecture)

| Paper | Authors | Year | Role in Project |
|---|---|---|---|
| **Autonomous Retrieval for Continuous Learning in Associative Memory Networks** | Saighi & Rozenberg | 2025 | Per-attractor self-inhibition (A_k accumulates on visited attractors; subtracted during settling). Solves blocker #2 (pattern death) via local per-pattern dynamic, not binary threshold. *Underexploited:* self-inhibition as role-fidelity probe (inhibit one role, re-cue, check whether settling lands on different role). Dynamics-based fidelity test, not distance-based. Also: robust at low β — maps to slow-timescale replay-pressure actuator. |
| **Online Continual Learning with Maximally Interfered Retrieval (OCL-MIR)** | Aljundi et al. | 2019 | Replay efficacy: sample by maximum-interference (items whose loss will most increase given foreseen update) rather than uniformly. *Underexploited:* MIR as role-fidelity selector — sample bindings/roles currently maximally-interfered by consolidation; if fidelity holds under MIR, it holds in general. Self-supervised criterion; aligns with codebook geometry. |
| **Information-Theoretic Online Memory Selection (InfoRS)** | Sun et al. | 2022 | Online memory selection by surprise (negative log conditional probability) + learnability (model improvement on point itself). Stochastic InfoRS, rank-one Bayesian updates, GPU-cheap. *Underexploited:* Surprise + learnability as consolidation selector — two-knob combination is anti-homunculus clean; neither knob *decides*; both local geometric properties. |
| **Replay in Deep Learning: Current Approaches and Missing Biological Elements** | Hayes et al. (Kanan) | 2021 | Survey: biological replay is selective, partial, multi-region, reward-modulated, NREM-vs-REM, spontaneous, temporally-structured. Deep learning replay is mostly veridical, single-layer, random-sampled. *Underexploited:* partial-experience replay (role-only vs. filler-only); NREM-vs-REM separation maps to Phase 5 (consolidate known) vs Phase 6 (recombine) split; reward modulation as binding-error-gradient weight on replay. |
| **MESH: Towards Continual Learning Exploiting Heteroassociation and Compression** | Sharma, Chandra & Fiete | 2022 | CAM continuum with no memory cliff via (a) fixed pre-stabilized scaffold + (b) heteroassociation. O(N²) information bound at every point. Phase 6 benchmark target. *Underexploited:* pre-orthogonalized role-codebook as bootstrap (fixed MESH-style scaffold vs. emergent roles); does fixed-ETF initialization eliminate role-fidelity noise floor at D=4096? |
| **Latent Variable Energy-Based Models** | Dawid & LeCun | 2023 | EBM lens on autonomous machine intelligence. *Underexploited:* role-binding as latent variable optimized jointly with energy; role-fidelity reduces to posterior-sharpness (entropy measurement on posterior over roles given binding) — differentiable, potentially D-independent. |
| **A Tutorial on Energy-Based Learning** | LeCun et al. | 2006 | Foundational EBM. Loss functionals, contrastive vs non-contrastive, margins. *Underexploited:* margin-loss framing (push down on data, push up on contrastive points) directly prescribes role-codebook training: minimize `cos(unbind(S,role_true), role_true)`, maximize `cos(unbind(S,role_true), role_other)` — equivalent to Ganesan's Jp + Jn. |
| **Autonomous Machine Intelligence** | LeCun | 2022 | Position paper: H-JEPA + world model + cost + actor + perception. *Underexploited (negatively):* "configurator" is anti-homunculus failure point the May-9 note warns about — useful as negative example, not path forward. |
| **Hopfield 1982** | Hopfield | 1982 | Founding paper. Energy descent, attractor dynamics, 0.14N capacity. Foundational; no new angles for Phase 5. |

### Recent Additions (PAM Family & JEPA)

| Paper | Authors | Year | Role in Project |
|---|---|---|---|
| **Predictive Associative Memory (PAM)** | Dury | 2026 | Temporal-co-occurrence-based retrieval via JEPA-style Inward predictor on continuous experience stream. Association ≠ similarity. A25=0.97, cross-boundary Recall@20=0.42 vs cosine 0. *Load-bearing:* closest thing in bookmarked literature to non-distance fidelity measure. Role is "faithful" if it reliably predicts surrounding bindings. Predictor accuracy is fidelity score without pairwise-distance pathway. |
| **From Topic to Transition Structure** | Dury | 2026 | PAM at corpus scale (29.4M params, 373M co-occurrence pairs). Capacity-constrained forces compression → concept clusters capture narrative function. *Underexploited:* capacity-constrained contrastive training as consolidation actuator; bottleneck *is* the selector; forced abstraction without explicit controller. Concept discovery from co-occurrence as alternative role-codebook acquisition path. |
| **LLM-JEPA (Language Model Prediction Improves with JEPA)** | Huang, LeCun, Balestriero | 2025 | JEPA embedding-space prediction objective on LLM training. Outperforms baseline on NL-RX, GSM8K, Spider, RottenTomatoes. Robust to overfitting. *Underexploited:* JEPA as substrate-level training signal for roles — if two bindings share a role, unbinds along that role should be predictable from each other; use predictability as loss for role codebook. Bypasses noisy unbind-distance signal. |

### Brainstorm-Derived Papers (May 2026 Deep-Research Phase)

| Paper | Authors | Year | Role in Project |
|---|---|---|---|
| **Free Energy Principle Applied to Attractor Networks** | Spisak & Friston | 2025 | Derives attractor networks directly from FEP. Attractor orthogonalization (which project wants for codebook diversity) is FEP consequence, not design choice. *Impact:* FEP formalizes anti-homunculus filter — diagnostic IS free energy gradient; actuator IS response to gradient. Verifiable with GitHub repo (pni-lab/fep-attractor-network). |
| **Dense Associative Memory with Epanechnikov Energy** | [2026, arXiv 2506.10801] | 2026 | LSR (Log-Sum-ReLU) kernel solves regime-disjoint without changing β. At intermediate β: all K stored patterns stable AND emergent blend attractors (centroids of overlapping clusters) coexist. Automatic routing: unambiguous cues→originals, ambiguous→centroids. Not approximation; structural property. *Impact:* addresses regime-disjoint failure the May-9 note flags; creative recombination without supervisor. |
| **Input-Driven Plasticity Hopfield** | [Science Advances 2025] | 2025 | Per-cue effective temperature without supervisor. Saliency weights computed from cue-pattern alignment. Energy landscape continuously reshaped: dominant matches→deep wells, weak→shallow. Ambiguous cues→graded states. Anti-homunculus clean. *Impact:* composes with LSR kernel; per-cue routing between blend/commit modes without module reading ambiguity. |

### Foundations / Interpretive (Lower Leverage for Phase 5)

- **Spin Glass Neural Networks** — Scanned PDF, OCR pending. Likely foundational stat-mech material adding nothing actionable beyond Hopfield 1982 + Krotov-Hopfield 2016.

---

## II. Cross-Paper Synthesis Themes

### A. Anti-Homunculus as Architectural Discipline (Load-Bearing)

**Statement (from 2026-05-09 note):**  
> Every proposed addition to the architecture must either be a local geometric dynamic or be expressible as a measurement of one — never an arbitration over them.

**Operationalized via FEP (from May 2026 brainstorm):**  
A mechanism passes the filter if it has natural description as gradient descent on variational free energy. A mechanism fails if it requires inspecting a metric and triggering a response.

**Implications for Phase 5:**
- Diagnostic-actuator pairs must identify the slow-timescale dynamic being snapshotted, not a rule that reads the diagnostic.
- No conditional logic of the form `if metric(X) > threshold then action(Y)`.
- Each diagnostic contributes to an energy-like quantity; response is natural evolution of that quantity.

**Papers supporting this framing:** Spisak & Friston (FEP), Saighi (self-inhibition as local plasticity), Krotov (energy descent), LeCun (EBM margins).

---

### B. Dimensionality & Noise Floor — The Core Phase 5 Blocker

**The Problem (from May 2026 brainstorm literature context):**  
At D=4096, role-fidelity measurements based on pairwise distance of unbinds are noise-dominated. The discriminability signal (true role vs. false role in unbind space) is swamped by HRR cross-talk variance that grows with both binding count (W) and dimension (D).

**The Krotov-Hopfield Diagnosis (2016):**  
```
P_error ≈ √((2n-3)!!/2π) · (K/N^(n-1)) · exp(-N^(n-1)/(2K(2n-3)!!))
```
Where:
- K = number of co-bound roles (W in binding case)
- N = effective dimension (D)
- n = interaction order (1 for linear softmax at low β; ≥2 for higher kernels)

At n=1 (the current project regime), noise floor is structural and scales as `1/√D`. At n≥2, floor recedes super-linearly. This is not a tuning problem; it is a kernel choice problem.

**The Ganesan Solutions (2021):**
1. Complex unit-magnitude projection π makes unbind variance constant as W scales.
2. Jp + Jn loss (dimensionless, scale-free) has structural ceiling 1, floor -1; does not noise-floor with D.
3. Binding-capacity-vs-dimension curve (Appendix D) shows linear scaling if projection maintained.

**Vangara-Gopinath Reframe (2026):**  
Instead of measuring role-fidelity as pairwise distance, measure cap-coverage of unbind cluster around true role. The `(θ'/d̄)^(d_eff/2)` bound applies; if effective dimension d_eff of unbind cluster is much smaller than D, the cluster is in tight regime regardless of D=4096.

**Papers addressing this:** Ganesan (unit-magnitude projection), Krotov (analytic SNR), Vangara-Gopinath (cap-coverage), HEN (separability in latent space), MESH (pre-orthogonal scaffold).

---

### C. Regime Disjunction: Blend vs. Sharp Retrieval

**The Problem (from 2026-05-13 brainstorm):**  
At D=4096 with exponential (softmax) kernel, blend and sharp retrieval are mutually exclusive. Low β (0.001–0.01): diffuse blend, no retrieval. High β: sharp memorization, no blend. No intermediate regime where both coexist. The project has been trying to solve this with per-query temperature control (supervisory) or dual-mode switching (also supervisory).

**The LSR Kernel Solution (2026):**  
The Log-Sum-ReLU (compact-support) kernel exhibits qualitatively different phase structure. At intermediate β, **all K stored patterns are simultaneously stable fixed points AND exponentially many emergent blend attractors (centroids of overlapping clusters) appear**. The system naturally routes: unambiguous cues→originals, ambiguous cues→emergent centroids. No supervisor; no temperature control per query.

**The Input-Driven Plasticity Solution (2025):**  
Per-cue effective temperature without supervisor. Memory saliency weights computed from cue-pattern alignment. Landscape continuously reshaped: dominant matches get deeper wells, weak matches shallower. Ambiguous cues automatically produce graded states. Composes with LSR kernel.

**Papers addressing this:** LSR kernel (2506.10801), Input-Driven Plasticity (Science Advances 2025), Krotov (regime temperature analysis).

---

### D. Diagnostics as Actuators (Architectural Threshold — Partially Explored)

**The Principle (from 2026-05-09 note):**  
> Diagnostic and actuator are the same physical process viewed at different temporal resolutions. The diagnostic measures the process right now; the actuator is the process running.

**Example Pairs (from May 9):**

| Diagnostic | Standard (Wrong) | Non-Controller (Right) | Physics |
|---|---|---|---|
| High drift | Read drift, trigger replay | Drift contributes to replay-tension energy that drives replay when crossed | Slow energy accumulation |
| High spread d̄ | Read spread, reduce consolidation | Consolidation gate threshold rises with d̄ naturally | Spread → precision change in likelihood |
| Bimodality | Read bimodality, trigger split | Persistent bimodality contributes to split-tension energy | Redundancy cost |
| Metastability | Read metastability, boost priority | Per-atom m_i accumulates when atom in metastable settle, paid down on replay | Local softmax-weight statistic |
| Low cap-coverage | Read cap-coverage, trigger restructuring | Cap-coverage failure raises local error gradient on consolidation; restructuring is the response | Reconstruction surprise |

**Current Status:**
- **Pair #1 (death ~ inhibition):** Closed by Saighi mechanism (per-pattern self-inhibition A_k accumulates on use, subtracted during settling). Dynamic form specified.
- **Pair #2 (spread ~ consolidation-gate):** Closed by Vangara-Gopinath bound (gate rises with d̄ → reduced update weight). Dynamic form specified.
- **Pair #3 (bimodality ~ split):** Attempted; failed at D=4096 (hit FHRR noise floor 1/√D ≈ 0.016). K-branch state_divergence saturated.
- **Pair #4 (metastability ~ replay-prioritization):** In progress (per-atom m_i EMA via `dm_i/dt = ζ · 𝟙{atom_i metastable} − μ · m_i · 𝟙{atom_i replayed}`). Dynamic form drafted.
- **Pair #5 (cap-coverage ~ restructuring):** Not yet open.

**Papers supporting:**  Saighi (self-inhibition), Benna-Fusi (bidirectional coupling), Krotov HAM (top-down energy), Aljundi MIR (interference gradient), Sun InfoRS (surprise + learnability), Dury (capacity-constrained consolidation).

---

### E. Cleanup Networks, Basin Sharpening, and Iterative Settling

**Not yet a named paper cluster in project reading, but adjacent themes:**

**From Krotov 2016 & 2021:**  
Higher-order interactions (n ≥ 2) push capacity from 0.14N to N^(n-1)/log(N). Energy functions `E = -ΣF(ξ·σ)` with F polynomial of n-th degree. At n=2 (quadratic), basins sharpen dramatically; noise floor recedes from 1/√D to 1/D.

**From HEN (Kashyap 2024):**  
Encoder→Hopfield→Decoder pipeline improves separability in latent space. This is a cleanup network in latent space, not in the original high-dimensional space.

**From HAM (Krotov 2021):**  
Top-down feedback from higher layer. Iterative settling between layers. Basin sharpness is the *consistency between levels* — does the lower layer's settled state activate the higher layer's expected attractor?

**From MESH (Sharma et al. 2022):**  
Pre-stabilized scaffold (fixed attractors with known separability). Heteroassociation wraps around it. The scaffold *is* the cleanup network.

**Papers addressing:**  Krotov (2016, 2021), HEN, MESH, Papyan (NC2 tight frame), SQHN (quantization as discrete cleanup).

---

### F. Multi-Timescale Consolidation & Replay

**From Benna-Fusi (2016):**  
Bidirectional fast↔slow coupling produces near-linear capacity scaling (vs. √N). Slow→fast feedback is necessary, not just consolidation. Memory lifetime scales as synapse count N, not √N.

**From Hayes/Kanan (2021):**  
Biological replay is selective (not random), partial (not veridical), multi-region, reward-modulated, NREM-vs-REM distinct, spontaneous, temporally-structured. Deep learning replay is missing most of these.

**Operationalization from 2026-05-13 brainstorm:**
- **Upgrade 4a — tag_count vs. age:** Prioritize by reactivation frequency during awake rest, not just age (Joo & Frank 2023, Science).
- **Upgrade 4b — graded u_1 initialization:** Larger SWRs → stronger consolidation (Neuron 2025). Initialize u_1 ∝ gate_signal.
- **Upgrade 4c — inhibition of return:** Prevent monopolization by highest-gate entry; decay + recovery of suppression multiplier (SFMA, Biderman et al. 2023).

**Papers addressing:**  Benna-Fusi, Hayes/Kanan, Joo & Frank, SFMA, Aljundi MIR, Sun InfoRS, Saighi (self-inhibition as decay dynamic).

---

### G. Role Codebook Acquisition Without Binding-Task Gradient

**Alternative pathways identified in literature:**

1. **Dury 2603 (From Topic to Transition Structure):**  
   Co-occurrence structure of substrate states across many bindings. Capacity-constrained forces compression. Concept discovery from transitions. Bypasses binding gradient entirely.

2. **LLM-JEPA (Huang et al. 2025):**  
   JEPA predictability between unbinds of same role across different bindings. Predictor accuracy is role-fidelity signal. Trains codebook as side-effect, not main objective.

3. **SQHN (Alonso & Krichmar 2024):**  
   Neurogenesis (grow new units on demand) when discriminability falls. Anti-homunculus: decision to grow is implicit in cap-coverage gradient.

4. **MESH (Sharma et al. 2022):**  
   Pre-orthogonal scaffold (fixed, not learned). Heteroassociation wraps around it. Question: does eliminating role learning eliminate Phase 5 noise floor?

**Papers addressing:**  Dury (both papers), LLM-JEPA, SQHN, MESH, Papyan (NC2 ETF initialization).

---

## III. Open Questions Flagged But Not Closed

### A. Diagnostic-Actuator Pairs Still Open (from 2026-05-09 note, §"What Remains Open")

1. **Pair #3 (bimodality ~ splitting):** K-branch state_divergence operationalization saturated at FHRR noise floor (1/√D ≈ 0.016). The mechanism is sound; the measurement is noise-dominated. *Requires:* Either (a) non-distance fidelity metric that beats the noise floor, or (b) architectural change (LSR kernel, HEN-style encoder, ETF initialization).

2. **Pair #4 (metastability ~ replay-prioritization):** Dynamic form drafted (per-atom m_i EMA). *Requires:* (a) Anti-homunculus reviewer audit, (b) empirical validation on Phase 4 data.

3. **Pair #5 (cap-coverage ~ restructuring):** Not yet opened. *Would require:* Identifying the local error gradient that cap-coverage failure contributes to, and verifying that restructuring is its natural response.

### B. Empirical Calibrations Flagged (from May 9)

1. **θ′(β) calibration via Geometry-of-Consolidation E1 protocol:**  
   The consolidation-geometry diagnostic uses θ′ ≈ 1/β as approximation. Running the E1 protocol (controlled d_eff, d̄, θ grid) on FHRR substrate would give calibrated mapping. *Status:* Not blocking; 2 days of work; worth doing before Phase 3 relies on regime classifier.

2. **Binding-capacity-vs-dimension curve calibration (Ganesan Appendix D):**  
   Where does the project's W=4 binding stack sit on the linear-vs-saturated spectrum? *Status:* Pre-Phase-5 calibration; required if Phase 5 commits to unit-magnitude projection audit.

3. **Effective d_eff of unbind cluster (Vangara-Gopinath):**  
   If d_eff ≪ D, role-fidelity cap-coverage may be tight-regime even at D=4096. *Status:* High-leverage diagnostic; would reframe noise-floor problem.

### C. Architecture-Level Questions (from Brainstorm & Synthesis Notes)

1. **Does LSR kernel (Epanechnikov energy) solve the regime-disjunct?**  
   *Test:* Implement LSR kernel; scan β at D=4096; measure whether emergent intermediate regime with blend + retrieval coexists. Phase 2 masked-token eval on LSR vs. exponential.

2. **Does Input-Driven Plasticity + LSR compose additively?**  
   *Test:* Run IDP on LSR kernel; measure whether per-cue saliency reshaping improves regime routing without loss on either mode.

3. **Does pre-orthogonal role initialization (ETF, MESH-style scaffold) eliminate D=4096 noise floor?**  
   *Test:* Initialize role codebook at simplex ETF; lock it (don't learn); run Phase 5 binding eval. Does role-fidelity become a non-issue?

4. **Is role-fidelity genuinely structural at n=1 (linear softmax), or is the project operating at effective n ≥ 2?**  
   *Test:* Vary β systematically; fit observed role-discrimination SNR to Krotov formula. Solve for implied n. If n ≥ 2, noise floor should be much tighter than observed.

5. **Does JEPA-on-roles or PAM-style co-role prediction outperform pairwise unbind distance?**  
   *Test:* Implement PAM-style predictor (given one unbound role, predict other W-1 roles). Implement JEPA-style predictor (unbind-to-unbind across bindings). Compare fidelity signal to distance-based baseline.

### D. Conceptual Questions (Unsettled Framing)

1. **Is the role-fidelity problem a role-codebook problem, a binding-substrate problem, or a measurement problem?**  
   - Role-codebook: roles not well-separated in FHRR space (fix: ETF init, Papyan NC2, MESH scaffold).
   - Binding-substrate: FHRR binding operation degrades separability (fix: Ganesan projection verification, unit-magnitude audit).
   - Measurement: pairwise distance is the wrong metric for FHRR binding (fix: Ganesan Jp/Jn, Vangara cap-coverage, PAM prediction, JEPA).
   
   *Status:* All three are plausible. The Ganesan audit (unit-magnitude projection) and d_eff measurement (Vangara) would narrow the diagnostic space significantly.

2. **Is the regime-disjunct a fundamental property of softmax kernels, or fixable with the right β schedule?**  
   *Status:* LSR and IDP papers suggest it's a kernel property, not a β tuning problem. But not yet tested in the project context.

3. **Should Phase 6 consolidation be modeled after biological hippocampal replay (selective, partial, NREM-REM split) or after something else (Dury capacity-bottleneck, Benna-Fusi cascade, MIR interference)?**  
   *Status:* Hayes/Kanan survey establishes biological properties; Dury, Benna-Fusi, MIR provide mechanistic alternatives. Not yet integrated into Phase 6 design.

---

## IV. What the Literature Suggests About D=4096

### The Noise Floor (Scaling Behavior)

**Krotov-Hopfield 2016 prediction:**  
At interaction order n=1 (linear softmax, low β), error scales as `exp(-D^(n-1)/(2K(2n-3)!!)) = exp(-1/(2K))` — *independent of D*. The noise floor does not improve with increasing D at n=1; it's structural.

**Ganesan 2021 solution:**  
Unit-magnitude projection + Jp/Jn loss. With projection, Appendix D shows binding capacity scales *linearly* with D. Without it, capacity saturates and variance explodes.

**Vangara-Gopinath 2026 insight:**  
The relevant dimension is not D (nominal substrate dimension) but d_eff (effective dimension of the data cluster). The bound `(θ'/d̄)^(d_eff/2)` applies to d_eff. If role unbind clusters have d_eff ≪ D=4096, they live in tight regime regardless.

**Empirical observation (May 2026 brainstorm context):**  
At D=4096, N=1064 atoms, K=4 roles (W=4 binding), noise-energy scale is ~5.5e-3. Signal ΔE from Phase 5 Decision-5 spike was +2.6e-5 (0.45× noise scale). Not merely noisy; structurally unresolvable at current setup.

### What Scales Well to D=4096

1. **FHRR substrate itself (if unit-magnitude maintained):** Ganesan shows binding capacity scales linearly with D to at least d=256; extrapolation suggests d=4096 is well into linear regime.

2. **Cap-coverage metric (Vangara-Gopinath):** Bound applies in all dimensions. If d_eff is calibrated, it holds at d=4096.

3. **Jp + Jn loss (Ganesan):** Dimensionless, scale-free by construction. Structural ceiling/floor regardless of D.

4. **PAM-style prediction (Dury):** Predictor accuracy is bounded by model capacity, not by dimension. Scales orthogonally to D.

5. **JEPA-on-roles (LLM-JEPA pattern):** Predictability between unbinds is a dynamic property, not a distance metric. Orthogonal to D scaling.

### What Does NOT Scale Well to D=4096

1. **Pairwise unbind-distance as fidelity metric:** Cross-talk noise floor 1/√D ≈ 0.016. Signal-to-noise ratio degrades as `1/√D`.

2. **Softmax regime disjunction:** LSR and IDP papers suggest exponential kernel fundamentally cannot blend + retrieve simultaneously without supervisor. Not a D issue; a kernel issue.

3. **State-divergence as bimodality detector (K-branch):** Saturation at noise floor; cannot resolve finer-grained bimodality at high D.

---

## V. Cleanup Networks, Iterative Settling, Sharp vs. Soft Basins

### The Architectural Crux for Phase 5

**The question:** How sharp should basins be? Extremely sharp basins (hard decisions, single-pattern retrieval) prevent blending and creativity. Soft basins (smooth overlap) enable creativity but degrade retrieval specificity. Can both coexist?

### What the Literature Proposes

1. **Krotov (2016, 2021): Interaction Order as Basin Shaper**  
   Higher-order interactions (n ≥ 2) sharpen basins without changing learning rules. Energy function `E = -ΣF(ξ·σ)` with F = polynomial of degree n. At n=2, basins sharpen; noise floor tightens from 1/√D to 1/D. The project could test whether moving from n=1 (softmax) to n=2 (pairwise product-kernel) solves the noise floor.

2. **Kashyap et al. (HEN): Latent-Space Cleanup**  
   Encoder→Hopfield→Decoder. The encoder maps high-D messy input to low-d latent space where patterns are well-separated. Hopfield operates in latent space (sharp basins possible because d is small). Decoder reconstructs. The "cleanup" happens in the bottleneck; latent basins can be sharper than input space basins.

3. **Krotov (2021): HAM — Hierarchical Cleanup via Top-Down Feedback**  
   Two-layer network. Lower layer stores filler+role pairs. Upper layer stores expected shape (how should fillers be related given role?). Bottom-up settling drives lower layer. Top-down feedback from upper layer sharpens lower-layer basins by constraining which mixed states are compatible with upper-layer attractors. Iterative settling between layers.

4. **Sharma et al. (MESH): Pre-Stabilized Scaffold**  
   Fixed set of well-separated scaffold attractors (e.g., random binary patterns or simplex ETF). Content-patterns heteroassociate to scaffold states. Scaffold states have structural sharp basins (they're not learned; they're constructed to be maximally separated). Content retrieval settles into scaffold basin + associated content.

5. **Papyan et al. (Neural Collapse): Tight Frames as Terminal Geometry**  
   Simplex equiangular tight frame (NC2) is the maximally-separated multi-class structure. All pairwise angles are equal: `cos(c_i, c_j) = -1/(K-1)` for i ≠ j. Basins would be razor-sharp if attractors were at ETF positions. The project could test: initialize roles at ETF, lock them, and measure whether Phase 5 noise floor disappears.

6. **Alonso & Krichmar (SQHN): Quantization as Discrete Cleanup**  
   Quantized neural codes (discrete activations, not continuous). Basins collapse to discrete points by construction. Local MAP learning. Provides a different notion of "sharpness" — not continuous gradient but discrete attractor.

### How This Frames the Phase 5 Crux

The project wants:
- **Soft basins for blending:** Multiple patterns contribute to retrieval, enabling creativity.
- **Sharp basins for specificity:** Retrieval fidelity, role separation, clean signal-to-noise.

The literature suggests these are not mutually exclusive if:
- **Basin sharpness is relative, not absolute.** What's sharp at the latent level (HEN, HAM) can be soft at the input level. Multi-scale sharpness (Benna-Fusi timescale cascade).
- **Sharpness is role-dependent, not global.** Roles (structure) have sharp basins. Fillers (content) have soft basins (blending encouraged). Two different storage systems.
- **Kernel choice matters.** Softmax at n=1 cannot simultaneously blend and retrieve. Softmax at n≥2, LSR kernel, or quantized codes can.

**For Phase 5 decision:** The regime-disjunct and noise-floor issues suggest the project may need to move away from pure softmax-at-n=1. Concrete candidates: (1) LSR kernel + IDP, (2) Higher-order interaction (n≥2), (3) HEN-style encoder, (4) HAM-style hierarchical, (5) MESH-style scaffold, (6) SQHN-style quantization.

---

## VI. Previous Brainstorm Topics & Scope

### 2026-05-13: Neuro-Personal-AI Architecture Brainstorm

**Context:** Identified three live tensions: (1) diagnostics-to-actuators threshold, (2) regime disjunct, (3) temporal structure.

**Ideas generated:**
- Idea 1: FEP as formal anti-homunculus test (diagnostic IS free-energy gradient).
- Idea 2: LSR kernel solves regime disjunct (emergent blend attractors at intermediate β).
- Idea 3: Input-Driven Plasticity as anti-homunculus temperature mechanism.
- Idea 4a–4c: Three replay-gate upgrades (tag-count vs age, graded u_1 init, inhibition of return).

**Outcome:** Framing document for Phase 5 architectural decisions. Not implementation; not testing.

### 2026-05-20: Phase 5 Graduation Brainstorm

**Context:** Phase 5 substrate saturation (role-fidelity noise-floor at D=4096). Report 051 falsified all Tier-1 alternative metrics (decode-margin, settling dynamics, pairwise distance). K-branch state_divergence saturated at FHRR noise floor.

**Deep-research output:** Comprehensive literature scan (21 papers, 8 flagged-findings clusters). Catalogued alternative fidelity metrics, consolidation selectors, codebook-acquisition pathways, architectural fixes for noise floor. Generated 180-line "Quick Reference" table mapping Phase 5 knobs to papers.

**Outcome:** Identified three ranked open pairs for next pivot:
1. **Metastability ~ replay-prioritization (Pair #4):** Most tractable. Per-atom m_i EMA dynamic form drafted. Anti-homunculus audit pending.
2. **Cap-coverage ~ restructuring (Pair #5):** Lowest ambition, highest tidiness. Single error-gradient flow.
3. **PAM-style co-role prediction vs. pairwise distance:** Most divergent. Would require new measurement infrastructure but cleanest win if it works.

**Papers emphasizing:** Dury PAM, Vangara-Gopinath, Krotov 2016, Ganesan, HEN, MESH.

---

## VII. Synthesis: Where the Research Base Converges

### A. The Anti-Homunculus Filter is Formalizable

**Papers backing formalization:** Spisak & Friston (FEP), Saighi (self-inhibition), Krotov (energy), Benna-Fusi (coupling).

FEP supplies the proof: diagnostic IS free-energy gradient; actuator IS gradient response. Verifiable mathematically and empirically. This closes the ambiguity in the May-9 note's "right-hand column" framing.

### B. The Noise Floor is a Kernel Choice, Not a Tuning Problem

**Papers backing this:** Krotov 2016 (n=1 vs n≥2), Ganesan (unit-magnitude projection), LSR kernel (compact-support phase structure), IDP (saliency reshaping).

At n=1 (softmax), noise floor is structural, D-independent, and unfixable by tuning. Moving to n≥2, LSR, or quantized codes is the architectural move, not retuning β.

### C. Pairwise Distance is Not the Right Metric for FHRR Binding

**Papers backing alternatives:**
- **Ganesan Jp/Jn:** Dimensionless, scale-free. Structural bounds. Loss function, not metric.
- **Vangara cap-coverage:** Regime-aware. Published bound applies. Measurements of cluster geometry, not pairwise distance.
- **PAM prediction:** Co-role predictability. Bypasses unbind-distance entirely.
- **JEPA-on-roles:** Unbind-to-unbind predictability across bindings. Trains codebook as side-effect.
- **HEN separability:** Encoder makes patterns separable in latent space before Hopfield.

All of these are dimensionally orthogonal or sub-linear in D scaling. All avoid the 1/√D cross-talk floor.

### D. Cleanup / Basin Sharpening Requires Architectural Choice, Not Just Tuning

**Papers backing this:** Krotov (n vs n≥2), HEN (encoder), HAM (top-down), MESH (scaffold), Papyan (ETF), SQHN (quantization).

Pure softmax at n=1 cannot support both sharp basins and soft blending. The options:
- **Higher interaction order (n≥2):** Sharper basins, tighter noise floor.
- **LSR or other compact-support kernel:** Structural intermediate regime with both modes.
- **Latent-space bottleneck (HEN, HAM):** Sharp latent basins, soft input basins.
- **Pre-stabilized scaffold (MESH):** Structural sharp scaffold basins.
- **Fixed role codebook at ETF (Papyan):** Roles have razor-sharp separation by construction.

### E. Diagnostics-as-Actuators is the Right Architectural Grammar

**Papers backing:** Saighi (self-inhibition), Benna-Fusi (bidirectional cascade), Krotov HAM (top-down energy), Aljundi MIR (interference gradient), Sun InfoRS (surprise + learnability), Dury (capacity bottleneck).

The May-9 note's intuition is confirmed by six different papers across six different mechanisms. The grammar is: diagnostic snapshot of a slow-timescale variable; actuator is the evolution of that variable. No separate "if X then Y" logic.

---

## VIII. Recommended Next Steps for Phase 5 Deep-Research

### High-Priority (Blocking Phase 5)

1. **Ganesan audit:** Verify unit-magnitude FFT projection is maintained end-to-end in FHRR substrate and Hebbian updates. (May reduce top1 regression of blocker #6'; confirms noise floor is not FHRR numerical artifact.)

2. **d_eff measurement:** Run PCA on unbind clusters of stored roles. Measure effective dimension. If d_eff ≪ 4096, cap-coverage may be tight-regime at D=4096. (Vangara bound applies; may reframe noise-floor problem entirely.)

3. **LSR kernel pilot:** Implement Log-Sum-ReLU; scan β at D=4096. Check whether emergent intermediate regime with blend + retrieval coexists. (Tests whether regime-disjunct is architectural or tuning.)

4. **PAM-style predictor:** Train simple predictor (given one unbound role, predict other W-1 roles). Compare predictor accuracy to pairwise-distance fidelity score. (Tests whether non-distance metric beats noise floor.)

### Medium-Priority (Architecture-Level Options)

5. **ETF role initialization:** Lock role codebook at simplex ETF; run Phase 5 eval. Does fidelity noise floor disappear? (Tests whether role-codebook separation is the bottleneck.)

6. **HEN-style encoder:** Train an encoder on filler space; store fillers in latent space; decode on retrieval. (Tests whether latent-space cleanup solves noise floor without architectural restructure.)

7. **Pair #4 anti-homunculus audit:** Review per-atom m_i EMA for metastability-replay-prioritization. Check whether dynamic form is truly supervisory-free. (Gate-keeper for next implementation cycle.)

### Lower-Priority (Phase 6 Groundwork)

8. **Benna-Fusi timescale calibration:** Measure slow/fast variable coupling and capacity scaling. Validate prediction of near-linear scaling. (Phase 6 design foundation.)

9. **PAM co-occurrence extraction:** Run PAM concepts extraction on Phase 5 data. Check whether role structure emerges from co-occurrence without binding-task gradient. (Alternative codebook-acquisition path.)

10. **MESH-style scaffold:** Pre-generate fixed scaffold of well-separated attractors. Run Phase 5 with scaffold as role codebook basis. (Architectural pivot option if learning-based roles remain intractable.)

---

## Summary

The project has read 21+ papers that directly address or provide adjacent framing for the Phase 5 substrate-saturation blocker. The literature converges on several high-confidence insights:

1. **Anti-homunculus filter can be formalized via FEP** — diagnostic IS gradient, actuator IS response. Verifiable.
2. **D=4096 noise floor is kernel-driven, not tuning-driven** — moving from n=1 (softmax) to n≥2, LSR, or quantized codes is the fix.
3. **Pairwise unbind-distance is the wrong metric** — dimensionless metrics (Ganesan Jp/Jn, cap-coverage, prediction-accuracy) scale orthogonally to D.
4. **Regime disjunction is not fundamental to Hopfield** — LSR kernel and IDP together provide simultaneous blend + retrieve without supervisor.
5. **Diagnostics-as-actuators works when framed as slow-timescale snapshots** — six papers across six mechanisms validate the grammar.

The next Phase 5 deep-research session should prioritize: (1) Ganesan audit, (2) d_eff measurement, (3) LSR kernel pilot, (4) PAM predictor prototype. These four would clarify whether the blocker is noise-floor-of-measurement or noise-floor-of-architecture, and which of the six architectural pivots (higher-order kernel, LSR, HEN, ETF, HAM, quantization) is most promising.

