# Brainstorm: Neuro-AI / Personal-AI Architecture
> Generated 2026-05-13 from context in: `/Users/dypatterson/Desktop/Neuro-AI`, `/Users/dypatterson/projects/personal-ai`

---

## Project Understanding

You are building a memory-first AI companion where personality is consolidation history, not model weights. The system uses FHRR (D=4096 holographic vectors) as its algebraic substrate, Modern Hopfield Networks for energy-settling retrieval, a growing emergent codebook of atomic patterns, and a Benna-Fusi multi-timescale cascade for consolidation. Two branches of the repository represent two stages of the same vision: `personal-ai` is a clean Phase 1–2 implementation with rigorous testing; `Neuro-AI` is the living research arm that has pushed experimentally through Phase 5.

The two most architecturally live tensions as of today are: (1) the **diagnostics-to-actuators threshold** — the system can measure geometric states but doesn't yet have a principled grammar for how measurements become responses without importing a hidden controller — and (2) the **regime disjoint** — the Hopfield landscape at D=4096 can blend or retrieve but not both, and this feels like a bug rather than a feature. The research below upends both of these framings in useful ways.

The third active thread is **temporal structure**: temporal context bags (unordered co-occurrence bundles) are proven to commit two named failure modes (temporal inaccuracy and temporal fragmentation), and the research community has developed at least three distinct algebraic approaches to directed temporal binding in FHRR — none of which is the same as named temporal roles, but all of which are stronger than current bags.

---

## Ideas and Approaches

---

### Idea 1: The Free Energy Principle as a Formal Pass/Fail Test for the Anti-Homunculus Filter

**What:** The Free Energy Principle (FEP) provides a rigorous criterion that resolves the ambiguity in the anti-homunculus filter. A mechanism passes the filter if and only if it has a natural description as gradient descent on a variational free energy functional. A mechanism fails if it requires inspecting a metric and triggering a response. This reframes the diagnostics-to-actuators problem: the diagnostic IS the free energy gradient; the actuator IS the local response to that gradient. Nothing reads the diagnostic and decides — the gradient is the dynamics.

**Why it's relevant:** The 2026-05-09 note identifies the right-hand column formulation ("consolidation gate threshold rises with d̄, so spread atoms naturally get smaller updates without anyone reading d̄") but lacks the formal grounding to prove it's not a hidden controller. FEP supplies that proof: d̄ contributing to the free energy that drives dynamics is provably non-supervisory by the FEP's own mathematical definition. Every diagnostic-actuator pair in the existing architecture can be audited against this criterion.

Concretely, the translation table is:
| Diagnostic | FEP quantity | Gradient response |
|---|---|---|
| High drift | KL divergence from prior model | Replay drives KL toward 0 |
| High spread d̄ | Precision of consolidation likelihood | Low precision → smaller update weight |
| Bimodality | Complexity cost of redundant attractors | Gradient penalizes redundancy → split |
| Metastability | Prediction error on settling trajectory | Higher error → larger replay weight |
| Low cap-coverage | Surprise at reconstruction | Surprise gradient drives restructuring |

None of these require a controller. All are free energy gradients.

**How to explore it:**
1. Read the Spisak & Friston paper (arXiv:2505.22749, May 2025) — it derives attractor networks directly from FEP and shows that the attractor orthogonalization the project wants for codebook diversity is a *free energy consequence*, not a design choice. The GitHub repo (pni-lab/fep-attractor-network) has simulation code.
2. Run a quick test: train the Hebbian codebook on Phase 2 data and measure pairwise cosine similarity between atoms over time. If they drift toward orthogonality without explicit repulsion, the FEP prediction is confirmed.
3. For each Phase 4 mechanism you add, write out the FEP gradient translation before coding. This is not extra work — it is the anti-homunculus filter, formalized.

**Sources:**
- Spisak & Friston (May 2025): https://arxiv.org/abs/2505.22749
- FEP made simpler: https://www.sciencedirect.com/science/article/pii/S037015732300203X
- pymdp reference implementation: https://github.com/infer-actively/pymdp

---

### Idea 2: The LSR (Epanechnikov) Kernel Solves the Regime Disjoint — Without Changing β

**What:** The "regime disjoint failure" at D=4096 (soft-blend and sharp-retrieval are mutually exclusive) is not a tuning problem — it is a known theoretical property of the exponential (softmax/LSE) kernel. But a 2026 paper proves that the Epanechnikov/LSR (Log-Sum-ReLU, compactly-supported) kernel has a qualitatively different phase structure: at **intermediate β**, all M original stored patterns are simultaneously stable fixed points AND exponentially many emergent blend attractors (centroids of overlapping pattern clusters) appear as additional attractors. The system navigates to originals from unambiguous cues and to emergent centroids from ambiguous ones — automatically, through energy minimization.

**Why it's relevant:** This is not an approximation of the behavior the project wants. It is the behavior the project wants, achieved without per-query temperature control, without a supervisor routing to blend vs. retrieve mode, and without abandoning the Hopfield substrate. The emergent centroids are the creative recombination outputs the project has been targeting.

**How to explore it:**
1. Implement the LSR kernel (replace `softmax(β * X^T @ ξ)` with `relu(β * X^T @ ξ) / sum(...)`) and scan β at D=4096. The emergent intermediate regime should be visible as a phase where K-graded ambiguity signal AND retrieval accuracy coexist.
2. Map the phase diagram: find β_low (diffuse blend, no retrieval), β_intermediate (the emergence regime), β_high (sharp memorization only). The project's current crossover zone (0.001–0.01) is likely where the LSR emergence regime lives.
3. Run the Phase 2 masked-token experiments on the LSR kernel and compare cap-coverage vs. the exponential baseline.

**Sources:**
- Dense Associative Memory with Epanechnikov Energy (2026): https://arxiv.org/html/2506.10801
- LSE vs LSR thermal robustness: https://arxiv.org/html/2603.13350

---

### Idea 3: Input-Driven Plasticity as an Anti-Homunculus Temperature Mechanism

**What:** The Input-Driven Plasticity (IDP) Hopfield model (Science Advances 2025) achieves per-cue effective temperature without any supervisor and without changing the global β parameter. Each memory's "saliency weight" is computed from the alignment between the incoming cue and each stored pattern. The energy landscape is continuously reshaped: dominant matches get deeper wells, weak matches get shallower wells. Unambiguous cues automatically produce sharp commitment (one deep well dominates). Ambiguous cues automatically produce graded states (multiple wells remain viable).

**Why it's relevant:** IDP is anti-homunculus clean. No module reads the ambiguity of the cue and adjusts temperature — the "decision" between blend and commit emerges from the local geometry of well depths relative to each other. This is exactly the language the 2026-05-09 note is reaching for. IDP would also compose with the LSR kernel (Idea 2) — the LSR emergence regime provides the coexisting attractor structure; IDP provides the per-cue depth differential that routes to the appropriate attractor.

**How to explore it:**
1. Implement IDP as a modulated memory matrix: `M_eff = M * diag(sim(cue, M)^γ)` before running Hopfield dynamics. The γ parameter controls aggression of the depth differential; γ=0 recovers standard Hopfield.
2. Measure: does IDP at γ > 0 reduce the number of retrieval failures on ambiguous cues without degrading performance on unambiguous cues?
3. Test whether IDP + LSR kernel are additive improvements or redundant.

**Sources:**
- IDP paper (2025): https://arxiv.org/html/2411.05849v1
- Science Advances PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12017325/

---

### Idea 4: Three Specific Upgrades to the Replay Gate (Biologically Grounded)

**What:** The biological literature on hippocampal replay provides three distinct, concrete refinements to the current `engagement × (1 - resolution)` gate — each addresses a different gap, each is anti-homunculus clean.

**Upgrade 4a — Tag count vs. age:** The current store uses `age` (time since first entry). Biological evidence (Joo & Frank 2023, *Science*) shows that sleep replay priority is predicted by **how many times** a trace was tagged during awake rest — not how old it is. Implement a `tag_count` field: when a new query arrives with high gate signal and its trajectory neighborhood overlaps with a stored trace (cosine > threshold), increment that trace's `tag_count` rather than creating a duplicate. Sample replay probability proportional to `gate_signal × tag_count × suppression`.

**Upgrade 4b — Graded u_1 initialization:** Large SWRs correlate with stronger consolidation (Neuron 2025). The current `u_state.initialize(new_idx, u_1=novelty_strength)` uses a fixed value. Replace with: `u_1_init = base_novelty_strength × gate_signal_at_resolution`. High-gate traces initialize stronger u_1 → faster transfer through the consolidation chain.

**Upgrade 4c — Inhibition of return:** The SFMA model (Biderman et al. 2023, *eLife*) prevents monopolization by the highest-gate entry: `priority = gate_signal × tag_count × suppression_multiplier` where suppression decays after each replay attempt and recovers over time. This mimics synaptic depression and ensures the replay store samples across its full inventory.

**Why it's relevant:** Each upgrade is a single-line change to the replay store data structure. Each is motivated by a specific empirical finding in the neuroscience literature. Together, they bring the gate much closer to biological replay priority without adding any controller logic.

**How to explore it:**
- Implement all three as a Phase 4b experiment: run consolidation with old gate vs. upgraded gate on a controlled set of trajectories. Measure whether tag_count vs. age changes which patterns survive to u_3+.

**Sources:**
- SFMA model: https://elifesciences.org/articles/82301
- Joo & Frank (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10659301/
- Large SWRs + consolidation (2025): https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1

---

### Idea 5: Retrieval-Frequency Weighted Coupling in the Benna-Fusi Chain

**What:** The Benna-Fusi cascade uses a fixed coupling coefficient α between each u_k and u_{k+1}. The biological critique ("intelligent plasticity" review, arXiv:2405.16922) shows this is not theoretically optimal — patterns retrieved at irregular intervals can saturate fast variables before transferring. The fix: make α depend on the pattern's retrieval count.

```
α_eff(pattern) = α_base × (1 + λ × normalized_retrieval_count(pattern))
```

Patterns retrieved frequently get faster transfer through the chain. Patterns that consolidate but are never retrieved again stall in fast variables and eventually decay. This is the compression mechanism: infrequently-queried consolidated patterns don't propagate to slow variables, which keeps the slow variables reserved for patterns that are regularly called upon.

**Why it's relevant:** This also serves as the compression regime the Dury trilogy confirms is necessary for generalization. The slow variables (u_m) hold only patterns that have been retrieved enough to justify slow-variable storage. This is structurally under-capacity: the slow variables "see" a filtered, frequently-reinforced subset of all consolidation candidates. The Dury finding (under-capacity → abstraction, at-capacity → memorization) predicts this filter should improve transfer learning on structurally similar held-out episodes.

**How to explore it:**
- Run the Phase 4 m=2 vs m=4 experiment (already on the roadmap) with and without retrieval-frequency-weighted α. Four conditions: (m=2, fixed α), (m=2, freq-weighted α), (m=4, fixed α), (m=4, freq-weighted α). Measure transfer to held-out structural episodes.

**Sources:**
- Intelligent plasticity review: https://arxiv.org/html/2405.16922v1
- Dury concept-discovery: https://arxiv.org/abs/2603.18420

---

### Idea 6: Replace Temporal Context Bags with FPE-Encoded Directed Offset Bundles

**What:** Temporal context bags (unordered bundles of co-occurred atoms) are documented to commit two named failure modes: temporal inaccuracy (can't distinguish simultaneous from sequential) and temporal fragmentation (loses relative ordering). The VSA literature now has a practical fix: Fractional Power Encoding (FPE) on a temporal axis with LCD cleanup (arXiv:2412.00488, 2024), which makes this computationally tractable.

Replace each bag bundle with:
```
atom_A ⊗ (Vt^Δt ⊗ atom_B)
```
where Δt is the signed temporal offset (positive = B comes after A, negative = before). Bundle across the temporal window. Query by applying `atom_A*` and decoding the temporal axis via LCD gradient ascent.

For discrete slot encoding (simpler, exact): follow the "Attention as Binding" paper (arXiv:2512.14709) and use permutation-indexed positions: `π^k(slot_role) ⊗ atom`, where k is the slot offset. This is algebraically identical to RoPE positional encoding in transformers — not a coincidence, but a confirmation that the substrate is doing the right thing.

**Why it's relevant:** The NeurIPS 2024 hippocampal compositionality paper (Kymn et al.) independently confirmed that the biological hippocampal-entorhinal system uses component-wise complex multiplication (standard FHRR binding) with a residue number system for multi-scale encoding. The project's FHRR substrate is already the right algebra — the only gap is that temporal bags discard ordering. Adding FPE or permutation encoding upgrades bags to ordered structures without adding a new substrate.

Anti-homunculus check: the binding is purely local. Each co-occurrence event writes one role-filler pair. Directionality comes from the sign of Δt (a geometric property of the temporal axis), not from a supervisor. Passes.

**How to explore it:**
1. Implement permutation-indexed slot encoding (discrete, simpler) as a Phase 3b variant. Compare against bags on the temporal shuffle ablation — if the temporal-structure signal improves under directed encoding vs. bags, the hypothesis is confirmed.
2. The FPE continuous encoding is the Phase 5 upgrade for multi-scale temporal hierarchy.

**Sources:**
- FPE cleanup (2024): https://arxiv.org/abs/2412.00488
- GC-VSA grid-cell structured: https://arxiv.org/html/2503.08608v1
- Attention as Binding / RoPE equivalence: https://arxiv.org/abs/2512.14709
- NeurIPS 2024 hippocampal binding: https://arxiv.org/abs/2406.18808

---

### Idea 7: REM-Phase Generative Replay as the Missing Creative Insight Mechanism

**What:** The project's replay mechanism is modeled on NREM-style precise reactivation (re-settling specific stored trajectories). The neuroscience literature distinguishes NREM (stabilization, precise reactivation, LTD) from REM (generalization, creative recombination, LTP). The REM-phase function — generating *novel* variants of stored patterns via stochastic recombination — is architecturally missing from the current design and is likely the mechanism behind "creative insight" in biological systems.

In the project's substrate, REM-phase replay could be implemented as: bundle operations over mixed-episode trajectory fragments with stochastic perturbation. Rather than re-settling a specific stored trajectory, draw two stored traces at random, mix their snapshots (preserving temporal structure within each trace but interleaving steps across traces), and settle the resulting composite. Traces that recombine into a resolved settling → novel candidate pattern. Traces that fail to resolve → discard.

The Dury PAM paper explicitly identifies cross-trajectory association as requiring entity identity persistence. The REM phase is the mechanism by which the system begins probing for such associations — not by knowing entities in advance, but by discovering which recombinations resolve.

**Why it's relevant:** The project targets creative recombination as an emergent property of retrieval dynamics (two ideas combining into a novel blend). But blend-retrieval from a fixed β landscape produces the *same* blends every time — the Hopfield dynamics are deterministic. REM-phase replay introduces controlled stochasticity specifically targeted at the recombination function. The resolved outputs of REM replay are candidates for new atoms or new layer-2 bindings — hierarchical structure discovered through creative retrieval.

**How to explore it:**
1. Implement a "dream mode" replay variant: sample two traces from the replay store, interleave their trajectory snapshots, settle the composite. Track: (a) resolution rate (what fraction of recombinations converge?) and (b) novelty rate (what fraction of resolved composites are outside the retrieval neighborhood of both source traces?).
2. The fraction of time in NREM vs. REM phase can be governed by the relative magnitudes of accuracy vs. complexity terms in variational free energy (Idea 1) — no controller picks the mode.

**Sources:**
- NREM/REM sleep as dual FEP dynamics: https://pubmed.ncbi.nlm.nih.gov/40422982/
- SCM dual-phase architecture: https://arxiv.org/abs/2604.20943
- NeuroDream latent-space dream phase: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5377250

---

### Idea 8: GIB Synergy as the Phase 5 Headline Metric for Abstraction Quality

**What:** The Generalized Information Bottleneck (GIB) framework (arXiv:2509.26327, 2025) defines synergy as information obtainable only via *joint* processing — not recoverable from any single component vector. GIB proves that synergistic representations generalize better (OOD) across architectures where the original IB framework failed. In the project's substrate, a schema or role-filler structure at Phase 5 should be synergistic by design: the schema cannot be decoded from any single atom in the binding — it requires the full bound composite. If synergy is measurable (the paper provides a computable estimator), it becomes a principled headline metric for Phase 5.

**Why it's relevant:** The project already has the headline-metric principle (from the 2026-05-09 note: one headline metric per phase, drill-downs explain anomalies). Phase 5 currently lacks a clear headline metric for "has the system actually learned structure?" The synergy score on consolidated schema representations vs. individual atoms is that metric: high synergy = the binding is doing meaningful compositional work. The Benna-Fusi cascade's cascaded compression is structurally a synergy-inducing bottleneck — the slow variables cannot encode any single episode's details and must integrate across episodes. GIB predicts this should produce synergistic (generalizing) representations as a consequence of the cascade architecture.

**How to explore it:**
1. Read the GIB paper's synergy estimator (average interaction information) and implement it for FHRR-bound composites.
2. Phase 5 experiment: measure synergy of role-filler bindings (atom ⊗ role) vs. individual atoms. If synergy is higher for bindings, the cascade is doing abstraction work.
3. Use synergy score as the Phase 5 headline metric with cap-coverage on held-out compositional queries as the drill-down.

**Sources:**
- GIB framework: https://arxiv.org/abs/2509.26327
- Dury compression trilogy: https://arxiv.org/abs/2603.18420

---

### Idea 9: Partial Credit Along the Settling Path (Reverse Replay for Credit Assignment)

**What:** The current design gives u_1 consolidation credit only to the final resolved pattern. Biological reverse replay (documented in the SWR literature) propagates credit backward from the outcome to prior states — used for credit assignment in planning and decision making. In the project's settling trajectory, the snapshots represent a path through the energy landscape: earlier steps explored, later steps committed. The final resolved state earns all the credit; the intermediate attractors that shaped the trajectory earn none.

Implementing temporal-difference credit assignment along the settling path:
```
u_1_credit[step] = base_credit × discount^(n_steps - step)
```
Attractors that appeared near the end of settling get more credit; those that appeared early get less. This turns consolidation into a TD update along the settling path — earlier attractors that consistently appear across many trajectories before resolution accumulate credit even though they don't resolve.

**Why it's relevant:** This is the mechanism by which "intermediate concepts" — patterns that frequently appear in the middle of settling trajectories but never as final outputs — can enter the codebook. These patterns may be the most valuable: they are the step-stones that connect cues to their retrievals. Without partial credit, they are invisible to consolidation.

Anti-homunculus check: credit is a mathematical function of the trajectory itself. No supervisor chooses which intermediate attractors matter. The discount rate is an architectural constant. Passes.

**How to explore it:**
1. Implement reverse-credit sweeping over `TrajectoryTrace.snapshots`. Compare codebook composition after Phase 4 with vs. without partial credit: do intermediate patterns enter the codebook, and are they more useful for future retrieval?

**Sources:**
- Biological forward/reverse replay: https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/
- PNAS Nexus replay cascades: https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348

---

### Idea 10: Cross-Attention Injection as the Phase 7 Workspace-to-LLM Interface

**What:** The dominant LLM memory interface is text prepending (RAG: retrieve context → prepend to prompt). The two strongest 2024–2026 approaches bypass text entirely: MemoryLLM/M+ (arXiv:2502.00592) injects compressed hidden-state memory vectors into each LLM transformer layer via cross-attention, extending effective context from 20K to 160K tokens without fine-tuning the LLM. Coconut (arXiv:2412.06769) feeds LLM hidden states back as next-input embeddings, enabling continuous-thought latent reasoning.

Phase 7 workspace-to-LLM interface design using the M+ pattern:
1. Hopfield settling produces settled FHRR vector `z` (D=4096 complex, 8192 real dimensions).
2. Learned projection `W_proj: ℝ^8192 → ℝ^{k × d_model}` maps `z` to k "memory tokens" in the LLM's hidden space.
3. These memory tokens are presented as cross-attention keys/values at each LLM layer (or selected layers).
4. LLM generates response conditioned on these memory tokens — no settled state decoded to text.

The LLM is pure voice: its autoregressive generation is conditioned on the pre-linguistic workspace state. W_proj is the only learned parameter; the LLM does not need fine-tuning. A LoRA adapter suffices.

**Why it's relevant:** The project is already architecturally ahead of the mainstream RAG paradigm. RAG decodes the memory state into text before conditioning the LLM; the project commits to conditioning on a pre-linguistic settled state. M+ proves this pattern works. The differentiation vs. HippoRAG (the closest published architecture) is exactly here: HippoRAG uses graph+PageRank and returns text; the project uses FHRR+Hopfield and returns a latent vector. That is a meaningful architectural differentiator with testable predictions.

Anti-homunculus check: W_proj always fires. There is no gate or switch deciding when to condition the LLM. The settled state is always the conditioning. Passes.

**How to explore it:**
1. Phase 7 prototype: freeze a small local LLM (e.g., Llama 3.2 1B). Add a trainable W_proj. Feed Phase 2/3 settled workspace states as cross-attention keys/values. Fine-tune W_proj on a held-out text reconstruction task. Measure whether LLM outputs are coherent with the workspace content.

**Sources:**
- MemoryLLM / M+: https://arxiv.org/abs/2502.00592
- Coconut continuous thought: https://arxiv.org/abs/2412.06769
- HippoRAG (NeurIPS 2024): https://arxiv.org/abs/2405.14831

---

## Cross-Cutting Themes

### Theme 1: The FEP vocabulary dissolves the diagnostics-to-actuators problem rather than solving it

The project has been searching for a mechanism that converts diagnostics into actuators without a controller. FEP reframes the question: the diagnostic and the actuator are the same object viewed at different temporal resolutions of the free energy gradient. The project doesn't need a new mechanism — it needs to audit existing mechanisms against the FEP gradient criterion. Importantly, this means the Benna-Fusi chain, the Hopfield settling dynamics, and the replay gate are *already* FEP-compatible in their general shape. The codebook Hebbian updates are already performing approximate free energy minimization. The anti-homunculus filter is the FEP criterion in disguise.

### Theme 2: The blend/retrieve "failure" is a generalization/memorization "feature"

Every 2024–2026 Hopfield theory paper converges on the same reframing: the blend regime (low β, global centroid attractor) and the sharp retrieval regime (high β, individual pattern attractors) are not degraded versions of each other — they are distinct computational modes corresponding to generalization and memorization respectively. The project should not try to eliminate the regime structure; it should architect a system that *selects* the appropriate mode per cue. The LSR kernel and IDP together give two orthogonal mechanisms to achieve this selection through local geometry.

### Theme 3: Compression-forces-abstraction is robustly confirmed from three independent directions

Dury's trilogy (PAM → AAR → Concept-Discovery) shows that under-capacity training extracts transferable structure. NeuroDream shows latent-space replay produces 17.6% zero-shot transfer improvement. The GIB framework provides formal proof that cascaded compression induces synergistic (generalizing) representations across architectures. Three independent groups, three domains, same conclusion. The Benna-Fusi chain with retrieval-frequency-weighted coupling is already the right shape for an under-capacity bottleneck — the slow variables hold only patterns that have been re-activated enough to justify slow-variable storage.

### Theme 4: The project's workspace design satisfies Global Workspace Theory without implementing it

Three independent 2025–2026 AI architecture papers (Global Workspace Agents, Cognitive Workspace, Theater of Mind) all converge on the same broadcast mechanism the project already has. The settled FHRR workspace state makes retrieved information globally available to all downstream processing — this is exactly GWT's broadcast function. The project doesn't need to implement GWT; it already is GWT. This is useful for communication and framing in Phase 7 documentation.

### Theme 5: Temporal ordering is a first-class information channel that bags discard

TiMem's failure mode taxonomy (temporal inaccuracy, temporal fragmentation) is a rigorous argument against unordered bags. The hippocampal time-cell literature (Nature 2024) shows the brain maintains separate fast-discrete (time cell) and slow-continuous (ramping cell) temporal signals. The VSA literature now has three concrete algebraic approaches to directed temporal binding (FPE, permutation, GHRR non-commutative). The project's Phase 3/4 temporal co-occurrence work is in the right direction — the upgrade from bags to directed binding is a substrate-level change that would improve every downstream phase.

---

## Challenges and Counterarguments

**On the LSR kernel:** The Epanechnikov/LSR energy is a new result (2026 paper). The emergent-memory intermediate regime has been theoretically characterized but not yet empirically validated in a real memory task at D=4096. The phase diagram at this dimensionality needs to be mapped before committing to an LSR architecture for Phase 3+. It may turn out the intermediate β zone is very narrow and sensitive to memory load.

**On FEP integration:** The VSA/FHRR community and the FEP community have not met — there is essentially no literature at the FHRR × active inference intersection. The bridge runs through the Hopfield network (which both communities share), but the FHRR binding structure has no direct FEP counterpart. Any integration would be novel, not drawing on established prior work. The audit criterion (does this have a natural description as a free energy gradient?) is useful; a full FEP mathematical treatment of the FHRR substrate would require original theoretical work.

**On temporal FPE offset binding:** No published paper has validated named temporal roles (BEFORE-role, DURING-role, AFTER-role) as VSA binding primitives. The algebraic foundation (FPE, permutation) exists, but the specific claim that directed causal structure is preserved through the binding-bundling-retrieval cycle at scale has not been empirically demonstrated. The permutation approach (discrete, exact, no continuous decoder) is lower risk as a first implementation.

**On REM-phase generative replay:** This is the highest speculative idea in this document. No published system has implemented generative replay from mixed FHRR trajectory fragments. The creative insight function is real in biological systems but may require entity-identity persistence (Dury's PAM finding) as a precondition — which the project doesn't yet have. REM-phase replay without entity persistence may produce noise rather than novel associations.

**On cross-attention injection (Phase 7):** The M+ approach requires training the projection layer W_proj, which means the LLM and the Hopfield substrate need to be jointly trained or at minimum sequentially fine-tuned. Without shared training, the LLM has no learned basis for interpreting FHRR-derived memory tokens. This is a real integration cost that should be scoped before Phase 7.

---

## Rabbit Holes Worth Following

1. **arXiv:2505.22749 (Spisak & Friston, May 2025) — Self-orthogonalizing attractor networks from FEP.** This paper was published three weeks ago, is directly about the intersection of FEP and attractor networks, proves that attractor orthogonalization is a free energy consequence (directly relevant to the NC1 reformulation and codebook collapse concerns), and includes a GitHub repository with simulation code. Should be read before Phase 3 Session 1. https://arxiv.org/abs/2505.22749

2. **arXiv:2406.18808 (Kymn et al., NeurIPS 2024) — Hippocampal compositionality via residue number system.** This paper models hippocampal-entorhinal circuits using exactly the algebra the project already has (component-wise complex multiplication, resonator networks, path integration via FPE carry-free identity). It is essentially a normative computational model of what the project's FHRR + Hopfield substrate is doing at the systems level. The RNS multi-scale encoding with exponentially spaced moduli (~1.42× factor) is a direct design input for Phase 5 temporal hierarchy. https://arxiv.org/abs/2406.18808

3. **arXiv:2506.10801 — Dense Associative Memory with Epanechnikov Energy.** The 2026 paper demonstrating the LSR kernel's emergent-memory intermediate regime. This is the first known kernel that achieves coexistence of memorized originals and emergent blend attractors. Very new; should be read before running any Hopfield architecture comparison. https://arxiv.org/html/2506.10801

4. **Resonator networks + emergent codebook gap.** The resonator network literature (for hierarchical VSA factorization) assumes a known codebook. The project's Hebbian codebook grows through self-organization — the two assumptions are in direct tension. No paper has bridged this. For Phase 5 layer-2 discovery (binding relationships between atoms), resonator factorization would be the natural retrieval mechanism if the codebook were fixed. The design question: periodically snapshot the emergent codebook as the resonator's search space, updated on a slow timescale. https://openreview.net/forum?id=FNrZd3Ls1d

5. **HippoRAG v2 (monitor).** HippoRAG is the closest published architecture to the project's design philosophy (neocortex/hippocampus split, pattern completion as retrieval). The Phase 7 differentiation hypothesis is: Hopfield energy-settling degrades more gracefully under noisy cues than graph-PageRank. The project's Phase 2 cue-degradation experiments are the right setup for this comparison. Keep an eye on whether HippoRAG v2 moves toward vector/energy-state conditioning (which would reduce the project's differentiation). https://arxiv.org/abs/2405.14831

6. **The stochastic Hopfield critical noise zone (arXiv:2509.17152).** The β ∈ (0.001, 0.01) crossover zone the project identified may overlap with a dynamical criticality regime where the stochastic MHN develops long-range temporal correlations — distinct from the blend/retrieve phase transition. This could mean the crossover zone is not a broken retrieval state but a critical state with interesting computational properties. Worth mapping the stochastic phase diagram at D=4096 alongside the deterministic phase diagram.

---

## Sources by Research Angle

### Hippocampal Replay & Consolidation
- Joo & Frank (2023), Science: https://pmc.ncbi.nlm.nih.gov/articles/PMC10659301/
- Mattar et al. (2018), Nature Neuroscience: https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620/
- SFMA (Biderman et al. 2023), eLife: https://elifesciences.org/articles/82301
- Large SWRs + memory (2025), Neuron: https://www.cell.com/neuron/abstract/S0896-6273(25)00756-1
- Aitchison et al. recall-gated plasticity, eLife: https://pmc.ncbi.nlm.nih.gov/articles/PMC11257680/
- Intelligent plasticity review: https://arxiv.org/html/2405.16922v1
- Buzsaki SWR review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6794196/
- PNAS Nexus replay cascades: https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348

### Active Inference & FEP
- Spisak & Friston (May 2025): https://arxiv.org/abs/2505.22749
- FEP + attractor networks (Cerebral Cortex 2025): https://pubmed.ncbi.nlm.nih.gov/40422982/
- Nature Human Behaviour memory consolidation (2023): https://www.nature.com/articles/s41562-023-01799-z
- pymdp: https://github.com/infer-actively/pymdp
- Scale-free active inference: https://arxiv.org/abs/2407.20292

### Hierarchical VSA & Temporal Binding
- NeurIPS 2024 hippocampal compositionality: https://arxiv.org/abs/2406.18808
- FPE cleanup (2024): https://arxiv.org/abs/2412.00488
- GC-VSA (2025): https://arxiv.org/html/2503.08608v1
- GHRR non-commutative: https://arxiv.org/abs/2405.09689
- Attention as Binding: https://arxiv.org/abs/2512.14709
- SRMU streaming memory: https://arxiv.org/html/2604.15121
- TiMem temporal tree: https://arxiv.org/html/2601.02845v1
- Human time cells (Nature 2024): https://pubmed.ncbi.nlm.nih.gov/39322671/
- Resonator networks: https://openreview.net/forum?id=FNrZd3Ls1d

### Hopfield Temperature Dynamics
- LSR/Epanechnikov energy (2026): https://arxiv.org/html/2506.10801
- LSE vs LSR thermal robustness: https://arxiv.org/html/2603.13350
- Phase transitions in MHN: https://arxiv.org/abs/2311.18434
- IDP Hopfield (2025): https://arxiv.org/html/2411.05849v1
- Sparse Hopfield (NeurIPS 2023): https://arxiv.org/html/2309.12673
- Hopfield-Fenchel-Young Networks: https://arxiv.org/html/2411.08590
- Dynamic Manifold Hopfield: https://arxiv.org/html/2506.01303

### Compression & LLM Integration
- Dury concept-discovery: https://arxiv.org/abs/2603.18420
- Dury AAR: https://arxiv.org/abs/2604.20850
- GIB framework: https://arxiv.org/abs/2509.26327
- NeuroDream: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5377250
- MemoryLLM / M+: https://arxiv.org/abs/2502.00592
- HippoRAG: https://arxiv.org/abs/2405.14831
- Coconut: https://arxiv.org/abs/2412.06769
- MemOS: https://arxiv.org/abs/2505.22101
- GWT for LLMs: https://arxiv.org/abs/2604.08206
- SCM sleep-consolidated memory: https://arxiv.org/abs/2604.20943
