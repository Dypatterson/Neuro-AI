I now have the exact bound, the 124 directional NULL (directional buys +0.006, ~1/15 of global), and the 119 d_eff-collapse failure mode (341→165) fully grounded. I have everything I need to curate. Let me produce the ranked menu.

---

# RANKED MENU — Entirely-Different Phase-3 Substrates (beyond the flat co-occurrence codebook)

**The bound being escaped (Report 123 §4, verified against the report):** paradigmatic ("substitutability," king/queen) structure lives in the **subdominant modes** of the SPPMI co-occurrence operator. A *global* SVD reaches them (**+0.109**, king/queen 0.222); *local iterative* `S'@G` pull+repulsion runs to the **dominant collocational** mode and reaches **≤+0.021** clean. The killer gate is `corr(log cooc, drift) CI-hi < 0.15` — every collapse-free positive flat-code cell *fails* it (clustering is collocational, not paradigmatic). Report 124 confirmed the bound holds for **directional/successor** operators on a flat code (+0.006, ~1/15 of global). Report 119 confirmed that a **contrastive objective on the flat value code collapses d_eff** (341→165). The deduplicated insight across 28 candidates: there are **exactly four distinct structural escape-routes**, and most candidates are variations on them.

**The four escape-routes (the real taxonomy under the 28 memos):**
- **(I) Diagonalize a *different operator* (transition/successor, not symmetric co-occurrence), and read its *subdominant eigenbasis* via local replay dynamics** — the "grid-cells-are-SR-eigenvectors" route. *Tension: 124 already nulled the successor operator on a flat code; the live claim is the **eigenbasis**, not the operator-as-growth-rule.*
- **(II) Factor structure × content into separate axes, recombine by FHRR binding** (TEM-family) — paradigmatic = "same structural slot, swappable filler." Promotes the subdominant axis to *dominant in a different space*.
- **(III) Replace the linear average with a *nonlinear/competitive/sparse demixing* dynamic** (rectification, k-WTA, assembly cap, branch-committee, SOM quantization) — the bound is a property of *linear* iteration; a nonlinear fixed point is not the leading eigenvector.
- **(IV) Add a coding axis co-occurrence does not have** (relative time/phase, timescale, polychronous delay) — paradigmatic signal moves to an *orthogonal channel* the SPPMI spectrum can't see.

I rank by **(upside on the bound) × (cheapness of the kill-test) ÷ (reinvents-killed risk)**.

---

## TIER 1 — Highest-upside, cheapest-to-falsify, genuinely-different

### 1. **SR-Eigenbasis Map** — successor representation, read as its *subdominant spectral basis* (Route I)
- **Bio + primary:** Stachenfeld, Botvinick & Gershman 2017 (*Nat Neuro*, "hippocampus as a predictive map"; grid cells = SR eigenvectors); the 2026 arXiv "Word Class Representations Spontaneously Emerge from Successor Representations" (paradigmatic word-classes cluster in SR space — *the* on-target literature result); Sprekeler 2024 (SFA ≡ SR).
- **Mechanism:** store a learned transition operator M (not per-token vectors); build M_SR=(I−γT)⁻¹ via **TD-on-replay**; the token code = its projection onto the **top-k eigenvectors of M_SR**, reached by replay-driven power iteration (local Oja/Hebbian), never `torch.svd`.
- **Different-from-flat:** the code is a token's *position in the operator's spectrum*, a function of the whole transition graph, not a centroid of neighbors.
- **Bound-escape:** the SR eigenspectrum *is* the subdominant-mode structure the global SVD reaches — but reached by local replay-power-iteration over the *transition* operator. The dominant SR eigenvector is the collocational common-mode; the *subdominant* eigenvectors are the substitutability axes.
- **Project-fit:** anti-homunculus ✅ (TD-on-replay, power iteration are local); batch-offline ✅ (native consolidation idiom); FHRR partial-extension (M is a new object, but the eigenbasis vectors live in the hypervector space and read via existing MHN).
- **Cheap test:** build empirical M_SR, eigendecompose, run Report-123's gate (king/queen lift, collapse-free, corr<0.15) on the **SR-eigenvector embedding** vs co-occurrence centroid. ~hours, pure numpy, reuses `experiments/62`/`63` harness. Greenlight if it clears +0.021→+0.109 *and* passes corr<0.15.
- **REINVENTS-KILLED:** **SHARES-A-FAILURE-MODE (guarded).** 124 nulled the successor operator *as a flat-code growth rule*. The non-trivial claim is the **eigendecomposition** — the cheap test MUST use the eigenbasis, or it relapses to 124. *This is the crux discriminator and the reason it's #1 only if the eigenbasis read is explicit.*

### 2. **Hebbian-TEM** — factorized structure × content, recombined by FHRR bind (Route II)
- **Bio + primary:** Whittington et al. 2020 (*Cell*, Tolman-Eichenbaum Machine); Behrens et al. 2018 ("What is a cognitive map?"). MEC structural code `g` ⊗ LEC content `x` = HPC conjunction `p`.
- **Mechanism:** represent a token as a factored FHRR pair `g ⊛ x`; grow `g` from the *transition graph* by offline path-integration (`g_{t+1}=g_t ⊛ r`), write conjunctions associatively (Hebbian) into the MHN — **not** backprop.
- **Different-from-flat:** two factors + a relational operator; paradigmatic = property of the *structure* factor, content-independent. "King:queen :: man:woman" becomes a displacement in g-space — impossible in one entangled vector.
- **Bound-escape:** re-bases the previously-*subdominant* substitutability axis into the *dominant* axis of an explicit structural factor grown from *transitions* (a different operator than SPPMI), shared across content by construction.
- **Project-fit:** anti-homunculus ✅ (path-integration local; binding is representational, no arbiter); batch-offline ✅; **FHRR-native ✅✅ — `g⊛x` is literally the existing bind/unbind/MHN.** Strongest substrate-fit in the whole pool.
- **Cheap test:** factor the *directed* transition operator into k≈8 relational slots (NMF/clustering); give each word a soft slot-distribution; test cos(slot(king),slot(queen)) ≫ random, beating +0.021 toward +0.109, with cooc≈0. One afternoon, numpy. Reuses the SR-eigenbasis from #1 as the slot-basis stand-in.
- **REINVENTS-KILLED:** **GENUINELY-NEW (one guard).** Distinct from 124 (which kept a flat code + direction; this adds a *second factor*) and 119 (no contrastive loss). **Mandatory pre-commit: grow `g` by Hebbian path-integration, NOT backprop** (TEM-as-published is backprop — that would violate the bet + the online-error ban). Whether Hebbian path-integration alone factorizes is the open risk the cheap test must probe.

### 3. **Slow-Feature substrate (Bio-SFA)** — grow the code from the *slow* modes, spectrally inverted by a local rule (Route III)
- **Bio + primary:** Wiskott & Sejnowski 2002 (Slow Feature Analysis; reproduces place cells); **Lipshutz, Pehlevan & Chklovskii (Bio-SFA, arXiv 2010.12644)** — SFA as a *local Hebbian/anti-Hebbian* network.
- **Mechanism:** over the replayed token trajectory, run Bio-SFA's local feedforward-Hebbian / lateral-anti-Hebbian updates to extract the slowest-varying directions; token code = projection onto the slow axes.
- **Different-from-flat:** SFA *minimizes* the temporal derivative (extracts the *opposite end* of the spectrum from dominant-variance iteration) — a slow latent, not a centroid.
- **Bound-escape:** the most surgical fit to the *exact shape* of the bound. The bound says "local iteration reaches dominant, not subdominant, modes"; SFA is **spectrally inverted by construction** and reaches the bottom of the spectrum via a *proven local rule*. If "paradigmatic = swappable without changing the slow context" (king↔queen leave the royal-context slow-variable unchanged), paradigmatic structure *is* a slow feature.
- **Project-fit:** anti-homunculus ✅ (pure local Hebbian/anti-Hebbian, no arbiter); batch-offline ✅ (runs on replayed stream); FHRR extension-not-replacement (new growth rule, slow axes carried as FHRR vectors).
- **Cheap test:** closed-form linear SFA (generalized eig on signal vs time-derivative covariance) on the existing windowed token-trajectory; measure king/queen lift vs +0.021/+0.109; if the closed-form clears it, commit to *local* Bio-SFA (whose value-add is reaching it locally). Half a day, reuses the 123 gate.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Different objective (temporal slowness, not co-occurrence frequency). Honest tension: SFA's closed form is a *global* generalized-eigendecomposition (flashlight only); the build must show **local Bio-SFA approaches it** — exactly the project's "global = flashlight, local = mechanism" discipline.

### 4. **Assembly-overlap codebook** — Assembly Calculus; the cap (k-WTA) clips the common-mode every step (Route III)
- **Bio + primary:** Papadimitriou, Vempala, Mitropolsky, Collins & Maass 2020 (*PNAS*, "Brain computation by assemblies of neurons"); Dabagia et al. 2022 (assemblies classify well-separated distributions).
- **Mechanism:** one sparse *assembly* (k-subset of D neurons) per token, grown by project+cap+associate during sleep; similarity = assembly overlap under a recurrent attractor.
- **Different-from-flat:** representation is a sparse *membership set*, similarity is set-overlap under a learned attractor — not cosine of accumulated co-occurrence mass. `associate` raises overlap *via a shared third assembly* (transitive), with no "must co-occur to be near" constraint.
- **Bound-escape:** the cap (k-WTA) is a hard per-step sparsity nonlinearity that **clips the dense collocational common-mode every step** — a *local nonlinear analog of subtracting the dominant SVD component*. king/queen acquire overlap through shared context-assemblies without co-occurring.
- **Project-fit:** anti-homunculus ✅ (cap + Hebbian + E-I, no arbiter); batch-offline ✅; substrate-change but *adjacent* (sparse VSA cousin; FHRR can still bind roles onto assemblies). **The escape operation (cap) is intrinsic to normal operation — it cannot quietly degenerate back to the flat code.**
- **Cheap test:** implement project+cap+associate over co-occurrence windows on a 5k-token slice; measure overlap(king,queen) − overlap(king,random) with corr<0.15. Hours, no GPU.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** The cap nonlinearity is exactly what all killed linear variants (121/123/124) lacked.

---

## TIER 2 — Strong, genuinely-different, slightly higher cost or narrower scope

### 5. **Self-Organizing Context Map (SOM / Growing Neural Gas over context-signatures)** (Route III)
- **Bio + primary:** Willshaw & von der Malsburg 1973/76; **Ritter & Kohonen 1989 "Self-organizing semantic maps"** — a SOM on *average-context vectors* places paradigmatically-similar (incl. non-co-occurring) words in adjacent map regions (*direct historical evidence the escape is real*); Fritzke 1994 Growing Neural Gas (learns its own topology by competitive-Hebbian edge growth).
- **Mechanism:** build each token's context signature (FHRR bundle of neighbors); run SOM/GNG — BMU competition + neighborhood pull + edge birth/prune. The grown object is a *graph of prototype units + adjacency edges*, not a per-token vector.
- **Different-from-flat:** adds two new state variables the flat code lacks — *prototype units tokens quantize onto* and *learned adjacency edges*. "Similar" = graph-distance / shared-unit.
- **Bound-escape:** WTA quantization on *second-order* context signatures is *not* a power-iteration of `S'` — it minimizes within-cell signature variance (clusters by similar-neighborhood = paradigmatic), which Ritter-Kohonen already demonstrated. Edges encode topology where paradigmatic relation is a short geodesic between never-co-firing units.
- **Project-fit:** anti-homunculus ✅ mostly (BMU=argmax-similarity, neighborhood-pull local; the one tension is error-driven *unit insertion* — frame as homeostatic density growth or use fixed-size SOM = clean); batch-offline ✅; FHRR-compatible (signatures are bundles, BMU uses existing similarity).
- **Cheap test:** run a tiny SOM/GNG (200–500 units) on SimLex-vocab context signatures; measure paradigmatic map-adjacency/shared-unit rate − random, with corr<0.15. One afternoon, numpy.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Uses context signatures (like the killed WS-InfoNCE *target*) but the *mechanism* is quantization+topology, not a contrastive objective on a flat code (SOM's neighborhood update *spreads* units — opposite of 119's d_eff collapse).

### 6. **Nonnegative Similarity-Matching codebook (NSM; rectified Hebbian/anti-Hebbian manifold-tiling)** (Route III)
- **Bio + primary:** Pehlevan & Chklovskii; **Sengupta, Tepper, Pehlevan & Chklovskii 2018 (NeurIPS, "Manifold-tiling localized receptive fields")** — local Hebbian/anti-Hebbian rectifying network ≡ symmetric NMF / soft-clustering; optimal solutions are localized place-cell-like tilings.
- **Mechanism:** two-population net `y=[Wx−My]_+` run to fixed point; offline replay updates W (Hebbian) / M (anti-Hebbian); token code = rectified hidden response.
- **Different-from-flat:** a *rectifying nonlinearity inside a recurrent competitive loop* — the fixed point is a soft-clustering/manifold-tiling, **not** the leading eigenvector.
- **Bound-escape:** the bound is for *linear* `S'@G`. Rectification means NSM does not maximize variance along the dominant collocational axis; it preserves the *full* similarity structure (incl. subdominant) under nonnegativity. **It is the brain-legal cousin of the global SVD the project uses as a flashlight.**
- **Project-fit:** anti-homunculus ✅ (purely local rules, no supervisor); batch-offline ✅; FHRR hybrid (NSM net is a real-valued extension; outputs re-phased into FHRR for the recall floor).
- **Cheap test:** 2-line NSM fixed-point + local update in numpy on the *existing* SPPMI context vectors; grow G_NSM; straight through the Report-123 gate. ~1 day. Kill if collapse-free specificity ≤ +0.021.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Explicitly the non-flat, nonlinear-competitive alternative 123 §4 gestures at; the rectifying recurrence is the new degree of freedom. (Closely related in spirit to #4 — both are nonlinear-competitive; NSM is the most direct SVD-analog, A1 the most substrate-intrinsic.)

### 7. **TCM-FHRR drifting-context-phase substrate** — paradigmatic from the order/position channel (Route IV)
- **Bio + primary:** Howard & Kahana Temporal Context Model; **Howard, Shankar & Jagadisan 2011** (semantics from gradually-changing temporal context); **Kelly et al. 2020 Holographic Declarative Memory** (BEAGLE lineage). BEAGLE's load-bearing result: **order information yields paradigmatic structure; context co-occurrence yields syntagmatic.**
- **Mechanism:** maintain a drifting FHRR context vector `c_t=ρ·c_{t-1}⊕φ(token)`; grow each token's code as the bundle of context states + an **order channel** (`Π^k` position powers). Two words at the same offset in parallel sentences acquire similar codes with zero co-occurrence.
- **Different-from-flat:** growth target is a recurrent *drifting trajectory* (second-order by construction), plus a position-spectrum the centroid lacks.
- **Bound-escape:** the order/position channel is **orthogonal to the co-occurrence operator the bound is about**; BEAGLE empirically separates the two channels. Paradigmatic similarity rides the channel the bound doesn't constrain.
- **Project-fit:** anti-homunculus ✅ (leaky integrator + bundle, pure local recurrence); batch-offline ✅ (drift replayed in a sleep pass); **FHRR-native ✅✅ (ρ-leak, bundle, Π^k are existing ops; zero substrate change).**
- **Cheap test:** ~30 lines numpy — build a pure *context* vector and a pure *order* vector per token; compare king/queen paradigmatic specificity per channel. Kill the whole family's premise if order ≤ context. Greenlight if order clears +0.109.
- **REINVENTS-KILLED:** **GENUINELY-NEW** with one sharp guard: if you collapse the position channel to k=±1 you degenerate into Report 124's successor operator — the position *spectrum* `{Π^k}` is what makes it different.

### 8. **Benna-Fusi metaplastic cascade edges** — separate collocational from paradigmatic by *timescale* (Route IV)
- **Bio + primary:** **Benna & Fusi 2016 (*Nat Neuro*, "Computational principles of synaptic memory consolidation")** — each synapse is a multi-timescale cascade; fast variables capture recent, slow integrate long history. *This IS a consolidation model — the sleep/wake split is native.*
- **Mechanism:** replace each scalar weight (in the existing co-occurrence accumulation) with a 3+-variable Benna-Fusi cascade; read structure off the **slow variables only**; replay writes the fast end, consolidation transfers to slow.
- **Different-from-flat:** each connection is a dynamical object with internal state across timescales — a built-in fast(syntagmatic)/slow(paradigmatic) decomposition a scalar code cannot hold.
- **Bound-escape:** reframes the bound as *timescale-mixing*, not spectral. Collocational signal is fast/bursty (king-the); paradigmatic is slow/diffuse (king/queen share contexts across the whole corpus, never adjacent). A scalar weight *sums* them, swamping the slow mode; the cascade **low-pass-filters them apart** — a local operation isolating the slow context-similarity.
- **Project-fit:** anti-homunculus ✅ (cascade = local linear dynamical system per edge); **batch-offline ✅✅ (Benna-Fusi *is* the consolidation model)**; FHRR add-on (default to scalar = byte-identical, matching the project's default-off discipline).
- **Cheap test:** **the single cheapest discriminating probe in the pool** — take the *exact* 121/123 WikiText pipeline, replace the scalar co-occurrence accumulation with a 3-variable cascade, read paradigmatic specificity off the slow variable. If corr(cooc,drift) drops below 0.15 while specificity stays positive (vs the scalar's +0.41/+0.82), that's a same-harness demonstration. Reuses `experiments/61`'s gauge-free gate ~verbatim; ~half a day.
- **REINVENTS-KILLED:** **SHARES-A-FAILURE-MODE (honestly flagged).** It still ingests co-occurrence statistics — if paradigmatic isn't separable by timescale, it collapses to 121/123. *But it's the cheapest falsifiable test of a sharp, untouched mechanism — run it first as near-free insurance.*

---

## TIER 3 — High novelty, higher substrate cost or higher uncertainty

### 9. **Clone-Structured Cognitive Graph (CSCG)** — a learned latent automaton, not a codebook (Route I/II hybrid)
- **Bio + primary:** George, Lázaro-Gredilla et al. 2021 (*Nat Comms*, clone-structured graphs); "Space is a latent sequence" (*Sci Adv* 2024). Each token has multiple *clones* (latent states for the same surface token in different contexts); splitter cells, transitive inference, transferable schemas emerge.
- **Mechanism:** a sparse latent transition graph; offline EM-during-replay splits tokens into context-specific clones and learns the latent transition matrix. Paradigmatic = clones in structurally-equivalent graph positions.
- **Different-from-flat:** a probabilistic graphical model / latent automaton — one surface token → many latent states. Natively solves polysemy and transitive inference (both outside a flat code).
- **Bound-escape:** clones give a token *multiple* positions; paradigmatic neighbors = graph-structural equivalence of clones' transition profiles (second-order, context-conditioned) — the subdominant-spectral structure accessed *combinatorially* (graph motifs) rather than spectrally.
- **Project-fit:** anti-homunculus ⚠️ (clone-splitting must be framed as energy/likelihood-driven state allocation during replay — a local geometric dynamic, NOT "split this token" as an if-then); batch-offline ✅ (EM = consolidation); **FHRR ✗ — genuine substrate change** (graphical model). Highest novelty, highest integration cost.
- **Cheap test:** run a public CSCG/CHMM learner on a WikiText stream; check (a) polysemous words get multiple clones; (b) paradigmatic pairs' clones sit at structurally-equivalent positions even when words never co-occur.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Guard: the EM/likelihood step must be local replay-driven allocation (else trips anti-homunculus), and must not become "a vector DB of clones + summaries" (forbidden memory sense).

### 10. **Grid-cell / Spatial-Semantic-Pointer code (fractional-power FHRR)** — multi-scale periodic re-encoding (Route II/IV)
- **Bio + primary:** Gardner et al. 2022 (*Nature*, toroidal grid-cell manifold); Komer/Stewart/Voelker/Eliasmith Spatial Semantic Pointers; **fractional power encoding is HRR/FHRR with phase scaling** (arXiv 2503.08608, 2412.00488).
- **Mechanism:** encode a token's *relational-graph coordinates* (from #1's SR-eigenvectors) as a fractional-power FHRR vector `z=⊛ⱼ baseⱼ^{φⱼ}`, with a *bank of modules* at different frequencies; fractional binding = phase scaling, native to FHRR.
- **Different-from-flat:** a continuous, translation-equivariant manifold *with a group action* (bind = translate) — "move from king by the man→woman displacement" is well-defined. Multi-scale modules give hierarchy for free.
- **Bound-escape:** two-part — coordinates come from the SR spectrum (#1's escape), AND the periodic multi-scale re-encoding puts collocational (high-freq/local) and paradigmatic (low-freq/global) structure in **different modules** (orthogonal frequency bands) — the frequency-domain analog of the SVD's mode-separation, as a fixed biological basis.
- **Project-fit:** anti-homunculus ✅ (fractional binding fixed algebra; CAN/path-integration local); batch-offline ✅; **FHRR-native ✅✅ (one-line phasor extension).** *On its own it's an encoding scheme, not a complete growth story — it needs #1's coordinates.*
- **Cheap test:** take 2–8 SR-eigenvector coordinates/token, encode as fractional-power FHRR with a small module bank; verify the *algebra* (`z(king)⊛z(man)⁻¹⊛z(woman)` cleans up to `z(queen)`?) and Report-123 specificity.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Guard against collapse-to-#1: must earn its keep via multi-scale module separation + clean group-action algebra; if a single-scale version just reproduces #1's eigenvectors, drop the grid framing.

### 11. **Relational-edge VSA substrate (Kanerva "Dollar of Mexico")** — structure on *edges*, read algebraically (Route II)
- **Bio + primary:** Kanerva 2010 ("What's the Dollar of Mexico?"); Gayler & Levy 2009 (distributed analogical mapping). Analogy = a single VSA unbinding — *the project's own FHRR algebra used for its designed purpose.*
- **Mechanism:** grow a relation memory — write bound edges `e=a⊛r⊛b` during replay; substitutability read by the algebra (same role-profile), not node geometry.
- **Different-from-flat:** structure = membership in bound relational edges; the carrier is the *binding operation* (which the project has used only for memorization, never to grow structure).
- **Bound-escape:** substitutability is *natively an edge property* (Kanerva: falls out of unbinding with zero neighbor-averaging). Moving structure onto bound edges + role-profiles orthogonalizes the dominant collocational node-statistic from the paradigmatic signal — the subdominant-mode problem *dissolves* because no node-average is ever formed.
- **Project-fit:** **maximally FHRR-native (zero new primitives)**; anti-homunculus ✅ (superposition/unbind/cleanup are local). **Tension:** roles `r` must come from somewhere without a homunculus — the legal version derives them from emergent position/dependency structure already in `encoding.py`.
- **Cheap test:** role = relative-position bucket (−2,−1,+1,+2, already substrate-native); role-profile cosine for king/queen vs +0.021/+0.109, cooc≈0. An afternoon on existing FHRR ops.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Distinct from 124 (node geometry) — structure on bound edges read algebraically. *Crux to grill: "where do roles come from without supervision."*

### 12. **Structured-connectivity (latent) Hopfield** — relations in the *weights between* attractors (Route II/III)
- **Bio + primary:** Latent Structured Hopfield Network (arXiv 2506.01303); correlated-pattern Hopfield theory; CA3 recurrent collaterals encode relations *between* memories (path-equivalence, PMC2966971).
- **Mechanism:** keep the MHN; add a relational coupling `E(x)=−β·lse(Ξx)−x·(W_rel·x)` where W_rel is grown by sleep-phase **second-order role/context** co-activation, with the common-mode as a separable rank-1 term.
- **Different-from-flat:** structure in the *coupling between attractors* (a relational object/manifold), not in independent vector positions.
- **Bound-escape:** modeling the common collocational mode as a *separable shared term* leaves the subdominant relational structure in the *residual* coupling — which is what W_rel learns. Explicit common-mode separation is the missing operation.
- **Project-fit:** anti-homunculus ✅ (one energy function, additive term, no if-then); batch-offline ✅; **FHRR+MHN extension — smallest reach; defaultable to 0 → byte-identical** (matches the project's knob discipline). Reuses `torch_hopfield.py`.
- **Cheap test:** add W_rel (grown from role co-activations on a slice) to the existing MHN energy; probe basin-adjacency for king/queen vs collocational, vs shuffled-W_rel control, with corr<0.15. **Cheapest to *wire* of all (reuses existing MHN).**
- **REINVENTS-KILLED:** **SHARES-A-FAILURE-MODE (flagged).** If W_rel is grown by *first-order* co-occurrence it collapses to 121. The escape *requires* second-order role/context co-activation + separable common-mode — pre-commit to that.

---

## TIER 4 — Beautiful but speculative / largest substrate change (fallback bets)

### 13. **Polychronous-group substrate (Izhikevich)** — structure as reproducible spike-timing groups in a delay-line net (Route IV)
- **Bio + primary:** Izhikevich 2006 ("Polychronization"); Abeles synfire chains. PNG capacity ≫ neuron count (delay-line nets are formally infinite-dimensional).
- **Mechanism:** tokens drive a recurrent spiking net with heterogeneous delays + STDP; relations = delay-defined polychronous groups; paradigmatic words slot into the same PNGs (same delay role).
- **Different-from-flat:** representation is *not a vector* — a set of active delay-defined groups; binding is by delay-coincidence, not circular convolution.
- **Bound-escape:** the bound is dimensional over a D×D rate operator; PNG capacity lives in the **delay dimension** (no axis in the co-occurrence operator). King/queen = structurally-equivalent nodes in the same timing-motif (graph-role equivalence = paradigmatic substitutability) without sharing co-occurrence mass.
- **Project-fit:** anti-homunculus ✅; batch-offline ✅ (PNGs crystallize in sleep-like oscillation); **FHRR ✗ — substrate change (spiking delay-net adjunct).** Highest novelty, lowest fit.
- **Cheap test (substrate-free):** compute **structural-role embeddings** (RolX / regular-equivalence, NOT node2vec) on the transition graph; if structural role-equivalence doesn't surface king/queen specificity, the spiking net won't either. ~50 lines.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Risk = capacity/credit-assignment opacity + FHRR-readout bridge; the cheap graph-role probe greenlights before paying.

### 14. **Cerebellar expansion-recoding sparse code** — random high-D lift + k-WTA (Route III)
- **Bio + primary:** Marr 1969 / Albus 1971; **Litwin-Kumar, Harris, Axel, Sompolinsky & Abbott 2017** (optimal sparse fan-in for pattern separation); cerebellum-as-kernel-machine.
- **Mechanism:** project each token's context vector through a fixed random sparse expansion (D→kD) + k-WTA; codebook = high-D sparse code; Hebbian accumulation in the *expanded* space.
- **Different-from-flat:** overcomplete sparse SDR; similarity = overlap of active sets, not dense cosine. Collocational and paradigmatic structure land in *different feature units*.
- **Bound-escape:** random expansion + k-WTA is a nonlinear lift — subdominant linear modes become first-order *readable overlaps* (the kernel-machine/pattern-separation argument); no single granule unit is dominated by the global frequency axis.
- **Project-fit:** anti-homunculus ✅ (random projection fixed/developmental; k-WTA local); batch-offline ✅; **FHRR tension (largest substrate-change in its family — SDRs aren't phasors).**
- **Cheap test:** apply fixed sparse R + k-WTA to WikiText context vectors; measure paradigmatic specificity in expansion-overlap space vs dense baseline, sweeping sparsity (Litwin-Kumar predicts an optimum), with corr<0.15.
- **REINVENTS-KILLED:** **GENUINELY-NEW.** Caveat: **k-WTA is load-bearing** — Hebbian-grow linearly *without* it re-enters the bound.

---

## Families that collapse back toward co-occurrence (honest flags)

- **Theta-compressed STDP-successor (T2)** and **directional/successor-operator-as-growth-rule** → these are **Report 124 territory** (NULL, +0.006). Any candidate that diagonalizes/iterates the transition operator *as a flat-code growth rule* (rather than reading its **eigenbasis**, #1) is reinventing 124. The eigenbasis read is the *only* thing that escapes — guard it explicitly.
- **CCGP/parallelism-growth (C3) and map-stitching (A4)** are strong *diagnostic oracles* but as *growth rules* risk re-walking 119 (CCGP-as-loss) or degenerating to the flat code if charts are pooled before growing — keep them as probes, not builds.
- **Self-organized-criticality codebook (D3)** — most at risk of being "theoretically beautiful but a co-occurrence cascade in disguise." High-upside/high-uncertainty; only its cheap branching-process probe is worth running.
- **Active-inference / hierarchical-PC generative substrate (G4)** — overlaps the explicitly out-of-scope "latent layer on top" effort and brushes the online-error-write ban; only legal scoped as the *substrate-level* generative model replacing the flat code, sleep-phase writes only.

---

## TOP 3–4 HIGHEST-UPSIDE GENUINELY-DIFFERENT BETS

1. **Hebbian-TEM (#2) — the highest-ceiling structurally-correct bet.** It is the canonical neuroscience answer to "relational structure that generalizes," it is **FHRR-native (`g⊛x` is the existing bind)**, and its bound-escape is the cleanest: it re-bases the *subdominant* substitutability axis into the *dominant* axis of an explicit structural factor — the local rule no longer has to claw a weak mode out of one entangled operator. It also *natively delivers analogy* (king:queen::man:woman as a g-space displacement), which is the Phase-5 deliverable. Highest upside; the only real risk is whether Hebbian path-integration factorizes *without* backprop — which the cheap test directly probes.

2. **SR-Eigenbasis Map (#1) — the most literature-triangulated direct hit on the bound.** Three independent literatures (Stachenfeld grid=SR-eigenvectors; SFA≡SR; the 2026 "word-classes emerge from SR-of-language" paper that *already passes the paradigmatic test on non-co-occurring substitutable words*) converge that the SR's **subdominant eigenmodes carry exactly the king/queen structure**. It is the natural *foundation* #2 and #10 build on (both need its coordinates). Strongest only if the cheap test uses the **eigenbasis** (not the operator-as-growth-rule, which is 124).

3. **Slow-Feature / Bio-SFA (#3) — the most surgical fit to the literal shape of the bound.** The bound says "local iteration reaches dominant, not subdominant, modes"; SFA is **spectrally inverted by construction** and reaches the bottom of the spectrum via a *proven local Hebbian/anti-Hebbian rule*. Lowest substrate change (a new growth rule, not a new representation), clean falsifiable hypothesis ("paradigmatic = slow-context-invariant").

4. **Benna-Fusi cascade edges (#8) — the cheapest high-information bet.** Not the highest ceiling, but it tests a sharp, *untouched* mechanism (timescale-separation) in the *exact* 121/123 harness for ~half a day, with byte-identical fallback. It is near-free insurance and would bank a clean finding either way. *(Shares-a-failure-mode if paradigmatic isn't timescale-separable — which is precisely what the test settles.)*

---

## THE SINGLE CHEAPEST DIVERGENT PROBE TO RUN FIRST

**Run the TCM-FHRR order-vs-context channel split (#7) — ~30 lines of numpy, no GPU.** Build two vectors per token from a single corpus pass — a pure *context* (window co-occurrence) vector and a pure *order* (`Π^k` position-bound) vector — and compare king/queen paradigmatic specificity per channel against the +0.021/+0.109 anchors.

Why this one first, over the equally-cheap #8: it is the **most decisive per dollar** because it tests the single load-bearing premise shared by the *entire* "add an orthogonal axis" route — *does paradigmatic structure live in the order/position channel that co-occurrence structurally misses?* — and BEAGLE/Howard-Kahana have **already shown empirically** that order yields paradigmatic where context yields syntagmatic, so a positive result is high-prior and a negative result kills an entire family for 30 lines. It is fully FHRR-native (zero substrate change), and a green result immediately greenlights #7 and de-risks the order/phase channel that #10 and #13 also lean on.

**Recommended cheap-test batch (all substrate-free, all reuse the Report-123 gauge-free gate + corr<0.15, ~1–2 days total, directly comparable to the +0.021→+0.109 anchors):** #7 (order-vs-context, ~30 lines) → #8 (Benna-Fusi slow-variable in the 121/123 harness, ~half day) → #1 (SR-eigenbasis, hours) → #2 (transition-slot factorization, an afternoon, reuses #1's basis). These four oracles triage the four distinct escape-routes (orthogonal-axis / timescale / spectral / factorization) before any substrate is built — exactly the project's oracle discipline. Honest caveat for the convergence pass: #8 and #12 share-a-failure-mode (still ingest co-occurrence) and #1 must use the eigenbasis or it relapses to the 124 NULL — these are the three guard-rails to hold the line on.