# Null-audit under the coupled capability-DAG framework

*Created 2026-06-01 · branch `audit/coupled-null-reeval` · READ-ONLY re-classification (nothing
re-run). Method: a 9-agent fan-out read every verdict-bearing null in `reports/` (plus the
`notes/status-log/` orphans), an adjudicator applied one consistent bar + deduped, and a completeness
critic adversarially re-scanned for missed nulls, over-generous MAYBEs, and homunculus-smuggling. Full
machine output archived in the session task log.*

## Why this audit exists

The §4 reframe (CONTEXT.md, 2026-06-01) made capabilities a **coupled DAG**, not a linear sequence.
The worry it operationalizes: the **121–127 arc tested Codebook-growth in ISOLATION** (iterate a
*static* operator over a flat code) and may have banked **false negatives** for mechanisms whose
faithful form is a **coupled / two-timescale / replay-interleaved** dynamic. This audit re-evaluates
every past null against that worry, **safely** — read-only classification, no re-runs.

**Classification rubric**

- **NO** — tests a *fixed operator iterated locally*; **operator-reweighting-invariant**; it **is** the
  bound. Replay only reweights *which windows* feed the same fixed operator → cannot move structure out
  of the dominant subspace. *(target ~80%.)*
- **MAYBE** — its **faithful** form is a consolidation/replay/two-timescale dynamic, tested only as a
  **static / single-timescale / isolation surrogate**. A valid MAYBE names a **specific LOCAL coupling**
  (a local priority scalar: surprise / tension / settling-residual). If the only flip-path is a hand-set
  curriculum or a metric-triggered supervisor, it is the banned homunculus → downgrade to NO.
- **UNTESTED-COUPLING** — never run under *any* coupling; faithful form inherently requires it.

**The key discriminator (applied ruthlessly):** does replay-interleaving change the **EFFECTIVE
OPERATOR** (the co-occurrence statistics the dynamic integrates), or merely the **ORDER** the same
operator is iterated? Effective-operator change → genuine MAYBE. Reachable-subspace fixed regardless of
order → NO.

---

## Headline

**64 distinct nulls triaged → 47 NO (73 %) · 12 MAYBE · 5 UNTESTED-COUPLING.** The 73 % NO sits on one
of **three independently order-invariant bound families** (below); coupling cannot flip any of them. The
genuinely live set is **small and convergent**: essentially **one channel — emergent replay-interleaving
feeding a NONLINEAR-PARTITION writer (CE-1 ⊗ 127, merged)** — plus a handful of distinct probes
(Oracle-E TEM local writer; 052 Pair-#2 drift-pressure; Oracle-C eligibility×surprise; 013
tag_count→u_k). **Most of what the audit feared (the Phase-4/5 replay-mechanism nulls) is NO because
those mechanisms were recall STABILIZERS, not structure GENERATORS** — coupling a stabilizer to growth
cannot make it generate subdominant-mode structure.

### "NO" is not monolithic — the three bound families

| # | Bound family | Governs | Why order-invariant |
|---|---|---|---|
| **1** | **Paradigmatic subdominant-mode bound** | 121–127 flat-code growth; 017/018/020-021; 119 | A fixed linear operator iterated locally amplifies its **dominant** (collocational) mode; reweighting which/whose windows feed it cannot **reorder eigenvectors**. Paradigmatic structure lives in **subdominant** modes reachable only by a **global** computation or a **nonlinear partition**. |
| **2** | **1/√D FHRR crosstalk noise floor** | Phase-5 ΔE / per-atom fidelity / metastability (041–058) | At D=4096 the unbind-crosstalk floor (≈0.016) drowns per-atom signal and pins the role-vs-content ΔE below a **dimensional** noise scale. Replay cannot change 1/√D. |
| **3** | **bind(k,v) substrate-algebra geometry** | MQAR key-only / GHRR / retrieval-side spikes (062–066) | A single factor of an elementwise FHRR/GHRR bind has **no addressable basin** (≈orthogonal to k), route-invariant across binding algebras. The demonstrated escape is the **bundle-first ARCHITECTURE**, not a replay coupling. |

This is a **framing correction the audit forces**: the rubric told classifiers to reason against bound
#1, but batches D/E/G are governed by #2/#3. The NO verdicts stand either way (all order-invariant), but
when this doc says "it is the bound," check **which** bound — for 041–066 it is the substrate/encoding
limit, *not* the paradigmatic-mode limit, and the right escape (if any) is a **lower-D run** or an
**architecture change**, not a replay reorder.

---

## The short list (actionable) — critic-incorporated ranking

*The raw fan-out produced 19 MAYBE/UNTESTED rows. The completeness critic's two corrections —
(a) merge CE-1/127 into one channel, (b) collapse the range-shaped lane to one representative gated on a
nonlinear-partition writer — tighten it to the ranking below. Plausibility = P(coupling flips the
verdict-bearing null), audit-subjective.*

### Tier 1 — the one genuinely live channel (decisive, anti-homunculus clean)

**1. CE-1 ⊗ 127 (MERGED) — emergent Replay-interleaving × a NONLINEAR-PARTITION (online streaming
k-WTA) writer.** `UNTESTED-COUPLING` · **plausibility 0.45.**
- **Coupling:** a *local* settling-residual / surprise / co-activation-tension priority re-orders WHICH
  windows the growth dynamic consolidates (pattern-separating dissimilar contexts), feeding an
  **online streaming k-WTA on `build_S`** — so interleaving amplifies a paradigmatic pair's
  shared-context signal relative to its collocational neighbors. Arm A = as-is stream + same writer
  (the 121-127 null reproduced); gauge = **random-reorder** + same writer (must NOT reproduce);
  headline = within-set **label-shuffle B-KILL**.
- **Why it's the lead:** it is the **only** experiment that asks the audit's key-discriminator question
  directly, lives on the **nonlinear-partition side where paradigmatic structure IS reachable** (123's
  linear channel is closed but k-WTA/k-means reach it — 127), is **anti-homunculus clean by
  construction** (combination-experiments.md pre-registers the local-priority requirement + random-reorder
  gauge + kill criterion), and is the **cheapest decisive test** (reuses `experiments/61/65/68`).
- **THE LOAD-BEARING DESIGN CORRECTION:** a **linear-writer CE-1 is predicted to NULL** (it collapses
  into 123). CE-1 must be wired to the **127 online-streaming k-WTA writer**, not the `grow_G`
  linear projection. CE-1 and 127 are the *same* live channel; this merge is the audit's single most
  important output.
- **Kill criterion (decisive either way):** if emergent-interleaved replay nulls (B-KILL ≈ Arm A), the
  **flat-code-from-small-text program is CLOSED** — the bound is banked as capability-level **AND
  replay-invariant**, and the next move is an **Abstraction-node build-gate decision** (substrate/data
  change — the user's call), not another oracle.

### Tier 2 — distinct live couplings, each worth a controlled one-at-a-time test

**2. 125 Oracle E — TEM slot-factorization LOCAL writer** (online Hebbian path-integration over
fixed-random structural slots). `UNTESTED-COUPLING` · **0.4.** The one local-writer route **not reducible
to k-means** — its *dynamic* differs (not a single projection, so route-invariance does not bind it).
Oracle E is a confirmed **global-flashlight POSITIVE** (recovers ~all paradigmatic spec in 8–32 nonneg
dims); only the **local writer** is untested (= the SVD's status for the flat code). *Anti-homunculus:*
slots fixed-random, writer Hebbian-**not-backprop** (backprop-trained slots = the banned global step);
NULL if matching E needs global optimization for slot assignment.

**3. 052 Pair-#2 — drift / replay-pressure** (per-atom cumulative drift δ_i → replay priority; MIR /
Aljundi). `UNTESTED-COUPLING` · **0.25.** The **only** Phase-5 replay-priority pair structurally
**immune to the 1/√D foreclosure** that killed Pair-#4 / 050 / 051, because its signal is a
consolidation-step **displacement**, not a retrieval-pulse softmax quantity. *Caveat:* the same D=4096
substrate may still cap a role-vs-content ΔE headline — **run at lower D** or headline a
**drift-consolidation metric**, not ΔE.

**4. 013 tag_count → u_k chain reroute** (route consolidation reinforcement through **replayed traces**,
not raw cue frequency). `MAYBE` · **0.25.** The replay→consolidation coupling was **severed by
construction** (the Benna-Fusi u_k chain read cue frequency, not replay survival); the 013 addFq result
(77 %→67 %) is direct evidence the coupling **moves the consolidation distribution**. *Anti-homunculus:*
a local product of existing per-trace scalars (gate × tag_count × suppression) driving the existing
`reinforce()`; clean unless `tag_overlap_threshold` is hand-set per-pair.

**5. 125 Oracle C — eligibility × surprise two-timescale INTERACTION** (fast eligibility trace gated by a
slow **MHN settling-residual** surprise). `MAYBE` · **0.25, borderline.** Tested only as a static
frequency-novelty reweighting; the two-timescale interaction is genuinely untested. *Risk:* the
`whats_missing` caveat — surprise ≈ PMI, already inside SPPMI — means the discriminator must be the
two-timescale **interaction beating plain SPPMI**, or it collapses straight back to 123. Surprise MUST be
the measured settling-residual, never an if-rare-then-write rule.

### Tier 3 — coupling-shaped but downstream-capped / lower confidence

**6. 062/064 M2 training-time path** (EqProp + **uniform-random** role-shuffled negatives + DSM over the
replay buffer). `UNTESTED-COUPLING` · **0.25.** A genuine growth⇄replay coupling that changes the
*effective operator* (carves basins during consolidation, not at retrieval). **But** 065/066 give
independent **geometric** evidence against it (bind(k,v) key-only is geometrically marginal; the
demonstrated escape is **bundle-first architecture, not M2**). *Anti-homunculus:* hard-negative **mining**
= the banned homunculus; must stay uniform-random.

**7. 115-118 Frame-B — CLS pattern-separating replay buffer** to suppress consolidation-injected
variance. `MAYBE` · **0.2.** A **variance-reduction bet** (make the real +0.05 lift detectable, σ 2.3×→6×
is the wall), categorically **not** a move-structure-into-a-new-subspace claim. The global-stream-shuffle
arm already ruled out cheap reordering, so the residual rests on a genuine **effective-input-distribution**
change. *Danger phrase:* "control the landscape draw" (118) must NOT become a designed schedule.

**8. 068 range-shaped replay sampler (REPRESENTATIVE of the whole range-shaped lane).**
`UNTESTED-COUPLING` · **0.2.** Genuinely **manufactures off-support (role,atom) pairs** (effective-operator
change), never coupled to growth. **Critic caveat (load-bearing):** the consolidation *write* it would
feed is the 055-058 **linear** Hebbian projection — rectangular input support does **not** give a linear
writer a partition operator, so it likely manufactures rectangularized **collocational** support and stays
**inside the bound** unless wired to a **nonlinear-partition** writer (→ Tier 1). The 070/071-072/073/074/
110-111 entries **collapse into this one** and are gated on that writer. *Also:* the Dorrell–Whittington
modularization theorem 068 invokes assumes a **nonneg energy-efficient autoencoder** trainer — **not** the
project's local Hebbian writer; the guarantee does not transfer (same mis-shapen-oracle trap PM-6 caught).

**9. 047/049 discovery-channel de-duplication** (pattern-separating replay diversifies which atoms the
discovery channel writes). `MAYBE` · **0.2, borderline-homunculus.** The natural fix ("skip add if
re-settle lands in an existing basin") is an if-X-then-Y rule (049 flags `resolve_threshold` as already
this shape) — valid only if de-duplication falls out of a **local self-throttling r_ema**. Even if it
restores diversity, downstream ΔE still hits the 1/√D floor → flips an **intermediate** criterion, not
graduation.

**10. 084/090 context-source provenance.** `MAYBE` → **near-NO** · **0.15.** A **source-quality** coupling
on an **already-passing** retrieval path (top1 0.95), downstream-capped by the 109 readout-saturation wall
→ flipping the source does not flip any banked null. *(Critic: closer to NO than to the live MAYBEs.)*

**11. 012/013 IoR keep-and-sweep.** `MAYBE` · **0.12-0.15, weakest survivors.** Two independent coupling
configs already nulled it (012 measurement-only, 013 candidate-addition); the only untested variant
(persist unresolved traces, re-sample over many cycles) is **one step from a hand-tuned curriculum** —
PASS only if persistence is an intrinsic per-trace suppression-decay EMA.

---

## Full triage table

*Grouped by cluster. NO rows are compressed (reason only); MAYBE/UNTESTED rows carry the specific
coupling. `→bound#` tags which of the three bound families a NO sits on.*

### Cluster A — flat-code growth arc (121–127, the core worry)

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 121 1st-order Hebbian centroid growth | token vec → centroid of co-occurring context | **NO** →1 | Fixed centroid operator iterated locally; the canonical dominant-collocational-mode family. |
| 122 2nd-order SPPMI graduation-arm null | row-centered SPPMI (B′) S′@G | **NO** →1 | One step of power iteration toward its dominant mode; reordering reweights inputs, not the reachable subspace. (SVD/FHRR oracles were POSITIVE flashlights, not nulls.) |
| 123 B′ + H_anti D=4096 sweep | SPPMI pull + energy-native anti-collapse | **NO** →1 | Canonical statement of the bound: reweighting cannot reorder eigenvectors. The replay channel lives on the **output-nonlinearity** side (127), not here. |
| 124 R3 directional/successor operator | γ-discounted fwd/bwd directional-SPPMI | **NO** →1 | Another fixed operator iterated locally; forward-only is WEAKEST → channel closed for linear reads regardless of stream manipulation. |
| 125 Oracle B (SFA/SR) | slowness / successor-representation via grow_G | **NO** →1 | Single-projection power iteration converges to the dominant mode; subdominant slow modes unreachable by local iteration. |
| **125 Oracle C (eligibility × surprise)** | **faithful = fast eligibility × slow settling-residual surprise; tested as static freq-novelty gate** | **MAYBE** 0.25 | **Two-timescale interaction (settling-residual surprise) genuinely untested; surprise gates WHICH co-activations consolidate (cooc→cooc×surprise) = effective-operator change. Risk: surprise≈PMI → must beat plain SPPMI or collapses to 123.** |
| 125 Oracle D (order channel) | position-spectrum operator via grow_G | **NO** →1 | Syntagmatic by construction; route-invariant, cannot reach paradigmatic subdominant modes. |
| **125 Oracle E (TEM slot factorization)** | **faithful = online Hebbian path-integration LOCAL writer over fixed-random slots; tested as GLOBAL NMF** | **UNTESTED-COUPLING** 0.4 | **Global flashlight POSITIVE; local writer never run. Dynamic ≠ single projection → route-invariance does not bind. Hebbian-not-backprop; slots fixed-random.** |
| 126 behavioral substitutability | does graduated 055-058 memory admit queen as king-completion | **NO** →1 | Read-only probe of a FIXED memory; confirms the bound is capability-level. Replay can't rewrite a read-only probe. |
| **127 build_S k-WTA / online next-move** | **nonlinear hard competition on build_S, tested OFFLINE/GLOBAL (k-means reproduces it)** | **MAYBE** 0.4 | **The LIVE nonlinear-partition channel; 127 itself prescribes an ONLINE/STREAMING local k-WTA fed a replay-reordered stream, headlined on the B-KILL. = Tier-1 merged with CE-1.** |
| **CE-1 (registry)** | **Codebook-growth × emergent Replay-interleaving (SPEC, un-built)** | **UNTESTED-COUPLING** 0.45 | **The user's lead; asks the discriminator directly; anti-homunculus clean by construction. ONLY escapes the bound in its NONLINEAR-PARTITION form (linear-writer CE-1 = the 123 null). → Tier-1 #1.** |

### Cluster B — Phase-4 replay/consolidation STABILIZERS (high-relevance-but-NO)

*Honest framing: these are not "the paradigmatic bound" — they are **orthogonal to it**: recall
stabilizers/pruners that never touch structure-generation, so coupling to growth cannot make a stabilizer
generate subdominant-mode structure.*

| Report | Mechanism | Class | One-line reason |
|---|---|---|---|
| 011 GIB synergy estimator | re-settle recovers structure under fixed Hopfield | **NO** | Pure measurement; fixed-operator basin geometry, reweighting can't change it. |
| 019 three growth objectives (atom-collapse) | Hebbian/reconstruction/error-driven | **NO** (was MAYBE) | Collapse-prevention already exercised (H_anti, 122-123) without flipping the bound. |
| 022 death/reencode + HAM aggregation | stabilizer knobs on frozen codebook | **NO** | Prunes/reweights which patterns feed the fixed retrieval operator; generates no structure. |
| 023 drift-magnitude sweep | replay-discovery as recall-recovery | **NO** | Recovery stabilizer compensating for drift, not a generator. |
| 026 design-spec verification | ΔR@K + cap-coverage under drift | **NO** (was MAYBE) | The cited flip touches only the SECONDARY headline (cap-coverage variance, see 037), not the bound. |
| 030 reencode_discovered_patterns | re-settle stale patterns | **NO** | Re-settling pulls toward existing basins; cannot relocate structure into new modes. |
| 032 Phase-3+4 integration n=10 | online Hebbian growth COUPLED to replay-discovery | **NO** (was MAYBE) | Closest existing growth×replay test, but PASSIVE corpus-shuffle over a fixed dominant-mode operator. Faithful emergent-priority version = **CE-1**. |
| 033 death-mechanism diagnostic | Benna-Fusi decay + threshold death | **NO** | A pruning/forgetting stabilizer; removes support, generates no structure. |
| 035 Saighi A_k n=10 falsification | per-attractor self-inhibition | **NO** (was MAYBE) | A_k ran on the RETRIEVAL operator; competition-over-the-growth-operator is subsumed by 127/CE-1. |
| 036 A_k decay sweep | inhibition-decay lever | **NO** | Tunes inhibition magnitude on the same fixed retrieval operator's dominant attractors. |
| 037 seed-3 cap collapse / binary-vs-graded death | mass death reshapes substrate | **NO** (was MAYBE) | Graded soft-death tightens cap-coverage variance only (secondary headline); report says it does NOT reach paradigmatic structure. |
| 040 freq-weighted Benna-Fusi α | graded two-timescale cascade-rate filter | **NO** (was MAYBE) | A fixed RATE-reweighting of the same operator; functional only in the death-free transient, nulled because mass-death absorbed it. Operator-reweighting-invariant. |

### Cluster C — Phase-3 codebook-comparison + Path C/γ closures

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 017 three fixed update rules (Recall@1) | Hebbian/reconstruction/error-driven static codebooks | **NO** →1 | Fixed first-order co-occurrence updates over a flat code; the dominant-mode family. |
| 018 same rules under settled-synergy | richer metric on same rules | **NO** →1 | Structure is first-order neighborhood-merging (collocational); metric exposes more of the SAME operator's output. |
| 020-021 per-scale reconstruction codebooks | orthogonality-as-target | **NO** →1 | Orthogonal only because UNDER-TRAINED; a dosage artifact of one first-order objective. |
| 039 data-integrity forensic | artifact-reuse labeling bug | **NO** | Not a mechanism null; confirms 017's nulls are genuine. |
| **112 Path-C C.2 two-timescale consolidation** | **metastability/drift-tension replay-priority (C.2.4/C.2.5) + α_anti, on a Hebbian centroid-pull** | **MAYBE** 0.2 | **The one batch-C mechanism whose faithful form IS a two-timescale replay loop; but ran on a single timescale over a dominant-mode operator, and the headline was gauge-vacuous (voided 2026-05-28). Critic: lean NO unless re-specified with a nonlinear-partition writer.** |
| 113-114 Γ1 context-residual consolidation | fixed per-event residual update | **NO** →1 | Another single-projection operator; ran INSIDE the same C.2 replay pipeline (coupling already present) and did not flip; deltas gauge-vacuous. |

### Cluster D — Phase-5 ΔE / death-dynamic substrate (governed by bound #2, the 1/√D floor)

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 041 K=4 schema-prior branching ΔE n=5 | bundle + re-settle on post-death substrate | **NO** →2 | Settles a fixed substrate; K-branch averaging buys nothing on a set landscape. |
| 042 branch-collapse diagnostic | K=4 branches collapse to one attractor | **NO** →2 | Wide Hopfield basins at fixed atom count; reweighting can't narrow them. |
| 043 substrate-scale discrimination | pre- vs post-death divergence | **NO** →2 | Residual bimodal ΔE on a dense substrate is a substrate-geometry fact, not an operator replay reshapes. |
| 044 consolidation-geometry diagnostic | death collapses d_eff ~10× | **NO** →2 | Measurement of a fixed substrate's spanned subspace; the diagnostic reads, never acts. |
| 045-046 A+B death-as-continuous-decay | redundancy-EMA + repulsion + step-3 | **NO** 0 (was MAYBE) | A faithful two-timescale dynamic, but even fixing the construction degeneracy hits the order-invariant 1/√D ΔE floor; flips an intermediate metric (d_eff) only. |
| **047 K-branch state-divergence FAIL** | **discovery-channel near-duplicates dominate schema-store** | **MAYBE** 0.2 | **Genuinely a function of HOW the replay/discovery loop adds atoms (effective-codebook change). Borderline homunculus (skip-add rule); downstream still 1/√D-capped → flips intermediate criterion, not graduation.** |
| 048 A1 r_ema init | mean Gram-row RMS proxy | **NO** | A measurement-operationalization bug (mean vs max), fixed in 049. |
| **049 A1′ tied-duplicate residual** | **max-over-others r_inst self-throttle** | **MAYBE** 0.2 | **Substrate throttle works (99.8 %); residual is the replay loop adding tied near-duplicates faster than they decay. Same coupling family as 047; same caveats.** |
| 050 β continuous role-fidelity prior | mean pairwise FHRR unbind distance | **NO** →2 | Hard encoding-dimension fact: f_i ≈ 0.984 zero-variance at D=4096 regardless of content. |
| 051 Tier-1 fidelity disambiguation | 3 alternative fidelity metrics | **NO** →2 | All three hit the 1/√D floor — route-invariant per-atom-variance bound. |
| 052 Pair-#4 metastability→replay-priority | softmax-derived per-atom variance | **NO** →2 | This IS the worried-about coupling and it WAS run; nulls because its signal source is the same D=4096 noise-floor-zero quantity. |
| **052 Pair-#2 drift/replay-pressure** | **per-atom cumulative drift δ_i → replay priority (proposed, NEVER RUN)** | **UNTESTED-COUPLING** 0.25 | **The only pair structurally IMMUNE to the 1/√D foreclosure (signal = consolidation-step displacement, not softmax). Run at lower D / headline a drift metric, not ΔE. → Tier-2 #3.** |
| 053 headline ΔE n=10 | E_content − E_role, gate ≥5.5e-3 & CI>0 | **NO** →2 | Real but sub-noise; the magnitude floor derives from 1/√D crosstalk; replay reshapes atoms, not the dimensional floor. |
| 057 cross-seed β-sweep ΔE | β∈{3,5,10} | **NO** →2 | Settling-hyperparameter sweep confirming sub-floor robust to β. |
| 058 cross-seed cue-regime sweep ΔE | 24 cells, ΔE + basin-membership | **NO** →2 | Exhaustive measurement surface; route-invariant failure across 24 cells × 3 metrics = D=4096 saturation. |

### Cluster E — Phase-5 retrieval-intervention + MQAR (governed by bound #3, substrate-algebra)

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 059 (+060/061) additive log-prior spike | logit spike on selected schema atom | **NO** →3 | Fixed additive logit field on a frozen substrate; the only large effect (no-schema-store) is an arbitration-shape positive control. |
| 062 D1 pseudo-inverse storage | storage-rule swap at retrieval | **NO** →3 | Re-expresses existing basins; cannot create role basins. |
| 062 D3 cross-K branch coupling | additive softmax branch coupling | **NO** →3 | Amplifies asymmetry only if it exists; replay can't inject missing initial-state asymmetry. |
| 063 E1 centered log-prior field | zero-mean per-atom log-prior | **NO** →3 | Role channel anti-correlated with role-target identity (substrate noise, not basin structure). |
| 064 M1 retrieval-side | role-weighted MHN stack | **NO** →3 | Retrieval-side surrogate over a frozen substrate; faithful coupled form is the M2 training-time path (below). |
| **062/064 M2 training-time path** | **EqProp + uniform-random role-shuffled negatives + DSM over the replay buffer (NEVER RUN)** | **UNTESTED-COUPLING** 0.25 | **Changes the consolidation dynamic (carves basins), not the read = the growth⇄replay coupling the reframe asks about. BUT 065/066 give geometric evidence against it (bundle-first, not M2, is the demonstrated escape). Hard-negative mining = homunculus. → Tier-3 #6.** |
| 065/066 MQAR key-only / GHRR / bundle-first | key-only Hopfield basin over bind(k,v) | **NO** →3 | Substrate-algebra bound (single bind factor ≈ orthogonal to k), route-invariant across algebras. Escape = bundle-first ARCHITECTURE, not a replay coupling. |

### Cluster F — range-shaped replay lane (collapsed; gated on a nonlinear-partition writer)

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| **068 RangeShapedReplaySampler** | **rebind-on-the-fly MANUFACTURES off-support (role,atom) pairs; never fed any growth dynamic** | **UNTESTED-COUPLING** 0.2 | **Genuine effective-operator change (manufactures pairs absent from the corpus). REPRESENTATIVE of the lane. → Tier-3 #8. Caveat: linear write probably stays inside the bound; gate on a nonlinear-partition writer.** |
| 070 range_shaped → UnifiedReplayMemory | sampler → rebind → settle → insert (frozen codebook) | **MAYBE** 0.2 → collapse into 068 | Downstream consolidation ran but codebook FROZEN; faithful form grows the codebook. Same banked-bound tension. |
| 071-072 real-corpus end-to-end smokes | static range_shaped through exp/19 | **MAYBE** 0.2 → collapse into 068 | Candidate inflation without retrieval movement; codebook never grew from replayed pairs. |
| 073 candidate-quality diagnostic | fixed Hopfield snaps candidates back (near-dup 1.0) | **MAYBE** 0.2 → collapse into 068 | Isolates the erasure mechanism; a co-evolving growth/consolidation coupling would change the operator the candidate settles against. |
| 074 pre-settle novelty diagnostic | store pre-settle query (novelty preserved, no utility) | **MAYBE** 0.2 → collapse into 068 | Missing piece is turning captured novelty into USEFUL structure over many slow cycles. Store-before-cleanup is homunculus-prone. |
| 110-111 range_postsettle bounded viability | production lane, precommitted stop criteria | **MAYBE** 0.2 → collapse into 068 | "not_viable" scoped to the FROZEN-codebook settle-then-insert lane; explicitly does NOT falsify the training-time/growth-coupled form; says a NEW mechanism is needed. |

### Cluster G — Phase-5′ scene/context-source + bridge lane (075–109)

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 075/080/097 scene-anchor + observed-prefix | candidate retrieval path (+ embedded global-anchor null) | **NO** | Candidate path SUCCEEDS (top1 0.95+); the global-anchor null is a positive control confirming specificity, not a falsified mechanism. |
| **084/090 learned/replay-derived context-source** | **frozen one-shot snapshot (chance-aligned) / dirty-control static source** | **MAYBE** 0.15 → near-NO | **Source-quality coupling on an already-passing retrieval path, downstream-capped by 109 saturation. 084's own next-work proposes hand-matching role universes = homunculus.** |
| 104/107/109 ΔE readout bridge (terminal) | raw_scene_energy over bundle-first scene-MHN | **NO** | Attractor-saturation limit (stored bundle self-similarity = 1.0 → energy readout saturates); no iterated operator for replay to move. |

### Cluster H — Frame-B level-DiD + WS-InfoNCE generalization

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| **115-118 Frame-B consolidation-injected variance** | **graduated consolidation write as a single-timescale batch pass; CAPACITY-WALL / variance-irreducible** | **MAYBE** 0.2 | **Closure is a VARIANCE/SNR wall (real +0.05 lift swamped, σ 2.3×→6×), NOT the paradigmatic bound. A pattern-separating two-timescale replay buffer attacks the named root cause. Variance-reduction bet, not move-structure. Global-stream-shuffle arm already ruled out cheap reordering. → Tier-3 #7.** |
| 119 WS-InfoNCE value-codebook shaping | static within-scene InfoNCE pass | **NO** →1 | Fixed shaping operator run as a single-pass optimization; real text has no generalizable low-rank role→filler structure (056 concurs at capability level). |

### Cluster I — older / stale / status-log orphans

| Report | Mechanism | Class | One-line reason / coupling |
|---|---|---|---|
| 008 LSR/Epanechnikov kernel sweep | swap softmax for log-sum-ReLU | **NO** | Fixed retrieval kernel, algebraically β-invariant; replay can't make a β-cancelling ratio non-invariant. |
| 010 permutation-indexed temporal slots | (POSITIVE result; lone weak seed) | **NO** | Headline is a positive directional-encoding result; weak seed is FHRR crosstalk, not a banked null. |
| **012 Inhibition-of-Return (measurement-only)** | **IoR decay/recovery, landscape fixed** | **MAYBE** 0.15 | **Faithful form is two-timescale keep-and-sweep; tested as static measurement-only. Weakest survivor (see 013). → Tier-3 #11.** |
| **013 IoR with candidate-addition** | **same IoR, landscape DOES change** | **MAYBE** 0.12 | **Nulled because resolve-and-remove denies IoR a time window; the keep-and-sweep flow was never built. Flip-path flirts with a re-sampling curriculum. → Tier-3 #11.** |
| **013 tag_count → u_k chain** | **does replay diversity reach the consolidation chain? (severed by construction — chain read cue freq)** | **MAYBE** 0.25 | **The replay→consolidation coupling was severed; addFq (77→67 %) shows the coupling moves the distribution. → Tier-2 #4.** |
| 034 Saighi A_k seed-1 | (POSITIVE prototype; falsified at n=10 = 035) | **NO** | A_k is a fixed local inhibition on the retrieval operator; banked null carried in 035. |
| 2026-05-13 orphan: Hebbian online updater | streaming Hebbian codebook reinforcement | **NO** →1 | First-order centroid pull = the dominant-mode dynamic; preservation-only at current settings. |
| 2026-05-13 orphan: Phase-4 replay-and-discover | ΔR@K with active drift, default knobs | **NO** | Headline-inert (0.8 % sparse discovery); named flip-path (death_window/reencode) is the NO stabilizer family. Emergent-priority version = CE-1. |
| 2026-05-13 orphan: error-driven contrastive updater | streaming pull/push codebook update | **NO** | Divergent fixed-operator instability (pull ≫ push toward a position-mask direction); not a structure the operator failed to reach. |

---

## Surfaced qualifications & disagreements (findings, not buried)

1. **CE-1 = 127 (the merge).** The audit's most important output: a **linear-writer CE-1 is predicted to
   NULL** (collapses into 123). The live experiment is **replay-reordered stream → online nonlinear-
   partition writer**. Ranking CE-1 and 127 as two top bets double-counts the one live channel.
2. **Range-shaped lane is probably inside the bound.** The sampler genuinely changes the effective
   operator, but the linear consolidation write it feeds cannot partition. A reasonable **stricter
   reading demotes the whole lane (068/070/071-072/073/074/110-111) to NO** pending a nonlinear-partition
   writer. Kept as one representative MAYBE because the reports themselves bound their nulls *away* from
   the coupled regime — but treat it as **probably-inside-the-bound**.
3. **The Dorrell–Whittington theorem mis-transfer.** Report 068 invokes a modularization guarantee that
   assumes a **nonneg energy-efficient autoencoder** trainer — **not** the project's local Hebbian writer.
   Any range-shaped flip-claim leaning on that theorem is unsound for the actual writer (same mis-shapen-
   oracle trap PM-6 caught with the one-pass TEM/Hebbian probe).
4. **Category mismatch (the three bound families).** Batches D/E/G are governed by bound #2 (1/√D) and #3
   (bind-algebra), not #1 (paradigmatic). NO verdicts stand (all order-invariant), but the *escape* for
   041–066 (if any) is a **lower-D run** or an **architecture change**, not a replay reorder.
5. **Homunculus discipline held.** Every coupling that flirts with a supervisor/curriculum/metric-branch
   is flagged in-table (IoR keep-and-sweep; Frame-B "control the landscape draw"; range-shaped "store-
   before-cleanup"; 047/049 skip-add; M2 hard-negative mining). The critic found **no new unflagged
   homunculus**.
6. **Oracle-C borderline.** Surprise ≈ PMI (already in SPPMI) is a real argument it collapses to 123;
   kept MAYBE because the **two-timescale interaction** (not the static surrogate) is genuinely untested.
   The discriminator must be the interaction beating plain SPPMI.

Completeness critic verdict: *"The table is SOUND and the short list is TRUSTWORTHY"* — coverage complete
across 001–127 + dir-reports; NO classifications rigorously the bound; homunculus discipline excellent;
the two qualifications are about **not over-stating the number of independent live escape routes**, and
overturn no verdict.

---

## Recommended Stage-2 lead (one at a time, frozen precommit + full control battery)

**CE-1 ⊗ 127 (merged): emergent Replay-interleaving × an online streaming k-WTA writer on `build_S`.**

- **Arms:** A = as-is corpus-order stream + the online k-WTA writer (the 121-127 isolation null,
  reproduced); B = the **same** writer fed a stream **re-ordered by a local emergent replay-priority**
  (settling-residual / surprise / co-activation tension) with **pattern separation**.
- **Full control battery (freeze in the precommit):** within-set **label-shuffle B-KILL** (pair-specific
  residual CI-lo > 0 in ≥4/5 seeds — the headline that killed 119/126/127 false positives); a
  **random-reorder gauge** (reordering stripped of emergent priority must NOT reproduce the signal); the
  **competent k-means/offline-partition control** (the 127 lesson — frozen-random is incompetent);
  **multi-seed**; gauge-free para-vs-random specificity; calibration anchor (+0.1092 / kq 0.222).
- **Interaction headline:** Arm B must beat **both** Arm A (isolation) **and** the offline-partition
  control, AND the emergent-priority signal (not a hand-set schedule) must be the operative ingredient.
- **Anti-homunculus check (must pass before building):** replay priority is a *local* scalar per stored
  trace; the interleaving is a *measurement of a local dynamic*, not an arbitration; if the only pass
  needs hand-picking the schedule, the result is **void**.
- **Kill criterion:** B-KILL ≈ Arm A ⇒ the **flat-code-from-small-text program is CLOSED** (bound banked
  as capability-level **and** replay-invariant) → Abstraction-node build-gate decision (user's call). A
  PASS ⇒ replay-as-structure-generator is the real Phase-4 capability — write its grounding + a faithful,
  carded precommit.

*If parallelized: it MUST be a pre-registered, controlled set with family-wise correction + one shared
adjudication — never "iterate permutations and accept whichever shows a signal."*
