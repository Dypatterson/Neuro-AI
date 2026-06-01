# RE-GROUNDING MAP — Neuro-AI

*Drafted 2026-05-31. A single coherent re-anchoring of the project to its
original thesis and biological North Star. It corrects the "memorization"
drift, maps validated assets onto the original six-phase plan, and names the
genuine next step.*

**How to read this document.** It is a map, not a verdict. Every status is
marked plainly — CLEARED / PARTIAL / UNTESTED-vs-original-gate / FAILED /
UNBUILT / PAUSED — and citations are carried verbatim from the dimensional
analyses (`file:line`). Where a claim rests on a paper not opened this session,
it is flagged **second-hand**. Where a capability is designed but not shown, it
is flagged **claimed-not-shown**. Nothing here is invented; PARTIAL/UNTESTED/
PAUSED flags from the inputs are preserved.

The single load-bearing finding: **the Phase-3 floor was written down as the
project's ceiling.** The correction is framing + headline re-anchoring, *not* a
code rewrite. The substrate, the graduated completion write + L2 decorrelator,
the emergent codebook, and the bundle-first scene-memory are validated assets.

> **Citation-correction (2026-05-31, appended).** Citations to
> `experimental-progression.md` in this snapshot use **pre-insertion** line numbers
> (the file gained a ~11-line note that day). Current mapping — prefer the **section
> anchors** (stable): the "phases 3-5" compositional-generalization line `:125` →
> **§"What to test against" (now :136)**; *"Structural retrieval starts working at
> Phase 5"* `:135` → **§"How to know it's actually working" (:146)**; the Phase-5 gate
> `:88-98` → **§"Phase 5 — Binding discovery and atom splitting" (:99-109)**. The live
> charter `CONTEXT.md` already uses section anchors.

---

## 1. The bet and the thesis (North Star)

**The wager.** Current LLMs are amazing but (a) energy-inefficient, (b) cannot
learn continuously — their "understanding" is "locked in frozen weights"
(`notes/briefing.md:36-37`), and (c) need enormous data. The human brain is the
opposite: it starts *not-smart*, learns continuously over time from *little*
data, is *energy-frugal*, and has *no homunculus* controlling it. The bet is
that the data-hungry / backprop / autoregressive path is **wrong**, and
biology/neuroscience gives the answer.

> "The industry's bet is 'build a god that knows everything.' Dylan's bet is
> 'build a companion that knows *me*.'" (`notes/briefing.md:38`) — "The system
> should not have to start smart. It should have the capacity to *become*
> knowledgeable alongside its user, through real shared experience."
> (`notes/briefing.md:40`)

The goal sentence: "Build a local, memory-first AI substrate that can
continuously learn from lived experience, remember by association, reason in
latent energy space, and use an LLM only as an interface rather than as the
cognitive core." (`docs/PROJECT_PLAN.md:3-7`). The hippocampal stability-
plasticity solution was **rederived from first principles, not imported**
(`notes/briefing.md:65`). **Biological fidelity is the organizing principle:**
every mechanism (emergent codebook, replay-as-deep-sleep consolidation, Hopfield
associative memory, FHRR binding, coupled energy terms) was chosen because it is
analogous to the brain.

### Core principles

| Principle | Project's words | Citation |
|---|---|---|
| **No-homunculus** | "Every proposed addition… must either be a local geometric dynamic or be expressible as a measurement of one — never an arbitration over them." | `docs/PROJECT_PLAN.md:16-18`; `notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md:93` |
| **Memory-is-the-self** | "the durable system identity lives in the learned landscape and its trajectories, not in frozen LLM weights." / "The LM is the voice; the memory is the self." | `docs/PROJECT_PLAN.md:19-20`; `notes/briefing.md:85` |
| **Contextual-completion (not sequence-prediction)** | "the native question is 'what does this remind the system of, and what fills this unresolved gap?'" / "'What does this remind me of?' not 'what comes next?'" | `docs/PROJECT_PLAN.md:21-22`; `notes/emergent-codebook/overview.md:23` |
| **Continuous learning** | "live experience writes traces; replay consolidates, abstracts, and reshapes the landscape." (flagged a *principle*, not yet a multi-domain evidence claim) | `docs/PROJECT_PLAN.md:23-26` |
| **Latent reasoning** | "reasoning should happen in vectors/energy states before language is generated." | `docs/PROJECT_PLAN.md:27-28`; `notes/briefing.md:58` |
| **Energy efficiency** | "compact latent operations, sparse active memory, replay… instead of large autoregressive decoding loops." (**aspiration, not benchmarked**) | `docs/PROJECT_PLAN.md:29-30` |
| **Local-first** | "the working research system should run on a MacBook Pro class machine." | `docs/PROJECT_PLAN.md:31-32` |

> **Honesty flag on the North Star.** "Energy-frugal" is a *design commitment*,
> not a shown property — there is no `reports/NNN` energy benchmark in the
> corpus (grep for FLOP/joule/watt/kWh found only an unrelated Dorrell hit). It
> is architecturally real (Hopfield settling, not token-by-token decoding) but
> never measured head-to-head against an autoregressive baseline.

---

## 2. What this system actually IS

Three distinct things get conflated under the word "memory." Pulling them apart
is what dissolves the drift.

1. **Storage / lookup** — a vector DB plus summaries. Retrieve the nearest
   stored item. This is what the project's non-negotiable design rule
   **forbids** collapsing into: *"Do not collapse the memory into a vector
   database plus summaries"* (`docs/PROJECT_PLAN.md:276`).

2. **Contextual-completion** — pattern-complete an unresolved gap by settling
   into an attractor: "what does this remind me of, and what fills this gap?"
   (`docs/PROJECT_PLAN.md:21-22`). Retrieval is *generative*, not lookup: in a
   continuous energy landscape, "retrieving from memory can produce blends that
   were never explicitly stored… Two ideas combining is not an extra operation
   bolted onto memory; it is what memory *does*." (`notes/briefing.md:95`).
   Memory and exploration are "the same event" (`notes/briefing.md:60`).

3. **Abstraction-as-self** — the durable identity is the *learned landscape and
   its trajectories* (`docs/PROJECT_PLAN.md:19-20`), which can resolve
   previously-unresolvable trajectories into genuinely new patterns: "It was
   never experienced. It was *discovered* by the system through its own settling
   dynamics… The system isn't just remembering anymore. It is **learning from
   the act of remembering**." (`notes/notes/2026-05-02-trajectory-trace-and-
   replay.md:56,65`).

**Why "memorization is the target" was a drift.** The Phase-3 consolidation
write graduated as a role-selective associative *memory* doing
contextual-completion (#2) — and it does so at the *information ceiling*,
multi-seed, integrated bit-identically (Reports 055/056/057/058). That is the
correct, validated Phase-3 result. The drift was **relabeling that floor as the
project's identity** ("just a memory," "memorization = the target," "a memory by
design"), which:

- **Collapses #2 into #1.** "It's just a memory that recalls stored episodes" is
  uncomfortably close to the very thing `docs/PROJECT_PLAN.md:276` forbids the
  project from becoming.
- **Mislabels the floor as the ceiling.** The original plan **predicted** Phase
  3 would not generalize compositionally and located that capability as a
  **3→5 gradient maturing in Phase 5**:

  > **Compositional generalization:** SCAN or COGS subsets… **Don't expect to
  > crush these; expect meaningful signal that increases with phases 3-5.**
  > (`experimental-progression.md:125`, verified this session)

  > **Structural retrieval starts working at Phase 5** — retrieval-shaped
  > analogical queries being above-chance is the clearest "the structural part
  > is doing real work" signal. (`experimental-progression.md:135`, verified)

The held-out real-text null at Phase 3 is therefore the **predicted floor
behavior**, not a verdict on the architecture. Contextual-completion vs
sequence-prediction is about the *prediction objective* (recall a stored episode
vs predict the next token); it is **orthogonal** to whether the system should
ever generalize compositionally. The axiom was stretched to license judging the
Phase-3 write in-sample and declaring generalization out-of-scope — a category
error (see §6).

---

## 3. The original six-phase plan and per-phase gates

*Source of truth: `notes/emergent-codebook/experimental-progression.md`
(2026-05-04, headline metrics added 2026-05-09). Architectural commitment
framing every gate (`:17-19`): "contextual-completion architecture… not a
sequence-prediction architecture. 'What does this remind the system of?' is the
load-bearing question."* **Phase 3 is the foundation; Phase 5 is where
"more-than-memory" lives.**

| Phase | Goal | ORIGINAL headline gate (verbatim + `file:line`) | Where compositional-gen / structural-retrieval was located |
|---|---|---|---|
| **1 — Substrate validation** | FHRR bind/unbind/role-filler recovery at 4096-d (`:33-41`) | *Pre-metric, qualitative:* "clean recovery of bound structures, no surprises in the algebra at 4096 dimensions" (`:37`) | Not here (substrate only) |
| **2 — Static codebook baseline (dual objective)** | Fixed random codebook; masked- vs next-token as a **design comparison** (`:43-50`) | "masked-token Recall@1 and next-token Recall@1, **separately**, vs bigram baseline with non-overlapping 95% CIs across the majority of the matrix" (`:58`); ≥1 objective clearly above chance (`:62`) | Not here. Phase 2 only decides which objective Phase 3 adopts (`:52-56`) |
| **3 — Codebook growth (FOUNDATION)** | Two-pathway hybrid update, gated per-experience by retrieval quality (`:66-68`) | **"Recall@K on masked-token contextual completion, stratified by regime…, evaluated against the shuffled-token control. One number + a controlled comparison + a stratification axis"** (`:70`; same at `phase-3-deep-dive.md:215`). Success: codebook stabilizes; improves over Phase 2; "similar tokens have similar hypervectors"; **shuffled-token control fails to produce the same organization** (`:74`, `:134`) | **Tier-3 drill-down only, NOT the gate:** "Don't expect to crush this at Phase 3 — analogical structure usually emerges more strongly at Phase 5" (`phase-3-deep-dive.md:267-269`) |
| **4 — Hierarchical compression** | Second Hopfield layer; frequent L1 bundles → L2 atoms (`:80-82`) | *Qualitative:* "layer 2 atoms emerge that correspond to interpretable units… longer-range retrieval improves, retrieval at layer 2 is faster and more abstract" (`:84`) | Partial — abstraction emerges; analogical structure still located in Phase 5 |
| **5 — Binding discovery & atom splitting (THE CEILING)** | Learned bind-vs-bundle + atom splitting on persistent bimodality (`:88-98`) | *No single-number headline ("Hardest and least well-specified phase," `:98`).* Three emergence signals (`:96`): (a) **"bind-versus-bundle distinction emerges from data rather than being declared"**; (b) **"Analogical retrieval starts to work — patterns like 'X did Y to Z' surface stored patterns with similar structural shape but different content"**; (c) polysemous tokens → distinct atoms | **THIS IS where compositional/structural capability lives.** Reinforced at `:135`: "Structural retrieval starts working at Phase 5." |
| **6 — Integration** | LLM-in-workspace (Option A); replay re-encode (`:100-104`) | *Qualitative:* "SONAR-replacement is non-regressive on retrieval tasks, structural reasoning capabilities are present and measurable" (`:104`) | Structural reasoning *expected present* by Phase 6, built on Phase-5 capability |

**Net (the re-anchoring statement).** Phase 3's gate was *retrieval-quality-on-
real-vs-control + emergent semantic geometry* (`:70,74,134`) — explicitly **not**
held-out compositional generalization. The compositional/structural capability
was a **3→5 gradient** (`:125`) maturing into the **Phase-5 analogical-retrieval
gate** (`:96,135`). Phase 3 only *tracks the precursor* (bimodality flagging);
splitting "fires in Phase 5" (`phase-3-deep-dive.md:162-164`).

---

## 4. Biological-fidelity scorecard

*Five pillars from the North Star. Grounding: Benna-Fusi 2016 and the Kanan/
Hayes replay review were re-read this session; MESH/SQHN/predictive-coding/
Geometry-of-Consolidation primaries were not (card-routed claims flagged).*

| Pillar | Verdict | Summary | Key citation |
|---|---|---|---|
| **1. Continuous learning** | **TENSION** (faithful where it counts; one genuine gap) | Runtime error-driven write BAN is **not** drift — it is the faithful sleep/wake CLS split (offline cortical transfer over a frozen buffer). The gap: the *waking-online* half (Hebbian path) is under-evidenced relative to doctrine, and the wake/sleep state oscillation is deferred. | `STATUS.md:43`; `online_codebook.py:122,131`; `Kanan.replay.21.pdf` p.2,25; `phase-4-unified-design.md:344-345` |
| **2. No homunculus** | **PARTIAL** (honored in shipped code; one fenced violation) | The filter is real and enforced; shipped consolidation write passes AH review (`reports/057:85`). One **live** arbitration-shaped violation: the Phase-5′ `min_branch` aggregator FAILS the filter — but Phase 5′ is **PAUSED** so it cannot ship. De-arbitration fix already specified (logsumexp-soft-min / two-stream landscape). | `PROJECT_PLAN.md:16-18`; `audit-phase5-2026-05-26.md:381-385`; `STATUS.md:18,20` |
| **3. Data efficiency** | **TENSION** (efficient by architecture; validated result is memorization, opposite of "little data → generalization") | One-shot associative write (obs=1) is genuinely brain-like episodic encoding (no backprop-through-time). But what was *demonstrated* is memorization, not data-efficient generalization (held-out ≈ chance, `reports/056:7-12,72-74`). The capability that would prove "little data → broad competence" is the Phase-3→5 gradient — **not yet built.** No sample-efficiency benchmark run. | `reports/056:31-47,72-74`; `Kanan.replay.21.pdf` p.19 |
| **4. Energy efficiency** | **PARTIAL** (architecturally real; never benchmarked) | Structurally low-energy (Hopfield settle over fixed landscape, LLM interface-only). **No energy benchmark in the record** — grep found only an unrelated hit. MPS/GPU work is throughput engineering, not a head-to-head comparison. Clean on the AH filter. | `PROJECT_PLAN.md:29-30,277`; grep of `reports/` |
| **5. Replay-as-sleep** | **PARTIAL** (bridge wired & validated; deep-sleep *cycle* aspirational) | Strongest design-level analogue, partly earned empirically: Benna-Fusi multi-timescale chain + engagement-gated replay, offline consolidation write graduated & integrated (055→058). **Deferred:** the modeled wake/sleep oscillation, SWS/REM phase structure, selective/multi-region replay, REM novelty, buffer purging (`phase-4-unified-design.md:344-345`; `Kanan.replay.21.pdf` p.24-25). | `phase-4-unified-design.md:26-45,158-181`; Reports 012/013/055-058 |

**The biggest drift from biology (omission, not wrong-shape):** the **deep-sleep
*cycle* and the waking-online learning loop are both deferred**, leaving the
system with biology's offline-transfer half but not its rhythm. Honestly logged
as deferred, not hidden (`phase-4-unified-design.md:344-345`).

**The most faithful brain-analogue we have:** the **Benna-Fusi multi-timescale
offline consolidation write, integrated bit-identically.** Traced to a re-read
primary (plasticity–rigidity via fast↔slow synaptic cascades), runs offline like
cortical sleep-transfer, empirically graduated with multi-seed CIs, AH-clean. The
one caveat that keeps it from "brain solved": it is a memory, not a learner — the
cortical *generalization* that completes the CLS story is the unbuilt 3→5
gradient.

**No-homunculus risk points to watch:** (1) the fenced `min_branch` aggregator
(must de-arbitrate before any 5′ reopen); (2) borderline `bridge_readouts.py`
"choose a bridge readout" + `freq ≤ 32` cap (card-level flag,
`audit:323-325`); (3) any move toward **metric-triggered replay scheduling**
would re-import a controller (`literature_matrix.md:11` — not currently
violated).

---

## 5. Asset inventory mapped to phases + biological role

*Gate-status legend:* **CLEARED** = met the original gate with CI evidence;
**PARTIAL** = real validated signal but the original full bar is not met / was
reframed; **UNTESTED-vs-original-gate** = validated for some other metric, never
evaluated against the original gate; **FAILED/CLOSED** = run against the gate and
did not clear; **UNBUILT** = mechanism does not exist in `src/`.

### Phases 1-2 — substrate + retrieval (CLEARED)

| Asset | What it is | Validating reports | Phase / gate status | Biological role |
|---|---|---|---|---|
| `substrate/torch_fhrr.py` + `substrate/fhrr.py` | Batched FHRR algebra (bind/unbind/bundle) + pure-Python reference backend (non-negotiable) | 007/014/015; exercised at D=4096 in 055/056/058 | **Phase 1 — CLEARED.** Bind/unbind clean at D=4096 | Cortical distributed VSA representation; binding ≈ phase-coding of role-filler conjunctions |
| `memory/torch_hopfield.py` | Modern Hopfield retrieval (softmax settling, basin `top_index`, entropy, energy trace) | 001-004, 022 (β-regime split); leaf of 055/056/058 | **Phase 2 — CLEARED** as retrieval substrate | Hippocampal/attractor pattern completion — "what does this remind me of?" |
| `phase2/encoding.py` | Window encoding, position vectors, masked-window — the contextual-completion substrate | 016 (matrix audit), 005/006; reused by 055/056/058 | **Phase 2 — CLEARED.** Masked-token chosen as primary objective (016) | Episodic encoding of a scene as bound role-filler bundle |
| `phase2/metrics.py` | cap-coverage, meta-stable rate, Wilson CIs | 038; two-floor Wilson in 055/056 | **CLEARED** (canonical metrics) | n/a (measurement) |

### Phase 3 — codebook growth (the foundation; where the drift lives)

| Asset | What it is | Validating reports | Phase / gate status | Biological role |
|---|---|---|---|---|
| **`phase4/hetero_write.py`** (the graduated deliverable) | Error-correcting **delta-rule heteroassociative write** `H·k_i ≈ D·v_i` over a frozen cue→target buffer; batch-offline only; no `sims.argmax` thermostat; read terminates in a `top_index` basin | **055 GRADUATION** (D=4096 WikiText, multi-seed, info ceiling), **056 G-D PASS**, 057 (integrated), 058 (wiring bit-identical) | **CLEARED against the *reframed* in-sample memorization headline; NOT cleared against the original "generalizes over Phase-2" gate** — held-out real-text ≈ chance. This is the Phase-3 **floor**, exactly as `:125` predicted. | Hippocampal heteroassociative recall — pattern-complete a stored episode (role→filler) from a sparse cue. The core "memory is the self" recall |
| **`phase4/decorrelator.py`** (the sole active ingredient) | Batch **ZCA whitening** of cue space `P=(Σ+εI)^(−1/2)`, **L2-renorm** on apply (raw write alone = floor) | **054** (one-line L2 fix: floor 0.10→ceiling 0.43), 055/056 (elementwise ablation ties floor) | **CLEARED** as the load-bearing fix; AH-exempt (offline batch statistic) | Pattern separation (dentate-gyrus-style) — decorrelate overlapping cues so correlated contexts don't collide |
| `phase34/online_codebook.py` (`OnlineCodebookUpdater`) | Production consolidation-write orchestrator; integrated `observe(cue=)`→`consolidate_hetero()`→`recall_hetero()` hetero path; byte-identity guard | 028/032 (integration n=10), **057** (102 tests green, byte-identical), **058** (bit-identical) | **CLEARED for what it ships** (validated home of the surgical write). As a Phase-3-Recall graduation vehicle: **PARTIAL** (pull/push + context-residual paths headline-inert/closed) | Orchestration of wake/sleep codebook updates — "where the write happens" |
| `phase2/codebook_learner.py` (Hebbian) | Hebbian co-occurrence update | 017, 018 | **PARTIAL.** On Recall@1 no learned codebook beat random (017); on settled synergy @ mask beats random ~125× disjoint CIs (018). Headline was *reframed* from Recall@1 to settled-synergy — original "Recall@K over Phase-2" bar **not** the one cleared | Distributional semantic organization — "king/queen" test |
| `phase2/error_driven_learner.py` / `reconstruction_learner.py` | Error-driven contrastive / dense-reconstruction updates | 017, 018, 019 | **PARTIAL / superseded.** Beat random on settled synergy; tie/lose on Recall@1. Streaming error-driven form **BANNED** at runtime; survives batch-offline only | Error-correcting plasticity (BCM-like); dense predictive coding |
| `phase3/regime_diagnostic.py`, `bimodality_diagnostic.py`, etc. | Consolidation-geometry diagnostics; bimodality flag = Phase-5 atom-split **hook** | 044; STATUS pre-phase commitments (grep-confirmed resolved) | **CLEARED as diagnostics.** Bimodality flag is a hook, **not yet exercised for splitting** | Measurement of representational regime (AH-clean offline statistics) |
| **Frame-B continual-learning track** (pull/push, Γ1, slope/level-DiD) | The attempt to show the write does corpus-specific *learning* (DiD designs) | 113 (Γ1 FAIL), 114 (gauge-vacuous control), 115-118 (level-DiD CLOSED) | **FAILED / CLOSED.** Level-DiD not economically rescuable (117); slope dropped (116); variance irreducible (118). Old shuffled-token control **retired as gauge-vacuous**. **This is the documented drift.** | n/a — the negative result that motivated the (mis-scoped) reframe to memory |

### Phase 4 — replay / consolidation machinery

| Asset | What it is | Validating reports | Phase / gate status | Biological role |
|---|---|---|---|---|
| `phase4/consolidation.py` (Benna-Fusi) | Multi-timescale synaptic consolidation; knobs default 0 for reproducibility (each has a numbered report) | 022, 033/036/037, 040 (freq-α: null) | **PARTIAL.** D1 meta-stable-rate graduated (038, n=10, CI-disjoint) on a *substrate-pure* headline; Recall@10/cap-coverage stayed variance-bound. Several knobs empirically **null** (040). Original *hierarchical-compression* gate (L2-atom emergence) **never measured** | Sleep-replay-driven systems consolidation; synaptic tagging |
| `phase4/replay_loop.py`, `trajectory.py`, `snapshot.py` | Unified replay loop (trajectory→engagement gate→re-settle→Benna-Fusi→death GC); all-local (AH-clean) | 012/013, 022, 038 | **PARTIAL → graduated on D1 only** (038). Death mechanism flagged possibly-wrong-shape (binary vs soft) | Hippocampal replay + neurogenesis/pruning |
| `phase4/range_shaped_replay.py` | Static replay sampling from `p(role)·p(atom)` marginals (Dorrell-motivated) | 068, 069-074, 111 | **PARTIAL/CLOSED on storage side** (111 closed `range_postsettle` proxy); **Dorrell nonneg-AE training objective remains OPEN** (Stage-1 candidate, fenced) | Experience-replay distribution shaping — "what gets replayed" |

### Phase 5 / 5′ — structural retrieval (the one sanctioned building-time addition)

| Asset | What it is | Validating reports | Phase / gate status | Biological role |
|---|---|---|---|---|
| **`phase5/bundle_first_scene_memory.py`** | Reusable **bundle-first scene-memory**: store each scene as one MHN pattern = normalized Σ bind(role,filler); cue → Stage-1 scene-MHN identify → Stage-2 algebraic unbind → Stage-3 cleanup. *The structural-retrieval machinery Phase 5 always wanted.* | **067** (multi-role MQAR: passes to N=256 ≥88%, **3 seeds**), 066, 075-081 (scene-token anchor; PAM held-out scene-key=0.000) | **PARTIAL / UNTESTED-vs-original-gate.** Works **as a memory** (in-sample scene-ID memorization, n=10). But this is *stored-item retrieval, not the emergent bind-vs-bundle discovery* the gate names; held-out scene-key = chance. **The original Phase-5 discovery/analogy gate is NOT yet tested.** n=3 MQAR is below the n≥10 verified bar | Hippocampal scene/episode index — "which event was this?" before role-filler readout |
| `phase5/ham_aggregator.py` (Krotov-2021 HAM) | Coupled multi-scale settling: bottom-up activate, geometric-mean consensus, top-down bias. AH-clean | 022 (HAM-arithmetic), `reports/phase5_ham_full/` | **PARTIAL.** Validated as a retrieval aggregator; regime-split adopted (β=30 retrieval / β=10 replay). Did **not** demonstrate the original Phase-5 structural-discovery win | Cortical hierarchical attractor consensus |
| `phase5/ham_with_layer2.py` | HAM + Phase-4-replay-discovered L2 attractors | `reports/phase5_layer2_validation/` | **PARTIAL/UNTESTED-vs-original-gate.** L2 discovery pathway built + control-tested; not shown to clear "interpretable L2 atoms + longer-range retrieval improves" | Chunking / phrase-level abstraction |
| `phase5/m1_role_energy.py` (`M1` role-energy stack) | Path-D/M1 retrieval-time role-prior primitives | 064 (null), 060-063 | **FAILED at smoke scale** (retrieval-time interventions null, 062-066). **Training-time M2 path remains OPEN** | Top-down role/content prior on retrieval |
| `phase5/bridge_readouts.py`, `role_fidelity.py`, etc. (Phase 5′) | ΔE bridge readouts, role-fidelity, natural-source protocol | 099-111 | **PARTIAL / PAUSED.** Pre-pause ΔE headline spec exists (`phase-5-unified-design.md:282-297`); 2026-05-26 audit §9 flags **saturating `raw_scene_energy_v0`** + **arbitration-shaped `min_branch` that FAILS anti-homunculus** — de-arbitrate before any reopen | Role-prior vs content-prior energy differential (the 5′ structural signature) |
| **Phase-5 signature mechanisms** (atom-splitting *action*, bind-vs-bundle *discovery*, analogical-retrieval) | The three capabilities the original Phase-5 gate names | — (grep: none exist in `src/`) | **UNBUILT.** This is the genuine frontier (§8) | Lexical sense differentiation; learned compositional binding; structural analogy |

### Secondary / exploratory (honestly net-negative or design-only)

| Asset | What it is | Validating reports | Status |
|---|---|---|---|
| `phase4/ws_infonce.py` (WS-InfoNCE) | Within-scene InfoNCE value-codebook shaping (generalization track Stage-1) | **119** (held-out margin ≤ 0 at every cue richness, 0/6 cells, net-negative at rich-cue end; 10/10 tests) | **FAILED** (no held-out lift; confirms memory-not-learner on *value-codebook shaping specifically* — codebook collapse d_eff 341→~165). DON'T-SCALE-A-NULL honored. Graduated path unaffected |
| MESH-scaffold | Fixed random scaffold for N≫D scaling | **120** (RESOLVED → **defer**; graduated H intrinsically low-rank, SVD-truncated byte-identical at r=D/16) | **NOT BUILT / deferred — wrong tool.** Cost fallback is factored low-rank H (~30 LOC, also not built) |

---

## 6. The headline-metric drift: timeline, root cause, corrections

### Drift timeline

| # | When | Headline WAS | Drifted TO | Mechanism | Citation |
|---|------|--------------|------------|-----------|----------|
| **0 (baseline)** | 2026-05-09 | **Phase-3 original gate:** regime-stratified masked-token Recall@K vs **shuffled-token control**; compositional generalization is a *drill-down* "increasing with phases 3-5" | — | The anchor. Generalization located as a 3→5 gradient → Phase 5 | `experimental-progression.md:70`, `:125`, `:88-98` |
| **1** | 2026-05-28 | Original gate | **Frame-B endpoint level-DiD** | C.3 shuffled control found **gauge-vacuous** (`E[Δ]=0` by construction). *Legitimate* (control was broken) — but severed headline from "retrieval quality" → "variance differencing" | `STATUS.md:16` |
| **2** | 2026-05-30 | Level-DiD | **value-codebook `top_index_hits` Selectivity-Δ** (two-floor Wilson) | Level-DiD "not economically rescuable" (116/117/118). New spec authored — **does NOT cite or reconcile with `experimental-progression.md:70`** | `STATUS.md:17`; `phase-3-consolidation-write-design.md:50-71` |
| **3** | 2026-05-31 | Selectivity-Δ on clean 4-role toy, held-out (chance 0.25) | Selectivity-Δ on masked-encoding substrate, chance → `1/\|value codebook\|` | §G-D substrate amendment (7-agent panel: clean toy provably can't reproduce the signature). Defensible — but moved chance baseline 0.25 → ~0.002 | `phase-3-consolidation-write-design.md:73-101` |
| **4 (load-bearing)** | 2026-05-31 | Selectivity-Δ judged **held-out** (spec literally requires it) | **In-sample** Selectivity-Δ; held-out demoted to "secondary"; **"memorization = the target"** elevated to project identity | Adversarial verification caught the read was in-sample. Instead of accepting the held-out null *as the result*, it was re-interpreted via "contextual-completion NOT prediction" into "memorization is the target → judge in-sample (PASS)." A **narrow true negative** hardened into a project-identity claim | `phase-3-consolidation-write-design.md:103-116`; Report 056 title; `STATUS.md:18` |
| **5** | 2026-05-31 | (consequence of #4) | Memorization framing propagated to **permanent memory** as standing identity | Two memory notes + STATUS encode "memorization = the target" as settled; Report 119 then treats it as "the FLOOR, not relitigated" → self-reinforcing across 056→119 | `neuro_ai_memory_not_learner.md`; Report 119 preamble; `STATUS.md:28` |

**Net displacement (0 → 4):** from *"regime-stratified masked-token Recall@K vs a
corpus-artefact control, compositional generalization explicitly deferred to
Phase 5"* to *"in-sample role-selective recall passes, and held-out
generalization failing is by design."* **The drill-down the original plan
deferred (`:125`) was promoted into a disqualifying test the mechanism is then
declared exempt from.**

### Root cause — four moves

1. **The held-out null is real and correctly measured** (056:67-99; D=4096 panel
   056:73-77) — *and it is exactly what `:125` predicted.* The original plan
   already told us this would be near-null at Phase 3.
2. **The substrate swap removed the comparison to the original gate.** The live
   headline migrated to a new spec (`phase-3-consolidation-write-design.md`) that
   never cites `experimental-progression.md:70` or `:125`. With no back-link,
   nothing forced the question *"is held-out-null a Phase-3 floor or a project
   ceiling?"* — the precise failure CLAUDE.md §session-start-rule-3 warns about.
3. **A correct architectural axiom was over-applied.** "Contextual-completion NOT
   sequence-prediction" (`PROJECT_PLAN.md:276`) is about the *prediction
   objective*; it is **not** a statement that the system should never generalize
   compositionally. `experimental-progression.md:88-98` is the project's *own*
   commitment to compositional generalization — as a later-phase capability, not
   an abandoned one. The axiom was stretched to license judging G-D in-sample.
4. **The reframe was ratified into identity before being scoped to the phase.**
   It now lives in permanent memory and Report 119's preamble as "the FLOOR, not
   relitigated," so every subsequent session inherits it as a premise. **The
   Phase-3 floor got written down as the project's ceiling** — brushing the
   non-negotiable `PROJECT_PLAN.md:276`.

**This is the same failure CLAUDE.md documents twice as precedent:** the Phase-5
ΔE drift (CLAUDE.md:39-45, an unread design-spec headline) and the grep-by-
default walk-backs (CLAUDE.md:64-71, a stale-but-trusted STATUS claim driving
downstream artifacts). The experiment-preamble forcing function *did* fire —
Report 056 cited a spec — but it cited the **new** spec that had already absorbed
the drift, so the citation chain never reached back to `:70/:125`. The drift was
laundered through a freshly-authored spec rather than caught.

**Bottom line:** no result is wrong; the *scoping* is. The Phase-3 floor
(in-sample recall passes; held-out compositional generalization ≈ chance, exactly
as `:125` predicted) was written down as the project ceiling.

---

## 7. Honest current position vs original gates

| Phase | Original gate | Actual status | Honest assessment | Citation |
|---|---|---|---|---|
| **1 — Substrate** | Clean bind/unbind at D=4096 (`:33-39`) | **CLEARED** | Genuinely done. Memorization retrieval exact (1.000 on stored windows). Never re-litigated | `reports/016:124-129`; `reports/015` |
| **2 — Static codebook** | Masked/next Recall@1 above chance, CI-disjoint (`:43-64`) | **CLEARED** | Done, with a *sharpening*: argmax-Recall@1 was the wrong **shape** (random ties learned codebooks); real signal in settled synergy (~125×, disjoint CIs). Masked-token confirmed primary → vindicates contextual-completion thesis | `reports/016:65-159`, `reports/018` |
| **3 — Codebook growth** | Regime-stratified Recall@K vs shuffled control, over Phase 2 (`:66-78`) | **PARTIAL** | **Drift epicenter.** Shuffled control proven gauge-vacuous; Frame-B DiD replacement closed (variance structurally irreducible, 116-118). **What IS cleared:** the consolidation **write** graduated as a role-selective associative memory (in-sample, all controls, D=4096 WikiText 3/3), integrated byte-identically. Phase-3 **floor** solid; original **corpus-specificity headline** open | gauge: `2026-05-28-...gauge-control-finding.md:48-90`; nulls `116/117/118`; write `055-058` |
| **4 — Hierarchical compression** | L2 atoms emerge; longer-range/abstract retrieval (`:80-86`) | **PARTIAL** | Graduated on a **substituted** headline (D1 Δmeta-stable-rate, n=10, CI-disjoint), **not** the hierarchical-compression gate. Replay/Benna-Fusi machinery real & verified; **L2-atom emergence never measured** (death mechanism vacuous A12; pattern-age never measured D3; entropy exit fails E3). What graduated: "replay stabilizes the substrate," not "interpretable phrase-level atoms emerge" | `phase-4-checklist.md:57,140,41,78,90` |
| **5 — Binding discovery + atom splitting + analogical retrieval** | Bind-vs-bundle emerges; bimodal atoms split; analogical retrieval works (`:88-98,:135`) | **UNBUILT (mechanisms) / PAUSED (energy headline)** | The three signature capabilities **do not exist in `src/`** (grep confirms). The Phase-5 *energy* headline (ΔE role-prior vs content-prior) reached n=10 **directional but sub-noise-floor** (+0.0013, 4.2× below the 5.5e-3 floor) → **PAUSED**. The bridge feeding it is **closed at smoke scale** (104-111, `min_branch` collapse + AH-failing aggregator) | `phase-5-checklist.md:39`; `STATUS.md:20`; `phase-5-prime-checklist.md:98-104` |
| **5′ — Bundle-first scene memory** | (Not an original phase — the machinery Phase 5 always wanted) | **PARTIAL (validated primitive) / PAUSED (graduation)** | Bundle-first scene-MHN **validated as a memory** (MQAR multi-role, K_roles≤16, clean controls; Reports 067/092-099) — a genuine asset. But **no n≥10 graduation run**; range-shaped replay downstream closed-not-viable (F9); ΔE bridge closed; WS-InfoNCE generalization probe **null** | `phase-5-prime-bundle-first-design.md:29-38`; `phase-5-prime-checklist.md:68`; `STATUS.md:28` |
| **6 — Integration** | SONAR-replacement non-regressive; structural reasoning measurable (`:100-118`) | **UNBUILT** | Not started; correctly downstream of 3-5 | n/a |

### What the drift cost

The drift is precisely *a Phase-5 capability (compositional/held-out
generalization) used to judge the Phase-3 write, then the FLOOR result labeled
the project's CEILING.* Cost in sessions (lower-bound, from the dated chain):

- **~3-4 sessions** chasing the Frame-B corpus-specificity headline through a
  control that was **gauge-vacuous from the start** (Path C / Path α / Γ1 nulls,
  Reports 112-114) — the `E[Δ]=0` proof shows none of those nulls were ever
  interpretable as evidence about learning.
- **~3 sessions (2026-05-30)** closing three rescue levers for the level-DiD
  (slope-DiD 116, n-escalation 117, landscape-size 118) — all confirming the
  variance is consolidation-injected and structurally irreducible (the contrast
  itself was the wrong instrument).
- **~2 sessions** building + running the generalization track (Dorrell port → 3
  fatals → design-only; WS-InfoNCE Stage-1 → held-out null, Report 119) to
  re-confirm what `:125` already predicted.

The throughline: each spent effort proving a **Phase-5-shaped claim** (corpus-
specific generalization beyond memorization) **at the Phase-3 substrate**, where
the original plan says it cannot and should not appear yet.

---

## 8. The frontier: the genuine next step toward the thesis

**The next step is to BUILD the Phase-5 structural-retrieval mechanisms on the
validated Phase 3-4 + bundle-first foundation — this is where the original plan
locates the generalization the project keeps trying to extract one phase too
early. It is reframe-PLUS-a-scoped-build, NOT "rewrite everything."**

### Reframe (zero new code, do first)

1. **Relabel the consolidation write as the Phase-3 contextual-completion floor
   that graduated** (`reports/055-058`, byte-identical-integrated), *not* "just a
   memory." Per `experimental-progression.md:125`, in-sample role-selective
   memorization is the *expected* Phase-3 result; held-out compositional
   generalization is a Phase-5 deliverable. Re-scope the "memory-not-learner"
   note to "Phase-3 floor cleared; generalization deferred to Phase 5 by original
   design" — removing the tension with the no-vector-DB rule.
2. **Retire the original shuffled-token-control headline as gauge-vacuous**
   (already done in spec) and **stop treating the Frame-B corpus-specificity DiD
   as a Phase-3 graduation gate** — Reports 117-118 show its estimand is
   structurally unable to detect the effect cheaply. Note it as "structurally
   underpowered at this operating point," not "Phase 3 failed."

### Scoped build (the genuine forward move)

The original Phase-5 success signal is **analogical/structural retrieval**
("retrieve stored patterns with similar structural shape but different content,"
`:96,:135`) and **atom-splitting** (`:88-98`). The bundle-first scene-MHN is the
**already-validated storage primitive** for exactly this
(`phase-5-prime-bundle-first-design.md:29-38`) — it is the structural-retrieval
machinery Phase 5 always wanted, not a deviation. The targeted build:

- Make the validated bundle-first scene-MHN a **first-class Phase-5 retrieval
  unit** (it currently lives only in MQAR/diagnostic harnesses + the null
  `ws_infonce.py`), and define a **held-out *structural-shape* analogical query**
  as the Phase-5 headline — distinct from the sub-noise ΔE energy headline, which
  is paused for separate reasons.

### Two binding gates before any code (do not skip)

- **De-arbitrate `min_branch` first.** The pre-existing Phase-5′ aggregator FAILS
  the anti-homunculus filter (`STATUS.md:18,20`); branch selection must be
  energy-only (`phase-5-checklist.md:194` H3). Hard prerequisite for any Phase-5
  reopen.
- **A held-out PASS is the only admissible generalization evidence.** Report 119
  established the paper-backed generalization cluster rests on PAM alone (itself
  memory-not-learner), so a structural-retrieval headline above-chance **on
  held-out structural shapes** is the load-bearing claim — not an in-sample
  memorization re-run (already done).

### What a scoped build is NOT

Not a substrate rewrite (Phase 1-2 cleared), not re-running the closed Frame-B
DiD, not reopening the sub-noise ΔE energy run, and not scaling any confirmed null
(gauge control, range-replay F9, WS-InfoNCE) — "don't scale a null" is already
honored (`STATUS.md:28`).

> **Reframe-vs-rebuild recommendation: REFRAME + a single scoped Phase-5 build.**
> The substrate, the graduated completion write + L2 decorrelator, the emergent
> codebook, and the bundle-first scene-memory primitive are validated **assets to
> build on**. We are at the **Phase-3 floor, cleanly cleared as a memory**, with
> Phase-5's structural-discovery gate **unopened** — which is exactly where the
> original 3→5 gradient predicted we'd be. There is no evidence base for a
> rewrite; there is a clear, plan-sanctioned next mechanism.

---

## 9. Immediate corrections (de-drift the docs)

Framing + headline re-anchoring only. No code changes. The fix is to **re-attach
every "memorization = the target" claim to "at Phase 3,"** restore compositional
generalization as the **Phase-5** target it always was (`:88-98`), and add the
missing back-citations from the live spec to `experimental-progression.md:70/
:125`.

1. **`STATUS.md:18`** — delete the bare standalone "**memorization = the
   target**." Replace the held-out-null clause with: held-out compositional
   generalization ≈ chance **at the Phase-3 floor** — which
   `experimental-progression.md:125` **predicted**; in-sample role-selective
   **recall PASSES** (the Phase-3 floor per `:70`); compositional generalization
   is **deferred to Phase 5** (`:88-98`), **not** out-of-scope. The mechanism is
   a validated completion-write, **not** "just a memory" (cf.
   `PROJECT_PLAN.md:276`).

2. **`phase-3-consolidation-write-design.md:103-116`** — keep "judge G-D
   in-sample" as the *Phase-3 gate*. **Strike** the inference "the target
   capability is memorization-recall, *not rule generalization*"; replace with
   "the target capability *at Phase 3* is memorization-recall; rule/compositional
   generalization is the **Phase-5** target and a held-out null here is the
   predicted floor." Scope "a memory, not a learner" → "a memory **at this
   phase**." **Add** a one-line back-citation to `experimental-progression.md:70`
   explaining Selectivity-Δ is the operationalization of "regime-stratified
   Recall@K vs control" after the C.3 control was retired as gauge-vacuous.

3. **Report 056** (`reports/056_gd_selectivity_panel/report.md`) — reports are
   historical record; **append** (do not rewrite) a framing-correction note: the
   in-sample PASS is the **Phase-3 floor** (`:70`); the held-out null on real text
   is the **predicted** floor behavior (`:125`), generalization deferred to
   **Phase 5** (`:88-98`). Strike the parenthetical equation "memory… not…
   generalization" as a project-level claim — it conflates the prediction
   objective with the (orthogonal) capability axis.

4. **`neuro_ai_memory_not_learner.md`** (memory note) — re-scope to the phase: "At
   **Phase 3**, the floor capability is memorization-recall (PASSES).
   Compositional generalization is the **Phase-5** target (`:88-98,:125`), not a
   secondary/abandoned capability. The held-out real-text null is the **predicted
   Phase-3 floor**." Remove the unscoped "memorization is the target" from the
   `description:` frontmatter.

5. **`project_status.md` (memory note) + `STATUS.md:28` + Report 119 framing** —
   memorization-as-"the FLOOR" is the *right word*, but must be paired with "the
   **ceiling** (compositional generalization) is **Phase 5**, currently unbuilt,
   not dead." The "a memory by design" verdict applies to *those specific
   value-codebook-shaping levers* (scene-ID, WS-InfoNCE) being dead ends — it must
   **not** be generalized into "the project is a memory by design." Drop the
   standalone "Memorization = the target" identity claim from `project_status.md`.

6. **Cross-cutting structural fix (prevents recurrence):** add to
   `experimental-progression.md` (near `:70`) a one-line note recording that the
   headline was operationalized as Selectivity-Δ after the C.3 control was retired
   (2026-05-28), so the **original gate doc and the live spec point at each
   other**. The bidirectional citation is the structural fix that would have made
   `:125` impossible to leave unread — the root cause of this entire drift was the
   live headline migrating into a new spec with no back-link to the original
   per-phase gate.

---

*End of RE-GROUNDING MAP. Validated assets to build on (do not rewrite): FHRR
substrate + pure-Python reference, Modern Hopfield basin readout, masked-token
encoding + Phase-2 metrics, the emergent (Hebbian/synergy) codebook, the
graduated heteroassociative write + L2 decorrelator and its production
integration, and the bundle-first scene-memory + HAM machinery. Open frontier:
the three unbuilt Phase-5 signature mechanisms (bind-vs-bundle discovery,
atom-splitting action, analogical retrieval), gated behind de-arbitrating
`min_branch` and a held-out structural-shape headline.*
