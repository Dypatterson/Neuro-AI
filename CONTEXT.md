# CONTEXT — Neuro-AI (read at session start)

*The stable charter: what this project IS, the original phase gates, and the
invariants that keep the work from drifting. This file changes slowly. For
**where we are right now**, read [STATUS.md](STATUS.md) (the volatile bookmark).
For the **2026-05-31 deep re-grounding snapshot** that seeded this file, see
[notes/RE-GROUNDING-MAP.md](notes/RE-GROUNDING-MAP.md).*

> **Why this file exists.** Over 2026-05-28→31 the project drifted: a *narrow,
> correct* Phase-3 result (held-out compositional generalization ≈ chance — which
> the original plan **predicted**) hardened into a project-identity claim
> ("memorization is the target / it's just a memory"). That brushed the
> non-negotiable *"do not collapse the memory into a vector database plus
> summaries."* The root cause was structural: the binding session-start read
> (`STATUS.md`) drifted, and the original per-phase gates
> (`experimental-progression.md`) were never in the session-start read set, so
> nothing forced the question *"is this null a Phase-3 floor or a project ceiling?"*
> This file is the durable fix — the original gates + the invariants, read first.

---

## 1. The bet

Current LLMs are amazing but (a) energy-inefficient, (b) cannot learn
continuously — their "understanding" is locked in frozen weights, and (c) need
enormous data. The human brain is the opposite: it starts **not-smart**, learns
**continuously** from **little** data, is **energy-frugal**, and has **no
homunculus** controlling it. **The wager: the data-hungry / backprop /
autoregressive path is the wrong bet, and biology/neuroscience gives the answer.**
Every mechanism here — emergent codebook, replay-as-deep-sleep consolidation,
Hopfield associative recall, FHRR binding, coupled energy terms — was chosen
because it is **brain-analogous**. *(Goal sentence: `docs/PROJECT_PLAN.md:3-7`;
the bet: `notes/briefing.md:36-40`. "Energy-frugal" is a design commitment, NOT a
benchmarked result.)*

## 2. What this system IS — three senses of "memory," and which one we are

1. **Storage / lookup** (a vector DB + summaries). **Forbidden** — the
   non-negotiable rule is *"Do not collapse the memory into a vector database
   plus summaries"* (`docs/PROJECT_PLAN.md:276`).
2. **Contextual completion** — pattern-complete an unresolved gap by *settling
   into an attractor*: *"what does this remind me of, and what fills this gap?"*
   not "what comes next?" (`PROJECT_PLAN.md:21-22`, `overview.md:23`). Retrieval
   is generative, not lookup — blends never explicitly stored can emerge. **This
   is the Phase-3 floor, and it is built + graduated** (Reports 055–058).
3. **Abstraction — "the landscape geometry IS the rule, grown from experience"**
   (`overview.md:17`); the durable identity is the *learned landscape*, not frozen
   weights (`PROJECT_PLAN.md:19-20`). **This is the "more than memory" thesis — and
   it lives in Phase 5, which is not yet built.**

> **THE LOAD-BEARING INVARIANT (do not re-drift):** *Memorization is the Phase-3
> FLOOR, not the project's ceiling or identity.* Compositional / structural
> generalization is a **Phase-3→5 gradient** that the original plan explicitly
> deferred to Phase 5 — *"Don't expect to crush these; expect meaningful signal
> that increases with phases 3-5"* (`experimental-progression.md` §"What to test
> against"); *"Structural retrieval starts working at Phase 5"* (§"How to know
> it's actually working"). A held-out
> generalization null at Phase 3 is the **predicted floor behavior**, NOT a verdict
> on the architecture. Never let "it's a memory" (true of the Phase-3 floor) become
> "the project is a memory" (false — that's sense #1, which is forbidden).

## 3. Core principles (invariants)

- **No homunculus** — every addition is a *local geometric dynamic or a
  measurement of one, never an arbitration over them* (`PROJECT_PLAN.md:16-18`;
  `notes/notes/2026-05-09-...:93`). No supervisor picks a winner; no
  metric-triggered branch. Runtime error-driven codebook writes are **BANNED**
  (batch-offline only — the sleep/wake split; `STATUS.md` live policy).
- **Memory is the self** — identity is the learned landscape + its trajectories
  (`PROJECT_PLAN.md:19-20`).
- **Contextual completion over token prediction** (`PROJECT_PLAN.md:21-22`).
- **Continuous learning** — live experience writes traces; replay consolidates,
  *abstracts*, and reshapes the landscape (`PROJECT_PLAN.md:23-26`).
- **Latent reasoning** — reason in vectors/energy before language
  (`PROJECT_PLAN.md:27-28`).
- **Energy efficiency / local-first** — compact latent ops, MacBook-class
  (`PROJECT_PLAN.md:29-32`; aspirational, unbenchmarked).
- **Headline-vs-drill-down** — each phase has ONE headline gate; the rest explain
  it, they don't redefine success.
- **Phase order** — Phase-N work presupposes Phase N-1's gate is cleared; skipping
  is allowed but must be acknowledged, and **building a later phase on an
  unverified earlier gate is the drift this file exists to prevent.**

## 4. The original six-phase plan + per-phase gates

*Source of truth: `notes/emergent-codebook/experimental-progression.md` (2026-05-04;
metrics added 2026-05-09). Phase 3 is the FOUNDATION; Phase 5 is where
"more-than-memory" lives.*

| Phase | Goal | Original headline gate | Current status |
|---|---|---|---|
| **1 Substrate** | FHRR bind/unbind/role-filler recovery at D=4096 (§Phase 1) | Clean recovery, no surprises at scale | **CLEARED** |
| **2 Static codebook** | Masked- vs next-token as a design comparison (§Phase 2) | Recall@1 vs bigram, ≥1 objective above chance, CI-disjoint (§Phase 2) | **CLEARED** (masked-token chosen → vindicates contextual-completion) |
| **3 Codebook growth (FOUNDATION)** | Two-pathway hybrid update (§Phase 3) | Regime-stratified masked Recall@K vs a **valid** control; **"similar tokens → similar hypervectors"; structure ABSENT in the control** (§Phase 3; §"How to know it's actually working") | **PARTIAL** — see §5 (floor cleared; structure-gate OPEN) |
| **4 Hierarchical compression** | L1 bundles → L2 atoms (§Phase 4) | Interpretable L2 atoms emerge; longer-range retrieval improves (§Phase 4) | **PARTIAL** (replay stabilizes substrate; L2-emergence never measured, deferrable) |
| **5 Binding discovery + atom-splitting + analogical retrieval (THE CEILING)** | Learned bind-vs-bundle + split persistently-bimodal atoms (§Phase 5) | Bind-vs-bundle **emerges from data**; **analogical retrieval works** ("similar structural shape, different content"); polysemy splits (§Phase 5; §"How to know it's actually working") | **UNBUILT** (mechanisms absent from `src/`) / energy headline PAUSED |
| **6 Integration** | LLM-in-workspace; replay re-encode (§Phase 6) | SONAR-replacement non-regressive; structural reasoning measurable | **UNBUILT** (correctly downstream) |

*(Citations to `experimental-progression.md` are by section header, which is stable across edits — the line numbers shifted once already and broke the references they fed.)*

## 5. Where we are — and the current crux

The contextual-completion **floor** (Phase 3, sense #2) is **cleared**: the
surgical heteroassociative write + L2 decorrelator recalls role→target bindings
role-selectively at the information ceiling, multi-seed, integrated bit-identically
(Reports 055/056/057/058), scaling settled (dense H; Report 120).

**THE CRUX — structure-gate "3b" RAN (Report 121, 2026-05-31) → the meaningful
(paradigmatic) gate is a NULL → RE-SCOPE the codebook-growth dynamics.** 3b asked:
does the codebook develop corpus-specific structure ("similar tokens → similar
hypervectors," vs a gauge-safe corpus-stream-shuffle control)? Decomposed answer:
- **COLLOCATIONAL** (co-occurring words cluster): **YES** — small (+0.010 real−shuffle
  on WikiText, ~12% of the drift; the rest is corpus-independent) — but a co-occurrence
  learner produces this ~by construction. It is the floor, not the thesis signal.
- **PARADIGMATIC / semantic** (similar, *non-co-occurring* words cluster — the king/queen
  test; the substrate Phase 5's bind-vs-bundle discovery / atom-splitting / analogical
  retrieval operate on): **NO** — WikiText paradigmatic subset −0.0084, CI [−0.013, −0.004]
  (*negative*); repo_sample null; corr(log co-occurrence, drift) = +0.41/+0.82 → the effect
  scales with co-occurrence, i.e. it is collocational, not similarity-based.

So the **first-order Hebbian distributional-centroid growth dynamic is the wrong shape**
for the thesis: it captures syntagmatic (co-occurrence) structure, not paradigmatic
(substitutional / second-order) structure. (An adversarial verification caught that a
tempting statistical "pass" on the overall semantic arm was an artifact of a 93%-collocation-
contaminated pair list — the genuinely paradigmatic pairs, including king/queen itself, do
not cluster.) The original Phase-3 control was gauge-vacuous; the Frame-B DiD is a closed
drift-artifact — neither is re-chased here.

> **RE-SCOPE (per the pre-registered prior — honored, NOT a retry):** **Phase 5 remains
> un-founded** (its mechanisms need paradigmatic structure, which does not emerge). The
> genuine next move is a **Phase-3 growth-mechanism redesign** — a candidate that clusters
> by **context similarity** (second-order: tokens with similar *neighborhoods*, not tokens
> that are *neighbors*) — e.g. a context-vector / SQHN / predictive-coding-style update,
> replacing the first-order co-occurrence centroid. This is NOT a Phase-5 build and NOT a
> larger-scale re-run of this null. **This is now the most important open question.**

## 6. Session-start read order (with this file)

0. **This file (`CONTEXT.md`)** — the thesis, the gates, the invariants, the crux.
1. [STATUS.md](STATUS.md) — current position (volatile bookmark).
2. The active phase checklist under `notes/emergent-codebook/`.
3. The §Headline + §Required-controls of the active phase's design doc (cite line
   numbers — STATUS banners drift; the design spec + §4 gates above are the truth).
4. `docs/PROJECT_PLAN.md` + relevant dated `notes/notes/`.

If `CONTEXT.md`, `STATUS.md`, and a design spec disagree, that contradiction is a
**finding to surface**, not paper over (it is exactly the failure that produced this
file).
