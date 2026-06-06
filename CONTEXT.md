# CONTEXT — Neuro-AI (read at session start)

> **⚑ TWO BETS (added 2026-06-05).** This file is the charter for **Bet A** —
> *"the data-hungry / backprop / autoregressive path is the wrong bet; biology
> IS the answer"* (no-backprop, no-global-as-mechanism, local-only). A parallel
> line, **Bet B** — *"biology is the spotlight, not the box; use whatever
> mechanism works, held to the same discipline"* — has its own binding charter at
> **[CONTEXT-B.md](CONTEXT-B.md)**. **For Bet-B work, CONTEXT-B.md governs**, and
> the **mechanism bans below (the no-backprop / no-global / local-only invariants
> in §1 and §3) are Bet A's — NOT binding for Bet B.** The **discipline** (§3
> headline-vs-drill-down, multi-seed, controls, anti-homunculus-as-no-supervisor,
> "do it right / no shortcuts") is binding for **both** bets. Do not read Bet A's
> mechanism bans back onto Bet B (that re-import is exactly why CONTEXT-B exists).

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
  **Structural priors are not homunculi** (clarified 2026-06-01). A *fixed, problem-generic*
  architecture — number of layers, the wiring, the operation each layer runs, the precommitted k-WTA cap,
  the dimensionality D — is an evolution-style **scaffold** (set in advance, identical across problems)
  and is **LEGAL**. The ban is only on (a) a **supervisor that arbitrates outcomes** (reads a metric →
  picks a winner / routes content), and (b) a **global error signal** (backprop) reshaping the scaffold
  toward a global objective. Brain-analogy: evolution fixes the layered/compartmentalized architecture;
  *local* plasticity fills it from experience. **Boundary:** the scaffold must stay problem-GENERIC (it
  discovers whatever differentiating axes the data has) — a scaffold *hand-shaped to the target answer*
  is a **design-time homunculus**, the structural cousin of tuning-to-pass.
- **Memory is the self** — identity is the learned landscape + its trajectories
  (`PROJECT_PLAN.md:19-20`).
- **Contextual completion over token prediction** (`PROJECT_PLAN.md:21-22`).
- **Continuous learning** — live experience writes traces; replay consolidates,
  *abstracts*, and reshapes the landscape (`PROJECT_PLAN.md:23-26`).
- **Latent reasoning** — reason in vectors/energy before language
  (`PROJECT_PLAN.md:27-28`).
- **Energy efficiency / local-first** — compact latent ops, MacBook-class
  (`PROJECT_PLAN.md:29-32`; aspirational, unbenchmarked).
- **Headline-vs-drill-down** — each capability/experiment has ONE headline gate;
  the rest explain it, they don't redefine success.
- **Dependency order, not an exclusive sequence** — a capability presupposes its
  *dependencies'* gates are cleared (the DAG in §4), but capabilities are NOT worked
  one-at-a-time-to-completion. **Any capability whose dependencies are met is
  workable, and cross-capability *combination* experiments are first-class**;
  testing a mechanism in isolation when the real one is an *interaction* is itself a
  failure mode (it produces false negatives — see §4). The one hard rule that
  survives the reframe: **building an abstraction capability on an unverified floor
  is the drift this file exists to prevent.**

## 4. The capability map (dependency DAG + gates) — *was "the six-phase plan"*

*Reframed 2026-06-01. The six "phases" were really a set of **coupled capabilities over one
substrate**, not a linear curriculum — and the linear framing forced isolation-testing + the
recurring phase-boundary drift this file keeps fighting. **The GATES below are unchanged** (source
of truth: `experimental-progression.md`, 2026-05-04; metrics 2026-05-09); only the **structure**
changed — a ladder → a dependency graph that licenses combination experiments. The historical
`notes/emergent-codebook/phase-<N>-*` design docs remain the per-capability design records (Substrate
= P1, Static-codebook = P2, Codebook-growth = P3, Replay = P4, Abstraction = P5).*

**The DAG** ( `→` = "depends on the gate of" ):
`Substrate → Static-codebook → Consolidation-write (FLOOR) → { Codebook-growth ⇄ Replay } → Abstraction (CEILING) → Integration`

- **Codebook-growth** and **Replay** are **siblings that COUPLE** (`⇄`): replay shapes what growth
  consolidates; growth supplies what replay interleaves. **They must be testable in COMBINATION, not
  only in isolation** — the 121-127 arc tested growth alone and may have produced false negatives for
  exactly this reason.
- A capability is workable the moment its *dependencies'* gates clear — NOT after the prior "phase"
  finishes. **Cross-capability combination experiments are first-class.**
- The **build fence** (don't commit a new substrate / Abstraction architecture without a decision) is
  a **build-gate on the Abstraction node**, not a phase-number rule.

| Capability (was Phase) | Goal | Original headline gate | Depends on | Status |
|---|---|---|---|---|
| **Substrate** (P1) | FHRR bind/unbind/role-filler recovery at D=4096 | Clean recovery at scale | — | **CLEARED** |
| **Static codebook** (P2) | Masked- vs next-token design comparison | Recall@1 vs bigram, ≥1 objective above chance, CI-disjoint | Substrate | **CLEARED** (masked-token chosen → vindicates contextual-completion) |
| **Consolidation-write = the FLOOR** (P3/P4 write) | Surgical heteroassoc write + L2 decorrelator | Value-codebook Selectivity-Δ at the info ceiling, multi-seed (055-058) | Substrate, Static-codebook | **CLEARED** |
| **Codebook-growth** (P3 structure) | Grow paradigmatic (substitutional) structure | "similar tokens → similar hypervectors"; structure ABSENT in a **valid** control (§"How to know it's working") | Consolidation-write | **OPEN/STUCK** — §5 (flat-code/linear-local family EXHAUSTED) |
| **Replay-consolidation** (P4) | Replay turns settling paths into grooves; interleaving may *manufacture* structure | Replayed trajectories improve retrieval; (CLS) interleaving builds shared abstraction; L2 atoms emerge | Consolidation-write | **PARTIAL** (stabilization only; **structure-generation UNTESTED — the untapped lever**) |
| **Abstraction = THE CEILING** (P5) | Bind-vs-bundle discovery + atom-splitting + analogical retrieval | Bind-vs-bundle **emerges from data**; **analogical retrieval works**; polysemy splits (§"How to know it's working") | Codebook-growth ⇄ Replay (needs paradigmatic structure) | **UNBUILT** (mechanisms absent from `src/`; energy headline PAUSED) |
| **Integration** (P6) | LLM-in-workspace; replay re-encode | SONAR-replacement non-regressive; structural reasoning measurable | Abstraction | **UNBUILT** (correctly downstream) |

*(Gate citations to `experimental-progression.md` are by section header — stable across edits; line numbers shifted once and broke the references.)*

## 5. Where we are — and the current crux

The **Consolidation-write FLOOR** (contextual completion, sense #2) is **cleared**: the surgical
heteroassociative write + L2 decorrelator recalls role→target bindings role-selectively at the
information ceiling, multi-seed, integrated bit-identically (Reports 055-058), scaling settled
(dense H; Report 120).

**THE CRUX (updated 2026-06-01) — Codebook-growth is STUCK against a now-characterized wall, and
Replay is the one untapped lever.** Structure-gate 3b (Report 121) nulled the paradigmatic gate, and
the 121-127 arc then **exhausted the entire flat-code / linear-local growth family**: paradigmatic
(substitutional, king/queen) structure lives in the **SUBDOMINANT modes** of the co-occurrence
operator; it is reachable by **GLOBAL** computation (SVD / NMF / k-means, +0.11-0.25) but by **NO
LOCAL single-projection dynamic over a flat code** (route-invariant across 7 operators; confirmed
*capability*-level by an independent behavioral probe (126); the one nonlinear-partition "escape"
(127) reduces to global k-means). **Every working read is a GLOBAL computation — which the
no-homunculus / local-growth invariant forbids.** That is itself the finding: **the flat code gives
the floor and NOT paradigmatic structure; the latter needs a different lever or substrate, not
another local-growth oracle.**

> **THE LIVE DIRECTION (2026-06-01):**
> 1. **Replay-consolidation is the untapped lever.** The CLS reframe: paradigmatic structure may be
>    *manufactured by interleaved, pattern-separated REPLAY* (a **Codebook-growth ⇄ Replay
>    combination**), not grown by any static operator — never tested, and the cheapest decisive next
>    probe. The interleaving must EMERGE from a local replay-priority dynamic, not a hand-set
>    curriculum — else it is the banned homunculus.
> 2. **Bank-the-bound decision:** a clean null on the replay combination would *close* the
>    flat-code-from-small-text program and point at a substrate/data change — an **Abstraction-node
>    build-gate** decision (the user's to make).
> 3. **Why the §4 reframe happened:** the old linear-phase framing forced the isolation-testing that
>    produced the arc's likely false negatives. The blow-by-blow lives in [STATUS.md](STATUS.md).

## 6. Session-start read order (with this file)

0. **This file (`CONTEXT.md`)** — the thesis, the §4 capability map + gates, the invariants, the crux.
1. [STATUS.md](STATUS.md) — current position (volatile bookmark).
2. The active *capability's* gate/checklist under `notes/emergent-codebook/` (the `phase-<N>-*`
   files are the per-capability design records; see the §4 map for which is which).
3. The §Headline + §Required-controls of the active *capability's* design doc (cite line
   numbers — STATUS banners drift; the design spec + §4 gates above are the truth).
4. `docs/PROJECT_PLAN.md` + relevant dated `notes/notes/`.

If `CONTEXT.md`, `STATUS.md`, and a design spec disagree, that contradiction is a
**finding to surface**, not paper over (it is exactly the failure that produced this
file).
