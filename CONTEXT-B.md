# CONTEXT-B — The Spotlight Bet (Bet B)

*The binding charter for the **Bet-B** line of work, forked 2026-06-05. Read
this FIRST for any Bet-B session. The original charter [CONTEXT.md](CONTEXT.md)
is **Bet A** — a different wager, not falsified, kept intact as its own program.
Where the two disagree on **mechanism rules**, CONTEXT-B governs Bet-B work.
Where they agree on **discipline**, both are binding.*

> **Why this file exists.** Over one session (2026-06-05) the assistant
> re-imported Bet A's mechanism bans (no-backprop, no-global, local-only,
> "the fence") **three times** as if binding, even after they were explicitly
> lifted — because those bans are written as line-numbered imperatives in
> CONTEXT.md/CLAUDE.md and are read as law by every agent and every grounding
> pass. You cannot fix a context-gravity problem with willpower; you fix it by
> changing what is written. This file is that fix: it states Bet B's rules as
> the binding ones so they win on every retrieval.

---

## 1. The two bets, side by side

- **Bet A (CONTEXT.md):** *the data-hungry / backprop / autoregressive path is
  the wrong bet; biology/neuroscience IS the answer.* Every mechanism must be a
  local geometric dynamic; global computation (SVD/NMF/k-means) is a *diagnostic
  flashlight, never the mechanism*; backprop is banned. **This bet produced a
  rigorous, real result:** across 130+ controlled experiments it mapped a true
  local-vs-global bound (the single-layer / local / no-backprop writer family is
  genuinely exhausted — see §4). That record stands. We are **not** falsifying
  Bet A; we are forking a different wager alongside it.

- **Bet B (this file):** *biology is the **spotlight**, not the **box**.* Biology
  tells us **what to build** (the target capability) and **what to aim at**
  (contextual completion, consolidation, "more than memory"). It gets **no vote
  on the mechanism.** Use whatever mechanism best reaches the target — held to
  the **same discipline** as Bet A.

## 2. The lifted rules (precise) — and what is KEPT

**Lifted for Bet B:**
- **Backprop is allowed** as a learning rule (not a homunculus — it's gradient
  descent on a loss, not a supervisor reading a metric and branching).
- **Global / non-local computation is allowed AS A MECHANISM**, not merely as a
  diagnostic flashlight — *with one exception* (next bullet).
- **Local-only / single-projection / single-layer is no longer required.**
- **The substrate is open** — need not be FHRR+Hopfield; a different EBM,
  encoder, or world-model substrate is on the table.

**Kept (binding for Bet B, no exceptions):**
- **"Do it right — no shortcuts."** The one mechanism still fenced as a
  *flashlight*: a **one-shot closed-form SVD / eigendecomposition**. It stays a
  *reference/ceiling*, never the mechanism — because handing the system a
  precomputed operator and calling `torch.svd` is the easy way out. The legal
  global mechanisms are **iterative / learned** (backprop, multiplicative-update
  NMF, Lloyd k-means, EM, streamed gradient) — not closed-form factorization of
  a materialized operator.
- **The anti-homunculus filter, in its TRUE form only:** no supervisor module
  that reads a metric and triggers a response; no `if X then do Y` arbitration.
  This is KEPT. It is **not** a ban on global or backprop learning rules.
- **The full experiment discipline** (identical to Bet A): the experiment
  preamble; ONE headline gate (headline-vs-drill-down); multi-seed with CIs;
  control conditions; **gauge-free pair-specific B-KILL** (within-paradigmatic-set
  label-shuffle) as the hubness-immune arbiter; the named metric traps
  (glob-double-subtraction, contraction-artifact, density-INVALID, low-dim cosine
  inflation, incompetent-control, para-set-hubness); **grep `reports/` before
  re-running any mechanism by name.**

## 3. The goal (UNCHANGED) and the architecture sketch

The goal is the same as it always was: a continually-learning **contextual-
completion** system whose durable identity is a learned landscape — and whose
"**more than memory**" is **paradigmatic / structural generalization** (the
king~queen substitutability the codebook never reached). Memorization is the
**floor**; structure is the prize.

The Bet-B architecture sketch (a hypothesis to test, not a result):
- **One allocation currency: learning PROGRESS, not raw prediction error.**
  Progress (the *change* in competence) is zero on noise, zero on the mastered,
  peaks on the learnable frontier. It is the **allocation** signal — what to
  write, what to replay, what to seek — *not* the weight-update (that stays
  error-driven). **It is direction-blind** (a scalar can't pick *which*
  structure to build), so it allocates; it does **not** manufacture.
- **The structure-MANUFACTURER is an offline, iterative-global consolidation
  pass** (the "sleep" step), now a legal mechanism. A fast online episodic store
  (the floor) feeds a slow offline consolidator that distills replayed episodes
  into paradigmatic structure.

## 4. The honest current state (cleaned of dead-rule contamination)

*Audited 2026-06-05, guarding against both over-negative (rule-bound) and
over-positive (over-correction) bias.*

- **TRUE and durable (rule-independent):** the **single-layer / local /
  single-projection / no-backprop writer family is exhausted** — every such
  writer NULLs on pair-specific paradigmatic structure (Reports 123 +0.021, 124
  directional ≈0, 125 B/C/D clean nulls, 127 `grow_G` +0.002, 129 frozen-slot
  B-KILL ≤4/10). Replicated across seven operators. **Do not re-run this family.**
- **TRUE and now a DESIGN SPEC (not a dead end):** every *working* read of
  paradigmatic structure is **global or nonlinear-partition** (SVD +0.109, NMF
  +0.19, k-means/k-WTA +0.25); the structure lives in **subdominant modes** local
  power-iteration can't reach but an iterative-global pass can. Under Bet B this
  reads as *"include an iterative-global / joint-assignment consolidation step,"*
  **not** *"impossible."*
- **Oracle-E is a FLASHLIGHT, not a writer.** Global NMF of the transition/
  successor operator recovers king~queen on **passive WikiText** (+0.124/0.176/
  0.188; label-shuffle + random-nonneg controls REAL; SVD-300 ceiling +0.183).
  This proves the **signal exists and is globally recoverable from passive
  text** — it does **not** prove a learned *writer* reaches it, and it does
  **not** privilege "predict-same-future" (the direct SR/SFA lead, Oracle B,
  nulled locally). Magnitudes are low-dim-inflated; **no "beats SVD" claim.**
- **THE BIGGEST CLEANED CORRECTION — embodiment is NOT structurally necessary.**
  Because the signal is recoverable from passive text by a global/iterative
  mechanism, there are **at least two live escapes**, not one:
  - **Door A (in-hand, cheap, UNTESTED-under-new-rules):** a passive-text
    **offline iterative-global / backprop joint-assignment writer** — exactly the
    TEM-style route Report 129 was *forced to abandon* ("global slot-ASSIGNMENT is
    load-bearing; TEM does it via backprop, banned"). Now buildable.
  - **Door B (optional, heavier):** richer grounding / **cross-view embodiment**
    — a *data upgrade*, no longer a structural necessity.
- **DOOR A TESTED → REDUNDANT-BUT-VALID** (Report 133, `exp79`, 2026-06-05). A
  streamed SGNS gradient writer (operator never formed, no SVD) reaches
  pair-specific paradigmatic structure (spec +0.107, **B-KILL 10/10**) where
  single-projection local writers null (+0.0002) — **but MATCHES, doesn't beat,
  the SVD reference** (−0.0018, weaker B-KILL) = **Levy-Goldberg 2014**. NOT a
  graduation (the auto-PASS over-claim was caught by the 3-lens — the documented
  over-correction mode, inverted). **Attribution 2×2: the lever is the MECHANISM
  CLASS (joint/global optimization vs single-projection), NOT input and NOT
  backprop** (NMF isn't backprop) — single-projection nulls on both 1st/2nd-order;
  joint-opt works on both (NMF +0.164, SGNS +0.107). So the 121–132 bound is
  **re-confirmed from the legal side, not overturned**; "embodiment not necessary"
  was already 125's. The genuinely-novel question is untouched: **Stage 2**.

## 5. The next experiment — gated two-stage (the load-bearing guard is anti-redundancy)

**The dominant risk is the redundancy trap:** "train a net to factorize the
materialized operator" lands on the NMF/SVD subspace Oracle-E already
characterized — a near-certain +0.18 that **teaches nothing**, dressed up as
"consolidation." That is this project's documented false-positive failure mode
(126, 127) in the inverted direction. Guard against it explicitly.

- **STAGE 1 (cheap, make-or-break, ~`exp76` scale — run FIRST):** an iterated /
  backprop **predict-the-future** writer.
  - **Loss = streamed predict-the-future over windows** (predict held-out
    successor/context tokens). **The operator is NEVER materialized; NO
    closed-form SVD** (keeps "do it right"). This is what makes it a *mechanism*
    question, not a re-factoring of Oracle-E.
  - **Headline = pair-specific B-KILL** (within-para label-shuffle; CI-lo > 0,
    ≥8/10 seeds). **Drop g3 entirely** (dead at n=40 — the SVD anchor fails it
    too).
  - **Floor:** the nulled local writers (`grow_G` +0.002; 129 frozen-slot).
    **Reference (no longer a fence — a mechanism you must BEAT-OR-DIFFER-FROM):**
    closed-form NMF + SVD-300 of the **same operator at matched k**, reported
    side-by-side — the **"you-are-just-NMF" control.** Novelty = a measured
    **delta** (better B-KILL, or reachability from streamed windows the
    closed-form route needed the full operator for), never a re-hit of +0.18.
  - **Anti-inflation:** random-init / no-learning control; report B-KILL, never
    absolute cosine.
  - **GATE:** graduates only if it **beats the nulled floor on B-KILL AND shows a
    defensible delta from the closed-form reference.** If it merely re-hits NMF's
    +0.18 with NMF-level B-KILL, **bank it as REDUNDANT and stop** (the honest
    Oracle-E-reproduction outcome).
- **STAGE 2 (only if Stage 1 graduates) — the genuinely NEW part:** the
  two-timescale consolidation claim. Fast online episodic store + slow offline
  consolidator. **Headline is NOT "does structure appear"** (Stage 1 settled
  that) but the **consolidation-specific payoff vs a single-timescale ablation:**
  does interleaving the slow consolidator over replayed episodes manufacture
  pair-specific structure the fast store alone never accumulates, and/or resist
  catastrophic forgetting an online-only encoder suffers? Anti-homunculus: a
  fixed fast/slow split with a smooth local replay-priority is a *dynamic*, not a
  supervisor — legal; replay-priority must stay a smooth function of codes (no
  "if king/queen then boost").

## 6. Honest novelty caveat

Most of this is recombination: world-models, surprise-gated memory, prioritized
replay, active inference, and curiosity each exist; a 4-of-5 unified-currency
agent (AXIOM, 2025) already beats DreamerV3. The genuinely-undone piece is
**Stage 2** — making offline consolidation of a surprise/progress-prioritized
episodic store be the **structure-manufacturer** for the slow model. Build only
the loop and reviewers correctly call it a recombination; the novelty lives or
dies on whether **consolidation manufactures structure the components can't** —
which is exactly the wall Bet A mapped, now to be re-asked under the lifted rules.

## 7. Read order (Bet-B session)

0. **This file (`CONTEXT-B.md`)** — the bet, the lifted rules, the kept
   discipline, the cleaned state, the gated experiment.
1. [CONTEXT.md](CONTEXT.md) §4 — Bet A's capability map (the shared target +
   the durable bound), read as *history/reference*, not as binding mechanism law.
2. The relevant numbered reports (121–132) for anchors — grep by mechanism name
   before re-running anything.

If this file and CONTEXT.md disagree on a **mechanism rule**, CONTEXT-B wins for
Bet-B work and the disagreement is expected (two bets). If they disagree on
**discipline**, that is a finding to surface.

## 8. THE RE-GROUNDED TARGET (2026-06-05) — continual compounding-transfer

*The §3 goal, re-grounded. The whole 121–133 representation arc (king~queen /
"can a method represent paradigmatic structure") was a **drift**: representation
is cheap and solved (word2vec, 2013). It was never the actual critique of LLMs.*

**The actual spotlight (the user's, restored):** an LLM is a **frozen function** —
it learns once, then can't learn a new thing without retraining, forgets nothing
new after training, needs the whole internet, and burns a power plant to do what a
child does on oatmeal. All four are symptoms of "frozen." The brain is the
opposite: **never frozen** — it learns continually, from little, on ~20W, and
(the deep part) **abstracts**: it takes a rule and applies it to the genuinely
novel. The hypothesis tying it together: *abstraction is not a module you add — it
is what a continual, sparse learner is **forced** to grow.*

**Operationalize, don't mystify (the vitalism rule):** do NOT target "thinking" /
"understanding" / consciousness — unmeasurable, unfalsifiable, irrelevant to the
build. Target the **behavior**: learn → retain → abstract → transfer. Whatever
passes the behavioral test, passes — no "but does it *really* understand."

**THE TRACER BULLET (the smallest thing with the essential property).** A single
small model, on a laptop, that learns 3 tasks in sequence and **(1) retains** them
(no catastrophic forgetting), **(2) does them genuinely** (model produces answers
on *held-out* inputs; no external solver), **(3) compounds** — each task learned
*faster* than the last via shared structure (positive forward transfer).

*The grounding sharpened the open niche (Report 134 spec):* retention is
solved (a replay buffer); positive forward transfer is **usually zero/negative**
in the literature, and the cases that work (joint training; embedding transfer)
are published. **The unclaimed, genuinely-open niche:** does an **emergent local
consolidation ("sleep") dynamic manufacture positive forward transfer on a single
sequential model, beating a plain replay buffer at equal retention, where the
transfer is provably structural (not a confound)?**

- **Tasks (modular arithmetic, for verifiability + a scramble control):**
  T1 `a+b mod p`; T2 `a−b mod p` (shares the number-circle); T3 `a+b mod p` over a
  **disjoint symbol alphabet** (shares structure, not surface tokens); **T3′
  scrambled** (no shared structure — speedup MUST vanish here).
- **Headline (one, falsifiable):** **Forward-Transfer Speedup Ratio** FTSR_k =
  (from-scratch steps-to-95%-held-out on task k) / (sequential-with-sleep
  steps-to-criterion on task k). **PASS = compounding (FTSR_3 > FTSR_2 > 1.0,
  CI-disjoint from 1) AND FTSR strictly beats the replay-only arm (CI-disjoint)
  AND T1+T2 retention ≥ 90% at end-of-stream.** Multi-seed (≥8), bootstrap CIs.
- **Controls (all mandatory):** (1) from-scratch per task = the denominator;
  (2) joint-train-all-3 = ceiling, our sequential final-acc must match it;
  (3) **plain replay-only (no sleep restructuring)** = isolates consolidation as
  the manufacturer — the load-bearing control; (4) **scrambled T3′** = speedup
  must die; (5) frozen-model-in-context on T3 = guards the "frozen model already
  does it few-shot" escape.
- **Anti-homunculus (KEPT):** what gets replayed is set by a **local surprise
  signal** (prediction-error weighting, Report 128), NOT a supervisor picking
  tasks. The slow/fast split + replay schedule is a fixed dynamic, legal.
- **The consolidation mechanism is a SWAPPABLE part.** If surprise-replay NULLs
  on the FTSR-beats-replay-only headline, that is **iterate-fuel, not a dead end**
  (user-binding) — swap in the next "sleep" recipe and re-run the same harness.
  A clean null here is a few hours, not a year.

*Honest bar:* the field mostly gets zero forward transfer; the likeliest single
outcome is sleep does NOT beat the plain buffer (null). That is fine — it is
falsifiable, laptop-sized, and the first real test of "experience compounds into
transferable skill," which is what this project was always actually about.

**RESULT — Report 134 (`exp80`, n=8): PARTIAL.** The harness works; forward
transfer is **REAL, strong, robust, retained** (sub groks ~18× faster after add;
matches the joint ceiling). BUT: (a) that transfer is a **KNOWN result**
(grokking-transfer lit ~5×) — redundant, like the SGNS run; (b) the distinctive
bet — **emergent "sleep" manufacturing transfer beyond a plain replay buffer —
NULLS** (sleep ≈ replay, deltas straddle 0; the strong transfer is carried by
replay + the shared representation, not the surprise-replay recipe); (c) the
scramble control is **invalid** (relabels an isomorphic addition → doesn't break
the operation-transfer; redesign before trusting T3 structurality; T2's 18× is
structurally unambiguous regardless). **ITERATE (per the user-binding note):** the
swappable sleep recipe nulled in v1 → try (1) a **harder regime where plain replay
is insufficient** (long streams / tight memory / negative-transfer tasks — sleep
can only win where replay doesn't already saturate), (2) a **restructuring/
abstracting** consolidation (not rehearsal), (3) a **redesigned scramble** that
breaks operation-transfer. The harness is ready for all three.

**ITERATED → FIRST POSITIVE (Report 134 ADDENDUM, 2026-06-06).** Hunted (1): starve the buffer
(cap=2/task) → plain replay **BREAKS** (forgets old tasks to chance ~0.05; even shows negative transfer).
**Self-generated "pseudo-sleep"** — the model regenerates its past from its own weights and rehearses
that, storing ZERO raw examples (pseudo-rehearsal, Robins 1995) — **beats raw replay on retention
(+0.45 [0.39,0.53]) AND transfer (FTSR 11.7 vs 3.8), n=8.** The **first** time this session a
brain-shaped mechanism beats the simple baseline — **but a KNOWN mechanism, partial (~0.48), tiny toy.**
The validated strategy: **test the brain mechanism where the baseline FAILS, not where it already wins.**
Next: push pseudo→full retention (interleave generation during learning), longer streams
(loss-of-plasticity), redesign scramble, then a harder domain.
