# TWO-BET RETROSPECTIVE — Neuro-AI (2026-06-06)

*An honest internal reckoning of what the program has actually established, across
both bets and ~139 numbered experiments. It sits **above** and **extends**
[RE-GROUNDING-MAP.md](RE-GROUNDING-MAP.md) (the 2026-05-31 Bet-A Phase-1–5 deep
snapshot): that doc covers Phases 1–5 + the memorization-drift correction; this one
adds the Bet-A codebook-growth **exhaustion arc (121–132)**, the **Bet-A→Bet-B
fork**, the **Bet-B continual arc (133–139)**, and the **two-bet meta-synthesis.***

**How to read this.** A map, not a victory lap. The same skepticism the project
applied to experiments is applied here to the *meta*-claims: findings are tiered
**DURABLE / SUGGESTIVE-but-confounded / UNCONFIRMED**, the contribution is sized
*without inflation*, and the central honesty checkpoint (§4) is that the recurring
"brain-distinctive mechanisms reduce to simple ones" pattern is **confounded with
task selection** — it is as much a statement about the toys as about biology.

---

## 1. The two bets (what each wagered)

- **Bet A** ([CONTEXT.md](../CONTEXT.md)) — *the data-hungry / backprop / autoregressive
  path is the wrong bet; biology IS the answer.* Mechanism must be a local geometric
  dynamic; global computation (SVD/NMF/k-means) is a diagnostic flashlight, never the
  mechanism; backprop banned. No-homunculus, contextual-completion-not-prediction.
- **Bet B** ([CONTEXT-B.md](../CONTEXT-B.md), forked 2026-06-05) — *biology is the
  **spotlight**, not the box.* Backprop / global-as-mechanism / multi-layer **allowed**
  (one-shot closed-form SVD still fenced as a flashlight); same discipline kept. Forked
  because Bet A's mechanism bans kept being re-imported as binding after being lifted.

Both bets share the **discipline** (experiment preamble; one headline; multi-seed +
CIs; control conditions; grep-before-rerun; anti-homunculus = no metric-reading
supervisor; "do it right / no shortcuts").

---

## 2. The arc in brief

- **Phases 1–5 (Bet A), Reports 001–120.** Substrate (FHRR) + Hopfield retrieval +
  masked-token encoding **CLEARED**; the consolidation **write** (heteroassoc + L2
  decorrelator) **GRADUATED as a role-selective associative memory** (055–058,
  byte-identically integrated). The "memorization is the target" **drift** was caught
  and corrected (RE-GROUNDING-MAP, 2026-05-31): in-sample recall is the Phase-3
  **floor**; compositional generalization was always a Phase-5 deliverable.
- **Codebook-growth exhaustion (Bet A), Reports 121–132.** The structure-gate (does the
  codebook grow *paradigmatic*/king~queen structure?) **NULLed** and the whole
  single-layer / local / no-backprop / flat-code growth family was systematically
  **exhausted** (see §3.1).
- **Fork → Bet B (2026-06-05).** King~queen/representation was declared a drift
  (representation is cheap/solved, word2vec 2013); the real LLM critique is
  **frozen-function** (can't learn continually, from little, on low power). Target
  re-grounded to **continual compounding-transfer**.
- **Bet-B continual arc, Reports 133–139.** SGNS (133) redundant; sleep≈replay (134)
  null; pseudo-sleep (135) doesn't scale; baseline compounding = **interleaved replay**
  (138); the "first graduation" (139, Benna-Fusi) **deflated to recombinant EWC+replay**
  under the matched-protection control.

---

## 3. What the program has ESTABLISHED (tiered)

### 3.1 DURABLE (defensible, multi-seed + controls + adversarial verification)

- **The Bet-A local-vs-global bound.** Paradigmatic (substitutional, king~queen)
  structure lives in the **subdominant modes** of the co-occurrence operator. It is
  reachable by **global / nonlinear-partition** computation (SVD +0.109, NMF +0.19,
  k-means/k-WTA +0.25) but by **no local single-projection dynamic over a flat code**.
  Route-invariant across ~7 operators (123 +0.021; 124 directional ≈0; 125 SR/SFA/
  eligibility/order all null; 127 k-WTA reduces to global k-means; 129 fixed-slot null;
  130 phase channel-invariant; 131 capacity necessary-not-sufficient; 132 depth doesn't
  compose). Confirmed **capability-level, not a metric artifact**, by an independent
  behavioral probe (126). *This is the most concrete, citable result of the program.*
- **The Bet-B continual finding (138 + 139).** On the modular-arithmetic continual toy,
  (a) the on-target **compounding transfer is carried by interleaved replay**, not by an
  offline rehearsal pass (138, n=64, log-FTSR); (b) a **soft weight-anchor + replay**
  clears the §8 interaction gate (RC beats both replay-alone and consolidation-alone,
  all 3 g, n=48) — and a **plain frozen-L2 anchor (EWC-lite) reproduces it as well as
  Benna-Fusi** at matched protection (139 + addendum). No brain-distinctive consolidation
  beat this; the graduating mechanism is **recombinant** (replay + soft circuit-anchoring
  + the graded continuation of 136's frozen-circuit reuse).
- **The Phase-3 floor is a real validated asset.** Role-selective associative recall at
  the information ceiling, multi-seed, integrated bit-identically (055–058); the FHRR
  substrate + Hopfield readout + masked-token encoding beneath it (RE-GROUNDING-MAP §5).
- **The methodology / false-positive-mode catalogue (§5).** The single most reproducibly
  valuable output: a worked record of how disciplined controls repeatedly converted
  apparent brain-mechanism wins into "a simple/known method does it."

### 3.2 SUGGESTIVE but CONFOUNDED (the honesty checkpoint — see §4)

- **"Brain-distinctive mechanisms reduce to simple/known ones."** True *across the arc*
  (Bet A: local writers → "global computation does it"; Bet B: consolidation recipes →
  "replay + EWC does it"), **but confounded with task selection.** In every case the task
  was *solvable by the simple method*, so the brain-distinctive mechanism had no
  **necessary** role. The honest, narrower statement: *on the small/solvable toys chosen,
  simple/global/known methods suffice and brain-distinctive mechanisms add nothing
  load-bearing.* This is as much about the toys as about biology (§4).

### 3.3 UNCONFIRMED / open

- **The central thesis** — that a brain-distinctive mechanism manufactures structure or
  transfer a simple/known method **cannot** — was never tested in a **discriminating
  regime** (one where protection / replay / global-factorization demonstrably do *not*
  suffice). It is neither confirmed nor falsified; it is **untested where it matters** (§6).

---

## 4. The honesty checkpoint — what we did NOT establish

Three guardrails against over-reading the meta-result:

1. **Task-selection confound (the big one).** Both bets chose tasks whose structure was
   *reachable/protectable by simple means*: Bet A's paradigmatic structure is in the data
   and recoverable by SVD; Bet B's shared "+/−" circuit is reusable and protectable by
   EWC+replay. When the simple method already saturates the task, no mechanism can show a
   *necessary* advantage — so "the brain mechanism added nothing" is **expected by
   construction**, not a verdict on the mechanism. The program never built the regime
   where the simple method *fails* (§6). This is the Bet-B echo of the Bet-A locality
   trap, now generalized.
2. **Scope of "biology."** We tested a finite set of *specific operationalizations*
   (FHRR local writers, Benna-Fusi, surprise-replay, pseudo-rehearsal, two-timescale
   freeze, …). Finding that these reduce to simple methods on these toys is **not**
   "biology-inspired ML doesn't work." It is "these mechanisms, on these tasks, were not
   load-bearing."
3. **Negative results on toys are weak evidence about scale.** Everything is
   modular-arithmetic / small-WikiText / D≤4096, n≤64. None of it speaks to whether the
   mechanisms matter at scale or on real continual streams.

---

## 5. The false-positive-mode catalogue (the most reproducibly valuable output)

The discipline caught, with a control or re-derivation, *at least* these — each a real
worked example where an apparent positive was deflated:

- **Single-seed false-PASS** → multi-seed + within-set label-shuffle (126).
- **Para-set hubness ≠ pair-specific** → gauge-free B-KILL arbiter (126).
- **Competent-control-reproduces-the-fancy-mechanism** → k-WTA reduces to global k-means
  (127); **Benna-Fusi reduces to a plain frozen-L2 EWC anchor (139 addendum)**.
- **n=2 smoke false-GRADUATES** → n=8/48 + per-k drill-down (137).
- **Metric traps** → glob-double-subtraction, contraction-artifact, density-INVALID,
  low-dim cosine inflation (121–124).
- **Heavy-tailed ratio metric** → log-FTSR instead of raw FTSR (138).
- **Under-powered/over-reaching go/no-go** → design-audit-before-trusting-numbers caught
  an "abandon the domain" rule firing on the low-power branch (138, `ws9bfc18f`).
- **Additive mistaken for super-additive** → `d_super` separated from the "beats both
  parents" gate (139 verify, `wuybsd91u`).
- **Confound resolved the wrong way** → the m=2-collapse looked like "depth is needed"
  but was a bidirectional artifact; the frozen-anchor control overturned it (139 addendum).

The throughline: **adversarial verification + competent controls before banking** flipped
several "first real win" moments into "a simpler/known method does it." That engine —
*not* any single mechanism — is the program's most transferable product.

---

## 6. What would actually test the central thesis (so a future program can resume)

The thesis can only be tested where **the simple method fails**. Concretely:

- **A discriminating (compositional) regime.** A continual task where forward transfer
  requires **recombining learned primitives**, not reusing/protecting one shared circuit
  — so replay + a soft anchor demonstrably do *not* saturate it. (SCAN/COGS-style
  compositional splits; or a modular task where T3 = T1∘T2 must be *composed*.) Then ask
  whether a genuinely **restructuring** consolidation (one that does more than
  re-presentation or protection) beats replay+anchor where they fail.
- **The Bet-A analogue:** a substrate/representation where the paradigmatic structure is
  *not* in the dominant modes a global SVD trivially reaches — i.e. where the "flashlight"
  itself fails — so a learned/iterative local mechanism could show a necessary advantage.

Until such a regime exists, the central thesis stays UNCONFIRMED, and further mechanism
recipes on the current toys will keep reducing to simple methods (the §4 confound).

---

## 7. Honest contribution inventory (sized, not inflated)

- **Most concrete:** the **Bet-A local-vs-global bound** — a clean negative with a precise
  mechanism (subdominant modes; route-invariant; capability-level). Could anchor a focused
  write-up.
- **Most transferable:** the **discipline + false-positive catalogue (§5)** — genuinely
  useful, but lives in a crowded "be rigorous" space; its value is the *concrete worked
  examples*, not the exhortation.
- **Real but weak standalone:** the **Bet-B negative** ("on a solvable continual toy,
  replay + a plain weight anchor suffices; brain-distinctive consolidation adds nothing").
- **Validated assets** (not contributions per se, but real): the graduated completion-write
  + decorrelator, the FHRR/Hopfield substrate, the bundle-first scene-memory primitive.

Not a breakthrough; not nothing. The program's honest output is **one durable bound, one
reusable discipline, a set of validated components, and a sharp map of where the central
thesis remains untested.**

---

## 8. Pointers

- Bet-A Phase-1–5 deep snapshot + the memorization-drift correction:
  [RE-GROUNDING-MAP.md](RE-GROUNDING-MAP.md).
- Bet-A 64-null reclassification under the §4 DAG:
  [null-audit-coupled.md](emergent-codebook/null-audit-coupled.md).
- Charters: [CONTEXT.md](../CONTEXT.md) (Bet A), [CONTEXT-B.md](../CONTEXT-B.md) (Bet B).
- Bet-B continual arc detail: Reports 133–139 (esp. [138](../reports/138_betb_baseline_decomposition/report.md),
  [139](../reports/139_betb_replay_x_consolidation_bennafusi/report.md) + its EWC addendum).
- Current bookmark: [STATUS.md](../STATUS.md).
