# PRE-COMMIT — Bet B: the COMPOSITIONAL discriminating regime (retrospective §6)

*Status: DRAFT for user review (2026-06-15). Extends [CONTEXT-B.md §8](../CONTEXT-B.md)
with a NEW headline regime; once signed off, CONTEXT-B §8 gets a pointer here.*

---

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** Bet-B continual **compounding-transfer**
> ([CONTEXT-B.md §8](../CONTEXT-B.md)), in the **discriminating / compositional
> regime** named by [RETROSPECTIVE-two-bets-2026-06-06.md §6:156-167](RETROSPECTIVE-two-bets-2026-06-06.md).
> This is a *combination*: continual-transfer × **composition** (forward transfer
> that requires **recombining** learned primitives, not protecting one shared circuit).
>
> **Headline metric (Step 2 graduation) per [this doc §4](#4-step-2--the-graduation-headline-only-if-step-1-validates):**
> the §8 interaction gate (RC beats both replay-only and consolidation-only, CI-disjoint
> log-FTSR) **on the COMPOSITION tasks**, in a regime Step 1 has *proven* replay+anchor
> do not already saturate.
>
> **Headline metric (Step 1 = THIS run, a VALIDITY DIAGNOSTIC, explicitly NOT a graduation)
> per [§3](#3-step-1--the-validity-gap-diagnostic-run-first-make-or-break):** the
> **composition gap** — under the simple method (replay + EWC-lite soft anchor),
> `FTSR_primitive > 1` (the parts transfer) **AND** `FTSR_composition ≈ 1` (the
> recombination does not), with an oracle confirming the composition is reachable
> (headroom is real, not impossibility).
>
> **Required controls per [§2](#2-arms--controls):** scratch denominator; floor
> (no replay/no consol); replay-only; replay+EWC-lite (the simple method that must
> leave a gap); joint-train ceiling; frozen-MLP composability oracle (136-style);
> redesigned scramble T3′ (difficulty-matched).
>
> **Last verified result:** [Report 139](../reports/139_betb_replay_x_consolidation_bennafusi/report.md)
> — on the *non*-discriminating regime, replay + soft weight-anchor cleared the §8
> gate, but a plain EWC-lite anchor reproduced it (recombinant; no brain-distinctive
> ingredient). The retrospective §4 diagnosed the cause: **task selection** — every
> task reused one circuit, so protection saturated.
>
> **Why this experiment now:** it is the *one* move (retrospective §6) that could move
> the central thesis from UNCONFIRMED to tested — build a regime where the simple method
> demonstrably **fails**, then ask whether a restructuring consolidation succeeds where
> it fails. Step 1 is the cheap make-or-break: if replay+EWC already composes, the regime
> is not discriminating and we re-pick primitives (hours, not weeks).

**Why Step 1 is a "drill-down" not a "graduation" (preamble rule):** the Step-1
headline (the composition gap) is NOT the §8 graduation metric — it is the
*precondition* that licenses Step 2, exactly as [Report 136](../reports/136_betb_transfer_gap_diag/report.md)'s
gap-diagnostic preceded the 137 mechanism attempt. Labeling it as a validity
diagnostic is mandatory and done here.

---

## 1. Anti-homunculus / discipline (must pass before building)

- **The regime is data, not a supervisor.** The compositional stream is a fixed
  task schedule; no module reads a metric and routes. ✅
- **The simple-method arms are unchanged** from exp85 (replay = content-blind
  interleaved re-presentation; EWC-lite = a fixed local L2 pull toward a frozen
  snapshot). No `if task==C then ...`. ✅
- **Step-2 consolidation (later) must stay a fixed local loss** — deferred; not built
  in Step 1. The Step-1 run contains **no** novel mechanism, only the regime + the
  known simple-method arms + oracles. ✅
- **Fence (Bet B):** no one-shot closed-form SVD/eig as a *mechanism*. Step 1 has no
  mechanism at all; the joint-train ceiling is iterative SGD (legal). ✅

---

## 2. Arms & controls

Tasks are modular arithmetic over `Z_p` (p=17), tokens offset by a per-alphabet
`base` (so a new alphabet = disjoint tokens, the established cross-block transfer
probe). **Three task TYPES per alphabet block:**

| type | definition | role |
|------|------------|------|
| **A** (primitive 1) | `A(a,b) = (a + b) mod p` | learn the additive circuit |
| **B** (primitive 2) | `B(a,b) = (a · b) mod p` | learn a *genuinely different* (nonlinear) circuit |
| **C** (composition) | `C(a,b) = B(A(a,b), b) = ((a+b)·b) mod p` | **chains A then B** — the recombination |

The target `C` is a deterministic `Z_p×Z_p → Z_p` table; the *composition* is not in
the target definition but in **whether the model reuses its learned A and B to learn
C faster**. At least one primitive (B) is **nonlinear**, so `C = B∘A` is not
collapsible to a single linear map (a linear-only C would transfer trivially → no
discriminating gap; see §6 ladder).

**Stream:** `n_alph` alphabet blocks, each emitting `[A_j, B_j, C_j]` in order
(j = block index). The model sees A and B before C within every block, so the
primitives are always *available* when C is presented — isolating "can it
**compose** them" from "are the parts present."

**Headline task-set split:**
- **x-block COMPOSITION tasks** = `C_j` for `j ≥ 1` (after ≥1 full block of
  primitive experience). FTSR here is the graduation headline (Step 2) and the gap
  metric (Step 1).
- **x-block PRIMITIVE tasks** = `A_j, B_j` for `j ≥ 1`. FTSR here is the **contrast**:
  the parts *should* transfer under the simple method.

**Arms (Step 1):**
- `scratch` — fresh model per task (the FTSR denominator).
- `floor` — sequential, no replay, no consolidation (= exp85 anchor).
- `replay_only` — interleaved replay (= exp85 anchor ~6.3 on the old regime).
- `replay_plus_ewc` — replay + EWC-lite soft anchor on the shared MLP (**the simple
  method that must leave a composition gap**; consol_mode=`ewc` in exp85).
- `joint` (ceiling) — train all blocks jointly; C must grok to ceiling fast (proves C
  is learnable and reachable with simultaneous structure access).
- `frozen_oracle` (composability, 136-style) — after the stream learns A,B (and early
  C), **freeze the shared MLP**, learn `C_j` on a fresh alphabet training only the
  head + embeddings. High FTSR_C ⇒ the composition is *latent in the circuit* (real
  headroom the simple method fails to assemble); ≈1 ⇒ C needs genuinely new circuitry.

**Scramble control (redesigned — Report 134's was invalid):**
- `C′` = composition of **fresh random primitives the model never learned**:
  `C'(a,b) = B_rand(A_rand(a,b), b)` with `A_rand, B_rand` fixed random `Z_p` tables
  of **matched output entropy** to A,B. Same compositional *form*, but the inner/outer
  pieces are not the learned ones → any FTSR_C lift from prior A,B knowledge **must
  vanish**. Difficulty-match is itself a Step-1 check: `scratch` steps on `C` vs `C′`
  must be comparable (else the scramble confounds difficulty, the 134 failure).

---

## 3. Step 1 — the VALIDITY gap-diagnostic (run FIRST, make-or-break)

Multi-seed (≥8), log-FTSR bootstrap CIs (heavy-tailed, per Report 138).

**The regime is DISCRIMINATING (proceed to Step 2) iff ALL hold:**
1. **Parts transfer:** under `replay_plus_ewc`, `FTSR(A_j) > 1` and `FTSR(B_j) > 1`
   (CI-lo > 1). Protection works on the primitives.
2. **Composition does NOT transfer (the gap):** under `replay_plus_ewc`,
   `FTSR(C_j) ≈ 1` — CI **includes 1 / not CI-lo > 1**; ideally CI-hi not far above 1.
   The recombination is the hard part the simple method cannot assemble.
3. **The gap is real headroom, not impossibility:** `frozen_oracle` and/or `joint`
   show `FTSR(C_j) ≫ 1` (composition *is* reachable from the learned structure).
4. **Scramble is valid:** `scratch` difficulty on `C ≈ C′`, and any FTSR_C lift dies
   on `C′`.

**If the regime is NOT discriminating** (most likely failure: condition 2 fails —
replay+EWC already composes C, FTSR_C > 1): **bank it and re-pick primitives up the
§6 ladder.** This is iterate-fuel, not a dead end (user-binding note). A clean null
here is a few hours.

**Disposition (decisive either way):**
- All 4 hold → **discriminating regime established**; Step 2 (restructuring
  consolidation) is licensed and the §8 headline finally has teeth.
- Condition 2 fails → the simple method composes; **the central thesis takes another
  confirmation that simple methods suffice on solvable toys** (retrospective §4) — and
  we climb the ladder to a harder composition. Either way it is a *result*, written up.
- Condition 1 or 3 fails → the task is mis-built (parts don't transfer, or C
  unreachable); fix the harness, not the science.

---

## 4. Step 2 — the graduation headline (ONLY if Step 1 validates)

Unchanged from [CONTEXT-B.md §8](../CONTEXT-B.md) / exp85, but on the **C tasks** in
the now-validated discriminating regime:

- **Headline:** the interaction gate — a **restructuring** consolidation (a fixed
  local non-reconstruction objective that aligns/factorizes the A,B representations so
  they *compose*) + replay beats **both** replay-only and consolidation-only on
  `logFTSR(C_j)`, CI-disjoint; RC compounds (FTSR_C grows across blocks); retention held.
- **The bar Step 1 sets:** replay+EWC leaves `FTSR_C ≈ 1`. A restructuring
  consolidation **graduates** iff it lifts `FTSR_C` above that floor (CI-lo > the
  simple-method arm) — i.e. it manufactures composition transfer protection cannot.
- **Recipe is swappable** (user-binding): if recipe-1 nulls on the interaction, swap
  the next and re-run the same harness. Candidate recipes (deferred to Step-2 design):
  a factorizing/decorrelating offline pass, GERM, a slow alignment of A's output
  subspace to B's input subspace. Each must pass anti-homunculus review before build.

---

## 5. Build checklist (`experiments/86_betb_compositional_regime.py`)

- [x] Reuse exp83 (`ContinualNet`, `train_task`, `evaluate`, `to_tensors`, `boot_ci`)
      and exp85 (`EWCAnchor`, `train_task_bf`) — **no model change** (nested-binary).
- [x] `make_task_composed` (the C target) + `build_compositional_stream` (per-block
      `[A,B,C]`, rotating alphabets, vocab sized for `n_alph`).
- [x] Random-primitive scramble `C′` with matched-entropy random tables.
- [x] Arms: scratch / floor / replay_only / replay_plus_ewc / joint / frozen_oracle.
- [x] Report FTSR split by type (A, B, C) per arm; the 4 validity conditions; log-FTSR
      bootstrap CIs; per-block FTSR_C vs j (compounding drill-down).
- [x] `--smoke` (n=2, small) for mechanical validation before the multi-seed run.
- [ ] **Multi-seed validity run (≥8 seeds)** — the make-or-break; gated on user sign-off
      of this doc. Triggers the experiment preamble + the §3 disposition.

---

## 6. Fallback ladder (if Step-1 condition 2 fails — replay+EWC already composes)

Climb primitive distinctness / composition depth until the gap appears or the ladder
is exhausted (an exhaustion is itself the retrospective-§4 confirmation):

1. **Primary:** A=add, B=mult, C=B(A(a,b),b). (this doc)
2. More distinct B: a fixed nonlinear permutation, or `B(a,b)=(a+b²) mod p`.
3. **Deeper chain:** C = a 3-step nesting (needs a 3rd primitive) — more composition to
   assemble, less reachable by a flat protect.
4. Unary `g∘f` function composition (model-signature change) — the canonical
   compositional-generalization setup, held in reserve.

If the gap never appears across the ladder → **the simple method composes everything
these toys can express**; the discriminating regime requires a domain change (out of
modular arithmetic), and *that* is the honest finding to report.

---

## 7. Open questions to resolve before freeze

- **mult-mod-p grokking:** can scratch reliably grok `B` and `C` within `max_steps`?
  If scratch hits the cap, FTSR is censored/noisy → tune `max_steps` or drop to ladder
  rung 2 (a more reliably-grokkable nonlinear B). (Step-1 smoke checks this.)
- **block count vs compute:** `n_alph` blocks × 3 types × 6 arms × ≥8 seeds. Pick the
  smallest `n_alph` that gives ≥3 x-block C measurements for a compounding CI (n_alph≈4
  → C_1,C_2,C_3). Shard if needed (exp85 has a merge path).
- **scramble difficulty-match tolerance:** how close must `scratch(C) ≈ scratch(C′)` be?
  Pre-register a band (e.g. within 1.5×) before trusting the scramble.

---

## 8. Run log — de-risking probes (2026-06-15, n=1, BEFORE the multi-seed run)

Pre-flight grokkability probes on the flat `ContinualNet` MLP (p=17, wd=1.0, 8–30k
steps). **Caught a mis-built regime in ~10 min, exactly what Step-1-first is for.**

- **wd matters:** at `weight_decay=0.1` *nothing* groks; at `1.0` the symmetric
  polynomials grok. The harness default (1.0) is correct — keep it.
- **The original `C=((i+j)·j)` is PATHOLOGICAL:** CENSORED at 30k from scratch,
  primed, AND frozen (acc ≈ chance 0.069). No headroom → validity condition 3 would
  fail. **Replaced C with `A+B = (i+j+ij)`** (groks ~4k, uses both primitives).
- **Primitive interference is REAL:** `B=mult` groks at 4k *alone*, but was CENSORED
  when learned right after `A=add` with A-replay interleaved. Add and mult fight over
  the single shared MLP. → the pilot must check A,B even **coexist** (retention) before
  the composition question is meaningful; if they don't, that interference is itself a
  candidate discriminating lever (a factorizing consolidation could separate them where
  protection can't).
- **Grokkability sweep (the strategic caveat):** mod-p compositions in this flat MLP
  look **bimodal** — grokked-directly-from-scratch (`add` 5.7k, `mult` 8k, `(i+j)²` 3k,
  `i²+j` 9.5k, `i+j+ij` 4k, `i+j−ij` 9.75k, `2(i+j)+3ij` 5.5k) **or never**
  (`(i+j)·j`, `(i+j)·ij` deg-3). **No "Goldilocks" composition observed that is
  learnable ONLY by composing the primitives.** If the pilot shows C transfers no
  differently than a same-difficulty non-composition, that bimodality is the finding:
  a flat MLP either fits the whole target directly (no composition needed) or can't
  represent it — so the discriminating gap may require a model where *composing learned
  primitives* is an architecturally distinct pathway from *direct fitting* (§6 rung 3–4,
  or a domain change). This is a substrate result, surfaced honestly, not a failure.

**Pilot (n_alph=3, seeds=2) launched** to read the actual FTSR_A/B/C pattern + A,B
coexistence on the fixed regime before committing to the ≥8-seed validity run.

### 8.1 Pilot result + confound check → the regime is NULL; the SUBSTRATE is too brittle (2026-06-15)

**Pilot (n_alph=3, n=2, fixed C=i+j+ij), replay+EWC x-block FTSR:**
`A=2.05[0.96,3.13]  B=0.60[0.57,0.62]  C=0.61[0.61,0.61]  frozen_oracle C=0.61  joint C-acc=0.09`.
`DISCRIMINATING=False` — **but NOT for the "C transfers easily" reason.** B and C both show
**negative transfer** (FTSR ≈ 0.60 < 1): the stream makes them *harder* than scratch.
No clean "parts transfer, composition doesn't" signature — interference dominates.

**Confound check (is the negative transfer / joint-failure real interference or under-training?):**
joint training over all 9 tasks (3 alphabets × {A,B,C}) at hid∈{256,512}, 15k steps →
**ALL types at chance** (`A=0.05, B=0.12, C=0.15`; chance=0.059). Even **plain addition**,
which groks alone at 5.7k, **collapses to chance under multi-task joint training**, and 2×
capacity / more steps don't help.

**VERDICT (decisive, despite small n — the signal is dramatic and consistent):** the
flat-MLP + mod-p + **grokking** substrate cannot host this experiment. Single-task
grokking does NOT survive multi-task circuit sharing — the phase transition collapses.
So (a) there is no valid joint ceiling to establish headroom, (b) the sequential signal
is swamped by interference, and (c) compositions are bimodal anyway (§8). **Modular
arithmetic is the wrong substrate for the discriminating-composition question.** This
converges with retrospective §4/§6: constructing a regime where simple methods fail is
itself the hard part, and toy-land can't express it. The gap-diagnostic-first discipline
earned its keep — ~20 min of probes, no multi-hour 8-seed run wasted.

**Disposition:** STOP the mod-p nested-binary line. The fork (user's call) is (A) move to
a real compositional-generalization benchmark where the discriminating regime is known to
exist (SCAN / COGS / relational), or (B) bank the durable results (write up the local-vs-
global bound) and treat "toy-land can't host the discriminating regime" as today's finding.
