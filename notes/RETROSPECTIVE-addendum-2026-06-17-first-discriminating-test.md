# Take-stock 2026-06-17 — the first discriminating test, and the structure-injection bind

*Extends [RETROSPECTIVE-two-bets-2026-06-06.md](RETROSPECTIVE-two-bets-2026-06-06.md) with the decisive
post-147 update. Written after the user chose "bank + re-assess the thesis." Synthesis of a 4-perspective
panel (`wf_a19c4cb7`: wall-is-fundamental skeptic, not-creative-enough optimist, literature-grounded check,
contribution-sizer), grounded in the numbered record.*

---

## 1. What changed — the confound the 06-06 retrospective flagged is now REMOVED

The 06-06 retrospective's load-bearing honesty caveat (§4): every prior "brain-mechanism reduces to a simple
one" was **confounded with task selection** — the toys were solvable by simple/known means, so a brain-shaped
mechanism could never show a *necessary* advantage. The thesis "could only be tested where the simple method
FAILS" (§6), a regime the program had **never built.**

This session built it. Tier-A SCAN add-jump was *discarded* precisely because it was GECA-saturated
([145](../reports/145_betb_scan_tierA_geca_saturated/report.md)). Canonical Keysers-2020 MCD
([146](../reports/146_betb_scan_mcd_stage0/report.md)) passed a pre-registered Stage-0 gate where **both** the
general-purpose neural learner (vanilla ~5–17%) **and** the known compositional fix fail (in-house GECA ~0% of
test; published strong fragment-GECA 51/30/12%; Lev-MAML 48/35/11%; T5 <17% — all <50% mean). The §4 alibi is
gone *by construction.* Then the first brain-shaped mechanism on that arena
([147](../reports/147_betb_scan_mcd_ccc/report.md), Clause-Compositional Consolidation) returned a clean,
β-robust result: `ccc − baseline = −0.029 [−.056, −.007]` (CI-disjoint **negative** — it *hurt*), and the
anti-homunculus-mandated separator `ccc ≈ ccc_nosplit` **falsified the composition hypothesis** (clause
splitting buys nothing over generic pooling).

## 2. The verdict on the central thesis — STRONGLY-DOUBTED, not falsified

The thesis ("a brain-shaped *consolidation* manufactures structure/generalization that simple or known methods
cannot") is **UNCONFIRMED, now joined by its first clean negative on a confound-free arena.** The pattern is
now over-determined: a brain-distinctive mechanism is **either** matched/beaten by a simple/known method
(REDUNDANT: 133 SGNS≈SVD, 134 sleep≈replay, 139 BF≈EWC, 144 GECA>consolidation) **or** distinctive-but-nulls
(147) — and this dichotomy now holds **even where nulls cannot be dismissed as task-selection artifacts.** The
bans were not the cause (Bet B lifted them and re-derived the same shape four times). The Bet-A subdominant-modes
bound re-appears intact on a brand-new axis (compositional generalization) and substrate (seq2seq, not FHRR).

**Why it is "strongly-doubted" and NOT "falsified" (the honest hedges, all real):** (1) CCC is **one** recipe on
the confound-free arena, and it is itself **confounded** — §3. (2) Everything is small toys (mod-arithmetic,
small-WikiText, SCAN, D≤4096); a mechanism inert at toy scale could matter where simple methods break at scale —
a regime never built. (3) "Brain-shaped" was tested as a finite operationalization set (FHRR local writers,
Benna-Fusi, surprise/pseudo-replay, two-timescale freeze, factored/clause consolidation), not the full space.
The surviving defenses are all "we haven't tried hard/big/different enough yet" — none is "here is a positive the
controls didn't kill." That asymmetry is itself a signal.

## 3. The SHARPEST finding — the structure-injection bind (this is the new thing)

The literature check (high-confidence, primaries opened this session) is decisive and reframes the whole
question: **NO brain-inspired / replay / consolidation / CLS / Hopfield mechanism appears anywhere on the
SCAN-MCD / CFQ / COGS leaderboards.** Every method that clears the floor **injects structure**:
- **LeAR** (Tree-LSTM + algebraic homomorphism) 90.9 MCD-mean — *ablation without the tree → 30.4: the symbolic
  bias IS the result.*
- **AuxSeq** 97.8 MCD1 — via **auxiliary symbolic-structure supervision** (the structure-injecting ceiling 146 cites).
- **NeSS / NSR** ~100% SCAN — neuro-symbolic stack/grammar machines.
- **Meta-learning without symbolic structure** (Conklin Lev-MAML) reaches only 48/35/11 = *the project's own floor.*
- **Scale** (T5-11B 40.9 CFQ-MCD) is insufficient; collapses as compound divergence rises.

So the project's 147 null is **EXPECTED** — it did not miss a brain-shaped success, because none exists. But this
exposes a bind the program has been walking into:

> **The arenas selected to defeat known methods are precisely the ones that ONLY structure-injection solves —
> and the charter treats structure-injection as borderline/un-brain-like (AuxSeq is "off-target" in 146).** CCC
> made this worst-case: it injected a *weak* structural prior (a single content-blind conjunction split) **and**
> bolted it on as a *regularizer* rather than the inference path — weak structure in the wrong vehicle, exactly
> the category the field shows does not clear MCD.

This is neither "fundamental wall" nor "not creative enough" cleanly. It is: **the thesis as framed may be
structurally hard to win on these arenas** — the only known wins use the one ingredient the charter forbids as a
*mechanism*, and there is no field example of an *emergent* mechanism clearing them.

## 4. What would change the bet — three concrete, cheap, decisive levers

1. **Composition-as-INFERENCE, not as a regularizer (the optimist's fairest unrun test, and 147's own
   §disposition).** CCC's null is confounded: the consolidation forced `h → e_comp` while the **frozen decoder
   still decoded the standard `h` better** (precommit §6 risk, realized). Build a decoder that *consumes* the
   per-clause composed encodings at inference (trained, not frozen) — the smallest change that converts
   "consolidation-as-regularizer fails" into "composition-as-inference fails" (the actual thesis claim). Reuses
   the validated MCD harness + GECA decider + random/nosplit controls. Laptop-sized.
2. **Run the field's WINNING ingredient in-harness as a ceiling-control (the literature's decisive test).**
   Add an AuxSeq-style auxiliary-structure arm (or a tree/structure-injecting decoder) to the *same* harness.
   If even structure-injection fails here → the harness/scale is the problem (and 147's null is uninterpretable).
   If it succeeds → the project has a **quantified structure-injection bar**, and the next brain-mechanism null
   becomes decisive (beat-or-differ-from a known ceiling) instead of ambiguous. This directly tests whether the
   "no structure-injection" constraint is the binding limit.
3. **Carry the 142 substrate-shape lever to the discriminating arena (the strongest existing pro-thesis datum).**
   [Report 142](../reports/142_betb_scan_factored/report.md): the *identical* consolidation went from **inert to
   load-bearing** (+0.079, 5/5 seeds) purely by changing the substrate to expose role/filler structure — and was
   *never carried to MCD.* If a substrate that exposes compositional structure makes consolidation load-bearing
   on MCD, the lever is **substrate-shape, not mechanism-absence**, and the thesis is alive.

*Genuinely untested (panel-vetted, distinct from tested-and-nulled):* learning-PROGRESS-prioritized allocation
(CONTEXT-B §3's *central* claim, never instantiated on any discriminating arena); compositional curriculum
(single-op → conjunction → nested); a self-attention/Transformer substrate (every MCD run used a GRU's holistic
`h`-bottleneck); generative pseudo-replay of *compositions* (not surface examples); weight-space (modular/low-rank)
consolidation (the only form for which the full 2×2 gate applies); the **COGS** arena (cleaner — field ≈0% on
structural splits).

## 5. The honest decision space (the user's call)

- **(a) Resolve the confound first** — run lever 1 (composition-as-inference) ± lever 2 (structure-injection
  ceiling). Cheap, and converts the one discriminating null from "thin & confounded" into "decisive." *This is
  the minimum to honestly claim the thesis was tested where it matters.*
- **(b) Relax the framing** — accept structure-injection (a learned compositional architecture / parse-as-
  inference) as the *mechanism*, and reframe the thesis from "emergent consolidation manufactures structure" to
  "the right structural prior + replay generalizes." A real weakening, but it is where every field win lives.
- **(c) Bank and reframe the contribution** — the durable outputs (§6) are real and citable; stop chasing the
  graduation and write the program up honestly as *a rigorous map of where brain-shaped mechanisms do NOT beat
  simple/known methods, plus a reusable discipline and a discriminating arena others can use.*

**Honest lean:** (a) before (c). The single biggest open question — is the 147 null about *consolidation* or
about the *frozen-decoder vehicle*? — is cheap to resolve (lever 1) and the project should not bank a
fundamental-wall reading on a confounded mechanism. But if lever 1 *also* nulls with the architecture confound
removed, the "wall" reading is then earned, and (c) is the honest close.

## 6. Durable contributions (sized, non-inflated)

1. **The Bet-A local-vs-global bound** — the most citable: paradigmatic structure lives in the subdominant modes
   of the co-occurrence operator, reachable by global/nonlinear-partition computation (+0.11–0.25) but no local
   single-projection dynamic over a flat code; route-invariant across ~7 operators (123–132), capability-confirmed
   (126). A clean, precise negative with a named mechanism.
2. **The GECA-resistant MCD arena + the first clean non-redundant test** (145–147) — built the discriminating
   regime the 06-06 retrospective named as missing; makes "beats known methods" testable, not redundant-by-construction.
3. **The false-positive-mode catalogue + discipline engine** — the most transferable output (label-shuffle,
   gauge-free B-KILL, competent-control reproduction, n=8-over-n=2, the named metric traps, GECA-as-Stage-0-gate,
   log-FTSR). Adversarial verification + competent controls before banking.
4. **Validated components** — the graduated role-selective associative memory (055–058), the FHRR/Hopfield/masked
   substrate, the MESH-scaling resolution (120), two continual-learning harnesses + the MCD harness.
5. **The Bet-B continual finding** — compounding transfer is carried by interleaved REPLAY, not the offline pass
   (138); the interaction gate clears via soft weight-anchor+replay = RECOMBINANT (139). A genuine negative on a
   solvable toy.

**What this is NOT:** not a proof "biology can't manufacture structure" (a finite operationalization set, one
arena, one mechanism); not a breakthrough (in ~25 experiments no brain-distinctive mechanism has cleanly beaten a
simple/known method on a discriminating task); not "it all failed" (the bound, the arena, the discipline, the
components are real). The thesis remains **UNCONFIRMED + one earned negative data point** — the start of testing
it where it matters, not its resolution.
