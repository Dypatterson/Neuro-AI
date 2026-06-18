# Report 149 — Lever-1 verification: Report 148 is OVERTURNED (composition is a TIE, not a wall); the discipline caught a premature program-closure

**Status:** a user-mandated "verify before deciding" pass on [Report 148](../148_betb_scan_mcd_cca/report.md) (whose "composition redundant → wall earned" conclusion had led to a recommendation to *close the program*) **overturned that conclusion.** The fair re-test shows composition-as-inference is a clean **TIE** with the holistic baseline — real (beats naive pooling) but not exceeding the encoder's own GRU summary. The "wall earned" reading is **retracted.** This report records the verification (a), the structure-injection ceiling probe (b), and the progress-allocation test (c), and states the honest corrected endpoint. (2026-06-17.) **Experiments:** `exp95` (reproduction), `exp98` (fair composer), `exp96` (oracle), `exp97` (progress).

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B Stage-1 on the GECA-resistant MCD arena (146) — *verifying* the lever-1 conclusion before any program-level decision.
- **Headline per [RETROSPECTIVE-addendum §4-5](../../notes/RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md):** does composition-as-inference, fairly tested (capacity-matched composer, multi-seed, trajectory), beat the matched non-compositional baseline?
- **Why now:** I had recommended closing the program on 148's "wall earned." Asked what I was least confident about, I flagged exactly this; the user mandated verifying (a) 148, (b) the arena ceiling, (c) the untested central hypothesis before deciding. The verification caught a real error — in *my* reasoning this time.

## (a) The 148 verification — the conclusion was wrong three ways

**1. The headline was a COIN-FLIP, not a redundancy (interpretation-skeptic agent).** 148's `cca − cca_holistic = −0.026` splits **4 seeds positive / 4 negative** (paired p≈0.4); the negative sign was driven *entirely by one seed* (leave-one-out → −0.003). The CI [−0.084,+0.024] means "indistinguishable," not "redundant." The "REDUNDANT-BUT-VALID a 5th time / wall earned" framing was **confirmation bias** — a draw relabeled to extend a streak.

**2. The composer was CONFOUNDED.** `cca` conditioned on `W·[mean-pool(L); mean-pool(R)]` (order-destroying); `cca_holistic` on the encoder's full **sequential GRU summary**. So 148 confounded "composition" with "weak mean-pool summarizer" — a fresh confound, structurally like CCC's frozen-decoder one (147).

**3. The code was CLEAN (code-audit agent), so the numbers were real — only the interpretation was wrong.** Same init, capacity, RNG stream, training, eval; the W params present-but-untrained in the holistic arm are harmless. The bug was epistemic, not in the code.

**4. A fresh-seeds reproduction FLIPPED the sign positive (`exp95`, seeds 8-15).** `cca − cca_holistic = +0.063, CI[+0.016,+0.107]` (disjoint **positive**); combined **n=16 = +0.019, 11/16 positive.** 148's negative was an **unlucky 8-seed draw** — even the handicapped mean-pool composer modestly *helps*.

**5. The FAIR (capacity-matched) re-test settles it (`exp98`, n=8, per-epoch trajectory).** Composer = per-clause **GRU readouts** (order-aware, same capacity as the holistic state), varying only clause-boundary-respect. Result:

| arm (mcd1, n=8, ep30) | mean | CI95 |
|---|---|---|
| cca_holistic | 0.424 | [0.359, 0.494] |
| fair_cca | 0.405 | [0.351, 0.460] |
| fair_nosplit | 0.327 | [0.253, 0.400] |

- `fair_cca − fair_nosplit` = **+0.078, CI[+0.020,+0.139]**, sign-robust LOO → **composition is REAL** (clause structure beats whole-pooling).
- `fair_cca − cca_holistic` = **−0.019, CI[−0.063,+0.023]**, sign-robust LOO → **a clean TIE** (indistinguishable).
- The n=1 smoke's dramatic early-peak-then-collapse (0.44@ep5→0.20@ep10) was **NOISE** — at n=8 `fair_cca` and `holistic` rise and plateau together (~0.40–0.43), no collapse, no robust crossover.

**(a) verdict:** composition-as-inference is a **genuine TIE** with the holistic baseline. Composition manufactures *real* structure (beats naive pooling robustly) but **does not exceed what the encoder's own sequential GRU summary already provides** — the GRU encoder is *already a good-enough composer* on this arena. The thesis bar (beat the matched baseline) is not cleared, but this is a draw, **not** the "redundant/worse/wall-earned" of 148.

## (b) The structure-injection ceiling probe (`exp96`)

- **REAL finding:** injecting **atomic clause data** into training (`--clause-aug`: the 102 single-clause SCAN commands, upsampled 10×; covers all 34 mcd1-test clauses + 17 single-clause test commands) **nearly doubled vanilla MCD: 0.158 → 0.297** — far beyond the +0.016 the memorized test-clauses could explain. *Atomic-semantics injection helps* compositional generalization, where *compound*-augmentation (GECA) failed (~0% reach, Report 146). A real structure-injection-via-data lever.
- **ANOMALY (flagged, not over-interpreted):** the post-hoc decompose-decode-concatenate oracle = **0.0**, with `single_clause_em = 0` *even after the model trained on those exact clauses 10×*. An exact 0/17-after-memorization is too anomalous to be a clean ceiling — most likely the holistically-trained GRU's **2-clause bias** (it over-generates / mis-times EOS on isolated single-clause inputs, since all 8365 train commands are 2-clause), but I did not fully isolate that from a subtle eval interaction. **Not reported as a clean ceiling measurement.**
- **Net:** the arena *is* crackable by structure-injection — but the kind that **enforces** clause-structured decoding (Tree-LSTM; LeAR 1.0, lit), which the holistic GRU *resists*. Atomic-data injection independently helps ~2×. This confirms the **bind** (the win needs structure the charter fences as a mechanism), now with an in-harness datum.

## (c) Progress-prioritized allocation — the untested central hypothesis (`exp97`)

*CONTEXT-B §3's central claim: learning PROGRESS (not error) is the allocation currency. Never instantiated on a discriminating arena. Tested here as a progress-weighted sampler vs uniform (+ hardness / antiprogress controls), n=8, mcd1.*

| arm (n=8) | mean | `− uniform` |
|---|---|---|
| uniform | 0.159 | — |
| progress | 0.130 | **−0.029, CI[−0.050,−0.009] DISJOINT** |
| hardness | 0.140 | −0.019, CI[−0.036,−0.005] DISJOINT |
| antiprogress | 0.153 | −0.006, CI[−0.025,+0.012] |

**RESULT: progress-allocation NULLS — actually HURTS** (CI-disjoint below uniform). Hardness-sampling also hurts; antiprogress ≈ uniform. So **any non-uniform allocation toward "interesting" (frontier/hard) examples reduces held-out MCD generalization — uniform coverage is best** (concentrating on a subset under-covers the compound space). The central CONTEXT-B §3 claim (learning-progress is the allocation currency that helps) is **not supported** on the discriminating arena; a clean negative for the project's own central untested hypothesis, as the honest prior predicted ("it allocates; it does not manufacture").

## Synthesis & disposition — the decision is GENUINELY OPEN (not "wall earned")

The corrected, **complete** picture across the three probes — **the emergent/brain-shaped levers do NOT beat the baseline on the discriminating arena, while the structure-injecting levers DO** (the bind, now multiply-confirmed):
- **(a) Composition-as-inference: a clean TIE** — real (beats naive pooling, +0.078 disjoint) but not exceeding the GRU encoder's implicit composition (−0.019, straddles 0); the model has no clause-level competence, which is *why* explicit composition adds nothing net. 148's "wall earned" was a coin-flip + weak composer + unlucky seed-draw.
- **(c) Progress-allocation — the project's OWN CENTRAL untested claim: HURTS** (−0.029, disjoint); uniform coverage beats frontier-prioritization. The allocation currency does not manufacture structure on this arena.
- **(b) What DOES help is structure-injection:** atomic-data injection ~doubles vanilla (0.158→0.297); structure-ENFORCING architectures (Tree-LSTM; LeAR 1.0) crack the arena — the ingredient the charter fences as a *mechanism*.

So the honest endpoint is **NEITHER 148's "wall earned"** (composition is a tie, not a loss; the negative was an artifact) **NOR "thesis alive"** (no brain-shaped lever beats the baseline; the central progress hypothesis hurts). It is: *on the discriminating arena, the emergent/brain-shaped levers tested (composition, learning-progress allocation) tie-or-hurt, while structure-injection consistently wins.* **The decision is GENUINELY OPEN and the user's, now on a correct foundation:** (i) relax toward a structure-ENFORCING architecture (Tree-decoder/Transformer — the field's winning ingredient, a thesis-weakening); (ii) a different substrate that develops clause competence; or (iii) bank the honest contribution. **The single most durable output of this pass: the false-positive-catching discipline (adversarial verification + reproduction + a fair composer) caught the ASSISTANT'S OWN premature, confirmation-biased program-closure — the project's core methodological asset applied to its hardest target.**

## Done-gate checklist

1. **Headline + CI:** ✅ fair composer n=8 + paired deltas + leave-one-seed-out; reproduction n=16.
2. **Control on same test set:** ✅ matched `cca_holistic`/`fair_nosplit`/`fair_random`; clause-aug ceiling probe; (c) uniform/hardness/antiprogress controls.
3. **Drill-downs explain the headline:** ✅ coin-flip per-seed analysis; weak-vs-fair composer; the trajectory (early-peak = noise); clause-competence deficit; clause-aug helps.
4. **Markdown report:** ✅ this file; raw json (`exp98_mcd1_n8`, `exp95_mcd1_repro_s8`, `exp96_mcd1[_clauseaug]`, `exp97_mcd1`) alongside.
5. **Status/memory updated:** ✅ STATUS walked back (148 retracted); memory to follow.
