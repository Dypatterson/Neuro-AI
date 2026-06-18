# Report 146 — Bet B / SCAN tier-B Stage-0 gate on canonical MCD: PASS (the regime discriminates against KNOWN methods)

**Status:** the tier-B Stage-0 gate **RAN → PASSES on all of mcd1/mcd2/mcd3.** On canonical maximum-compound-divergence SCAN, both vanilla seq2seq AND a faithful GECA (in-house lower bound + the published strong fragment-GECA) fail to solve the regime. So — unlike tier-A, where GECA reached 0.85 — **MCD discriminates against the known augmentation fix, and a Stage-1 mechanism is now licensed and testable against known methods.** (2026-06-17.) **Experiment:** `experiments/93_betb_scan_mcd_stage0.py`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer — establishing the GECA-resistant arena (the regime where the *known* method also fails) after [Report 145](../145_betb_scan_tierA_geca_saturated/report.md) showed SCAN-simple compound hold-outs are GECA-saturated.
- **Headline per [betb-geca-resistant-regime-precommit.md:89-90](../../notes/betb-geca-resistant-regime-precommit.md):** the **Stage-0 gate** — (0a) vanilla exact-match ≪ ceiling AND (0b) faithful-GECA exact-match ≪ ceiling, ≥3 seeds. PASS ⇒ the regime discriminates against known methods ⇒ a mechanism is licensed.
- **Controls per [precommit:95-96](../../notes/betb-geca-resistant-regime-precommit.md):** `vanilla_plain` floor · in-house faithful token-GECA (computed-from-buffer) · the **published strong fragment-GECA** (Conklin et al. 2021) as the binding 0b reference · ceiling ≈ 1.0 (MCD is learnable in-distribution).
- **Why now:** STATUS — tier-A failed the gate (GECA-saturated); MCD is the precommit's tier-B and is GECA-resistant by construction. This is the gate that decides whether the arena is real.

## Data

Canonical Keysers et al. 2020 MCD1/2/3 (maximum compound divergence, matched atom distribution), obtained as plain text from `SegwangKim/SCAN` fork `mcd_split/` (pinned SHA `9da9c8af…`, recorded in `data/scan/mcd_split/PROVENANCE_*.sha`). **Integrity verified:** 8365 train / 1045 test per split; **zero train/test command overlap; zero OOV** (all 13 input + 6 output atoms present in every train split — the MCD "atoms matched" property); `around right` present in test (349/308/356). No TensorFlow/TFDS needed.

## Method

Reuse exp87 (GRU enc-dec + attention, greedy exact-match) and exp92's faithful token-rewrite GECA. Arms (n=3 seeds, 30 epochs): `vanilla_plain` · `vanilla_geca` (in-house token-GECA augmentation). The published strong fragment-GECA is cited, not re-run.

## Result (headline, with CI)

| split | `vanilla_plain` | CI95 | `vanilla_geca` (in-house) | CI95 | in-house GECA test-reach | **published strong GECA** |
|---|---|---|---|---|---|---|
| **mcd1** | 0.170 | [0.075, 0.339] | 0.093 | [0.071, 0.111] | 0/1045 (0.0%) | 0.515 |
| **mcd2** | 0.138 | [0.127, 0.158] | 0.173 | [0.036, 0.272] | 0/1045 (0.0%) | 0.304 |
| **mcd3** | 0.023 | [0.014, 0.033] | 0.038 | [0.024, 0.046] | 1/1045 (0.1%) | 0.120 |

ceiling ≈ 1.0. **Gate: vanilla fails AND GECA fails on every split → PASS (all 3).**

## Verdict — PASS: MCD discriminates against known methods

- **0a holds:** vanilla seq2seq reaches at most 0.17 mean (mcd1) and as low as 0.02 (mcd3) — all ≪ ceiling. MCD is not solved by a general-purpose neural learner.
- **0b holds, doubly:** (i) the in-house token-GECA manufactures thousands of examples but reaches **~0% of the held-out test** (vs 52% on tier-A `around right`) — MCD's compound divergence is unreachable by substitution, so GECA augmentation ≈ vanilla; (ii) the *published* strong fragment-GECA (51/30/12%, Conklin et al. 2021, arXiv:2106.04252 Table 2) also fails (≪ ceiling). The known augmentation fix does not solve MCD.

This is the property tier-A lacked. **A Stage-1 restructuring-consolidation mechanism is licensed**, and for the first time in the arc the "beats known methods" question is genuinely testable rather than redundant-by-construction.

## Drill-down — two honest caveats

1. **In-house vanilla runs hotter than the published LSTM** (mcd1 0.170 vs published 0.047; one seed hit 0.339). Cause: our model has Luong attention + 200 hidden + 30 epochs vs the published plain LSTM; mcd1 seed variance is high (CI [0.075, 0.339]). It still **fails** (≪ ceiling) — but the Stage-1 "beat the floor" bar should use **our** in-house vanilla (0.17/0.14/0.02) as the floor, not only the published 0.05, to avoid crediting the mechanism for our baseline's strength.
2. **In-house token-GECA is a lower bound** (reaches ~0% of test). The binding 0b evidence is the *published* strong fragment-GECA (51/30/12%); both fail, so 0b is robust. (Symmetric to Report 145, where the lower-bound argument cut the other way.)

## The Stage-1 bar (quantified, from the scoping workflow `wf_a355e875`)

A brain-shaped local restructuring consolidation must beat the **general-purpose neural floor** — all < 50% mean MCD: vanilla (~5–17%), GECA (51/30/12%), Lev-MAML (48/35/11%), T5-base (26/8/12%). Beating that floor (CI-disjoint, ≥8 seeds, and ≥ the faithful-GECA arm) is an **achievable, genuine positive**. The ~99–100% ceiling (AuxSeq 99.9/90/98; symbolic LeAR 100/100/100) is held only by **structure-injecting / auxiliary-supervised** special-purpose systems — a brain-shaped local mechanism is not expected to reach it, and the honest claim is "beats every general-purpose neural learner + every augmentation/meta method," not "beats the structure-injecting SOTA." (COGS-structural is a cleaner tier-C target — field ≈0% — but a heavier build.)

## Disposition

- **Tier-B MCD Stage-0: DONE → PASS (all 3 splits).** The GECA-resistant arena is established.
- **NEXT = Stage-1 mechanism** — UNDESIGNED (the exp90 peer-alignment recipe does NOT transfer to compound hold-out; [precommit §2](../../notes/betb-geca-resistant-regime-precommit.md)). Requires: a Stage-1 mechanism precommit (slot⊕filler restructuring consolidation / compound→atomic decorrelation) → **MANDATORY anti-homunculus review** → build → ≥8-seed headline vs the in-house floor AND vs faithful GECA. Open decision (user's): **MCD vs COGS** as the mechanism arena.

## Done-gate checklist

1. **Headline + CI:** ✅ both arms × 3 splits, 3-seed bootstrap CI.
2. **Control on same test set:** ✅ vanilla floor + GECA on the identical MCD test sets; published strong-GECA + the full known-method bar cited.
3. **Drill-downs explain the headline:** ✅ the ~0% GECA test-reach (MCD property), the hotter-vanilla caveat, the lower-bound caveat.
4. **Markdown report:** ✅ this file; raw `experiments/exp93_mcd_stage0.json` alongside.
5. **Status/memory updated:** ✅ STATUS + memory note (this session).
