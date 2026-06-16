# Report 144 — Bet B / SCAN Stage 1: the GECA redundancy guard FALSIFIES the graduation claim (REDUNDANT-BUT-VALID)

**Status:** the critical guard on the [Report 143](../143_betb_scan_factored2/report.md) graduation **RAN → the distinctiveness CLAIM is falsified.** The known SCAN fix (GECA) **matches and beats** the consolidation. The headline *finding* (consolidation is load-bearing within the factored arch) stands; the *claim* ("…composition a known method can't") does not. (2026-06-16.) **Experiment:** `experiments/91_betb_scan_geca.py`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer in the discriminating regime — the **redundancy guard** on the 143 graduation.
- **Headline per [betb-scan-stage1-consolidation-precommit.md:97-103](../../notes/betb-scan-stage1-consolidation-precommit.md):** does **GECA-style augmentation** (the known add-jump fix) also reach the consolidation's jump-split exact-match (~0.868)? Lift ≈ GECA → bank REDUNDANT; consolidation ≫ GECA → distinctive.
- **Controls per [precommit:97-110](../../notes/betb-scan-stage1-consolidation-precommit.md):** `vanilla_plain` (Stage-0 floor) · `vanilla_geca` (known fix on the simple method) · `factored_geca` (augmentation on the *same substrate* as consolidation, no consolidation) · cited `factored_baseline` 0.111 + `factored_consolidation` 0.868 (Report 143, same seeds 0–7).
- **Why now:** STATUS Blocker #1 — "the critical guard"; must run before any "beats known methods" claim.

## Method — a generous, computed-from-buffer GECA

GECA (good-enough compositional augmentation, Andreas 2020) on add-jump reduces to **primitive substitution** (precommit §4.2: "swap `jump` into the template slots other verbs occupy"). `jump` appears in train ONLY standalone; `walk/run/look` appear in every template. GECA generates synthetic `jump`-in-template pairs by substituting `jump`/`I_JUMP` into the templates the other verbs occupy. The substitution is **label-preserving and sound for SCAN** (verb→action is 1:1 and position-preserving), so the generated outputs are correct.

**Generous-to-GECA by design** (makes the guard *stringent* — if even a best-shot GECA establishes redundancy, the verdict is robust). All of GECA's structural knowledge is **computed from the train buffer** (BUILD CONDITION 1 — nothing hardcoded):
- peer set = tokens appearing as a complete one-token command → `{jump, walk, run, look}` (asserted).
- held-out primitive = the peer verb(s) with **zero** template (multi-token) occurrences → `{jump}` (**discovered**, not assumed).
- verb→action = the standalone command→output map → `{jump:I_JUMP, walk:I_WALK, run:I_RUN, look:I_LOOK}` (asserted).

**Augmentation stats:** 14670 original → **+7706 generated** → 22376 total. (Sound example: `look opposite right thrice and jump left → I_TURN_RIGHT I_TURN_RIGHT I_LOOK ×3, I_TURN_LEFT I_JUMP`.) **Transparency note:** the 7706 generated pairs ≈ the **entire 7706-example jump-test distribution** — GECA reconstructs the held-out compositional manifold *as training data*, purely from train structure + the substitution rule (**no test labels read**). This is exactly why GECA solves add-jump, and it is the sharpest possible illustration of the data-space-vs-representation-space contrast below.

Arms (all n=8, seeds 0–7, matching the 143 stabilization run): `vanilla_plain` (exp87 Seq2Seq, plain train, 30 ep) · `vanilla_geca` (exp87, augmented train, 30 ep) · `factored_geca` (exp90 Factored2 on the augmented train, 50 ep, **no consolidation**). `factored_baseline`/`factored_consolidation` cited from Report 143 (same seeds).

## Result (headline, with CI)

| arm | mean jump-split | CI95 | min | per-seed |
|---|---|---|---|---|
| `vanilla_plain` (Stage-0 floor) | **0.008** | [0.002, 0.017] | 0.0003 | all 8 ≤ 0.037 |
| `vanilla_geca` (known fix, simple method) | **0.984** | [0.956, 1.000] | 0.888 | 6/8 = 1.000 |
| `factored_geca` (known fix, consolidation's substrate) | **0.9998** | [0.9995, 1.000] | 0.999 | 8/8 ≈ 1.0 |
| `factored_consolidation` (Report 143, cited) | 0.868 | [0.727, 0.975] | 0.568 | — |
| `factored_baseline` (Report 143, cited) | 0.111 | [0.030, 0.218] | 0.0 | — |

**Paired deltas (seeds aligned 0–7):**
- `consol − vanilla_geca` = **−0.117**, CI [−0.261, **−0.001**] — CI disjoint from 0 (negative).
- `consol − factored_geca` = **−0.132**, CI [−0.273, **−0.025**] — CI disjoint from 0 (negative).

## Verdict — REDUNDANT-BUT-VALID; the distinctiveness clause is FALSIFIED

GECA does not merely *match* the consolidation — it **beats** it, on both architectures, **more robustly** (GECA min 0.888 / 0.999 vs consolidation min 0.568), with both paired CIs disjoint from 0 on the negative side. Per the precommit §3 disposition ("Lift ≈ GECA / a known fix → bank as REDUNDANT … novelty would need the local/emergent form to **BEAT** the engineered one"), the consolidation does the opposite of beating it.

So the graduation **CLAIM** — *"a brain-shaped restructuring consolidation manufactures composition a simple/known method cannot"* — is **falsified in its "known method cannot" clause.** A known method (GECA) *can*, and does it better. This is the **133/139 pattern a 4th time** (133: SGNS matches-not-beats SVD; 139: EWC reproduces Benna-Fusi; now 144: GECA beats the consolidation).

**What still stands (do not over-walk-back):** the Report-143 headline *finding* is real and unaffected — within the factored architecture, the consolidation pass lifts jump-split composition 0.111→0.868 across 8/8 seeds; it is unambiguously load-bearing *for that architecture*. What this guard removes is the *interpretation* that the capability is one known methods lack.

## Drill-down — the one measured residual (a characterization, NOT a novelty claim)

The consolidation and GECA use the **same structural signal** (the computed peer set / held-out primitive) but spend it differently:
- **GECA** spends it in **data space** — it manufactures 7706 synthetic composed-`jump` examples (≈ the whole jump-test manifold) and trains on them. The model is *shown* composed `jump`.
- **Consolidation** spends it in **representation space** — it realigns `jump`'s role-embedding toward its peers from the standalone `jump→I_JUMP` signal, and reaches 0.868 **having seen zero composed-`jump` examples.**

This is a genuine mechanistic difference (zero-shot representation alignment vs few-shot data augmentation), and it is the only axis on which the consolidation is not simply dominated. **Per the charter it is explicitly NOT a novelty claim** — the bar for novelty is *beating* the engineered method, which it fails. GECA's augmentation is also trivially cheap to generate, so "data efficiency" is not a clear win either. We record the contrast honestly and bank the headline as **redundant**.

## Why this was foreseeable (the RETROSPECTIVE §4 confound, re-confirmed)

[RETROSPECTIVE-two-bets §4](../../notes/RETROSPECTIVE-two-bets-2026-06-06.md) flagged that "brain-distinctive mechanisms reduce to simple ones" is **confounded with task selection** — the regimes were solvable by simple means. SCAN add-jump was chosen because the *vanilla* method fails (0.003), making it a discriminating regime **against vanilla seq2seq**. But it is **not** discriminating against the *known compositional-augmentation* fix: GECA solves it trivially. A positive there was therefore always at redundancy risk — which is exactly what guard #1 was built to catch, and did, **before** any overclaim was banked.

## Disposition

- **Guard #1 (the critical one): DONE → claim falsified-as-redundant.** Banked.
- **Guards #2 (second-split generality) and #3 (computed-`verb_ids` + anti-homunculus pass): now lower-value** — they would firm up the generality/cleanliness of a mechanism already shown redundant. Run only if we want to characterize the redundant-but-real consolidation for the record.
- **The live decision (user's):** (a) **bank-and-stop** this SCAN-consolidation recipe (the honest 133/139 outcome), or (b) escalate to a regime where **GECA also fails** — SCAN MCD / length splits, or COGS — i.e. make the regime discriminate against *known* methods, not just vanilla. (b) is the only route on which "consolidation beats known methods" remains testable.

## Done-gate checklist

1. **Headline + CI:** ✅ all arms with bootstrap CI95 + paired deltas.
2. **Control on same test set:** ✅ all arms evaluated on the identical jump-test set; `vanilla_plain` floor + cited 143 anchors at the same seeds.
3. **Drill-downs explain the headline:** ✅ the data-space-vs-representation-space contrast + the 7706 ≈ test-manifold transparency note.
4. **Markdown report:** ✅ this file; raw `exp91_full.json` alongside.
5. **Status/memory updated:** ✅ STATUS.md walked back FIRST (Current state + Recent updates); memory note to follow.
