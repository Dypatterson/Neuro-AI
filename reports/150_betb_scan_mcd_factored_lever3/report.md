# Report 150 — Lever 3: a factored substrate does NOT rescue clause-composition consolidation on MCD (substrate-shape is not the missing lever; Report 142 retired on MCD)

**Status:** the user-chosen Lever 3 ([RETROSPECTIVE-addendum-2026-06-17 §4 item 3](../../notes/RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md)) — carry Report 142's factored substrate-shape lever to the GECA-resistant MCD arena (146) — RAN and **NULLS.** Factoring the substrate (structure-role ⊥ content-fill) does **not** make the clause-composition consolidation load-bearing on MCD, where it nulled on the holistic substrate (147). The pre-registered thesis-alive signature appears on **neither** split. This converts Report 142 (the strongest remaining pro-thesis datum) from "untested on MCD" into "tested on MCD, didn't transfer" — the "wall" reading earned on a **correct** foundation (vs 148's confounded coin-flip). **Adversarially verified** (thesis-defender + result-auditor): null safe to bank, no re-run required. (2026-06-21.) **Charter:** [notes/betb-mcd-lever3-factored-precommit.md](../../notes/betb-mcd-lever3-factored-precommit.md) (anti-homunculus PASS-WITH-FIXES). **Experiments:** `exp99` (mcd1+mcd2, n=8), `analyze_mcd_divergence_axis.py`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B Stage-1 mechanism on the GECA-resistant MCD arena — *Consolidation-write × substrate-shape* combination.
- **Headline per [betb-geca-resistant-regime-precommit.md:91-94](../../notes/betb-geca-resistant-regime-precommit.md) + [betb-mcd-stage1-mechanism-precommit.md:16-19](../../notes/betb-mcd-stage1-mechanism-precommit.md):** held-out MCD exact-match, `factored_ccc − factored_baseline` CI-disjoint > 0 AND `≥ faithful-GECA`, ≥8 seeds; PLUS the 2×2 interaction (consolidation inert-on-holistic, load-bearing-on-factored) AND conjunction-split specificity (beats both random-split and nosplit).
- **Required controls per [precommit §4](../../notes/betb-mcd-lever3-factored-precommit.md):** `vanilla_plain`, `vanilla_geca`, `factored_baseline`, `factored_ccc_random` (NULL), `factored_ccc_nosplit` (must-not-match), `random_partition` (NULL), holistic `ccc` (the 2×2 top-right).
- **Last verified:** [149](../149_betb_scan_mcd_lever1_verification/report.md) (composition TIES; progress HURTS); [147](../147_betb_scan_mcd_ccc/report.md) (CCC NULLS on holistic, −0.029); [142](../142_betb_scan_factored/report.md) (the same-shape consolidation went inert→load-bearing +0.079 when factored, on SCAN add-jump).
- **Why now:** the user-chosen lever 3, on a correct foundation; the honest prerequisite before any "bank / wall earned" decision.

## 0. The reframe — why the LITERAL 142 port is dead, and what was actually run

`experiments/analyze_mcd_divergence_axis.py` (this session): MCD's **primitive-filler axis is saturated** (mcd2 = **0.0%** filler-novel clauses; mcd1/mcd3 ~6%), 100% of test templates novel; and 142's `len==1` peer predicate yields the **empty set** on mcd1/mcd3. So a literal 142 port (align the 4 primitive verbs' role-halves) has no purchase on MCD and can't even define its peer set — **doubly dead.**

The faithful reframe (this run): carry 142's **substrate shape** (input embedding split into a role channel [encoder runs on it] ⊥ a content-fill channel [read by a separate head]; the content partition **discovered in-code** by a biconditional input↔output predicate → `{jump,walk,run,look,left,right}`, identical on all 3 splits) but match the **consolidation** to MCD's actual axis = the CCC clause-composition operation (force the holistic role-state `h` toward a learned composition of role-channel clause-pools + decode-consistency + anti-collapse). The decisive design is the **2×2** `{holistic, factored} × {±consolidation}`; thesis-alive ⟺ consolidation inert-on-holistic AND load-bearing-on-factored.

## 1. Premise gate

| split | factored TRAIN-EM | factored_baseline test-EM | gate (train≥.90 ∧ base≥.10) |
|---|---|---|---|
| mcd1 | 0.988 | 0.207 | **PASS** |
| mcd2 | 0.852 | 0.104 | **FAIL** (2/8 seeds collapsed: train-EM 0.67, 0.23) |

mcd1 is a clean, competent, premise-passing substrate. **mcd2 FAILED the premise gate** (factored substrate unstable on 2 seeds) → its arm-level means are partly collapse-polluted; mcd2 conclusions below lean on the **paired controls** (premise-independent), not arm means.

## 2. Headline + the 2×2 interaction — fails on both splits

| arm | mcd1 | mcd2 |
|---|---|---|
| vanilla_plain | 0.166 [.118,.237] | 0.131 [.110,.151] |
| **ccc** (holistic+consol) | 0.131 [.074,.206] | **0.204** [.157,.254] |
| **factored_baseline** | **0.207** [.169,.247] | 0.104 [.085,.120] |
| **factored_ccc** | 0.163 [.140,.186] | 0.123 [.099,.150] |
| factored_ccc_random | 0.076 [.056,.098] | 0.116 [.093,.140] |
| factored_ccc_nosplit | 0.168 [.146,.191] | 0.091 [.068,.119] |
| random_partition | 0.119 [.083,.156] | 0.172 [.108,.255] |
| vanilla_geca | 0.121 [.097,.144] | 0.162 [.118,.207] |

**Headline `factored_ccc − factored_baseline`:**
- **mcd1 = −0.044 [−0.078, −0.017] — CI-disjoint NEGATIVE** (7/8 seeds neg). Consolidation *hurts* the factored substrate.
- **mcd2 = +0.019 [−0.011, +0.047] — straddles 0 (null).**

→ The headline (disjoint-positive) is met on **neither** split. (Secondary conjunct `factored_ccc ≥ vanilla_geca`: mcd1 clears it +0.042 disjoint; mcd2 fails it −0.039 — moot, since the primary already fails.)

**The 2×2 interaction runs the WRONG way for the thesis.** The thesis needs consolidation inert/hurt on holistic and load-bearing on factored. Observed:
- mcd1: consolidation hurts **both** (factored −0.044; holistic `ccc − vanilla = −0.035 [−0.070,+0.004]`, re-confirms 147).
- mcd2: consolidation is **null on factored** (+0.019) but **helps the HOLISTIC substrate** (`ccc − vanilla = +0.074 [+0.043, +0.109]`, disjoint positive) — the *opposite* direction. CCC's only win on this run is *without* factoring.

## 3. The controls — composition-specificity fails on both splits

- **mcd1:** `factored_ccc − factored_ccc_nosplit = −0.005 [−0.019,+0.009]` (straddle) → clause-composition buys **nothing over generic whole-pooling** — the exact 147 separator that falsified CCC, reproduced on the factored substrate.
- **mcd2:** `factored_ccc − factored_ccc_random = +0.007 [−0.013,+0.027]` (straddle) → the conjunction split is **indistinguishable from a random interior split**. mcd2 *does* show `factored_ccc − nosplit = +0.032 [+0.014,+0.049]` (disjoint), but read with the random-split straddle this is "**any split helps a little, the conjunction structure is irrelevant**" = generic split-capacity, not composition (verdict-ladder rung 2). **This control is premise-independent** (a paired comparison of two consolidation variants on the same substrate), so it survives the mcd2 instability.
- `random_partition` (the design-time-homunculus / generic-capacity decider): mcd1 0.119 (below factored_ccc); mcd2 0.172 but inflated by one collapsed-seed outlier (0.425) — not over-read.

## 4. Honest drill-downs (the nuance the controls force)

1. **"Consolidation hurts" is NOT the right blanket claim.** It hurts the factored substrate on mcd1 (−0.044) and both substrates on mcd1, but it **helps the holistic substrate on mcd2** (+0.074 disjoint). So consolidation's effect is **small and split-/substrate-dependent**, not a clean directional hurt — and mcd2 partially *reverses* 147 on the holistic substrate. That split-dependence itself argues against a robust "consolidation manufactures structure" claim: a real structure-manufacturer would show consistently, and it does not.
2. **The factored ARCHITECTURE is split-dependent and structure-injection-flavored.** factored_baseline edges holistic vanilla on mcd1 (0.207 vs 0.166) but is *below* it on mcd2 (0.104 vs 0.131) and unstable there (premise fail). So even the mild mcd1 architectural benefit doesn't generalize, and it is an architectural *structural prior*, not the emergent consolidation. (Descriptive only — no paired CI computed.)
3. **The steelman recompute kills the "instability understated it" escape.** Dropping mcd2's 2 collapsed seeds, the kept-6 `factored_ccc − factored_baseline = +0.014 [−0.023, +0.051]` — **still straddles**, and *weaker* (the collapsed seeds carried +0.041/+0.029). A stabilized mcd2 would look worse, not better.

## 5. Adversarial verification (verify-before-deciding, the 149 lesson applied proactively)

Two independent agents on the raw JSON + precommit:
- **Thesis-defender** (tasked to OVERTURN the null): the strongest pro-thesis reading (mcd2 fccc>nosplit) **fails its own pre-registered specificity control** (ties random-split), fails the headline on both splits, fails GECA on mcd2, and is *not* understated by the collapsed seeds (kept-6 recompute shrinks it). "**The null is safe to bank. No re-run required.**"
- **Result-auditor**: arithmetic reproduces exactly; headline + controls read correctly; flagged 3 honesty fixes (all applied above: drop "consolidation hurts" as a blanket; label mcd2 premise-failed + lean on paired controls; state the dropped conjuncts). "**The NULL is real, correctly computed, and safe to write up.**"

## 6. Verdict & disposition

**LEVER 3 NULLS — substrate-shape is not the missing lever.** Factoring the substrate does not make clause-composition consolidation load-bearing on the GECA-resistant MCD arena: the headline fails on both splits (mcd1 disjoint-negative, mcd2 null), the predicted substrate-shape interaction never appears (mcd1 hurts both; mcd2 helps the holistic substrate — wrong direction), and composition-specificity fails on both (mcd1 ≈ nosplit; mcd2 ≈ random-split). **Report 142 is retired as the missing lever for MCD.**

This is the high-value, most-likely outcome the precommit anticipated (verdict-ladder rung 1, reinforced by rung 2 on mcd2), now **more robust** than a clean-mcd1-only null: across two splits no configuration produces the thesis-alive pattern, and the effects that do appear are small, split-dependent, and **structure-injection-flavored** (the architecture, not the emergent consolidation) — consistent with the multiply-confirmed structure-injection bind.

**The program decision (the user's), now earned on a correct foundation.** The emergent/brain-shaped levers tested on the discriminating arena — CCC consolidation (147 null), composition-as-inference (149 tie), progress-allocation (149 hurts), and now substrate-shape (150 null) — **uniformly fail to beat the baseline**, while structure-injection (atomic-data ~doubles vanilla; Tree-LSTM/LeAR ceiling) uniformly wins. The live options narrow to: **(a)** relax to a structure-ENFORCING substrate (Tree-decoder/Transformer — where the field's wins live; a thesis-weakening), or **(b)** bank the honest contribution (the local-vs-global bound, the GECA-resistant arena, the discipline engine, the now-complete map of where brain-shaped mechanisms do not beat simple/known methods).

## Done-gate checklist

1. **Headline + CI:** ✅ `factored_ccc − factored_baseline`, n=8, bootstrap CI, both splits + per-seed + steelman kept-6 recompute.
2. **Controls on same test set:** ✅ matched `factored_baseline`, holistic `ccc`/`vanilla_plain` (the 2×2), `factored_ccc_random`, `factored_ccc_nosplit`, `random_partition`, `vanilla_geca` — all n=8.
3. **Drill-downs explain the headline:** ✅ split-dependent consolidation; factored-architecture split-dependence; the premise-gate fail + its bias analysis; the composition-specificity controls.
4. **Markdown report:** ✅ this file; raw json (`exp99_mcd1_n8`, `exp99_mcd2_n8`, `exp99_premise`) + `analyze_mcd_divergence_axis.py` alongside.
5. **Status/memory updated:** ✅ STATUS walked back (active deliverable → lever-3 null); memory updated.
