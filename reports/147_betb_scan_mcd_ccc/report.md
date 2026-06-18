# Report 147 — Bet B / SCAN-MCD Stage-1: Clause-Compositional Consolidation NULLS (and hurts) — the composition hypothesis is falsified by the no-split control

**Status:** the first Stage-1 mechanism on the GECA-resistant MCD arena **RAN → clean, well-controlled NULL.** The consolidation does NOT manufacture compositional generalization; it *degrades* the model (ccc 0.126 < baseline 0.155, CI-disjoint), and the composition hypothesis is independently falsified by `ccc ≈ ccc_nosplit` (clause composition ≈ generic whole-pooling). This is the **first genuinely non-redundant test in the arc** — the arena defeats known methods, so this is a clean *negative*, not the 133/139/144 redundant-positive pattern. (2026-06-17.) **Experiment:** `experiments/94_betb_scan_mcd_ccc.py`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer — the Stage-1 restructuring-consolidation mechanism on the GECA-resistant SCAN-MCD arena (the "does consolidation manufacture compositional generalization a known method can't" test).
- **Headline per [betb-mcd-stage1-mechanism-precommit.md:90-94](../../notes/betb-mcd-stage1-mechanism-precommit.md):** mcd exact-match, **ccc − ccc_baseline CI-disjoint > 0 AND ccc ≥ faithful-GECA**, ≥8 seeds, bootstrap CI, on ≥ mcd1.
- **Controls per [precommit §4](../../notes/betb-mcd-stage1-mechanism-precommit.md):** vanilla floor · ccc_baseline (λ=0) · **ccc_random (must NULL)** · **ccc_nosplit (must NOT match ccc — the composition-vs-tightening separator)** · faithful GECA (redundancy decider). Anti-homunculus review: PASS-WITH-FIXES (folded in).
- **Why now:** STATUS — Stage-1 licensed by Report 146; this is the crux mechanism test, designed by a 4-angle panel + a decisive data verification (MCD hardness = clause composition, not verb-filler).

## The mechanism (CCC) — provenance

A 4-angle design panel (`wf_dff3f687`) converged on a *filler-invariance* family, but a direct grep on the MCD data **re-selected the mechanism**: mcd1 verb×modifier cells are saturated (only add-jump-shaped holes), while **100% of test templates and 88% of operator-skeletons are novel** → MCD's hardness is clause/operator *composition*, not verb-filler. So CCC composes in representation space: a fixed content-blind parser splits at the single top-level conjunction (`and`/`after`); a learned compose operator forms `e_comp = W_conj·[pool(L); pool(R)]`; an offline consolidation (decoder frozen) minimizes `CE_whole + β·‖h − e_comp‖² + γ·CE(decode_from(e_comp), gold) + δ·hinge`. Anti-homunculus review: PASS-WITH-FIXES.

## Result — headline (mcd1, n=8, 30 epochs, 800 consolidation steps, β=γ=3, δ=0.1)

| arm | mean exact-match | CI95 | min | max |
|---|---|---|---|---|
| vanilla_plain | 0.159 | [0.114, 0.225] | 0.077 | 0.370 |
| ccc_baseline (λ=0) | 0.155 | [0.104, 0.219] | 0.067 | 0.355 |
| **ccc** (mechanism) | **0.126** | [0.073, 0.197] | 0.032 | 0.345 |
| ccc_random (must-null) | 0.064 | [0.037, 0.102] | 0.027 | 0.181 |
| ccc_nosplit (separator) | 0.116 | [0.069, 0.178] | 0.041 | 0.306 |
| vanilla_geca (redundancy) | 0.107 | [0.074, 0.142] | 0.044 | 0.186 |

**Paired deltas (bootstrap CI):**
- `ccc − ccc_baseline` = **−0.029, CI[−0.056, −0.007]** — CI-disjoint **NEGATIVE**. The consolidation **hurts**.
- `ccc − ccc_nosplit` = **+0.010, CI[−0.006, +0.025]** — NOT disjoint. **Clause composition ≈ generic whole-pooling.**
- `ccc − ccc_random` = +0.062, CI[+0.018, +0.107] — disjoint, but this is dominated by *random-split corrupting* the pooling (pools across clause boundaries → garbage `e_comp`), not by conjunction-composition helping.
- `ccc − vanilla_geca` = +0.019, CI[−0.041, +0.096] — NOT disjoint.

## Verdict — clean NULL; the composition hypothesis is falsified

1. **The headline FAILS in the negative direction:** `ccc − baseline` is CI-disjoint *below* 0. The consolidation degrades performance rather than manufacturing generalization.
2. **The composition hypothesis is independently falsified by the AH-mandated control:** `ccc ≈ ccc_nosplit` (not CI-disjoint). Splitting at clause boundaries and composing buys **nothing** over pooling the whole command — so any consolidation effect is generic representation-tightening, **not** clause composition. This holds *regardless* of the headline sign, which is the decisive finding.
3. `ccc > ccc_random` is real but is "a random split corrupts pooling," not evidence the conjunction structure helps.

**Why it likely hurts:** the consolidation freezes the decoder and forces `h → e_comp` (a composed-from-clause-pools state); the frozen decoder decodes the standard-training `h` better than the forced one, so the self-consistency transfer is counterproductive (the precommit §6 risk, realized — the effect flows through the decoder *init* while attention over the original positions already carries the load).

## Robustness — β-sensitivity check (β=1, n=8)

*The smoke (β=1, undertrained) showed a faint +0.018; the headline locked β=3 a-priori (test decodes from `h`, so the transfer term is load-bearing). To confirm the null is not a β=3-too-strong artifact, a β=1, n=8 run was added.*

| arm (β=1, n=8) | mean | `ccc − baseline` | `ccc − nosplit` |
|---|---|---|---|
| ccc_baseline 0.145 / ccc 0.126 / ccc_nosplit 0.118 | — | **−0.018, CI[−0.060, +0.020]** (straddles 0, leans negative) | **+0.008, CI[−0.007, +0.021]** (NOT disjoint) |

**The null is β-ROBUST.** At β=1 the consolidation is net-neutral-to-harmful (never lifts over baseline); at β=3 it significantly hurts. Critically, **`ccc ≈ ccc_nosplit` at BOTH β** — clause composition is never load-bearing over generic whole-pooling. The smoke's +0.018 (β=1, n=1, undertrained) was undertraining noise. **CONCLUSION: across both a-priori configs the consolidation never helps, and the composition hypothesis is robustly falsified.**

## Meaning in the arc — a clean NULL, not a redundancy

This is the **first Stage-1 test on a genuinely GECA-resistant arena** (Report 146). Unlike 133/139/144 (where a brain-shaped mechanism *worked* but a known method matched it → REDUNDANT-BUT-VALID), here the mechanism **fails outright** on an arena where known methods *also* fail. The honest statement: *neither the general-purpose baseline (vanilla 0.16) nor this brain-shaped clause-composition consolidation solves MCD; the consolidation makes it worse, and clause-composition is not load-bearing over generic pooling.* The wall mapped by Bet A (structure is not read off the surface; the brain-shaped mechanism does not beat the simple baseline) **re-appears on the compositional-generalization axis** — now on an arena selected precisely so the null cannot be dismissed as redundancy.

## Disposition

- **CCC (clause-composition-as-consolidation): NULL → banked.** A clean, well-controlled negative; the composition hypothesis falsified by the no-split control.
- **The honest iterate question (user's):** the controls (`ccc ≈ ccc_nosplit`) suggest composition is *not* carrying signal under a consolidation-as-regularizer with a frozen decoder — so a redesign that makes the compose path the actual *inference* path (a compositional architecture, decode from `e_comp` at test) is the natural next try, **but** its premise is weak (composition isn't beating pooling even in the favorable control). Alternatives: a different mechanism family, or banking the program with the durable contribution (the GECA-resistant arena + the first clean non-redundant null).
- **Durable assets:** the MCD harness + the GECA-resistant arena (Report 146) + this clean null stand.

## Done-gate checklist

1. **Headline + CI:** ✅ all 6 arms, 8-seed bootstrap CI + paired deltas.
2. **Control on same test set:** ✅ all arms on the identical mcd1 test; vanilla floor + GECA + the random/nosplit cleanliness controls.
3. **Drill-downs explain the headline:** ✅ ccc≈nosplit (composition not load-bearing); ccc>random (random corrupts); the frozen-decoder disruption interpretation.
4. **Markdown report:** ✅ this file; raw `experiments/exp94_mcd1_n8.json` (+ β=1 robustness) alongside.
5. **Status/memory updated:** ✅ STATUS + memory (this session).
