# Report 145 — Bet B / SCAN tier-A Stage-0: the compound hold-out is GECA-SATURATED → pivot to tier-B canonical MCD

**Status:** the tier-A Stage-0 gate (front-loaded GECA guard, the 144 lesson) **RAN → FAILS in substance.** A custom `around right` compound hold-out on SCAN-simple is solved by the known augmentation fix (GECA), so it does **not** discriminate against known methods. A 4-agent verification+scoping workflow confirmed the verdict (high-conf), killed the cheaper "harden tier-A" alternative, and established that **canonical MCD is the right arena, obtainable as plain text with zero heavy deps.** Decision: **pivot to tier-B canonical SCAN-MCD.** (2026-06-17.) **Experiment:** `experiments/92_betb_scan_compound_stage0.py`; **workflow:** `wf_a355e875-6dd`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer — establishing a regime where the *known* compositional-generalization fix also fails (after [144](../144_betb_scan_geca_guard/report.md) showed SCAN add-jump is GECA-solvable).
- **Headline per [betb-geca-resistant-regime-precommit.md:36-46,89](../../notes/betb-geca-resistant-regime-precommit.md):** the **Stage-0 gate** — (0a) vanilla seq2seq exact-match ≪ ceiling AND (0b) a *faithful* GECA exact-match ≪ ceiling. PASS ⇒ a mechanism is licensed; FAIL ⇒ the regime is GECA-saturated ⇒ harden or escalate.
- **Controls per [precommit:89-96](../../notes/betb-geca-resistant-regime-precommit.md):** `vanilla_plain` floor · a faithful GECA computed-from-buffer (BUILD CONDITION 1) · ceiling ≈ 1.0 (SCAN-simple is learnable).
- **Why now:** STATUS — guard #1 (144) falsified the add-jump consolidation's distinctiveness; the recurring redundancy (133/139/144) is the task-selection confound; the fix is a regime that defeats known methods. This is the cheap, data-in-hand first test of whether SCAN-at-scale can host one.

## Method

**Split (tier A).** SCAN-simple's train+test = the *entire* 20910-command set (a random 80/20), and the grammar is left↔right and verb symmetric. We hold out the contiguous command compound `around right` (every command containing it → test), keeping every atom + primitive in train. Result: **train=15225, held-out test=5685**, zero missing atoms.

**Faithful GECA (new code; exp91's primitive-substitution GECA does not apply to a compound hold-out).** A general **token-rewrite GECA**, all computed-from-buffer: discover *sound* directed rewrite rules `(a→b, action_sub)` where (i) `a,b` share ≥3 command environments (exchangeable) and (ii) every train minimal pair differs by a single consistent 1:1 action-token substitution. This correctly finds `left↔right` (action sub `I_TURN_LEFT↔I_TURN_RIGHT`) and the verb swaps, and correctly *rejects* `around↔opposite` / `twice↔thrice` (action length changes ⇒ no 1:1 sub). It manufactured **+18935** examples (16 rules), of which **2985 land in the held-out region**.

**Arms** (n=3 seeds, 30 epochs, GRU enc-dec + attention, greedy exact-match): `vanilla_plain` (no aug) · `vanilla_geca` (GECA-augmented train). Eval on the 5685 held-out test.

## Result (headline, with CI)

| arm | mean exact-match | CI95 | per-seed |
|---|---|---|---|
| `vanilla_plain` (Stage-0 floor) | **0.0055** | [0.000, 0.012] | 0.0, 0.012, 0.004 |
| `vanilla_geca` (known fix, lower bound) | **0.848** | [0.808, 0.885] | 0.851, 0.808, 0.885 |

ceiling ≈ 1.0 (SCAN-simple is fully learnable, cf. Report 140 random-split 0.9947).

## Verdict — GECA-SATURATED; the Stage-0 gate FAILS in substance

- **0a holds:** vanilla ≈ 0 (the compositional gap is real).
- **0b FAILS:** GECA reaches **0.848** — 85% of ceiling, **not** "≪ ceiling." A known method largely solves the regime.

**The experiment script's mechanical flag (`mean ≤ 0.9 ⇒ "GECA fails" ⇒ GATE PASS`) is a FALSE POSITIVE** and is overridden here. The precommit's binding definition of 0b is "faithful GECA ≪ ceiling" ([precommit:41,89](../../notes/betb-geca-resistant-regime-precommit.md)); 0.848 does not meet it. This is itself a finding — a hard-coded threshold drifting from the binding intent, exactly the metric-trap class the project guards against.

**The 0.848 is a genuine LOWER BOUND** (the load-bearing argument, adversarially verified, high-confidence):
- The token-rewrite GECA is the *weakest* faithful GECA — global token substitution (`left→right` flips *all* lefts) can only reach the **2985/5685 = 52.5%** of held-out examples that contain no `left`; the 47.5% mixing left+right are unreachable. It scored 0.848 anyway (the model generalized from the 52.5% it was shown).
- A faithful **fragment-GECA** (Andreas 2020, local within-environment substitution) reaches **≥92.1%** of the region verbatim with sound labels (derived on the data), including the mixed examples token-GECA misses → projected **~0.95–1.0**, anchored to the add-jump GECA's *measured* 0.984/0.9998 ([144](../144_betb_scan_geca_guard/report.md)).
- Adding correctly-labeled copies of the eval distribution to train cannot lower expected exact-match, so a stronger GECA scores ≥ 0.848. **0.848 is decisive for FAIL.**

A brain-shaped mechanism scoring anywhere in (0.848, 1.0] here would not beat the relevant known method (faithful fragment-GECA ≈ ceiling) → it would reproduce the **133/139/144 REDUNDANT-BUT-VALID pattern a 5th time.** This matches the literature: SCAN template splits are GECA-solvable (Loula 2018 → Andreas 2020); MCD (Keysers 2020) was built precisely to defeat them.

## Scoping workflow (`wf_a355e875-6dd`, 4 parallel agents) — informs the branch

1. **Adversarial verdict check (high-conf): FAIL is correct.** Re-derived every number from the data; confirmed the lower-bound logic. **One honest caveat:** fragment-GECA was *projected, not run* — resting FAIL on a projection is "assert rather than run." The airtight closeout would implement+run fragment-GECA (expected ≈ ceiling). **Moot for the pivot** (we leave tier-A regardless), recorded here for honesty.
2. **MCD data (high-conf): plain-text, zero heavy deps.** Canonical mcd1/2/3 obtainable as `IN:/OUT:` text from `SegwangKim/SCAN` fork `mcd_split/` (8365 train / 1045 test each). **Removes the TensorFlow/TFDS env-change blocker.** Downloaded + integrity-verified (pinned SHA `9da9c8a`; zero leakage, zero OOV, `around right` present in test).
3. **Baseline bar (high-conf): MCD is GECA-resistant; the floor is wide open.** Published GECA on MCD = **51/30/12%** (Conklin 2021) — the known fix *fails*. The general-purpose neural floor (vanilla ~5%, GECA ~31%, MAML ~32%, T5 <17%) is all **< 50% mean** — beating it is an achievable genuine positive. The ~99–100% ceiling (AuxSeq, symbolic LeAR) is held only by **structure-injecting / auxiliary-supervised** special-purpose systems. **COGS-structural is even cleaner** (whole field ≈0%) but a heavier build (logical-form output).
4. **Harden-tier-A is DEAD (high-conf).** A structural `<verb> around <dir>` hold-out is vanilla-hard and token-GECA-closed, **but fragment-GECA cracks it** — verified on the data via the verb-slot bridge (`turn opposite right` ≈ `walk opposite right` → manufacture `walk around right` from `turn around right`). No hand-buildable SCAN-simple split escapes the vanilla-easy / fragment-GECA-vulnerable squeeze — which is *why* Keysers built MCD algorithmically.

## Disposition

- **Tier-A Stage 0: DONE → GECA-saturated → gate FAILS.** Banked. (The cheap, designed-to-be-cheap confirmation that SCAN-simple is GECA-saturated, per precommit §3.)
- **Branch: pivot to tier-B canonical SCAN-MCD** (data acquired + verified; GECA-resistant by construction). Stage-0 MCD gate is `experiments/93` (running) → Report 146.
- **Deferred to the Stage-1 precommit:** the mechanism is UNDESIGNED (slot⊕filler restructuring consolidation; anti-homunculus review required before build); the **MCD-vs-COGS arena choice** (user's) given the MCD ceiling is held by structure-injecting methods.

## Done-gate checklist

1. **Headline + CI:** ✅ both arms, 3-seed bootstrap CI.
2. **Control on same test set:** ✅ `vanilla_plain` floor on the identical 5685 held-out set; ceiling cited (140).
3. **Drill-downs explain the headline:** ✅ the 52.5% region-coverage / lower-bound decomposition; the false-positive-flag finding.
4. **Markdown report:** ✅ this file; raw `experiments/exp92_stage0_around_right.json` + workflow result alongside.
5. **Status/memory updated:** ✅ STATUS walked back (Current state + Recent updates); memory note to follow.
