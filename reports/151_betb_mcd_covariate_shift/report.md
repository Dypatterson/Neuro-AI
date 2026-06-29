# Report 151 — SCAN-MCD is compound covariate-shift; the clause-aug floor has zero atom/clause headroom (and is partly leaked)

*Bet B. **Type: CHARACTERIZATION** (no training, no gate moved) — the today-feasible salvage of
"re-ground the bar." Reads data only. Script:
[experiments/analyze_mcd_covariate_shift.py](../../experiments/analyze_mcd_covariate_shift.py);
JSON: `experiments/analyze_mcd_covariate_shift.json`. Plan:
[notes/betb-bar-regrounding-part1-precommit.md](../../notes/betb-bar-regrounding-part1-precommit.md).
Date: 2026-06-29.*

## Preamble (CLAUDE.md)

> **Active capability:** none reopened by this report — it is a characterization that *strengthens the
> banked compound-divergence wall* (Reports 146–150) by quantifying it, and bounds the headroom for the
> held exp100 entropy soft-prior.
> **Headline metric:** N/A (nothing is trained — no headline-with-CI; this is a drill-down/characterization,
> NOT a graduation experiment). Per CLAUDE.md the "experiment done" gates that apply are only (4) report
> exists + (5) STATUS updated.
> **Why now:** the 2026-06-22 cloud session ranked "re-ground the bar on ReCOGS/SLOG" #1, but no
> COGS/ReCOGS/SLOG data exists on this repo (a 3–5 day build). The covariate-shift HALF of that move is
> analyzable today on our own mcd1/2/3, and it answers — before any GPU run — whether the held exp100
> entropy soft-prior could beat the clause-aug floor.
> **Scope honesty:** this addresses ONLY the covariate-shift seam. The ReCOGS *decoder-artifact* seam
> (COGS logical-form string/length confound) does NOT exist on SCAN's flat-action exact-match and remains
> un-addressable here.

## Method

Reads `data/scan/mcd_split/{train,test}_{mcd1,mcd2,mcd3}`. Convention matches the project's existing
[analyze_mcd_divergence_axis.py](../../experiments/analyze_mcd_divergence_axis.py): primitive verbs =
{jump, walk, look, run}; a clause is templatized by abstracting its primitive to `V`; commands split at
the single top-level conjunction (and/after). Four measurements per split:

1. **Covariate-shift signature** — total-variation distance train→test of the empirical distribution at
   five structural scales (atom unigram → input bigram → clause-template → (prim, clause-template) →
   whole-command compound), with the count of test types never seen in train.
2. **Clause-aug headroom** — fraction of test commands whose clauses are all already provided (exact and
   template) by the 102 single-clause forms `clause-aug` (exp96) injects.
3. **Leakage of the 0.297 floor** — clause-aug is sourced from `tasks_train_simple` **+ `tasks_test_simple`**;
   count the forms present only via test_simple, and the MCD test commands handed verbatim.
4. **Template-level leakage invariant** — pre-register the audit any future augmentation must pass.

## Results (mcd1 / mcd2 / mcd3 — same pattern on all three)

**[1] Covariate-shift signature — divergence climbs monotonically with structural scale:**

| scale | TV (mcd1/2/3) | test-novel types (mcd1) | reading |
|---|---|---|---|
| atom unigram | 0.095 / 0.109 / 0.097 | **0 / 13** | atoms fully shared; only frequency reweighted |
| input bigram | 0.135 / 0.192 / 0.129 | 2 / 46 | local bigrams ~96–100% shared |
| clause-template | 0.938 / 0.865 / 0.937 | 12 / 17 | clause skeletons substantially diverge |
| (prim, clause-template) | 0.999 / 0.857 / 0.994 | 28 / 29 | which primitive fills which skeleton diverges |
| **whole-command compound** | **1.000 / 1.000 / 1.000** | **326 / 326** | **the compound is 100% disjoint** |

This is the textbook maximum-compound-divergence (Keysers 2020) signature: **atoms matched, compounds
maximally divergent.** MCD's hardness is *compound covariate-shift*, not atom OOV — names the wall
precisely and confirms the framing the 2026-06-22 R2 raised.

**[2] Clause-aug headroom bound — ZERO atom/clause headroom above the floor:**

| metric (all splits) | value |
|---|---|
| test cmds with ALL clauses EXACT-covered by clause-aug | **100.0%** |
| test cmds with ALL clause TEMPLATES covered | **100.0%** |
| per-clause exact / template covered | **100.0% / 100.0%** |

Every clause of every test command is *already* supplied by clause-aug. The residual MCD gap above the
floor is therefore **100% recombination of already-covered clauses** — exactly the compound-divergence
wall. **No atom/clause-level prior can close it**, because there is nothing at the atom/clause level left
to provide.

**[3] The 0.297 "atomic floor" is partly leaked:**

| leak metric | mcd1 / mcd2 / mcd3 |
|---|---|
| clause-aug forms present ONLY via `test_simple` | 18 / 18 / 18 |
| MCD test commands handed VERBATIM by clause-aug | **17 / 4 / 19** |
| per-clause gold-decode handed over | 100% / 100% / 100% |

`clause-aug` draws from `tasks_test_simple`, so single-clause MCD *test* commands are handed verbatim with
gold, and every test clause's gold decode is provided. So Report 149's "atomic injection ~doubles vanilla
(0.158→0.297)" is **partly leakage-inflated** — not a clean no-leakage floor.

**[4] Template-level leakage invariant (pre-registered for PART 2):** an augmentation example is safe iff
its whole-command clause-skeleton template matches **zero** held-out test template. clause-aug currently
violates this on 11 / 4 / 10 single-clause test commands (the verbatim leaks). PART 2 must enforce
the invariant in-code and source the floor from `train_simple` only.

## Interpretation & decision

- **The compound-covariate-shift characterization is the durable deliverable.** It quantifies the banked
  wall (atoms TV≈0.10/0 novel; compounds TV=1.000/100% novel) on all three splits — strengthening, not
  reopening, the bank.
- **exp100 (the entropy soft-prior) is predicted NULL by construction and is HELD per its own gate.** The
  precommit pre-committed: "run PART 2 only if PART 1 reveals real atom/clause headroom that clause-aug
  failed to supply." PART 1 reveals **zero** such headroom (100% clause coverage). A Wold-style entropy
  prior — even the genuinely-distinct fixed-budget occupancy *sweep* — reweights where already-present
  atoms/clauses appear; it cannot manufacture novel compound recombination, which is the entire residual
  gap. Running it would confirm-not-discover a structurally-foregone null and burn GPU to do so.
- **If exp100 is still run** (empirical confirmation of the predicted null has modest citable value), it
  MUST use the non-leaked train_simple-only floor, the fixed-token-budget occupancy sweep with the
  manipulation check, VOL/SHUF guards, the template-level leakage assertion, and `(ENT − ATOM)` as the
  sole headline vs the **published fragment-GECA ≈ 0.51** bar — per the PART 2 spec in the precommit.

## What this does NOT do

It does not re-ground the bar against the ReCOGS decoder-artifact (that confound is absent on SCAN's
flat-action exact-match) and it is a small-toy single-arena characterization. The genuinely cleaner arena
(ReCOGS/SLOG logical forms with semantic eval) remains a separate 3–5 day build.

## Provenance

`experiments/analyze_mcd_covariate_shift.py` (+ `.json`). Cross-checks: novel-atom=∅ and ~100% clause
coverage independently reproduce the 2026-06-29 seam-audit workflow (`wf_54a7cfe6-f9c`) findings; the
0.297 figure and clause-aug mechanism are from [Report 149](../149_betb_scan_mcd_lever1_verification/report.md)
§(b) and [exp96](../../experiments/96_betb_scan_mcd_oracle_ceiling.py):116–132.
