# Verified diagnosis synthesis (WF-1 output, refuter-checked)

> Source: workflow `neuro-ai-ground-and-brainstorm` (run wf_efb36586-c3a, 17 agents,
> ~2.1M tokens). Diagnosis verdict: **HOLDS WITH CORRECTIONS** — zero fabricated
> citations; the L=64 variance block was independently recomputed from raw JSON to
> an exact match with Report 117; the EBM-2006 root-cause was verified at the primary.
> Confidence: **medium**. Lean: **surgical-in-place** (rebuild-from-Phase-1 premature
> before the cheap write-then-read test).

## 1. The root cause (verified at primary, medium confidence)

The Phase-4 consolidation objective writes each consolidated atom as the **members
mean** (`reconstruction_learner.py` centroid) — a **push-down-only, zero-margin
write**. EBM-2006 (LeCun 2006, `pdf:energy-based-learning-2006`, opened: reader-p.10/24)
proves a push-down-only objective **can** collapse to a *flat zero-energy surface* when
nothing automatically pulls up competitor energies. With no contrastive / margin / swap
term, the consolidation step **never writes a role→target association**; the role-target
ends up sitting where a random atom sits.

Two independent lines converge on this same locus:
- **Phase 5 role-binding:** `hit_role = 0.000` **exactly** across four retrieval families
  (D1/D3/E1/M1; Reports 062–064), and **sharpening the representation made recovery
  monotonically worse** (D1 rank_role 205→403, E1 205→498, ΔE monotone-negative; 063:99-101)
  — the **anti-overlap signature** of an *absent* write, not a signal drowned by crosstalk.
- **Phase 3 Frame B:** consolidation **injects** variance (σ_A=0.157 vs frozen σ_C=0.069,
  2.30×, recomputed exactly), and the real−shuffle DiD can't cancel it (corr_AC_BD=0.10;
  Report 116). Landscape lever closed (Report 118).

## 2. Calibration — the diagnosis is one rung softer than the STATUS banner

The "never-written" verdict is reached by **elimination, not a positive write-then-read**.
The refuter folded in four corrections (none overturn the spine):

1. **The "95-point" top-1-vs-basin gap is borrowed from a *discarded* GHRR-confounded cell.**
   For the live FHRR readout the gap is **~7 points** (066:37). Basin-membership readout is
   still the right call, but the load-bearing number that rules out "written-but-not-top-1"
   is **rank_role ≈ 205–540 (mid-pack)**, not the dramatized 95.
2. **"Soft-TPR proves fixed-random roles are fine" is `link_only`** — not load-bearing per
   HARD RULE 3. It is *suggestive* that the roles aren't the bug, but must not *alone*
   exclude the role/binding-algebra locus, which Reports 065/066 keep live.
3. Minor: a stray "9 consolidation events" should read "~29" (phase3b).
4. **"the production objective IS that flat regime" is an analogy EBM-2006 does not prove**
   for FHRR/MHN — the write-then-read control is what would confirm it.

## 3. What is NOT excluded (the fork-deciders)

- **Binding algebra (deeper locus).** Reports 065/066: `bind(k,v)` has **no key-only basins
  for ANY operator**, and GHRR-native top_index_hits = 0/3072. The reports do **not**
  disambiguate "objective wrong" from "the binding algebra has no key basins." An
  objective-rewrite does not address this.
- **Capacity / crosstalk floor.** 065 key-only decays to chance with N; 058 leaves lower-D
  open; the Frame-B wall fired at β=10. Part of the null may be a 1/√D crosstalk floor
  fixable by **dimension**, not objective. → if so, GSBC (confined crosstalk) un-parks as a
  justified efficiency swap.
- **Eval design / metric.** smokes init from target atoms (bypass the schema-store selector,
  062 caveat); ΔE measures something real that is *not* role-target retrieval (058).

## 4. The "already tried" nuance (verified)

A push/pull contrastive write **was already built and run** (`error_driven_learner.py`,
lr_pull=0.1/lr_push=0.05) — phase3b: **Fail Rate 1.0** across ~29 events. BUT: **single
seed 17**, on **masked-token Q** (not hit_role), with a **self-mined `predicted_id`
negative** (a collapsing negative), and **top-1** readout. So "just add a contrastive term"
is *not untried* — but it was tried in the worst possible configuration. The surgical
re-run must change the negative (role-**swap**, not self-mined), the readout
(basin-membership), and the seed count (multi-seed).

## 5. Candidate structure-writing objectives (ranked + brainstorm additions)

All must be **batch-offline** (runtime error-driven updates are BANNED) and anti-homunculus
clean. Negatives must be **precommitted augmentation**, never runtime-arbitrated.

| # | Objective | Sources | AH-shape | Key risk |
|---|---|---|---|---|
| 1 | Contrastive / margin (push-down-correct + pull-up-offending) | `energy-based-learning-2006`, `dense-associative-memory-2016`, EqProp `1602.05179` | offline term | **already failed single-seed**; negative-construction in phasor space; EqProp wants symmetric real fixed points |
| 2 | Predictive / JEPA self-target (cue-derived, EMA/stop-grad) | `pam-2026`, `2502.05164`, DreamWeaver `2501.14174` | local dynamic | every PC/JEPA primary writes into an *external* Euclidean predictor; FHRR-native unbind-then-rebind self-target is the unproven port |
| 3 | Range/rectangular-support + separation pressure | `2410.06232` (Dorrell-Whittington), FEP `2505.22749` | offline | **both link_only**; nonnegativity vs signed FHRR; 068 deferred the training step, 111 closed a *different* (storage-side) variant |
| + | **Swap-reconstruction** (ArSyD/Soft-TPR/Dual-Swap) | `2412.19847`, `2412.04671`, `1805.10583` | offline | the FHRR-native form of the rank-1 negative; avoids the phase3b self-mined-negative trap |
| + | **FEP single-phase self-orthogonalizing** (anti-Hebbian = the missing margin) | `2505.22749` | local dynamic | single-phase defuses the EqProp symmetry risk; **empty falsification record here** (111/068 falsified a different mechanism) |
| + | **Mixture-prior EM write** (Probabilistic Slot Attention, von-Mises) | `2406.07141` | local dynamic | negatives-free; sidesteps phasor negative-construction; von-Mises port |
| + | **BTSP one-shot instructive write** (targets the binding-algebra locus) | bioRxiv 2025.05.15.654220, `PMC10484462` | local dynamic | the only candidate aimed at the 065/066 key-only-basin defect |

## 6. The cheapest decisive experiments (run-first gates)

- **G-A. Frozen-substrate refit-readout gate (run FIRST; one afternoon, zero new arch).**
  Run the *existing* centroid-mean consolidation on a 4-role toy, freeze it, then **refit a
  fresh basin-membership readout** on held-out role cues (`2310.05644`). **Refit recovers →
  the defect is the READOUT, not the objective** (flips rank-1, re-opens GSBC). **Refit
  fails → structure was truly never written.** This is the read-half of write-then-read the
  project has never executed.
- **G-B. phase3b A/B with ONE surgical change:** swap-negative + basin-membership readout,
  multi-seed, vs the existing Fail-Rate-1.0 baseline. Cheapest adjudicator of the
  "already-tried" risk.
- **G-C. Lower-D / dense-AM control arm:** every write rule run at current-D and lower-D.
  This is the **force-rebuild adjudicator**: null across contrastive AND predictive families
  AND lower-D → rebuild warranted; lower-D rescues → defect was capacity → GSBC un-parks.
- **G-D. Selectivity-Δ as the ONE Stage-1 headline:** Δ = recoverability(true-role cue) −
  recoverability(role-shuffled cue), basin-membership, Wilson CI **strictly > 0**, with
  no-negatives and random-codebook ablations collapsing Δ→0 (Hewitt-Liang control tasks
  `1909.03368`).

## 7. The surgical-vs-rebuild fork

**Lean: surgical-in-place.** Rebuild-from-Phase-1 is **premature** before Stage-1.
- Stage-1 toy writes a recoverable role basin (Δ>0) → integrate the winning objective into
  Phase 3/4 consolidation; the existing phase ladder mostly stands.
- Stage-1 null across contrastive **and** predictive families **and** the lower-D arm →
  rebuild-from-Phase-1 becomes warranted (the defect is substrate/algebra-deep, not
  objective-shaped).
- Lower-D rescues recovery → the defect is **capacity**; fix is dimension/orthogonalization
  (and GSBC un-parks as an efficiency swap), not the objective.

## 8. Findings to surface (per the CLAUDE.md contradiction rule)

- **STATUS may be stale on the range lever.** STATUS says the range-shaped-replay downstream
  lane is "closed" (Report 111), but **111 closed a storage-side post-settle variant**, NOT
  the Dorrell-Whittington *training-objective* variant (consolidate under a nonneg+energy
  objective on rectangular support), which **Report 068 explicitly deferred**. This mirrors
  the documented freq-weighted-α walk-back. → the training-objective variant is **open**, not
  closed-by-association.
- **The project's eval doctrine is uncarded.** Every source in the eval row of the literature
  matrix (`2312.04927` MQAR, `2406.03980`, `2503.23390`, `2507.11393`) is `link_only` and
  uncarded. There is no carded write-then-read eval methodology yet.
- **Phase-5′ stays PAUSED.** The Stage-1 toy is legitimately *pre-Phase-5′* and does not
  require lifting the fence. Targeting the paused ΔE headline directly **does** require
  lifting it (and clearing the `raw_scene_energy_v0`/`min_branch` audit fence first).

## 9. Open user decisions (from the diagnosis)

1. Run the Stage-1 write-then-read toy (the direct test) before any rebuild, or close on
   elimination?
2. Does Report 111's "novelty-that-didn't-convert" re-open a *readout/retrieval-geometry*
   reading (which would un-park GSBC)?
3. Add a lower-D arm to Stage-1 to separate objective-never-wrote-it from D=4096-crosstalk?
4. Card the link-only flashy candidates (Dorrell `2410.06232`, FEP `2505.22749`) before
   leaning on them?
5. Is the range-shaped-replay *training-objective* variant open, or closed-by-association?
6. Scope = the consolidation OBJECTIVE, or does the 065/066 binding-algebra wall warrant the
   deeper substrate question first (changing surgical vs rebuild-from-1)?
