# Frame B pre-build smoke — corrected plan + harness notes (DRAFT)

**Date:** 2026-05-29 · DRAFT, nothing run, no spec edit.
**Supersedes** the earlier `04-smoke-harness-DRAFT.py` sketch, which imported
`_build_substrate` / `_make_world` from `gate0_frame_a.py` — **those functions
do not exist** (fabricated; caught on a clean read). The real primitive is
`c3._run_single_seed_condition(...)`.

## Key discovery: half the smoke is already committed code
Reading `experiments/gate0_frame_a.py` end-to-end shows the variance question
behind grill items 3/6/7 is already implemented:

- **`variance_report_from_summary(summary)`** (gate0:566) → `did_sd`,
  `single_arm_binomial_floor`, `did_sd_over_binomial_floor`, **`corr_AC_BD`**,
  `n_for_80pct_power_at_effect`. (Uses `_corr`, gate0:553.)
- **`run_variance_decomposition(atom_seeds, window_seeds, ...)`** (gate0:668) →
  nested ANOVA splitting σ(lift) into **atom-draw / window-draw / binomial**,
  with a `recommendation` that *already* says (gate0:776-784): when atom-draw
  dominates, "change the estimand to an exposure–recall SLOPE (Frame B) which
  pools within-seed and is higher-SNR."

So the reframe's premise was *motivated* by this decomposition — but the
decomposition was run (per the audit) only at a non-slope operating point, and
`corr_AC_BD` is the ENDPOINT analog of the smoke's `corr(intercept, slope)`,
not the slope itself.

## Two-stage smoke (both BEFORE the full build, per user ruling)

### Stage 1 — zero new code, run now
Run / re-use Gate 0's own variance tools on the corrected operating point:
```
# (a) decompose the lift variance on HOLDOUT pilot seeds (item 6 fix: not 0..9)
.venv/bin/python experiments/gate0_frame_a.py --variance-decomp \
    --atom-seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 \
    --window-seeds 0,1,2,3 --device <cuda|mps> --corpus-source wikitext
# (b) if a gate0_summary.json exists, decompose its DiD variance (no re-run)
.venv/bin/python experiments/gate0_frame_a.py --variance-report \
    reports/<gate0_run>/gate0_summary.json
```
**Reads on the reframe:**
- `var_fraction_atom` high + `corr_AC_BD` low → per-seed luck is structural AND
  the worlds' lifts do NOT move together → within-pair differencing won't
  cancel it → **the slope reframe likely buys little power** (item 3 fires at
  the endpoint level already). 
- `corr_AC_BD` high → differencing cancels seed variance → reframe is promising;
  proceed to Stage 2 to confirm it survives at the slope level.

This stage can FALSIFY the pivot cheaply with code that already exists. It does
NOT settle slope-specific gates (G2 corr(intercept,slope), G3 fanning, G4
β_shuffle) — those need checkpoints.

### Stage 2 — requires the checkpoint hook (the one piece of build code)
`_run_single_seed_condition` scores only at the END of consolidation. The
slope/intercept/fanning gates need Recall@K at the corrected exposure
checkpoints mid-stream. That hook (defect-02 item-9 axis: lower anchor ≥100)
is the minimal reusable core of the Frame B build itself — so Stage 2 is
"build the hook, nothing else, and read it before committing to the full rig."

Per-(pilot seed, world): fit OLS slope+intercept of Recall@K on log(exposure)
at the corrected checkpoints, then:
- **G1** σ(Δβ) ≤ 0.05 vs σ_level 0.196
- **G2** |corr(intercept_real, β_real)| ≤ 0.40   *(user-approved)*
- **G3** slope of [r_real(e)−r_shuffle(e)] vs log e > 0 with CI excluding 0
- **G4** mean β_shuffle CI excludes 0 (positive)
Build the full Frame B rig only if Stage 1 didn't falsify AND G1–G4 pass.

## Corrected checkpoint schedule (defect 02, item 9)
All anchors ≥ consolidation_k=100 so none sits on the frozen intercept:
`{100, 178, 316, 562, 1000}` (~even in log10; span = log10(1000/100) = 1 decade
= 2.303 e-folds). NOTE this changes the floor span again — the floor
recalibration (defect 02-b) must use the FINAL agreed schedule's e-fold span.
Open question for you: is a 2.3-e-fold span enough resolution, or extend e_max
above 1000 to widen it? (Wider span = more consolidations = the schedule needs
`n_observations` > 1000.)

## Signature checklist before ANY run (learned from the fabricated imports)
- `c3._run_single_seed_condition` — confirm full kwarg list (gate0:210-238 shows
  the real call; copy it verbatim).
- `c3._load_wikitext_corpus(repo_root, wikitext_name, vocab_cap)` — gate0:189.
- `c3._delta_ci_stats`, `gate0._overall_recall`, `gate0._corr` — exist, confirmed.
- the checkpoint hook does NOT exist yet — must be written for Stage 2.
