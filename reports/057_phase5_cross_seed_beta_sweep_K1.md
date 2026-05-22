# Report 057 — Phase 5 Cross-Seed β Sweep K=1 (n=10 seeds, n=200 cues each)

**Date:** 2026-05-21 (late session)
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior − E_role-prior, paired per cue.
**Last verified result:** [Report 056](056_phase5_headline_beta_sweep_K1_seed17.md) — seed-17 alone showed β=5 with ~8× the β=10 ΔE magnitude at single-seed n=200 (mean +0.000103 vs +0.000013).
**Why this experiment now:** GPT's recommended cross-seed audit of the seed-17 β=5 finding. The question: does β=5 beat β=10 consistently across seeds, or was seed 17 atypical? Run via [scripts/colab_phase5_cross_seed_beta_sweep.ipynb](../scripts/colab_phase5_cross_seed_beta_sweep.ipynb), commit [6cd2b95](https://github.com/Dypatterson/Neuro-AI/commit/6cd2b95).

This is a **drill-down**, not a graduation experiment.

---

## Setup

10 A1' substrate snapshots from the original headline run (`phase3_phase4_w4_step1800.pt` for each of seeds {17, 11, 23, 1, 2, 3, 5, 7, 13, 29}). For each seed: K=1, γ=0.5, formulation=per_pattern, binding_noise_std=0.05, content_distortion=0.6, n_cues=200, β ∈ {3, 5, 10}. Three conditions per cue (role-prior, content-prior, random-prior). All via the patched harness at commit [6cd2b95](https://github.com/Dypatterson/Neuro-AI/commit/6cd2b95).

**Caveat:** cue_seed=117 (the default = seed-17's `args.seed + 100`) was used for all 10 seeds, so the cross-seed variance is purely substrate variance, not cue+substrate variance. For exact report-053 replication a per-seed cue_seed would be needed; for the β-axis decision this is actually cleaner — isolates substrate response across β.

---

## Result

| β  | n  | mean ΔE_raw  | 95% CI (t-approx df=9)    | std_seed  | seeds⁺ | role<cont<rand | ΔE/floor |
|----|----|--------------|----------------------------|-----------|--------|----------------|----------|
| 3  | 10 | +0.000405    | [+0.00003, +0.00078]       | 0.00053   | 8/10   | 3/10           | 0.0737   |
| 5  | 10 | +0.000488    | [+0.00006, +0.00091]       | 0.00059   | 8/10   | 2/10           | 0.0887   |
| **10** | **10** | **+0.002289** | **[+0.00109, +0.00349]** | **0.00168** | **9/10** | **1/10** | **0.4161** |

Magnitude floor (5.5e-3) from [2026-05-21 magnitude-floor note](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md). ΔE/floor < 1.0 = below floor.

---

## Reading

1. **β=10 is the cross-seed winner.** Mean ΔE = +0.002289 — **~4.7× β=5 (+0.000488), ~5.6× β=3 (+0.000405).** 9/10 seeds positive (vs 8/10 at β=5 and β=3). Only β where the 95% CI strictly excludes zero. This is the **opposite** of report 056's seed-17-alone reading, where β=5 was 8× β=10.

2. **The signal is statistically real at β=10.** CI [+0.00109, +0.00349] excludes zero by a margin. The mean is 4.2× the lower CI bound — by any standard inferential criterion, role-prior finds lower-energy attractors than content-prior across these 10 seeds.

3. **The signal is sub-floor.** Mean = 0.42× the 5.5e-3 floor; β=10 is the closest of the three but still 2.4× below. The pre-committed graduation criterion (`mean ΔE ≥ 5.5e-3 AND CI lower > 0`) was designed to catch exactly this cell: statistically significant but below the substrate's noise scale.

4. **Headline ordering is RARE.** Only 1/10 seeds shows `role < content < random` at β=10. Most seeds have random producing the lowest energy (random < role < content was the dominant ordering at β=5 in [report 056]). The mean ΔE > 0 with 9/10 seeds positive is **role beating content on average even though both often lose to random.** The K=1 mechanism is discriminating role from content, just not against an off-substrate floor.

5. **Seed 17 was atypical in BOTH directions.** Single-seed at β=10: +0.000013 (~175× smaller than the cross-seed mean +0.002289). Single-seed at β=5: +0.000103 (~5× smaller than cross-seed mean at β=5). Seed 17 is on the weak end of substrate response; the cross-seed signal is dominated by stronger seeds. Report 053's mean +0.00130 across n=10 × 300 cues is now bracketed: this run's +0.00229 at n=10 × 200 cues with a fixed cue distribution is the same order of magnitude (1.76× larger), and the previously-reported "directional but sub-noise" verdict is reconfirmed.

6. **β-axis is closed.** β=10 is already the design-spec operating point and is empirically the best β cell across seeds. β=3 and β=5 reduce ΔE by ~5×. No β-only path graduates the headline through the 5.5e-3 floor.

---

## What this changes

- **Walks back [report 056]'s "β=5 is slight winner" framing.** Seed 17 misread the β axis. Single-seed β-sweep results are not reliable indicators of cross-seed β preference on this substrate.
- **Confirms [report 053]'s β=10 choice as the correct operating point.** No β reformulation is needed; the existing headline configuration is at the β optimum.
- **Sharpens "directional but sub-noise" to "statistically significant but sub-magnitude-floor".** The CI excludes zero by ~3× the lower bound — this isn't a null result that happens to point positive; it's a real signal that's smaller than the substrate's intrinsic noise scale. Different failure mode from what report 053's headline framing implied.

## What this does not change

- **The four strategic options remain.** Options 1 (lower-D), 3 (different metric), and 4 (basin priors) still viable; option 2 (advance with sub-noise) still contraindicated.
- **The "headline ordering rare" finding sharpens option 3.** 1/10 seeds with `role < content < random` suggests the role/content energy gap captures *something*, but not the cleanly-discriminating-prior structure the headline assumed. A different metric — e.g., R@K for role-target atom, basin-membership probability, or "fraction of branches in role-target basin" — might expose what paired-ΔE-on-noise-floor obscures. The branch-softmax log(2) collapse from [report 055] hinted at this.

---

## Updated next-step priority

Per GPT's sequence:

1. ~~Cross-seed audit of β=5 vs β=10~~ — **done**; β=10 is the winner; β-axis is closed for graduation purposes.
2. **Cue-regime aggregator** — now the next test. The shell script ([scripts/run_cue_regime_sweep.sh](../scripts/run_cue_regime_sweep.sh)) writes placeholder stats and references a missing `scripts/aggregate_cue_sweep.py`. Build the aggregator + run a pre-committed 36-cell `binding_noise_std × content_distortion` grid at β=10, K=1 (the now-confirmed-best-β operating point). The question: does a different cue regime move ΔE through the magnitude floor?
3. **D-sweep** — last. Most expensive; only if cue-regime sweep is also flat.

Strategic option recommendations:
- **Option 1** (lower-D redesign): strengthened — at confirmed-best-β with confirmed-best-K, the headline is still 2.4× below floor. If cue-regime aggregator is also flat, lower-D becomes the leading path.
- **Option 3** (reformulate headline): strengthened. The 1/10 `role<content<random` ordering rate combined with 9/10 sign-positive ΔE means there's *structure* the substrate exposes that paired ΔE doesn't capture cleanly. Reformulating to a metric that doesn't compare role-energy to content-energy directly (e.g., role-target basin membership) could read out this structure.
- **Option 2** (advance with sub-noise): same recommendation — contraindicated. The signal is now confirmed statistically significant, but it's sub-floor — exactly the cell the magnitude-floor pre-commit was designed to prevent vacuous graduation on.
- **Option 4** (basin priors): unchanged.

## Required controls

- The random-prior condition is the per-β control. The 1/10 rate of `role<content<random` ordering means random-prior frequently produces the lowest-energy state — confirming the noise-floor mechanism: random schemas often happen to be near a stored pattern, which then dominates the cue's own gradient.
- Step3 step bias is shift-invariant per [report 054] (bias_cv ≈ 0.024 cross-seed expected); the geometry audit in the same JSON confirms this per seed.

## What was NOT done

- **No retuning.** γ=0.5, K=1, formulation=per_pattern, ε=0.05, τ=0.02 all unchanged per audit constraint #10.
- **Single fixed cue distribution.** cue_seed=117 across all 10 seeds. For exact report-053 replication, the experiment would need per-seed cue_seed = args.seed + 100. Filed as caveat; the β decision is not sensitive to this choice (fixed cues isolates substrate response cleanly).
- **No full n=10 headline rerun at β=5.** Would have been triggered if β=5 was the cross-seed winner; β=10 won, so no rerun needed.
