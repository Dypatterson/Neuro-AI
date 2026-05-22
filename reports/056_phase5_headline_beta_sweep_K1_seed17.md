# Report 056 — Phase 5 Headline β Sweep K=1 (seed 17, n=200) — corrects report 055

**Date:** 2026-05-21 (late session)
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior − E_role-prior, paired per cue.
**Last verified result:** [Report 053](053_phase5_headline_n10_directional_subnoise.md), mean ΔE_raw = +0.00130 at K=1, γ=0.5, n=10 seeds × 300 cues.
**Why this experiment now:** GPT pointed out that [report 055](055_phase5_headline_beta_sweep_seed17.md) ran the β sweep at K=4, but report 053's headline was K=1. K=4 collapses the role/content distinction because all 4 branches can collectively reach the same global minimum regardless of prior source; K=1 only lets the prior's chosen schema settle. The two are not equivalent. **Report 055's "FP noise" reinterpretation of report 053 was therefore unsafe.** This experiment runs the exact K=1 condition.

This is a **drill-down**, not a graduation experiment.

---

## Setup

```
PYTHONPATH=src .venv/bin/python scripts/phase5_frozen_snapshot_audit.py \\
  --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \\
  --headline-beta-sweep --headline-n-cues 200 --headline-k-main 1 \\
  --output reports/phase5_audit/a1prime_seed17_headline_beta_K1.json
```

K_main=1, γ=0.5, formulation=per_pattern, binding_noise_std=0.05, content_distortion=0.6, cue_seed=117 (matches `args.seed + 100` from experiments/40 for seed 17). Three conditions per cue: role-prior, content-prior, random-prior. β ∈ {1, 3, 5, 10, 30}.

---

## Result

| β  | ΔE_raw     | f⁺_raw | ΔE_step3   | f⁺_step3 | E_role     | E_content  | E_random   | Ordering                  |
| -- | ---------- | ------ | ---------- | -------- | ---------- | ---------- | ---------- | ------------------------- |
| 1  | +0.000030  | 0.12   | +0.000027  | 0.13     | −7.40236   | −7.40233   | −7.40240   | random < role < content   |
| 3  | +0.000010  | 0.07   | +0.000008  | 0.07     | −2.78723   | −2.78722   | −2.78724   | random < role < content   |
| **5**  | **+0.000103** | **0.10**   | **+0.000120**  | **0.07**     | **−1.91751**   | **−1.91740**   | **−1.91716**   | **role < content < random**   |
| 10 | +0.000013  | 0.03   | +0.000013  | 0.03     | −1.33508   | −1.33506   | −1.33431   | role < content < random   |
| 30 | −0.000000  | 0.00   | −0.000000  | 0.00     | −1.00915   | −1.00915   | −1.00915   | content < random < role   |

Branch-softmax entropy is trivially 0 at K=1 (single branch). `max_w_proxy = 1.000` at β=30 (q_settled lands exactly on stored pattern); 0.99 across β≥3.

---

## Reading

1. **β=5 is the largest-signal cell at n=200, seed 17.** ΔE_raw = +0.000103, ordering role < content < random as the headline predicts. That's ~8× the β=10 magnitude (+0.000013).

2. **β=30 reverses ordering.** Content < random < role — the headline prediction inverts. With ΔE_raw ≈ 0 to 5dp, this is on the FP-noise floor, but it's not arbitrary direction-flipping at every β: low-β consistently has random as the lowest-energy condition; mid-β (5, 10) lands role < content < random as predicted; high-β tips to content < random < role.

3. **Magnitudes stay 50× below the 5.5e-3 magnitude floor.** Even the β=5 best at +0.000103 is 50× below the noise-scale floor from [2026-05-21 magnitude-floor note](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md). β does not pull the signal above the floor.

4. **Comparison to report 053.** Report 053's mean ΔE_raw = +0.00130 at K=1, γ=0.5, β=10, n=10 seeds × 300 cues. My seed-17-alone n=200 at β=10 gives +0.000013 — two orders of magnitude smaller. Possible explanations: (a) seed 17 is an unusually weak ΔE seed at this n, and the cross-seed mean is dominated by 1-2 high-ΔE seeds; (b) `randperm` ordering for cue pair selection differs between cue_seed=117 here and the headline run's cue generation; (c) sample variance — 200 cues at sub-noise magnitudes has wide CI. **Walking back report 055's claim that report 053 was "residual FP noise":** that claim was based on a K=4 sweep where the K=4 mechanism collapses the role/content signal by construction. K=1 actually does carry a small β-dependent signal; report 053's mean isn't refuted by this evidence.

5. **The K=4 collapse from [report 055] is its own finding, just not the one I claimed.** When K=4 branches all reach the same global minimum, the paired ΔE goes to exact zero — confirming that scaling K *kills* the role/content discrimination on this substrate. K=1 (and K=2, K=3 probably) is the necessary cell for the headline to even have a chance. The branch softmax saturating at log(2) in the K=4 sweep was a hint of this: 2 schemas hit the same attractor as 2 others.

---

## What this changes

- **Report 055's "β is irrelevant" claim was too strong.** β=5 shows ~8× larger headline magnitude than β=10 at K=1. Still 50× below the magnitude floor, so β=5 doesn't graduate the headline — but it IS the right operating point if there's a "best β" cell to investigate.
- **Report 055's "report 053 was FP noise" reinterpretation is rescinded.** Report 053's +0.00130 isn't on the floor, but it isn't obviously refuted by K=4 evidence — K=4 was a different question.
- **K=1 vs K=4 dichotomy is itself a finding.** Phase 5's design treats K_main=4 as the production cell; the freq-α experiment ([report 040](040_freq_weighted_alpha_sweep.md)) and the headline both ran with K=4 or 1 depending on condition. **At K≥4 the headline mechanism can never discriminate priors on this substrate** because the branches collectively span the schema store. Only K=1 (or maybe K=2) preserves the prior-dependence.

## What this does not change

- **Substrate-saturation framing still has multiple instances** ([reports 050, 051, 052, 053, 054]). What I added in [report 055] (β-axis flat at K=4) is now correctly read as "K=4 collapses the discrimination, not the substrate." The substrate-level saturation findings 050–054 are independent of this K dependence.
- **β=5 doesn't graduate the headline.** Even at the largest-signal cell, ΔE is 50× below the 5.5e-3 magnitude floor. The "β corner" hypothesis is partially salvaged (β=5 > β=10) but β alone cannot move the headline through the floor.
- **The four open strategic options remain unchanged.** Option 1 (lower-D), option 3 (reformulate headline), option 4 (basin priors) all still viable; option 2 (advance to Phase 6 sub-noise) still contraindicated.

---

## Updated next-step priority (per GPT's sequence + this correction)

GPT's sequence was: cross-seed audit → β sweep → cue sweep → D-sweep. The β sweep is now done correctly. β=5 is the slight winner. Next:

1. **Cross-seed audit of β=5 vs β=10** (needs Drive snapshots, or run the harness in the headline Colab notebook for the other 9 seeds). If β=5 shows consistent ~8× advantage over β=10 across seeds, the cross-seed mean ΔE at β=5 might still be sub-floor but worth knowing. If β=5 has high cross-seed variance, β isn't the right axis.
2. **Cue-regime aggregator** — still missing, still the natural next big test. Higher leverage than D-sweep, lower cost.
3. **D-sweep** — last.

## Required controls

- The random-prior condition serves as the per-β control. At β=5 it ranks lowest among the three (highest energy), as the headline predicts when the substrate has any structural signal. At β=1 and β=3, random ranks LOWEST energy (lowest = best), which is the wrong direction — confirms low β is in a "smoothed" regime where the random prior's diffuseness wins. At β=30 ordering tips toward content < random < role — confirms over-sharpened regime where the cue's intrinsic content alignment overwhelms the role prior.

## What was NOT done

- **No retuning of γ, K_main, formulation, content_distortion, or binding_noise_std.** Per audit constraint #10. The K=1 vs K=4 swap is *not* retuning — K=1 was already part of report 053's headline and the design spec.
- **No cross-seed verification.** Single-seed result; cross-seed is the next test.
