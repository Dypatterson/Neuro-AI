# Report 026 — Phase 4 verification against the design-spec headlines (supersedes 025)

**Date:** 2026-05-14
**Status:** Supersedes [reports/025_rt0p85_5seed_verification.md](025_rt0p85_5seed_verification.md), which framed the result against the wrong headline metric (top1 instead of design-spec Recall@K + cap-coverage) and missed several systematic issues uncovered in the post-25 scope check.

**Raw data:**
- `reports/phase4_rt0p85/phase4_unified_results.json` (seed 17)
- `reports/phase4_rt0p85_seed{11,23,1,2}/phase4_unified_results.json`
- `reports/phase4_rt0p85_random_cb/phase4_unified_results.json` (control)
- `reports/phase4_drift_0p{15,30,50}/phase4_unified_results.json` (drift sweep)

**Specs cross-referenced:**
- [notes/emergent-codebook/phase-4-unified-design.md](../notes/emergent-codebook/phase-4-unified-design.md) — Phase 4 design + headline metric + required controls
- [notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md](../notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md) — headline metric principle
- [docs/PROJECT_PLAN.md](../docs/PROJECT_PLAN.md) — Phase 4 exit criteria

## Headline result

Against the design-spec headline (Δ Recall@K on masked-token contextual
completion with active drift):

- **Δ Recall@10 at step 2000: +0.010, 95% CI [+0.001, +0.022], 5/5 seeds non-negative.**
- **Random-codebook control: 0 candidates discovered, Δ = 0.000 exactly at every checkpoint.**

This passes both design-spec requirements: (a) the headline metric
moves in the predicted direction with CI disjoint from zero, and (b)
the mechanism produces no effect when codebook structure is absent.
**This is the first verified positive Phase 4 result in the project.**

The second design-spec headline (Δ cap-coverage) is **not verified**:
no checkpoint has Δ cap_t05 CI disjoint from zero, and per-seed
behavior splits qualitatively (some seeds get +6 pp, others -10 pp).

## Headline metrics (5 seeds, bootstrap 95% CI, rt=0.85, drift=0.30, β=10)

### Δ Recall@10 (primary headline)

| step | per-seed                                | mean   | 95% CI            | non-neg |
| ---: | --------------------------------------- | -----: | ----------------- | ------: |
|  500 | {-0.024, -0.007, -0.013, +0.037, +0.000} | -0.001 | [-0.017, +0.019]  | 2/5     |
| 1000 | {+0.003, +0.000, -0.013, +0.010, +0.003} | +0.001 | [-0.007, +0.007]  | 4/5     |
| 1500 | {+0.027, +0.000, -0.007, +0.017, +0.003} | +0.008 | [-0.002, +0.020]  | 4/5     |
| **2000** | {+0.031, +0.000, +0.003, +0.017, +0.000} | **+0.010** | **[+0.001, +0.022]** | **5/5** |

The lift **grows through the run** and clears the CI bar at step 2000.
All 5 seeds non-negative. This is the opposite-shaped trajectory from
top1's "fade by step 2000" — report 025's main framing was an artifact
of the wrong metric.

### Δ cap-coverage @ τ=0.5 (secondary headline)

| step | per-seed                                | mean   | 95% CI            | positive |
| ---: | --------------------------------------- | -----: | ----------------- | -------: |
|  500 | {-0.007, -0.011, +0.003, +0.027, -0.038} | -0.005 | [-0.024, +0.013]  | 2/5      |
| 1000 | {+0.000, -0.007, -0.070, +0.061, -0.080} | -0.019 | [-0.062, +0.023]  | 1/5      |
| 1500 | {+0.017, -0.014, -0.099, +0.067, -0.073} | -0.020 | [-0.072, +0.031]  | 2/5      |
| 2000 | {+0.010, +0.000, -0.040, +0.044, -0.035} | -0.004 | [-0.030, +0.022]  | 2/5      |

**Δ cap-coverage is not robust.** All CIs cross zero, means slightly
negative, per-seed variance ~5× larger than Δ Recall@K's. Phase 4
does *not* reliably preserve confidence-weighted retrieval at this
drift.

### Random-codebook control (design-spec required)

Single seed (17), same config except codebook swapped for the
unlearned `phase2_full_matrix/phase2_codebook.pt` random codebook:

| step | baseline R@10 | phase4 R@10 | ΔR@10 | Δtop1 | Δcap_t05 |
| ---: | ------------: | ----------: | ----: | ----: | -------: |
|    0 | 0.024 | 0.024 | **+0.000** | +0.000 | +0.000 |
|  500 | 0.024 | 0.024 | **+0.000** | +0.000 | +0.000 |
| 1000 | 0.058 | 0.058 | **+0.000** | +0.000 | +0.000 |
| 1500 | 0.058 | 0.058 | **+0.000** | +0.000 | +0.000 |
| 2000 | 0.061 | 0.061 | **+0.000** | +0.000 | +0.000 |

**Phase 4 produced zero candidates at all three scales (W=2, W=3, W=4)
across all 2000 cues.** Deltas are exactly zero at every checkpoint
on every metric. The mechanism is structurally inert without codebook
structure — exactly what the design ([phase-4-unified-design.md:320](notes/emergent-codebook/phase-4-unified-design.md)) requires.

This control rules out: corpus-statistical artifacts, drift-magnitude
artifacts, and replay-mechanism-as-noise artifacts. The +0.010 ΔR@K
with learned codebook is causally tied to learned codebook structure.

## Drill-downs (5-seed aggregates)

### Multi-scale candidate production

The per-checkpoint display in exp 18 prints W=2 stats only. The
W=3 and W=4 channels were not reported in prior session findings.
Aggregated across 5 seeds, candidates at step 2000:

| scale | mean ± std | per-seed                       |
| :---: | ---------: | :----------------------------- |
| W=2   | 9.0 ± 11.1 | {31, 5, 3, 1, 5}               |
| W=3   | 56.4 ± 27.2 | {107, 32, 33, 55, 55}         |
| W=4   | **179.6 ± 97.7** | {154, 3, 280, 213, 248} |

**Three important corrections to the session's narrative:**

1. **W=2's "30 candidates per checkpoint" was a seed-17 phenomenon.**
   Mean across 5 seeds is 9.0 ± 11.1; seed 17 was the only outlier.
   Every prior report this session that quoted "30 candidates" was
   reporting a single-seed result.

2. **W=4 is the actual workhorse of multi-scale Phase 4** — by far
   the most productive discovery channel. But also the highest-variance:
   seed 11 produced 3 candidates, seed 23 produced 280. Two orders of
   magnitude across seeds.

3. **W=3 saturates fast** — all 5 seeds reach their final candidate
   count between step 500 and step 1000, then stop discovering. W=3
   has a tight effective ceiling.

### Engagement / entropy over time

Project plan exit criterion: *"repeated solution paths become faster
and lower entropy."*

Per-scale mean entropy (5-seed mean ± std):

| step | W=2 baseline | W=2 phase4 | W=2 Δ | W=4 baseline | W=4 phase4 | W=4 Δ |
| ---: | -----------: | ---------: | ----: | -----------: | ---------: | ----: |
|    0 | 0.717±0.008  | 0.717±0.008 | 0.000 | 0.734±0.027 | 0.734±0.027 | 0.000 |
|  500 | 0.723±0.017  | 0.710±0.009 | -0.013 | 0.734±0.027 | 0.660±0.058 | **-0.074** |
| 1000 | 0.705±0.010  | 0.707±0.008 | +0.002 | 0.723±0.033 | 0.690±0.035 | -0.032 |
| 1500 | 0.700±0.008  | 0.703±0.005 | +0.002 | 0.713±0.033 | 0.724±0.035 | +0.011 |
| 2000 | 0.698±0.008  | 0.700±0.006 | +0.003 | 0.697±0.034 | 0.754±0.018 | **+0.057** |

- **At W=2: entropy is flat.** Δ ≈ +0.003 throughout. Project plan
  exit criterion not met at this scale.
- **At W=4: non-monotonic.** Phase 4 sharpens retrieval early
  (step 500-1000), then crosses over and ends with *higher* entropy
  than baseline. The discovered W=4 patterns (180 of them on average)
  add basins to the landscape rather than sharpening existing ones —
  by step 2000 they're contributing competing pull on otherwise-
  resolvable retrievals.

**The "faster, lower entropy" exit criterion is not met.** The
mechanism does the opposite: by end of run, retrieval at W=4 is
*more* ambiguous with Phase 4 than without. This is a real
architectural finding, not just a measurement gap.

### u_k variable distribution (Benna-Fusi graduation check)

Per the design spec, "u_k variable distributions should show graduated
consolidation." Seed 17 W=2 mean_u trajectory:

| step | u_1 | u_2 | u_3 | u_4 | u_5 | u_6 | mean strength |
| ---: | --: | --: | --: | --: | --: | --: | ------------: |
|    0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
|  500 | 0.058 | 0.088 | 0.084 | 0.060 | 0.032 | 0.013 | 0.133 |
| 1000 | 0.025 | 0.041 | 0.047 | 0.043 | 0.032 | 0.017 | 0.065 |
| 1500 | 0.015 | 0.025 | 0.029 | 0.028 | 0.022 | 0.012 | 0.039 |
| 2000 | 0.010 | 0.016 | 0.018 | 0.018 | 0.014 | 0.008 | 0.026 |

Initial spike at u_1 (novelty_strength=1.0) diffuses across the chain
into a bump centered on u_3/u_4. **This matches the design's expected
graduated-consolidation pattern.** ✓ This drill-down is healthy.

### Pattern death — mechanism is structurally broken

Across all 5 seeds, all 3 scales, all 5 checkpoints: **0 patterns
have died and 0 patterns are below the death_threshold.** This is not
a sample-size issue — the mechanism cannot fire under any settings we
explored. Two separate problems:

**Problem 1: death_window units are consolidation-steps, not cues.**
With `replay_every=50`, a 2000-cue run produces only 40 consolidation
steps. `death_window=1000` (my "10× faster pruning" setting from
report 022) requires 1000 *consecutive* sub-threshold steps =
~50,000 cues. Even the default `death_window=100` requires ~5,000
cues. Death is structurally impossible in any practical run length.

**Problem 2: death_threshold (0.005) is below the chain's equilibrium
strength.** With α=0.25, m=6, observed mean_strength converges to
~0.026 — five times above the threshold. Active patterns (those
getting replay reinforcement) never drop below the floor. Even with
death_window=1, no pattern would qualify for death under this
threshold.

**Implication:** every Phase 4 result this session — including the
verified +0.010 ΔR@K — was produced *without* the death mechanism
firing. The "pattern replacement" story I told in earlier reports
(report 022's "death and discovery balance") was wrong. There is no
death; n_patterns only grows. The Δ Recall@K we measured is from
*pure accumulation* of discovered patterns, not turnover.

This is an architectural bug in current usage, not a result-
invalidating bug. It just means the project hasn't actually exercised
the design's pattern-death dynamic yet.

### Drift sweep summary (from report 023, multi-seed not run)

Single seed (17) across drift ∈ {0.15, 0.30, 0.50}:

| drift | baseline R@10 (step 2000) | phase4 R@10 | ΔR@10 |
| ----: | -----------------------: | ----------: | ----: |
| 0.15  | (already computed below) | | (re-run needed) |
| 0.30  | 0.310                    | 0.341       | +0.031 |
| 0.50  | (need recompute)         | | (re-run needed) |

The drift sweep was framed against Δtop1 in report 023. Should be
recomputed against ΔR@K for consistency with the design-spec headline.
(Cheap — the data is in the JSON files.)

## What the project plan exit criteria say now

From [docs/PROJECT_PLAN.md](../docs/PROJECT_PLAN.md):

| Exit criterion | Status |
| --- | --- |
| Replayed trajectories improve future retrieval | **✓** Verified +0.010 ΔR@10, CI disjoint, random-codebook control passes |
| Stale atom drift can be corrected without global rewrites | **partial** — reencode_every=100 runs but its specific contribution isn't ablated |
| Repeated solution paths become faster and lower entropy | **✗** Phase 4 entropy *increases* at W=4 by step 2000 |

One of three exit criteria fully met, one partial, one failed.

## What we missed in the session and have now resolved

| # | Missed thing | How resolved |
| --: | --- | --- |
| 1 | Reported top1 as headline instead of design-spec R@K | Re-aggregated; ΔR@K +0.010 CI disjoint at step 2000 |
| 2 | "30 candidates / checkpoint" was seed-17-only at W=2 | Multi-seed: W=2 mean 9.0 ± 11.1, W=4 actually 179.6 ± 97.7 |
| 3 | Random-codebook control absent | Run; 0 candidates, Δ = 0.000 at every checkpoint (passes) |
| 4 | Entropy drill-down never aggregated | Aggregated; exit criterion *not* met at W=4 late in run |
| 5 | u_k graduation never inspected | Inspected; consistent with Benna-Fusi expectation ✓ |
| 6 | death_window mechanism assumed functional | Confirmed structurally vacuous (units + threshold both wrong) |
| 7 | Drift sweep framed against wrong metric | Acknowledged; recomputation against ΔR@K is the cheap follow-up |

## What this does NOT settle

- **drift=0.30 only.** rt=0.85 hasn't been 5-seed tested at drift=0.15
  (likely no-gap regime) or drift=0.50 (severe degradation, single-
  seed only).
- **Single drift trajectory per seed.** Substrate variance and drift-
  realization variance are entangled.
- **Per-seed W=4 split (3 to 280 candidates) is not understood.** Some
  upstream substrate-state variable splits the seeds; never identified.
- **Entropy increase at W=4 by step 2000 is not understood.** Likely
  related to W=4 producing 180 candidates that crowd the basin
  structure, but mechanism not verified.
- **Death mechanism not tested.** Either need much longer runs
  (>50,000 cues), or retune death_window/threshold so pruning actually
  fires within run length.
- **The Phase 3 multi-seed shuffled-token control inherited from the
  05-09 framework hasn't been applied at Phase 4.** Random-codebook
  is the Phase 4 design-spec control; shuffled-token is a Phase 3
  discipline that the Phase 4 design doesn't separately require.

## Anti-homunculus / FEP audit

- All knobs touched this session (`resolve_threshold`, `death_window`,
  `reencode_every`, `store_threshold`) are constants on geometric
  diagnostics, not inspect-and-trigger controllers. ✓
- The death-mechanism bug is a tuning issue, not an architectural
  homunculus introduction. ✓
- u_k chain dynamics are pure Benna-Fusi diffusion; no supervisory
  routing introduced. ✓

Passes cleanly.

## Recommended next steps (re-prioritized)

### Tier 1 — cheap and high-information

1. **Recompute drift sweep against ΔR@K.** Existing JSON has the data;
   just re-extract. Frame all Phase 4 results against the design-spec
   headline. ~5 min.

2. **Fix death mechanism settings and rerun rt=0.85 5-seed.** Tune
   `death_threshold` to ≈ 0.05 (above equilibrium mean_strength) and
   `death_window` to ≤ 20 (so death is possible in 40-step runs).
   Tests whether actual pattern-turnover changes the result. ~90 min.

3. **rt=0.85 at drift=0.15, 5 seeds.** Confirms no-gap regime
   prediction; cheap robustness check. ~90 min.

### Tier 2 — required for Phase 4 to fully clear exit criteria

4. **Investigate the W=4 seed-variance split.** Why do some seeds
   produce 3 candidates and others 280? Likely candidates: initial
   landscape coverage of validation distribution, cue-stream
   composition. ~1-2 h instrumentation.

5. **Investigate the W=4 entropy-rise-by-step-2000.** Does it predict
   Phase 4 hurting retrieval if runs continue past 2000 cues?
   Longer-window single-seed run (n_cues=5000). ~30 min.

### Deferred (not blocking Phase 4 conclusion)

6. **Shuffled-token control.** Phase 3 discipline; Phase 4 design
   doesn't separately require. Worth running before claiming Phase 4
   "done" but not before the Tier 1/2 items.

7. **Codebook geometry follow-up** (Spisak-Friston direction).
   Report 021 closed this line; reopen only if a Phase 5 limit
   clearly traces back to atom collapse.

## One-line takeaway

Against the design-spec headline (Δ Recall@K), Phase 4 is verified
positive at +0.010 (CI [+0.001, +0.022]) at step 2000 across 5 seeds,
with random-codebook control producing exactly 0 candidates and 0
deltas — but the mechanism is doing so without any pattern death
firing, without sharpening retrieval entropy (which actually grows
at W=4), and with enormous seed variance concentrated at the most
productive scale (W=4: 3 to 280 candidates across seeds).
