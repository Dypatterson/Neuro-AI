# Report 058 — Phase 5 Cross-Seed Cue-Regime Sweep K=1, β=10 (n=10 seeds × 24 cells × 100 cues)

**Date:** 2026-05-23
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior − E_role-prior, paired per cue. (Headline reused from report 057; this experiment is a **drill-down**, not a graduation attempt.)
**Required controls per [phase-5-unified-design.md:285-293](../notes/emergent-codebook/phase-5-unified-design.md):** random-prior condition per cue (catches "any prior helps" pathology); K=1 (the operating point report 057 established as the β optimum). No new graduation criteria — the magnitude floor pre-commitment from the [2026-05-21 magnitude-floor note](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md) (`mean ΔE ≥ 5.5e-3 AND CI lower > 0`) remains binding and remains the only graduation gate.
**Last verified result:** [Report 057](057_phase5_cross_seed_beta_sweep_K1.md) — β=10 cross-seed mean ΔE = +0.002289, CI [+0.00109, +0.00349], 9/10 seeds positive. Signal statistically significant but 0.42× the 5.5e-3 magnitude floor; β-axis closed. Report 057 closed with: "1/10 ordering rate + 9/10 sign-positive ΔE points to 'different metric reads out same structure' (e.g., R@K for role-target, basin-membership probability)."
**Why this experiment now:** The cue-regime sweep was the last load-bearing pre-phase-graduation drill-down per the STATUS.md sequence ("Cue-regime aggregator is now next; D-sweep stays last"). Its purpose: factor the substrate's failure mode along two orthogonal axes (paired-ΔE vs basin-membership) and test report 057's "different metric reads out same structure" hypothesis directly. The grid varies cue construction at fixed (β=10, γ=0.5, K=1) per the locked scope; cue-regime is **diagnostic of the measurement surface**, not a tunable for graduation.

This is a **drill-down**, not a graduation experiment. No cell, including any above the magnitude floor, can graduate Phase 5 or justify post-hoc operating-point selection.

Run via [scripts/colab_phase5_cross_seed_cue_sweep.ipynb](../scripts/colab_phase5_cross_seed_cue_sweep.ipynb), commit [e3cad90](https://github.com/Dypatterson/Neuro-AI/commit/e3cad90).

---

## Setup

**Substrate set:** 10 A1' snapshots — `phase3_phase4_w4_step1800.pt` for seeds {1, 2, 3, 5, 7, 11, 13, 17, 23, 29}. Same set used in reports 053 and 057.

**Locked scope (no retuning):** β=10, γ=0.5, K=1, formulation=per_pattern. β=10 per report 057; γ=0.5 + K=1 per design spec headline.

**Grid (24 cells):**
- `binding_noise_std ∈ {0.01, 0.05, 0.10, 0.20}` — perturbation magnitude on the role-binding axis
- `content_distortion ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` — fraction of cue content replaced with random noise

**Per cell:** n_cues = 100 (cue_seed = 117, fixed across cells so cross-cell variance is pure substrate response to cue-regime, not cue-seed variance).

**Conditions per cue:** role-prior, content-prior, random-prior (the required random control per design spec).

**Diagnostics per cell (cross-seed aggregated):** paired mean ΔE_raw with 95% CI + seeds-positive count + ΔE/floor ratio; ordering counts (`role<content<random`, `random_lowest`); per-condition basin-membership (`hit_role` = role-target = argmax-similarity; `rank_role` = 1-indexed similarity rank of role-target out of 1064 atoms).

**Execution:** sequential per seed on T4 via the [Drive-resumable Colab path](../scripts/colab_phase5_cross_seed_cue_sweep.ipynb). Optimized profile cell at 20.9s confirmed harness was not the bottleneck; full sweep ran 26.0 min wall-clock to completion. Each seed JSON validated and atomically published to Drive before proceeding.

---

## Results

### Full cell table (ordered by mean ΔE_raw)

| bns  | cd   | mean ΔE_raw | 95% CI                 | seeds⁺ | ΔE/floor | role<c<r | random_lowest | hit_role | rank_role |
|------|------|-------------|------------------------|--------|----------|----------|---------------|----------|-----------|
| 0.05 | 0.60 | +0.002478   | [+0.00113, +0.00383]   | 9/10   | +0.4505  | 0.07     | 0.34          | 0.00     | 462.1     |
| 0.01 | 0.60 | +0.002460   | [+0.00110, +0.00382]   | 9/10   | +0.4472  | 0.07     | 0.34          | 0.00     | 462.1     |
| 0.20 | 0.60 | +0.002431   | [+0.00117, +0.00370]   | 9/10   | +0.4420  | 0.07     | 0.34          | 0.00     | 462.2     |
| 0.10 | 0.60 | +0.002424   | [+0.00108, +0.00376]   | 9/10   | +0.4407  | 0.07     | 0.34          | 0.00     | 462.3     |
| 0.01 | 0.80 | +0.002176   | [+0.00098, +0.00337]   | 9/10   | +0.3956  | 0.10     | 0.30          | 0.00     | 502.3     |
| 0.10 | 1.00 | +0.002156   | [+0.00079, +0.00352]   | 8/10   | +0.3921  | 0.10     | 0.30          | 0.00     | 503.9     |
| 0.20 | 1.00 | +0.002152   | [+0.00079, +0.00352]   | 8/10   | +0.3912  | 0.10     | 0.30          | 0.00     | 503.9     |
| 0.01 | 1.00 | +0.002149   | [+0.00078, +0.00352]   | 8/10   | +0.3907  | 0.10     | 0.30          | 0.00     | 503.9     |
| 0.05 | 1.00 | +0.002148   | [+0.00078, +0.00352]   | 8/10   | +0.3906  | 0.10     | 0.30          | 0.00     | 503.9     |
| 0.10 | 0.80 | +0.002093   | [+0.00092, +0.00327]   | 9/10   | +0.3805  | 0.10     | 0.30          | 0.00     | 501.8     |
| 0.20 | 0.80 | +0.001989   | [+0.00099, +0.00299]   | 9/10   | +0.3617  | 0.10     | 0.30          | 0.00     | 501.4     |
| 0.05 | 0.80 | +0.001964   | [+0.00093, +0.00300]   | 8/10   | +0.3571  | 0.10     | 0.30          | 0.00     | 501.3     |
| 0.05 | 0.40 | +0.000012   | [-0.00090, +0.00092]   | 7/10   | +0.0022  | 0.03     | 0.43          | 0.01     | 290.2     |
| 0.01 | 0.40 | +0.000008   | [-0.00091, +0.00093]   | 7/10   | +0.0014  | 0.03     | 0.43          | 0.01     | 290.0     |
| 0.20 | 0.40 | +0.000001   | [-0.00093, +0.00094]   | 6/10   | +0.0002  | 0.03     | 0.43          | 0.01     | 290.3     |
| 0.10 | 0.40 | -0.000026   | [-0.00097, +0.00092]   | 6/10   | -0.0048  | 0.03     | 0.43          | 0.01     | 290.1     |
| 0.05 | 0.00 | -0.000107   | [-0.00041, +0.00019]   | 5/10   | -0.0195  | 0.01     | 0.48          | 0.02     | 263.7     |
| 0.01 | 0.00 | -0.000109   | [-0.00041, +0.00019]   | 4/10   | -0.0198  | 0.01     | 0.48          | 0.02     | 263.7     |
| 0.20 | 0.20 | -0.000117   | [-0.00067, +0.00043]   | 6/10   | -0.0213  | 0.01     | 0.46          | 0.01     | 268.4     |
| 0.10 | 0.00 | -0.000118   | [-0.00042, +0.00018]   | 5/10   | -0.0214  | 0.01     | 0.48          | 0.02     | 263.8     |
| 0.20 | 0.00 | -0.000122   | [-0.00041, +0.00017]   | 5/10   | -0.0222  | 0.00     | 0.48          | 0.02     | 263.9     |
| 0.05 | 0.20 | -0.000150   | [-0.00072, +0.00042]   | 5/10   | -0.0272  | 0.01     | 0.46          | 0.01     | 267.9     |
| 0.01 | 0.20 | -0.000159   | [-0.00070, +0.00038]   | 4/10   | -0.0289  | 0.01     | 0.46          | 0.01     | 267.9     |
| 0.10 | 0.20 | -0.000164   | [-0.00071, +0.00039]   | 4/10   | -0.0298  | 0.01     | 0.46          | 0.01     | 268.0     |

Magnitude floor = 5.5e-3 per [2026-05-21 magnitude-floor note](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md). ΔE/floor < 1.0 = below floor. 95% CIs are t-approximation (df=9), matching the [scripts/aggregate_cue_sweep.py](../scripts/aggregate_cue_sweep.py) convention and report 057's. The magnitude-floor pre-commit note specifies "bootstrap 95% CI lower bound"; t-approximation at n=10 is the closest analytical proxy used across the cross-seed reports (057, 058) and is the operational test. The pre-commit gate (`CI lower > 0 AND mean ≥ 5.5e-3`) is evaluated on this t-approx CI here; no cell satisfies the second conjunct regardless of CI method.

### Drill-down extrema

- **Best ΔE cell:** bns=0.05, cd=0.6 → ΔE_raw = +0.002478, ΔE/floor = +0.4505, **hit_role = 0.00**, rank_role = 462.1
- **Best basin-hit cell:** bns=0.01, cd=0.0 → hit_role = 0.02, **ΔE_raw = -0.000109** (negative), rank_role = 263.7
- **Best (lowest) rank cell:** bns=0.05, cd=0.0 → rank_role = 263.7, hit_role = 0.02, **ΔE_raw = -0.000107** (negative)

**No cell crosses the 5.5e-3 magnitude floor.** Best magnitude is 0.45× the floor.

---

## Reading

### 1. ΔE is a pure function of content distortion; binding noise is invisible.

Within every cd row, the four bns values produce essentially identical mean ΔE:
- cd=0.6: {+0.002460, +0.002478, +0.002424, +0.002431} across bns ∈ {0.01, 0.05, 0.10, 0.20}
- cd=1.0: {+0.002149, +0.002148, +0.002156, +0.002152}
- cd=0.0: {-0.000109, -0.000107, -0.000118, -0.000122}

The grid factored cleanly: cd controls everything; bns is noise on top. This is consistent with K=1's structure — the role-prior shapes settling toward the schema; cue binding-noise affects which cue arrives but not how the prior interacts with the substrate.

### 2. A content-distortion phase transition lives between cd=0.4 and cd=0.6.

ΔE structure across cd (averaging over bns):
- cd=0.0: −0.0001 (negative; 4–5/10 seeds positive)
- cd=0.2: −0.0001 (negative; 4–6/10 seeds positive)
- cd=0.4: ≈ 0 (6–7/10 seeds positive — null zone)
- cd=0.6: +0.0024 (9/10 seeds positive — best magnitude)
- cd=0.8: +0.0020 (8–9/10 seeds positive)
- cd=1.0: +0.0021 (8/10 seeds positive)

Mechanistic reading: at low cd, the cue's content carries enough information that the substrate retrieves the right schema without role-prior help, so role-prior provides no differential lift. At high cd, content is degraded enough that the role-prior's hint becomes informative. The transition is sharp and lives around cd ≈ 0.4–0.6.

### 3. ΔE and basin-hit are *anti-correlated* across the cd axis.

This is the central finding and the killer for report 057's "different metric reads out same structure" hypothesis.

| cd  | mean ΔE   | hit_role | rank_role | seeds⁺ |
|-----|-----------|----------|-----------|--------|
| 0.0 | −0.0001   | 0.02     | 263.7     | 4–5/10 |
| 0.2 | −0.0001   | 0.01     | 267.9     | 4–6/10 |
| 0.4 | ≈ 0       | 0.01     | 290.1     | 6–7/10 |
| 0.6 | +0.0024   | **0.00** | 462.1     | 9/10   |
| 0.8 | +0.0020   | **0.00** | 501.4     | 8–9/10 |
| 1.0 | +0.0021   | **0.00** | 503.9     | 8/10   |

The two metrics move in opposite directions on the cd axis. ΔE+ region (cd≥0.6) has zero role-target argmax hits and rank ~462–504 out of 1064. Best basin-hit cells (cd=0.0) are in the ΔE− region.

**No single cell carries both signals.** Whatever paired ΔE is measuring in the cd≥0.6 region is *not* the role-target basin. Whatever the role-target basin lookup is finding in the cd=0 region is *not* a positive ΔE-on-priors.

### 4. Even the best basin-hit cell places role-target far from top-1.

The best rank_role across the entire grid is 263.7 — out of 1064 atoms. That's the top 25%, not the top-1 or top-K. The hit_role = 0.02 means the role-target is argmax in 2% of cues. Even rounding generously, no plausible R@K reformulation of the headline can claim "the system retrieves the role target" — K would need to be ≥ 264 for basin coverage to reach 50%, which is absurd as a "retrieval works" claim.

### 5. Random-prior dominates across the grid.

`random_lowest` ranges from 30% (cd ≥ 0.6) to 48% (cd ≤ 0.2). The random-prior condition produces the lowest energy in roughly a third of cues even at the best ΔE cells, and in nearly half of cues at the basin-hit-favorable cells. This is the same pathology report 057 flagged at the cross-seed level; it persists across the entire cue-regime grid. **Random-prior outperforms both role-prior and content-prior more often than role-prior wins outright.**

The mean ΔE > 0 with 9/10 sign-positive in the cd≥0.6 region means role-prior beats content-prior on average even though random-prior often beats both. The headline mechanism IS discriminating role from content; it just doesn't dominate the unbiased competition.

### 6. Headline ordering (`role < content < random`) is rare across the entire grid.

Maximum `role<c<r` rate is 10% (at cd ≥ 0.8). At the best ΔE cell (cd=0.6), only 7% of cues exhibit the headline ordering. The cross-seed pattern from report 057 (1/10 seeds with the predicted ordering at β=10) generalizes across the cue grid: the predicted three-way ordering is a minority event even where paired ΔE is positive.

---

## Strategic implications

### Option ranking update

| Option | Pre-058 status | Post-058 status |
|--------|----------------|-----------------|
| **1. Close + lower-D redesign** | Leading per 057 | **Strengthened.** Now the only path with a plausible mechanism. |
| **2. Sub-floor advance to Phase 6** | Contraindicated | Still contraindicated. Real ΔE at 0.45× floor + zero basin hits is the exact cell the floor was designed to prevent. |
| **3. Headline reformulation (basin/R@K)** | Leading per 057's "different metric reads out same structure" hint | **Weakened, near-foreclosed.** The hypothesis is empirically falsified: ΔE and basin metrics are anti-correlated; no cell carries both. Best basin-hit cell still places role-target at rank 264/1064. |
| **4. Basin-shape priors** | Unchanged per 057 | **Weakened.** With basin hit ≈ 0 and rank 264–504 across the grid, there is no exploitable role-target basin geometry for a shaped prior to bias toward. |

### Substrate-saturation finding count

Pre-058: five instances at the same root cause (A+B+A1' clean-retrieval geometry forecloses softmax-derived per-atom variance, role-vs-content energy gap, step-3 sigmoidal suppression mechanism, β-axis discrimination at K≥4, prior-source discrimination at K≥4).

**Post-058: sixth instance** — at the K=1 β=10 operating point report 057 established as optimal, the substrate's directional ΔE signal lives at content_distortion ≥ 0.6 (a *high-corruption* cue regime), and the same substrate carries no role-target basin structure (rank 264+ out of 1064 across the grid). The substrate carries a directional energy signal of the predicted sign, but the signal does not correspond to role-target basin retrieval at any cue regime. **The headline measures something real; that something is not "retrieves the role target."**

### What this does not change

- The magnitude floor is unchanged. No cell graduates Phase 5.
- The directional ΔE finding from report 057 is reconfirmed (cd=0.6 cells reproduce +0.002 mean, 9/10 seeds positive).
- The substrate is not corrupted or pathological — it is doing what it was designed to do (sharp self-retrieving basins). The Phase 5 question was whether *role-binding cues at moderate K and γ* could traverse the substrate's geometry to recover role-target schemas. The answer is no, at this D=4096 substrate; whether it would be yes at lower D is the open question.

---

## Recommendation for next session

The substrate-side diagnostic chain is now exhausted at D=4096. With options 3 and 4 effectively foreclosed by this drill-down, the live strategic question collapses to:

1. **Lower-D redesign (option 1):** the only path with a plausible mechanism. Would require a re-scoped Phase 5 design that targets a smaller substrate dimension where role-binding can be measured. Significant scope: D-sweep diagnostic spike first to characterize the dim-dependence of the failure modes, then redesign.

2. **Close Phase 5 as graduation-unattained:** acknowledge the substrate-saturation finding (now six instances), document the strategic constraint (architectural incompatibility between sharp-basin self-retrieving substrates and role-binding-prior-traversal at D=4096), and move on.

These are not mutually exclusive — the D-sweep is informative regardless of which closure direction is chosen. The D-sweep was already on the deferred queue ("D-sweep stays last" per STATUS.md); this report makes it the next actionable diagnostic.

**No implementation commitment in-session.** Strategic-direction decision is the user's call.

---

## Audit trail

- Run config locked in commit [e3cad90](https://github.com/Dypatterson/Neuro-AI/commit/e3cad90); harness optimizations validated by 20.9s profile cell on T4.
- Aggregator wording (no post-hoc cell selection) per commit [19132e9](https://github.com/Dypatterson/Neuro-AI/commit/19132e9).
- 292 tests pass at e3cad90; equivalence pin for K=1 q_settled ≡ q_bundle on sharp-basin substrate at `tests/test_phase5_branching.py::test_k1_q_settled_equiv_q_bundle_on_sharp_basin_substrate`.
- Audit constraint #10 (no retuning of κ/μ_obs/β/γ/K_main/formulation post-hoc) preserved: this drill-down operates at the report-057-confirmed β=10, K=1, γ=0.5 operating point; cd and bns are the per-design measurement axes, not graduation tunables.
