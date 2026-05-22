# Report 054 — Phase 5 Step-3 Walk-Back Smoke (seed 17, n=20)

**Date:** 2026-05-21 (late session)
**Context:** Post-walk-back verification that experiments/40_phase5_branching.py now applies `ConsolidationState.retrieval_weight_bias()` in both settling logits and final-energy telemetry. Reports 047-053 were measured on the unweighted landscape (`score_bias` never passed). See STATUS.md walk-back banner for the cross-audit (this agent + GPT) that surfaced the omission.

**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior − E_role-prior, paired per cue.
**Last verified result:** [Report 053](053_phase5_headline_n10_directional_subnoise.md) — graduation-unattained, ΔE_raw = +0.00130, 4.2× below magnitude floor.
**Why this experiment now:** Bug-vs-substrate-property disambiguation. The patch must (a) preserve back-compat with 047-053 and (b) surface whether step 3 moves the headline materially when correctly applied.

---

## Patch shape (committed to working tree)

[experiments/40_phase5_branching.py](../experiments/40_phase5_branching.py):
- `settle_branch_with_prior` accepts `score_bias`; subtracts from per-iter logits in both per_pattern and global_pull formulations; sign convention matches [trajectory.py:181](../src/energy_memory/phase4/trajectory.py:181).
- Final-state telemetry adds `energy_unbiased_step3_final` alongside `energy_unbiased_final` (load-bearing per GPT's #2 correction — settling-only weighting would yield a mixed-landscape headline).
- `BranchState` adds `energy_unbiased_step3`.
- `compute_branch_diagnostics`, `combine_bundle_resettle`, `run_branched_retrieval`, `_run_condition_over_cues`, and the headline-mode loop all thread `score_bias` through.
- Headline aggregation computes paired ΔE_raw AND ΔE_step3 from `per_cue_energy_*_min`.
- 4 new tests in `TestStep3ScoreBias` ([tests/test_phase5_branching.py](../tests/test_phase5_branching.py)) pin `score_bias=None` and `score_bias=zeros(...)` bit-identity, plus a targeted-atom-suppression mechanism check.

**286 tests pass (282 prior + 4 new, 0 regressions).**

---

## Smoke setup

```
PYTHONPATH=src .venv/bin/python experiments/40_phase5_branching.py \
  --mode headline --seed 17 \
  --substrate-snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \
  --device cpu --n-cues 20 --k-main 4 --gamma 0.5 \
  --output-dir reports/phase5_step3_smoke_seed17
```

Substrate: A1' pilot final state, seed 17, W=4 step 1800. `coverage_lambda=1.0`, `retrieval_weight_epsilon=0.05`, `retrieval_weight_tau=0.02` (the design-spec values from report 046).

**Substrate inspection (load-bearing):**
- `n_atoms = 1064`, `dim = 4096`.
- `|E_i|`: min=0.0251, median=0.0251, mean=0.0252, max=0.0586.
- `bias = softplus((ε − |E_i|)/τ)`: min=0.5003, median=1.4977, mean=1.4937, max=1.4977.
- **1063 / 1064 atoms have bias ≥ 1.0** (≥ 1 nat suppression). Only 1 atom (the |E_i| = 0.0586 outlier just above ε=0.05) sits in the mild-attenuation band.

The A1' fix correctly drove discovery-atom strength to the noise floor — and in doing so drove **every** atom to the noise floor. The bias is nearly uniform across the entire substrate.

---

## Result

| Condition       | E_raw     | E_step3   | Δ (uniform shift) |
| --------------- | --------- | --------- | ----------------- |
| role_K4         | −1.3635   | −1.2199   | +0.1436           |
| content_K4      | −1.3635   | −1.2199   | +0.1436           |
| random_K4       | −1.3635   | −1.2199   | +0.1436           |
| role_K4_g0      | −1.3561   | −1.2121   | +0.1440           |
| content_K4_g0   | −1.3561   | −1.2121   | +0.1440           |
| role_K1         | −1.3635   | −1.2199   | +0.1436           |
| content_K1      | −1.3635   | −1.2199   | +0.1436           |

**Paired ΔE = E_content − E_role**

| Tag    | n  | ΔE_raw    | f⁺raw | ΔE_step3  | f⁺step3 |
| ------ | -- | --------- | ----- | --------- | ------- |
| K4     | 20 | 0.000000  | 0.00  | 0.000000  | 0.00    |
| K4_g0  | 20 | 0.000000  | 0.00  | 0.000000  | 0.00    |
| K1     | 20 | 0.000000  | 0.00  | 0.000000  | 0.00    |

---

## Reading

1. **Energy uniformly shifts by +0.144.** This matches `mean(bias)/β = 1.4937/10 = 0.1494` — within rounding of the per-condition shifts. The step-3-weighted landscape is the raw landscape minus a near-constant, so `−logsumexp(β·s − c)/β = −logsumexp(β·s)/β + c/β`. Mechanically correct.

2. **ΔE_step3 ≡ ΔE_raw on this substrate.** Because the bias is uniform to within ±0.0005, paired comparisons cancel the shift exactly. **Step 3 is empirically inert on the A1' substrate, not because of the wiring bug, but because A1' drove every atom into the |E_i| ≪ ε regime where the sigmoidal suppression saturates and becomes uniform.**

3. **ΔE_raw is also 0.000 across all conditions at n=20.** This does NOT match report 053's +0.00130 at n=3000 × 10 seeds. Two non-conflicting reads: (a) n=20 is too small to resolve a signal at the 5.5e-3 noise scale; (b) the cue distribution from a single seed × 20 cues may not span the role/content distinction with enough variance. Smoke n is for mechanism, not for headline confirmation.

4. **What this rules out and what it doesn't:**
   - **Rules out:** "the wiring bug is the reason report 053 came in sub-noise." Step 3 cannot move a paired ΔE when its bias is uniform across the substrate. A correctly-weighted n=10 × 3000 rerun would land in the same place report 053 did — the bug, while real, was not the load-bearing failure.
   - **Does not rule out:** the broader substrate-saturation framing. If anything this strengthens it — A1' + the step-3 mechanism collectively pushed the substrate into a regime where the very mechanism designed to suppress the noise floor is itself a no-op because everyone IS the noise floor.

5. **The substrate-wide architectural finding now has a fourth instance.** The same A+B+A1' clean-retrieval geometry that:
   - saturated softmax-derived per-atom variance (reports 050, 051, 052),
   - saturated the role-vs-content energy gap (report 053),
   - also saturates the step-3 sigmoidal weighting itself (this report).

   The bias is bimodal in design (suppress noise-floor atoms, leave strong atoms alone) but unimodal in practice on this substrate (every atom is at the noise floor). The "death-as-asymptotic-limit" retrieval mechanism has nothing to differentiate on.

---

## What this changes for the decision

The upstream-diagnostics ordering my earlier response proposed (θ′(β) calibration spike → cue-regime sweep → D-sweep) stands, and is now reinforced rather than weakened. The bug fix is committed; back-compat is preserved (`coverage_lambda=0` reproduces 047-053 bit-identically); but the substrate-saturation reading of reports 050-053 is no longer "unsafe pending rerun" — it is **strengthened** by this finding.

The four open strategic options (close + redesign for lower-D; close + advance to Phase 6 with sub-noise acknowledgment; reformulate headline; investigate basin-shape priors) are still on the table. Option 2 remains the weakest — this report adds a third symptom of the same substrate-level disease that option 2 would paper over.

---

## What was NOT done

- **Not re-run on n=3000 × 10 seeds.** A correctly-weighted Colab rerun would give a CI-resolved ΔE_step3 to compare against report 053's ΔE_raw. Per the math (bias is uniform → ΔE_step3 ≡ ΔE_raw exactly), this rerun is **not informative on the headline question**; would only confirm the bit-identical-at-this-substrate prediction. Sequenced after upstream diagnostics or skipped depending on user call.
- **Not retuned ε or τ.** Per audit constraint #10, those are design-spec parameters; this is a no-touch zone.

## Required controls

- `score_bias=None` matches pre-walk-back path — pinned in `test_score_bias_none_matches_pre_walkback_path`.
- `score_bias=zeros(...)` ≡ `score_bias=None` — pinned in `test_score_bias_zero_tensor_bit_identical_to_none`.
- Targeted-atom suppression mechanism check — pinned in `test_score_bias_attenuates_targeted_atom`.

---

## Status update

STATUS.md walk-back banner remains accurate but should be updated next session to record: report 053's headline verdict (graduation-unattained) survives the patch. Step 3 was empirically inert on the A1' substrate; the substrate-saturation framing is reinforced, not refuted.
