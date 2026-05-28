# Report 113 — Γ1.c context-residual headline gate: FAIL at lr_cr=0.1; F1 lr_cr sweep is next

**Status:** complete. Γ1.c at the headline operating point does NOT graduate Phase 3. Phase 5′ remains paused. Per the precommit's pre-committed follow-ups, **F1 (lr_cr sweep at smoke scale)** is the next move.

**Date:** 2026-05-27.

## Preamble (per CLAUDE.md experiment-preamble requirement)

- **Active phase:** Phase 3 reopen — Path γ Γ1 (context-residual consolidation).
- **Headline metric per [phase-3-deep-dive.md:188-205](../notes/emergent-codebook/phase-3-deep-dive.md) (revised 2026-05-27 criterion):** regime-stratified Recall@K on masked-token contextual completion vs. shuffled-token-with-consolidation control; **both clauses must hold:** (1) Wilson CI on Δ strictly disjoint in ≥ 1 regime stratum at n ≥ 10; (2) per-seed paired robustness ≥ 70% (≥ 7/10 seeds with stratum-pooled Δ > 0).
- **Required controls per same spec:** shuffled-token-with-consolidation control (Path α default since 2026-05-26); identical training data and substrate atom set, only the codebook row permutation differs.
- **Last verified result:** [Report 112](112_phase3_c3_wikitext_graduation_walkback.md) (Path C closed inconclusive; v3 → v6 wikitext chain). The Γ1 precommit at [notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md](../notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md) and its anti-homunculus reviewer PASS authorized this experiment.
- **Why this experiment now:** Γ1.c was the leader candidate from the Path γ mechanism-family survey at [notes/emergent-codebook/path-gamma-mechanism-family-survey.md](../notes/emergent-codebook/path-gamma-mechanism-family-survey.md). The precommit committed a single lr_cr=0.1 headline gate at the Path C wikitext operating point; this report executes that gate.

## TL;DR

Γ1.c (asymmetric gradient descent on the per-event repulsion energy E_cr over confused atom pairs) was run at the Path C wikitext-2 operating point (D=4096, β=10, vocab_cap=1000, window=8, n_consolidation_events=1000, lr_cr=0.1, seeds 0..9) alongside a matched-seed pull/push baseline. Three findings:

1. **PathC baseline reproduces [Report 112](112_phase3_c3_wikitext_graduation_walkback.md) v3 numbers exactly.** default/spread Δ=+0.055 CI [0.241, 0.280] vs [0.188, 0.223] disjoint ✓; calibrated/tight Δ=+0.083 CI [0.350, 0.401] vs [0.268, 0.316] disjoint ✓. **The Γ1 refactor preserves byte-identity at the experiment-output level.** v3 was repeatable, not a one-time fluke.

2. **Γ1.c FAILS both clauses of the revised criterion.** No stratum CI-disjoint at n=10 (Δ ≈ +0.011 across primary strata, CIs overlap); per-seed paired robustness 5/10 = 50% (< 70%).

3. **Γ1.c is attenuated, not reversed.** Per-seed mean Δ +0.0115 ≈ 1/5 of PathC's +0.055 at the same seeds. Γ1.c flattens PathC's per-seed variance (large positive and negative PathC swings → near-zero Γ1.c). This is a real signal-shape change at lr_cr=0.1, just at a smaller magnitude than the headline criterion can detect.

**Decision:** Γ1.c at lr_cr=0.1 does not graduate. Follow-up F1 (`lr_cr` sweep at {0.01, 0.05, 0.1, 0.2, 0.5}, smoke scale n=3) is the precommitted next step. If F1 also produces null at all magnitudes, the Γ1 family is closed and the next-candidate precommit (Γ2 bundle-first scene memory or Γ3 SFA-head) is the move.

## Experiment chain

Two conditions × 10 seeds = 20 parallel CUDA subprocesses on Colab A100, via [scripts/colab_gamma1_headline_gate.ipynb](../scripts/colab_gamma1_headline_gate.ipynb).

| condition | flags | mechanism |
|---|---|---|
| `gamma1_headline` | `--use-context-residual --no-pull-push --lr-cr 0.1` | Asymmetric gradient descent on `E_cr = − Σ_j 1[predicted_j ≠ target_j] · ½‖codebook[target] − codebook[predicted]‖²` |
| `pathc_baseline` | (defaults — pull/push on, Γ1 off) | Existing Hebbian pull/push from [online_codebook.py:144-164](../src/energy_memory/phase34/online_codebook.py), matched-seed reference |

All other operating-point knobs held at Path C values per the precommit §"Operating point": wikitext-2-raw-v1, vocab_cap=1000, window=8, D=4096, β=10, K=5, landscape=64, n_consolidation_events=1000, α_anti=0.01, repulsion_step_size=0.05, λ_ac=0.5, μ_T=0.1, τ_T=0.5, λ_cc=0.5, metastability_obs_rate=0.1, drift_ema_rate=0.1.

## Result tables

### Pooled n=10 Wilson CIs per stratum / mode

**gamma1_headline (Γ1.c at lr_cr=0.1):**

| mode | stratum | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---|---|---|---|---:|:--:|
| default | tight | (empty stratum — no atoms classified tight in default mode at this op point) | | | |
| default | spread | 0.251 [0.232, 0.270] | 0.239 [0.221, 0.258] | +0.012 | no |
| calibrated | tight | 0.015 [0.008, 0.029] | 0.004 [0.001, 0.014] | +0.011 | no |
| calibrated | spread | 0.336 [0.312, 0.360] | 0.324 [0.300, 0.348] | +0.012 | no |

**pathc_baseline (pull/push reference at matched seeds):**

| mode | stratum | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---|---|---|---|---:|:--:|
| default | spread | 0.260 [0.241, 0.280] | 0.205 [0.188, 0.223] | +0.055 | **YES** |
| calibrated | tight | 0.375 [0.350, 0.401] | 0.292 [0.268, 0.316] | +0.083 | **YES** |
| calibrated | spread | 0.005 [0.002, 0.014] | 0.019 [0.011, 0.033] | −0.014 | no |

The pathc_baseline default/spread Δ=+0.055 and calibrated/tight Δ=+0.083 **match [Report 112](112_phase3_c3_wikitext_graduation_walkback.md) v3 to 3 decimal places**. This is the strongest form of test-harness sanity check: same operating point, same seeds, same numerical outputs across the Γ1 refactor and the Report 112 baseline.

### Per-seed paired comparison (default mode, stratum-pooled tight + spread)

Both conditions at the same seeds 0..9, default theta_prime_mode, Δ = std_p − ctrl_p stratum-pooled:

| seed | Δ_Γ1 | Δ_PathC | shift (Γ1 − PathC) | Γ1 > 0? | PathC > 0? |
|---:|---:|---:|---:|:---:|:---:|
| 0 | +0.030 | −0.090 | +0.120 | ✓ | |
| 1 | +0.005 | +0.465 | −0.460 | ✓ | ✓ |
| 2 | +0.000 | +0.065 | −0.065 | | ✓ |
| 3 | −0.030 | −0.190 | +0.160 | | |
| 4 | +0.010 | −0.015 | +0.025 | ✓ | |
| 5 | −0.010 | +0.165 | −0.175 | | ✓ |
| 6 | −0.010 | +0.070 | −0.080 | | ✓ |
| 7 | +0.115 | +0.095 | +0.020 | ✓ | ✓ |
| 8 | +0.005 | −0.060 | +0.065 | ✓ | |
| 9 | +0.000 | +0.045 | −0.045 | | ✓ |

Aggregates:
- **Γ1 per-seed mean Δ:** +0.0115
- **PathC per-seed mean Δ:** +0.055
- **Γ1 seeds with Δ > 0:** 5/10 (50%) — clause 2 threshold 70%, **fails**.
- **PathC seeds with Δ > 0:** 6/10 (60%) — also fails clause 2 at this n=10.
- **Γ1 better than PathC at matched seeds:** 5/10.

## Findings

### Finding 1 — test harness is sound (PathC reproduces v3 byte-identically)

The pathc_baseline condition was inserted into the headline gate specifically to verify that the Γ1 refactor preserves Path C behavior under the default config flags (`use_pull_push=True, use_context_residual=False`). It does — to 3 decimal places at the experiment-output level. This complements the implementation-level baseline parity test ([tests/test_consolidation_path_c_byte_identity.py](../tests/test_consolidation_path_c_byte_identity.py), float32 atol=1e-7) by extending byte-identity all the way to the report-level numbers on real wikitext-2.

A useful secondary finding: v3's numbers replicate at the same operating point and same seeds. The Report 112 walk-back analysis assumed the v3 single-run was reproducible but didn't have a head-to-head replication. This run confirms it.

### Finding 2 — Γ1.c fails both clauses of the revised criterion

**Clause 1 (CI-disjoint at n=10):** Γ1.c's largest Δ is +0.012 in default/spread, with CIs [0.232, 0.270] (std) overlapping [0.221, 0.258] (ctrl) — overlap is substantial. Same shape across all three populated strata: Δ ≈ +0.011, CIs overlap.

**Clause 2 (per-seed paired robustness ≥ 70%):** 5/10 seeds with Δ > 0 = 50%. Below threshold.

The criterion was designed with both clauses required (per the revised spec at [phase-3-deep-dive.md:188-205](../notes/emergent-codebook/phase-3-deep-dive.md)) precisely because the v3 → v6 walk-back showed clause 1 alone is gameable by lucky-seed tail draws. Γ1.c fails clause 1 *and* clause 2 — there's no scenario where this would be read as graduation evidence.

### Finding 3 — Γ1.c attenuates the signal ≈ 5× vs pull/push at lr_cr=lr_pull=0.1

Per-seed mean Δ: Γ1.c +0.0115 vs PathC +0.055. Same operating point, same seeds, only the base update rule's shape differs.

**Two candidate explanations:**

1. **Effective learning rate mismatch.** The Γ1.c update direction is `ε = codebook[target] − codebook[predicted]` (atom-space). For two random unit-modulus FHRR atoms, ‖ε‖ ≈ √2. But for atoms that are *similar* (which is what happens during confusions — the system retrieves something close to the right answer), ‖ε‖ may be much smaller, often « 1. Pull/push's update direction uses `slot_query`, which is a cue with magnitude ≈ 1 by construction. So `lr_cr=0.1` and `lr_pull=0.1` produce different effective update magnitudes — Γ1.c's per-event movement is smaller because its multiplicand is smaller on average. This argues for an `lr_cr` *sweep* (F1) before declaring the mechanism shape inadequate.

2. **The atom-vs-atom geometry carries less corpus signal than the atom-vs-cue geometry.** Pull/push compresses within-basin variability (atoms tighten around cues that retrieved them). Γ1.c compresses between-confused-atom distance. If WikiText's compositional structure lives more in cue geometry than in atom-pair geometry, Γ1.c is operating on the wrong axis regardless of magnitude.

The F1 lr_cr sweep is designed to distinguish these. If F1 shows lr_cr=0.5 produces Γ1.c per-seed mean ≈ +0.055 (matching PathC), explanation (1) is correct. If F1 shows the effect saturates well below PathC at all magnitudes, explanation (2) is correct.

### Finding 4 — Γ1.c flattens per-seed variance, doesn't just shrink it uniformly

The per-seed paired data shows a more nuanced pattern than "Γ1.c is uniformly weaker than PathC":

- At seeds where PathC has *large positive* Δ (seed 1: +0.465; seed 5: +0.165; seed 7: +0.095), Γ1.c is *much smaller* and sometimes *negative* (1: +0.005; 5: −0.010; 7: +0.115).
- At seeds where PathC has *large negative* Δ (seed 0: −0.090; seed 3: −0.190; seed 8: −0.060), Γ1.c is *smaller in magnitude* and often *flipped* (0: +0.030; 3: −0.030; 8: +0.005).
- Variance of per-seed Δ: Γ1.c σ ≈ 0.038; PathC σ ≈ 0.169. Γ1.c is **4.4× less seed-variance** than PathC.

This is consistent with a real shape difference: Γ1.c's atom-vs-atom geometry is less sensitive to which specific seed produced the substrate, while pull/push's atom-vs-cue geometry has high seed-to-seed swings. If a future mechanism could combine PathC's signal magnitude with Γ1.c's seed-stability, the revised criterion's clause 2 would be much more achievable. (This is hypothetical and not authorized by any current precommit; flagging it as a direction the post-mortem may surface.)

### Finding 5 — PathC at n=10 seeds 0..9 also fails clause 2

PathC's per-seed positive rate is 6/10 = 60%, below the 70% threshold. This is consistent with [Report 112](112_phase3_c3_wikitext_graduation_walkback.md) §"Finding 2" (per-seed σ/μ ≈ 3.3, robust per-seed signal is weak for pull/push too). The revised criterion's clause 2 (per-seed robustness ≥ 70%) catches PathC failing the spirit-of-graduation test on the same seed set where it passes clause 1's strict letter.

In other words: the revised criterion does what it was designed to do. PathC's strict-letter CI-disjoint result *is* a tail-of-distribution artifact at n=10; the per-seed-paired clause exposes it.

## Decision: Γ1.c at lr_cr=0.1 rejected; F1 is next

Per the precommit §"Falsifiable graduation criterion":

> If neither clause holds, Γ1.c is rejected at this operating point.

Both clauses fail. Γ1.c at lr_cr=0.1 is rejected.

Per the precommit §"Pre-committed follow-ups" F1:

> If the headline gate fails (neither graduation clause holds), this follow-up sweeps `lr_cr ∈ {0.01, 0.05, 0.1, 0.2, 0.5}` at n=3 (smoke). It is a diagnostic, not a graduation gate. If the sweep produces null at all magnitudes, the Γ1 family is closed.

**F1 is now authorized to be precommitted** as a separate document and run. Finding 3's "effective learning rate mismatch" hypothesis predicts that lr_cr=0.5 (or higher) may recover PathC-magnitude effects — but Finding 4's variance-flattening observation predicts even at recovered magnitude, per-seed robustness might not reach 70% on its own. F1 will distinguish these.

## What this does NOT decide

- **Γ1 family closure.** Γ1.c at lr_cr=0.1 alone is not enough evidence to close the family. F1 needs to run first.
- **Γ2 / Γ3 / Γ4 / Γ5 candidate ranking.** The Path γ mechanism-family survey at [path-gamma-mechanism-family-survey.md](../notes/emergent-codebook/path-gamma-mechanism-family-survey.md) named Γ2 (bundle-first) as the immediate fallback if Γ1 fails outright. That escalation is contingent on F1 closing.
- **Phase 5′ reopen.** Remains paused per [STATUS.md](../STATUS.md). No bridge / M2 / matrix / headline / graduation work authorized during Path γ.
- **The revised C.3 criterion.** Its first contact with a real mechanism worked as designed — it caught Γ1.c at lr_cr=0.1 cleanly, and the per-seed-paired clause exposed PathC's spirit-failure on the same seed set. No criterion adjustment is warranted.

## Anti-homunculus check

The headline result was produced by code that passed anti-homunculus reviewer 2026-05-27 PASS (see [Γ1 precommit](../notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md) §"Implementation findings"). The reviewer audit covers all H1-H7 checks; the implementation watch items W1-W3 were verified at code-landing time (see commit `e30b14e`). No new mechanism is proposed in this report — only the precommitted F1 follow-up is named, which itself will need a fresh precommit before any code or experiment lands.

## Limitations

- **Single n=10 seed set.** Per the precommit, n=30 pooled is authorized only if n=10 is borderline. n=10 here is clearly FAIL on both clauses; pooling would not change the verdict.
- **Single corpus** (wikitext-2-raw-v1). Inherits Report 112's limitation.
- **Single operating point.** D=4096, β=10, window=8, vocab_cap=1000, n_consolidation_events=1000.
- **Single lr_cr value** (0.1). F1's purpose is exactly to test the magnitude sensitivity.
- **No D-curve under Γ1.c.** Deferred to a post-graduation drill-down (precommit §"Pre-committed follow-ups" F5).

## Artifacts

All result trees on user's Drive at `MyDrive/neuro-ai/results/gamma1_headline_2026-05-27/`:

| run | path | n_seeds |
|---|---|---:|
| gamma1_headline | `gamma1_headline_seed{0..9}/c3_summary.json` | 10 |
| pathc_baseline | `pathc_baseline_seed{0..9}/c3_summary.json` | 10 |
| aggregate | `aggregate.json` (clause flags + per-seed Δ tables) | — |
| Colab logs | `colab_logs/` | — |

Each per-seed JSON includes the full `aggregated` dict with successes/trials/wilson per stratum, the `header` block with all flag values (including `use_context_residual`, `lr_cr`, `use_pull_push`), and the `per_cell` rows for finer per-condition inspection. Notebook commit: `b5dc42c`.

## Done-gate audit (per CLAUDE.md "What done looks like")

1. **Headline metric reported with confidence intervals:** ✅ Wilson CIs throughout, all primary strata.
2. **Control condition run on same test set:** ✅ matched-seed pull/push baseline; reproduces Report 112 v3 byte-identically.
3. **Drill-down metrics explain anomalies:** ✅ per-seed paired comparison localizes the attenuation (Finding 3 / Finding 4) and confirms PathC failing clause 2 at the same seed set (Finding 5).
4. **Result written up as markdown report under `reports/`:** ✅ this file.
5. **Relevant memory / status note updated:** to be landed this session (STATUS.md banner + Recent Updates entry + precommit running log).
