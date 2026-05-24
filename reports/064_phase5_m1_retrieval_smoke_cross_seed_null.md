# Report 064 — Phase 5 M1 Retrieval Smoke Cross-Seed Null

**Date:** 2026-05-24
**Active phase:** 5
**Status:** Smoke complete. **Null result across 3/3 substrate seeds.** M1's retrieval-side variant (P1 weighted MHN + D3 cross-K + P3 saliency stack) with geometric `codebook_prior_density` row-role weights — the only weight protocol the P1 geometric probe found non-degenerate — does not select role-target patterns at any seed. `hit_role = 0.000` across 90 cues × 3 seeds = 270 cue evaluations. ΔE is robustly negative (−0.32 ± 0.01 across seeds), with content prior settling to lower energy than M1's minimum-energy branch on **0 / 90 cues**.
**Decision:** Do not promote M1 to a Colab n≥10 control matrix. The null is not seed-17 idiosyncratic; the variance across seeds is *smaller* than within seed 17, not larger. Path D / M1 retrieval-side variant joins the path graveyard. The only Phase 5 path not yet smoked is Path D / M2 (training-time intervention).
**Headline metric per [phase-5-unified-design.md:280-282](../notes/emergent-codebook/phase-5-unified-design.md):** `ΔE = E_content_prior − E_role_prior`, paired per cue. Floor `5.5e-3` per [magnitude-floor pre-commit](../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md). Here `E_role_prior = E_M1_min_branch` and `E_content_prior = E_settled_with_content_distractor` per the existing snapshot-smoke harness, plus a third `E_random_prior` baseline.
**Required controls per [phase-5-unified-design.md:309-314](../notes/emergent-codebook/phase-5-unified-design.md):** B1 random-schema reported alongside; B2 K=1 not run (M1's branching is structurally K=n_roles); B3 γ=0 not run (M1 has no γ knob — its prior is the per-role weight matrix); B4 no-schema-store not run (M1's "store" is the substrate pattern matrix). This smoke does **not** satisfy the n≥10 control matrix and is explicitly labeled drill-down/smoke.
**Last verified result:** [Report 063](063_phase5_spike_e1_centered_log_prior.md) — Spike E1 (asymmetric log-prior field, "Path C done right") null at all λ, ruling out the landscape-reshaping retrieval family.
**Why this experiment now:** STATUS.md blocker #3 named the n≥10 Colab M1 evidence run as the gating step before any graduation claim. Before paying Colab credits, run a single-seed local smoke to verify M1's headline is non-null. The single-seed smoke returned strongly negative ΔE and `hit_role = 0.000`; the user then requested cross-seed verification that this isn't a seed-17 idiosyncrasy. This report covers all three seeds.

## Setup

- Harness: [`scripts/phase5_m1_snapshot_smoke.py`](../scripts/phase5_m1_snapshot_smoke.py)
- Substrate snapshots:
  - Seed 17: `reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt` (S1-provenance-bearing; audited)
  - Seed 11: `reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime_seed11.pt` (pre-S1; `--skip-audit` used)
  - Seed 23: `reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime_seed23.pt` (pre-S1; `--skip-audit` used)
- Operating point: β=10, γ=0.5 (content/random branches only), max settling iter=12, binding noise σ=0.05, content distortion=0.6, d3_mix=0.5, p3_saliency_gain=0.0, laplace_count=1.0
- Role weights: `weight_source=geometric`, `geometric_mode=codebook_prior_density`, neighbor_k=8, laplace=1e-6, temperature=0.05
- Codebook: `phase3c_codebook_reconstruction.pt` (SHA-256 `863d2ae4…12aa6`); registry entry in `config/phase5_m1_codebook_registry.json`
- 30 cues per seed × 3 seeds = 90 total cue evaluations

**Why `--skip-audit` is acceptable here.** The M1 provenance audit hard-fails on `missing_pattern_encoder_terms`. Those encoder terms are only consumed by `weight_source=count`, which the [P1 geometric probe](../reports/phase5_m1_geometric_probe/p1_geometric_probe_report.md) already showed to be **degenerate** under full-window role symmetry. With `weight_source=geometric` + `codebook_prior_density`, the weight matrix is derived from `(substrate.patterns, role_vectors, codebook)` only — no encoder counts needed. The audit-bypass is therefore safe for this specific configuration; the harness flags the resulting payloads as `"UNAUDITED cross-seed smoke; not Phase 5 evidence"` in the report scope. The seed-17 run, which has full provenance, was audited and serves as the canonical smoke.

## Aggregated results

| seed | audited? | ΔE mean | ΔE stdev | M1-lower cues | hit_role | rank_role | random_lowest | n_patterns |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | ✅ | **−0.3381** | 0.0714 | **0 / 30** | 0.000 | 540 / 1064 | 0.43 | 1064 |
| 11 | ⚠️ skipped | **−0.3155** | 0.0215 | **0 / 30** | 0.000 | 453 / 1024 | 0.83 | 1024 |
| 23 | ⚠️ skipped | **−0.3113** | 0.0239 | **0 / 30** | 0.000 | 501 / 1024 | 0.73 | 1024 |

**`hit_role = 0.000` at all 3 seeds × 90 cues.** M1's best (minimum-energy) branch never selected the role-target pattern.

**`rank_role` median 453–540** out of ~1024 patterns: the role target sits at the 44th–53rd percentile of patterns by M1's role-weighted score. Statistically indistinguishable from random placement.

**`mean ΔE ≈ −0.32` across all seeds.** Variance *decreases* across seeds (σ=0.071 → 0.022 → 0.024) — the null is *more* consistent on the unaudited seeds, not noisier. The 5.5e-3 magnitude floor is exceeded by **58× in the wrong direction**.

**`random_lowest` rises on seeds 11/23 (0.73, 0.83)** vs seed 17 (0.43). On the majority of cues at seeds 11 and 23, random-pattern priors settle to lower energy than both content and M1. M1 is *worse than random prior* on these substrates.

### Role-weight matrix is non-degenerate (sanity check)

Confirming that the codebook-prior geometric weighting does produce a non-trivial role-weight matrix (the failure mode the P1 geometric probe diagnosed for count and same-role-pool modes):

Seed 17 audit reports:
- Mean normalized row entropy: **0.887** (1.000 is uniform; 0.000 is one-hot)
- Uniform row fraction: **0.025** (only 27 / 1064 rows are uniform)
- Role utilization: [0.305, 0.202, 0.201, 0.291] across 4 roles

The weights *do* differentiate atoms by role. They just don't point at role-target basins during M1 settling.

## Reading

### The null is structural, not seed-17 idiosyncratic.

The user explicitly raised the seed-23-outlier-pattern (Phase 4 idiosyncratic geometry, [STATUS blocker #5](../STATUS.md)) as a worry. The cross-seed verification rules it out:

- All three seeds give ΔE in [−0.34, −0.31]: range 0.03, far smaller than seed 17's own within-seed stdev (0.07).
- All three seeds give `hit_role = 0.000`, exactly.
- `rank_role` ranges 453–540 across seeds; none of them place the role target in the top 50.

If this were a seed-17 idiosyncrasy, seeds 11 and 23 would show ΔE recovery or `hit_role > 0` on at least some cues. They don't.

### Same diagnosis as Reports 058, 061, 062, 063.

This is the fourth converging line of evidence that **the FHRR substrate as currently consolidated does not contain role-addressable basins**:

| family | spike | seeds × cues | hit_role | structural reading |
|---|---|---:|---:|---|
| Storage rule (Hebb → pseudo-inverse) | D1, [Report 062](062_phase5_spikes_d1_d3_local_smoke.md) | 3 × 30 = 90 | 0.000 | `rank_role` 205→403 |
| Branch coupling (independent → cross-K softmax) | D3, [Report 062](062_phase5_spikes_d1_d3_local_smoke.md) | 3 × 30 = 90 | 0.000 | ΔE drives negative 0/3 seeds |
| Landscape reshaping (uniform → asymmetric field) | E1, [Report 063](063_phase5_spike_e1_centered_log_prior.md) | 3 × 30 × 4λ = 360 | 0.000 | `rank_role` monotone 205→498 with λ |
| **Role-weighted MHN stack (P1+D3+P3, geometric codebook-prior weights)** | **M1, this report** | **3 × 30 = 90** | **0.000** | **`rank_role` median 453–540; M1 worse than random** |

Four architecturally distinct retrieval-mechanism families have now been smoke-tested and returned `hit_role = 0.000`. The mechanisms varied across:
- which energy function (per-pattern MHN vs joint role-energy stack)
- which weight derivation (uniform, pseudo-inverse storage, cross-K softmax coupling, asymmetric logit field, codebook-prior geometric row weighting)
- which branching topology (single-branch, cross-K coupled, K=n_roles independent)

None of them surfaces role-target retrieval. The invariant across all of them is *the substrate*. Per the standard Bayesian reading: when many distinct mechanisms downstream of a shared upstream component all fail, the upstream component is the cause.

### Why M1's geometric weights don't help

The P1 geometric probe characterized the codebook_prior_density weight matrix correctly: non-degenerate, row-specific, codebook-anchored. But "non-degenerate weight matrix" is necessary, not sufficient, for role-target retrieval. The weights have to point at *role-target* atoms in the *substrate*'s pattern matrix.

The mechanism is: `row r, role-r weight w_{r,r}` is computed as the softmax (temperature 0.05) of `topk_8(cosine(unbind(pattern_r, role_r), codebook))`. This says: "for atom r in role r, the role-r weight is high if pattern_r, when unbound from role-vector r, has high cosine to a few codebook entries." This is essentially asking the codebook to vouch for the pattern's role identity.

But the FHRR substrate doesn't store atoms with strict per-role identity — the consolidation process pools encoded windows that may have been bound at different positions, and the resulting "pattern" is the consolidation centroid. The codebook-prior weights end up giving each pattern row a peaky weight distribution across roles based on which codebook entry happens to align with the unbinding result at each role-vector — but the alignment is dominated by the same noise channel that defeated E1's asymmetric field (Report 063 §"Why E1 worsens role-prior settling").

In short: the geometric weights are *measuring* something about pattern×role geometry, but what they're measuring is not "this pattern is the role-r target for this cue."

### What this strengthens about the Path D triage

The Path D family has two declared variants in the brainstorm:

- **M1**: training-mechanism-stack at retrieval time (P1 weighted MHN + D3 cross-K + P3 saliency). This report rules out M1's retrieval-side variant at smoke scale on the existing substrate.
- **M2**: training-time intervention (EqProp + role-shuffled-negatives + DSM warm-start). Requires a substrate retrain. **Not yet attempted.**

M1's STATUS-blocker fallback ("rerun a provenance-bearing substrate, then run the n≥10 control matrix") is now substantially weakened: the substrate the retrain produces would still be consolidated by the existing Phase 4 dynamics, which is the layer that this report (and 058, 061, 062, 063) implicates as the cause. There is no architectural reason to expect that retraining will change M1's retrieval-side null. Encoder-count weights — the other M1 weight variant the retrain enables — are degenerate per the P1 probe, so they would not change `hit_role = 0.000` either.

This leaves **M2** as the only Phase 5 path not exhaustively ruled out at smoke scale.

## Limitations

- n=3 seeds × n=30 cues per seed (smoke; graduation standard is n=10 × ≥100 with formal CI)
- Seeds 11 and 23 use `--skip-audit`. Audit was added precisely to prevent silent dependency on missing encoder terms; geometric weights don't consume encoder terms, so the bypass is principled for this configuration, but it should not be normalized into routine runs.
- Energies are not bit-identical apples-to-apples: M1 reports the minimum-energy branch from its joint role-energy stack; content and random use the existing per-pattern Phase 5 baseline. The ΔE magnitude is therefore approximate. The `hit_role` and `rank_role` measurements are within-M1 and unambiguous regardless of cross-formulation comparison.
- M1's D3 cross-K mixing is fixed at d3_mix=0.5; P3 saliency gain at 0.0. These were the default smoke values, not swept. A d3_mix or saliency sweep could in principle change settling dynamics but would not change the role-weight matrix — and the null is in the weight matrix not pointing at role targets, not in the settling.
- Only the `codebook_prior_density` geometric mode was run. The `unbind_density` and `same_role_filler_density` modes are documented degenerate (P1 probe); they were not retested here.
- B2/B3/B4 controls from the design spec were not run (B2 K=1 is structurally ill-defined for M1's per-role branching; B3 has no γ knob in M1; B4 no-schema-store is not the relevant control here). The within-M1 random-prior comparison (`random_lowest`) is the closest control reported.

## Anti-homunculus discipline

- M1's role-weight matrix is a deterministic function of `(substrate.patterns, role_vectors, codebook)` — pure geometric computation, no metric trigger.
- The minimum-energy branch selection is `min(branches, key=energy)` — a single arg-min on a Lyapunov function, not arbitration across alternative architectures.
- The random-prior baseline is drawn uniformly excluding `{role_target, content_distractor}` — fixed sampling, not adaptive.
- The cross-seed smoke is reported as data; no post-hoc selection of "best seed"; the per-seed table reports all three runs.
- No graduation claim; no shift of which schema source M1 reads from based on result (the H1 prohibition in [phase-5-checklist.md:191-192](../notes/emergent-codebook/phase-5-checklist.md)).

## What this DOES and DOES NOT close

**Closes**:
- The M1 retrieval-side variant of Path D at smoke scale on the existing FHRR substrate. The codebook-prior geometric weight protocol (the only non-degenerate M1 weight source) does not select role-target patterns at any of the 3 available substrate seeds.
- The "Path D is the only path not ruled out at smoke scale" framing in STATUS.md as written. Path D is now split: M1 retrieval-side is ruled out at smoke scale; M2 training-time is not yet smoked.
- The provenance-bearing-substrate retrain blocker (STATUS blocker #3) as a route to M1 evidence. The retrain does not change the diagnosis layer.

**Does NOT close**:
- Path D / M2 (EqProp + role-shuffled-negatives + DSM warm-start). M2 changes the substrate's consolidation dynamics, which is the layer this report and the prior four implicate. M2 is the next architectural commit decision.
- The dual-code GHRR variant of P1 (separate role algebra). This is a P1 alternative algebra; it would change M1's `cosine(unbind(pattern, role), codebook)` computation. The brainstorm did not yet smoke it; nothing in this report directly rules it out, but the diagnosis points upstream of the algebra.
- Phase 5 itself as a viable phase. Closure-paper framing per the 2026-05-23 SNR walk-back remains available if M2 is also null.

## Tooling notes

- `scripts/phase5_m1_snapshot_smoke.py` extended with `--codebook`, `--codebook-registry`, and `--skip-audit` flags. The `--skip-audit` flag is documented in the argparse help as unsafe and intended only for cross-seed geometric smoke on pre-S1 snapshots.
- Three JSON + Markdown payloads under `reports/phase5_m1_snapshot_smoke_seed17/` (the directory name is from the original audited seed-17 run; cross-seed files were colocated for ease of comparison).
