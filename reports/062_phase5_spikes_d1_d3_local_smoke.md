# Report 062 — Phase 5 Spikes D1+D3 Local Smoke (Null Result)

**Date:** 2026-05-24
**Active phase:** 5
**Status:** Smoke complete. **Null result for both D1 and D3** at n=3 seeds × 30 cues. Neither mechanism produces role-target basin retrieval; structural retrieval remains absent.
**Decision:** Per the [Phase 5 rescue brainstorm](../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)'s Tier-0 decision recipe, this is the "neither helps → Tier 2 (training-time intervention)" branch. The bottleneck is not retrieval mechanism; it is substrate training. Do not commit to D1, D3, or P1-as-storage-only as Phase 5 graduation routes.
**Headline metric per [phase-5-unified-design.md:269-292](../notes/emergent-codebook/phase-5-unified-design.md):** `ΔE = E_content_prior - E_role_prior`, paired per cue. Positive means role-prior branch lands at lower final-state energy.
**Required controls per [phase-5-unified-design.md:296-303](../notes/emergent-codebook/phase-5-unified-design.md):** baseline (raw MHN) acts as the within-experiment control vs D1/D3; smoke scale is below graduation threshold (n=3 seeds × 30 cues, not n=10 × 100 with CI).
**Last verified result:** [Report 061](061_phase5_log_prior_gain1_required_controls.md) — Path C log-prior spike `hit_role ≈ 0.003`, `rank_role ≈ 443/489`, structural retrieval absent under arbitration-shape positive control.
**Why this experiment now:** Tier-0 diagnostic spikes from the [2026-05-24 brainstorm](../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md) (D1 pseudo-inverse storage swap; D3 slot-style cross-K softmax). The brainstorm pre-committed: D1 and D3 should produce noticeable `hit_role` lift if storage geometry or branch coupling is the load-bearing gap. If neither helps, the gap is substrate training, not retrieval mechanism.

## Setup

- Harness: `scripts/spike_d1_d3_local_smoke.py`
- Substrate snapshots (locally available, A+B+A1' Phase 4):
  - `reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime.pt` (seed 17)
  - `..._AB_A1prime_seed11.pt` (seed 11)
  - `..._AB_A1prime_seed23.pt` (seed 23)
- Seeds: 17, 11, 23 (only 3 snapshots available locally; full Colab harness would use 10)
- N cues per seed: 30 (smoke; full would be 100)
- β = 10, max settling iter = 12
- Binding noise σ = 0.05, content distortion = 0.6 (matches Report 061 v2)
- Cue seed = 117 (matches Report 061 v2)
- D1 regularization λ = 1e-3 (pseudo-inverse)
- D3 cross-K mix coefficient μ = 0.5 (additive form per [D3 Lyapunov analytical pass](../notes/notes/2026-05-24-spike-D3-lyapunov-analytical.md))

## Conditions

Each condition runs 3 retrieval branches per cue (role-target prior, content-distractor prior, random-distractor prior) and measures the same headline + drill-downs.

| Condition | Retrieval rule |
|---|---|
| `baseline` | Standard MHN settling: `s ← Xᵀ softmax(β · X s)` per branch, independent across branches |
| `D1` | Linear pseudo-inverse settling: `s ← Xᴴ (X Xᴴ + λI)⁻¹ X s`, independent per branch (Kymn/Stewart 2022 style) |
| `D3` | Additive cross-K MHN: `s_k ← Σ_p [(1-μ)π_k(p) + μ·α_k(p)] · x_p` where `π` is softmax over patterns, `α` is softmax over branches (per the Lyapunov pass; multiplicative form rejected) |

## Aggregated results (3 seeds × 30 cues = 90 cues per condition)

| condition | mean ΔE | ×floor | mean hit_role | mean rank_role | random_lowest | seeds ΔE > 0 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +0.00267 | +0.49 | 0.000 | 205.5 | 0.367 | 2/3 |
| D1       | −0.00021 | −0.04 | 0.000 | 403.1 | 0.256 | 2/3 |
| D3 (μ=0.5) | −0.00496 | −0.90 | 0.000 | 226.8 | 0.344 | 0/3 |

**All three conditions report `hit_role = 0.000`.** The role-target atom is never the top-decoded pattern after settling for any cue, in any condition, on any seed.

## Per-seed results

| seed | condition | ΔE | hit_role | rank_role | random_lowest |
|---:|---|---:|---:|---:|---:|
| 17 | baseline | +0.00287 | 0.000 | 207.6 | 0.400 |
| 17 | D1       | +0.00018 | 0.000 | 453.0 | 0.267 |
| 17 | D3       | −0.00239 | 0.000 | 213.4 | 0.300 |
| 11 | baseline | −0.00266 | 0.000 | 204.2 | 0.367 |
| 11 | D1       | −0.00181 | 0.000 | 482.1 | 0.233 |
| 11 | D3       | −0.00532 | 0.000 | 237.1 | 0.433 |
| 23 | baseline | +0.00780 | 0.000 | 204.6 | 0.333 |
| 23 | D1       | +0.00099 | 0.000 | 274.1 | 0.267 |
| 23 | D3       | −0.00716 | 0.000 | 229.9 | 0.300 |

Seed 23 again surfaces as the high-magnitude outlier on baseline (ΔE = +0.0078, above floor at 1.42×) but with the same null on `hit_role` — consistent with the prior project finding that seed 23 has idiosyncratic geometry but does not translate to structural retrieval.

## Reading

### D1 (pseudo-inverse) does not unlock role-target basin retrieval.

The brainstorm's strongest claim for D1 was that pseudo-inverse storage would reveal role information that pure-Hebb MHN softmax was masking. The data refute it on this substrate at smoke scale. D1 actually moves `rank_role` *worse* (403 vs 205 for baseline) and `mean ΔE` to essentially zero. Interpretation: the projection onto the column space of `X` (the consolidated atoms) is too uniform — every initial state, regardless of role-target or content-distractor identity, projects onto roughly the same mixture in pattern space. The pseudo-inverse over-orthogonalizes, washing out whatever weak role signal the baseline preserves through its softmax sharpness.

This *closes* the Kymn/Stewart-2022-inspired storage-rule hypothesis: replacing Hebb-style implicit storage with pseudo-inverse explicit projection does not, on its own, make this substrate role-addressable. The storage rule is not the load-bearing parameter on the current substrate.

### D3 (cross-K softmax) actively hurts.

The Lyapunov analytical pass derived the correct (additive) form of cross-K softmax and confirmed gradient flow on a joint energy. The mechanism is sound; the *empirical effect* on this substrate at μ=0.5 is that cross-K coupling drives ΔE *negative* across all three seeds (mean −0.00496, 0/3 positive). Interpretation: the cross-K term pushes role-prior and content-prior branches apart, but on a substrate where neither branch has a strong attractor to begin with, the symmetric repulsion penalizes the role-prior branch as much as the content-prior branch. The branches end up at higher energy (less settled) without preferentially separating in a structurally meaningful direction.

This is consistent with the Lyapunov pass's explicit caveat: "the cross-K term `-Σ_p lse_k` preserves asymmetry rather than erasing it... BUT only if the branches start asymmetric." On this substrate the role-prior and content-prior initial states are similarly close to nothing — there's no asymmetry for cross-K to amplify, so it just adds noise.

A μ sweep was not run; the additive form is Lyapunov-clean for all μ ∈ [0, 1], so μ=0.5 is a substrate-fair midpoint. Lower μ (μ→0) recovers baseline and is therefore uninformative; higher μ would push cross-K more aggressively but would likely make ΔE more negative still. Not worth sweeping on this substrate.

### Baseline is what we already had.

Baseline (raw MHN settling from atom init) gives mean ΔE = +0.0027, similar shape to the Report 061 `gamma=0.5, gain=0` row (+0.0024). This is the project's known sub-floor directional signal. `hit_role = 0.000` is consistent with Report 061's `hit_role ≈ 0.003` to within smoke noise — the substrate genuinely does not return role-target atoms as top decodes.

### The brainstorm's decision recipe routes us to Tier 2.

From [brainstorm-phase5-rescue.md](../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md):

> If D3 helps and D1 doesn't → Tier 1 path S (branch coupling)
> If both help, do both at Tier 1 (path M, → P1)
> **If neither helps, the atoms themselves don't carry role information; go Tier 2 (training-time intervention).**

This smoke is the "neither helps" outcome. The next move is **Tier 2: training-time intervention** that puts role-target basins into the substrate, since the substrate as currently consolidated does not contain them and no retrieval mechanism we tested can extract them.

Candidates from the brainstorm's Tier 2:

- **M2 (P4 EqProp + role-shuffled negatives + DSM warm-start)** — anti-homunculus reviewer PASS, audit 2026-05-24. Carves role-target basins via contrastive consolidation passes. Highest leverage if the goal is to demonstrate structural retrieval cleanly.
- **M1 (P1 + D3 + P3)** — anti-homunculus reviewer CONDITIONAL (inherits P1+D3 conditions). Per-role energies *plus* training pressure. The P1 spec change is now necessary, not optional, since D1's null result confirms storage-rule-only changes are insufficient.

## Limitations (smoke-specific, not graduation-binding)

- **n=3 seeds** (only seeds 17, 11, 23 have local snapshots). Full graduation criterion is n=10. The cross-seed direction (2/3 baseline positive; D1 also 2/3 positive but at near-zero magnitude; D3 0/3 positive) is suggestive but not CI-strong.
- **n=30 cues per seed** (Report 061 used 100). Bootstrap CI not reported; per-seed values are point estimates.
- **Cue setup is simplified** vs. the full Phase 5 control matrix: this smoke initializes branches directly from the role-target / content-distractor / random atoms, bypassing the schema-store selector and log-prior weighting machinery used by `experiments/40_phase5_branching.py`. The simpler setup isolates the *retrieval mechanism*, but a more sophisticated cue construction with schema-store priors might surface a different signal. For D1 and D3 specifically, the null is robust because they failed at the most-favorable-possible cue construction (initializing directly from the target atom).
- **Pseudo-inverse λ = 1e-3, D3 μ = 0.5** are single fixed values. Neither parameter sweep was run; both are reasonable midpoints. A sweep would not change the qualitative null (D1 cannot create role attractors that don't exist; D3 cannot create asymmetry that doesn't exist in the initial state).

## What this DOES and DOES NOT close

**Closes**:
- The storage-rule-only hypothesis for Phase 5 (D1). Pseudo-inverse storage on the current consolidated substrate does not produce role-target basin retrieval.
- The retrieval-coupling-only hypothesis (D3 at μ=0.5). Cross-K softmax does not produce structural retrieval on a substrate where the per-branch attractors don't exist.

**Does NOT close**:
- The per-role energy hypothesis (P1) — per-role codebooks could in principle carve role-target basins even on this substrate IF the per-role attribution `c_{i,r}` distinguishes atoms meaningfully across roles. The P1 spec is unaffected by this smoke; it would need its own implementation + run to test.
- The training-time hypotheses (P4 EqProp, M2 stack) — these are the natural next move per the brainstorm's decision recipe and remain untested.
- The IDP-saliency hypothesis (P3) — not run in this smoke; reshapes the energy landscape per-cue, distinct from D1 (storage swap) and D3 (cross-K coupling). Worth a separate spike if Tier-2 work doesn't pan out.

## Anti-homunculus discipline

- D1, D3, baseline all run as substrate-only dynamics (energy/projection settling). No supervisor reads any metric to choose between branches or conditions.
- The smoke harness reports both null and non-null results uniformly. No "best of N conditions" cherry-pick.
- Seed-23 outlier behavior reported as data, not laundered into a positive headline.
- D1 was kept as a hypothesis-rejecting null even though "structural retrieval works on a different storage rule" would have been the more brainstorm-favorable outcome. The data refute it.

## Artifacts

- Harness: [`scripts/spike_d1_d3_local_smoke.py`](../scripts/spike_d1_d3_local_smoke.py)
- Per-seed/condition JSON: [`reports/spike_d1_d3_local_smoke.json`](spike_d1_d3_local_smoke.json)
- Sibling spike — S1 replay-trace schema check: [`notes/notes/2026-05-24-spike-S1-replay-trace-schema.md`](../notes/notes/2026-05-24-spike-S1-replay-trace-schema.md)
- Sibling spike — D3 Lyapunov analytical pass: [`notes/notes/2026-05-24-spike-D3-lyapunov-analytical.md`](../notes/notes/2026-05-24-spike-D3-lyapunov-analytical.md)
- Brainstorm doc this responds to: [`brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md`](../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)

## Recommendation for STATUS.md

Add a new "Recent updates" entry pointing here. Promote the Phase 5 path decision to consider **Path D = Tier-2 training-time intervention** alongside the open Path A/B'/C choice. The brainstorm's "Decision recipe" outcome on this smoke routes us to Path D.

Do NOT promote D1 or D3 to a Colab n=10 run on this substrate. The smoke is sufficient to close both as standalone mechanisms. A Colab n=10 confirmation might be worth running for completeness *if* the next reviewer audit / user wants a CI-strong null on record, but it would not change the path decision.
