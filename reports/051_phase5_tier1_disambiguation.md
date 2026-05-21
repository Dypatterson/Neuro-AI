# Report 051 — Phase 5 Tier-1 disambiguation: three alternative fidelity metrics FAIL

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** All three Tier-1 alternative fidelity metrics from the
[2026-05-20 brainstorm](../brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md)
fail the pre-committed `std > 0.02` rescue threshold on the seed-17 A1'
substrate. The β rescue path is binding-falsified at n=1. **Per the
brainstorm's decision tree, Phase 5 pivots to pair #4
(metastability ~ replay-prioritization).**
**Driver:** [scripts/inspect_alternative_fidelities.py](../scripts/inspect_alternative_fidelities.py)
against [reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt](phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt).
**Output:** [reports/phase5_a1prime_pilot_seed17/alt_fidelities_w4_step1800.json](phase5_a1prime_pilot_seed17/alt_fidelities_w4_step1800.json)

## Experiment preamble

**Active phase:** 5

**Headline metric for this report:** per-atom variance (`std`) of three
fidelity metrics on the existing A1' seed-17 substrate. Pre-committed
pass threshold (binding): `std > 0.02` rescues β. Smaller std means
the metric is substrate-encoding-dominated.

**Required controls:** the existing `compute_role_fidelity` (mean
pairwise unbind distance) provides the baseline — report 050 found it
uniform at 0.9858 ± 0.0000 across 1064 atoms.

**Last verified result:** [report 050](050_phase5_beta_smoke_seed17.md)
— β smoke test FAILed at n=1; root cause attributed to FHRR
crosstalk `1/√D ≈ 0.016` setting f_i ≈ 0.984 structurally at D=4096.

**Why this experiment now:** the brainstorm
([brainstorm-workspace/2026-05-20-phase5-graduation/](../brainstorm-workspace/2026-05-20-phase5-graduation/))
identified three alternative per-atom fidelity metrics with per-atom
variance guaranteed *by metric construction* rather than by pattern
content. Tier 1 of the brainstorm's decision tree runs them on the
existing snapshot before any retraining, to disambiguate "substrate is
empty" from "metric is wrong."

## Headline result

| Metric | mean | std | min | max | in band [>0.02]? |
| --- | ---: | ---: | ---: | ---: | :---: |
| `pairwise_distance_current` (the existing f_i) | 0.9858 | **0.0000** | 0.9858 | 0.9858 | ✗ |
| `decode_margin_ganesan` (research C #1) | 0.0023 | **0.0012** | 0.0001 | 0.0090 | ✗ |
| `settling_based` (research B B2, research C #5, research E §3.3) | 0.9858 | **0.0000** | 0.9858 | 0.9858 | ✗ |

**All three FAIL the pre-committed rescue threshold.** The substrate
has no recoverable per-atom signal of sufficient magnitude in any of
these metric classes.

## Drill-down: decode-margin DOES discriminate, but at a tiny scale

While the decode-margin's overall std (0.0012) is well below threshold,
its distribution is *qualitatively correct* — it discriminates between
original Phase-3 atoms (idx 0–1023) and Phase-4 discovery atoms
(idx 1024–1063) in exactly the way A1' work was designed to produce:

| Atom class | mean | std | max | min |
| --- | ---: | ---: | ---: | ---: |
| Original Phase-3 (0–1023) | 0.0024 | **0.0012** | 0.0090 | 0.0001 |
| Discovery (1024–1063) | 0.0012 | **0.000006** | 0.0012 | 0.0012 |

Top-10 atoms by decode-margin (full-substrate vocabulary, ranked):

| Rank | Index | Class | Decode-margin |
| ---: | ---: | --- | ---: |
| 1 | 231 | original | 0.0090 |
| 2 | 724 | original | 0.0089 |
| 3 | 178 | original | 0.0089 |
| 4 | 488 | original | 0.0081 |
| 5 | 448 | original | 0.0080 |
| 6 | 554 | original | 0.0076 |
| 7 | 734 | original | 0.0068 |
| 8 | 517 | original | 0.0066 |
| 9 | 186 | original | 0.0064 |
| 10 | 817 | original | 0.0063 |

**All top-10 are original Phase-3 atoms.** Discovery atoms are tied at
0.0012 with std 6e-6 (FP-level identical, just as A1' was designed to
produce — they have no role-binding structure to discriminate).

The decode-margin metric **correctly identifies which atoms have
intact role-binding** — it just does so at a tiny absolute scale
(max margin 0.009 vs ~0.9 achievable for sharp role-binding in the
unit test). The substrate's geometric density at D=4096 forces every
decode to be diffuse across the 1064-atom vocabulary, compressing the
discriminative signal into an effective range below the threshold.

## Why settling didn't help

The `settling_based` metric was the cheapest free idea from the
brainstorm (3 research angles converged on it). It returns
**bit-identical** stats to the raw f_i (mean 0.9858, std 0.0000, min
= max).

The reason, by inspection: the Hopfield substrate has very sharp
basins for each atom (a consequence of A+B+A1+A1' preserving d_eff at
35.23 with continuous death). Each atom retrieves itself with near-FP-
precision probability. The settled state q_settled_i ≈ s_i, so its
pairwise-distance fidelity inherits the raw schema's uniformity. The
"settling reshapes the unbind structure" hypothesis was correct in
principle but doesn't fire here because the substrate has
*self-retrieving* basins.

This is itself a substrate finding: **A+B+A1+A1' produced sharp
self-retrieval basins as a side effect of the redundancy throttle.**
That's actually a good property (no spurious retrievals) but it
eliminates settling-based diagnostics as a fidelity rescue route.

## Pre-committed binding remains intact

All three Tier-1 metrics were committed to before observation:
- pairwise_distance_current: the existing baseline
- decode_margin_ganesan: per the formula in research C #1
- settling_based: per research B B2 / C #5 / E §3.3

The `std > 0.02` pass threshold was set BEFORE running. None of the
metrics' parameters were tuned post-hoc. Per H4, this is a
falsification result, not an invitation to retune (e.g., to lower
the threshold to 0.001 to "pass" decode-margin).

The disciplined verdict: the metric category "intrinsic per-atom
fidelity computable on the existing A1' substrate" is exhausted.

## The four-step debugging chain is now complete

| Layer | Mechanism | Verdict at n=1 |
| --- | --- | --- |
| Substrate-construction (report 047) | A+B | FAIL (discovery atoms dominate) |
| Measurement (report 048) | A1 (substrate-derived r_ema init) | FAIL (RMS too weak) |
| Reduction operator (report 049) | A1' (max-over-others) | FAIL (selector ties) |
| Selector layer (report 050) | β (continuous fidelity-weighted prior) | FAIL (uniform f_i at D=4096) |
| **Metric rescue (this report)** | **decode-margin / settling / pairwise** | **ALL FAIL (no per-atom signal)** |

A+B+A1+A1' built the substrate as designed (d_eff preserved,
duplicates throttled, original atoms dominate). Beyond that, no
substrate-side measurement at D=4096 produces enough per-atom
variance to ground "structural retrieval" as currently
operationalized via K-branch state_divergence under role-prior.

## What this report does NOT do

- It does NOT declare the architecture failed. The substrate is
  empirically what the design notes specified.
- It does NOT exhaust the brainstorm's idea space. Three Tier 2 paths
  remain viable: PAM predictor-distance (a fundamentally different
  paradigm), permutation-binding (the algebraic fix), and pair #4
  metastability (a Phase-5 headline pivot). All three were ranked
  in the brainstorm.
- It does NOT exercise n=5. The decisive n=1 finding (std 1000× below
  threshold) makes cross-seed confirmation low-value; the encoding-
  noise-floor argument is dimensional, not seed-dependent.
- It does NOT exhaust the cue-regime question. The brainstorm's Tier-1
  cue-regime sweep was sketched but not run; it tests a different
  question (whether the headline-test operating point can rescue β
  even with uniform f_i). If you want to be exhaustive about Tier 1,
  the sweep is a half-day of Colab. The empirical case for pivoting is
  already strong without it.

## Updated recommendation (the disciplined next move)

**Phase 5 pivots to pair #4: metastability ~ replay-prioritization
([brainstorm Idea 5](../brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md)).**

Per [research D's analysis](../brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md):

- The substrate already computes `meta_stable_rate` every retrieve() call (in `phase2/metrics.py`, surfaced via `TrajectoryTrace`).
- The actuator is a Benna-Fusi/Saighi-shape per-atom EMA `m_i` multiplied into the existing `ReplayStore` priority (which already composes `gate × tag × suppression`).
- **The Phase 5 headline becomes `Δ meta_stable_rate at W=3`** — the same substrate-pure metric class that just graduated Phase 4 on D1 ([report 038](038_phase4_d1_graduation.md): Δms_w3 = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds).

This is the "stop fighting K-branch state_divergence; the substrate
is fine; use a metric the substrate has variance in" move. The
substrate has variance in meta_stable_rate (per Phase 4 D1
graduation); it does NOT have variance in role-fidelity at D=4096
(per this report).

Estimated implementation: ~½ day plumbing + design note + anti-
homunculus audit. Colab n=10 retrain for graduation verification.

## What was confirmed (positive finding)

A1' is doing its job. The decode-margin's qualitative discrimination
(original atoms 2× higher margin than discovery atoms; top-10 all
originals; discovery atoms tied at FP precision) is the *exact*
substrate property A1' was designed to produce. The four-step
debugging chain has built the substrate correctly — what it didn't
build is a substrate with the *operationalization-of-structural-
retrieval* the K-branch headline requires. **That operationalization
was the wrong test for this substrate, not the substrate that's the
wrong substrate for the test.**

Pair #4 graduates the project on the substrate-pure metric class
the project's discipline has already validated.

## Sequencing

1. Draft pair #4 design note (analogous to the 2026-05-20 death-
   dynamic note structure). Pre-committed falsification criteria.
2. Anti-homunculus reviewer audit. PASS required before code.
3. Implement `m_i` per-atom metastability EMA in `ConsolidationState`;
   plumb into `ReplayStore.add` as multiplicative priority modifier.
4. Tests + full suite (262 → 265+).
5. Colab n=10 retrain.
6. Headline measurement: Δ meta_stable_rate at W=3 with CI.
7. STATUS walk-back.

The user has indicated substantial runs go to Colab. The local work is
~1 day to commit + push, then the Colab n=10 is the graduation
attempt.
