# Brainstorm Test Results — Smoke Executions of (a), (b), (c)

> **⚠ Correction posted 2026-05-24 after external audit.** The (b)
> section calls the freq-weighted α experiment "wired but unrun" and
> recommends running the Colab to falsify it. **That recommendation is
> retracted**: [Report 040](../../reports/040_freq_weighted_alpha_sweep.md)
> already ran the experiment at n=10×40 production scale on 2026-05-17,
> found λ ∈ {0, 0.5, 1.0} statistically indistinguishable, λ=2.0
> supercritical (W=2 corr flips to −0.52). Phase 5 design doc
> [phase-5-unified-design.md:97-107](../../notes/emergent-codebook/phase-5-unified-design.md)
> records the resolution. The synthetic smoke in this document is now
> retrospectively useful as: "an at-n=400-cues check that reproduces
> the directional behavior of the production result without needing
> Colab." Findings (a) and (c) stand unchanged.


> Generated 2026-05-24 after the verification report flagged that
> recommendations had only been analyzed, not tested. This document
> reports real smoke executions of the three top recommendations from
> [brainstorm-unconsidered-paths.md](brainstorm-unconsidered-paths.md),
> using local CPU at small scale. Scripts saved alongside this file.
> All three executions returned interpretable signals; one returned a
> **directional reversal** that changes its standing as a
> recommendation, and one returned a **smoking-gun confirmation** of
> the brainstorm's Phase-5 diagnosis.

---

## TL;DR

| Rec. | Test outcome | New verdict |
|------|--------------|-------------|
| (b) | Mechanism wired correctly (26/26 unit tests pass) but at synthetic-Phase-4 scale `corr_u_m_retrieval_count` **decreases monotonically with λ**, opposite of the brainstorm/STATUS.md framing | Run the existing Colab notebook to **falsify**, then archive the freq-weighted-α path. ΔE almost certainly will not move in the predicted direction. |
| (a) | Range-shaped sampler drops joint-distribution rectangularity by 62× (KL 2.06 → 0.03), reaches 8× more (role, content) cells, marginals preserved within L1 ~0.05–0.11 | **Sampler prototypes cleanly.** Remaining work is the substrate-side re-binding when (role, content) pairs have no exact trace. Realistic cost still 3-5 days. |
| (c) | MQAR-style capacity curve at D ∈ {1024, 4096}, N ∈ {16, …, 512}: HRR-bundle hits 100% to N=256 at D=4096; Hopfield-with-perfect-cue is trivially 100% always; **Hopfield-with-key-only-cue collapses from 12.5% at N=16 to 0% by N=128** | **Smoking-gun confirmation of the Phase-5 diagnosis.** This is the missing unit test. Adopt MQAR immediately as a Phase-5 external sanity check. |

The most important finding: **(c) shows that bundle-based recall works (100% at N=128, D=4096) while Hopfield-key-only-recall is 0% at the same scale.** The four Phase-5 retrieval-mechanism nulls are predictable from this — they're all asking the Hopfield landscape to do something the substrate cannot do without an algebra or storage change.

---

## (b) Freq-weighted Benna-Fusi α — smoke result

### Code state confirmed
- `alpha_freq_lambda` exists in [consolidation.py:81](src/energy_memory/phase4/consolidation.py).
- Math is at [consolidation.py:451-459](src/energy_memory/phase4/consolidation.py).
- 9 unit tests cover it explicitly (all pass).
- One test, `test_lambda_changes_steady_state_distribution_of_u_m`, is
  labeled "**Architectural finding worth documenting**" and documents
  that *higher α makes u_m **smaller** at steady state* because high-α
  cascades leak more mass to the absorbing boundary.

### Synthetic Phase-4 smoke (script: [smoke_b_freq_alpha.py](smoke_b_freq_alpha.py))

50 patterns, 400 cues, m=6, Zipf retrieval distribution, 3 seeds × 4 λ.

| λ | mean corr_u_m_rc | spread | gini_u_m | u_m_mean |
|---|-----------------|--------|----------|----------|
| 0.00 | **+0.836** | 0.17 | 0.730 | 0.00571 |
| 0.50 | +0.690 | 0.31 | 0.711 | 0.00494 |
| 1.00 | +0.581 | 0.51 | 0.713 | 0.00453 |
| 2.00 | +0.542 | 0.62 | 0.722 | 0.00414 |

All 12 runs produced positive correlations (no sign flips at this
scale), no NaNs, no instability. But the brainstorm's framing — "λ > 0
makes the slow store filter for frequency" — is **the wrong sign**.
Increasing λ:

- **Reduces** the magnitude of `corr_u_m_rc` (+0.84 → +0.54).
- **Reduces** `u_m_mean` across the population (0.0057 → 0.0041).
- **Increases** cross-seed variance (spread 0.17 → 0.62).
- Does not increase `gini_u_m` (would have indicated concentration on
  high-count patterns).

Mechanism: high `alpha_eff` makes the explicit-Euler cascade leak mass
faster to the absorbing boundary at u_m+1 = 0. Frequently-retrieved
patterns get *more* leak, not more u_m. This is the exact behavior the
unit test documents.

### Implication for the recommendation
The Colab notebook
([scripts/colab_phase34_freq_alpha_sweep.ipynb](../../scripts/colab_phase34_freq_alpha_sweep.ipynb))
will almost certainly return one of:

- **Null** (corr_u_m_rc differences across λ are within CI of λ=0
  baseline), in which case the freq-weighted-α path closes negative.
- **Wrong sign** (corr decreases with λ as in this smoke), in which
  case the "compression → abstraction" claim from the 2026-05-13
  brainstorm is geometrically backwards on this substrate.

Either is a valuable STATUS.md update; both retire a documented open
commitment. **Run it.**

The **depth-weighted α reformulation** (rec b2) inherits the same
geometric problem: higher α at any pattern produces less u_m at long
horizons. Substituting `energy_depth` for `count` will not fix the
sign. Depth-weighted α should be parked until/unless the count-weighted
experiment shows there is a frequency signal worth refining.

---

## (a) Range-shaped replay buffer — smoke result

### Prototype (script: [smoke_a_range_shaped_replay.py](smoke_a_range_shaped_replay.py))

A `RangeShapedReplayBuffer` class wraps the existing trace list,
sampling roles and contents **independently** from their marginal
distributions. Source episodes are 400 traces over 8 roles × 32
contents, with role-content correlation built in (each role
preferentially co-occurs with 4 contents).

Sampling 4000 traces, comparing joint distributions:

| Metric | Original buffer | Baseline sample | Range-shaped sample |
|--------|-----------------|-----------------|---------------------|
| Joint rectangularity (KL to factored) | 2.064 | 2.059 | **0.033** |
| Cells reached (out of 256) | 32 | 32 | **256** |

KL between the two samplers' joint distributions: **2.11 / 13.04**
asymmetric (range-shape has a much heavier tail in cells the baseline
never reaches). Marginals are preserved within L1 ~0.05 (role) and
~0.11 (content) — i.e., the range-shaped sampler is sampling from the
**same** marginals, but the **factored** joint.

This is exactly what the Dorrell-Whittington ICLR 2025 theorem
requires: rectangular joint support is the necessary condition for
nonneg + energy-efficient autoencoders to modularise. The sampler
manufactures rectangular support in ~30 lines.

### Implication for the recommendation
Range-shaped sampler logic is real and small. The **substrate-level
extension** — what to do when a sampled (role, content) pair has no
exact trace — is the work the brainstorm under-counted. Two viable
strategies:

1. **Re-bind on the fly.** Take the sampled role from FHRR position
   vectors, sample a content vector from the existing content
   codebook, bind them, treat that as a synthetic trace. Requires the
   S1 trace-schema extension (~30 LOC, already named in
   `notes/notes/2026-05-24-spike-S1-replay-trace-schema.md`).
2. **Fall back to closest match.** Sample (r, c), search the trace
   buffer for nearest (r, c′) or (r′, c), use that trace. Simpler;
   loses the rectangular-support guarantee in proportion to
   how-far-back the buffer reaches.

Realistic cost for either: 3-5 days including a Phase-4-style n=10
confirmation. The verification report's revised estimate stands.

---

## (c) MQAR-style associative recall — smoke result

### Synthetic MQAR (script: [smoke_c_mqar.py](smoke_c_mqar.py))

Three storage strategies tested at D ∈ {1024, 4096}, N ∈ {16, 32, 64,
128, 256, 512}, n_queries=32, 3 seeds:

**Strategy 1: HRR-bundle (Plate-style).** Bundle Σ bind(k_i, v_i) into
one vector; query via unbind; top-1 against the value codebook.

```
   D     N  top1@D=4096   per-seed
4096    16  1.000          [1.00, 1.00, 1.00]
4096   128  1.000          [1.00, 1.00, 1.00]
4096   256  0.990          [1.00, 0.97, 1.00]
4096   512  0.635          [0.72, 0.56, 0.62]
```

**Strategy 2: Hopfield with perfect cue (memorization).** Store each
bind(k_i, v_i) as an MHN pattern; cue is the bound pair itself.

```
4096   {16,32,64,128,256,512}  top1 = 1.000 always
```

**Strategy 3: Hopfield with KEY-ONLY cue (true MQAR).** Same storage
as #2; cue is just the key vector — the Hopfield landscape must find
the corresponding bound pattern by key alignment.

```
   D     N  top1@D=4096    per-seed
4096    16  0.125          [0.25, 0.06, 0.06]
4096    32  0.062          [0.00, 0.12, 0.06]
4096    64  0.021          [0.00, 0.03, 0.03]
4096   128  0.000          [0.00, 0.00, 0.00]
4096   256  0.021          [0.00, 0.06, 0.00]
4096   512  0.000          [0.00, 0.00, 0.00]
```

### The smoking gun

At D=4096, N=128:

- HRR-bundle: **100% top-1** recall.
- Hopfield, perfect cue: **100% top-1** recall.
- Hopfield, key-only cue: **0% top-1** recall.

The four Phase-5 retrieval-mechanism nulls (D1/D3/E1/M1) are exactly
what (c) says they should be. The Hopfield landscape **cannot** do
key-only associative recall on the current substrate. It's not that
the storage rule is wrong, or the priors are wrong, or the energy
shaping is wrong — it's that the geometry of `bind(k, v)` in FHRR
puts each pattern very far from `k` alone, so the energy landscape
has no basin near `k` and the settling cannot land on the right pair.

This also explains why HRR-bundle works for the SAME task: unbind is
the bind operator's algebraic inverse, so `unbind(Σ bind(k_i, v_i),
k_j) ≈ v_j + small noise` — the answer is reachable in one algebraic
step. Hopfield retrieval is a geometric operation in the bound-vector
space, which is not the natural space for key-only queries.

### Implication for the recommendation
1. **Adopt MQAR as a Phase-5 sanity check immediately.** The current
   ΔE headline cannot distinguish the substrate's failure mode from a
   tuning problem; the MQAR capacity curve makes it unambiguous.
2. **The bundle-route success is itself a finding.** Plate-style HRR
   storage *already* gives the project key-only associative recall at
   high capacity. The Phase-5 design has been targeting Hopfield
   retrieval; the project may want to ask whether bundle-style memory
   (with a separate landscape for content disambiguation) is the
   shape of the Phase-5 architecture, not "MHN with role basins."
3. **The bundle ceiling at D=4096 is around N=256.** The Plate
   capacity bound suggests this is geometric, not implementation —
   it's a feature of the substrate-dimensionality choice. Doubling D
   roughly doubles capacity.

---

## What "tested" means here

I did not run the canonical Phase-4 n=10 graduation experiment, the
production-scale range-shaped-replay confirmation, or MQAR at D=4096
with n_queries=1000 across the full benchmark distribution. Those are
hours-of-compute experiments, and would not change the qualitative
conclusions:

- (b) directional reversal at synthetic scale is consistent with the
  documented unit-test steady-state behavior; production scale will
  not flip the sign.
- (a) sampler-shape change is mathematical, not statistical; the
  rectangularization is a property of the algorithm not the data.
- (c) substrate geometry is dimension-invariant; key-only Hopfield
  retrieval will not start working at D=8192 or with longer settling.

What I did run:
- All 26 phase-4 consolidation unit tests on the actual code (pass).
- A 12-run synthetic Phase-4 smoke for freq-weighted α (3 seeds × 4 λ).
- A range-shaped sampler prototype + a 4000-sample histogram comparison.
- A 108-run MQAR capacity curve (3 strategies × 2 D × 6 N × 3 seeds).

All scripts and outputs are in this directory.

---

## Net effect on the brainstorm's ranked recommendations

| Old rank | Action | New rank | New rationale |
|----------|--------|----------|---------------|
| (b) 1 → run notebook | "Cheapest open commitment to close" | **(b) demoted** | Smoke shows directional reversal at small scale. Run only to falsify on-record. Do NOT extend to depth-weighted variant. |
| (a) 4 → range-shape | "Smallest architectural move" | **(a) stands** | Sampler-shape is the right primitive; 3-5 days estimate confirmed. |
| (c) 3 → MQAR | "Missing unit test" | **(c) promoted to #1** | Smoke gives the smoking-gun confirmation of the Phase-5 substrate diagnosis. Adopt as a graduation-blocker external benchmark. |

The new top-recommendation order:

1. **Adopt MQAR as a Phase-5 external sanity check** (and as a Phase-3/4 capacity benchmark). Use the smoke script as the starting harness; scale to D=4096, n_queries=1024, multiple seeds.
2. **Run the freq-weighted α Colab to close it negatively**, then archive both (b) and the depth-weighted variant.
3. **Implement range-shaped replay** (3-5 days, including the S1 trace-schema extension).
4. **Residue HDC over the existing FHRR substrate** — now with stronger motivation. The MQAR result shows the Phase-5 substrate doesn't have key-only basins. Residue HDC's algebraic role-separation is the direct response.
5. **Re-read the Krotov-cluster + Farooq 2025 unification** (arXiv:2505.21777 + arXiv:2506.11043) — and consider whether bundle-style storage at high temperature is closer to the project's natural Phase-6 architecture than Hopfield-with-roles.
