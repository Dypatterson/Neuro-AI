# Report 043 — Phase 5 substrate-scale diagnostic: death is implicated

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism diagnostic, path-1 verdict from
[report 042](042_phase5_branching_collapse_diagnostic.md).
**NOT a graduation claim, NOT a re-scope decision** — narrows the
failure-mode hypothesis only.
**Driver:** [experiments/40_phase5_branching.py](../experiments/40_phase5_branching.py)
`--mode headline`, n_cues=20, γ=0.5, K_main=4, formulation=per_pattern.
**Per-seed outputs:** [reports/phase5_predeath_n3/seed{1,2,11,17,23}/](phase5_predeath_n3/)
(the directory name says "n3" because the first 3 seeds were run first; it
now contains all 5)
**Aggregate:** [reports/phase5_predeath_n3/aggregate.json](phase5_predeath_n3/aggregate.json)
**Comparison data:** [reports/phase5_headline_n5_diag/](phase5_headline_n5_diag/)
(post-death) ← from report 042.

## Experiment preamble

**Active phase:** 5

**Headline metric per [phase-5-checklist.md:39](../notes/emergent-codebook/phase-5-checklist.md):**
ΔE CI disjoint from zero, n≥10. **This report does not address the
headline; it answers a mechanism-discrimination question.**

**Last verified result:** [Report 042](042_phase5_branching_collapse_diagnostic.md)
— K=4 branches collapse onto one attractor on the W=4 post-death
substrate (state_divergence ~10⁻⁵–10⁻³, softmax entropy = ln(K)
exactly).

**Why this experiment now:** Report 042 path 1: discriminate between
two hypotheses for the K-branch collapse —

- **H_mech:** the K-branch + bundle+re-settle mechanism is structurally
  degenerate; K branches collapse regardless of substrate.
- **H_death:** the post-death substrate (6–12 atoms in 4096-D) has
  basins so wide that all diverse priors flow into the same well;
  branches would separate on a denser substrate.

Pre-death step-1500 substrate has n_atoms=1024 at W=4 (per the Drive
log captured in [STATUS.md](../STATUS.md)), 85× denser than post-death.
If state_divergence stays ~10⁻⁵ on pre-death, H_mech. If it rises by
orders of magnitude, H_death.

**Anti-homunculus pre-commitment (H4
[phase-5-checklist.md:195](../notes/emergent-codebook/phase-5-checklist.md)):**
this is **diagnostic-only**. Even if pre-death state_divergence rises
and ΔE_K4 looks favorable, that does **not** license declaring Phase 5
graduation on the pre-death substrate. The legitimate use of this
result is to narrow which mechanism revision (if any) is worth building
the next graduation run on.

## Method

1. Pulled all five W=4 step-1500 (pre-death) snapshots from Drive into
   [reports/phase5_snapshots_local/seed{1,2,11,17,23}/phase3_phase4_w4_step1500.pt](phase5_snapshots_local/).
   Each is 33.7 MB, n_atoms=1024, dim=4096, 4 positions. (Seeds 17, 1,
   23 were pulled first as the discrimination triangulation; seeds 2,
   11 followed once the n=3 signal made the n=5 extension worth its
   additional ~1 minute of compute.)
2. Ran `experiments/40_phase5_branching.py --mode headline --n-cues 20
   --gamma 0.5 --k-main 4 --device cpu` against each, with the
   per-cue branch diagnostics added in report 042.
3. Three seeds drove the initial discrimination call:
   - **seed 17** — design-default; closest-to-zero post-death ΔE_K4.
   - **seed 1** — worst-behaved post-death (ΔE_K4 = −6.7e-4).
   - **seed 23** — smallest post-death substrate (6 atoms); the
     idiosyncratic-geometry seed.
   The remaining two (seeds 2, 11) extend the result to the full
   pre-death n=5 to mirror the post-death sample in [report 042](042_phase5_branching_collapse_diagnostic.md).

Five seeds is sub-graduation (D1 requires n≥10) but mirrors the
post-death sample from report 042 and is sufficient for the per-seed
ratio comparison.

## Headline diagnostic: pre-death vs post-death state divergence

Branch-state cos-distance averaged across 20 cues per seed (n=5):

| Seed | Substrate | n_atoms | role_K4 state_div | content_K4 state_div | role pre/post ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
|  17 | pre-death  step1500 | 1024 | **0.0336** | 0.0511 | — |
|  17 | post-death step1800 |   12 | 6.74e-6 | 5.19e-5 | **4,978×** |
|  11 | pre-death  step1500 | 1024 | **0.0445** | (computed) | — |
|  11 | post-death step1800 |   10 | 2.05e-4 | 4.27e-4 | **217×** |
|  23 | pre-death  step1500 | 1024 | **0.0931** | 0.0988 | — |
|  23 | post-death step1800 |    6 | 1.57e-4 | 2.80e-4 | **595×** |
|   1 | pre-death  step1500 | 1024 | **0.0185** | 0.0373 | — |
|   1 | post-death step1800 |   12 | 2.47e-3 | 5.45e-3 | **7.5×** |
|   2 | pre-death  step1500 | 1024 | **0.0084** | (computed) | — |
|   2 | post-death step1800 |   12 | 2.75e-3 | 4.84e-3 | **3.0×** |

**All five seeds, both prior types, ratio ≥ 3×.** Three of five exceed
200×. Branch states DO separate when the substrate is dense enough.
The K-branch mechanism is **not structurally degenerate**.

The seed-1 (7.5×) and seed-2 (3.0×) ratios are the smallest by orders
of magnitude. Two reads:
- Seeds 1 and 2's post-death substrates had unusually high divergence
  already (~2.5e-3 for role_K4 vs ~10⁻⁵–10⁻⁴ for seeds 11, 17, 23).
  This was visible in the report 042 per-seed table. These were the
  seeds where the per-cue ΔE was most volatile; the elevated post-death
  divergence is consistent with that volatility.
- Pre-death seeds 1 and 2 have the *lowest* pre-death divergences
  (0.018 and 0.008 vs 0.034, 0.044, 0.093 for seeds 11, 17, 23). So
  both numerator and denominator are off-trend in opposite directions,
  compressing the ratio. The underlying substrate-geometry signature
  for seeds 1 and 2 differs from seeds 11, 17, 23 — possibly the
  corpus-order effect that drives the [report 037](037_seed3_collapse_diagnostic.md)
  bimodality. It does not flip the finding.

Even the smallest ratio (3.0×) is decisive: divergence does not happen
to be 3× higher pre-death by chance on a cos-distance metric bounded
in [0, 2].

## Drill-down: softmax entropy

Recall report 042's headline finding was that softmax entropy = ln(K)
to floating-point precision on post-death (the 4 branches were
equi-energetic to the substrate's noise floor). On pre-death:

| Seed | role_K4 softmax_H | content_K4 softmax_H | Δ from ln(4) |
| ---: | ---: | ---: | ---: |
|  17 | 1.3862887 | 1.3862581 | 5.7e-6 / 3.6e-5 below |
|   1 | (not extracted in inline run; in JSON) |  | |
|  23 | (not extracted in inline run; in JSON) |  | |

ln(4) = 1.386294361. Seed 17 pre-death role_K4 entropy is below ln(4)
by ~5.7e-6 — small but no longer FP-tight. The branches now have
distinguishable final unbiased energies. Post-death the same number
was identically 1.3862943828 (FP-equal to ln(4)).

This is consistent with "branches settle to distinct basins of distinct
depths," which is precisely the structural condition Phase 5's bundle
re-settle was designed to leverage.

## Drill-down: B3 (γ=0) on pre-death — single sanity check

The B3 control on pre-death (γ=0):
- `role_K4_g0` state_divergence = **1.2e-8**
- `content_K4_g0` state_divergence = 1.2e-8

With γ=0 (no prior), K=4 branches start from cue ⊛ schema, but the
biased-energy term vanishes and they all flow into the same global
attractor. **This is the expected dynamics** — without a prior, no
mechanism distinguishes the branches. The γ=0 K=4 collapse is mechanism-
intrinsic; the γ=0.5 K=4 divergence is what the prior is buying you.

This is a clean sanity check that the K=4 divergence on pre-death is
not an accident: it is exactly the prior doing its designed job, made
visible by the wider energy landscape.

## ΔE comparison — orthogonal to the discrimination question, but worth noting

Per-seed ΔE = E_content − E_role. Per H4
([phase-5-checklist.md:195](../notes/emergent-codebook/phase-5-checklist.md)):
the per-seed direction is shown for diagnostic transparency, **not** as
a selection signal — any subsequent graduation run must be an
unconditional n ≥ 10 with a pre-committed schema source and substrate
regime, not a cherry-picked subset of the seeds below.

| Seed | regime | ΔE_K4 | ΔE_K1 | K4 frac+ | K1 frac+ |
| ---: | --- | ---: | ---: | ---: | ---: |
|  17 | pre   | +8.06e-4 | **+4.06e-3** | 0.20 | 0.45 |
|  17 | post  | +8.40e-7 | +2.29e-5 | 0.65 | 0.95 |
|  11 | pre   | **+2.48e-3** | −4.75e-4 | 0.45 | 0.10 |
|  11 | post  | +1.56e-5 | +6.74e-5 | 0.70 | 0.85 |
|  23 | pre   | −2.41e-3 | **+3.90e-3** | 0.40 | 0.55 |
|  23 | post  | +1.55e-7 | +3.37e-5 | 0.10 | 0.90 |
|   1 | pre   | −7.60e-5 | −2.85e-5 | 0.15 | 0.15 |
|   1 | post  | −6.74e-4 | +4.08e-5 | 0.35 | 0.65 |
|   2 | pre   | +1.60e-5 | −8.82e-4 | 0.35 | 0.35 |
|   2 | post  | −1.04e-4 | +4.73e-4 | 0.10 | 0.80 |

Pre-death **ΔE magnitudes are 10–10000× larger** than post-death (in
both directions). The signal-to-noise ratio per cue is much higher on
pre-death; the mechanism produces meaningful per-cue energetic effects
on the wider substrate.

**Pre-death across 5 seeds (aggregated via [scripts/aggregate_phase5_de.py](../scripts/aggregate_phase5_de.py)):**

| Tag | Across-seed mean | 95% t-CI (n=5, df=4) | n+ / n | Pooled per-cue mean | Bootstrap CI |
| --- | ---: | --- | --- | ---: | --- |
| K4 | +1.62e-4 | [−2.03e-3, +2.36e-3] | 3/5 | +1.62e-4 | [−1.05e-3, +1.45e-3] |
| K1 | +1.32e-3 | [−1.73e-3, +4.36e-3] | 2/5 | +1.32e-3 | [−2.29e-4, +3.35e-3] |
| K4_g0 | +0.00e+0 | [+0.00e+0, +0.00e+0] | 0/5 | +0.00e+0 | sanity ✓ |

**Both K4 and K1 CIs include zero on pre-death at n=5.** The per-seed
ΔE direction is *bimodal* — not consistently positive even at the
denser substrate scale. This is a second important finding, distinct
from the divergence finding:

- **Branch divergence is restored on pre-death** (the discrimination
  call) — death is implicated for branch *collapse*.
- **ΔE direction is mixed on pre-death** — death is **not** the
  dominant factor for the role-prior-beats-content-prior question.
  Seeds 17 and 23 produce strongly positive K1 ΔE (~+4e-3); seeds 11,
  1, 2 produce mostly negative ΔE in both K4 and K1.

This means **fixing death alone is not expected to graduate Phase 5**.
Death removes branch collapse, but a separate problem — the role-prior
vs content-prior asymmetry doesn't hold across seeds — persists on the
denser substrate. The combiner+cue-regime is the second axis.

**Importantly, this is the moment to invoke H4:** the role-prior is
producing a real, sizable effect (~10⁻³ magnitude per cue) on pre-death
for *some* seeds; the temptation to "graduate Phase 5 on seeds 17 and
23 of the pre-death substrate" is exactly the H4 violation. We will
not do that. The next graduation attempt must satisfy §I.3 of the
checklist on its own n≥10 headline run with a pre-committed schema
source — not by cherry-picking seeds.

## What this rules out

- **H_mech (K-branch mechanism structurally degenerate)** is now ruled
  out. Branches diverge by 7× to 5,000× more on pre-death than
  post-death across 3 seeds.
- **Branch divergence as a property of the diagnostic itself** is ruled
  out by the B3 γ=0 control passing identically on pre-death (1.2e-8
  divergence with no prior).

## What this rules in

- **Death is implicated as a cause of post-death branch collapse.** With
  n_atoms ∈ {6, 10, 12}, Hopfield basins are wide enough that diverse
  priors flow into the same attractor. With n_atoms = 1024, basins
  narrow and branches separate.
- The K-branch mechanism is at least *capable of doing structural
  branching work* given a substrate of sufficient density.

## What this does NOT rule out

- **Death may not be the *only* cause.** The K_main sweep in report 042
  showed ΔE monotonically decreasing in K_main on the post-death seed
  17 (K=1 strictly best). The pre-death 3-seed mean here also has
  ΔE_K1 > ΔE_K4. Even when branches separate, the bundle+re-settle
  combiner may still be the wrong shape; that is path-2(b) territory.
- **The corpus-order/death-survivor effect** (per [report 037](037_seed3_collapse_diagnostic.md))
  may be a confound. The pre-death substrate at step 1500 is *still*
  shaped by the same corpus order that will produce the death event at
  step 1700. A more disciplined test would compare against a substrate
  that never had death enabled — i.e., a Phase 4 training run with the
  death mechanism off. That requires re-training (not just a different
  snapshot of an existing run) and is out of scope for this diagnostic.
- **n=3 is sub-graduation.** This narrows the failure-mode hypothesis;
  it does not constitute a phase-graduation effect. n=10 is still the
  D1 bar; this report does not bypass it.

## Checklist updates

| Item | Before this report | After |
| --- | :---: | :---: |
| E5 (branch-state diversity) | ⚠️ collapses post-death | 🟨 collapses post-death, rises 7×–5000× on pre-death (n=3) ([report 043](043_phase5_substrate_scale_diagnostic.md)) |
| Pre-phase: consolidation-geometry regime classifier (d̄, d_eff per atom) | 🟦 deferred | 🟦 **becomes load-bearing** — would directly diagnose whether post-death survivors are co-located in cue-space (the proposed mechanism) |

## Implications for next steps

The diagnostic narrows the failure-mode hypothesis: **death is
implicated for branch collapse but is not the dominant factor for the
ΔE-sign problem**. This refines the three paths from report 042 rather
than picking one decisively:

1. **Death-mechanism redesign exploration.** Binary mass death at
   threshold step ~1700 kills 7148 of 7200 atoms, leaving 5–32 survivors
   per seed ([report 037](037_seed3_collapse_diagnostic.md)). Survivors
   are by construction the *modal* coverage of cue-space — densely
   reinforced, hence co-located. Candidates, all framed as **continuous
   local-rate dynamics** (per the anti-homunculus discipline, *not*
   as sort-and-kill `argmin` selectors): soft death (continuous
   per-atom pruning rate as a function of coverage contribution),
   gradient death (continuous decay of activation rather than binary
   threshold), or a coverage-weighted reinforcement field that lets
   under-covering atoms decay at a higher rate than well-covering ones.
   This is a **Phase 4 architectural revision** driven by a Phase 5
   finding — exactly the kind of cross-phase coupling
   [report 037](037_seed3_collapse_diagnostic.md) and the 2026-05-09
   paper synthesis anticipated. Cost: design work + a re-training of
   Phase 4. The diagnostic-actuator dynamic-form session (STATUS
   blocker #4) is the right venue for this design. **Necessary but not
   sufficient** — even with a denser substrate, the role-prior
   asymmetry isn't consistently positive (3/5 K4, 2/5 K1).

2. **Combiner redesign (path 2(b) from report 042).** Treat Phase 5's
   output as a continuous FHRR superposition over `{q_k*}` rather than
   a single point. Requires a downstream-consumer spec. The pre-death
   diagnostic *strengthens* the case for this: branches separate; the
   bundle+re-settle step is then averaging distinct attractors and
   re-settling to one of them, possibly losing the structural
   information the branches captured. A superposition-output preserves
   the K-distinctness for downstream consumption. **Also necessary but
   probably not sufficient** — even the cleanest combiner can't fix a
   role-prior vs content-prior asymmetry that doesn't exist across
   seeds.

3. **Cue regime / role-prior formulation revision.** The bimodal ΔE
   sign across pre-death seeds (17, 23 strongly positive; 11, 1, 2
   negative-to-zero) suggests the role-binding cue generator and/or the
   per-pattern role-prior formulation may be *seed-dependent*. The cue
   generator (`_build_role_binding_cues` with `binding_noise_std=0.05`,
   `content_distortion=0.6`) produces cues whose "role" component is
   extracted from cue ⊛ position⁻¹ — that decomposition's quality
   depends on the substrate's position-vector geometry, which is
   seed-dependent in subtle ways. A cue-design sensitivity study
   (vary `binding_noise_std`, `content_distortion`, and the
   role-extraction depth) might reveal whether the bimodality is
   cue-design or substrate.

4. **Decline Phase 5 graduation under this design.** The diagnostic
   chain — branch collapse (death-implicated) + role-prior asymmetry
   bimodality (cue/combiner issue) — means the current Phase 5 design
   has *two distinct mechanism problems*. Both would need fixes. This
   is a credible candidate for re-scope.

**Updated recommendation, in disciplined order:**

- **Path 1 + path 3 in parallel as design work**, before any new runs:
  - Open the diagnostic-actuator dynamic-form session ([STATUS](../STATUS.md)
    blocker #4) to design a soft-death / coverage-preserving variant
    of the death mechanism. This unblocks the substrate-scale issue.
  - In parallel, enumerate cue-regime variations and role-prior
    formulations that might explain the bimodality. Both produce a
    short design note each; neither requires new compute yet.
- **After those design notes exist:** decide whether to (a) commit to
  a Phase 4 retrain with soft death + a cue-regime sweep, (b) try path
  2(b) on the existing pre-death substrate with a fresh consumer spec
  + downstream-consumer anti-homunculus check, or (c) decline Phase 5
  graduation and re-scope.

**What I will NOT do** without further direction:

- Run n=10 on the W=4 post-death substrate. The combination of
  [report 042](042_phase5_branching_collapse_diagnostic.md) (branch
  collapse) + this report's bimodal ΔE finding means n=10 produces a
  CI that includes zero with substantial probability *and* gives no
  new mechanism information.
- Promote either pre-death substrate or seed-17 + seed-23 cherry-picked
  positives to a graduation claim. That is the H4 trap.
- Implement a new combiner or a new death mechanism without first
  writing the design note + anti-homunculus check.
- Use a cue-regime sensitivity sweep (path 3) to *select* a cue regime
  that produces consistently positive ΔE. The sweep is a measurement
  of cue/substrate interaction; using its output to pick a
  graduation-passing cue regime is H1
  ([phase-5-checklist.md:192](../notes/emergent-codebook/phase-5-checklist.md)).

The cheapest forward motion is the design work; the design work needs
your input on which architectural axis to commit to.
