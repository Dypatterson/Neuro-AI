# Report 044 — Consolidation-geometry diagnostic: death collapses effective dimensionality ~10×

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism diagnostic, prerequisite step for path (a′) per the
post-research-review recommendation. **Not graduation, not falsification.**
Builds the load-bearing local-geometric signal that was specified
pre-Phase-3 in [consolidation-geometry-diagnostic.md](../notes/emergent-codebook/consolidation-geometry-diagnostic.md)
and listed as "pre-phase commitment still open" in STATUS, finally
populated.
**Diagnostic:** [scripts/consolidation_geometry_diagnostic.py](../scripts/consolidation_geometry_diagnostic.py)
**Per-snapshot outputs:** [reports/phase5_geometry_diag/seed{1,2,11,17,23}_step{1500,1800}.json](phase5_geometry_diag/)
**Companion:** [reports/phase5_memory_cliff/README.md](phase5_memory_cliff/)
(MESH-style memory-cliff check, ran in parallel)

## Experiment preamble

**Active phase:** 5
**Headline metric per [phase-5-checklist.md:39](../notes/emergent-codebook/phase-5-checklist.md):**
ΔE CI disjoint from zero, n≥10. **This report does not address the
headline.** It builds and runs a prerequisite local-geometric diagnostic
that informs whether death is the mechanism causing the K-branch
collapse identified in [report 042](042_phase5_branching_collapse_diagnostic.md)
and partially answered in [report 043](043_phase5_substrate_scale_diagnostic.md).
**Required controls per [consolidation-geometry-diagnostic.md:113-117](../notes/emergent-codebook/consolidation-geometry-diagnostic.md):**
the θ′(β) calibration spike is **NOT** run here (still open as a
pre-phase commitment); regime classification at β=10, θ′=1/β=0.1 is
uncalibrated and produces 0 tight atoms across all 10 snapshots. The
load-bearing finding is the d_eff ratio comparison, which does not
depend on the θ′ calibration.

**Last verified result:** [report 043](043_phase5_substrate_scale_diagnostic.md)
— pre-death branches separate by 3×–4978× over post-death; K-branch
mechanism not structurally degenerate; per-seed ΔE direction bimodal
even on pre-death.

**Why this experiment now:** Post-research-review (see this session's
transcript) recommended path (a′) — continuous-rate death as the
slow-timescale dynamic of which d̄/d_eff is a fast-timescale snapshot —
with two prerequisites: (i) build the consolidation-geometry diagnostic;
(ii) hold the diagnostic-actuator dynamic-form session. This report
closes (i). Note 2 closes (ii).

## Method

Operationalized [consolidation-geometry-diagnostic.md](../notes/emergent-codebook/consolidation-geometry-diagnostic.md)
against the substrate-state regime (the original spec assumed live
per-atom context-bag history during consolidation; snapshots don't
carry that history). Substitution: substrate-level d̄ / d_eff over the
full pattern matrix, and per-atom d̄_t / d_eff_t over each atom's k=5
nearest-neighbor cluster in FHRR space. Same geometric question;
different operationalization for the post-hoc snapshot case.

Anti-homunculus check applied: the script reports measurements; nothing
reads them and acts. The d̄ / d_eff numbers do not enter any
control-flow path in production code. ([scripts/consolidation_geometry_diagnostic.py](../scripts/consolidation_geometry_diagnostic.py)
docstring §"Anti-homunculus discipline".)

Implementation note: substituted the Gram matrix
`X̃ X̃^H / N ∈ ℂ^{N×N}` for the feature covariance
`X^H X / N ∈ ℂ^{D×D}` in the participation-ratio computation. The two
share their non-zero eigenvalues, so the ratio is identical; the
[N, N] eigh is O(N³) instead of O(D³). For N=1024, D=4096 this is
~64× speedup. The initial implementation called eigh on the [4096,
4096] complex matrix once per atom and was intractable on CPU.

## Headline diagnostic: substrate effective dimensionality collapses ~10× pre→post

10 snapshots: 5 seeds × {step 1500 (pre-death, n_atoms=1024), step 1800
(post-death, n_atoms ∈ {6, 10, 12})}, all W=4, dim=4096.

| Seed | Regime | n_atoms | substrate d̄ | substrate d_eff | d_eff / D | regime (β=10) |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
|  17 | pre   | 1024 | 0.7401 | **40.51** | 0.0099 | spread |
|  17 | post  |   12 | 0.6165 | **5.36**  | 0.0013 | spread |
|  11 | pre   | 1024 | 0.7506 | **42.67** | 0.0104 | spread |
|  11 | post  |   10 | 0.5377 | **3.88**  | 0.0009 | spread |
|  23 | pre   | 1024 | 0.7534 | **40.98** | 0.0100 | spread |
|  23 | post  |    6 | 0.4707 | **2.52**  | 0.0006 | spread |
|   1 | pre   | 1024 | 0.7500 | **45.38** | 0.0111 | spread |
|   1 | post  |   12 | 0.5420 | **3.91**  | 0.0010 | spread |
|   2 | pre   | 1024 | 0.7517 | **42.61** | 0.0104 | spread |
|   2 | post  |   12 | 0.4591 | **6.41**  | 0.0016 | spread |

**Headline finding: across all 5 seeds, d_eff collapses by 7×–16× from
pre-death to post-death.** Pre-death sits at ~40 effective dimensions
across all seeds (tight cluster around 40.51–45.38). Post-death sits at
~2.5–6.4 effective dimensions. The substrate's reachable subspace
collapses from ~40 dimensions to ~5 dimensions in a single death event.

**This is the geometric mechanism for K-branch collapse from [report 042](042_phase5_branching_collapse_diagnostic.md).**
With 4 branches and ~5 effective dimensions, settling drains them to a
degenerate point — there isn't enough span for K distinct attractors.
The B2 control firing in [report 041](041_phase5_de_n5_partial.md) and
the equi-energetic branches finding in
[report 042](042_phase5_branching_collapse_diagnostic.md) are downstream
of this single geometric fact.

Pairwise distance d̄ also drops, by ~25–40% (0.75 → 0.46–0.62). But d̄
alone doesn't capture the collapse — the atoms in the surviving cluster
are still moderately far apart pairwise. The dominant story is **the
subspace they span**, not the distance between them.

## Per-atom diagnostics

For each atom, d̄_t / d_eff_t over its k=5 nearest-neighbor cluster:

| Seed | Regime | n_atoms | per-atom d̄ mean | per-atom d_eff mean | n_tight (β=10) |
| ---: | --- | ---: | ---: | ---: | ---: |
|  17 | pre   | 1024 | 0.442 | 3.66 | 15 / 1024 (1.5%) |
|  17 | post  |   12 | 0.524 | 3.23 | 0 / 12 |
|  11 | pre   | 1024 | 0.458 | 3.71 | 9 / 1024 (0.9%) |
|  11 | post  |   10 | 0.421 | 2.99 | 0 / 10 |
|  23 | pre   | 1024 | 0.449 | 3.63 | 14 / 1024 (1.4%) |
|  23 | post  |    6 | 0.471 | 2.33 | 0 / 6 |
|   1 | pre   | 1024 | 0.463 | 3.72 | 0 / 1024 (0%) |
|   1 | post  |   12 | 0.371 | 2.94 | 0 / 12 |
|   2 | pre   | 1024 | 0.453 | 3.69 | 12 / 1024 (1.2%) |
|   2 | post  |   12 | 0.360 | 3.19 | 0 / 12 |

Per-atom d_eff is ~3.7 pre-death and ~3.0 post-death (modest shift).
This is k=5 nearest neighbors, so the maximum possible per-atom d_eff
is 5; saturating near 3.0–3.7 means most atoms' immediate neighborhoods
are already low-dimensional, even pre-death. Death narrows this further
but the bulk of the dimensionality story is at the substrate level, not
the per-atom level.

## Regime classification calibration is uninformative

At β=10, θ′ = 1/β = 0.1, n_tight = 0 across all 10 substrate-level
classifications and 0 across post-death per-atom classifications. The
spec ([consolidation-geometry-diagnostic.md:113-117](../notes/emergent-codebook/consolidation-geometry-diagnostic.md))
explicitly flags this approximation as needing the empirical θ′(β)
calibration spike — listed as a pre-phase commitment that is still
open. **Until that spike runs, the tight/spread regime classification
is uncalibrated.** The load-bearing finding (d_eff collapses ~10×) does
not depend on it.

This is itself a small finding for the design note: any continuous-rate
death dynamic that consumes a tight/spread classifier as input would
need that calibration to run first. A dynamic that consumes d_eff
directly would not.

## Cross-reference: MESH-style memory-cliff check

[reports/phase5_memory_cliff/README.md](phase5_memory_cliff/README.md)
ran in parallel. Setup: TorchFHRR + TorchHopfieldMemory at β=10,
n_atoms ∈ {2..1024}, noise scales ∈ {0.1..0.9}, n=20 trials × 3 seeds.

**Headline finding from the cliff check: no memory cliff in the
relevant noise range for any n_atoms ≥ 2.** Post-death sizes (6, 10,
12) and pre-death (1024) all retrieve at recall=1.000 across the spec
noise range. Even at noise=24 (well beyond plausible), post-death sizes
still recall 77–90%.

**Combined read:** the substrate at n_atoms=6 is *trivially separable
for single-pattern retrieval* (MESH check), and yet the K-branch
mechanism collapses on it (reports 042, 043) because the *substrate's
effective dimensionality* is 5, not because the *capacity* is
insufficient. The two diagnostics are testing different things, and
together they bracket the failure: capacity is fine; subspace span is
insufficient for K-branch separation.

## What this rules out

- **Death-induced loss of retrieval capacity** as the mechanism. The
  MESH check is decisive: n_atoms=6 retrieves perfectly. Capacity is
  not the issue.
- **Pairwise co-location alone** as the mechanism. d̄ does drop, but
  not catastrophically. Atoms are not "clustered tightly"; they are
  "embedded in a low-dimensional subspace."

## What this rules in

- **Substrate effective-dimensionality collapse** as the mechanism for
  K-branch failure. d_eff ~5 post-death cannot support K=4 distinct
  attractors. A death mechanism that *preserves* d_eff would, in
  principle, restore branch separation.
- **Continuous-rate death candidates with a d_eff or coverage signal**
  as the right shape for a redesigned mechanism. The d_eff per atom
  (or substrate-level) is a measurable local-geometric quantity that a
  slow-timescale dynamic can be coupled to in the diagnostic-actuator
  sense.

## What this does NOT rule out

- **A second failure mode beyond d_eff collapse.** [Report 043](043_phase5_substrate_scale_diagnostic.md)
  showed per-seed ΔE direction is bimodal even on pre-death where
  d_eff ≈ 40. So preserving d_eff is necessary but not sufficient. The
  cue regime / role-prior asymmetry is a second axis (the next
  prerequisite for any graduation-level run).
- **The θ′(β) calibration question.** Regime classification at the
  current β=10 approximation is uninformative; the spike remains open.
- **Whether d_eff is the right input to the actuator.** Coverage,
  effective_strength (Benna-Fusi), or some combination might be more
  natural for the slow-timescale dynamic.

## Checklist updates

| Item | Before this report | After this report |
| --- | :---: | :---: |
| Pre-phase commitment: consolidation-geometry regime classifier (d̄, d_eff per atom) | 🟦 deferred (specified pre-Phase-3, never built) | ✅ **built** ([scripts/consolidation_geometry_diagnostic.py](../scripts/consolidation_geometry_diagnostic.py)) — substrate-state operationalization; original live-consolidation form remains for future implementation |
| Pre-phase commitment: empirical θ′(β) calibration spike | 🟦 deferred (recommended pre-Phase-3, never done) | 🟦 still open — current β=10, θ′=0.1 produces uninformative regime classification; calibration would resolve |
| STATUS blocker #4 (diagnostic-actuator dynamic-form session) | ❌ open since 2026-05-09 | 🟨 prerequisite (d_eff diagnostic) built; session still to be held in Note 2 |
| Phase 5 E5 (branch-state diversity) | ⚠️ post-death collapses; pre-death restores | ⚠️ **mechanism identified: d_eff/D collapses from 0.010 to 0.001 across all 5 seeds; with K=4 branches and d_eff~5 post-death, branches cannot occupy distinct subspaces** |

## Implications for next steps

This report closes the first of the two prerequisites for path (a′).
The signal a continuous-rate death dynamic should be coupled to is
**substrate-level d_eff** (or a related quantity), not pairwise
similarity. The post-death d_eff/D ratio (~0.001) is ~10× too small to
support K=4 branching with the current bundle+re-settle combiner. A
soft-death mechanism that preserves d_eff at ~0.01 (pre-death levels)
would, in principle, restore the geometric conditions for branching to
work.

The second prerequisite is the diagnostic-actuator dynamic-form
session — Note 2, written next. That note must:

1. Frame the slow-timescale dynamic as one whose fast-timescale snapshot
   is d_eff (or a related quantity), not as "if d_eff < threshold then
   stop pruning."
2. Enumerate 2–3 candidate continuous local dynamics that would
   preserve d_eff while still producing under-capacity (per brainstorm
   Idea 5 / freq-weighted α reasoning).
3. Anti-homunculus-audit each before any code lands.

After Note 2 lands, the path-(a′) commit decision is informed by both
geometric prerequisite (this report) and dynamic-form discipline
(Note 2). Only then does the Phase 4 retrain become the right next
unit of compute work.
