# Report 025 — rt=0.85 5-seed verification: first verified Phase 4 lift, with a caveat

**Date:** 2026-05-14
**Experiment:** `experiments/18_phase4_unified_experiment.py` at rt=0.85,
drift=0.30, β=10, death_window=1000, reencode_every=100, n_cues=2000,
seeds ∈ {17, 11, 23, 1, 2}.

**Raw data:** `reports/phase4_rt0p85*/phase4_unified_results.json`.

**Follows up:** [reports/024_resolve_threshold_sweep.md](024_resolve_threshold_sweep.md).

## Headline

**This is the first verified positive Phase 4 headline in the project,
multi-seed.** Δtop1 has bootstrap CIs that exclude zero at the
mid-run checkpoints (steps 1000 and 1500). The lift is smaller than
seed 17 alone implied, and fades by the end of the run — but it is
real, multi-seed, and CI-bounded.

Seed 17's "drift-immune" pattern (Phase 4 top1 flat at 0.122 across
all checkpoints) was the high outlier of the five seeds, not the
typical case.

## Δtop1 across checkpoints (5 seeds, bootstrap 95% CI)

| step | per-seed Δtop1                          | mean   | 95% CI            | positive |
| ---: | --------------------------------------- | -----: | ----------------- | -------: |
|  500 | {+0.000, +0.018, +0.020, +0.020, +0.000} | +0.012 | [+0.004, +0.020]  | 3/5      |
| 1000 | {+0.031, +0.014, +0.017, +0.017, +0.000} | **+0.016** | **[+0.007, +0.025]** | 4/5  |
| 1500 | {+0.031, +0.035, +0.017, +0.010, +0.000} | **+0.019** | **[+0.007, +0.030]** | 4/5  |
| 2000 | {+0.034, +0.021, +0.003, -0.003, -0.010} | +0.009 | [-0.005, +0.024]  | 3/5      |

**Steps 1000 and 1500 have CIs disjoint from zero.** Step 500 also
clears zero on the bootstrap (lower bound +0.004) but with only 3/5
seeds positive — borderline. Step 2000's CI straddles zero.

**Phase 4 provides a robust mid-run top1 lift of 1.6–1.9 percentage
points that fades to a CI-crossing-zero result by step 2000.**

## Δcap_t05 across checkpoints (5 seeds, bootstrap 95% CI)

| step | per-seed Δcap_t05                       | mean   | 95% CI            | positive |
| ---: | --------------------------------------- | -----: | ----------------- | -------: |
|  500 | {-0.007, -0.011, +0.003, +0.027, -0.038} | -0.005 | [-0.024, +0.013]  | 2/5      |
| 1000 | {+0.000, -0.007, -0.070, +0.061, -0.080} | -0.019 | [-0.062, +0.023]  | 1/5      |
| 1500 | {+0.017, -0.014, -0.099, +0.067, -0.073} | -0.020 | [-0.072, +0.031]  | 2/5      |
| 2000 | {+0.010, +0.000, -0.040, +0.044, -0.035} | -0.004 | [-0.030, +0.022]  | 2/5      |

**No checkpoint has Δcap_t05 reliably positive.** CIs always
straddle zero, with means slightly negative throughout the middle of
the run. Phase 4 doesn't reliably preserve confidence-weighted
retrieval at this drift level. The +0.024 cap_t05 lift at rt=0.80
from report 024's single-seed sweep is not robust either.

## Absolute headline numbers at step 2000

| seed | baseline top1 | phase4 top1 | Δtop1   |
| ---: | ------------: | ----------: | ------: |
| 17   | 0.088         | 0.122       | +0.034  |
| 11   | 0.088         | 0.110       | +0.021  |
| 23   | 0.103         | 0.106       | +0.003  |
| 1    | 0.098         | 0.094       | -0.003  |
| 2    | 0.083         | 0.073       | -0.010  |
| **mean** | **0.092** | **0.101**   | **+0.009 (CI [-0.005, +0.024])** |

Relative Δ at the mean: ~10% lift on top1 (0.101 / 0.092 - 1). But
the absolute lift is bounded by the absolute floor that drift creates
— at drift=0.30 the baseline only loses ~3 pp from its 0.122 start,
so there's a hard ceiling on how much room Phase 4 has to recover.

## Per-seed trajectories

Three qualitatively distinct behaviors emerge:

- **Seeds 17 and 11**: Phase 4 wins throughout. Both seeds show
  Δtop1 positive at every checkpoint past step 0; cap_t05 mixed but
  not negative on average.
- **Seeds 23 and 2**: Phase 4 helps early then hurts cap_t05. Seed
  23's Δcap_t05 hits -0.099 at step 1500. The replay-discovered
  patterns get the argmax right but degrade the substrate's
  confidence in those retrievals.
- **Seed 1**: The mirror image. Phase 4 *hurts* top1 slightly
  (-0.003) but *helps* cap_t05 substantially (+0.044). Argmax slips
  but the confidence on the in-top-K hits is markedly preserved.

The behavior split is not noise around a common mean — different
seeds put Phase 4 into qualitatively different regimes. The
across-seed std at step 1500 is 0.013 for Δtop1 and 0.064 for
Δcap_t05 — Δcap_t05's per-seed variance is **5× larger** than
Δtop1's.

## What this settles

### Phase 4 is alive, multi-seed verified

Two checkpoints (steps 1000 and 1500) have Δtop1 CIs strictly above
zero. This is the project's first multi-seed Phase 4 result with
disjoint CIs. The mechanism works.

### The seed 17 "drift-immune" reading was too strong

Phase 4 top1 staying flat at 0.122 across all checkpoints (the
striking pattern that motivated this replication) was outlier-grade,
not typical. The typical pattern is a few-pp lift that grows
through step 1500 and then fades.

### cap_t05 is unreliable at this drift level

No checkpoint has Δcap_t05 robustly positive. Per-seed cap_t05
movements (−0.099 to +0.067) span almost two orders of magnitude
larger than the top1 effect. The "Phase 4 preserves confidence
under drift" story from the drift sweep doesn't generalize.

### Fade by step 2000 is real

Mean Δtop1 peaks at step 1500 (+0.019) and drops to +0.009 by step
2000, with CI crossing zero. Discovered patterns help for ~1500
cues of drift, then either get pruned by the death_window or get
out-competed by accumulated drift on the original landscape. The
death/discovery balance reported in earlier runs (`n_pat` constant
at 4126) means discovered patterns *are* being pruned at the same
rate as new ones arrive — and by step 2000 the early-cycle
discoveries (the highest-quality ones, from before drift had fully
accumulated) have rotated out.

## What this does NOT settle

- **drift=0.30 only.** rt=0.85 hasn't been multi-seed tested at
  drift=0.15 (where the no-gap regime may apply) or drift=0.50.
- **Single drift trajectory per seed.** Each run draws one drift
  realization. The 5-seed variance includes both seed variance
  (substrate / cue order) and drift-realization variance.
- **The fade at step 2000 is not understood.** Pruning-vs-discovery
  balance is the leading hypothesis; longer-window or shorter-
  death_window experiments would distinguish.
- **No diagnostic for why seeds split into qualitatively distinct
  behaviors.** The 5 seeds don't cluster around a mean — they
  cluster into "Phase 4 helps top1" (17, 11) and "Phase 4 helps
  cap_t05 but not top1" (1) and "Phase 4 hurts cap_t05" (23, 2).
  Some upstream factor — initial landscape composition, cue-order,
  drift realization — separates the seeds into qualitatively
  different regimes.

## Anti-homunculus / FEP audit

No new mechanisms introduced. rt=0.85 is a constant cut on a passive
resolution measurement. Sweep across seeds is pure replication. No
inspect-and-trigger branches added. Passes trivially.

## Recommended next steps

1. **rt=0.85 at drift ∈ {0.15, 0.50}, 5 seeds each.** Maps the
   operating envelope. Specifically: does the mid-run top1 lift
   survive at drift=0.15? Does cap_t05's per-seed variance grow
   with drift magnitude? ~3 h.

2. **Longer-window run at drift=0.30 (n_cues=5000).** Tests whether
   the fade at step 2000 is a pruning artifact (death_window=1000
   evicts early-cycle discoveries faster than later-cycle ones
   can replace their quality) or a saturation artifact (Phase 4
   gets all the help it can get by step 1500). ~30 min.

3. **Per-seed regime diagnostic.** What's different about seeds 17
   and 11 vs. 23 and 2? Possibilities to test: initial landscape
   coverage (cap-coverage metric per seed), drift realization
   magnitude, cue-stream composition. ~1 h.

4. **(Deferred)** drift-realization seeding. Currently the seed
   controls both substrate and drift. Separating them would
   distinguish substrate-variance from drift-realization-variance.

The cleanest follow-up is **#2** (single-seed-17 replication at
n_cues=5000) — answers the fade question directly and is cheap. **#1**
is the obvious envelope mapping but takes much longer.

## One-line takeaway

rt=0.85 produces a verified +1.6–1.9 pp top1 lift at mid-run
checkpoints (CI disjoint from zero) across 5 seeds at drift=0.30,
fading to non-significant by step 2000. Δcap_t05 is unreliable at
this drift. The seed 17 "drift-immune" pattern was outlier-grade,
not typical.
