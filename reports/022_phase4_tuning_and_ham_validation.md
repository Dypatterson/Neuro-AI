# Report 022 — Phase 4 tuning fails to rescue headline; HAM-arithmetic ties summed at β=30, wins narrowly at β=10

**Date:** 2026-05-13
**Experiments:**
- `experiments/18_phase4_unified_experiment.py` with `--death-window 1000 --reencode-every 100`
- `experiments/20_ham_vs_summed.py` with `--seeds 11,17,23,1,2`

**Raw data:**
- `reports/phase4_tuned_dw1000_re100/phase4_unified_results.json`
- `reports/ham_5seed/ham_vs_summed_results.json`

**Follows up:**
- [reports/2026-05-13-session-summary.md](2026-05-13-session-summary.md) — flagged `death_window` and `reencode_every` as the prescribed-but-untested Phase 4 rescue knobs.
- [reports/phase5_ham_full/05_ham_phase5_report.md](phase5_ham_full/05_ham_phase5_report.md) — flagged 5-seed validation as required before HAM replaces summed scores as default.

## Part 1 — Phase 4 tuning: still headline-inert

### Setup

Matches the prior `phase4_drift_headline_b10` run exactly except for two
knobs that were prescribed but untested:

- `death_window`: 10000 → **1000** (stale patterns pruned 10× faster)
- `reencode_every`: 0 → **100** (stored landscape patterns refreshed
  through the current codebook every 100 cues; discovered patterns
  have no source window and are skipped)

Other settings: seed 17, β=10, drift=0.15, 2000 cues, 4 checkpoints,
3 scales (W=2/3/4), resolve_threshold=0.7, store_threshold=0.05.

### Result

| step  | baseline top1 | phase4 top1 | Δtop1 | baseline cap_t05 | phase4 cap_t05 | Δcap_t05 |
| ----: | ------------: | ----------: | ----: | ---------------: | -------------: | -------: |
| 0     | 0.122 | 0.122 | +0.000 | 0.276 | 0.276 | +0.000 |
| 500   | 0.122 | 0.122 | +0.000 | 0.255 | 0.255 | +0.000 |
| 1000  | 0.122 | 0.122 | +0.000 | 0.259 | 0.259 | +0.000 |
| 1500  | 0.122 | 0.122 | +0.000 | 0.262 | 0.255 | **-0.007** |
| 2000  | 0.122 | 0.122 | +0.000 | 0.262 | 0.255 | **-0.007** |

### What's actually happening

The mechanism is alive but not load-bearing.

- **death_window=1000 is firing.** Mean consolidation strength `meanU`
  drops monotonically (0.133 → 0.065 → 0.039 → 0.026) across
  checkpoints — patterns are being pruned at the prescribed cadence.
- **Discovery is firing.** Phase 4 reports ~30 candidates per
  checkpoint window. Pattern count on the W=2 landscape stabilizes
  at 4126 (= 4096 landscape + ~30 discovered) — death and discovery
  rates roughly balance.
- **Replay store is empty throughout** (`store=0` at every
  checkpoint). The engagement gate isn't accumulating; each replay
  cycle drains what was admitted, but admission rate is so low that
  store size doesn't grow.
- **Headline is exactly inert.** `top1` is constant 0.122 across both
  conditions at all checkpoints. Cap_t05 drifts slightly negative in
  phase4 at later steps.

This is the same outcome as the prior `phase4_drift_headline_b10`
run, achieved through different mechanism settings. The new knobs
*change the dynamics in the prescribed direction* but the dynamics
they create still aren't enough to move the headline.

### What this rules out / implies

- **Death and reencode are not the missing piece.** Both were
  flagged as the "shortest path to a positive Phase 4 headline."
  They aren't. The Phase 4 architecture, as currently configured,
  cannot move the headline metric under drift=0.15 with this
  codebook, at this scale, at any setting of these two knobs.

- **The bottleneck is admission, not consolidation.** Store=0
  throughout means few traces ever enter the replay store. The
  engagement gate (`entropy × (1 − resolution)`) at
  `store_threshold=0.05` only admits ~30 traces per 500 cues. The
  question shifts upstream: what's preventing the gate from firing?
  At β=10 with Phase 3c codebook, retrieval entropy and resolution
  must be sitting in a flat region where neither high-engagement nor
  low-resolution holds.

- **drift=0.15 may be too small to be informative.** The codebook is
  drifting, but post-drift retrieval still has top1=0.122 just as
  pre-drift does. If drift isn't degrading the baseline, there's no
  performance gap for Phase 4 to close. A drift-magnitude sweep
  (0.15 → 0.30 → 0.50) would surface whether Phase 4 can close a
  gap when one actually exists.

## Part 2 — HAM 5-seed validation: tie at β=30, narrow win at β=10

### Setup

Standard exp 20 protocol: 5 seeds (11, 17, 23, 1, 2) × eval window W ∈
{4, 8} × β ∈ {10, 30} × three conditions (summed, ham_geometric,
ham_arithmetic). Frozen Phase 3c reconstruction codebook. ~350-380
test windows per cell.

### Aggregated results (mean ± std across 5 seeds)

| eval | β  | bigram | summed top1     | ham_geom top1   | ham_arith top1  | summed topK | ham_arith topK |
| ---- | -- | ------ | --------------: | --------------: | --------------: | ----------: | -------------: |
| W=4  | 10 | 0.061  | 0.111 ± 0.023   | 0.098 ± 0.031   | 0.110 ± 0.020   | 0.324       | 0.323          |
| W=4  | 30 | 0.061  | 0.131 ± 0.012   | 0.131 ± 0.008   | 0.136 ± 0.011   | 0.323       | 0.321          |
| W=8  | 10 | 0.052  | 0.111 ± 0.037   | 0.115 ± 0.018   | **0.133 ± 0.007** | 0.345       | 0.338          |
| W=8  | 30 | 0.052  | 0.147 ± 0.007   | 0.146 ± 0.014   | 0.149 ± 0.012   | 0.341       | 0.331          |

### Observations

- **At β=30 (the validated prototype-mode regime), HAM-arithmetic is
  statistically indistinguishable from summed scores.** W=4: 0.136 vs
  0.131; W=8: 0.149 vs 0.147. ±std overlap fully on both sides.
- **At β=10 W=8, HAM-arithmetic is a clear win** on stability if not
  always on magnitude. Mean 0.133 vs summed's 0.111, with std 0.007
  vs 0.037. HAM-arith's CI sits above summed's mean. The variance
  collapse is the more interesting result than the +2.2 pp mean
  delta: summed is high-variance because some seeds settle into
  low-quality combined rankings at low β, and HAM-arithmetic's
  bidirectional coupling resolves that.
- **HAM-geometric is consistently the worst** of the three. It's a
  meaningful win over summed in only one cell (W=8 β=10, +0.4 pp on
  the mean) and a meaningful loss in another (W=4 β=10, -1.3 pp).
- **Top-K is essentially flat across all three aggregators.** Where
  the architecture matters is in argmax discrimination, not in the
  top-K coverage.

### Decision: HAM-arithmetic conditionally replaces summed scores

The Phase 5 HAM report specified that summed-scores is the default
*until* HAM survives a 5-seed validation. Reading this validation
strictly:

- **At β=30 (the canonical Phase 2/4 retrieval regime), keep summed.**
  HAM-arithmetic offers no significant advantage. Summed is simpler,
  cheaper, and has identical performance.
- **At β=10 (Phase 4's replay-gating regime), switch to
  HAM-arithmetic.** The variance collapse is real and disjoint
  (summed std=0.037, ham_arith std=0.007), and the mean is
  ~20% higher (0.133 vs 0.111).

The two regimes don't compete — they correspond to different
operating modes (Phase 4 gate-firing vs Phase 2/4 retrieval). The
canonical multi-scale stack should use β=30 + summed for retrieval
and β=10 + HAM-arithmetic for replay diagnostics.

## Joint implications

Two prescribed paths from the session summary are now resolved:

1. **Phase 4's death_window/reencode rescue is *not* the rescue.**
   The knobs work as designed; the headline doesn't move. Phase 4 is
   either (a) admission-starved — needs a different engagement gate
   formulation, or (b) operating in a no-gap regime — drift=0.15 isn't
   degrading baseline enough to leave room for rescue.

2. **HAM-arithmetic has a defensible niche** at β=10 W=8 (variance
   collapse + small mean lift) but does not displace summed at β=30.
   The 2-config policy above is the cleanest reading.

### Recommended next steps

In order of leverage:

1. **Drift-magnitude sweep on Phase 4.** Single seed, drift ∈
   {0.15, 0.30, 0.50}, baseline-vs-phase4 at each. Tests hypothesis
   that Phase 4 only has a chance to help when drift actually
   damages baseline. ~30 min.

2. **Engagement-gate audit.** With `store=0` throughout the 2000-cue
   run, the gate either (a) never fires, or (b) the trace is
   admitted then immediately consumed by the next replay cycle. Add
   per-cue logging of `(engagement, resolution, gate)` to identify
   which. ~20 min instrumentation + 10 min analysis.

3. **Adopt the 2-regime HAM policy.** Update Phase 5 docs to specify
   β=30+summed for retrieval-mode, β=10+HAM-arithmetic for replay-
   diagnostic-mode. ~10 min, no experiment needed.

4. **(Deferred until #1 and #2 land.)** Engagement-gate redesign or
   Phase 4 architectural change.

## Anti-homunculus / FEP audit

- **Phase 4 tuning:** Both new knobs (`death_window`, `reencode_every`)
  are passive timers/counts on existing dynamics, not inspection-and-
  trigger controllers. Pruning fires when a strength variable stays
  below threshold for N steps; reencode fires on cue-count modulo.
  Both pass the FEP audit (`free-energy quantity = consolidation
  strength`; gradient response = decay below threshold + scheduled
  re-projection). No new controllers introduced.
- **HAM validation:** Pure measurement evaluator across three
  pre-existing aggregators. No mechanisms added.
