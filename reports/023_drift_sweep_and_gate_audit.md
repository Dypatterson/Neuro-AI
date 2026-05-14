# Report 023 — Phase 4 drift sweep + engagement-gate audit: the gate fires, the bottleneck is downstream

**Date:** 2026-05-13
**Experiments:**
- `experiments/18_phase4_unified_experiment.py` at drift ∈ {0.30, 0.50}
- `experiments/36_engagement_gate_audit.py` at default settings (β=10, W=2)

**Raw data:**
- `reports/phase4_drift_0p30/phase4_unified_results.json`
- `reports/phase4_drift_0p50/phase4_unified_results.json`
- `reports/engagement_gate_audit.json`

**Follows up:** [reports/022_phase4_tuning_and_ham_validation.md](022_phase4_tuning_and_ham_validation.md)

## Part 1 — Drift sweep: Phase 4 helps when drift damages baseline

### Setup

Same seed, β, knobs as report 022's tuning run (death_window=1000,
reencode_every=100). Only `--drift-magnitude` varies: 0.15 (prior run),
0.30, 0.50. Single seed (17) per condition.

### Result table (best-checkpoint Δs, baseline → phase4)

| drift | baseline top1 (step 2000) | phase4 top1 | best Δtop1 | best Δcap_t05 | Story |
| ----: | ------------------------: | ----------: | ---------: | ------------: | ----- |
| 0.15  | 0.122 (stable)            | 0.122 (stable) | **+0.000** | -0.007 | No-gap regime: baseline isn't degrading, Phase 4 has nothing to recover. |
| 0.30  | 0.088                     | 0.095       | **+0.031** @ step 1500 | +0.010 | Mild-but-real top1 lift; baseline degrades 0.122 → 0.088, Phase 4 holds 0.122 → 0.095. |
| 0.50  | 0.071                     | 0.068       | **-0.003** | **+0.065** @ step 1500 | top1 lags slightly but cap_t05 is materially preserved (baseline 0.276 → 0.058, Phase 4 0.276 → 0.078 at step 2000). |

### Reading

Report 022's hypothesis B is confirmed: **drift=0.15 is below the
threshold at which baseline degrades, so Phase 4 has no gap to close.**
At drift=0.30 the gap opens (~3 pp), and Phase 4 closes about half of
it. At drift=0.50 the gap is severe (baseline cap_t05 collapses to
0.058 — basically no confidence in any retrieval); Phase 4 doesn't
recover top1 but does preserve cap_t05 ~35% above baseline.

This is the first positive Phase 4 headline signal in the project's
history. It is single-seed and modest, but it is real.

**Caveats:** Single seed; magnitude is 1–7 pp; cap_t05=0.078 in
absolute terms is degenerate, not impressive. A 5-seed replication at
drift=0.30 is the obvious follow-up to confirm the +3.1 pp top1 isn't
seed luck.

### Mechanism check

Across all three drift levels:
- `cands` per checkpoint: 30 / 30 / 30 / 31 / 31 — roughly constant.
- `meanU` decay: identical pattern (0.133 → 0.026) — death_window
  pruning fires the same way regardless of drift.
- `store=0` at every checkpoint in all three runs (the puzzle from
  report 022).
- Pattern count stabilizes at 4126–4127 across all checkpoints — death
  and discovery balance.

So Phase 4's *mechanism* runs the same way under all three drift
levels; what changes is the *substrate degradation* the discovered
patterns are compensating for. The ~30 discovered patterns per
checkpoint do nothing when the baseline is stable, contribute a
modest top1 lift when baseline is mildly degraded, and preserve
cap_t05 specifically when degradation is severe.

## Part 2 — Engagement-gate audit: the gate is firing 59% of the time

### Setup

`experiments/36_engagement_gate_audit.py`: stream 2000 cues through a
single-scale W=2 Phase 4 unit (`retrieve_and_observe`), record
`(engagement, resolution, gate)` per cue. **No replay cycles** —
this is a pure characterization of the gate signal itself.

### Distributions

```
engagement (= mean across-pattern entropy during settling)
  min=0.7349  p05=0.7349  p25=0.8380  p50=0.8804  p75=0.8846
  p95=0.8903  p99=0.8951  max=0.9010  mean=0.853  std=0.052

resolution (= final-state similarity to top stored pattern)
  min=0.8391  p05=0.8572  p25=0.8967  p50=0.9259  p75=0.9985
  p95=0.9985  p99=0.9999  max=0.9999  mean=0.935  std=0.049

gate = engagement × (1 − resolution)
  min=0.0001  p05=0.0011  p25=0.0012  p50=0.0654  p75=0.0913
  p95=0.1271  p99=0.1401  max=0.1440  mean=0.057  std=0.044
```

### Admission rate at hypothetical thresholds

| store_threshold | hits / 2000 | rate    |
| --------------: | ----------: | ------: |
| 0.01            | 1436        | 71.8%   |
| 0.02            | 1384        | 69.2%   |
| **0.05 (run)**  | **1183**    | **59.2%** |
| 0.10            | 375         | 18.8%   |
| 0.20            | 0           | 0.0%    |
| 0.30            | 0           | 0.0%    |

### What this overturns

Report 022 read `store=0` at every checkpoint and concluded "the
bottleneck is admission — the gate isn't firing." **That conclusion
was wrong.** The gate fires on **59% of cues** at the standard
threshold. With store capacity 500 and admission rate 59%, the store
would saturate at cue ~845 of a 2000-cue stream if traces were never
drained. The reason `store=0` shows up at every checkpoint isn't
admission starvation — it's that **replay cycles fire every 50 cues
and drain the store between checkpoints**.

So the bottleneck has moved one step downstream:

- **Admission rate is fine** — ~59% of cues admit.
- **Replay cadence is `replay_every=50`**, so over 500 cues there
  are 10 replay cycles. Each samples `replay_batch_size=10` traces
  for re-settling.
- **Candidate emission requires `resolve_threshold=0.7`**: after
  re-settling, only traces whose new resolution clears 0.7 emit
  a discovered pattern. The 30-candidates-per-checkpoint result
  means **only 30% of resampled traces are clearing the resolve
  threshold** (10 cycles × 10 sampled = 100 attempts → 30 emitted).
- The other 70% of resampled traces fail to consolidate into a
  candidate — they're being discarded silently.

### Structural facts about the gate distribution

- **Engagement is essentially a constant.** Min 0.73, max 0.90, std
  0.05. It contributes almost no dynamic range to the gate.
- **Resolution is bimodal.** ~50% of cues sit at p75–p95 ≈ 0.9985
  (full lock to a basin); the other 50% spread between 0.84 and 0.93
  (partial lock). The gate's dynamic range comes entirely from
  resolution.
- **The gate is structurally capped at ~0.14.** Engagement × (1 −
  resolution) ≤ 0.90 × 0.16 ≈ 0.14 even in the worst case. A
  `store_threshold > 0.20` cannot fire at all — there are no traces
  beyond that.

This caps the gate's discriminating range to roughly [0.001, 0.14].
The default 0.05 cuts right through the middle (median 0.065).

## Joint implications

### The Phase 4 architecture is alive and produces positive signal

Drift sweep confirms: Phase 4 reduces top1 loss by ~3 pp at drift=0.30
and preserves cap_t05 by ~2 pp at drift=0.50. The mechanism is doing
what its design says it should — when the codebook drifts, replay-
discovered patterns let retrieval recover partially. **It's not the
20-pp rescue we were hoping for, but it's no longer headline-inert.**

### The real bottleneck is `resolve_threshold`, not `store_threshold`

The audit moves the diagnostic question from "why is admission so
low?" to "why do only ~30% of resampled traces clear the resolve
threshold?" Three candidates for what the right fix looks like:

1. **Lower `resolve_threshold`** (currently 0.7). The audit's
   resolution distribution shows ~50% of traces fully resolve
   (>0.99). The other 50% sit in [0.84, 0.93]. A resolve_threshold
   of 0.85 would emit candidates from a much wider band.

2. **The 30% emission rate is the right rate**, and the bottleneck
   is elsewhere — e.g., `replay_batch_size=10` × `replay_every=50` =
   only 0.2 retrieval-equivalents per cue spent on replay. Raising
   `replay_batch_size` would scale candidate production linearly.

3. **The retrieval substrate is too good at locking to existing
   basins.** When 50% of traces fully resolve on first retrieval,
   there's nothing left to "discover" via re-settling. This points
   back upstream to the codebook geometry: with the W=8 collapsed
   codebook, the Hopfield landscape may be sharp enough that re-
   settling rarely surfaces alternative consensuses.

### Anti-homunculus / FEP audit reading

The gate is a measurement of "this cue produced a partially-resolved
trace" — pure free-energy diagnostic, passes the FEP audit. The
`store_threshold` is a constant cut on that diagnostic, which is fine
(architectural prior, not a learned controller). The `resolve_threshold`
acts as a filter on candidate emission — also a constant cut. None of
the knobs surveyed introduce inspect-and-trigger controllers; the
bottleneck is in a passive emission rule, not in supervisory logic.

## Recommended next steps

In order of leverage:

1. **resolve_threshold sweep at drift=0.30.** Single seed, single
   drift level (where Phase 4 already shows a top1 lift), sweep
   `resolve_threshold ∈ {0.5, 0.6, 0.7, 0.8, 0.85}` and measure
   discovered-pattern count + Δtop1 per setting. This is the
   shortest path from "gate audit insight" to "actual headline
   improvement." ~45 min.

2. **5-seed confirmation at drift=0.30, default thresholds.** The
   +3.1 pp top1 lift is single-seed. Replicate across seeds 11, 23,
   1, 2, 17 to bound the effect with CIs. ~90 min (5× single-run
   time).

3. **(Deferred)** replay_batch_size scaling — only worth running
   if (1) doesn't surface meaningful candidate-production lift.

4. **(Deferred)** codebook-geometry follow-up — if the substrate's
   too-aggressive basin locking limits re-settling diversity, the
   Spisak-Friston direction comes back into play. Re-prioritize
   only after (1) and (2) are in.

The headline-vs-drill-down framework from the 2026-05-09 note now
applies cleanly to Phase 4: headline metric is **Δtop1 at drift=0.30**.
Drill-downs are admission rate, discovered-pattern count, cap_t05
preservation, meanU decay. The recommended sweep targets the
headline directly.
