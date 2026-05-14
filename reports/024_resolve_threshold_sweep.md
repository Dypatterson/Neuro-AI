# Report 024 — resolve_threshold sweep at drift=0.30: rt=0.85 produces drift-immune retrieval

**Date:** 2026-05-13
**Experiment:** `experiments/18_phase4_unified_experiment.py` swept across
`resolve_threshold ∈ {0.50, 0.60, 0.70, 0.80, 0.85}` at drift=0.30,
single seed (17), other knobs matched to report 023's drift=0.30 run
(death_window=1000, reencode_every=100, β=10, n_cues=2000).

**Raw data:** `reports/phase4_rt0p{50,60,80,85}/phase4_unified_results.json`,
prior `reports/phase4_drift_0p30/` for rt=0.70.

**Follows up:** [reports/023_drift_sweep_and_gate_audit.md](023_drift_sweep_and_gate_audit.md).

## Headline

**rt=0.85 makes Phase 4 retrieval immune to drift=0.30.** Phase 4
top1 stays constant at the pre-drift baseline value (0.122) across
all 2000 cues while the no-replay baseline degrades from 0.122 to
0.088. Δtop1 climbs monotonically (+0.031, +0.031, +0.034) — the
only setting where the discovered-pattern channel fully insulates
retrieval from substrate degradation.

Below rt=0.70 the sweep is *exactly* saturated: rt=0.50, 0.60, 0.70
produce byte-identical headline numbers, implying lowering rt
below 0.70 is wasted permissiveness.

## Full sweep results

Single seed (17), drift=0.30, 4 checkpoints. Phase 4 top1 / baseline
top1 / Δtop1 / Δcap_t05 at each checkpoint:

### rt=0.50, 0.60, 0.70 (identical)

| step | baseline top1 | phase4 top1 | Δtop1   | Δcap_t05 |
| ---: | ------------: | ----------: | ------: | -------: |
| 0    | 0.122 | 0.122 | +0.000 | +0.000 |
| 500  | 0.122 | 0.122 | +0.000 | -0.007 |
| 1000 | 0.092 | 0.095 | +0.003 | +0.000 |
| 1500 | 0.092 | 0.122 | **+0.031** | +0.007 |
| 2000 | 0.088 | 0.095 | +0.007 | +0.010 |

### rt=0.80

| step | baseline top1 | phase4 top1 | Δtop1   | Δcap_t05 |
| ---: | ------------: | ----------: | ------: | -------: |
| 0    | 0.122 | 0.122 | +0.000 | +0.000 |
| 500  | 0.122 | 0.122 | +0.000 | -0.007 |
| 1000 | 0.092 | 0.122 | **+0.031** | +0.000 |
| 1500 | 0.092 | 0.116 | +0.024 | +0.007 |
| 2000 | 0.088 | 0.082 | -0.007 | **+0.024** |

### rt=0.85 (winner)

| step | baseline top1 | phase4 top1 | Δtop1     | Δcap_t05 |
| ---: | ------------: | ----------: | --------: | -------: |
| 0    | 0.122 | 0.122 | +0.000 | +0.000 |
| 500  | 0.122 | 0.122 | +0.000 | -0.007 |
| 1000 | 0.092 | 0.122 | **+0.031** | +0.000 |
| 1500 | 0.092 | 0.122 | **+0.031** | +0.017 |
| 2000 | 0.088 | 0.122 | **+0.034** | +0.010 |

The phase4_top1 column for rt=0.85 reads `0.122, 0.122, 0.122,
0.122, 0.122` — perfectly flat at the pre-drift baseline. **No other
setting holds top1 throughout the drift trajectory.**

## What the sweep settles

### The dedup ceiling is real and lives around rt=0.70

rt=0.50, 0.60, 0.70 produce byte-identical baseline / phase4 / Δ
columns at every checkpoint. This is the architectural saturation
predicted from the gate audit: the candidate-handler in the W=2 unit
emits ~30 distinct attractors regardless of how liberally re-settled
traces are accepted. Lowering rt below 0.70 just routes more resampled
traces into the same set of outcomes — no new patterns, no new
information. The cands counter (30–31 across all five settings) is
consistent with this read.

**Practical implication:** Don't sweep rt below 0.70. Anything
permissive is in the saturation regime.

### Higher rt = sharper, more drift-resistant patterns

The interesting band is above rt=0.70:

- **rt=0.70** (and below): Δtop1 spikes to +0.031 at step 1500 then
  *drops* to +0.007 at step 2000. The discovered patterns help at
  one checkpoint but get washed out as drift accumulates.
- **rt=0.80**: Δtop1 +0.031 *earlier* (step 1000), then degrades
  (+0.024, -0.007). Best Δcap_t05 (+0.024 at step 2000) — confidence
  preserved even when argmax slips. Stricter threshold front-loads
  the lift.
- **rt=0.85**: Δtop1 *climbs monotonically* (+0.031, +0.031, +0.034).
  Phase 4 top1 stays at the pre-drift value throughout — drift
  immune.

The pattern is: stricter resolve_threshold → fewer but cleaner
patterns → patterns that survive subsequent drift cycles rather than
getting reprocessed into noise. At rt=0.85 the candidates are
high-resolution lock-ins that the substrate retrieves reliably even
as the codebook drifts further around them.

### Cands count is a misleading drill-down

Across all five rt settings, the run displays `cands=30` (give or
take 1). The headline differs by ~3 pp in top1 between the rt=0.70
band and rt=0.85, but the cands counter doesn't move. **The number
of discovered patterns isn't the load-bearing variable; their
quality is.** This is a methodological note: the existing drill-down
under-reports what changes between settings. A useful next-iteration
diagnostic would track *which* patterns get added (their resolution
at admission, their lifetime in the landscape, their retrieval-hit
rate after admission).

## What this does NOT settle

- **Single seed.** Δtop1=+0.034 at step 2000 with rt=0.85 could be
  seed luck. The pattern (monotonic climb, flat phase4 top1) is
  qualitatively distinct from any other setting, which makes it
  unlikely to be pure noise — but a 5-seed replication is the
  correct confirmation.
- **drift=0.30 only.** The sweep tests one drift magnitude. rt=0.85
  may fail at drift=0.50 (cleaner patterns may have less margin
  against severe drift) or be unnecessary at drift=0.15 (no gap to
  close).
- **Phase 4 top1 at the pre-drift value isn't the same as retrieval
  is "correct."** Phase 4 maintaining top1=0.122 means it gets the
  same fraction of test cues right as the pristine substrate did —
  but it may be getting different cues right via different
  mechanisms. A cue-by-cue overlap analysis between pre-drift and
  Phase 4 post-drift retrievals would clarify what "drift immune"
  means at the retrieval level.

## Anti-homunculus / FEP audit

`resolve_threshold` is a constant cut on a post-settling resolution
quantity. The audit (report 023) confirmed resolution is a passive
measurement of free-energy basin lock. Raising the threshold doesn't
introduce a controller — it raises the architectural prior on what
counts as "consolidated enough to emit a candidate." Passes the FEP
audit cleanly.

## Recommended next steps

1. **5-seed confirmation of rt=0.85 at drift=0.30.** Seeds {11, 17,
   23, 1, 2}. Headline metric: Δtop1 across the trajectory with 95%
   bootstrap CIs. If the monotonic-climb pattern survives the
   replication, **this is the first verified positive Phase 4
   result.** ~90 min.

2. **rt=0.85 at drift ∈ {0.15, 0.50}.** Tests robustness of the
   setting across drift regimes. Specifically: does rt=0.85 still
   help at drift=0.15 (the no-gap regime from report 023)? Does it
   help at drift=0.50 where rt=0.70 only preserved cap_t05? ~30 min.

3. **Pattern-quality drill-down.** Instrument the candidate handler
   to log per-pattern: admission resolution, retrieval-hit count
   after admission, lifetime (cycles before death). Re-run rt=0.50
   and rt=0.85 side-by-side and characterize the difference in
   pattern-set composition. ~45 min instrumentation + ~20 min
   analysis.

In order of leverage: #1 is the obvious replication; #2 is cheap and
clarifies the operating envelope; #3 is the right "why does this
work?" follow-up after replication.

## One-line takeaway

For drift=0.30 at this codebook + this β, the resolve_threshold knob
has a saturation regime (rt ≤ 0.70, all identical) and a quality
regime (rt ∈ {0.80, 0.85}) where stricter selection of discovered
patterns produces qualitatively better drift resistance. rt=0.85 is
the first setting that makes Phase 4 fully drift-immune at this
magnitude on this seed.
