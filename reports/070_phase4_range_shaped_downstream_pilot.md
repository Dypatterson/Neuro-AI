# Report 070 — Phase 4 Range-Shaped Downstream Pilot

**Date:** 2026-05-25
**Active phase:** 5′ precommit
**Status:** Synthetic downstream pilot completed. **No Phase 5 graduation
claim.** This report measures Phase 4 replay-cycle behavior, not Phase 5
Delta E.

## Boundary

Report 069 landed the reusable range-shaped sampler and static Phase 4 wiring.
This report checks whether that integration changes downstream Phase 4 replay
behavior when the real `UnifiedReplayMemory.run_replay_cycle()` path is used:

```text
replay store -> sampler -> optional rebind -> Hopfield re-settle ->
candidate insertion -> consolidation update
```

The run is still synthetic and diagnostic. It does not use the full Phase 3/4
corpus harness, does not measure held-out recall, and does not measure Phase 5
`Delta E`.

## Setup

Command:

```text
PYTHONPATH=src:. .venv/bin/python experiments/45_phase4_range_shaped_downstream_pilot.py \
  --seeds 17 11 23 1 2 3 5 7 13 29 \
  --dim 512 --n-traces 128 --n-roles 8 --n-atoms 128 \
  --window-size 8 --skew-concentration 4 \
  --replay-cycles 12 --replay-batch-size 16 \
  --resolve-threshold 0.2 \
  --out reports/phase5_prime_range_shaped_downstream_pilot/results_n10.json
```

Conditions are static config only:

| condition | sampler | fallback | rebind mode |
| --- | --- | --- | --- |
| `standard` | standard whole-trace replay | none | none |
| `range_skip` | range-shaped | skip | none |
| `range_closest` | range-shaped | closest | none |
| `range_rebind_single` | range-shaped | rebind | single-binding |
| `range_rebind_window` | range-shaped | rebind | window-preserving |

Smoothing alpha was fixed at `0.0`. No condition switches behavior based on
observed metrics.

## Aggregate Results

Means over 10 seeds:

| condition | candidates | candidate cells | candidate rect | atom entropy | winner unique | u3 max | final d_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 128.0 | 32.0 | 2.0794 | 0.997 | 3.2 | 0.1074 | 15.95 |
| `range_skip` | 25.1 | 32.0 | 2.0794 | 0.982 | 3.0 | 0.1070 | 27.95 |
| `range_closest` | 128.0 | 32.0 | 2.0794 | 0.997 | 3.2 | 0.1074 | 15.96 |
| `range_rebind_single` | 192.0 | 149.2 | 0.8204 | 0.987 | 3.1 | 0.1074 | 13.50 |
| `range_rebind_window` | 191.8 | 254.7 | 0.1036 | 0.993 | 3.1 | 0.1074 | 13.51 |

Raw JSON:

```text
reports/phase5_prime_range_shaped_downstream_pilot/results_n10.json
```

## Findings

**F-1 — Rebind is load-bearing.** `range_skip` and `range_closest` do not
expand provenance support: both remain at the original 32 skew cells with
rectangularity `2.0794`. Skip also under-samples because most range-shaped
pairs are unbacked. This matches Report 068's sampler-only warning.

**F-2 — Window-preserving rebind produces the intended downstream provenance
geometry.** `range_rebind_window` expands candidate encoder support from
`32.0` to `254.7` cells and drops candidate rectangularity from `2.0794` to
`0.1036`. This is the first test showing the reusable Phase 4 integration can
carry range-shaped support into replay-generated candidate traces.

**F-3 — Single-binding rebind helps but is weaker.** `range_rebind_single`
expands support to `149.2` cells, but rectangularity remains much higher
(`0.8204`). If the goal is to preserve factored event structure, the
window-preserving mode is the better default for downstream tests.

**F-4 — No consolidation-quality win is shown here.** `u3_max` is essentially
unchanged across conditions, and winner diversity stays around three basins.
The final `d_eff` drops for rebind conditions, likely because the toy Hopfield
landscape settles many newly synthesized queries into a few existing basins
before candidate insertion. That is a caution signal, not a falsification:
this pilot measures replay-cycle mechanics on a tiny synthetic landscape, not
full Phase 3/4 learning or held-out recall.

## Anti-Homunculus Check

Pass. The experiment compares predeclared static configurations. The sampler,
fallback, smoothing alpha, and rebind mode are fixed before each run. No
condition is selected or switched based on intermediate metrics.

## What Remains Unverified

- No real-corpus Phase 3/4 integrated comparison has been run with
  `range_shaped`.
- No held-out retrieval or codebook-quality metric improved in this report.
- No Phase 5 `Delta E` movement has been measured.
- No smoothing-alpha downstream comparison has been run.
- The full Phase 5′ bundle-first grid remains a diagnostic harness, not a
  graduation result.

## Bottom Line

The downstream pilot validates the integration mechanics: range-shaped replay
only changes Phase 4 candidate provenance when rebind is enabled, and
window-preserving rebind is the strongest mode. The next meaningful test is a
real Phase 3/4 integrated comparison with `range_shaped + rebind +
window_preserving` against `standard`, with held-out recall/codebook geometry
readouts before any Phase 5 `Delta E` run.
