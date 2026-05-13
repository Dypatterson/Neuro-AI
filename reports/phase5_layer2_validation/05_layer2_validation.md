# Phase 5: Layer-2 Discovery Validation

## Purpose

This run tests whether Phase 4 replay-discovered layer-2 attractors improve
HAM retrieval beyond both:

- an empty layer-2 HAM baseline, and
- a matched-count random layer-2 control.

The random control matters because layer-2 attractors alter the consensus
distribution even when they carry no learned structure. If discovered and random
attractors behave similarly, the layer-2 pathway is only a generic smoothing or
bias mechanism.

## Setup

- Experiment: `experiments/21_ham_with_phase4.py`
- Substrate: TorchFHRR, `D=4096`
- Codebook: Phase 3c reconstruction codebook
- Corpus: WikiText-2
- Eval window: `W=8`, center mask
- Beta: `30`
- HAM consensus: arithmetic mean
- Seeds: `11,17,23,31,42`
- Held-out samples per seed: `400`
- Cue stream per seed: `1000`
- Replay settings: `store_threshold=0.05`, `resolve_threshold=0.2`,
  `replay_every=25`, `replay_batch_size=10`
- Conditions: `ham_baseline`, `ham_with_layer2`, `ham_random_layer2`

## Headline Metric

The headline metric is held-out masked-token top-1 accuracy at the final
checkpoint.

Top-k, cap@0.5, layer-2 size, trace counts, and candidate counts are drill-down
diagnostics.

## Final Aggregate

At `1000` streamed cues:

| Condition | Top-1 | Top-K | Cap@0.5 | Layer-2 size |
|---|---:|---:|---:|---:|
| HAM baseline | 0.123 ± 0.016 | 0.319 ± 0.017 | 0.063 ± 0.006 | 0.0 |
| HAM + discovered layer-2 | **0.132 ± 0.010** | **0.339 ± 0.028** | 0.063 ± 0.007 | 109.4 |
| HAM + random layer-2 | 0.121 ± 0.018 | 0.318 ± 0.033 | 0.018 ± 0.008 | 111.0 |

Mean top-1 deltas:

| Comparison | Delta |
|---|---:|
| Discovered layer-2 vs baseline | **+0.0094** |
| Discovered layer-2 vs random layer-2 | **+0.0108** |
| Random layer-2 vs baseline | -0.0013 |

## Per-Seed Results

| Seed | Baseline top-1 | Discovered top-1 | Random top-1 | Disc - base | Disc - random |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.148 | 0.148 | 0.148 | +0.000 | +0.000 |
| 17 | 0.133 | 0.136 | 0.133 | +0.003 | +0.003 |
| 23 | 0.106 | 0.119 | 0.103 | +0.013 | +0.017 |
| 31 | 0.122 | 0.128 | 0.122 | +0.007 | +0.007 |
| 42 | 0.105 | 0.129 | 0.102 | +0.024 | +0.027 |

Discovered layer-2 beats baseline on 4 of 5 seeds and beats random layer-2 on
4 of 5 seeds. The one non-improving seed is a tie, not a regression.

## Checkpoint Trend

Mean discovered-layer-2 top-1:

| Cues seen | Top-1 |
|---:|---:|
| 250 | 0.131 |
| 500 | 0.132 |
| 750 | 0.133 |
| 1000 | 0.132 |

Most of the gain appears early, after the first batch of replay-discovered
attractors. Additional streamed cues keep the lift but do not compound it in
this run.

## Findings

1. **Replay-discovered layer-2 attractors are doing more than generic bias.**
   The discovered condition beats the matched random condition by about one
   absolute percentage point top-1, while random layer-2 is slightly worse than
   baseline.

2. **Random layer-2 harms confidence.** Random layer-2 drops cap@0.5 from
   `0.063` to `0.018`, even though top-1 is nearly flat. Arbitrary consensus
   attractors can flatten or miscalibrate the probability distribution.

3. **Discovered layer-2 improves top-1 and top-k without improving cap@0.5.**
   The top candidate is more often correct, and the target appears in the top-k
   list more often, but consensus confidence does not become sharper under the
   current cap threshold.

4. **The discovery gate is active but broad.** The discovered condition adds
   roughly 100 candidates per checkpoint and stabilizes around 108-112 live
   attractors. This is enough to improve top-1, but it is probably still too
   permissive for an abstraction mechanism.

## Decision

Layer-2 discovery is validated as a real signal, but not yet as a mature
abstraction mechanism.

The next architecture step should keep layer-2 in the system and tighten the
candidate admission rule. The current result clears the important control:
discovered attractors beat matched random attractors.

## Next Step

Refine the layer-2 discovery criterion before scaling it further.

Candidate drill-downs to add:

- activation concentration of each layer-2 attractor
- reuse count across distinct cue neighborhoods
- consensus entropy before vs after layer-2 influence
- discovered-attractor age and survival curves
- top-1 delta stratified by whether a cue strongly activates a layer-2 attractor

Candidate admission rule:

> Add a layer-2 attractor only when replay convergence recurs across multiple
> related traces, not after a single resolved replay.

That keeps the mechanism local: recurrence and activation concentration are
properties of the settling/replay dynamics, not a supervisor choosing which
discovery matters.
