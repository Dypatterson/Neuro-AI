# Phase 5: HAM Aggregation and Layer-2 Discovery

## Purpose

Phase 5 tests whether multi-scale retrieval should move from summed decoded
scores to a coupled hierarchical associative memory (HAM) dynamic.

The specific question is not "can another aggregator beat bigrams?" Phase 4
already showed that multi-scale retrieval beats the bigram baseline. The Phase
5 question is sharper:

> Does bidirectional cross-scale settling improve the validated summed-score
> aggregator enough to become the default retrieval path?

## Setup

Experiment 20 compares three multi-scale retrieval conditions:

- `summed_scores`: the Phase 4 validated baseline, summing decoded token scores
  across scales.
- `ham_geometric`: coupled HAM retrieval with geometric-mean consensus.
- `ham_arithmetic`: coupled HAM retrieval with arithmetic-mean consensus.

Common setup:

- Substrate: TorchFHRR, `D=4096`
- Codebook: Phase 3c reconstruction codebook
- Scales: `W={2,3,4}` with landscapes `{4096,2048,1024}`
- Corpus: WikiText-2
- Eval windows: `W=4` and `W=8`, center mask
- Seeds: `11,17,23`
- Betas: `10,30`
- HAM alpha: `0.3`
- HAM max iterations: `12`
- Decode K: `10`

Experiment 21 is a smaller smoke test for Phase 4 discoveries as layer-2 HAM
attractors. It compares an empty-layer-2 HAM baseline against HAM with replay
discoveries inserted as layer-2 consensus profiles.

## Headline Metric

The headline metric is held-out masked-token top-1 accuracy.

Top-k and cap-coverage are drill-downs: they explain whether the correct token
is present in the candidate set and whether the model assigns it high enough
confidence, but they are not alternate success definitions.

## Experiment 20 Results

Mean across all 12 seed/window/beta rows:

| Method | Top-1 | Top-K | Cap@0.5 |
|---|---:|---:|---:|
| Bigram | 0.055 | - | - |
| Summed scores | 0.121 | 0.333 | 0.265 |
| HAM geometric | 0.121 | 0.328 | 0.029 |
| HAM arithmetic | **0.129** | **0.334** | 0.040 |

HAM arithmetic improves mean top-1 by `+0.0074` absolute over summed scores.
HAM geometric is effectively tied on top-1 but slightly worse on top-k.

By evaluation window:

| Eval window | Summed top-1 | HAM geometric top-1 | HAM arithmetic top-1 |
|---|---:|---:|---:|
| W=4 | 0.111 | 0.102 | **0.114** |
| W=8 | 0.132 | 0.141 | **0.143** |

The HAM gain is clearer at `W=8`, where longer context creates more room for
cross-scale constraint.

By beta:

| Beta | Summed top-1 | HAM geometric top-1 | HAM arithmetic top-1 | Mean HAM iters |
|---:|---:|---:|---:|---:|
| 10 | 0.107 | 0.103 | **0.115** | ~11.8 |
| 30 | 0.135 | 0.139 | **0.142** | ~8.6 |

At `beta=10`, HAM often reaches the iteration cap and has zero `cap@0.5`.
At `beta=30`, HAM converges faster and improves top-1 more consistently.

## Drill-Down Findings

1. **HAM arithmetic is the best Phase 5 aggregator so far.** It wins the mean
   top-1 comparison overall, by evaluation window, and by beta.

2. **The gain is real but modest.** Mean top-1 improves from `0.121` to `0.129`.
   This is useful, but not yet a decisive replacement for summed scores unless
   future validation confirms the effect across more seeds and larger samples.

3. **Geometric consensus is not the default winner.** It underperforms at
   `W=4, beta=10`, likely because geometric means punish scale disagreement too
   harshly when one scale is uncertain.

4. **HAM confidence is not calibrated like summed score confidence.** HAM
   `cap@0.5` is much lower than summed scores even when top-1 is better. This
   is expected because HAM returns a normalized consensus distribution, while
   summed scores aggregate raw decoded similarities. Use top-1/top-k for direct
   comparison until cap thresholds are recalibrated for consensus space.

5. **Beta 30 remains the stronger operating regime.** It improves top-1 and
   settles faster. Beta 10 may still be useful diagnostically because it exposes
   feature-mode dynamics, but it is not the strongest retrieval setting here.

## Layer-2 Discovery Smoke Results

Experiment 21 tests whether Phase 4-style replay discoveries help when added as
layer-2 consensus attractors.

At `beta=10`, after 400 streamed cues:

| Condition | Top-1 | Top-K | Cap@0.5 | Layer-2 size |
|---|---:|---:|---:|---:|
| HAM baseline | 0.151 | 0.362 | 0.000 | 0 |
| HAM + layer-2 | 0.099 | 0.355 | 0.000 | 11 |

At `beta=30`, after 400 streamed cues:

| Condition | Top-1 | Top-K | Cap@0.5 | Layer-2 size |
|---|---:|---:|---:|---:|
| HAM baseline | 0.171 | 0.342 | 0.125 | 0 |
| HAM + layer-2 | 0.171 | **0.375** | 0.118 | 97 |

The layer-2 channel is not validated yet. It can populate attractors and, at
`beta=30`, improves top-k in this smoke test without hurting top-1. But at
`beta=10` it hurts top-1, which means the discovery criterion is too permissive
or the layer-2 coupling is too strong for that regime.

## Architectural Interpretation

HAM arithmetic is the current best candidate for Phase 5's default aggregation
dynamic. It keeps the anti-homunculus constraint intact: cross-scale influence
comes from consensus geometry rather than a supervisor selecting which scale
wins.

Layer-2 attractors are promising but not ready to count as a result. The current
discovery pathway produces attractors, but the attractors are not yet proven to
encode reusable abstraction rather than reinforcing frequent or noisy consensus
profiles.

## Decision

Do not replace summed aggregation globally yet.

Use `HAM arithmetic, beta=30` as the leading Phase 5 candidate, and treat summed
scores as the control baseline until HAM arithmetic survives a Phase 4-style
multi-seed validation.

Layer-2 discovery should remain experimental. The next test should compare:

- HAM arithmetic baseline
- HAM arithmetic with random layer-2 attractors
- HAM arithmetic with replay-discovered layer-2 attractors

on the same seeds, same held-out windows, and same layer-2 capacity.

## Next Experiment

Run a focused layer-2 validation with:

- `eval_window_size=8`
- `beta=30`
- seeds `11,17,23,31,42`
- fixed held-out test set per seed
- matched layer-2 counts for random vs discovered controls
- headline: top-1 delta vs HAM arithmetic baseline
- drill-downs: top-k, consensus entropy, layer-2 activation concentration,
  attractor reuse rate, and cap coverage with thresholds calibrated to
  consensus probabilities rather than raw summed scores.

Success criterion:

> Replay-discovered layer-2 attractors improve top-1 over HAM arithmetic and
> over random layer-2 attractors on the same evaluation windows.

If replay-discovered and random layer-2 attractors behave similarly, the layer-2
channel is acting as a generic smoothing/bias mechanism rather than a genuine
discovery pathway.
