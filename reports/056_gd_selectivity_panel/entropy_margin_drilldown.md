# Report 056 drill-down — entropy / margin: a miss is sharp-WRONG, not flat; and the Bayes↔memorization boundary

> **Drill-down** (not a graduation). Disambiguates `tix=0` (a recall miss) from a
> sharp-wrong-basin per the spec's headline requirement
> ([phase-3-consolidation-write-design.md:53](../../notes/emergent-codebook/phase-3-consolidation-write-design.md)
> "ship entropy + margin drill-downs so tix=0 is disambiguated from a sharp-wrong-basin").
> Data: the per-arm `mean_entropy` / `mean_margin` (+ rule-vs-noise split) already
> recorded in `headline_D2048.json` (synthetic) and `corpus_repo_D1024.json` (repo).
> `entropy` is normalized to [0,1] (1 = uniform over the codebook); `margin` = top1−top2
> settled score.

## Finding 1 — a miss is a confident **sharp-WRONG** basin, never a flat one

At β=10 the Modern Hopfield cleanup settles essentially one-hot, so **every real arm's read
is sharp** regardless of whether it hits:

| | recall | entropy | margin |
|---|---:|---:|---:|
| synthetic, all real arms (obs 2/3) | 0.11–0.87 | **≈0.002** | **≈0.978** |
| repo_sample, all real arms (obs 1/2) | 0.009–0.793 | **≈0.041** | **≈0.93** |

Entropy is near zero and margin near maximal **even for the arms that mostly miss** (e.g.
store-as-is at repo obs=1 recalls 0.009 but still reads at entropy 0.041 / margin 0.937).
**→ `tix=0` is a sharp-WRONG basin: the cue confidently addressed the wrong atom**, not a
flat "couldn't decide." A miss therefore means the (cue → recoverable association) was not
stored for that cue, not that the readout was uncertain.

## Finding 2 — the random-codebook control is the **only** flat failure

The one arm that goes flat is the readout-leak control (the headline H-output read against an
**unrelated** random codebook):

| repo_sample | recall | entropy | margin |
|---|---:|---:|---:|
| random_codebook obs=1 | 0.001 | **0.276** | 0.706 |
| random_codebook obs=2 | 0.001 | **0.373** | 0.618 |

High entropy + low margin = **no aligned basin** (the H-output, trained to point at real
atoms, aligns with no random atom). This is the signature of a genuine non-readout: it
confirms the real arms' sharp basins are *aligned recall*, not an argmax artifact — i.e. the
readout is sound and the misses in Finding 1 are real geometry, not a flat readout.

## Finding 3 — rule-vs-noise split: the Bayes-optimal ↔ memorization boundary

The synthetic toy injects a noise fraction `eps=0.25` (the target is a random draw, not
topic-determined). Splitting write+L2's recall by rule-vs-noise scene:

| obs | rule_rate | noise_rate | reading |
|----:|----------:|-----------:|:--------|
| 2 | 0.857 | **0.221** (≈ chance 0.125) | **Bayes-optimal**: recovers the recoverable rule structure, *correctly fails* the unrecoverable noise — cue-collision at the sparse cue caps H at the per-cue majority (the topic target) |
| 3 | 0.938 | **0.639** (≫ chance) | **memorization kicks in**: the richer, more distinctive cue lets H memorize *individual* noise scenes' (cue → random-target) pairs |

This is the cue-collision boundary made visible: at sparse cues the write behaves as a
**Bayes-optimal aggregator** (it cannot, and does not, recover targets that are not a function
of the cue), and the eps=0.25 noise floor on `noise_rate` ≈ chance is the toy analogue of the
information ceiling. At rich cues the cue is distinctive enough that the associative **memory**
stores each pair individually — consistent with the memory-not-learner verdict (Report 056):
the mechanism is a memory, and "Bayes-optimal recall" at sparse cues is the cue-collision limit,
not generalization.

## Why store-as-is < write+L2, in entropy/margin terms

Both read sharp (entropy ≈0.002), so store-as-is does not fail by going flat — it fails by
**confidently settling on the wrong atom** more often. On the rule scenes where the signal
exists, write+L2 recovers more (synthetic obs=2 rule_rate 0.857 vs store 0.799; repo obs=2
0.793 vs store 0.019): the write's marginal advantage is **more correct sharp basins on the
recoverable scenes**, not a sharpness difference. The scene-MHN blend-corruption (store-as-is)
produces a *sharp wrong* unbind, not a flat one — which is why entropy alone cannot rank the
arms and the `top_index_hits` basin-membership readout (not a confidence threshold) is the
correct metric.

## Bottom line

`tix=0` is **sharp-wrong**, not flat (Finding 1); the readout is sound (Finding 2, the only
flat arm is the unrelated-codebook control); and the rule-vs-noise split exposes the
**Bayes-optimal-at-sparse-cues → memorization-at-rich-cues** boundary (Finding 3). The
drill-down confirms the G-D headline is real aligned recall and explains *how* the arms differ
(more-correct sharp basins, not sharper ones).
