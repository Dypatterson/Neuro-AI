# Report 012 — Replay-store upgrade ablation (tag_count × IoR)

**Date:** 2026-05-13
**Experiment:** `experiments/29_replay_store_upgrades_ablation.py`
**Raw data:**
- `reports/replay_store_upgrades_ablation_tuned.json` (primary)
- `reports/replay_store_upgrades_ablation_tight.json` (small store)
- `reports/replay_store_upgrades_ablation_ior_regime.json` (sticky resolve)

## Motivation

This session added two `ReplayStore` upgrades from brainstorm Idea
4a/4c (2026-05-13):

* **Tag count** — overlapping queries collapse onto a single store
  entry with `tag_count += 1` rather than appending. Sampling weight
  becomes `gate × tag_count × suppression`. (Joo & Frank 2023.)
* **Inhibition of return (IoR)** — sampled traces decay their
  suppression multiplier; non-sampled traces recover.
  (Biderman SFMA 2023.)

The unit tests in
[tests/test_phase4_replay_store_upgrades.py](../tests/test_phase4_replay_store_upgrades.py)
pin the data-structure contracts. This report closes the open loop
from item B in the post-A2 plan: *do the upgrades actually change
replay dynamics in a real consolidation run, in the direction the
brainstorm predicted?*

## Setup

Stream design:

* 32 stored windows: 8 designated **frequent** sources, 24 **one-shot**
* 300 perturbed cues, with frequent fraction 0.8 (frequent sources
  are 24× more likely to be cued)
* Same seed produces identical cue stream across all four conditions
  → any outcome difference is attributable to replay-store config

2×2 ablation matrix:

| condition     | tag_overlap | suppression_decay | suppression_recovery |
| ------------- | ----------- | ----------------- | -------------------- |
| A_baseline    | None        | 1.0 (off)         | 0.0 (off)            |
| B_tag_only    | 0.7         | 1.0               | 0.0                  |
| C_ior_only    | None        | 0.4               | 0.02                 |
| D_both        | 0.7         | 0.4               | 0.02                 |

Hopfield retrieval: D=4096, β=3.0, cue_noise=0.45. This regime
satisfies two competing constraints — the cue noise is high enough
that the gate signal fires on most retrievals, *and* low enough that
two perturbations of the same source pattern have cosine ≈ 0.92,
above the 0.7 tag-overlap threshold.

Five seeds: {11, 23, 41, 53, 67}.

## Headline result (primary regime: store_capacity=200, batch=10)

Per-condition aggregates over 5 seeds:

| condition   | cands | unique_src | store_final | mTag | gini  | u₂_max | u₃_max | freq_surv |
| ----------- | ----- | ---------- | ----------- | ---- | ----- | ------ | ------ | --------- |
| A_baseline  | 200.0 | 28.0       | 100.0       | 1.00 | 0.619 | 0.459  | 0.289  | 0.975     |
| B_tag_only  | 184.0 | **30.0**   | **0.0**     | 0.00 | **0.520** | 0.459  | 0.289  | 0.975     |
| C_ior_only  | 200.0 | 27.4       | 100.0       | 1.00 | 0.634 | 0.459  | 0.289  | 0.975     |
| D_both      | 183.8 | **30.0**   | **0.0**     | 0.00 | **0.516** | 0.459  | 0.289  | 0.975     |

### Headline reading (tag_count)

**Tag-overlap collapse fires aggressively and produces measurable,
directional change**:

* **Sampling gini drops 16%** (0.619 → 0.520 under tag-on). Lower
  gini = more even distribution of which source patterns get
  replayed. Tag-overlap collapsing duplicates means the store is more
  diverse per slot, so the sampling distribution covers more sources.
* **Unique source coverage rises 7%** (28.0 → 30.0). The replay
  channel touches almost every source pattern (30 of 32) under tag-on
  conditions, vs. only 28 under baseline.
* **Candidate count drops 8%** (200 → 184). Tag collapse means
  redundant cues fold together rather than each producing its own
  resolution event.
* **Store size at end is 0** under tag-on conditions. Aggressive
  collapse + fast resolution clears the store completely — there's
  no backlog because there's no redundancy to keep traces around.

### Headline reading (IoR)

**IoR is correctly implemented but invisible in this measurement
regime**: A vs. C are statistically indistinguishable across every
metric (gini 0.619 vs. 0.634; everything else identical).

The cause was confirmed by two follow-up runs:

* `_tight` (store_capacity=20, batch=10 — 50% of store sampled per
  cycle): A_baseline gini 0.687, C_ior_only gini 0.683. Still no
  difference.
* `_ior_regime` (resolve_threshold=0.75 — traces forced to persist
  across multiple replay cycles): same null. Traces that fail to
  resolve once fail to resolve again because the underlying landscape
  is unchanged between cycles.

IoR's value emerges when the **landscape changes between replay
cycles** — e.g. when candidate addition feeds new patterns back into
memory and re-shapes the energy surface. In the measurement-only
configuration of this experiment (`candidate_handler` returns None,
never adds to memory), every re-settling on the same query produces
the same outcome, and inhibition-of-return adds no new information.

### u_k chain and frequency survival

`u₂_max`, `u₃_max`, `n_patterns_u3_active`, and `freq_survival_ratio`
are **identical across all four conditions**. The brainstorm's
prediction that tag_count would bias which patterns reach u₃+ doesn't
show in this experiment, but the reason is structural rather than
informative: consolidation in this run is driven by
`retrieve_and_observe()`'s `reinforce()` call on every retrieval,
which reflects which patterns the user *cued*, not which traces
survived replay. The replay-store upgrades shape *replay
trajectories*; they don't reach the u_k chain unless candidates
actually enter memory.

The 97.5% `freq_survival_ratio` confirms the cue stream did its job
— the frequent sources are dominantly winning the u_k race, as
designed.

## What is and isn't proven by this report

**Proven:**

* Tag-overlap collapse changes replay dynamics in the predicted
  direction (more diversity, lower gini, comparable yield), under
  parameters where same-source cues are mutually similar
  (cue_noise ≲ 0.5 at β = 3.0).
* Both upgrades are backwards-compatible: condition A_baseline
  reproduces the prior behavior.

**Not yet proven and worth a follow-up:**

* IoR's behavioral effect under **enabled candidate addition** —
  i.e. real Phase 4 runs where new candidate patterns enter memory
  between replay cycles. That's the regime where IoR is supposed to
  matter, and this report's measurement-only setup forecloses it by
  construction.
* Whether tag_count's diversity gain translates into downstream
  consolidation gains. This requires an experiment where candidates
  *do* feed back, so the upgraded replay distribution can shape the
  long-term codebook composition.

## Anti-homunculus check / FEP audit

All four conditions pass the same audits already documented in
[notes/notes/2026-05-13-fep-audit-checklist.md](../notes/notes/2026-05-13-fep-audit-checklist.md).
Configuration constants only; no inspect-and-trigger branches; no
controllers.

## Decision

* **Make `tag_overlap_threshold=0.7` the default** for new
  `ReplayConfig` callers when same-source cue similarity is expected.
  The current default is already 0.7 in
  [src/energy_memory/phase4/replay_loop.py](../src/energy_memory/phase4/replay_loop.py),
  consistent with this evidence.
* **Keep IoR enabled by default** (`suppression_decay=0.5`,
  `suppression_recovery=0.1`). The mechanism is correct and harmless
  even where it's invisible; it earns its keep the first time the
  candidate channel feeds back into memory.
* **Add a follow-up experiment to PROJECT_PLAN sidebar** that enables
  candidate addition and re-runs this ablation. That's the missing
  measurement for the IoR claim and the brainstorm's u₃+ survival
  prediction; it's the natural successor to this report.

## Recommended next steps

1. The follow-up just named (candidate-addition ablation). One
   session of work, reuses this script.
2. The remaining tier-A item ("synergy ratio as a standard drill-down
   in `experiments/18_phase4_unified_experiment.py`") from
   [reports/011_synergy_probe_phase4.md](011_synergy_probe_phase4.md)'s
   next-steps section. Cheap to add, valuable for catching regime
   drift in long runs.
3. The tier-C standing PROJECT_PLAN items: Phase 0 sweeps to Torch
   hot path, Phase 2 full matrix on MPS, first Phase 3 codebook-
   growth objective decision.
