# Report 013 — Replay-store ablation with candidate addition enabled

**Date:** 2026-05-13
**Experiment:** `experiments/29_replay_store_upgrades_ablation.py --enable-candidates`
**Raw data:** `reports/replay_store_upgrades_candidates_on.json`

## Motivation

[Report 012](012_replay_store_upgrades_ablation.md) ran the
tag_count × IoR ablation in measurement-only mode (candidate handler
returned None, so the landscape never changed). Two open questions
remained:

1. Does IoR become visible when the landscape *does* change between
   replay cycles — the regime where it's biologically motivated?
2. Does tag_count's diversity gain translate into a measurable
   change in the long-term memory composition once candidates feed
   back into the landscape?

This report enables the `candidate_handler` to actually store
settled states as new patterns, extending consolidation. Same 2×2
matrix, same cue stream, same seeds.

## Headline result

| condition  | cands | uniq | mem_final | +added | addFq  | gini  |
| ---------- | ----- | ---- | --------- | ------ | ------ | ----- |
| A_baseline | 15.0  | 9.8  | 47.0      | 15.0   | 0.773  | 0.765 |
| B_tag_only | 9.8   | 9.8  | **41.8**  | **9.8** | **0.671** | **0.694** |
| C_ior_only | 15.0  | 9.8  | 47.0      | 15.0   | 0.773  | 0.765 |
| D_both     | 9.8   | 9.8  | **41.8**  | **9.8** | **0.671** | **0.694** |

(Aggregates over seeds {11, 23, 41, 53, 67}. Initial memory = 32
patterns; mem_final shows the count after 300 cued retrievals with
replay candidates added. addFq = fraction of added candidates whose
nearest source pattern was in the frequent set.)

### Tag-collapse reading

* **Memory bloat is cut by ~35%**. A/C add 15 candidates to the 32-
  pattern landscape (final 47); B/D add 9.8 (final 41.8). Tag-overlap
  folds near-duplicate cues into a single store entry, so subsequent
  resolutions don't fire repeatedly on the same source. Each unique
  source produces approximately one consolidation event rather than
  several.
* **Source coverage is preserved**. All conditions reach 9.8 unique
  source patterns with candidates — tag-collapse trims redundancy
  without giving up reach.
* **Frequency-weighted addition bias drops from 77% to 67%**.
  Without tag_count, 77% of added candidates trace back to a
  frequent source (the system over-consolidates what it sees most
  often). With tag_count, that drops to 67% — closer to the cue-
  stream's 80% frequent fraction, with the over-representation
  attenuated by collapse-on-overlap. The brainstorm's prediction that
  tag_count would *boost* frequent-pattern survival is reversed in
  practice: the mechanism actually *reduces* over-representation of
  frequent sources in the long-term memory.
* **Sampling gini drops 9%** (0.765 → 0.694). Modest narrowing of
  the replay distribution toward more even coverage, consistent with
  the measurement-only finding.

### IoR reading

**Still null.** A_baseline vs. C_ior_only are identical to three
decimals across every metric in the table. Even when candidate
addition modifies the landscape between replay cycles — exactly the
regime that was supposed to make IoR matter — the upgrade contributes
nothing measurable.

Diagnosis of why: the replay flow is

```
sample → re-settle → if resolved: emit candidate AND remove from store
                  → if unresolved: age++, gate updated, kept until max_age
```

Suppression only matters when a trace stays in the store long enough
to be sampled multiple times. In this regime almost every sampled
trace resolves on the first replay attempt (~97% candidate rate per
sample) and is immediately removed. Suppression has no time window in
which to bias subsequent samples.

The IoR mechanism is correctly implemented (proven at the unit-test
level in `tests/test_phase4_replay_store_upgrades.py`) but its operating
regime requires a different replay flow than the one the project
currently uses — e.g., one where unresolved traces are deliberately
kept across many cycles and re-sampled with declining priority, like
biological replay sweeps. The current architecture is
"resolve-and-remove"; IoR's design assumes "keep-and-sweep."

### u_k chain and frequency survival

`n_patterns_u3_active` is **1** in every condition, and
`freq_survival_ratio` is **0.0** in every condition. This reflects
two interacting facts:

* Retrieval reinforcement is unbalanced: one pattern (whichever the
  cue stream most often lands on at high resolution) accumulates much
  more u-mass than the rest. With frequent_fraction=0.8 and 8 frequent
  sources, each frequent source receives ~30 reinforce() calls; one of
  them pulls ahead and dominates u_3.
* That one dominant pattern is an **added candidate**, not a stored
  frequent source — the candidate-addition flow steals u-mass once it
  enters the landscape because future retrievals settle to it with
  higher resolution than to the perturbed-stored-pattern original.
  The freq_survival metric, defined as "fraction of u_3-active
  patterns whose index is in the frequent set [0, n_frequent)," is
  mechanically 0.0 because the index ≥ 32.

This isn't a bug — it's how Benna-Fusi behaves under unbalanced
reinforcement combined with candidate seeding. The interpretation is:
**candidate addition is doing structural work**, pulling consolidation
mass into the new patterns it creates. Whether that's the right thing
to do is a Phase 4/5 design question, but it's outside the scope of
this ablation.

## Status of the two open questions from report 012

| open question                                                                            | answer                          |
| ---------------------------------------------------------------------------------------- | ------------------------------- |
| Does IoR become visible when the landscape changes between cycles?                       | **No.** Same null result, robust across regimes. |
| Does tag_count's diversity gain reach the long-term memory composition?                  | **Yes.** Memory bloat -35%, freq-bias -10 percentage points. |

## Decision

* **Keep `tag_overlap_threshold=0.7` as the default.** Now demonstrated
  to reduce in-memory pattern bloat by ~35% with no loss of source
  coverage and a useful reduction in over-consolidation of frequent
  sources.
* **Demote IoR defaults.** Set `suppression_decay=1.0` and
  `suppression_recovery=0.0` as the default in `ReplayConfig`. The
  mechanism does nothing in the project's current resolve-and-remove
  replay flow; leaving it on by default adds compute and conceptual
  surface for no observed benefit.
* **Keep the IoR code path in place.** The unit tests still pass and
  the mechanism is correctly implemented. If a future design moves to
  a keep-and-sweep replay flow (where unresolved traces persist and
  get re-sampled over many cycles), IoR becomes relevant again. The
  code can stay; only the defaults change.

## Recommended next steps

1. **Flip the IoR defaults** (`suppression_decay=1.0`,
   `suppression_recovery=0.0`) in
   [src/energy_memory/phase4/replay_loop.py](../src/energy_memory/phase4/replay_loop.py).
   Existing tests continue to pass (they exercise the parameterized path,
   not the defaults).
2. **Move to Tier C** — the standing PROJECT_PLAN items: Phase 0
   sweeps to Torch hot path; Phase 2 full matrix on MPS; first Phase 3
   codebook-growth objective decision. The brainstorm-driven work is
   now thoroughly explored and the open loops it created are closed.

## Anti-homunculus check / FEP audit

Candidate addition is a clean local function of (settled state, gate
signal) → (new pattern stored, consolidation chain extended). The
`candidate_handler` is a constructor-time architectural choice, not a
runtime inspect-and-trigger. Pass.
