# Report 033 — Phase 4 death-mechanism diagnostic

**Date:** 2026-05-15
**Active phase:** 4
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift between cycles.
**Required controls per [phase-4-unified-design.md:318-323](../notes/emergent-codebook/phase-4-unified-design.md):** N/A — this is a mechanism diagnostic, not a headline measurement.
**Last verified result:** [Report 032](032_phase34_n10_verification.md) — ΔR@10 = +0.009, CI [−0.003, +0.021].
**Why this experiment now:** STATUS.md blocker #2 (death mechanism never fires). The 5-seed `phase34_integrated_hebbian_st03_death_*` runs on disk (death_threshold raised 0.005 → 0.05, window shrunk 10000 → 10) produced `deaths_total=0` on all 5 seeds. Before raising thresholds further — which is the homunculus pattern of picking numbers to make a metric move — characterize the mechanism empirically: is `effective_strength` never below threshold, or is the death counter resetting before reaching the window?

---

## Method

Added per-checkpoint death telemetry to [experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py): for each Phase 4 unit (scales 2/3/4), log `mean/min/p10/p50/p90/max` of `abs(effective_strength)`, count of patterns currently below `death_threshold`, max of `below_threshold_steps`, count of `dead_ready` patterns. Telemetry is additive — does not change any committed experiment behavior.

Ran one seed (seed 1, the cheap one — only 12 Hebbian commits) locally on MPS at the canonical st=0.3 + death-fix config:

```
--success-threshold 0.3  --death-threshold 0.05  --death-window 10
n_cues=1500  replay_every=50  consolidation_m=6  novelty_strength=1.0  retrieval_gain=0.1
```

Telemetry-bearing JSON: [reports/phase34_death_diag_seed1/phase34_results.json](phase34_death_diag_seed1/phase34_results.json).

---

## Result

Condition C (Phase 3 + Phase 4) consolidation-state trajectory across the run, scale-2 unit (4097 patterns):

| step | mean | min | p50 | p90 | max | below_thresh | counter_max | dead_ready |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300  | 0.211 | 0.208 | 0.208 | 0.208 | 9.14  | 0 / 4097 | 0 | 0 |
| 600  | 0.112 | 0.108 | 0.108 | 0.108 | 12.80 | 0 / 4097 | 0 | 0 |
| 900  | 0.073 | 0.069 | 0.069 | 0.069 | 14.72 | 0 / 4097 | 0 | 0 |
| 1200 | 0.053 | 0.048 | 0.048 | 0.048 | 15.81 | **4094 / 4097** | **1** | 0 |
| 1500 | 0.039 | 0.034 | 0.034 | 0.034 | 16.97 | 4094 / 4097 | **7** | 0 |

Identical pattern at scales 3 and 4.

### Two coupled failure modes

**Failure mode 1 — temporal:** patterns decay below `death_threshold=0.05` only at ~step 1200. The death counter is incremented per `step_dynamics()` call, which fires once per replay cycle = every `replay_every=50` cues. `death_window=10` therefore demands 500 consecutive cues below threshold. First death would fire at ~step 1700; the run is 1500 cues. By step 1500 the maximum counter is 7 — below the window of 10.

**Failure mode 2 — population structure:** at every checkpoint, the `min`, `p10`, `p50`, and `p90` of `effective_strength` are byte-identical. That means 4094 of 4097 patterns are *symmetric* — they have the same strength to four decimals. The only mechanism that breaks symmetry is `reinforce()` on retrieval ([consolidation.py:139–149](../src/energy_memory/phase4/consolidation.py)), so this telemetry shows that **>99.9% of patterns are never retrieved as top-1 across 1500 cues**. The handful of differentiated patterns (max=16.97 at scale 2) are the ~3 patterns that win retrieval repeatedly.

### Implication

Naively closing the temporal gap (extending the run or shrinking `death_window`) does not produce architecturally-meaningful death. It produces *mass death of the initial vocab codebook*: at the point patterns start crossing below threshold, 4094 of 4097 cross simultaneously. The architecturally-intended death — stale **discovered** patterns dying so the discovery channel can replace them with current-codebook geometry — is not what would fire. The design's `pattern_death` ([replay_loop.py:391–405](../src/energy_memory/phase4/replay_loop.py)) treats initial vocab atoms and discovered patterns uniformly. With this telemetry, almost every initial vocab atom would die together for the same reason: no test cue retrieved it as top-1.

This is a Phase 3 / vocab-statistics problem, not a Phase 4 mechanism problem. The reinforcement signal (top-1 retrieval) is too narrow for a 2048-token vocab × 1500-cue run: average expected reinforcements per pattern = 1500/2048 ≈ 0.73. The bulk of vocab atoms cannot be reinforced even once.

---

## FEP / anti-homunculus audit

| Check | Status |
|---|---|
| `step_dynamics` = local geometric dynamic on u-chain (no supervisor) | ✓ |
| `reinforce` = local geometric dynamic on top-1 winner (no supervisor) | ✓ |
| `dead_indices` = a measurement, not an arbitration | ✓ |
| `garbage_collect` = a mechanical removal driven by the measurement | ✓ |
| The choice of `death_threshold` and `death_window` is the place where a designer's intuition enters | ⚠ — this is where the homunculus risk lives |

The mechanism itself is FEP-clean. The way it's parameterized determines whether what dies is *meaningful*.

---

## What this report does and does not establish

**Does establish:** death cannot fire in n_cues=1500 at the canonical config. The 5-seed `st03_death_*` runs on disk produce headline numbers indistinguishable from report 029 because death didn't fire and consolidation state evolved identically.

**Does not establish:** what the right death configuration *is*. Three architecturally-distinct options:

1. **Scope death to discovered patterns only.** Initial vocab atoms are never killed; only the patterns added by the discovery channel are subject to garbage collection. This matches the design rationale (death is for purging stale discovered patterns when the codebook moves under them).
2. **Keep death uniform but accept that mass-death of unreached vocab is correct.** If the system is operating in a regime where most vocab tokens never appear in the cue stream, dropping them frees capacity. Requires accepting that "death" is a corpus-statistics phenomenon, not a stale-pattern-detection phenomenon.
3. **Extend the run and accept whatever fires.** Run with n_cues=3000+ and report what dies. Risk: deaths still dominated by unreached vocab atoms, just in a more drawn-out way.

Option 1 is the architecturally-clean change. Option 2 is honest if accepted explicitly. Option 3 is the experimental-evidence-without-design-clarity path and should be a last resort.

The next session should pick between (1) and (2) before any more compute.

---

## Status update

Blocker #2 stays open. Updates to STATUS.md:
- The 5-seed `st03_death_*` runs do not constitute evidence on the death mechanism; they reproduce report 029.
- The mechanism is correctly implemented; the parameterization cannot fire death in n_cues=1500.
- The architectural decision (uniform death vs. discovered-only death) is now the blocker.

No code changes beyond additive telemetry. The new artifacts on disk under `reports/phase34_integrated_hebbian_st03_death_*` should be treated as a failed configuration attempt, not as a Phase 4 verification.
