# Report 034 — Saighi A_k self-inhibition prototype (seed 1)

**Date:** 2026-05-15
**Active phase:** 4
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift between cycles.
**Required controls per [phase-4-unified-design.md:318-323](../notes/emergent-codebook/phase-4-unified-design.md):** No-replay baseline ✓ (Conditions A and B in this run). Random-codebook control: not re-run for this mechanism prototype (already verified in report 026).
**Last verified result:** [Report 033](033_phase4_death_mechanism_diagnostic.md) — death mechanism diagnostic. Blocker #2 reshaped from "scope decision" to "A_k self-inhibition is the right-shape mechanism" per the [2026-05-15 paper synthesis](../notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md).
**Why this experiment now:** Single-seed prototype of the Saighi & Rozenberg (2025) per-pattern self-inhibition `A_k` as a drop-in replacement for the threshold-based death mechanism. Verifies (a) the mechanism wires correctly through the existing Phase 4 path, (b) telemetry is sensible, (c) preliminary effect direction is positive enough to justify the 5-seed sweep.

---

## Method

### Mechanism

Per [notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md](../notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md):

```
A_k ← A_k + β·1{retrieval converged to k}        # accumulate per successful retrieval
logits_k = β·sims_k - A_k                         # subtract from per-pattern logits before softmax
```

`A_k` is a per-pattern scalar carried alongside the existing Benna-Fusi u-chain in [ConsolidationState](../src/energy_memory/phase4/consolidation.py). Untouched patterns accumulate zero inhibition; repeatedly-visited attractors lose dominance smoothly.

### Implementation diff (additive)

- [consolidation.py](../src/energy_memory/phase4/consolidation.py): new fields `inhibition_gain`, `inhibition_decay` on `ConsolidationConfig`; new tensor `self.A` parallel to `self.u`; new methods `accumulate_inhibition()`, `inhibition_bias()`; lifecycle (add/remove) extended; optional decay in `step_dynamics`.
- [trajectory.py](../src/energy_memory/phase4/trajectory.py): `retrieve_with_trace` accepts `score_bias` (defaults None — backwards compatible). When provided, subtracted from `β·scores` before softmax.
- [replay_loop.py](../src/energy_memory/phase4/replay_loop.py): `retrieve_and_observe` and `run_replay_cycle` thread `inhibition_bias()` through retrieval; `retrieve_and_observe` calls `accumulate_inhibition` after `reinforce`. Bias re-fetched per replay-cycle iteration because `candidate_handler` may grow memory + consolidation between iterations.
- [experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py): per-cue Phase 4 retrieval (inlined) mirrors the same wiring; CLI flags `--inhibition-gain` and `--inhibition-decay`; checkpoint telemetry extended with `inhibition_{mean,p50,p90,p99,max,nonzero}`.
- 6 new unit tests in [tests/test_phase4_consolidation.py](../tests/test_phase4_consolidation.py) `TestSaighiInhibition`. All pass. 28 existing Phase 4 tests still pass.

### Configuration

Identical to the [reports/phase34_death_diag_seed1](phase34_death_diag_seed1/) baseline (= identical to canonical st=0.3 config), with only one knob different:

```
--updater-kind hebbian --seed 1 --success-threshold 0.3
--death-threshold 0.05 --death-window 10
--inhibition-gain 0.01            # NEW
--inhibition-decay 0.0            # NEW (Saighi basic monotonic form)
```

n_cues=1500, beta=10, replay_every=50.

JSON: [reports/phase34_saighi_seed1/phase34_results.json](phase34_saighi_seed1/phase34_results.json).

---

## Result

### Headline (Condition C, step=1500)

| metric | baseline (death_diag seed 1) | A_k (this run) | Δ |
|---|---:|---:|---:|
| top1 | 0.0705 | **0.0793** | **+0.0088** |
| topk (R@10) | 0.3524 | **0.3744** | **+0.0220** |
| cap_t_05 | 0.2203 | **0.2423** | **+0.0220** |

Conditions A (baseline_static) and B (phase3_reencode) are **byte-identical** between runs (as designed — A_k is wired only in the Phase 4 path).

### Trajectory (Condition C, every checkpoint)

| step | top1 b | top1 s | Δtop1 | topk b | topk s | Δtopk | cap b | cap s | Δcap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 0.079 | 0.079 | +0.000 | 0.339 | 0.339 | +0.000 | 0.211 | 0.211 | +0.000 |
| 300  | 0.066 | 0.079 | **+0.013** | 0.366 | 0.374 | +0.009 | 0.216 | 0.242 | **+0.026** |
| 600  | 0.070 | 0.079 | +0.009 | 0.366 | 0.374 | +0.009 | 0.220 | 0.242 | +0.022 |
| 900  | 0.070 | 0.079 | +0.009 | 0.352 | 0.374 | **+0.022** | 0.220 | 0.242 | +0.022 |
| 1200 | 0.070 | 0.079 | +0.009 | 0.352 | 0.374 | +0.022 | 0.220 | 0.242 | +0.022 |
| 1500 | 0.070 | 0.079 | +0.009 | 0.352 | 0.374 | +0.022 | 0.220 | 0.242 | +0.022 |

The A_k effect appears almost immediately (step 300) and plateaus by step 900. No regression in any of the three metrics.

### A_k accumulation trajectory (Condition C, scale 2)

| step | A_mean | A_p99 | A_max | A_nonzero | n_patterns | strength_p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 300  | 0.0007 | 0.0000 | 2.47  | **2** | 4097 | 0.208 |
| 600  | 0.0015 | 0.0000 | 5.05  | **3** | 4097 | 0.108 |
| 900  | 0.0022 | 0.0000 | 7.54  | **3** | 4097 | 0.069 |
| 1200 | 0.0029 | 0.0000 | 9.96  | **3** | 4097 | 0.048 |
| 1500 | 0.0037 | 0.0000 | 12.48 | **3** | 4097 | 0.034 |

**Telemetry matches Saighi's prediction exactly.** Only 3 of 4097 patterns at scale 2 ever accumulate inhibition — corresponding to the same ~3 patterns the report-033 telemetry identified as the only ones retrieved as top-1. A_max grows to 12.48 (≈ −1.25 in cosine-similarity units at β=10), which is a substantial penalty applied selectively to the over-used attractors.

Scales 3 and 4 (final checkpoint): A_nz=22/2080 at scale 3, 16/1072 at scale 4 — broader top-1 distributions at coarser scales, also as expected.

### Death mechanism

`deaths_total=0` across all 5 checkpoints. With A_k active, death is no longer the load-bearing mechanism for dominance management; the inhibition handles dominance decay gracefully without removal. This is exactly the architectural argument from the [synthesis note](../notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md). The threshold-based death mechanism can be deprecated, kept as opt-in for future variants.

---

## FEP / anti-homunculus audit

| Check | Status | Note |
|---|---|---|
| `accumulate_inhibition` = local geometric dynamic on retrieval winner | ✓ | Per-pattern scalar incremented on a passive geometric event (which attractor the network settled to) |
| `score_bias` = a measurement of accumulated inhibition, not an arbitration | ✓ | Subtracted from logits as part of softmax dynamics, not as a gating condition |
| No supervisor decides which pattern is inhibited | ✓ | Inhibition follows retrieval outcomes, which follow geometry |
| Parameterization is rate-shaped not threshold-shaped | ✓ | `inhibition_gain` is a per-event magnitude, not "if metric > X then do Y" |

This is structurally a different shape from the death threshold + window. It passes the anti-homunculus filter cleanly.

---

## What this report establishes — and does NOT establish

**Does establish:**
- The mechanism wires correctly through both `retrieve_and_observe` and exp 19's inlined Phase 4 path.
- Telemetry is sensible and matches the Saighi prediction (selective inhibition of over-used attractors only).
- Backwards compatibility is preserved (default `inhibition_gain=0.0` keeps prior behavior; Conditions A and B byte-identical to baseline).
- 6 new unit tests pass; 28 existing Phase 4 tests still pass.
- On seed 1, all three Phase 4-relevant metrics improve simultaneously: top1 +0.009, R@10 +0.022, cap_t05 +0.022.

**Does NOT establish:**
- Cross-seed reliability. This is one seed. Seed 1 has been an *outlier-positive* seed in prior reports (largest C−B advantage in report 029). The effect must be verified on the full 5-seed set {17, 11, 23, 1, 2} before any claim about the headline.
- Independence from the seed-23 outlier (blocker #5). Seed 23 has been the persistent negative outlier; its behavior under A_k is the most informative single test.
- Phase 4 graduation. Per the design, both ΔR@K *and* Δcap-coverage need disjoint-from-zero CIs. Multi-seed required.
- The right value of `inhibition_gain`. 0.01 was a first-pass guess. A small sweep (0.005, 0.01, 0.02, 0.05) would characterize the trade-off; Saighi p. 7 Fig. 6 shows smaller β → more iterations / softer effect, larger β → faster but more spurious dynamics.

---

## Next session — recommended actions in order

1. **5-seed sweep at `--inhibition-gain 0.01`** with the canonical seeds {17, 11, 23, 1, 2}. On M5 Pro MPS this is ~2h; on Colab A100 ~30 min. Decision after: if 4/5+ seeds show R@10 ≥ baseline AND CI on Δtop1 includes positive territory, A_k graduates from "prototype" to "verified mechanism" and blocker #2 closes.
2. **Inspect seed-23 trajectory specifically.** If seed 23 also flips positive under A_k, blocker #5 partially closes too (the geometric idiosyncrasy was being amplified by the dominance feedback loop A_k breaks).
3. **β sweep (later, if 5-seed verifies).** Characterize the trade-off across `inhibition_gain ∈ {0.005, 0.01, 0.02, 0.05}` on seed 1 alone. Don't run before the 5-seed result — premature tuning.

---

## Status implications

- Blocker #2 stays open pending 5-seed verification, but the mechanism is implemented and the seed-1 evidence is in the expected direction.
- Blocker #6' (top1 regression) shows reversal under A_k on seed 1. If this holds across seeds, the regression is downstream of dominance feedback, not an inherent Hebbian property.
- Blocker #3 (Δcap-coverage) shows +0.022 on seed 1. Cap-coverage had been the persistently un-verified Phase 4 headline; if multi-seed holds, this is direct progress.
- Deaths remain at 0. The death mechanism's relevance is now contingent on whether A_k alone is sufficient — likely yes per Saighi's architecture, but multi-seed data will say.
