# Report 011 — Synergy probe across the Phase 4 pipeline

**Date:** 2026-05-13
**Experiment:** `experiments/28_synergy_probe_phase4.py`
**Raw data:**
- `reports/synergy_probe_phase4.json` (default operating point)
- `reports/synergy_probe_phase4_aggressive.json` (forced blend regime)

## Motivation

Brainstorm Idea 8 (2026-05-13) proposed the GIB-inspired synergy
estimator as a Phase 5 headline metric. The estimator landed as a
diagnostics primitive in this session and was unit-tested in
isolation — but had no measurements on real Phase 4 artifacts.

This probe applies the estimator to the three checkpoints in the
Phase 4 pipeline:

1. **Raw bindings.** Encoded windows `bundle({ bind(position_p,
   codebook[token_p]) : p in window })` — the substrate's own
   compositional output, the structural ceiling.
2. **Hopfield-settled states.** The output of `retrieve_with_trace`
   under perturbed queries, what downstream consumers actually see.
3. **Replay-resolved candidates.** Settled states that crossed
   `resolve_threshold` during a replay cycle — the artifacts that
   would feed back into the codebook through the consolidation
   channel.

This is also the **second of the two prerequisite checks** named in
[reports/008_lsr_kernel_sweep.md](008_lsr_kernel_sweep.md) for re-
opening the LSR coexistence question: does the project's substrate
actually exhibit a regime disjoint in measured artifacts, or was the
disjoint a theoretical concern that doesn't show up in practice?

## Setup

Minimal Phase 4 pipeline per seed:

- D = 4096, window size 3, codebook size 64, 32 random windows stored.
- `TracedHopfieldMemory` + `UnifiedReplayMemory` with the upgraded
  replay store (tag_overlap = 0.7, suppression on).
- Drive perturbed retrievals; capture settled states + replay candidates.
- For each captured artifact, measure `mean_synergy(roles=positions,
  fillers=codebook[token_ids], bindings=artifact)` — the recovery of
  every (position, atom) slot via unbinding.

Two operating conditions:

| condition                | cue_noise | beta | store_threshold | resolve_threshold |
| ------------------------ | --------- | ---- | --------------- | ----------------- |
| Default (intended)       | 0.15      | 10.0 | 0.05            | 0.5               |
| Aggressive (forced blend)| 1.5       | 3.0  | 0.001           | 0.2               |

Five seeds for default ({11,23,41,53,67}); three seeds for aggressive
({11,23,41}).

## Headline result

### Default operating point (β=10, cue_noise=0.15)

```
stage         recover  sim_role  sim_bind   synergy
raw            0.524    -0.001    -0.004     0.521
settled        0.524    -0.001    -0.002     0.520
```

| metric                    | value        |
| ------------------------- | ------------ |
| raw mean synergy          | 0.519        |
| settled mean synergy      | 0.519        |
| **settled / raw ratio**   | **1.000 ± 0.001** |
| n_seeds with candidates   | 0 / 5        |

At the project's intended operating regime, **Hopfield settling
preserves compositional structure exactly** — the synergy of the
settled state matches the synergy of the stored pattern to within
floating-point noise. The replay candidate channel never activated
because settling resolved every cue too cleanly (gate signal stayed
below the store threshold).

### Aggressive condition (β=3, cue_noise=1.5)

```
stage         recover  sim_role  sim_bind   synergy
raw            0.524    -0.002    -0.002     0.521
settled        0.097    -0.001    -0.000     0.092
candidates     0.097    -0.001    -0.000     0.090
```

| metric                    | value          |
| ------------------------- | -------------- |
| raw mean synergy          | 0.519          |
| settled mean synergy      | 0.097          |
| candidate mean synergy    | 0.090          |
| **settled / raw ratio**   | **0.186 ± 0.015** |
| candidate / raw ratio     | 0.174          |
| n_seeds with candidates   | 3 / 3          |

Under blend-regime conditions, **settled synergy collapses by ~5.4×**.
Crucially, the replay candidate channel does not recover structure —
candidates inherit the same destroyed synergy (0.174 ratio vs.
settled's 0.186 ratio). Replay re-settling under the *same* dynamics
that destroyed the synergy in the first place cannot reconstruct it.

### Interpretation of the absolute values

Raw synergy is 0.52, not the 0.8+ seen in the unit tests for single
role-filler pairs. This is correct: each window bundles 3 (position,
atom) bindings, so unbinding any one position produces the target
atom plus crosstalk noise from the other two bindings. The 0.52
figure is the **substrate's actual synergy on multi-slot bundles**;
it is the structural ceiling for Phase 4 settled states.

## Implication for the LSR re-decision

Report 008 deferred LSR with two prerequisite checks. With report 010
closing the first, this report closes the second:

> **Does the GIB synergy estimator detect an actual regime-disjoint
> when run on real Phase 4 consolidated bindings?**

**Answer: yes, the disjoint is real and measurable, but only outside
the project's intended operating regime.** The two conditions in this
report sit at opposite sides of the disjoint:

- At the intended operating point, the substrate behaves like a
  sharp-memorization regime: synergy 1.000, no candidate channel
  activation, no problem to solve.
- Under aggressive conditions designed to push into the blend regime,
  synergy collapses ~5× and the candidate channel — which the
  project hoped would recover structure — instead carries the
  destroyed structure forward.

This validates the LSR question as substantive — there *is* a regime
the project's current dynamics fall into where compositional
structure is genuinely lost. But it also clarifies the priority: the
project does not currently operate in that regime, so the LSR
mechanism is not the highest-leverage next thing to build. The
synergy probe is now itself the right tool for detecting drift: if
future experiments show `settled_synergy_ratio` falling below ~0.9,
that is the trigger for revisiting LSR (or any other regime-
preserving mechanism).

## Recommended decision: keep LSR deferred (re-confirmed)

The two prerequisite checks from report 008 are now both closed:

1. **(Report 010)** Permutation slots produce a 4.4× directionality
   gain in coupled recall — the structural gap that motivated the
   LSR question is being addressed at the temporal-binding layer, not
   the kernel layer.
2. **(This report)** The regime disjoint is real but is not biting at
   the project's intended operating point. Synergy is fully preserved
   when β and cue noise are inside their normal ranges.

The LSR `kernel="lsr"` parameter and tests in
`tests/test_torch_hopfield_lsr.py` remain in place as documentation
of the β-invariance finding, but no further work on the unnormalized-
gradient variant is recommended until either:

- A future Phase 5 or Phase 7 design forces lower-β operation
  (e.g. wider blend regimes for creative recombination), or
- The synergy probe on real long-run experiments shows ratio < 0.9
  even at intended β.

## Anti-homunculus check / FEP audit

The synergy estimator is a pure measurement primitive. It computes a
deterministic geometric quantity from (role, filler, binding) triples
and never feeds back into any update rule. No controller reads its
output, no branch fires based on its value. Passes trivially.

The probe experiment itself only reads from the pipeline — it does
not write back. The `candidate_handler` callback intentionally
returns `None` so candidates are recorded for measurement but never
added to memory or consolidation. The dynamics of the probe run are
identical to a baseline Phase 4 run.

## Recommended next steps

1. **Make synergy a standard drill-down in Phase 4 unified
   experiments.** Add `mean_synergy(settled_states) /
   mean_synergy(raw_bindings)` to the output of
   `experiments/18_phase4_unified_experiment.py` as a structural
   health diagnostic. Cheap to compute, valuable for catching regime
   drift.
2. **Use synergy as the proposed Phase 5 headline metric** (the
   original Idea 8 use case). Phase 5 introduces layer-2 bindings
   between codebook atoms; the synergy of those bindings vs. atoms
   alone is exactly the structural claim Phase 5 needs to make. The
   `atom_alone_synergy` baseline already provides the null
   hypothesis.
3. **Optional drill-down: scan β at fixed cue_noise** to map where
   the synergy ratio crosses 0.9. That gives a concrete operating
   envelope (β_min, noise_max) that the project can target with
   downstream architectural decisions.
