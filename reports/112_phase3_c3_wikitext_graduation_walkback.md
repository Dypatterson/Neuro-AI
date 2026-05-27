# Report 112 — C.3 wikitext signal is operating-point-fragile; Phase 3 graduation declined; Path γ pivot

**Status:** complete. Path C closed as inconclusive. Phase 3 NOT graduated. Phase 5′ remains paused. New active work: **Path γ — Phase 3 mechanism redesign per [literature-and-principles.md](../notes/emergent-codebook/literature-and-principles.md)**.

**Date:** 2026-05-27.

## Preamble (per CLAUDE.md experiment-preamble requirement)

- **Active phase:** Phase 3 reopen — Path C (closing this session).
- **Headline metric per [phase-3-deep-dive.md:180-189](../notes/emergent-codebook/phase-3-deep-dive.md):** regime-stratified Recall@K on masked-token contextual completion, vs. genuine shuffled-token-with-consolidation control, n ≥ 10 seeds, Wilson CIs; criterion = CI-disjoint in at least one regime stratum.
- **Required controls per same spec:** the standard condition runs C.2 consolidation on the real wikitext token-id assignment; the control runs the SAME consolidation pipeline on a random permutation of the codebook row order, with byte-identical training data and substrate atom set. Path α (2026-05-26) replaced the original "fresh random codebook" no-consolidation control with this stricter shuffled-token-with-consolidation control.
- **Last verified result:** [Report 111](111_phase5_prime_range_replay_downstream_viability.md) (the Reports 062-111 chain plus the 2026-05-26 Path C C.1-C.2 closure entries in [notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](../notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md)). The 2026-05-26 Path α smoke at n=3 returned null at the synthetic operating point.
- **Why this experiment now:** Path C exit criterion #3 (regime-stratified Recall@K beats shuffled-token control at n ≥ 10 with at least one CI-disjoint stratum). This experiment chain is the canonical attempt at meeting that criterion on real-corpus statistics, after the 2026-05-26 synthetic null and the precommit's recommended Path β (WikiText-2).

## TL;DR

A four-stage Colab experiment chain (v3 → v6) found that the Path C C.2 dynamics + α_anti repulsion produce a small, operating-point-fragile corpus-specific learning signal on WikiText-2. The chain initially produced what looked like Phase 3 graduation (v3, n=10), but progressive verification with disjoint seed sets, β robustness probes, a D-sweep, and a per-seed paired D-comparison revealed that the apparent v3 magnitude (Δ ≈ +0.083) was a tail draw from a noisy seed distribution whose typical effect is ≈ +0.03–0.04 — borderline at n=10, statistically detectable only at n ≥ 30 pooled, and non-monotonic in substrate dimensionality (positive in D ∈ {4096, 8192}, reversed at D ∈ {1024, 2048, 16384}). The strict reading of the C.3 spec is technically satisfied (pooled n=30 CI-disjoint at D=4096 and D=8192 in primary strata), but the spirit is not: the per-seed paired comparison falsifies a substrate-capacity story, the D=16384 reversal falsifies a monotonic-D story, and the typical effect magnitude sits at the noise floor of what the criterion can detect.

**Decision:** Phase 3 does not graduate. Path C closes as inconclusive. Path γ (Phase 3 mechanism redesign per literature-and-principles.md) is the next research direction.

## Experiment chain — four Colab runs

All runs share the v3 baseline configuration unless noted: `wikitext-2-raw-v1, lr_pull=0.1, lr_push=0.05, n_consolidation_events=1000, alpha_anti=0.01, repulsion_step_size=0.05, β=10, K=5, D=4096, landscape_size=64, window_size=8, vocab_cap=1000`. All runs use the same C.3 driver at [experiments/c3_phase3_exit_criterion.py](../experiments/c3_phase3_exit_criterion.py) with three load-bearing local patches applied via inline `git apply` in each notebook (CLI flags for `--lr-pull`/`--lr-push`, kernel-trick eigvalsh fix, Salesforce/wikitext namespace fix — see "Patches" section below).

### v3 — first apparent graduation (n=10, seeds 0..9)

Notebook [scripts/colab_c3_followup_v3_perseed_fanout.ipynb](../scripts/colab_c3_followup_v3_perseed_fanout.ipynb), Drive results `c3_followup_v3_2026-05-27/`.

The 4-condition × 10-seed parallel CUDA run was the first n=10 attempt at the C.3 criterion on wikitext. The `n10_wikitext_base` cell (lr_pull=0.1, n_events=1000) produced two CI-disjoint strata:

| stratum | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---|---|---|---:|---|
| calibrated/tight | 0.375 [0.350, 0.401] | 0.292 [0.268, 0.316] | +0.083 | **YES** |
| default/spread | 0.260 [0.241, 0.280] | 0.205 [0.188, 0.223] | +0.055 | **YES** |

By the strict letter of [phase-3-deep-dive.md:180-189](../notes/emergent-codebook/phase-3-deep-dive.md), this is graduation. **No STATUS.md update was made on the basis of v3 alone** — the result was flagged to the project owner with a request for robustness verification before committing.

Worth noting: the strongest-consolidation conditions in v3 (`n10_wikitext_best` at lr_pull=1.0, n_events=3000) produced *null* results, and `n10_wikitext_mid` (lr_pull=0.5) showed slightly negative deltas. The signal appears at the *weakest* consolidation setting, which is the opposite of what the prior synthetic sweep direction would predict.

### v4 — robustness probe (3 conditions × 10 seeds = 30 procs)

Notebook [scripts/colab_c3_robustness_probe_v4.ipynb](../scripts/colab_c3_robustness_probe_v4.ipynb), Drive results `c3_robustness_v4_2026-05-27/`.

Three orthogonal probes against v3's `wikitext_base` config:

| probe | varies | result vs v3 |
|---|---|---|
| `replicate_seeds10_19` | new seed set 10..19, same op point | Δ_default/spread = +0.023 (was +0.055), **not disjoint** |
| `beta30_seeds0_9` | β=30 instead of 10, same seeds 0..9 | Δ_default/spread = +0.017, **not disjoint** |
| `D2048_seeds0_9` | D=2048 instead of 4096, same seeds 0..9 | Δ_default/spread = **−0.011** (reversed), **not disjoint** |

**Direction preserved in 2/3 probes; magnitudes ≈ 1/3 of v3's; D=2048 reversed.** This was the first signal that v3's effect size was inflated by lucky seeds.

### v5 — D-sweep + pooled larger-n (4 D × 10 seeds = 40 procs)

Notebook [scripts/colab_c3_dsweep_and_largeN_v5.ipynb](../scripts/colab_c3_dsweep_and_largeN_v5.ipynb), Drive results `c3_v5_2026-05-27/`.

D-sweep with fresh seeds 30..39 at D ∈ {1024, 2048, 4096, 8192} plus pooled aggregation reading v3 + v4 replicate JSONs from Drive to produce an n=30 estimate at D=4096:

| D | n | default/spread Δ | disjoint? |
|---:|---:|---:|---|
| 1024 | 10 | −0.004 | no |
| 2048 | 10 | −0.027 | no |
| 4096 | 30 (pooled v3 + v4 + v5) | +0.020 | no |
| 8192 | 10 (v5 only) | +0.057 | **YES** |

Two surprising findings:
- **At D=4096 with seeds 30..39 alone, Δ = −0.017** (reversed). Three independent n=10 seed groups at D=4096 gave Δ ∈ {+0.055, +0.023, −0.017} — wide variance centered near zero.
- **D=8192 with seeds 30..39 produced v3-magnitude effects in both strata** (calibrated/tight Δ +0.081, default/spread Δ +0.057, both CI-disjoint).

This suggested a substrate-capacity story: the mechanism graduates only at D ≥ 8192, and the v3 D=4096 result was the borderline-detectable edge of the same effect. v6 was designed to test this hypothesis.

### v6 — D=8192 confirmation (D=8192 seeds 0..19 + D=16384 seeds 30..39 = 30 procs)

Notebook [scripts/colab_c3_d8192_confirmation_v6.ipynb](../scripts/colab_c3_d8192_confirmation_v6.ipynb), Drive results `c3_v6_2026-05-27/`.

Two probes:
- D=8192 seeds 0..19 (20 procs) — pool with v5 D=8192 seeds 30..39 → n=30 confirmation at D=8192.
- D=16384 seeds 30..39 (10 procs) — extends D-curve to test whether the effect grows or saturates.

The aggregation cell produced three outputs:

**(1) Pooled n=30 at D=8192:**

| stratum | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---|---|---|---:|---|
| calibrated/tight | 0.367 [0.352, 0.381] | 0.306 [0.293, 0.321] | +0.060 | **YES** (margin 0.031) |
| default/spread | 0.257 [0.246, 0.268] | 0.216 [0.206, 0.226] | +0.041 | **YES** (margin 0.020) |

**The strict criterion is met at D=8192 with n=30.** Both primary strata CI-disjoint, comfortable margins, both directions positive.

**(2) Per-seed paired D=4096 vs D=8192 for seeds 0..19** (the most diagnostic test of the substrate-capacity hypothesis):

Per-seed Δ values for default/spread:

| seed | Δ_4096 | Δ_8192 | shift | sign_flip neg→pos? |
|---:|---:|---:|---:|:---:|
| 0 | −0.090 | +0.105 | +0.195 | ✓ |
| 1 | +0.465 | +0.380 | −0.085 |  |
| 2 | +0.065 | +0.005 | −0.060 |  |
| 3 | −0.190 | +0.010 | +0.200 | ✓ |
| 4 | −0.015 | −0.145 | −0.130 |  |
| 5 | +0.165 | +0.060 | −0.105 |  |
| 6 | +0.070 | +0.090 | +0.020 |  |
| 7 | +0.095 | −0.095 | −0.190 |  |
| 8 | −0.060 | +0.025 | +0.085 | ✓ |
| 9 | +0.045 | +0.020 | −0.025 |  |
| 10 | +0.020 | +0.005 | −0.015 |  |
| 11 | +0.250 | −0.035 | −0.285 |  |
| 12 | +0.130 | +0.095 | −0.035 |  |
| 13 | −0.025 | −0.110 | −0.085 |  |
| 14 | +0.040 | −0.030 | −0.070 |  |
| 15 | −0.030 | −0.050 | −0.020 |  |
| 16 | +0.005 | +0.070 | +0.065 |  |
| 17 | −0.150 | +0.070 | +0.220 | ✓ |
| 18 | +0.005 | +0.035 | +0.030 |  |
| 19 | −0.010 | +0.145 | +0.155 | ✓ |

Counts:
- Seeds with Δ_8192 > Δ_4096: **8/20 (40%)**
- Seeds with Δ_8192 > 0: 14/20 (70%)
- Seeds with neg→pos flip: 5/20

Per-seed means:
- D=4096 (seeds 0..19): **+0.039**
- D=8192 (seeds 0..19): **+0.033**

**The substrate-capacity hypothesis is falsified.** If "more D = stronger effect" were true, we would expect a substantial majority of seeds to improve. Instead 40% improve and 60% degrade going from D=4096 to D=8192. The per-seed mean is actually slightly *lower* at D=8192. The pooled D=8192 > D=4096 difference in v5 was driven by regression-to-the-mean: seeds 30..39 happened to land in the unlucky tail at D=4096 (Δ = −0.017) and the lucky tail at D=8192 (Δ = +0.057), but resampling at D=8192 with seeds 0..19 produced the same mean effect as at D=4096.

**(3) Full D-dependence curve** at largest available n per D:

`default/spread`:

| D | n | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---:|---:|---|---|---:|---|
| 1024 | 10 | 0.236 [0.218, 0.256] | 0.240 [0.222, 0.259] | −0.004 | no |
| 2048 | 10 | 0.230 [0.212, 0.249] | 0.258 [0.239, 0.277] | −0.027 | no |
| 4096 | 30 | 0.254 [0.243, 0.265] | 0.233 [0.223, 0.244] | +0.020 | no |
| 8192 | 30 | 0.257 [0.246, 0.268] | 0.216 [0.206, 0.226] | +0.041 | **YES** |
| 16384 | 10 | 0.197 [0.180, 0.215] | 0.222 [0.204, 0.241] | −0.025 | no |

`calibrated/tight`:

| D | n | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---:|---:|---|---|---:|---|
| 1024 | 10 | 0.334 [0.310, 0.359] | 0.340 [0.316, 0.366] | −0.006 | no |
| 2048 | 10 | 0.344 [0.319, 0.370] | 0.371 [0.346, 0.398] | −0.027 | no |
| 4096 | 30 | 0.367 [0.353, 0.382] | 0.335 [0.321, 0.350] | +0.032 | **YES** (margin 0.003) |
| 8192 | 30 | 0.367 [0.352, 0.381] | 0.306 [0.293, 0.321] | +0.060 | **YES** (margin 0.031) |
| 16384 | 10 | 0.267 [0.245, 0.291] | 0.303 [0.280, 0.327] | −0.036 | no |

**D-dependence is non-monotonic with reversals at both small and large D.** The effect is positive in a narrow envelope D ∈ {4096, 8192} and reversed outside it.

## Key findings

**Finding 1 — strict criterion satisfied at n=30 in both directions.** At D ∈ {4096, 8192} pooled n=30, both primary strata are CI-disjoint with positive Δ. By the literal text of [phase-3-deep-dive.md:180-189](../notes/emergent-codebook/phase-3-deep-dive.md), this would qualify as graduation.

**Finding 2 — typical per-seed Δ ≈ +0.03 with huge variance.** Individual-seed Δ at the original operating point ranges from −0.19 to +0.465 across n=20 (seeds 0..19 at D=4096). The standard deviation across per-seed Δ is ≈ 0.13, compared to a mean of ≈ 0.04 — so the per-seed signal-to-noise ratio is ≈ 0.3. The v3 magnitude (+0.083) is ≈ 1.4σ above the mean of the per-seed distribution; the v5 D=8192 seeds-30..39 magnitude (+0.057) is ≈ 0.7σ above. Both are within the realistic range of tail draws.

**Finding 3 — D-dependence is non-monotonic.** The effect appears in D ∈ {4096, 8192} and reverses at D ∈ {1024, 2048, 16384}. No simple capacity-scaling story explains this shape.

**Finding 4 — substrate-capacity hypothesis falsified by paired test.** When the same n=20 seeds (0..19) are evaluated at both D=4096 and D=8192, only 40% improve. Per-seed means are essentially identical (+0.039 vs +0.033). The pooled D=8192 > D=4096 difference in v5 was a regression-to-the-mean artifact, not a substrate-dim effect.

**Finding 5 — the v3 graduation magnitude was a tail-of-distribution draw, not the typical effect.** The walk-back chain v3 (+0.083) → v4 replicate (+0.023) → v5 D=4096 seeds-30..39 (−0.017) demonstrates the per-seed variance. Pooling across all three groups (n=30) gives Δ ≈ +0.020 — the actual typical effect — with calibrated/tight just barely CI-disjoint (margin 0.003) and default/spread not.

## Decision: strict reading satisfied, spirit reading not, do not graduate

The strict text of the C.3 spec ("CI-disjoint in at least one regime stratum, n ≥ 10 seeds") is met multiple ways: v3 at n=10 met it, v5 D=8192 at n=10 met it, pooled n=30 at D=4096 met it in one stratum, pooled n=30 at D=8192 met it in two strata.

The spirit of the criterion is not met. The criterion was written assuming an effect that is CI-disjoint at n=10 is also a robust, generalizable effect that supports further development. Here the effect:

1. Has per-seed variance that completely swamps the mean (σ/μ ≈ 3.3).
2. Appears only in a narrow operating-point envelope (D=4096–8192, β=10) and reverses outside it.
3. Has a magnitude (typical Δ ≈ +0.03) at the noise floor of what the criterion can detect — the strict n=30 pooled disjoint margin in calibrated/tight at D=4096 is 0.003, which means a single different seed could flip it.
4. Does not follow any predicted mechanism story (substrate-capacity falsified, weak-consolidation-beats-strong unexplained, non-monotonic D-dependence).

**This is not a foundation for Phase 5′.** A Phase 5′ ΔE bridge built on top of this Phase 3 mechanism inherits all of its operating-point fragility. The 2026-05-26 audit's mandate that "any future bridge / readout attempt is diagnosable per-atom and per-stratum" assumes the underlying Phase 3 signal is real and stable enough that diagnostics localize meaningful failures. With per-seed σ ≈ 0.13, diagnostics would localize *noise*.

**Path C closes as inconclusive.** The diagnostic-as-actuator framework (C.1 + C.2) was implemented cleanly with anti-homunculus reviewer PASSes, and the framework produces measurable structural change in the codebook (regime distribution shifts under consolidation — see C.3 first smoke entry in the precommit running log). But the structural change does not produce a robust corpus-specific learning signal. The mechanism is too weak.

## What this says about the C.2 mechanism

The five C.2 diagnostic-as-actuator dynamics (anti-collapse force, splitting-tension modulation, cap-coverage gradient, metastability replay-priority, drift replay-tension) were anti-homunculus-clean and produced measurable structural change. Combined with α_anti repulsion (Path α), the system measurably reshapes the codebook geometry: ~90% of atoms re-classify from `spread` to `tight` under calibrated θ′ during consolidation. **That structural change does not translate into corpus-specific learning beyond a small, fragile signal.**

Two ways to interpret this:

1. **The mechanism is doing the right *kind* of thing but at the wrong *strength*.** The C.2 dynamics are tightening basins (within-basin variability reduces) but the resulting codebook geometry doesn't carry enough corpus-specific information to robustly beat the shuffled-token control. Stronger pull (v3 `wikitext_best` at lr_pull=1.0, n_events=3000) does *not* help — it produces a null. Weaker pull also doesn't help. The mechanism's signal floor appears to be intrinsic, not a hyperparameter problem.

2. **The mechanism is doing the wrong kind of thing entirely.** The structural change it produces (basin tightening) may be orthogonal to what's needed for corpus-specific contextual completion. A different mechanism family — e.g., predictive-coding-style local error gradients (Dorrell-Whittington direction), Hopfield/MHN training-time methods (M2 path), or Hyperseed-style content-addressable updates — might produce structural change of a different shape that does carry corpus-specific signal.

Path γ targets interpretation 2.

## Why not the M2 training-time-intervention path

The 2026-05-24 brainstorm flagged (a.1) M2 (EqProp + role-shuffled negatives + DSM warm-start on existing FHRR) as the heaviest of the open paths. It was deferred during Path C in favor of the diagnostic-as-actuator framework. M2 modifies *training* of the codebook, not its algebraic bind structure, and remains a defensible move if the diagnosis is "the C.2 mechanism's training signal is too weak". However:

- The v3 → v6 chain showed that *changing consolidation strength on the existing C.2 framework does nothing* (`wikitext_best` at lr_pull=1.0 nulled). M2 sits in the same training-signal-shape regime as C.2, just with a different specific loss function (energy-based vs Hebbian-pull). It's defensible to suspect M2 would hit the same noise floor.
- M2 was scoped at ~2-3 weeks of work for a defensible smoke. With Path γ being explicitly mechanism-redesign, the cost-benefit of M2 vs a Path γ design pass favors Path γ.

If a Path γ design pass *itself* identifies a mechanism whose shape is M2-adjacent (e.g., energy-based training of bundle-first scene memory), M2 work re-enters scope under a fresh precommit at that point.

## Path γ — recommended next research direction

**Path γ is not a single experiment; it is a design-and-precommit phase.** The deliverables for Path γ before any code lands:

1. **Mechanism-family survey.** Read [literature-and-principles.md](../notes/emergent-codebook/literature-and-principles.md) plus the 2026-05-24 brainstorm `unconsidered-paths` material and identify three to five candidate mechanism families that could plausibly produce robust corpus-specific learning on the FHRR substrate. Candidates currently on the table (from earlier brainstorms):
   - **Predictive coding** in the codebook layer (Dorrell-Whittington direction). Already partially explored by the range-shaped replay sampler (Reports 068-074); the algorithm is wired but the downstream lane was closed by Report 111. The mechanism question now is whether a *consolidation update rule based on prediction error* (not just sampling shape) carries more signal than Hebbian pull.
   - **Bundle-first scene memory** as the structural memory primitive (the (c.2) direction strengthened by Report 067). Reframes Phase 3 around storing scene bundles in scene-MHN rather than per-atom slots; consolidation becomes scene-level rather than atom-level.
   - **Energy-based training** of the codebook (M2-adjacent). EqProp or contrastive-divergence-style updates on the FHRR substrate.
   - **Hyperseed-style content-addressable updates** (per the 2026-05-24 brainstorm citations, with attribution caveats from that session's audit).
   - **Self-Organizing Language** (Eugenio direction per literature-and-principles.md) — a mechanism that builds codebook structure from sequential exposure rather than from Hebbian event triggering.

2. **Precommit selection.** Pick one candidate. The precommit must specify: (a) the diagnostic↔actuator identity (the C.2.1-C.2.5 H1-H7 discipline still applies — no metric-triggered routing); (b) a falsifiable graduation criterion that includes operating-point robustness (e.g., "CI-disjoint at n=10 *AND* per-seed Δ > 0 in at least 70% of seeds"); (c) an anti-homunculus reviewer pass on the design before code lands.

3. **The Path C exit criterion in [phase-3-deep-dive.md:180-189](../notes/emergent-codebook/phase-3-deep-dive.md) should be revised** before any Path γ candidate runs against it. The current spec is met by lucky-seed-set tail draws on a noisy mechanism. A reasonable revision: "CI-disjoint in at least one stratum at n ≥ 10 seeds, AND per-seed paired robustness — at least 70% of independent seed pairs show Δ > 0, computed from per-seed Δ stratum-pooled across `default` and `calibrated` regime classifiers." This rules out the v3 failure mode (lucky tail at n=10 with negative seed pair).

Path γ does not commit to a specific mechanism in this report. The mechanism-family survey is the immediate next session.

## Patches landed during this session

Three local patches to the working tree, applied via inline `git apply` heredoc in each Colab notebook. Not pushed to the remote branch as of this report. All three are load-bearing for reproduction:

1. **`experiments/c3_phase3_exit_criterion.py`** — added `--lr-pull` / `--lr-push` CLI flags so the consolidation-strength sweep was runnable. 12/12 driver tests pass post-patch.
2. **`src/energy_memory/phase4/consolidation.py`** — kernel-trick eigvalsh fix in `_spatial_bimodality_signal`. The original code computed `σ = diffs.conj().T @ diffs / n` (D×D, rank ≤ n_members ≤ 64 ≪ D) and called `torch.linalg.eigvalsh(σ)`. On CUDA, cuSOLVER raised `LinAlgError 4095` ("ill-conditioned ... too many repeated eigenvalues") on the (D − n) trivial zero eigenvalues; on CPU LAPACK raised `LinAlgError 5/12` similarly. The fix uses the n×n Gram matrix `diffs @ diffs.conj().T / n`, which shares the same non-zero spectrum (standard "kernel trick" identity) and is well-conditioned. Numerically byte-identical at the λ_1 / λ_2 layer (verified relative error ≈ 1.4e-6 at D=4096, float32). 26/26 splitting-tension tests and 76/76 phase4 tests pass post-patch. A defensive CPU fallback is preserved for pathological inputs.
3. **`src/energy_memory/phase2/corpus.py`** — switched `load_dataset("wikitext", name)` to `load_dataset("Salesforce/wikitext", name)`. The bare-name form worked with older HF stacks; from `huggingface_hub` ≈ 0.30+ the URI parser requires `namespace/name` and rejects the bare form with `HfUriError`.

## Anti-homunculus check

The kernel-trick eigvalsh fix is byte-identical at the diagnostic measurement level (top-2 eigenvalues match to float32 noise floor); it does not change the C.2.2 dynamic's behavior or introduce arbitration. The `--lr-pull` / `--lr-push` CLI flags expose existing internal parameters; they do not introduce metric-triggered routing. The Salesforce/wikitext fix is a corpus-loader correctness fix. No anti-homunculus risk introduced by any patch.

The verdict logic in the v6 aggregation cell (CONFIRMED / PARTIAL / FALSIFIED branching) is *not* mechanism logic — it is a post-hoc reporting branch in offline analysis, equivalence-class with reading the results table by eye and writing prose. No CP7 risk.

## Limitations

- **Single corpus.** wikitext-2-raw-v1 only. wikitext-103 and other natural corpora untested.
- **Single window size.** window=8 throughout. Shorter (window=4) and longer (window=16, 32) untested.
- **Single vocab cap.** vocab_cap=1000 (top-K wikitext tokens + UNK + MASK = 1002 effective). Larger vocab caps (5000, 10000) untested.
- **Single (lr_pull, n_events) operating point for the headline.** Tested only at (0.1, 1000) for the wikitext_base condition. The full v3 sweep tested other strengths but all at D=4096.
- **All runs at β=10** except v4 `beta30_seeds0_9`. β-sensitivity beyond the single-probe walk-back at β=30 untested.
- **No D=12288 or other intermediate-D probes.** The D-curve has gaps between 8192 and 16384.
- **No alpha_anti sweep.** alpha_anti=0.01 throughout. The C.2.2-Path-α interaction may be α-sensitive.
- **No alternate consolidation-trigger schedules.** Consolidation fires per consolidation-buffer fill; alternative schedules (periodic, error-magnitude-triggered) untested.
- **Single PyTorch version on Colab A100.** All runs on the same Colab runtime image. CUDA-specific or PyTorch-specific behavior cannot be ruled out.

## Artifacts

All result trees live on the user's Drive at `MyDrive/neuro-ai/results/`:

| run | path | n_seeds |
|---|---|---:|
| v3 follow-up | `c3_followup_v3_2026-05-27/` (4 conditions × 10 seeds = 40 per-seed dirs + `_merged_n10/`) | 10 per condition |
| v4 robustness | `c3_robustness_v4_2026-05-27/` (3 probes × 10 seeds = 30 per-seed dirs + `_merged_n10/`) | 10 per probe |
| v5 D-sweep | `c3_v5_2026-05-27/` (4 D × 10 seeds = 40 per-seed dirs + `_merged/`) | 10 per D |
| v6 D=8192 confirmation | `c3_v6_2026-05-27/` (D=8192 × 20 seeds + D=16384 × 10 seeds = 30 per-seed dirs + `_merged/`) | 20 (D=8192) + 10 (D=16384) |

SHA-256s of merged-pool JSONs are written to the corresponding `_merged*/` directories at run time. Per-seed `c3_summary.json` files each have their own header SHA via the C.3 driver. Reproduction path: any of the four Colab notebooks rebuilds the corresponding results dir from scratch from a clean Colab runtime.

## STATUS.md and Path C precommit log updates

Required this session before the report is filed:

1. STATUS.md banner: walk back any active-phase wording that implied imminent Phase 3 graduation; mark Path C as closed inconclusive; mark Phase 3 as not graduated; mark Phase 5′ as remaining paused; mark Path γ design as the new active work.
2. STATUS.md "Recent updates": one-line entry pointing at this report and at the per-month archive section.
3. Path C precommit running log at [notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md](../notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md): closure entry walking through the v3 → v6 chain and the not-graduate decision.

## Done-gate audit (per CLAUDE.md "What done looks like")

1. **Headline metric reported with confidence intervals:** ✅ Wilson CIs throughout, both pooled n=30 tables and per-D rows.
2. **Control condition run on same test set:** ✅ shuffled-token-with-consolidation control on identical wikitext windows and identical substrate atom set, only the codebook row permutation differs (Path α default since 2026-05-26).
3. **Drill-down metrics explain anomalies:** ✅ per-seed paired comparison localizes the substrate-capacity hypothesis to a regression-to-the-mean artifact; D-curve with reversals at small and large D documented; per-seed variance reported.
4. **Result written up as markdown report under `reports/`:** ✅ this file.
5. **Relevant memory / status note updated:** to be landed this session (see STATUS.md and Path C precommit log updates above).
