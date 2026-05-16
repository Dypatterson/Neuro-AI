---
date: 2026-05-15
project: neuro-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
  - phase-4
  - blocker/death-scope
  - blocker/top1-regression
---

# Saighi self-inhibition, FHRR unitarity, and the replay literature against current Phase 4 blockers

## Entry point

Same-day follow-up to [report 033](../../reports/033_phase4_death_mechanism_diagnostic.md), which showed that the Phase 4 "pattern death" mechanism cannot fire in n_cues=1500 because (1) patterns reach equilibrium too late and (2) >99.9% of vocab atoms are never reinforced via top-1 retrieval — so naively closing the temporal gap would mass-kill the initial codebook rather than purge stale discovered patterns. The morning's framing collapsed the question into "uniform-vs-discovered-only death scope." Four newly-added papers were reviewed against that question and the parallel top1 regression in [blocker #6'](../../STATUS.md).

The point of this note is *not* to catalogue the four papers. Two of them produce a mechanism-level reframe and a cheap audit; the other two are background or low-leverage. The load-bearing output is one mechanism substitution and one verification step.

## Papers in scope

| Paper | Role |
|---|---|
| Saighi & Rozenberg (2025), *Autonomous retrieval for continuous learning* | **Mechanism substitution for blocker #2.** Per-attractor self-inhibition replaces threshold-based death. |
| Ganesan et al. (2021), *Learning with Holographic Reduced Representations* | **Cheap audit for blocker #6'.** Unit-magnitude FFT projection as a stability requirement for HRR family substrates. |
| Hayes/Kanan et al. (2021), *Replay in deep learning* | Biological precedent backing the Saighi reframe. No new mechanism. |
| Sun et al. (2022), *Information-theoretic online memory selection (InfoRS)* | LOW relevance. Supervised replay-buffer triage; doesn't fit the discovery-channel shape. Defer. |

## The Saighi reframe of blocker #2

The morning's framing offered three options for the death scope decision: (a) scope death to discovered patterns only, (b) accept uniform death as a corpus-statistics phenomenon, (c) extend the run and report whatever fires. (a) is an arbitration over which population is subject to a binary mechanism. (b) accepts mass-death of unreached vocab as the design's actual behavior. (c) is the experimental-evidence-without-design-clarity path.

Saighi's mechanism is a *different shape entirely*. Instead of a global threshold + window driving binary removal, each stored attractor `k` carries a scalar **inhibition `A_k`** that accumulates with each successful retrieval of that attractor:

```
A_k ← A_k + β · v_k(t_f)         # after settling converges to attractor k
score_k(query) -= A_k             # subtracted from energy/score during settling
```

The dynamic is local, geometric, monotonic in use, and produces basin shrinkage *proportional to actual visitation*. Three consequences map exactly onto our blockers:

1. **Untouched vocab atoms never accumulate inhibition** → they are never "killed." The mass-death problem from report 033 dissolves because the mechanism is per-pattern and use-driven rather than threshold-and-window driven.
2. **Repeatedly-visited stale discovered patterns lose dominance smoothly** → they are effectively retired without binary removal. This is what `pattern_death` was *intended* to do; the threshold mechanism was the wrong shape.
3. **No global hyperparameter that says "what counts as dead."** β is a rate (per-retrieval inhibition magnitude), tuned to run length — Saighi p. 4: smaller β for longer runs, larger β for faster turnover but more spurious attractors. This is a usable trade-off curve, not the tuning hell of `death_threshold × death_window`.

Anti-homunculus filter: passes cleanly. `A_k` accumulation is a Vogels-2011 / spike-frequency-adaptation-style local plasticity rule. No supervisor, no `if X then Y`. Compare to `--reencode-discovered` (report 030) which made an `if discovered then refresh` decision at a wrong-shaped location and eroded ΔR@10.

### The biased / free phase split

Saighi pairs `A_k` accumulation with a settling protocol that is also the right shape for the discovery-channel refresh problem. Their "biased phase + free phase" decomposition:

```
biased:  settle with A_k(known stale) ≠ 0    # inhibition pushes trajectory off stale attractors
free:    release A_k → 0, settle freely      # trajectory finds the right basin under current codebook
```

This is the "refresh without arbitration" primitive that the `--reencode-discovered` rfix variant failed to be. The rfix re-encoded discovered queries through the current codebook, which pulled them toward existing attractors and destroyed the geometric property the discovery channel was meant to capture. Saighi's protocol does the opposite: it lets the network find the *current* basin a stale pattern *should* live in by repelling from the old one.

This is a Phase 4 architectural substitution. It should be prototyped against report 026 and report 029 configs and evaluated on the same headlines + drill-downs.

## The FHRR unitarity audit (Ganesan)

Ganesan et al. show that naive HRR binding/unbinding numerically degrades past ~10 bound terms — query magnitudes drift far outside [0,1] and variance grows with binding depth (their Fig. 1, p. 4). The fix is a **unit-magnitude FFT projection**:

```
π(x) = F⁻¹( F(x) / |F(x)| )
```

After this projection, `|F(x)_j| = 1` for every frequency bin `j`, which makes the inverse and pseudo-inverse mathematically equivalent and scales binding capacity to thousands of terms. They show that without it, variance compounds with each binding operation.

**Relevance to blocker #6'.** The top1 regression appears in both Phase 3-only and Phase 3+4 conditions under online Hebbian drift (report 030's reframe). The current hypothesis is "drift spreads the codebook in a way that improves the right neighborhood but moves rank-1 around." An alternative hypothesis, smaller and cheaper to test, is: the Hebbian update rule moves codebook vectors off the unit-magnitude FFT manifold. Retrieval scores then have *drifting meaning* over the course of a run — what `score = 0.7` represents at step 200 vs step 1500 is no longer the same quantity. This could produce a slow top1 degradation that looks like a Hebbian-reshaping property but is actually a binding-stability artifact.

The audit is one read of [src/energy_memory/substrate/torch_fhrr.py](../../src/energy_memory/substrate/torch_fhrr.py) and the Hebbian update path. Three states are possible:

1. **Already correct** — init and Hebbian-update both preserve `|F(c_k)_j|=1`. The audit closes; the top1 regression is a real Hebbian-reshaping property. Move on to mechanism-level investigation.
2. **Init correct, Hebbian breaks it** — retrofit a projection step after each update. Re-run report-030 condition B with drift on. If top1 regression softens or vanishes, blocker #6' is partly (or fully) explained.
3. **Init also incorrect** — bigger problem. All prior FHRR results inherit the same artifact, but it's been consistent across all of them so relative comparisons mostly hold.

### Audit result (2026-05-15, same session)

**Outcome (1): already correct end-to-end.** Empirical verification on the frozen phase3c init codebook + all 10 found `.pt` checkpoints (phase3a/3b/3c, all updater kinds — random, hebbian, error_driven, reconstruction) shows 100% of elements within `|v|∈[0.999, 1.001]`. A synthetic stress test of 1000 Hebbian updates with `lr_hebbian=0.05` keeps max deviation at 2.4e-7 across 8.4M elements (floating-point noise).

The substrate's `random_vector` uses `torch.polar(ones, phase)` so init is unit-magnitude by construction (FHRR is the frequency-domain representation, so this *is* the Ganesan projection at init). All four codebook-mutation paths — [HebbianOnlineCodebookUpdater](../../src/energy_memory/phase34/hebbian_online.py:103), [OnlineCodebookUpdater](../../src/energy_memory/phase34/online_codebook.py:127) pull/push, [StableOnlineCodebookUpdater](../../src/energy_memory/phase34/stable_online_codebook.py:131) repulsion — terminate in `self.substrate.normalize(...)` which is per-element `v / |v|`, exactly the Ganesan projection.

**Implication for blocker #6'.** The top1 regression under online Hebbian drift is *not* an FHRR numerical artifact. It is a real Hebbian-reshaping property. The Ganesan alternative-hypothesis branch closes; investigation focus stays on the mechanism itself.

Worth noting Ganesan's Table 5 (p. 9): "updating concept vectors during training had no benefit; sometimes hurt." Their task is supervised XML, not contextual completion, so it isn't directly transferable. But it is an uncomfortable circumstantial alignment with what blocker #6' is showing under online Hebbian.

## Hayes/Kanan as biological precedent

Two relevant points from the review:

- "ANN replay buffers don't purge their memory, which is inconsistent with biology" (Hayes/Kanan p. 23-24). Hippocampal replay strength *decays over hours* (Nádasdy 1999; Karlsson & Frank 2009). Direct biological precedent for *gradual decay of replay weight* rather than a hard threshold cutoff. Aligns with Saighi.
- "Old experiences that overlap with new memories are in the most danger of being damaged by new learning and are preferentially replayed" (citing McClelland et al. 2020). The principled answer to blocker #2's "which patterns should be subject to forgetting?" is: those *not* being damaged by current learning. Untouched vocab atoms (99.9% of the codebook in our setting) are by definition *not* being damaged, so they shouldn't be the things that die. Discovered patterns superseded by later ones *are*.

Both points back the Saighi mechanism rather than introduce a new one. Worth filing as a citation when the next Phase 4 report is written.

## What this does and does not commit to

**Commits to (next-session actions):**

1. Audit [src/energy_memory/substrate/torch_fhrr.py](../../src/energy_memory/substrate/torch_fhrr.py) for the unit-magnitude FFT projection invariant. Three outcomes are bounded above.
2. Prototype `A_k` self-inhibition as a drop-in replacement for the death threshold in [src/energy_memory/phase4/consolidation.py](../../src/energy_memory/phase4/consolidation.py) and [src/energy_memory/phase4/replay_loop.py](../../src/energy_memory/phase4/replay_loop.py). Re-run report-026 and report-029 configs; check whether the mechanism organically retires stale patterns *and* whether top1 regression softens because dominance leakage is graceful.

**Does not commit to:**

- A redesign of Phase 3 or the Hebbian update rule. Both are upstream of the death mechanism and the FHRR audit; do the cheap moves first.
- Building anything from InfoRS. Wrong shape for our discovery channel.
- A new design document. The Saighi substitution slots into the existing [phase-4-unified-design.md](../emergent-codebook/phase-4-unified-design.md) §4 ("pattern death (strength < threshold for window)") — that section becomes "per-pattern self-inhibition under settling" and the rest of the design is unaffected.

## Status implications

- Blocker #2 will be reshaped *again*: from "decide uniform-vs-discovered-only death scope" to "implement and verify A_k self-inhibition as the right-shape mechanism, deprecating the threshold-based death." The previous reshape (in report 033) is preserved as the diagnostic step that ruled out the threshold form.
- Blocker #6' gains a candidate cheap explanation: FHRR unitarity audit. Outcome of that audit will either close part of #6' or focus the investigation on Hebbian-reshaping itself.
- No change to blockers #3 (cap-coverage), #4 (diagnostic-actuator session), #5 (seed 23), #7 (Phase 3 codebook data integrity).
