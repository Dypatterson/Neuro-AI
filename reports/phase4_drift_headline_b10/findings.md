# Phase 4 headline-with-drift — initial results

The Phase 4 unified design (`notes/emergent-codebook/phase-4-unified-design.md`)
specifies its headline metric as *"Recall@K + cap-coverage on
masked-token contextual completion, measured before vs. after N
consolidation cycles, with active codebook drift between cycles."*
The prior phase4_unified_b10 run set `drift_magnitude=0.0`, so the
headline was never tested under its own design conditions.

This is the first run with drift > 0.

## Setup

- Substrate: TorchFHRR, D=4096, MPS
- Codebook: pretrained `phase3c_codebook_reconstruction.pt`
- Multi-scale: W={2,3,4} with landscapes {4096, 2048, 1024}
- Eval window: W=8, center mask
- **drift_magnitude = 0.15** (applied to codebook between checkpoints)
- 2000 cues, checkpoints every 500
- Test set: 400 held-out windows
- Single seed (17)

Two β configurations tested, since the Phase 4 design and HAM
aggregation use different β regimes:

- `phase4_drift_headline/`: β=30, resolve_threshold=0.85 (HAM regime)
- `phase4_drift_headline_b10/`: β=10, resolve_threshold=0.7 (Phase 4
  design regime, matches the prior phase4_unified_b10)

## Results

### β=30, rt=0.85 (HAM regime)

| Condition | step | top-1 | top-K | cap_t05 | candidates |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0.146 | 0.323 | 0.299 | — |
| baseline | 2000 | 0.133 | 0.320 | 0.296 | — |
| phase4 | 0 | 0.146 | 0.323 | 0.299 | 0 |
| phase4 | 2000 | 0.133 | 0.320 | 0.296 | **0** |

**Phase 4 produces zero candidates** at β=30. The engagement gate
(entropy × (1 − resolution)) does not fire because high-β retrieval
resolves traces too fast — every trace settles into a clean basin with
low entropy and high resolution, so the gate signal stays below the
0.05 store threshold. The Phase 4 condition is mechanically equivalent
to the baseline.

### β=10, rt=0.7 (Phase 4 design regime)

| Condition | step | top-1 | top-K | cap_t05 | candidates |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0.122 | 0.310 | 0.276 | — |
| baseline | 2000 | 0.122 | 0.323 | 0.262 | — |
| phase4 | 0 | 0.122 | 0.310 | 0.276 | 0 |
| phase4 | 500 | 0.122 | 0.310 | 0.255 | 33 |
| phase4 | 2000 | 0.122 | 0.306 | 0.255 | **33** |

Drift impact on baseline (β=10): top-1 unchanged, cap_t05 −0.014.

**Phase 4 vs baseline at step=2000:** Δtop1=0, Δtopk=−0.017, Δcap_t05=−0.007.

Phase 4 produces 33 candidates (all discovered in the first replay
cycle), but they do not translate to retrieval improvement. In fact
Phase 4 underperforms the baseline by small margins on topk and
cap_t05.

## Interpretation

Two regime issues sit upstream of the headline metric.

### Issue 1 — β controls the engagement gate

The gate signal `engagement × (1 − resolution)` is designed to fire on
traces with *high entropy across settling* (engagement) and *low final
resolution* (1 − resolution). At β=30 these are anticorrelated by
construction — sharp retrieval drops entropy fast and locks into a
basin. So β=30 makes the gate effectively unreachable.

The Phase 4 design implicitly assumes the low-β feature-mode regime.
This wasn't called out explicitly in `phase-4-unified-design.md`. Worth
documenting as a design constraint.

### Issue 2 — 33 candidates among 4129 patterns is too sparse

The replay-and-discover channel ran once in the first 500 cues, found
33 high-resolution trajectories, and stored them as new patterns.
Subsequent replays produced zero new candidates (the store_threshold-
passing traces had already been drained). Stored as 33 of 4129 total
patterns, the discovered candidates contribute roughly 33/4129 ≈ 0.8%
of the retrieval landscape — too sparse to measurably move the
aggregate metrics.

For replay to *rescue* retrieval under drift, one of these likely needs
to be true:

- **The Benna-Fusi pattern-death mechanism actually kills stale
  patterns**, so replays' new patterns replace rather than augment.
  Currently `death_threshold=0.005`, `death_window=10000` — the window
  is longer than the experiment's 2000 cues, so no patterns ever die
  during the run. The strength-decay observable (meanU 0.133 → 0.026)
  shows patterns approaching the death threshold but the window
  prevents pruning.
- **More aggressive drift** (e.g., drift_magnitude=0.3) that breaks
  retrieval enough to make replay's contribution proportionally larger.
- **Lower store_threshold + lower resolve_threshold** so many more
  candidates accumulate. Risk: noise candidates dilute rather than
  rescue.
- **Re-encode** stored patterns through the current codebook (this
  exists in the Phase 4 code via `reencode_patterns` but exp18 does
  not call it — it only adds new patterns from replay).

## What this run resolves

- **Phase 4 replay is wired up correctly.** It fires when β is in its
  intended regime, produces meaningful discovery activity (33
  candidates in the first replay cycle), and decays consolidation
  strength on un-reinforced patterns.

- **Phase 4 replay does not yet rescue retrieval under drift at default
  settings.** The mechanism is mechanically alive but headline-inert.

- **The β regime split is real.** β=30 disables Phase 4 by design;
  β=10 enables it. The project has been quietly running two different
  β regimes (HAM at 30, Phase 4 at 10) without explicit documentation.

## What this leaves open

- The headline metric ("Recall@K + cap-coverage before vs. after N
  cycles with active drift") is not yet positive for Phase 4. Without
  evidence that replay rescues retrieval, Phase 4 has not crossed its
  viability threshold.

- The pattern-death window of 10000 means stale patterns can't be
  pruned within a 2000-cue experiment. The design probably needs
  shorter death_window OR active re-encode to demonstrate rescue.

## Recommended next steps

In rough priority order:

1. **Re-run with shorter `death_window` (e.g., 1000)** so stale patterns
   actually die during the experiment. This would test whether Phase 4's
   *pruning* arm contributes to rescue.

2. **Re-run with `reencode_every=100`** (the mechanism exists in
   `phase34/reencoding.py`, just not invoked by exp18). This is the
   most direct path to "stored patterns track the drifted codebook" —
   no replay-discovery required.

3. **Sweep drift_magnitude** (0.10, 0.15, 0.20, 0.30) to find the
   regime where replay's contribution becomes measurable.

4. **Multi-seed validation** (5 seeds) once a configuration shows
   positive Δ vs baseline.

5. **Document the β regime split** in `phase-4-unified-design.md` and
   `notes/emergent-codebook/experimental-progression.md`. The current
   docs imply a unified β; the empirical reality is that Phase 4
   replay needs β=10 and HAM aggregation needs β=30.

The Phase 4 headline metric question — *does replay rescue retrieval
under drift?* — remains open. The path forward is parameter tuning
(specifically `death_window` and `reencode_every`), not a fundamental
redesign.
