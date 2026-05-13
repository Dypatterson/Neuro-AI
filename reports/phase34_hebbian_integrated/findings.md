# Phase 3+4 integrated experiment with Hebbian online + pretrained codebook

Re-run of `experiments/19_phase34_integrated.py` with the architectural
fix from `reports/phase34_hebbian/findings.md`: replace
`OnlineCodebookUpdater` with `HebbianOnlineCodebookUpdater`, and
initialize from the pretrained `phase3c_codebook_reconstruction.pt`
codebook instead of `--random-codebook`. This re-establishes the
integrated Phase 3+4 result on architecturally-sanctioned ground.

## Setup

- Substrate: TorchFHRR, D=4096, MPS
- Initial codebook: pretrained `phase3c_codebook_reconstruction.pt`
- Multi-scale: W={2,3,4} with landscapes {4096, 2048, 1024}
- Eval window: W=8, center mask
- β=10, decode_k=10
- 2000 cues, 300-window held-out test set, single seed (17)
- Online updater: `HebbianOnlineCodebookUpdater`, lr=0.01, success_threshold=0.5

## Results

Final state at cues=2000:

| Condition | top-1 | top-K | cap_t_05 | drift | Hebbian succ_rate | Phase 4 candidates |
|---|---:|---:|---:|---:|---:|---:|
| A_baseline (frozen) | 0.129 | 0.320 | 0.280 | 0.0000 | — | — |
| B_phase3 (Hebbian + reencode) | 0.124 | **0.333** | 0.280 | 0.0000 | 0.018 | — |
| C_phase34 (Hebbian + replay + reencode) | 0.124 | 0.320 | **0.289** | 0.0000 | 0.018 | **126** |

### Phase 4 replay candidate trajectory (condition C)

| cues_seen | candidates_total |
|---:|---:|
| 0 | 0 |
| 250 | 124 |
| 500 | 126 |
| 750 | 126 |
| 1000+ | 126 |

The candidate count jumps to 124 by step=250 (the first replay cycle
batch) and stabilizes around 126. This is the burst-of-discovery
pattern: replay finds the durable trajectories in the initial pool,
then settles as the landscape stops producing new unresolved-but-
engaged traces.

## What changed vs the original phase34_medium

The original `reports/phase34_medium/phase34_results.json` ran the same
experiment with `--random-codebook` and the (then unfixed) error-driven
updater. Final state at cues=2000:

| | Original (random + error-driven) | This run (pretrained + Hebbian) |
|---|---:|---:|
| Condition A (baseline) top-1 | 0.018 | **0.129** |
| Condition B top-1 | 0.000 (collapsed) | **0.124** |
| Condition C top-1 | 0.022 (collapsed) | **0.124** |
| Condition C candidates | **2** | **126** |

Three things flipped:
1. Baseline performance: random codebook starts at chance; pretrained
   starts at the validated Phase 3c level.
2. Online learning: error-driven destroyed the codebook; Hebbian
   preserves it.
3. Phase 4 replay: starved on a degrading codebook (2 candidates
   over 2000 cues) → produces meaningful discovery (126 candidates
   over 250 cues) when the substrate is stable.

The third one is the most architecturally significant. Phase 4 replay
was implemented correctly and tested in the wrong regime — when the
underlying codebook was destabilizing under it. The previous "replay
produces only 2 candidates" finding was an artefact of that
destabilization, not a property of the replay mechanism.

## What this run does NOT yet show

- Replay candidates landing as a measurable retrieval-quality improvement
  over the no-replay baseline. cap_t_05 ticks up +0.009 (0.280 → 0.289)
  but that's a single seed and modest. The actual Phase 4 headline
  metric is "before vs after N consolidation cycles, with active
  codebook drift" — this run has zero drift, so it doesn't yet test
  whether replay rescues retrieval against drift.

- Multi-seed validation. Single seed (17) only here.

## Next experiment

The Phase 4 unified design's headline metric requires drift > 0.
`experiments/18_phase4_unified_experiment.py` has `--drift-magnitude`
(default 0.15, but the run that produced `reports/phase4_unified_b10/`
set it to 0.0 in practice). Re-running exp18 with drift > 0 is the
actual Phase 4 headline test.

With the codebook now stable under streaming online updates (Hebbian,
not error-driven), the Phase 4 headline experiment can proceed.
