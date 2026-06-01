# HANDOFF — 2026-06-01 (end of night; fresh-session orientation)

**Branch:** `consolidation/role-structure` · **HEAD:** `dc642a0`
**One line:** The Phase-3 second-order growth re-scope ran end-to-end → **NULL** at every
stage that matters, but each null was *earned* (signal confirmed to exist; two metric traps
caught before banking) and the failures converged on one decisive finding → the genuine next
move is **R3 (a predictive/successor LOCAL growth)**, un-built. Clean stopping point.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — the charter (the bet, the gates, the invariants).
2. **[STATUS.md](STATUS.md)** — the bookmark (Active deliverable + Recent updates).
3. This file.
4. The load-bearing reports for the next move: **[Report 123](reports/123_growth_redesign_sweep_null/report.md)**
   (the sweep NULL + the subdominant-modes finding + the R3 disposition) and
   **[Report 122](reports/122_phase3_second_order_growth_oracles/report.md)** (the oracle
   reopen + the pressure-test that killed the basin-read rescue). Plus the design spec:
   **[precommit §GR](notes/emergent-codebook/phase-3-second-order-growth-precommit.md)**.

## 1. Where we are RIGHT NOW
Report 121 (first-order centroid) was a paradigmatic NULL → re-scope to a second-order
growth. This session built and ran that, exhaustively:
- **The signal EXISTS** — the §10 SVD-of-SPPMI oracle (`experiments/62`) shows WikiText-2's
  SPPMI carries paradigmatic structure (king/queen 0.222; specificity +0.109). So no Option-B
  capitulation: the corpus has the signal.
- **But the LOCAL growth can't write it.** The B′ (row-centered SPPMI) + energy-native
  `H_anti` anti-collapse sweep (`experiments/61`, D=4096, gauge-free gate) is a NULL — 0/60
  cells pass. H_anti prevents collapse, but the surviving clustering is **collocational, not
  paradigmatic** (Report 123).
- **THE DECISIVE FINDING:** paradigmatic ("substitutability") structure lives in the
  **subdominant modes** of SPPMI — the *global* SVD isolates them; *local* iterative
  pull+repulsion (which runs to the dominant collocational modes) **cannot reach them**.
  Local best +0.021 clean vs global +0.109. A real local-vs-global bound.

## 2. The genuine next move — R3 (un-built)
**A predictive / successor-context LOCAL growth** — model *which contexts follow from a token*
(a slowly-drifting FHRR context vector / successor representation), richer than the symmetric
co-occurrence centroid; it may promote the subdominant substitutability axis locally. Reuse the
graduated heteroassoc write (055–058) + FHRR bind/bundle/permute. **Needs its own grounding +
pre-commit** (don't build from first principles — the codebook-framing + anti-smush workflows
this session carded much of the relevant literature; check `docs/ground-truth/` + Reports 017/018
(error-driven/reconstruction = already-killed first-order) before treating a mechanism as new).
- **Honest contingency:** if R3 also can't reach the subdominant structure → paradigmatic
  structure needs a **latent/hierarchical layer** (PCN/SFA grounding said this) — bigger, but
  still local. NOT more `S'@G` knobs — that shape is bounded.

## 3. Invariants the user holds (do not violate)
- **Local growth is NON-NEGOTIABLE.** No global SVD/PCA/word2vec shortcut as the *mechanism*
  (global computations are diagnostics/flashlights only — the SVD oracle is used that way).
  "If we're going to do something, we're going to do it right." [[memory: do-it-right]]
- **Anti-homunculus** (local geometric dynamic, never an `if-metric-then` arbiter);
  **batch-offline only** (sleep/wake); **FHRR-native**; **floor (055–058) untouched**.
- **Anti-rationalization:** two metric traps were caught this session before banking a verdict
  (the stream-shuffle gauge LEAKS 0.79 for 2nd-order operators → retired; the gauge-free
  headline double-subtracted contraction → fixed `dc642a0`). Stay this suspicious of any metric.

## 4. Banked side-findings (don't relitigate)
- High static cosine ≠ merged Hopfield basin at β=30 (β-decoupling; Report 122). The
  basin-read reframe (R1) was pressure-tested and KILLED as a rescue — do not revive it.
- The gauge-free para-vs-random gate is the right paradigmatic test (the stream-shuffle gauge
  is invalid for SPPMI). `corr(log cooc, drift) < 0.15` is the collocational-vs-paradigmatic
  discriminator.

## 5. Artifacts
- Reports **121/122/123**; precommit **§GR**; `experiments/61` (gauge-free gate + force-normalized
  H_anti, both fixed), `experiments/62` (oracles, CUDA-fixed); notebooks **061** (first headline
  run) / **062** (the D=4096 sweep). Sweep JSON on Drive (`_sweep_all.json`, `_oracle_d4096.json`).
- Nothing is mid-run; tree is committed + pushed. The 3 pre-existing untracked artifacts
  (`brainstorm-workspace/.../_wf1,2_raw.json`, `reports/gate0_2026-05-28/`) are pre-existing — leave them.
