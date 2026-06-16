# Report 143 — Bet B / SCAN Stage 1 iterate 3: TWO-SIDED factorization → consolidation nearly solves it (strong but unstable)

**Status:** strong **near-graduation**, **unstable** (2026-06-15). Best seed 0.86; mean below the bar due to
high seed variance. **Experiment:** `experiments/90_betb_scan_factored2.py` (+ diagnostic chain).
**Builds on:** [Report 142](../142_betb_scan_factored/report.md) (input-only factorization, partial positive).

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B discriminating regime — the thesis graduation test.
- **Headline:** add-`jump` exact-match, consolidation − baseline, n=5. Bar = Δ CI-disjoint > 0 AND absolute ≥ 0.30.
- **Why now:** Report 142 was capped at a leaky premise ceiling (0.26). A diagnostic chain localized the leak
  and motivated the two-sided fix tested here.

## The diagnostic chain that motivated this (the leak was NOT where we guessed)

1. **Oracle filler** (force `ctx_fill = fill_jump` everywhere): 0.185 → 0.183 — **NOT the filler channel.**
2. **Canonical decoder feedback** (feed a single "a verb was emitted" token instead of the specific `I_JUMP`):
   single-verb premise **0.18 → 1.0000.** **The leak is the decoder's identity-FEEDBACK** — `I_JUMP` never
   appeared mid-sequence in training (jump was only the standalone single-token output), so emitting it
   knocks the autoregressive recurrence off-distribution.

## The fix + result

**Two-sided factorization:** the decoder now recurs on the **role-class** of its previous output (all
verb-actions → one canonical VERB class; turns/EOS keep identity); identity is read out by the head but does
**not** re-enter the recurrence. Input side unchanged from exp89.

| | baseline | + consolidation | Δ |
|---|---|---|---|
| mean (n=5) | 0.087 | 0.295 | **+0.208** |
| per-seed | 0.298 / 0.000 / 0.000 / 0.138 / 0.001 | **0.864** / 0.126 / 0.006 / 0.480 / 0.000 | +0.566, +0.126, +0.006, +0.342, −0.001 |

## Reading (honest)

- **The mechanism nearly SOLVES the discriminating regime.** Seed 0: consolidation lifts the jump split to
  **0.864** (vs the vanilla model's ≈0 and exp89's best 0.39). The two-sided factorization raised the ceiling
  from 0.26 (Report 142) toward 1.0, exactly as the diagnostic predicted, and consolidation is strongly
  **load-bearing** (mean Δ +0.21, ~3× exp89). This is the **strongest evidence in the program** for the Bet-B
  thesis: a restructuring consolidation manufactures composition the simple method (vanilla, Report 141) cannot.
- **But it is UNSTABLE — not yet a clean graduation.** 3/5 seeds lift big (+0.57, +0.34, +0.13); 2/5 are dead
  (baseline exactly 0.0 → consolidation ≈0; one seed consolidation even hurts −0.001). The mean (0.295) sits
  right at the 0.30 bar, driven by spectacular + dead seeds. The dead seeds have baseline *exactly* 0.0
  (systematic, not noise) — the base training sometimes lands in a bad factorization the consolidation can't
  rescue. **This is an optimization/stability problem, not a wall.**
- **Redundancy-guard note:** the factored architecture alone (baseline mean 0.087) does NOT solve it — known
  architectural SCAN fixes reach 0.8+ alone. The lift is the consolidation's. The claim remains narrow: the
  *consolidation mechanism* is load-bearing within the factorization.

## Disposition

A **stabilization run (8 seeds, 50 epochs, 1500 consol steps)** is in flight to test whether more training
reduces the dead-seed rate and tips this to a clean graduation (mean ≥ 0.30, Δ CI-disjoint). If it stabilizes →
graduation, then the mandatory follow-ups: the GECA redundancy guard, the second-split generality check
(BUILD CONDITION 2 — byte-identical predicate on a different held-out primitive), and ≥8-seed CIs. If it stays
unstable → characterize/fix the dead-seed factorization (init, regularization, or a cleaner role-space).

## Discipline notes

- 5 seeds; the diagnostic probes n=1 but the canonical-feedback result (0.18→1.0) is deterministic and decisive.
- Anti-homunculus: peer set computed-in-code; the role-class map is a fixed content-blind grouping of OUTPUT
  tokens by type (verbs-as-a-class), no per-token metric branch. FENCE: iterative gradient. (Pending a fresh
  anti-homunculus pass on the output-side role-class grouping before any graduation claim is banked.)
