# Phase 3 codebook-comparison data integrity — investigation brief

**Date opened:** 2026-05-15
**Status:** **CLOSED 2026-05-16** by [report 039](../../reports/039_phase3_codebook_comparison_integrity.md). Disposition: labeling bug in `31_phase3_comparison.py`, not a data integrity issue. Phase 4 graduation unaffected (`phase3c_codebook_reconstruction.pt` is genuinely unique).
**Severity:** HIGH (per [audit-report-2026-05-14.md §Appendix A](../../audit-report-2026-05-14.md)) — **downgraded to documentation cleanup** after investigation.
**Why this note exists:** The 2026-05-14 repo audit flagged a suspicious
result in the Phase 3 codebook comparison runs. Captured here so the
investigation can be picked up cold in a future session.

---

## The finding

Per [audit-report-2026-05-14.md §2.3](../../audit-report-2026-05-14.md):

> In both `phase3_comparison.json` and `phase3_comparison_6seed.json`,
> two pairs of codebooks produce byte-identical per-seed results:
> (random_baseline_matrix, random_phase3c) and
> (hebbian_phase3c, hebbian_phase3b) and (reconstruction, error_driven).
> Each pair produces the same `total_correct` count and the same per-seed
> `n_eval`. This persists in the 6-seed repeat. The most likely cause is a
> shared random state or codebook loading that accidentally reuses the
> same artifact for both labels.

Appendix A labels this HIGH severity and notes it was "mentioned as
suspicious in report 017 but not resolved."

## Why this is potentially load-bearing

Phase 3c was adopted as the canonical codebook for Phase 4 partly on the
strength of the phase3_comparison results. If the comparison was
silently reusing the same codebook tensor for what were supposed to be
different conditions, then:

- The "Phase 3c is the winner" claim may rest on an artifact, not a
  genuine comparison.
- Any downstream Phase 4 result that interprets effect sizes against
  "the Phase 3c codebook is the right one" inherits that uncertainty.
- The random-codebook control used in Phase 4 verification (report 026)
  is independent of this bug (different code path), so that control is
  not affected. But the *choice* to use Phase 3c is.

## Where the evidence lives

| Artifact | Path |
| --- | --- |
| Comparison run (3-seed) | [reports/phase3_comparison.json](../../reports/phase3_comparison.json), [.csv](../../reports/phase3_comparison.csv) |
| Comparison run (6-seed verification) | [reports/phase3_comparison_6seed.json](../../reports/phase3_comparison_6seed.json), [.csv](../../reports/phase3_comparison_6seed.csv) |
| Producing script | [experiments/31_phase3_comparison.py](../../experiments/31_phase3_comparison.py) |
| Earlier mention | [reports/017_*.md](../../reports/) — flagged but not resolved |

## Starting hypothesis

Most likely: a shared `torch.Generator` state or a cached codebook
tensor is reused between conditions inside `31_phase3_comparison.py`.
The 6-seed repeat reproducing the same pairs argues against a fluky
filesystem race or one-off bug — it's deterministic.

Less likely but worth ruling out: different label strings pointing at
the same loaded `.pt` file (e.g., a path-construction bug where
`hebbian_phase3c` and `hebbian_phase3b` resolve to the same artifact).

## First moves for the investigation session

1. **Reproduce locally on one pair.** Run `31_phase3_comparison.py` with
   just the suspect pair (e.g., `reconstruction` and `error_driven`) and
   confirm the byte-identical-results behavior reappears on a fresh run.
2. **Diff the codebook tensors that go into each condition.** Add a
   line that hashes (or `torch.equal`-checks) each codebook tensor right
   before it's used for evaluation. If two distinct labels show the
   same hash, the bug is upstream of evaluation.
3. **Check the RNG handling.** Per CLAUDE.md "Common failure modes":
   shared random state between conditions is a known landmine in this
   project. Verify the script captures and restores RNG state, or uses
   independent generators per condition.
4. **Read report 017** for what was already noticed and any partial
   diagnosis.

## What "done" looks like

- The mechanism producing identical results is identified (RNG, path
  reuse, or other).
- A clean rerun produces non-identical results for the affected pairs
  (or, if the pairs are genuinely equivalent under the comparison
  protocol, that's stated explicitly with rationale).
- If the Phase 3c adoption decision is affected, the implications are
  written up — either a confirmation that 3c remains the right choice
  on independent grounds, or a re-evaluation.
- STATUS.md blocker entry (see below) is closed with the new report
  cited.

## Disposition in STATUS.md

Added as blocker #7 in [STATUS.md](../../STATUS.md). Tag any session
that touches this with the report number when it lands.
