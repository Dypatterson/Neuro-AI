# Stage 1 smoke — PRE-COMMITTED read protocol

**Date:** 2026-05-29 · written BEFORE results, so interpretation is fixed.
Source run: `colab_frameb_smoke_stage1.ipynb` → `variance_decomp.json`
(+ optional `--variance-report` on the n=10 Gate-0 summary).

## The one question Stage 1 answers
Does within-seed pairing actually cancel per-seed codebook luck? (If not, the
slope reframe buys ~0 power and we should NOT reframe — Q1 conditional fires.)

## Pre-committed decision rule (apply verbatim, no rationalizing)

Read TWO numbers:
- **D = dominant_source** (atom-draw / window-draw / binomial)
- **C = corr_AC_BD** from Stage 1B if available; else use var_fraction_atom as
  a weaker proxy (high atom fraction ⇒ luck is structural, says nothing alone
  about cancellation — corr is the real test, so prefer 1B).

| condition | verdict | action |
|---|---|---|
| D=atom-draw AND C ≤ ~0.3 (or 1B unavailable) | **pairing won't cancel luck** | do NOT build Stage 2 yet; report to user that the reframe likely buys little power; revisit Frame A / CUPED |
| D=atom-draw AND C > ~0.3 | **pairing cancels seed variance** | reframe promising → proceed to Stage 2 (checkpoint hook) for the slope-level gates G2/G3/G4 |
| D=window-draw | luck is corpus-window, not codebook | cheaper fix = average K window draws; slope reframe not the lever → flag to user |
| D=binomial | op point too small to show structure | raise n_test, re-run Stage 1 (not a real read) |

## Hard guardrails (anti-improvisation)
- Stage 1 decomposes the ENDPOINT lift A−C. It can FALSIFY the reframe but
  CANNOT confirm it. A "proceed" here only buys the right to run Stage 2; it is
  NOT a graduation signal and NOT a substitute for G1–G4.
- corr threshold 0.3 is the doc's own `_variance_diagnosis` cutoff (gate0:652
  "if c > 0.3") — reused for consistency, not invented here.
- If results contradict the STATUS-2028-05-28 "σ structural" claim (e.g.
  D=binomial at the real op point), that is itself a finding to surface, not
  smooth over.
- Report numbers as returned; quote the JSON fields, do not paraphrase.
