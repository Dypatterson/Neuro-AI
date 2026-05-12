---
date: 2026-05-06
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Phase 2 Specification (Draft) — Hopfield Retrieval Baseline

This is the scope-of-work document for Phase 2 of the personal-ai project. Read `CLAUDE.md` for architectural context and `PHASE_1_SPEC.md` for what Phase 1 delivered.

## Goal

Measure whether Hopfield retrieval over FHRR-encoded sequences can recover masked or predicted tokens, and determine whether masked-token (contextual completion) outperforms next-token (sequential prediction) as a retrieval objective. The codebook is static and random — no learning occurs. Phase 2 is a measurement phase that produces empirical evidence for Phase 3 design decisions.

## Why this matters

Modern Hopfield networks are associative memories, not transition models. Masked-token prediction is pattern completion (fill in a gap using bidirectional context); next-token prediction is sequential generalization (predict what comes after using only leftward context). Comparing the two objectives on the same substrate is a critical architectural test: if masked-token consistently outperforms next-token, it confirms the contextual-completion framing committed on 2026-05-04 and tells Phase 3 which objective to optimize its refinement mechanism around.

## What "done" looks like

1. A Hopfield retrieval module that can store encoded sequences and retrieve against them.
2. A data pipeline that loads WikiText-2, tokenizes at word level with a frequency cutoff, and generates test cases across the full experimental matrix.
3. A complete experimental run across all conditions in the matrix.
4. A markdown report (`experiments/02_phase2_retrieval_baseline.md`) with:
   - Accuracy distributions per condition (not single numbers — distributions with confidence intervals).
   - Trivial baseline comparisons.
   - Analysis by token frequency.
   - Diagnostic plots.
   - A "Findings" section summarizing the pattern across conditions and its implications for Phase 3.

## Project layout (new files — Phase 1 files unchanged)

```
personal-ai/
├── PHASE_2_SPEC.md                          # This spec (copied to repo for build)
├── src/
│   └── pai/
│       ├── hopfield/
│       │   ├── __init__.py
│       │   └── retrieval.py                 # Modern continuous Hopfield retrieval
│       └── data/
│           ├── __init__.py
│           └── wikitext.py                  # WikiText-2 loading, word-level tokenization
├── tests/
│   ├── test_hopfield.py                     # Hopfield module unit tests
│   └── test_data_pipeline.py                # Data pipeline tests
└── experiments/
    ├── 02_phase2_retrieval_baseline.py       # Full experimental run
    └── 02_phase2_retrieval_baseline.md       # Generated report
```

## Dependencies (additions to Phase 1)

- `datasets>=2.0` — HuggingFace datasets library for loading WikiText-2
- `scipy>=1.10` — For confidence interval computation

Do not add other dependencies without asking. Notably: do not add tokenizers libraries (we are doing word-level tokenization, not BPE). Do not add scikit-learn or other ML frameworks.

## Dataset and Tokenization

### Corpus

WikiText-2, loaded via HuggingFace `datasets`. Use the standard train/validation/test splits as provided.

### Tokenization

Word-level with a frequency cutoff. Implementation:

1. Scan the training split. Count word frequencies (lowercase, strip punctuation).
2. Keep the top N most frequent words as the vocabulary (N to be determined during implementation — target range 5,000–10,000). All other words map to a single `<UNK>` token.
3. Add a `<MASK>` token to the vocabulary for the masked-token objective.
4. Record the final vocabulary size. This determines codebook size.

The tokenizer is a simple lookup table — no subword decomposition, no learned merges. Store the vocabulary mapping for reproducibility.

### Why word-level

The codebook is static and random — atoms carry no learned semantic content. BPE's advantages (handling rare words, subword compositionality) don't do work when atoms are random. Word-level gives shorter sequences (better SNR in bundles per Phase 1 Test 3's scaling exponent of −0.505), more interpretable retrieval results, and simpler implementation. Tokenization granularity will be revisited for Phase 3 when the codebook starts growing.

## Codebook Initialization

One random unit-modulus FHRR vector per vocabulary entry (including `<UNK>` and `<MASK>`), at D=4096. Generated via `substrate.random_vector(vocab_size)`. Store the codebook tensor for reproducibility across experimental runs.

The codebook is **static** for the entirety of Phase 2. No updates, no learning, no refinement. It exists to assign a consistent HD representation to each word so that sequence encoding and retrieval can be tested.

## Hopfield Retrieval Module (`hopfield/retrieval.py`)

Implement the modern continuous Hopfield update rule (Ramsauer et al. 2020):

```
ξ_new = X @ softmax(β * X^T @ ξ)
```

Where:
- `X` is the matrix of stored patterns (each column is an encoded sequence)
- `ξ` is the query vector (the masked/partial sequence encoding)
- `β` is the inverse temperature parameter
- The update is iterated until convergence (energy change below threshold) or a maximum number of iterations

### Required API

- `store(patterns: Tensor) -> None` — Add encoded sequences to the landscape. Patterns are columns of X.
- `retrieve(query: Tensor, beta: float, max_iter: int = 10, tol: float = 1e-6) -> Tensor` — Run the Hopfield update rule from the query until convergence or max_iter. Return the settled state.
- `energy(state: Tensor) -> float` — Compute the energy of a state (for convergence checking and diagnostics).
- `stored_count` (property) — Number of stored patterns.

### Implementation notes

- All operations through the substrate interface — no direct torchhd imports.
- Accept and return tensors on the same device (MPS/CPU).
- The softmax is over the similarity scores between the query and all stored patterns. At high β, this peaks sharply on the nearest pattern; at low β, it blends across patterns.
- Convergence criterion: `|energy(ξ_new) - energy(ξ_old)| < tol`.

### Deliberate simplification

In the committed architecture, temperature is a geometric property of the query, not a tunable parameter. Phase 2 treats β as an explicit parameter to be swept across values for substrate characterization. This is a deliberate simplification. The β sweep results will inform how temperature-as-geometry should be implemented in later phases.

## Experimental Matrix

### Variables

| Variable | Levels | Values |
|----------|--------|--------|
| Window size | 3 | 8, 16, 32 tokens |
| Mask count | 3 | 1, 2, span (3 contiguous) |
| Mask position | 3 | center, edge (position 1 or 2), end-of-window |
| Objective | 2 | masked-token, next-token |
| Landscape size | 3 | Small, medium, large (exact counts determined by corpus; target roughly 500, 5000, 50000 stored sequences) |
| Inverse temperature (β) | 3 | Low, medium, high (exact values determined empirically during implementation; start with 1, 10, 100 and adjust if needed) |
| Retrieval condition | 2 | Memorization (test on stored sequences), generalization (test on held-out sequences) |

### Total conditions

3 × 3 × 3 × 2 × 3 × 3 × 2 = 972 conditions.

### Samples per condition

Minimum 1,000 test sequences per condition. Each test sequence is a window drawn from the corpus, with masking/truncation applied per the condition's parameters. Report accuracy as a distribution with mean, standard deviation, and 95% confidence interval.

### How each objective works

**Masked-token:**
1. Draw a window of W tokens from the corpus.
2. Encode the full window as an FHRR sequence vector using the encoding module.
3. Create a masked version: replace the masked token(s) with `<MASK>` and re-encode.
4. Present the masked encoding as a query to the Hopfield landscape.
5. After settling, decode the masked position(s) from the settled state using `decode_position`.
6. Find the nearest codebook atom to the decoded vector (cosine similarity).
7. Score: 1 if the nearest atom matches the true token, 0 otherwise.

**Next-token:**
1. Draw a window of W tokens from the corpus.
2. Encode only the first W−1 tokens as the prefix.
3. Present the prefix encoding as a query to the Hopfield landscape.
4. After settling, decode position W from the settled state using `decode_position`.
5. Find the nearest codebook atom.
6. Score: 1 if the nearest atom matches the true W-th token, 0 otherwise.

### Populating the landscape

For each landscape size condition:
1. Draw the specified number of windows from the **training** split.
2. Encode each window as a full FHRR sequence vector.
3. Store all encoded vectors in the Hopfield module.

For the **memorization** condition: test sequences are drawn from the same set used to populate the landscape.
For the **generalization** condition: test sequences are drawn from the **validation** split (never stored in the landscape).

### Window generation

Windows are non-overlapping consecutive spans from the corpus, discarding any final span shorter than the window size. Sentence boundaries are ignored — windows are purely positional. This is a simplification; later phases may use sentence-aware windowing.

## Trivial Baselines

Three baselines, computed once per condition (no Hopfield retrieval involved):

1. **Random guess:** For each test case, pick a codebook atom uniformly at random. Expected accuracy: ~1/vocab_size.
2. **Unigram frequency:** For each test case, predict the most frequent word in the training corpus. Accuracy is the proportion of test positions where the most frequent word is the true answer.
3. **Bigram frequency:** For each test case, find the token immediately preceding (for next-token) or adjacent to (for masked-token) the target position. Predict the word most frequently following that token in the training corpus. If the preceding token is unknown or has no recorded bigrams, fall back to unigram.

These baselines use no HD encoding, no Hopfield retrieval — just corpus statistics. They establish the floor that substrate-based retrieval must clear.

## Success Criteria

### Part 1: Substrate gate (pass/fail)

Both retrieval objectives must consistently outperform the bigram baseline across experimental conditions with non-overlapping 95% confidence intervals. "Consistently" means across the majority of conditions — not cherry-picked favorable settings.

If neither objective clears bigram: the substrate has a fundamental problem. Do not proceed to Phase 3. Revisit substrate design.

If only one objective clears bigram: that objective is the candidate for Phase 3. Investigate why the other failed.

### Part 2: Architectural pattern (diagnostic, no fixed threshold)

The pattern of results across the experimental matrix answers these questions for Phase 3:

- **Which objective is better?** Does masked-token consistently beat next-token, or only under specific conditions?
- **Where does the advantage come from?** Does the gap disappear when mask position is end-of-window (where masked-token becomes structurally similar to next-token)?
- **How does landscape size affect retrieval?** Is there a capacity cliff, or does retrieval degrade gracefully?
- **How does β affect retrieval?** Which temperature regime produces the best results for each objective?
- **Memorization vs. generalization:** How much does retrieval quality drop when test sequences weren't stored? This directly predicts Phase 3's operating conditions.
- **Window size sweet spot:** Where does the tradeoff between encoding noise (longer windows = more bundle terms = lower SNR) and retrieval ease (longer windows = smaller proportional perturbation from masking) balance?

No fixed threshold is set for Part 2. The pattern is the deliverable. Phase 3 scoping decisions will be made based on what the pattern reveals.

## Post-Hoc Analysis Dimensions

These are computed on the same experimental runs — no additional conditions needed:

1. **Token frequency stratification:** Slice accuracy by the corpus frequency of the target token (e.g., quartiles: top-25% most frequent, 25–50%, 50–75%, bottom-25%). This reveals whether retrieval is dominated by common words or works across the frequency spectrum.

2. **Retrieval confidence:** For each test case, record not just whether the nearest atom was correct, but the similarity gap between the top-1 and top-2 nearest atoms. A large gap indicates confident retrieval; a small gap indicates the system was "uncertain." Distribution of confidence scores per condition is diagnostic.

3. **Energy at convergence:** Record the Hopfield energy of the settled state for each retrieval. Low energy = deep attractor basin = confident retrieval. Correlate energy with accuracy.

## Validation Report

`experiments/02_phase2_retrieval_baseline.py` runs the full experimental matrix and generates `experiments/02_phase2_retrieval_baseline.md`. The report should include:

1. **Dataset summary:** Vocabulary size, corpus statistics, window counts per split.
2. **Accuracy heatmaps:** For each pair of matrix variables (e.g., window size × mask count), show accuracy as a heatmap with separate panels for masked-token and next-token objectives.
3. **Baseline comparison plot:** Accuracy distributions for both objectives vs. the three trivial baselines, across landscape sizes.
4. **β sensitivity curve:** Accuracy vs. inverse temperature for both objectives.
5. **Memorization vs. generalization comparison:** Paired accuracy distributions.
6. **Token frequency analysis:** Accuracy broken down by frequency quartile.
7. **Retrieval confidence distribution:** Histogram of similarity gaps (top-1 minus top-2).
8. **Findings section:** Plain-language summary of the pattern across conditions. Explicit statement of implications for Phase 3: which objective to use, what operating parameters to target, what capacity limits were observed, and any surprises.

## Out of Scope for Phase 2

Do not implement:

- Codebook growth or refinement (Phase 3)
- Hebbian or error-driven update rules (Phase 3)
- Consolidation dynamics (Phase 3)
- Atom splitting or Hartigan's dip test (Phase 5)
- Hierarchical compression (Phase 4)
- LLM integration (Phase 6)
- Replay buffer (Phase 6)
- BPE or subword tokenization (revisit at Phase 3 boundary)
- Temperature-as-query-geometry (later phase — β is explicit in Phase 2)
- Sentence-aware windowing (possible later refinement)

If the design of any Phase 2 component seems to require one of these, the design is wrong. Stop and ask Dylan.

## Phase 1 Empirical Baselines

These numbers from Phase 1 validation are relevant to Phase 2 expectations:

| Metric | Phase 1 value | Phase 2 relevance |
|---|---|---|
| Round-trip fidelity | 1.0000 | Encode/decode pipeline is algebraically exact |
| Quasi-orthogonality std | 0.011 | Noise floor for random codebook atoms |
| Bundling recovery at K=10 | 100% | Window size 8 should encode cleanly |
| Bundling signal at K=10 | 0.282 | Expected per-position SNR for short windows |
| Scaling exponent | −0.505 | SNR degrades as ~1/√K with window size |
| Bundling recovery at K=50 | 100% | Window size 32 should still encode, but with lower SNR |
| Bundling signal at K=50 | 0.126 | Expected per-position SNR for longest windows |

## Open Implementation Questions

These can be settled during build without architectural discussion:

1. **Exact vocabulary cutoff:** Target 5,000–10,000. Pick based on frequency distribution of WikiText-2 training split — look for a natural elbow in the frequency curve.
2. **Exact landscape sizes:** Depends on how many non-overlapping windows each window size produces from the training split. Target three levels spanning roughly two orders of magnitude.
3. **β values:** Start with {1, 10, 100}. If all three produce similar results, the parameter isn't sensitive. If results vary wildly, add intermediate values.
4. **Max Hopfield iterations:** Start with 10. If most retrievals converge in fewer, lower it. If some don't converge, investigate whether those cases correlate with accuracy.
5. **Handling multi-token masks:** For mask count > 1, the scoring is per-masked-position. Report both per-position accuracy and all-correct accuracy (did the retrieval recover ALL masked tokens correctly?).
