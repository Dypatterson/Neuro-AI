---
date: 2026-05-06
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Phase 2 Specification (Draft v2) — Hopfield Retrieval Baseline

This is the scope-of-work document for Phase 2 of the personal-ai project. Read `CLAUDE.md` for architectural context and `PHASE_1_SPEC.md` for what Phase 1 delivered.

**v2 changes (2026-05-06):** Six bucket-1 amendments from the Phase 2 spec review. See change log at the bottom.

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
- Batched queries: support `[D × batch]` query tensors. Cosine cost is dominated by `X^T @ Q`; batching is essentially free relative to single queries and should be the default in the experimental driver.

### Deliberate simplification

In the committed architecture, temperature is a geometric property of the query, not a tunable parameter. Phase 2 treats β as an explicit parameter to be swept across values for substrate characterization. This is a deliberate simplification. The β sweep results will inform how temperature-as-geometry should be implemented in later phases.

## Mask Representation (committed)

For the masked-token objective, masked positions are represented by binding the position vector with the `<MASK>` codebook atom: `bind(p_k, <MASK>)`. The masked sequence is `bundle(... + bind(p_k, <MASK>) + ...)` — same arity (W bound terms) as the corresponding unmasked sequence.

This is one of two reasonable choices. The alternative is omitting the masked position from the bundle entirely, producing a query of arity W−1. Omission gives a slightly cleaner cue at small W (no quasi-orthogonal noise contribution from `bind(p_k, <MASK>) · bind(p_k, t_k)`); `<MASK>`-substitution matches BERT-style framing and produces a query whose arity matches stored patterns, which is the more plausible operating mode for downstream phases.

This spec commits to `<MASK>`-substitution. Omission-style masking is an explicit deferred ablation, out of scope for Phase 2. If Phase 2 results suggest small-W retrieval is noise-limited rather than signal-limited, the omission ablation is the first thing to revisit at the Phase 3 boundary.

## Experimental Matrix

### Variables

| Variable | Levels | Values | Applies to |
|----------|--------|--------|-----------|
| Window size | 3 | 8, 16, 32 tokens | both objectives |
| Mask count | 3 | 1, 2, span (3 contiguous) | masked-token only |
| Mask position | 3 | center, edge, end-of-window | masked-token only |
| Objective | 2 | masked-token, next-token | — |
| Landscape size | 3 | Small, medium, large (target ~500, 5,000, 50,000 stored sequences) | both objectives |
| Inverse temperature (β) | 3 | Low, medium, high (start with 1, 10, 100; refine empirically) | both objectives |
| Retrieval condition | 2 | Memorization (in-landscape), generalization (held-out) | both objectives |

### Total conditions (asymmetric matrix)

Mask count and mask position only apply to masked-token. Next-token has a single prediction target (position W) by definition. The matrix is therefore asymmetric across objectives:

- **Masked-token:** 3 × 3 × 3 × 3 × 3 × 2 = **486 conditions** (window × mask count × mask position × landscape × β × retrieval)
- **Next-token:** 3 × 3 × 3 × 2 = **54 conditions** (window × landscape × β × retrieval)
- **Total: 540 distinct conditions**

Reporting consequences:
- Heatmaps comparing the two objectives must align on the shared axes (window × landscape × β × retrieval). The masked-token-only axes (mask count, mask position) appear as additional facet dimensions for masked-token panels and are absent from next-token panels.
- The "consistently beats bigram" gate is evaluated *per objective* — masked-token clears bigram across the majority of its 486 conditions; next-token clears bigram across the majority of its 54.

### Mask position semantics (concrete)

For mask count L (where L ∈ {1, 2, 3}) and window size W, the masked positions (1-indexed) are:

- **edge**: positions [1, 2, …, L]
- **end**: positions [W − L + 1, W − L + 2, …, W]
- **center**: positions [⌈(W − L) / 2⌉ + 1, …, ⌈(W − L) / 2⌉ + L]

Concrete table for the matrix levels:

| W | L | edge | center | end |
|---|---|------|--------|-----|
| 8 | 1 | {1} | {5} | {8} |
| 8 | 2 | {1,2} | {4,5} | {7,8} |
| 8 | 3 | {1,2,3} | {4,5,6} | {6,7,8} |
| 16 | 1 | {1} | {9} | {16} |
| 16 | 2 | {1,2} | {8,9} | {15,16} |
| 16 | 3 | {1,2,3} | {8,9,10} | {14,15,16} |
| 32 | 1 | {1} | {17} | {32} |
| 32 | 2 | {1,2} | {16,17} | {31,32} |
| 32 | 3 | {1,2,3} | {16,17,18} | {30,31,32} |

### Test sampling

For each (objective, retrieval condition, window size) triple, draw a single test population of 1,000 windows that is reused across all other matrix axes (mask count, mask position, landscape size, β). This guarantees the post-hoc analyses (frequency stratification, retrieval confidence distribution, energy at convergence) operate on a fixed population per slice, rather than a re-drawn population that varies in composition across conditions.

Sampling rules:

- **Source:** all non-overlapping size-W windows from the appropriate split (training split for memorization condition, validation split for generalization condition).
- **Method:** uniform random sample of 1,000 windows per (objective, retrieval, window size) triple, without replacement, with a recorded seed.
- **If source pool < 1,000:** use all available windows and record the actual N. Likely relevant for size-32 generalization (~6,800 source windows is fine; tighter splits — if any — surface here).
- **Per-condition exclusion at evaluation time:** drop test cases where the target token (the masked token, or the position-W token for next-token) is `<UNK>`. Predicting `<UNK>` is uninformative. Report the effective N for each condition; expect a few-percent drop from 1,000 depending on UNK rate.
- **Frequency stratification** is computed post-hoc on the same population: each test case is bucketed by the corpus frequency quartile of its target token (training-split unigram distribution). No pre-stratified sampling.

### Samples per condition

Effective N per condition is reported (≤ 1,000 after UNK exclusion). Report accuracy as a distribution with mean, standard deviation, and 95% confidence interval (Wilson score for binary outcomes).

### How each objective works

**Masked-token:**
1. Draw a window of W tokens from the test population.
2. Construct the masked sequence by replacing the masked token(s) with `<MASK>`. Encode this masked sequence as an FHRR sequence vector (the cue).
3. Present the cue as a query to the Hopfield landscape.
4. After settling, decode the masked position(s) from the settled state using `decode_position`.
5. Find the nearest codebook atom to each decoded vector (cosine similarity).
6. Score: per-position correctness (1 if nearest atom matches true token, 0 otherwise) and all-correct (1 only if all masked positions decoded correctly).

**Next-token:**
1. Draw a window of W tokens from the test population.
2. Encode only the first W − 1 tokens as the prefix.
3. Present the prefix encoding as a query to the Hopfield landscape.
4. After settling, decode position W from the settled state using `decode_position`.
5. Find the nearest codebook atom.
6. Score: 1 if the nearest atom matches the true W-th token, 0 otherwise.

### Populating the landscape

For each landscape size condition:
1. Draw the specified number of non-overlapping windows from the **training** split.
2. Encode each window as a full (unmasked) FHRR sequence vector.
3. Store all encoded vectors in the Hopfield module.

For the **memorization** condition: test sequences are drawn from the same set used to populate the landscape. The masked/prefix cue is then a corrupted version of a known stored pattern; the test is whether Hopfield recovers the stored pattern from the corruption.

For the **generalization** condition: test sequences are drawn from the **validation** split (never stored in the landscape).

### Window generation

Windows are non-overlapping consecutive spans from the corpus, discarding any final span shorter than the window size. Sentence boundaries are ignored — windows are purely positional. This is a simplification; later phases may use sentence-aware windowing.

## Trivial Baselines

Three baselines, computed once per condition (no Hopfield retrieval involved). Bigram tables are computed from the training split: forward `P(t | t_prev)` and backward `P(t | t_next)`.

1. **Random guess:** For each test case, pick a codebook atom uniformly at random. Expected accuracy: ~1/vocab_size.

2. **Unigram frequency:** Predict the most frequent word in the training corpus. Accuracy is the proportion of test positions where the most frequent word is the true answer.

3. **Bigram frequency:** Computed differently for each objective.

   - **Next-token:** predict `argmax_t P(t | t_{W−1})` from the forward bigram table. Fall back to unigram if `t_{W−1}` is `<UNK>` or has no recorded bigrams.
   
   - **Masked-token, single mask at position k:** predict `argmax_t [P(t | t_{k−1}) + P(t | t_{k+1})]` using forward and backward bigram tables, with each term falling back to its marginal if the conditioning token is `<UNK>` or unobserved. (Bidirectional, because masked-token has bidirectional context. A forward-only baseline would be artificially weak in masked-token's favor.) Edge cases: if `k = 1` use only the backward term; if `k = W` use only the forward term.
   
   - **Masked-token, multi-mask:** for each masked position k, find the nearest non-masked position to the left (`k_L`) and right (`k_R`), even if those are several positions away (the immediate neighbor may itself be masked). Predict `argmax_t [P(t | t_{k_L}) + P(t | t_{k_R})]`. If one side has no non-masked tokens within the window, use only the other side. If both are absent, fall back to unigram.

These baselines use no HD encoding, no Hopfield retrieval — just corpus statistics. They establish the floor that substrate-based retrieval must clear.

## Success Criteria

### Part 1: Substrate gate (pass/fail)

Both retrieval objectives must consistently outperform the bigram baseline across their respective experimental conditions, with non-overlapping 95% confidence intervals. "Consistently" means across the majority of conditions for that objective — not cherry-picked favorable settings.

- Masked-token clears bigram across the majority of its 486 conditions.
- Next-token clears bigram across the majority of its 54 conditions.

If neither objective clears bigram: the substrate has a fundamental problem. Do not proceed to Phase 3. Revisit substrate design.

If only one objective clears bigram: that objective is the candidate for Phase 3. Investigate why the other failed.

### Part 2: Architectural pattern (diagnostic, no fixed threshold)

The pattern of results across the experimental matrix answers these questions for Phase 3:

- **Which objective is better?** Does masked-token consistently beat next-token, or only under specific conditions?
- **Where does the advantage come from?** Does the masked-token gap collapse when mask position is end-of-window with mask count 1 (where masked-token becomes structurally similar to next-token at the same window length)?
- **How does landscape size affect retrieval?** Is there a capacity cliff, or does retrieval degrade gracefully?
- **How does β affect retrieval?** Which temperature regime produces the best results for each objective?
- **Memorization vs. generalization:** How much does retrieval quality drop when test sequences weren't stored? This directly predicts Phase 3's operating conditions.
- **Window size sweet spot:** Where does the tradeoff between encoding noise (longer windows = more bundle terms = lower SNR) and retrieval ease (longer windows = smaller proportional perturbation from masking) balance?

No fixed threshold is set for Part 2. The pattern is the deliverable. Phase 3 scoping decisions will be made based on what the pattern reveals.

## Post-Hoc Analysis Dimensions

Computed on the same experimental runs — no additional conditions needed:

1. **Token frequency stratification:** Slice accuracy by the corpus frequency of the target token (quartiles: top-25% most frequent, 25–50%, 50–75%, bottom-25%). Reveals whether retrieval is dominated by common words or works across the frequency spectrum.

2. **Retrieval confidence:** For each test case, record the cosine similarity gap between the top-1 and top-2 nearest atoms. A large gap indicates confident retrieval; a small gap indicates the system was "uncertain." Distribution of confidence per condition is diagnostic.

3. **Energy at convergence:** Record the Hopfield energy of the settled state for each retrieval. Low energy = deep attractor basin = confident retrieval. Correlate energy with accuracy.

4. **Per-position-within-span (masked-token, multi-mask only):** For span/multi-mask conditions, slice accuracy by which position within the span was the target. Interior-of-span positions should be hardest because both immediate neighbors are also masked. Free analysis given the data being collected; reveals whether retrieval fails gracefully under heavier corruption.

## Validation Report

`experiments/02_phase2_retrieval_baseline.py` runs the full experimental matrix and generates `experiments/02_phase2_retrieval_baseline.md`. The report should include:

1. **Dataset summary:** Vocabulary size, corpus statistics, window counts per split, effective N per condition (after UNK exclusion).
2. **Accuracy heatmaps:** For pairs of matrix variables, with separate panels for masked-token and next-token where applicable. Note that masked-token panels include the mask-count and mask-position dimensions; next-token panels do not.
3. **Baseline comparison plot:** Accuracy distributions for both objectives vs. the three trivial baselines, across landscape sizes.
4. **β sensitivity curve:** Accuracy vs. inverse temperature for both objectives.
5. **Memorization vs. generalization comparison:** Paired accuracy distributions.
6. **Token frequency analysis:** Accuracy broken down by frequency quartile.
7. **Retrieval confidence distribution:** Histogram of similarity gaps (top-1 minus top-2).
8. **Per-position-within-span breakdown** (masked-token, multi-mask only).
9. **Findings section:** Plain-language summary of the pattern across conditions. Explicit statement of implications for Phase 3: which objective to use, what operating parameters to target, what capacity limits were observed, and any surprises.

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
- **Omission-style masking** as an alternative to `<MASK>`-substitution (deferred ablation; revisit at Phase 3 boundary if Phase 2 small-W retrieval is noise-limited)

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
3. **β values:** Start with {1, 10, 100} as a probe sweep on a single landscape size to identify the regime where retrieval transitions from soft blend to nearest-pattern commit. If the transition is bracketed by these values, use them. If the transition lies outside (e.g., between 0.1 and 3, or above 30), re-center the full sweep on the interesting regime. Record the probe results in the report.
4. **Max Hopfield iterations:** Start with 10. If most retrievals converge in fewer, lower it. If some don't converge, investigate whether those cases correlate with accuracy.
5. **Multi-token mask scoring:** report both per-position accuracy and all-correct accuracy.
6. **Mask construction across conditions:** the test population is fixed per (objective, retrieval, window size), but the masked variant is reconstructed per (mask count, mask position) condition from the same base window. This is mechanical, no architectural decision needed.

## Change Log (v1 → v2)

Six bucket-1 amendments from the 2026-05-06 spec review:

1. **Condition count corrected (asymmetric matrix).** Was `3⁴ × 2² × 3² = 972`. Now masked-token contributes 486 distinct conditions, next-token contributes 54. Total: 540. The success criterion is evaluated per objective, not globally.
2. **Mask representation committed.** `<MASK>`-substitution is the chosen scheme. Omission-style is documented as a deferred ablation and added to the out-of-scope list.
3. **Bigram baseline made bidirectional and explicit for multi-mask.** New rules cover single mask, multi-mask, and edge-of-window cases. Falls back through unigram cleanly.
4. **Mask position semantics made concrete.** Formula plus a 9-row table covering every (W, L) combination in the matrix.
5. **Dead step removed.** Step 2 of the masked-token procedure (encode the full window) is no longer present — it was never used downstream.
6. **Test sampling strategy specified.** Single 1,000-window population per (objective, retrieval, window size); reused across mask count, mask position, landscape size, β. UNK-target exclusion at evaluation time. Frequency stratification is post-hoc, not pre-stratified.

Two items from the review were intentionally **not** moved into the spec:

- **Reproducibility seeds** belong in `CLAUDE.md` / `STYLE.md` as a coding policy ("every random source has a recorded seed; codebook is persisted to disk after first generation; test windows are sampled from a recorded seed"), not in the architectural spec.
- **β range adaptive sweep** is now part of Open Implementation Question #3 as a probe-then-refine pattern, not a spec-level lock-down.

Three items from the review were noted as informational only and incorporated where relevant without separate amendments: memorization-condition framing (clarified in the populating-the-landscape paragraph), per-position-within-span analysis (added to post-hoc dimensions and report sections), and prefix-length asymmetry between objectives (already implicit in the scoring procedures; surfaces naturally in the end-of-window mask-count-1 diagnostic).
