# Levy & Goldberg 2014 (SPPMI) + Mikolov et al. 2013 (SGNS / negative sampling)

*DRAFT card (2026-05-31). Both primaries opened and verified verbatim this
session. **Manifest upgrade PENDING:** add two `source_manifest.jsonl` entries
(currently absent from the corpus) and checksummed local PDFs before this card is
treated as load-bearing (Hard Rule 3, `principles.md:81-84`). Until then, cite
this card, not a manifest stub.*

- **Source ids:** `levy_goldberg_2014_implicit_mf` (NeurIPS 2014, "Neural Word
  Embedding as Implicit Matrix Factorization", Levy & Goldberg, Bar-Ilan) and
  `mikolov_2013_distributed_reps` (NeurIPS 2013 / arXiv:1310.4546, "Distributed
  Representations of Words and Phrases and their Compositionality", Mikolov,
  Sutskever, Chen, Corrado, Dean).
- **Primary opened this session:** YES, both. L&G: 9-page NeurIPS primary
  (`papers.nips.cc/paper/5477`), pdftotext-extracted, Eqs 1–12 read directly.
  Mikolov: arXiv:1310.4546, §2.2 + Eq 4 read directly. NOT carded from summaries.

## Claim (load-bearing)

Skip-Gram with Negative Sampling (SGNS = the word2vec NEG objective) **implicitly
factorizes** a word-context matrix whose cell is **shifted PMI**: at the optimum
for sufficiently high dimension, the per-pair objective is optimized at
`w·c = PMI(w,c) − log k` (L&G **Eq 7**). The explicit sparse matrix that matches
this is **SPPMI** (L&G **Eq 12**); its rows, used directly or after SVD, perform
*slightly better* than word2vec vectors on word-similarity and analogy tasks.
Mikolov **Eq 4** is the NEG objective L&G analyze, with `k = 5–20` (small data) /
`2–5` (large), noise `Pn(w) = U(w)^{3/4}/Z`.

## The precise operators (verbatim from the primaries)

- PMI (Eq 9/10): `PMI(i,j) = log( P(i,j) / (P(i)P(j)) ) = log( #(i,j)·|D| / (#(i)·#(j)) )`.
- PPMI (Eq 11): `PPMI(i,j) = max(PMI(i,j), 0)`.
- **SPPMI (Eq 12):** `SPPMI[i,j] = max( log( P(i,j)/(P(i)P(j)) ) − log k, 0 )`.
  (`k` = the negative-sampling shift; `k ≥ 1`; larger `k` → sparser, more
  aggressive thresholding.)

## Frequency-correction argument (WHY variant B should beat variant A — the crux)

A raw row-normalized co-occurrence profile `P = rownorm(C)` has its largest
entries on the most *frequent* contexts (the/of/and), because a high-frequency
context inflates `#(i,j)` for nearly every word `i`. So `S_A = P @ Pᵀ` (cosine
between raw profiles) is **dominated by shared high-frequency contexts** and washes
out the mid/low-frequency contexts that carry word identity. PMI fixes this
**mechanically, not heuristically:** dividing the joint `P(i,j)` by the product of
marginals `P(i)P(j)` cancels exactly that frequency mass, measuring association
*above chance co-occurrence*. L&G §3.3 further note the raw matrix is
"inconsistent" — a frequent pair seen once gets a *negative* entry while an
unobserved frequent pair gets 0 — and PPMI/SPPMI's `max(·,0)` drops those
unreliable rare-pair negatives, leaving a sparse consistent matrix. Empirically
PMI > PPMI > raw on word-similarity (Bullinaria–Levy, Turney–Pantel).

**First-order vs second-order:** the matrix *cell* is first-order/syntagmatic
(`i` and `j` co-occur = collocational); word similarity is computed between *rows*
(context profiles) = second-order/paradigmatic/substitutional (`i` and `j` have
*similar contexts* even if they never co-occur — the king/queen test). SPPMI is
the principled weighting that makes the second-order row-similarity discriminative.

## How to turn SPPMI into the codebook-update operator S (the FHRR port)

1. Build windowed counts `#(i,j)`, marginals `#(i)`, `#(j)`, total `|D|` (same
   windows as `CodebookLearner.build_cooccurrence`).
2. `SPPMI[i,j] = max( log(#(i,j)·|D| / (#(i)·#(j))) − log k, 0 )` (real V×V, sparse,
   non-negative).
3. `S = rownorm( SPPMI @ SPPMIᵀ )` — the **second-order** context-profile-similarity
   matrix (row-normalized so update strength is fan-out-independent).
4. `centroid = S @ codebook`; `codebook ← substrate.normalize(α·centroid + (1−α)·codebook)`
   — drop-in replacement of `C` with `S` in `CodebookLearner.train`
   (codebook_learner.py ≈ :100–115, which today does `centroid = C @ codebook`).
   Codebook rows stay complex unit-modulus; `S` is real, so `S@codebook` is a
   real-weighted bundle of phasors, re-projected to unit modulus by the existing
   `substrate.normalize` — exactly as the current `C@codebook` path already does.

## FHRR-substrate caveats (HONEST: this is OUR extrapolation, not in the primaries)

1. **Both primaries use REAL vectors only.** Neither says anything about complex
   unit-modulus phasors, FHRR binding, or applying a context-similarity matrix to
   a complex codebook. The `S@codebook`-in-FHRR construction is the project's own
   bet, licensed by analogy, not by the literature.
2. The literature similarity is `cos(SPPMI_i, SPPMI_j)` between explicit SPPMI
   rows; our `S@codebook` instead uses `S` to *re-bundle the learned phasor
   codebook* — a second-order Hebbian write, not a row-vector readout. Whether the
   paradigmatic signal survives per-component renormalization of a complex weighted
   bundle is **UNTESTED**.
3. The incumbent already applies a *partial* frequency discount (`1/√count`,
   codebook_learner.py ≈ :42–50,66) — weaker than PMI's marginal-product division,
   no negative thresholding, still first-order. So the A→B contrast is
   "√-inv-freq first-order" vs "SPPMI second-order"; report it as such.
4. The `−log k` shift and `max(·,0)` are real scalar ops on the V×V matrix and are
   substrate-agnostic; the FHRR risk lives entirely in step 4 (the bundle).

## Neuro-AI relevance / why now

Directly grounds the variant-A/B Phase-3 growth-mechanism redesign that
[Report 121](../../reports/121_phase3_structure_gate_3b/report.md) opened. 121
found the current first-order Hebbian co-occurrence-centroid learner (= variant A
without the inner second-order step) develops **collocational/syntagmatic**
structure (PASS, ~by construction) but is **NULL/NEGATIVE on the paradigmatic
gate** (king/queen −0.005; cooc≤2 subset −0.0084, CI<0; `corr(log cooc, drift) =
+0.41/+0.82`). `S = rownorm(SPPMI @ SPPMIᵀ)` is exactly the "cluster by context
similarity (second-order)" operator 121 named, and L&G prove SPPMI is the
*principled* weighting for it. Prediction: **B moves the paradigmatic drift
positive where A is null/negative, with the collocational floor and the gauge-safe
stream-shuffle control unchanged.**

## Anti-homunculus

PASSES. `S` is a fixed offline batch statistic; the update is a local geometric
write with no runtime metric-reader, no `if-metric-then` branch, no supervisor.
The A-vs-B-vs-C choice is a precommitted offline data-shaping decision, not a
runtime arbitration. (See the precommit
[`notes/emergent-codebook/phase-3-second-order-growth-precommit.md`](../../notes/emergent-codebook/phase-3-second-order-growth-precommit.md)
§7 for the variant-C top-k boundary.)

## Built-in falsifiers (precommit — use the SAME 121 protocol)

1. **Paradigmatic real-beats-shuffle** drift CI > 0 under B where A is
   null/negative (gauge-safe corpus-stream-shuffle; **per-arm S** — a shared S
   leaks co-occurrence into the control).
2. **Matched specificity:** paradigmatic-similar drift > paradigmatic-random drift
   (same low-cooc regime), CI > 0.
3. **Collocation-decorrelation:** `corr(log cooc, drift)` under B must DROP toward
   0 vs the incumbent +0.41/+0.82 — else B is still collocation-driven and the
   second-order machinery bought nothing.
4. **k-sweep:** vary `k` ~1 order of magnitude; a lift that depends on one `k` is
   not robust. Watch SPPMI density (large `k` can zero out a small-vocab matrix).
5. Adversarial-verify the pair list for collocation contamination (121 caught a
   false PASS that was the co-occurring 34/45 subset in disguise).

## Transfer verdict

**transfers-with-caveats** — to the Phase-3 growth-mechanism redesign (variant B)
ONLY.
- **Paper proves:** SGNS optimum = PMI − log k (Eq 7); SPPMI = max(PMI − log k, 0)
  (Eq 12) is the explicit matching matrix; PMI's marginal-division is the frequency
  correction; second-order row-similarity (paradigmatic) is the standard
  word-similarity quantity.
- **Project has shown (121):** variant A (first-order co-occurrence centroid) is
  paradigmatic-NULL.
- **Project's OWN extrapolation (untested):** that `S=rownorm(SPPMI@SPPMIᵀ)` applied
  as `S@codebook` in complex FHRR space recovers the paradigmatic signal A misses.
- **Boundary:** the SGNS⇔SPPMI equivalence holds at the optimum for large d — it is
  a *what-is-being-factorized* result, NOT a held-out generalization claim (same
  boundary as the Dorrell card). It promises the *right-shaped* second-order growth
  signal, not Phase-5 compositional generalization.
- **STOP-RULE:** if B lifts real-text paradigmatic drift above the shuffle control
  AND decorrelates from log-cooc, that is the re-scope's first PASS — surface it.
- **DON'T-LAUNDER:** SPPMI row-similarity success in real-vector NLP ≠ FHRR
  phasor-bundle success; keep the substrate caveat attached.
