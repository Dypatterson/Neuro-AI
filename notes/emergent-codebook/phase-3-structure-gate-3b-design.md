# Phase-3 Structure-Gate "3b" — design spec

> Status: **RESOLVED — NULL on the meaningful (paradigmatic) gate → RE-SCOPE** ([Report 121](../../reports/121_phase3_structure_gate_3b/report.md), 2026-05-31; `experiments/60_phase3_structure_gate_3b.py`). The codebook develops corpus-specific **collocational** structure (co-occurring words cluster, ~by construction for a co-occurrence learner) but **NOT paradigmatic/semantic** structure (similar, non-co-occurring words do not cluster — WikiText paradigmatic subset −0.0084 CI [−0.013,−0.004]; corr(log cooc, drift)=+0.41/+0.82). Adversarial verification caught that the overall-semantic "pass" was a 93%-collocation-contaminated artifact. The pre-registered prior is honored: this is a **re-scope signal** (the first-order Hebbian co-occurrence-centroid growth dynamic is the wrong shape for paradigmatic structure → redesign the growth mechanism for **context-similarity / second-order** structure), **NOT a retry**. Phase 5 remains un-founded. Spec retained below as the as-run design.

## The question

**Does the emergent codebook develop *corpus-specific* structure from experience?**
This is the original Phase-3 emergent-structure deliverable
(`experimental-progression.md` §Phase 3 success criteria + §"How to know it's actually
working": *"Similar items develop similar hypervectors without explicit supervision —
if 'king' and 'queen' don't end up geometrically close after Phase 3, something's
wrong. Cross-checked against a control: structure should appear in the standard run and
be absent in the control."*). It is **distinct from** the consolidation-write *floor*
(role-selective recall, cleared by Reports 055–058): the floor asks "does the memory
recall a stored episode?"; **3b asks "does the codebook's geometry itself become a
learned rule grown from co-occurrence?"** — the load-bearing test of the "more than
memory / the landscape geometry IS the rule" thesis (`overview.md:17`). Every Phase-5
mechanism (bind-vs-bundle discovery, atom-splitting, analogical retrieval) depends on a
YES here; **Phase 5 is gated behind 3b** (dependency verdict §6–7).

## Why the original control was invalid (do not reuse)

The original Phase-3 shuffled-token control was proven **gauge-vacuous**: it permuted
*which i.i.d. atom each token-id wears* (`codebook_ctrl[i] = A_{π(i)}`), and since the
pipeline reads atoms only through codebook rows and the atoms are i.i.d. random-phase
FHRR vectors, `E[Δ] = 0` **by construction** for any corpus and any mechanism
(`notes/notes/2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md`).
The Frame-B level-DiD that chased corpus-specificity through this control is a **closed
drift-artifact — do NOT re-chase it.**

## §Headline metric

**Related-pair-cosine drift, real vs gauge-safe corpus-stream-shuffle, CI-disjoint.**

- For a precommitted set of **related token pairs** (synonyms / collocations; WordNet or
  hand-curated per `experimental-progression.md` §"What to test against"), measure the
  mean codebook cosine `sim(cb[a], cb[b])` at **init** and after **training**.
  `Δrelated = sim_end − sim_init`.
- **Control: gauge-safe corpus-stream shuffle** (the 2026-05-28-decided primary control):
  permute the **flat token sequence before windowing** — preserves unigram marginals,
  destroys co-occurrence. Train an identical codebook on the shuffled stream; measure
  `Δrelated_shuffle`.
- **HEADLINE PASS** = `Δrelated_real − Δrelated_shuffle` has a **CI strictly > 0**
  (multi-seed, bootstrap/Wilson): related pairs are pulled together by **real
  co-occurrence**, not by training dynamics per se. A matched **unrelated/random-pair**
  arm must NOT show the same lift (specificity).

## §Required controls (same protocol, same seeds)

1. **Gauge-safe corpus-stream-shuffle** (primary) — destroys co-occurrence, preserves
   marginals. Real-beats-shuffle is the genuine corpus-specificity signal.
2. **No-consolidation DiD baseline** — difference out the Phase-2 (init) landscape:
   report `(real_end − real_init) − (shuffle_end − shuffle_init)`, NOT raw end-state
   similarity (which carries init structure). (NOT "hold landscape fixed + shuffle only
   the consolidation corpus" — that degenerate noise-injection was rejected, gauge note.)
3. **Unrelated/random-pair arm** — related-pair lift must exceed random-pair lift
   (rules out global contraction/collapse masquerading as structure).
4. **NC1 / inter-basin-separability drill-down** (reuse `phase3/basin_diagnostics.py`:
   `compute_basin_nc1`, `compute_basin_separability_nc2`) — within-basin variability
   bounded-and-nonzero while inter-basin separability **grows** under real and **not**
   under shuffle. Explains *why* the headline moves; not a competing definition.
5. **RETIRED:** the atom-relabel shuffled-token control (gauge-vacuous) — must not appear.

## §Headline + §Controls — RATIFIED REVISION for the second-order re-scope (experiment 61, 2026-06-01)

*The as-run 3b headline above (raw related-pair drift CI > 0) was proven by the
2026-06-01 red-team grill to **PASS ON GLOBAL COLLAPSE** (a contracted codebook lifts
every pair, clearing the CI without any paradigmatic structure). For the second-order
growth re-scope — experiment 61, precommit
[phase-3-second-order-growth-precommit.md](phase-3-second-order-growth-precommit.md) —
the headline + controls are extended below. **Ratified with user agreement 2026-06-01.**
The original text is retained above as the as-run 3b design; this revision governs
experiment 61 (which grows a separate **paradigmatic codebook G** and reads 3b on it).*

- **HEADLINE (exp 61) = demeaned matched specificity, real − stream-shuffle,
  hierarchical-bootstrap CI > 0.** Per-pair paradigmatic drift **minus the per-token
  global-mean drift** (so a global contraction cancels) **minus the matched random-pair
  arm** (random pairs at the *same* low-cooc + frequency regime), real minus shuffle.
  Raw drift is **NON-DIAGNOSTIC** and is reported only alongside the collapse panel.
- **Collapse floor (HARD gate, co-equal with the CI):** `d_eff_end / d_eff_init ≥ 0.5`
  AND `max-off-diag cosine < 0.99` AND global off-diag mean-cosine drift below a frozen
  ceiling. Failing any → NOT a pass, regardless of the CI.
- **Decorrelation gate (wired + numbered):** `corr(log cooc, paradigmatic drift)`
  bootstrap 95 % CI upper bound `< +0.15` (> 60 % drop from the +0.41 incumbent), added
  to the pass conjunction.
- **Gauge-validity gate:** `corr(S_real_offdiag, S_shuffle_offdiag) < 0.40`; above it the
  stream-shuffle control is leaking marginal structure and the run is **INVALID (not a
  null)** → re-run at a `(W, k)` that separates.
- **Pair source:** **SimLex-999** (Hill et al. 2015) filtered to similarity ≥ 5.0,
  in-vocab, within-window cooc ≤ `paradigmatic_max_cooc`; frozen + hashed (precommit §5).
  The original hand-curated `SEMANTIC_PAIRS` list is a separately-reported secondary arm.
- **Pre-registered NULL (exp 61):** if **no** `(k, α)` grid point satisfies the
  specificity CI AND the collapse floor *simultaneously*, that is a NULL → latent-layer
  fork (the signal is in `S` but the FHRR growth cannot read it without collapse).

## Anti-homunculus check (PASSES)

- **Local dynamic:** codebook atoms drift under the fixed Hebbian/error two-pathway
  update (`phase2/codebook_learner.py`); the measured quantities are **offline** cosine
  + **offline** NC1/separability batch statistics (AH-exempt, same class as the
  Report-044 diagnostics). No runtime metric gates, branches, or selects.
- **No decision:** the geometry either separates related pairs or it does not, as a
  consequence of the fixed update — there is no arbiter.
- **The control is a data manipulation** (stream shuffle), not a mechanism. Read
  terminates in a similarity/separability measurement, never an energy→argmin→ΔE.

## Reuse (do not reinvent)

- Training: `experiments/03_phase3a_hebbian_codebook.py` + `phase2/codebook_learner.py`
  (`CodebookLearner`) + `phase2/corpus.py` (`build_vocabulary`, `encode_texts`,
  `make_windows`); add a `corpus_stream_shuffle(token_ids, seed)` before windowing.
- Structure metrics: `phase3/basin_diagnostics.py` (NC1, separability).
- CIs: `phase2/metrics.py` Wilson / bootstrap.

## §Done-gates

1. Headline `Δrelated_real − Δrelated_shuffle` with CI, multi-seed (n≥5).
2. All controls (corpus-stream-shuffle, no-consolidation DiD, random-pair, NC1/sep) on
   the same protocol/seeds.
3. Drill-downs explain the headline.
4. Written up under `reports/`; STATUS + CONTEXT updated walk-back-first.

## PRE-REGISTERED HONEST PRIOR (binding interpretation)

The C.3 evidence is a real prior that **3b MAY FAIL** at current scale (consolidation
change ≈ corpus-independent; atoms collapse spread→tight ~90% rather than bifurcating).
**A null 3b is a RE-SCOPE signal** — the emergent codebook does not yet develop
corpus-specific structure → rethink the **growth dynamics** (the codebook-learner update
rule / capacity / objective), NOT "run it bigger" and NOT "build Phase 5 anyway." A
PASS licenses opening Phase 5 (still gated behind de-arbitrating the Phase-5′
`min_branch` aggregator). **I will not reinterpret a null as a near-miss.** This
interpretation is committed before the run (CLAUDE.md anti-rationalization).
