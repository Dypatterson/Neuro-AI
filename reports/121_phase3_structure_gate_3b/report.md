# Report 121 — Phase-3 structure-gate "3b": NULL on the meaningful (paradigmatic) gate → re-scope signal

**Status:** RESOLVED — terminal finding (re-scope signal, per the pre-registered prior). NOT an open blocker.
**Classification:** Phase-3 structure-gate **graduation** experiment → **NULL on the meaningful (paradigmatic) gate**; COLLOCATIONAL structure only.
**Date:** 2026-05-31
**Spec:** [notes/emergent-codebook/phase-3-structure-gate-3b-design.md](../../notes/emergent-codebook/phase-3-structure-gate-3b-design.md) · charter [CONTEXT.md](../../CONTEXT.md) §5.
**Code:** `experiments/60_phase3_structure_gate_3b.py` (reuses `phase2/codebook_learner.CodebookLearner`, `phase2/corpus`).
**Builds on:** the 2026-05-31 re-grounding ([RE-GROUNDING-MAP.md](../RE-GROUNDING-MAP.md)) + dependency analysis (3b = the single genuine next build, gating Phase 5).

---

## TL;DR — verdict

**The emergent codebook develops corpus-specific COLLOCATIONAL structure (co-occurring words cluster — small, real, but ~by construction for a co-occurrence learner) but NOT the PARADIGMATIC/semantic structure Phase 5 needs (similar, non-co-occurring words do not cluster — they slightly anti-cluster).** This is the **meaningful 3b gate, and it is a NULL** (repo_sample) / **significant NEGATIVE** (WikiText). Per the pre-registered prior, **this is a RE-SCOPE signal, not a retry**: the current first-order Hebbian co-occurrence-centroid growth dynamics are the wrong shape to produce paradigmatic structure. **Phase 5 (bind-vs-bundle discovery, atom-splitting, analogical retrieval) remains un-founded** — it requires similar-things-cluster, which does not emerge.

The discipline that produced this: a pre-registered "null = re-scope" prior + an **adversarial verification** that caught a tempting statistical "PASS" was an artifact of a collocation-contaminated semantic-pair list.

---

## Experiment preamble (per CLAUDE.md)

- **Active phase:** Phase 3 — FOUNDATION (codebook growth). Genuine open gate: **structure-gate 3b**.
- **Headline metric** (per the 3b spec + `experimental-progression.md §"How to know it's actually working"` — "similar tokens develop similar hypervectors… vs a control"): **PARADIGMATIC** related-pair-cosine drift (end−init) on real text **vs a gauge-safe corpus-stream-shuffle control**, CI > 0, AND specificity (semantic > random) CI > 0. "Paradigmatic" = curated semantic pairs **that do not co-occur** (the king/queen test); the co-occurring subset (collocational) ≈ the PMI floor and is not the gate.
- **Required controls:** gauge-safe corpus-stream-shuffle (primary; destroys co-occurrence, preserves marginals); random-pair specificity arm; PMI-collocation floor; init-as-baseline DiD. The atom-relabel shuffled-token control is RETIRED (gauge-vacuous).
- **Last verified:** the consolidation-write **floor** cleared (055–058); 3b OPEN.
- **Why now:** 3b is the single genuine next build (dependency analysis), the load-bearing test of the "more than memory" thesis, gating Phase 5.
- **Pre-registered honest prior:** 3b MAY FAIL (the C.3 null). A null = re-scope of the growth dynamics, NOT "run it bigger." Committed before the run.

---

## Mechanism / method

`CodebookLearner` (`phase2/codebook_learner.py`) is a first-order Hebbian distributional learner: each token's vector is interpolated toward the unit-normalized centroid of its **co-occurring** context tokens, with a repulsion term. We train it on (a) the real token stream and (b) a **gauge-safe corpus-stream shuffle** (permute the flat token sequence before windowing → preserves unigram marginals, destroys co-occurrence), from the **same** random init per seed. We then measure codebook-cosine **drift** (end−init) for curated, corpus-independent semantic pairs (synonyms/antonyms/category-mates — the "king/queen" set), split by within-corpus co-occurrence into **PARADIGMATIC** (cooc ≤ 2, the meaningful gate) vs **COLLOCATIONAL** (cooc > 2, ≈ the PMI floor). Anti-homunculus: offline batch statistics (cosine, d_eff); the control is a data manipulation; no runtime gate; read never touches energy→argmin→ΔE.

---

## Results

### WikiText-2 (D=1024, W=6, 10 epochs, 5 seeds, max_vocab=2000) — the powered headline

| arm | n pairs | real−shuffle drift | bootstrap 95% CI | PASS |
|---|---|---|---|---|
| **PARADIGMATIC** (cooc ≤ 2) — *the meaningful gate* | 11 | **−0.0084** | **[−0.0129, −0.0036]** | **FAIL (CI < 0)** |
| COLLOCATIONAL (cooc > 2) — ≈ PMI floor | 34 | +0.0163 | [+0.0047, +0.0292] | PASS |
| overall semantic arm (mixed) | 45 | +0.0103 | [+0.0014, +0.0205] | (PASS — but contaminated) |
| PMI-collocation floor | 100 | +0.1249 | [+0.1024, +0.1495] | PASS |
| specificity (semantic − random, real) | — | +0.0458 | [+0.0363, +0.0555] | PASS |

- **corr(log co-occurrence, drift) = +0.405** — the related-pair lift **scales with co-occurrence**: it is collocation-driven, not similarity-driven.
- **No collapse:** d_eff 678 → 651 (real) / 669 (shuffle); max-pairwise-sim 0.53 (real) / 0.23 (shuffle).
- The overall semantic arm "passes" (+0.0103) **only because 34/45 curated pairs co-occur** — it is the collocational subset carrying a contaminated headline. The genuinely paradigmatic subset (king/queen itself, big/large, small/little, year/month) is **zero-to-negative**.

### repo_sample (companion, underpowered: 10 in-vocab pairs)

corr(log cooc, drift) = **+0.816**; PARADIGMATIC (n=8) +0.0021, CI [−0.0062, +0.0118] → **null (FAIL)**; COLLOCATIONAL (n=2) +0.0996 → PASS. **Same verdict** as WikiText: collocation-driven, paradigmatic null.

### Effect-size honesty

Even the *overall* (collocation-inflated) semantic real-drift is +0.085, of which only **~12% (+0.010) is corpus-specific** — the other ~88% (+0.075) happens under the shuffle too (corpus-independent training contraction). The C.3 prior ("consolidation change ≈ corpus-independent") is largely vindicated; the only corpus-specific structure is collocational.

---

## Adversarial verification (what flipped the verdict)

An adversarial verifier confirmed the statistics are sound (stream-shuffle correctly destroys co-occurrence while preserving marginals; valid percentile bootstrap; CI robust; real/shuffle share init) — **but falsified the "semantic/paradigmatic" framing**: corr(log cooc, drift)=+0.41; cooc≥5 pairs drive the effect (+0.024), cooc<5 don't (−0.005), and **king/queen itself drifts −0.005**. 93% of the curated list co-occurs, so the overall "pass" was the collocation floor in disguise. This report's co-occurrence split formalizes that finding reproducibly. **The verification is what prevented banking a false PASS.**

---

## What this means (and the re-scope direction)

3b asked: *does the codebook develop corpus-specific structure?* The honest, decomposed answer:
- **Collocational** (co-occurring tokens cluster): **YES** — but a co-occurrence-centroid learner produces this ~by construction; it is the floor, not the "more than memory" signal.
- **Paradigmatic / semantic** (similar, non-co-occurring tokens cluster — the king/queen test, the substrate Phase 5's bind-vs-bundle discovery / atom-splitting / analogical retrieval operate on): **NO** (null/negative).

So the *first-order* Hebbian distributional-centroid growth dynamic is the **wrong shape** for the thesis: it captures syntagmatic (co-occurrence) structure, not paradigmatic (substitutional/second-order) structure. **Phase 5 remains un-founded** — building it now would be the phase-order violation the re-grounding already flagged. The re-scope target is the **codebook-growth mechanism itself**: a candidate that clusters by **context similarity** (second-order distributional structure — tokens with similar *neighborhoods*, not tokens that are *neighbors*), e.g. a context-vector / SQHN / predictive-coding-style update, rather than the current first-order co-occurrence centroid. That is a Phase-3 mechanism redesign, **not** a Phase-5 build and **not** a re-run of this null at larger scale.

---

## Falsifier ledger

| Gate | Outcome |
|---|---|
| **PARADIGMATIC real-beats-shuffle (the meaningful 3b)** | **NULL/NEGATIVE** — WikiText CI [−0.013, −0.004] < 0; repo_sample null. |
| Collocational real-beats-shuffle (floor) | PASS (~by construction). |
| Specificity (semantic > random) | PASS — but carried by the collocational subset. |
| Collapse guard (d_eff) | PASS (no collapse). |
| corr(log cooc, drift) | **+0.41 / +0.82** — confirms the effect is collocation-driven, not similarity-driven. |

**Not implemented:** the NC1 / inter-basin-separability drill-down (spec control #4). It would explain *why* the headline moves — but the paradigmatic headline is null, so it is moot for this verdict (noted for any re-scoped follow-up).

---

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python experiments/60_phase3_structure_gate_3b.py \
  --corpus-source wikitext --seeds 5 --epochs 10 --max-vocab 2000 \
  --out reports/121_phase3_structure_gate_3b/wikitext.json
PYTHONPATH=src .venv/bin/python experiments/60_phase3_structure_gate_3b.py \
  --corpus-source repo_sample --seeds 5 --epochs 10 --max-vocab 2000 \
  --out reports/121_phase3_structure_gate_3b/repo_sample.json
```

Result JSON/logs are gitignored (`reports/**/*.json`, `*.log`); this `report.md` is the durable record.
