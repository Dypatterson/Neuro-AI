# Report 133 — Bet B, Stage 1: streamed predict-the-future writer (SGNS)

**Status:** BANKED — **REDUNDANT-BUT-VALID** (NOT a graduation). Auto-verdict printed PASS;
3-lens adversarial verification adjudicated it down (over-claim caught). Charter: [CONTEXT-B.md](../../CONTEXT-B.md).
Experiment: [`experiments/79_betb_streamed_sgns.py`](../../experiments/79_betb_streamed_sgns.py). Data:
[`wikitext.json`](wikitext.json).

## Preamble

- **Active capability:** Codebook-growth (P3 structure) under **Bet B** — grow paradigmatic structure via a *learned* writer.
- **Headline metric (CONTEXT-B §5):** within-para **label-shuffle B-KILL** pair-specific residual (CI-lo>0, ≥8/10 seeds) + g1 (spec CI-lo>+0.04) + g2 (beats nulled floor by +0.02). **g3 dropped** (dead at n=40). PLUS the "defensible delta from the closed-form reference" requirement (better B-KILL OR reachability the closed form needed the operator for).
- **Controls:** "you-are-just-SVD/NMF" reference (closed-form SVD-of-SPPMI + NMF, same metric); `grow_G` nulled-local floor; random-init anti-inflation; calibration anchor (must hit +0.1092/0.222); streamed loss with the operator NEVER materialized and NO closed-form SVD in the test path; planted smoke.
- **Last verified:** Report 129 (TEM-local NULL, joint-assignment load-bearing, backprop banned); Report 125 Oracle-E (global flashlight +0.124–0.188); Report 127 (nonlinear partition = global k-means).
- **Why now:** Bet B lifted the no-backprop ban; the predict-the-future + learned-writer idea was live but un-run.

## Result (WikiText-2, V=2002, n=10 seeds, d=300, mps; calibration anchor +0.1092/0.222 — VALID)

| arm | spec (para−rand) | B-KILL (kill-lo) | note |
|---|---|---|---|
| **SGNS (TEST)** — streamed, operator-free, no-SVD | **+0.107** (CI-lo +0.078) | **10/10** (+0.039…+0.067; mean +0.053) | epoch-stable (ep2 +0.093 / ep5 +0.107 / ep10 +0.114) |
| SVD-of-SPPMI **reference** | +0.109 (CI-lo +0.082) | +0.071 | the closed-form flashlight |
| NMF **reference** (Oracle-E family) | +0.164 | +0.072 | global factorization of build_S |
| `grow_G` nulled-local **floor** | +0.0002 | ~0 | single-projection on build_S |
| random-init **nolearn** (anti-inflation) | −0.008 | — | architecture alone = nothing |

Eff-dim 192/300 (no collapse). Planted-learner smoke passed (+0.538 on a topic-structured corpus).

## Verdict: REDUNDANT-BUT-VALID — not a graduation

**SGNS matches, does not beat, the closed-form reference.** spec −0.0018 vs SVD; B-KILL strictly
*weaker* than both SVD (+0.071) and NMF (+0.072). This is **Levy-Goldberg 2014** (SGNS implicitly
factorizes shifted-PMI) — a result the experiment's own docstring cited *a priori*. By CONTEXT-B §5's
own gate, the "defensible delta from the reference" fails on **both measurable axes**; the only surviving
delta is **procedural** (reached from a streamed loss without forming the operator), which is a known
equivalence, not a measured capability. **The auto-verdict's "PASS" was an over-claim — the documented
over-correction failure mode (126/127), inverted — caught by the 3-lens pass.**

**What is genuinely real (3-lens confirmed):**
- A **legal Bet-B streamed gradient writer** (operator never materialized, no SVD — verified in code) reaches **pair-specific** paradigmatic structure (B-KILL 10/10) where every **single-projection** local writer nulled.
- **Not a frequency artifact** (Lens 1): a freq-only model predicts a para−shuffle gap of +0.001 vs the actual +0.096; the freq-purged paradigmatic excess (+0.10) *is* the whole signal. B-KILL pins word-identity, frequency, and para-set membership and breaks only the pairing.

## The attribution 2×2 (Lens 2) — the lever is MECHANISM, not input, not backprop

Lens 2 correctly flagged that "the wall was the rules" equivocates: SGNS changed **both** the input (1st-order stream vs the 2nd-order `build_S` operator) **and** the mechanism (joint all-vocab co-adaptation vs single linear projection). Assembling the existing record resolves it:

| | single-projection | joint / global optimization |
|---|---|---|
| **1st-order** (SPPMI / stream) | +0.021 (R123, NULL) | **+0.107 SGNS** |
| **2nd-order** (build_S) | +0.002 (R127 grow_G, NULL) | **+0.164 NMF** (R125) |

Single-projection NULLs on **both** inputs; joint optimization WORKS on **both**. → **The lever is the
mechanism class (joint/global optimization vs single-projection), not the input representation — and not
backprop specifically** (NMF is multiplicative-update, not backprop). This *re-confirms* the 121–132
bound from the legal side; it does not overturn or extend it. "Embodiment not necessary" was **already**
established by Report 125's global pass — SGNS adds only the *legal streamed realization*, not the falsification.

## Lens caveats (verbatim, for the record)

- **L1 (artifact): HOLDS.** Genuine pair-specific structure, not frequency; a benign within-set freq→cosine corr (~+0.24) is held constant across all arms and contributes ~0 to specificity.
- **L2 (fair-comparison): HOLDS-WITH-CAVEAT.** Legal mechanism verified; but do NOT claim a clean "wall-break" — input and mechanism both changed (the 2×2 above is the honest attribution). "Confirms 129's joint-assignment" is soft — 129 was TEM *backprop* slot-assignment on a structural code; SGNS is a different joint optimization (implicit SPPMI factorization).
- **L3 (over-claim): HOLDS-WITH-CAVEAT.** This is the redundant-and-stop outcome dressed as a pass. Bank as "a legal streamed writer matches the reference," NOT a graduation or a beat.

## What this licenses

Stage-1-as-graduation is **retired** (redundant). What it *does* establish: a **legal, streamable,
operator-free gradient writer that reaches the reference level exists**. The only genuinely-novel question
remains **Stage 2** — does a **two-timescale consolidation** (fast episodic store + slow replay-driven
consolidator) *manufacture* pair-specific structure **beyond** what the one-shot writer (SGNS/SVD/NMF)
reaches, and/or resist catastrophic forgetting an online-only writer suffers? That must be licensed
**soberly** (on "a legal streamable writer exists"), judged against a single-timescale ablation — never
on a manufactured Stage-1 delta.
