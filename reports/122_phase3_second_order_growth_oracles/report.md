# Report 122 — Phase-3 second-order growth: WikiText NULL → §10 oracles → research arc → growth redesign

**Status:** Phase-3 structure-gate (second-order re-scope). The WikiText-2 graduation
run of `experiments/61` is a **NULL** on the 4-condition gate; the pre-registered §10
oracles (`experiments/62`) show the **paradigmatic signal EXISTS** in the corpus
statistics and survives a single FHRR port; an adversarial pressure-test then **killed
the "it was a static-metric artifact" rescue** and **redirected to the growth redesign**.
The live deliverable is unchanged from Report 121's re-scope: a *local growth rule that
writes second-order/paradigmatic structure, with a built-in anti-collapse force.*

**Classification:** Phase-3 structure-gate graduation experiment (exp 61) + its §10
oracle drill-down (exp 62) + a research/pressure-test synthesis. Floor (055–058)
untouched.

---

## 1. The graduation run — NULL (experiments/61, WikiText-2, Colab GPU)

Config: D=1024, W=6, max_vocab=2000, 5 seeds, k pinned by SPPMI density, α swept
{0.05,0.1,0.2,0.3}, SimLex-999 ≥ 5.0 (n=40 paradigmatic pairs), hierarchical
seed×pair bootstrap. Variants B′ (row-centered SPPMI, primary), B (uncentered), A
(first-order). **Verdict: NULL on all four gate conditions.** Key cells (α=0.05, the
only non-collapsing regime):

| variant | headline (demeaned matched, real−shuf) | corr(log cooc, drift) | gauge corr(S_real,S_shuf) | d_eff ratio |
|---|---|---|---|---|
| **B′** (row-centered SPPMI) | +0.0294, CI [0.025, 0.034] | **0.250** | **0.791 (LEAKS)** | 0.922 |
| **A** (first-order P·Pᵀ) | +0.0297 | **0.140** | **−0.017 (clean)** | 0.943 |

Three findings: (a) **A ≈ B′** — PMI's marginal-division bought *nothing* on real text
(the grill predicted it would only tie on the frequency-controlled toy; it tied on
WikiText too). (b) **The SPPMI gauge LEAKS (0.791)** — `SPPMI@SPPMIᵀ` is
frequency-dominated and frequency survives the stream-shuffle, so the real-vs-shuffle
framing is *insensitive* for a second-order operator; the SPPMI cells are gauge-INVALID,
exactly as grill-G4 predicted (W=6/k=5 → 0.79). (c) **The density target [0.3,0.5] was
unreachable** (max 0.186 at k=1; density only falls with k).

## 2. The §10 oracles — the signal EXISTS (experiments/62, gauge-free)

Per the precommit §10 pre-registered disposition for a NULL, two gauge-free oracles
(reusing exp-61's functions):

- **ORACLE 1 — SVD-of-SPPMI (substrate-free, L&G's own embedding):** the paradigmatic
  signal **EXISTS**. king/queen cos = **0.222**; paradigmatic (40 non-co-occurring
  SimLex pairs) 0.146 vs random 0.037; specificity **+0.109, CI [0.082, 0.137]**. Sanity
  ordering collocational (0.343) > paradigmatic (0.146) > random (0.037) — the embedding
  works. WikiText-2's SPPMI statistics carry genuine paradigmatic structure.
- **ORACLE 2 — FHRR-port (single-shot `S'@G`):** the signal **SURVIVES** (specificity
  +0.101, CI [0.062, 0.139]) **but with global contraction** (random pairs sit at 0.577,
  not ~0).

**→ the corpus + operator carry the signal; the NULL is a GROWTH-DYNAMICS / GAUGE-CONTROL
issue, NOT "the substrate can't carry paradigmatic structure" (NOT Option B).**

## 3. The research arc (3 workflows) — diagnosis + reframes

- **Anti-smush diagnosis (operator-spectrum collapse).** The growth operator
  `S' = rownorm(SPPMI@SPPMIᵀ) − rowmean` is intrinsically near-rank-1 *even after
  row-centering* (top eigenvalue 0.150 vs the next five ~0.005 — a 30× gap), because real
  tokens share high-PMI context profiles. So `centroid = S'@G` is *one step of power
  iteration* — it projects every vector onto that one dominant shared direction (random
  pairs → 0.58). Row-centering removes *one* shared eigenvector; a second survives. The
  per-coordinate unit-modulus normalize discards the magnitude spectrum that could
  re-spread them. **The fix shape:** a local force that removes the common-mode (rank-1),
  *not* whiten the whole Gram (whitening's fixed point is PCA/ZCA = the forbidden global
  word2vec shortcut).
- **Codebook-framing reframe (static vs dynamical).** Grounding found the codebook is
  used *both* ways: as faithful **landscape-seeds** in the recall floor (atoms bound into
  windows → stored as Hopfield patterns → retrieved by settling), but as a **static
  lookup table** in the growth + the 3b read (`cos(G_i,G_j)`, no bind/bundle/store/settle
  — grep-confirmed) — the "vector-DB + summaries" geometry `PROJECT_PLAN.md:276` forbids.
  This surfaced a real **β-decoupling finding** (see §4): high static cosine ≠ merged
  Hopfield basin.

## 4. The pressure-test — R1 (basin-read reframe) KILLED as a rescue; β-decoupling BANKED

The codebook-framing workflow proposed **R1**: swap the 3b read from static cosine to
**co-completion / basin-overlap** on the Hopfield settling loop, claiming the NULL was a
static-metric/smush artifact. An adversarial pressure-test (empirical probe + 4
adversaries) ran the real `TorchHopfieldMemory.retrieve()` loop and returned
**SURVIVES-WITH-FIXES as a drill-down; CRACKED as a graduation rescue:**

- **BANKED (real finding):** a faithfully-smushed codebook (rank-3 shared subspace so
  d_eff craters, mean cosine ~0.55–0.65) does **NOT** merge Hopfield basins at β=30 —
  retrieval acc=1.00, 40/40 distinct attractors, co-completion gap +0.15..+0.24. And it
  **β-decouples** (the *same* 0.65-cosine codebook is one merged attractor at β=10, 40
  distinct at β=30/100). So *high static cosine ≠ merged basin*; the operating box at
  β=30 holds to ~0.65 cosine / d_eff ~6.6, and collapses past ~0.67 (recoverable only at
  β≥100). Worth a one-paragraph drill-down someday.
- **KILLED (the rescue):** **the 3b paradigmatic null was never a smush.** Report 121's
  graduated arm passed the collapse guard (d_eff 678→651, max-pairwise 0.53,
  `report.md:50,86`). The null is the **absence of paradigmatic signal written**
  (king/queen ~0/negative after demeaning; corr with co-occurrence +0.41 → collocational).
  Co-completion can only read out structure retrieval-selectivity already encodes — it
  **cannot manufacture paradigmatic structure the growth rule never wrote.** R1 would, at
  best, re-pass a guard the project already passes; at worst be reinterpretation #3 of the
  same null. Two factual corrections it forced: "R1 == the NC1 control Report 121 skipped"
  is **false** (NC1 drills the *floor's* stored-window basins; the grown G is never
  stored), and `metastability_contribution` is a per-atom replay-priority statistic, **not**
  a pairwise co-completion read.

**The pressure-test did its job:** it caught a seductive, partly-true reframe being
mis-sold as a rescue for a null it cannot rescue, and redirected to the real deliverable.

## 5. Verdict + disposition (the live deliverable)

**Unchanged from Report 121's re-scope: a Phase-3 GROWTH mechanism that *writes*
second-order/paradigmatic structure — with a built-in, local, anti-homunculus
anti-collapse force.** All three workflows + the pressure-test converge on:

> **Row-centered SPPMI (B′) for "pull-similar," composed with the energy-native
> `H_anti = −α·log(d_eff)` repulsion (`torch_fhrr.py:147-182`, already built, currently
> inert) for "keep-apart"** — which removes the common-mode the bundle injects (the
> diagnosed cause), is the brain's lateral-inhibition / homeostatic-decorrelation solution
> (the local analog of the global SVD that provably doesn't contract), and **retires the
> threshold-gated `_apply_repulsion` homunculus** (`codebook_learner.py:144`). Evaluated
> on the **gauge-free para-vs-random specificity gate** (the stream-shuffle gauge is
> invalid for 2nd-order operators — §1) + the collapse floor.

Rejected on the hard local-not-global gate: full Földiák/Pehlevan-Chklovskii whitening of
G (its fixed point *is* PCA/ZCA — the global shortcut). Deferred: a true predictive /
successor-context growth target (the heavier "R3") — pursue if B′+H_anti does not write
the signal.

## 6. Anti-homunculus + discipline

- `H_anti`'s `α_anti` is **fixed at construction, never read back from observed d_eff**
  (`torch_fhrr.py:48-55`) — the gradient *is* the actuator; it passes the anti-homunculus
  filter by construction, unlike the threshold-`_apply_repulsion` it replaces.
- This report is the **anti-rationalization checkpoint**: the gauge reframe (Report-121
  follow-up) and the static-table/basin reframe (R1) were *both* tested and *both* found
  unable to rescue the paradigmatic null. The deliverable was not moved to fit a passing
  result; the signal-exists oracle (§2) is what reopened the *growth* problem, on its own
  gauge-free terms.

## 7. Artifacts

- `experiments/61_phase3_second_order_growth.py` (graduation harness; CUDA-aware;
  vectorized+cached co-occurrence), `experiments/62_exp61_oracles.py` (the §10 oracles),
  `notebooks/061_second_order_growth_wikitext_colab.ipynb` (the GPU run).
- Result JSON (gitignored): `_exp61_wikitext_headline.json` (Drive), local oracle stdout.
- Next: precommit growth-redesign § (`notes/emergent-codebook/phase-3-second-order-growth-precommit.md`)
  + the `H_anti` arm in `experiments/61`.
