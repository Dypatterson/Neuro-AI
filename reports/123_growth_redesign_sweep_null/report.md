# Report 123 — Growth-redesign sweep (B′ + H_anti) → NULL → the paradigmatic signal is in the *subdominant* modes

**Status:** Phase-3 second-order growth re-scope. The pre-registered WikiText-2 D=4096
sweep of **B′ (row-centered SPPMI) + the energy-native `H_anti` anti-collapse**
([precommit §GR](../../notes/emergent-codebook/phase-3-second-order-growth-precommit.md))
is a **NULL** on the gauge-free para-vs-random gate. Per the §GR pre-registration this
re-scopes the *growth-rule shape* (not "more knobs") → **R3: a predictive/successor-context
local growth**, with a latent-layer contingency. The decisive new finding: **the
paradigmatic ("substitutability") structure lives in the *subdominant* modes of SPPMI that
the global SVD isolates but the local iterative pull+repulsion cannot reach.**

**Classification:** Phase-3 structure-gate graduation experiment (the §GR growth redesign).
Floor (055–058) untouched.

---

## 1. What ran

`experiments/61` at **D=4096** on WikiText-2 (GPU), gauge-free gate
(`--gate gauge_free`), 5 seeds, SimLex≥5 (n≈40 paradigmatic pairs):
- **B′ + H_anti** sweep: `α_anti=1`, force-normalized `η_sep ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.4}`,
  growth-pull `α ∈ {0.05, 0.1, 0.2, 0.3}`, k-grid {1,2} (density-pinned) → 48 cells.
- Controls: uncentered **B** (8 cells), first-order **A** (4 cells) = 60 cells total.
- **CTRL-global reference** (the SVD-of-SPPMI oracle, `experiments/62`, D=4096):
  paradigmatic specificity **+0.1092, CI [0.082, 0.137]** — the global-whitening upper bound.

## 2. Result — NULL (0/60 cells pass the gate)

- **0 / 60** cells pass all three gate conditions (para-vs-random specificity CI > 0 AND
  collapse floor AND `corr(log cooc, drift) CI-hi < 0.15`).
- **11 / 60** cells show positive para-vs-random *and* no collapse — but **every one fails
  `corr<0.15`**: the surviving clustering is **co-occurrence-driven (collocational), not
  paradigmatic** — the exact Report-121 failure, persisting through the second-order operator
  + the anti-collapse force.
- **H_anti worked as designed** — it prevents the collapse (d_eff retained at low `η_sep`,
  lifted to ~1.0–1.5 at higher `η_sep`) — but preventing collapse does **not** make the
  clustering paradigmatic; it spreads everything roughly equally.
- **Best collapse-free specificity ≈ +0.021** vs the global SVD's **+0.109** — the local
  growth reaches at most ~1/5 of the achievable signal, cleanly.

## 3. Metric correction (honest, caught before banking)

The first pass of the gauge-free headline computed `para − glob − rand` (paradigmatic drift
minus the global-mean drift minus the matched-random arm). The matched-random arm **already**
controls for contraction, so subtracting `glob` *too* is a double-subtraction that biases the
headline **negative whenever the codebook contracts** — producing an apparent "the growth
anti-clusters king/queen," which was an artifact. **Fixed** (`experiments/61`, commit
`dc642a0`): gauge-free headline = `mean(para_drift) − mean(rand_drift)`, matching the §10 SVD
oracle; contraction is handled by the **separate** collapse floor. The corrected numbers above
were recomputed from the sweep JSON as `para−rand = biased_headline + glob`; the **verdict is
robust** to the correction (the `corr<0.15` gate, not the specificity sign, is what kills every
collapse-free positive cell), so per the user's call we accept the means-based NULL without a
re-run. *(This is the second metric issue this arc — both caught before any conclusion was
banked; the anti-rationalization discipline is doing its job.)*

## 4. The decisive finding — paradigmatic ≡ subdominant modes (a local-vs-global bound)

The same SPPMI matrix yields a clean paradigmatic signal under the **global** SVD (+0.109,
king/queen 0.222) but only a weak, collocational, or collapse-bound signal under the **local
iterative** `S'@G` growth (≤ +0.021 clean). The mechanism:
- `S'` is dominated by collocational/frequency structure (its leading modes); iterating
  `S'@G` runs toward those dominant modes (the smush).
- The **paradigmatic (substitutability) structure lives in the subdominant modes** — the global
  SVD isolates them as equal-footed orthogonal axes; the local Hebbian-style iteration cannot
  (it amplifies the dominant mode, not the subdominant ones).
- `H_anti` removes the dominant common-mode (prevents collapse) but does **not selectively
  promote** the subdominant paradigmatic modes — it spreads uniformly.

→ **Simple local pull + uniform repulsion cannot reach the paradigmatic modes.** This is a
genuine, empirically-bounded statement about the local-vs-global tension (the project's bet is
that the *local* path is right; this result says the *simple* local dynamics are insufficient
and a richer one is required).

## 5. Disposition (pre-registered §GR)

**R3 — a predictive / successor-context local growth.** Replace the symmetric co-occurrence
centroid with a rule that models *which contexts follow from a token* (a slowly-drifting FHRR
context vector / successor representation), reusing the graduated heteroassociative write
(055–058) and FHRR bind/bundle/permute. Predictive coding targets the higher-order/relational
structure the centroid misses, and *may* promote the subdominant substitutability axis locally.
- **Contingency (honest):** if R3 also cannot reach the subdominant structure, the result is
  telling us paradigmatic structure needs a **latent/hierarchical layer** (the PCN/SFA grounding
  said exactly this — paradigmatic structure lives in a latent layer *above* the flat code),
  which is a larger architectural step but still local (predictive-coding hierarchies are local).
- **NOT:** more `(α_anti, η_sep, k, α)` knobs on `S'@G` — that shape is now bounded.

## 6. Artifacts

- `experiments/61` (gauge-free gate fixed, force-normalized H_anti), `experiments/62` (oracle,
  CUDA-fixed), `notebooks/062_growth_redesign_sweep_wikitext_colab.ipynb` (the D=4096 run).
- Sweep JSON (gitignored / Drive): `_sweep_all.json`, `_oracle_d4096.json`.
- Banked side-finding (Report 122): high static cosine ≠ merged Hopfield basin at β=30.
- **Next:** ground + design R3 (predictive/successor local growth) — its own pre-commit.
