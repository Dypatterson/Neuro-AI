# Research Brief A — PAM / JEPA-style abstract prediction for Phase 5 graduation

Date: 2026-05-20
Mission: find a replacement for the structurally-degenerate role-fidelity
metric `f_i = mean(1 - |G_jk|)` that collapses to ~0.984 at D=4096 because
FHRR crosstalk is `~1/sqrt(D)` regardless of pattern content.

## Angle

The project is stuck because its headline weighting function has zero
per-atom variance: it measures *crosstalk magnitude*, which is a property
of the substrate dimension, not of any individual atom. PAM (Dury 2026a,b)
and LLM-JEPA (Meta 2025/26) reframe the underlying primitive: rather than
score atoms by *pairwise similarity in the substrate*, score them by
**predictability under a learned predictor trained on co-occurrence**.
Co-occurrence is a per-atom (per-state) signal — every atom has its own
neighborhood of true associates — so the resulting score has structural
per-atom variance by construction.

This is exactly the shape of fix the situation calls for: a measurement
of a local dynamic (a learned predictor's settled output near a stored
state), not an arbitration over them.

## Key findings

### 1. PAM (Dury 2026a) — the core formula already published, with numbers

Source: <https://arxiv.org/abs/2602.11322> (Dury 2026a, "Predictive
Associative Memory: Retrieval Beyond Similarity Through Temporal
Co-occurrence", Feb 2026).

- Drop-in metric the paper validates against cosine:
  - **Discrimination AUC** for the binary question "were these two states
    experienced within the same temporal window?". Computed per-query as
    a Wilcoxon-Mann-Whitney U-statistic over (positives = temporal
    co-occurrences, negatives = non-associates), then macro-averaged.
  - PAM: AUC = 0.916. Cosine: 0.789. **+0.13 absolute AUC purely from
    swapping the score function.** This is the exact "headline cannot
    discriminate atoms" symptom the project is currently hitting, in a
    published baseline.
- Architecture is minimal: 4-layer MLP residual predictor, 128→1024→
  1024→1024→128, GELU, LayerNorm, **2.36M params total**. Trains in
  hours on the synthetic benchmark.
- Loss is InfoNCE with in-batch negatives, stop-gradient on the target
  branch, temperature annealing 0.15 → 0.05.
- The "Inward JEPA" is the relevant variant for this project: it
  predicts associatively reachable *stored* states from the current
  state. That maps directly onto a stored-atom Hopfield substrate.

### 2. PAM-CC (Dury 2026b) — capacity-constraint as a consolidation knob

Source: <https://arxiv.org/abs/2603.18420> (Dury 2026b, "From Topic to
Transition Structure", Mar 2026).

- Same predictor shape (4-layer MLP residual, 1024→1024, learned
  residual mix `f(x) = norm(α·x + (1−α)·g(x))`, α converges to 0.756,
  29.4M params), scaled up to a 373M-pair corpus.
- **The "capacity constraint" is not an explicit term in the loss.** It
  is the natural under-fitting that occurs when model capacity is too
  small relative to the co-occurrence cardinality of the dataset
  (42.75% training accuracy at convergence). The paper frames this as
  *the* consolidation mechanism: compression-by-undercapacity is the
  mathematical analog of hippocampal replay producing stable concepts.
- This is highly relevant to the project's existing replay/consolidation
  story — it gives a sharp, single-knob operationalisation of "consolidate"
  that is purely local-geometric (just a small predictor that cannot fit
  everything).

### 3. LLM-JEPA (Sobal et al., arXiv 2509.14252) — the predictor reuses the encoder

Source: <https://arxiv.org/html/2509.14252v2> (ICLR 2026).

- Loss: `L = L_LLM + λ · d(Pred(Enc(Text)), Enc(Code))`.
- The predictor is **not a separate network** — it reuses the LLM with
  appended `[PRED]` tokens and a block-causal attention mask; the last
  predictor-token embedding is the prediction. For a Hopfield substrate
  this means the predictor can be a tiny re-projection through the same
  bound-state space, not a new module.
- Crucially: they fall back to cosine `d`, but the *interesting*
  diagnostic is the **SVD spectrum of `Enc(Text) - Enc(Code)`**. After
  training, the singular values are systematically smaller, and the
  mapping becomes approximately linear (low residual under
  least-squares fit). That is a *spectral fidelity* measure that does
  not depend on the magnitude of pairwise crosstalk — it depends on the
  rank-structure of the prediction error.

### 4. JEPA = Energy-Based Model with Hopfield ancestry

Sources:
- <https://arxiv.org/pdf/2306.02572> (Les Houches lecture notes on EBMs/JEPA).
- <https://arxiv.org/abs/2008.02217> (Ramsauer et al., Modern Hopfield).

JEPA's "energy" is literally `E(x,y) = d(Pred(Enc(x)), Enc(y))`. Modern
Hopfield retrieval is `softmax(β · S · x)` where S is the stored pattern
matrix. These are the same family: a predictor that lands you near a
stored state is a one-step Hopfield update with a learned readout. The
project already has the Hopfield half; PAM provides the missing
"predictor that lands you near *the right* stored state" half.

### 5. Successor-representation lineage

Sources:
- <https://elifesciences.org/articles/80680> (Fang et al., neural SR
  learning rules, 2023).
- PAM explicitly cites Dayan 1993 (successor representation) as a
  predecessor.

The SR `M(s,s') = E[Σ γ^t 𝟙(s_t = s') | s_0 = s]` is bounded in `[0, 1/(1-γ)]`
and is per-state by construction. PAM is essentially a continuous,
learned, embedding-space SR. This gives a second, simpler family of
candidate metrics with clean bounds.

## Promising leads

- **PAM Discrimination AUC** as the new headline. It is exactly the
  binary discrimination metric the project's K-branch state_divergence
  was trying to be, but computed in a space the substrate cannot
  trivially saturate. Cosine baseline already published (0.789), so the
  project can report PAM-style numbers against it directly.
- **Capacity-constrained predictor as the consolidation mechanism.** A
  single small MLP (29M or less) trained InfoNCE on co-occurring atom
  pairs from the existing replay buffer; the under-fitting *is* the
  consolidation. No supervisor, no thresholds, no if/then.
- **SVD spectrum of (Pred(query) − target) as a fidelity diagnostic.**
  Per-atom: take all queries that should reach atom `i`, compute the
  residual vectors, look at their top singular value. Atoms with a
  collapsed residual spectrum are well-bound; atoms with a flat
  spectrum are not.
- **Inward JEPA over W=4 role-filler windows.** For a stored atom with
  fillers `f_1..f_4` bound to roles `r_1..r_4`, train the predictor to
  map (unbound query of any 3 roles) → (the 4th filler's embedding).
  Per-atom variance comes from how predictable each atom's 4th filler
  is given the other 3 — this is a structural property of the atom's
  *content*, not of D.
- **PAM cross-boundary Recall@K.** The paper reports Recall@20 = 0.421
  where cosine = 0. If the project's substrate has stored atoms whose
  cosine retrieval has saturated, predictor-based retrieval can pull
  out structure the cosine cannot see. Direct probe of whether the
  D=4096 ceiling is a substrate ceiling or a *metric* ceiling.

## Concrete ideas for this project

### Idea 1 — Replace `f_i` with predictor-residual energy (PASS-likely, local-geometric)

Train a tiny MLP predictor `g_φ : R^D → R^D` on co-occurring (cue,
target) pairs drawn from the existing W=4 role-filler windows. For
each stored atom `i`, define

    f_i_new := exp(-||g_φ(cue_i) - z_i||^2 / σ^2)

with `z_i` the atom's substrate vector and `cue_i` a held-out
partial-binding cue. This is a per-atom quantity that varies because
different atoms have different prediction error under `g_φ` — a
property of the *atom's relationship to the rest of the substrate*,
not of D. Bounded in (0, 1]. Plays the same arithmetic role as the
old `f_i` in the weighted retrieval but cannot saturate at `1/sqrt(D)`.

Anti-homunculus check: PASS. No supervisor decides; `g_φ` is just a
function fixed at consolidation time, `f_i` is just a distance under
that function. Same shape as Hopfield energy with a learned kernel.

### Idea 2 — InfoNCE consolidation as the Phase-5 substitute for ad hoc death dynamics (PASS-likely)

Replace the A+B+A1+A1' continuous death rules with: at the end of each
phase, train a small `g_φ` on the replay buffer using PAM's exact
InfoNCE recipe (in-batch negatives, temperature anneal 0.15→0.05,
stop-gradient targets). Atoms whose post-consolidation predictor
residual exceeds a *learned* (not hand-set) percentile are pruned. The
"decision to die" is a position on a fitted geometric distribution, not
an if/then rule.

Anti-homunculus check: PASS. The threshold is a percentile of a fitted
distribution; the death is a local geometric event ("this atom is far
from the predictor manifold"). No module decides between two
mechanisms.

### Idea 3 — Per-atom SVD-residual fidelity (PASS-likely)

For each atom `i`, sample K cues that should retrieve it, run them
through `g_φ`, stack the residuals `R_i ∈ R^{K×D}`, compute the top
singular value `σ_1(R_i)`. Define

    f_i_svd := 1 / (1 + σ_1(R_i))

This is the spectral analog of "how cleanly does the predictor land on
this atom across its cue variants". Bounded in (0, 1]. Per-atom
variance is intrinsic. Connects directly to the LLM-JEPA SVD
diagnostic.

Anti-homunculus check: PASS. SVD is a local geometric operation on a
single atom's prediction residuals. No arbitration.

### Idea 4 — Headline metric swap: PAM Discrimination AUC (PASS-likely)

Replace K-branch state_divergence with: macro-averaged Wilcoxon-Mann-
Whitney AUC for the binary question "do roles `r_a` and `r_b` co-occur
in some stored atom?". Score function is `-||g_φ(z_a) - z_b||`. Cosine
baseline gives the project's current AUC; the PAM predictor gives the
replacement AUC. If the gap is comparable to PAM's published 0.789 →
0.916 swing, Phase 5 graduates.

Anti-homunculus check: PASS. AUC is computed over labels and scores;
the *score* is a single local distance, not a vote.

### Idea 5 — Capacity-constrained replay as a sharp consolidation knob (PASS-likely)

Adopt Dury 2026b's framing explicitly: rather than scaling the
predictor to fit the corpus, fix a small predictor (e.g. 2-layer,
hidden 512) and *target* an undertrained accuracy (~40%). The
under-fitting forces the predictor to compress across redundant
co-occurrences, producing concept-level structure as an emergent
geometric property. This replaces the project's current consolidation
story with a single-hyperparameter knob (predictor capacity) whose
effect is *measurable as a spectrum collapse* rather than a heuristic.

Anti-homunculus check: PASS. The capacity is a fixed scalar; the
"decision to consolidate" is the gradient-descent dynamics of a
single network. No module.

## Surprises

- **PAM's per-atom-variance fix is essentially free of hyperparameters
  the project doesn't already have.** A 2.36M-param MLP, InfoNCE, in-
  batch negatives — every component is standard. The *insight* (predict
  from co-occurrence, score by predictor distance) is the load-bearing
  part, not the recipe.
- **PAM's cosine baseline (0.789 AUC) and the project's current
  saturation are the same failure mode in two different problems.** PAM
  was literally written to address "cosine cannot discriminate states
  that should be associatively distinct" — a structurally near-identical
  failure to "FHRR crosstalk cannot discriminate atoms by role
  fidelity". The fix should transfer.
- **LLM-JEPA uses cosine `d` and still wins** — but their interesting
  *diagnostic* is the SVD of `Enc(Text) - Enc(Code)`. This implies the
  per-atom variance the project needs may live in the *residual rank
  spectrum* of a learned predictor, not in a scalar distance. That is a
  fundamentally different shape of fidelity measure and worth a separate
  probe (Idea 3).
- **The capacity constraint in Dury 2026b is not a regulariser; it is
  the consolidation mechanism.** Replay + insufficient capacity =
  compressed concept structure. This dissolves the project's distinction
  between "replay" and "consolidation": they are the same gradient
  descent, framed by capacity.
- **Modern Hopfield retrieval is one step of a JEPA-style predictor with
  a fixed kernel.** The project already has the substrate; it is
  missing the *learned* readout that gives per-atom predictability
  variance. The two halves compose without architectural surgery.

## Source list

- <https://arxiv.org/abs/2602.11322> — Dury 2026a, PAM
- <https://arxiv.org/html/2602.11322v1> — PAM HTML (technical details)
- <https://arxiv.org/abs/2603.18420> — Dury 2026b, PAM-CC / concept discovery
- <https://arxiv.org/html/2603.18420> — PAM-CC HTML (technical details)
- <https://arxiv.org/html/2509.14252v2> — LLM-JEPA (ICLR 2026)
- <https://arxiv.org/pdf/2306.02572> — EBM/JEPA lecture notes (Les Houches)
- <https://arxiv.org/abs/2008.02217> — Ramsauer et al., Modern Hopfield
- <https://elifesciences.org/articles/80680> — Fang et al., neural SR
- <https://arxiv.org/pdf/2506.14373> — Discrete-JEPA (less directly relevant)
- <https://arxiv.org/html/2512.14709v1> — Attention-as-binding VSA view
- <https://proceedings.neurips.cc/paper/2021/file/d71dd235287466052f1630f31bde7932-Paper.pdf>
  — Learning with HRRs (NeurIPS 2021)
