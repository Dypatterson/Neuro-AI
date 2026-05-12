---
date: 2026-05-08
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Consolidation Geometry Diagnostic

A diagnostic probe applying the regime-checking framework from Vangara & Gopinath's "The Geometry of Consolidation" (NeurIPS 2026 submission) to the personal-ai codebook. Not a build spec — a diagnostic instrument that can run alongside Phase 3 to tell us whether consolidation is safe on each atom's neighborhood before the update fires.

**Source paper:** [github.com/niashwin/geometry-of-consolidation](https://github.com/niashwin/geometry-of-consolidation)

> **Updated 2026-05-09:** added cap-coverage as a complementary identity measurement, strengthened the calibration recommendation from "would tighten the mapping" to "recommended pre-Phase-3 spike," and noted the connection to the headline metric. See `notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md` for the broader diagnostic-stack synthesis.

## What the paper gives us

The paper proves a lower bound on identity-retrieval error after any consolidation operator replaces $n$ cluster members with $m < n$ representatives on the unit sphere:

$$\varepsilon_{\text{id}} \geq 1 - c_1 \, m \left(\frac{\theta'}{\bar{d}}\right)^{d_{\text{eff}}/2}$$

Two numbers decide whether consolidation preserves retrievability:

- $\bar{d}$ — mean within-cluster cosine distance (how spread out the cluster is)
- $d_{\text{eff}}$ — effective dimension of the cluster's covariance spectrum (participation ratio)

A regime boundary at $\bar{d} = \theta'$ (where $\theta' = 1 - \theta$, the retrieval threshold slack) separates two worlds:

- **Tight regime** ($\bar{d} < \theta'$): any reasonable consolidation operator works. Centroid averaging is near-optimal. The paper's 17,813-cell sweep shows this is where real English text lives.
- **Spread regime** ($\bar{d} \geq \theta'$): consolidation destroys identity. No operator — including an oracle — can recover what the full cluster provided.

## Why this matters for the personal-ai codebook

The Phase 3 consolidation event is structurally a consolidator in the paper's sense. Every K steps, experiences accumulated in the buffer get processed, and atoms get updated: `t_new = (1 - α) × t_old + α × target`. The refined atom is a compressed representative of the experiences that shaped it. The theorem constrains how much identity survives that compression.

Four specific mappings:

| Paper concept | Codebook analogue |
|---|---|
| Cluster $\mathcal{C}$ | An atom's recent context bags — the set of co-occurrence bundles from experiences involving that atom |
| Representative $r$ | The atom's current hypervector after consolidation |
| Retrieval threshold $\theta$ | Hopfield retrieval sharpness, controlled by $\beta$. Higher $\beta$ → sharper retrieval → effectively higher $\theta$ → smaller $\theta'$ → tighter regime boundary |
| Identity-retrieval error $\varepsilon_{\text{id}}$ | Fraction of stored Hopfield patterns containing this atom that fail to retrieve after the atom drifts |

The Hebbian pathway (drift toward context-bag centroid) is doing exactly what the paper says is near-optimal in the tight regime. The error-driven pathway earns its keep only in the spread regime — on atoms whose context neighborhoods span distinct clusters.

## The diagnostic: two numbers per atom

After each consolidation event, compute for each atom $t$ that was updated:

### $\bar{d}_t$ — mean within-cluster cosine distance of the atom's context bags

```
context_bags = atom.context_bag_history[-N:]  # rolling window
pairs = all_pairs(context_bags)
d_bar = mean(1 - cosine_similarity(a, b) for a, b in pairs)
```

This measures how spread out the atom's recent usage contexts are. Small $\bar{d}$ means the atom appears in consistently similar contexts; large $\bar{d}$ means it's being used in diverse or contradictory contexts.

### $d_{\text{eff},t}$ — effective dimension of the atom's context-bag covariance

```
context_matrix = stack(context_bags)  # shape [N, D]
cov = covariance(context_matrix)
eigenvalues = eigenvalues_of(cov)
d_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)
```

This measures how many independent directions the atom's context neighborhood uses. Low $d_{\text{eff}}$ (1–5) means the contexts vary along one or two axes; high $d_{\text{eff}}$ means the variation is spread across many dimensions.

### Regime classification

For a given retrieval sharpness $\beta$, compute the effective $\theta'$ (this mapping is approximate — modern Hopfield's softmax is a smooth relaxation of hard threshold, so $\theta'$ needs empirical calibration against the softmax temperature; a reasonable starting point is $\theta' \approx 1/\beta$ for $\beta$ in the working range {0.01, 0.1, 1.0}):

```
theta_prime = 1.0 / beta  # starting approximation; refine empirically
regime = "tight" if d_bar < theta_prime else "spread"
```

## What to do with the regime classification

The diagnostic doesn't change Phase 3's mechanics — it annotates them. Three uses:

**1. Validate the Hebbian-dominance prediction.** On atoms classified as tight-regime, the paper predicts the Hebbian pathway (centroid drift) is already near-optimal and the error-driven pathway adds nothing. Track this: after consolidation, compare retrieval quality improvement on tight-regime atoms that got Hebbian-only updates vs. those that got error-driven updates. If the paper is right, the delta should be negligible on tight atoms and meaningful only on spread atoms.

**2. Sharpen the bimodality signal.** Phase 3 already tracks bimodality via dip test or GMM/BIC. The regime diagnostic adds a retrieval-native criterion: an atom can be bimodal (two distinct usage contexts) but still tight-regime (both contexts are close enough that a single centroid covers them). The dip test fires; the regime test doesn't. In that case, splitting would waste a codebook slot. Conversely, an atom can be unimodal with high $\bar{d}$ — a single diffuse cloud rather than two clusters — and still be in the spread regime where consolidation breaks. The $\bar{d} < \theta'$ test catches cases the dip test misses and filters cases the dip test over-flags.

**3. Early warning for Phase 3 failure modes.** If most atoms land in the spread regime after initial codebook population, that's a structural signal: the codebook's atoms are too coarse (each atom is trying to cover too much semantic territory) and Phase 3 dynamics will struggle regardless of learning rates. This surfaces the "retrieval signal richness" concern from the Phase 3 deep dive in geometric terms before running the full training loop.

## Cap-coverage as a complementary identity measurement

*(Added 2026-05-09.)*

The paper's Section 8 makes a distinction the regime diagnostic doesn't capture: **identity error** and **cap-coverage error** are different metrics that can differ by 10× on the same data. Identity = "did the right representative fire?" Cap-coverage = "did the cap of any representative cover the source?" The paper documents downstream EM on QA tasks tracking cap-coverage rather than identity, with the two coming apart on different corpora.

For the codebook setting, the analogue is:

| Paper measurement | Codebook analogue |
|---|---|
| Identity error | Fraction of stored Hopfield patterns containing this atom that fail to retrieve as the top-1 result after the atom drifts |
| Cap-coverage error | Fraction of stored Hopfield patterns containing this atom whose max cosine similarity to any stored pattern falls below threshold $\theta$ after the atom drifts |

The architectural implication: meaning in this project is supposed to live in the retrieval *neighborhood*, not in exact-token recovery (this is the contextual-completion commitment from `overview.md`). Identity error captures the wrong axis. Cap-coverage error captures the right one. An atom that drifts in a way that breaks identity but preserves cap-coverage has not necessarily broken anything the architecture cares about; an atom that drifts in a way that breaks cap-coverage has.

The regime diagnostic predicts when consolidation is *safe* (tight regime → Hebbian-near-optimal). Cap-coverage error measures whether consolidation actually *was* safe in the sense the architecture cares about. They are complementary, not redundant: the regime test is geometric prediction, cap-coverage is empirical outcome.

Implementation note: cap-coverage error is also being added to the Phase 2/3 per-retrieval metric set (see `PHASE_2_SPEC.md`'s Settled-state diagnostics — "Max stored-pattern cosine at convergence" enables post-hoc cap-coverage at any threshold). The Phase 2 measurement is on retrieval queries; the Phase 3 measurement here is on stored patterns containing each atom after consolidation. Same metric, different referent.

## What this does NOT do

- It does not replace the bimodality tracker. The two are complementary — bimodality detects polysemy; the regime diagnostic detects consolidation safety. An atom can be polysemous but tight (two close meanings, like "run a program" / "run a script") or unimodal but spread (a function word appearing in maximally diverse contexts).
- It does not tell you what the right $\theta'$ is for Hopfield softmax retrieval. The $\theta' \approx 1/\beta$ mapping is a starting point. Calibrating it empirically — running the paper's identity-probe protocol on synthetic clusters stored in the Hopfield layer at each $\beta$ — would tighten the mapping. **As of 2026-05-09 this calibration is recommended as a pre-Phase-3 spike**, not optional. The regime classifier is load-bearing for several Phase 3 decisions (which atoms get which pathway, what the bimodality test should filter); building those decisions on an uncalibrated approximation is a foreseeable problem worth heading off.
- It does not prove the theorem transfers to FHRR. The paper is proved for real-valued unit-norm embeddings under cosine threshold. FHRR atoms are unit-modulus per dimension (complex unit circle), and Hopfield uses softmax energy, not a hard threshold. The structural form should transfer (cap-volume arguments are geometric), but the calibrated constants won't. Measuring the project's own $c_1$ is a Phase 3 extension, not a prerequisite.

## Implementation sketch

Lightweight. Runs inside the existing consolidation event with no architectural changes.

```python
# After each consolidation event, for each updated atom t:

def regime_diagnostic(atom, beta, window=20):
    """Compute regime classification for a single atom."""
    bags = atom.context_bag_history[-window:]
    if len(bags) < 3:
        return None  # not enough data

    # d_bar: mean pairwise cosine distance
    sims = []
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            sims.append(cosine_similarity(bags[i], bags[j]))
    d_bar = 1.0 - mean(sims)

    # d_eff: participation ratio of context-bag covariance
    matrix = torch.stack(bags)  # [window, D]
    matrix = matrix - matrix.mean(dim=0)
    # For complex FHRR: use real and imaginary parts,
    # or compute on magnitudes — needs empirical comparison
    cov = (matrix.T @ matrix) / (len(bags) - 1)
    eigvals = torch.linalg.eigvalsh(cov.real)  # approx for complex
    eigvals = eigvals.clamp(min=0)
    d_eff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()

    # regime boundary
    theta_prime = 1.0 / beta
    regime = "tight" if d_bar < theta_prime else "spread"

    return {
        "d_bar": d_bar.item(),
        "d_eff": d_eff.item(),
        "theta_prime": theta_prime,
        "regime": regime,
        "ratio": d_bar.item() / theta_prime,  # <1 is tight, ≥1 is spread
    }
```

Log results per consolidation event. Aggregate statistics to track:

- Fraction of atoms in tight vs. spread regime over training
- Whether the fraction shifts as the codebook stabilizes
- Correlation between regime classification and retrieval quality improvement from consolidation

## When to build this

Not blocking. This is a Phase 3 diagnostic, not a Phase 3 prerequisite. The cleanest insertion point is after Phase 3's consolidation event is running and Tier 1 sanity checks pass — at that point the context-bag history is populated and the diagnostic has data to work with. Could be added as a lightweight logging pass inside the consolidation event with zero changes to the update mechanics.

The empirical $\theta'(\beta)$ calibration (paragraph above under "What this does NOT do") *is* recommended pre-Phase-3. Concretely: run the Geometry of Consolidation E1 protocol on the project's FHRR substrate — generate synthetic clusters with controlled $(d_\text{eff}, \bar{d}, \theta)$, store them in the Hopfield layer at each $\beta \in \{0.01, 0.1, 1.0\}$, measure the empirical retrieval-success boundary, and back out the calibrated $\theta'(\beta)$ mapping. Roughly two days of work; not blocking Phase 2 Session 3 but should land before Phase 3 starts.

## Open questions this diagnostic might answer

- Does the tight/spread boundary shift as the codebook stabilizes? (If atoms drift from spread to tight over training, that's the Hebbian pathway doing its job — pulling contexts closer together until consolidation becomes safe.)
- Is there a correlation between an atom's frequency and its regime? (High-frequency function words might be spread despite unimodal context distributions; low-frequency content words might be tight.)
- Does the regime classification predict which atoms benefit from error-driven updates? (The paper's headline finding is that centroid/Hebbian dominates in the tight regime. If the same holds here, the error-driven pathway can be gated by regime classification — only fire on spread-regime atoms — saving compute and reducing noise on atoms that don't need it.)
- At what $\beta$ does the regime boundary start excluding atoms that were previously tight? (This connects to the Phase 5 question about temperature-as-geometry replacing explicit $\beta$.)
- *(Added 2026-05-09)* Does the regime classification predict cap-coverage outcomes? Specifically: do tight-regime atoms have lower cap-coverage error than spread-regime atoms after consolidation, holding identity-error roughly constant? If yes, the regime classifier can be used to *predict* which atoms will preserve neighborhood meaning, not just which will preserve identity.

## Relationship to existing Phase 3 diagnostics

This diagnostic sits alongside — not above or below — the existing Phase 3 evaluation tiers:

- **Tier 1 (sanity):** regime diagnostic adds a geometric pre-check. If most atoms start in the spread regime, Tier 1 is unlikely to pass.
- **Tier 2 (distributional structure):** regime diagnostic explains *why* distributional structure emerges (tight-regime atoms can be safely consolidated toward their semantic neighborhoods) or fails to emerge (spread-regime atoms resist consolidation).
- **Bimodality tracking:** regime diagnostic adds a retrieval-native filter. See "sharpen the bimodality signal" above.
- **Shuffled-token control:** the regime diagnostic should show similar $d_{\text{eff}}$ and $\bar{d}$ distributions under shuffled and standard runs (these are properties of the corpus co-occurrence structure, not of learned representations). If the *regime classification* differs between shuffled and standard, that's a signal the codebook is actively reshaping its context geometry — which would be a Tier 2+ finding.
- *(Added 2026-05-09)* **Headline metric stratification:** the Phase 3 headline metric (Recall@K on masked-token contextual completion, regime-stratified, vs. shuffled-token control) uses this diagnostic's regime classification as the stratification axis. The "regime-stratified" qualifier is what this file produces. Strong performance only in tight-regime atoms is a different story from strong performance across both regimes — the stratification is what lets the headline metric distinguish them.
