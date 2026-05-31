# Research angle: x

## Searches
- a1
- a2
- a3
- a4
- a5
- a6
- a7
- a8
- a9
- a10

## Key findings
- (NEW) Chaudhry/Tang/Lu Learning Sequence Attractors with Hidden Neurons (2404.02729): LOCAL three-factor rule, error residual stops writing at a margin, hidden neurons one-hot encode transitions; writes heteroassociations. Opened HTML.  <https://arxiv.org/abs/2404.02729>
- (NEW) SURPRISE: covariance-PC (pcbi.1010719) writes cross-pattern structure but converges to a flat HYPERPLANE, reproducing the flat/zero-margin failure. Opened PLOS.  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719>
- (NEW) SURPRISE: Transformers Variable Binding (2505.20896) is separate-subspace allocation after a long plateau. Opened HTML.  <https://arxiv.org/abs/2505.20896>

## New sources
- **Learning Sequence Attractors with Hidden Neurons** (2024) <https://arxiv.org/abs/2404.02729> — Error-gated three-factor rule writing heteroassociations via hidden neurons.
- **Recurrent PC AM via covariance learning** (2023) <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719> — Covariance write collapses to a flat HYPERPLANE absent a margin.
- **Transformers Learn Variable Binding** (2025) <https://arxiv.org/abs/2505.20896> — Separate-subspace allocation after a plateau.

## Concrete ideas

- **Error-gated three-factor rule (2404.02729) on the FHRR toy as Stage-1 WRITE; hidden layer one-hot-encodes the role; writing stops at correctness.** [local-dynamic; Stage-1 toy]
  - writes structure: Error residuals write a margin not a flat centroid; hidden one-hot units allocate a basin per role.
  - test: Report-068 toy ~4 roles; basin-membership/top_index_hits (not top-1, Report 066) above chance; controls ablate-hidden, shuffled-pairing, lower-D.

## Surprises


## Leads

