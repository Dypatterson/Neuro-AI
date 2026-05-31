# Production build: cue-space decorrelator + integrated surgical mechanism

> **Status:** the validated surgical mechanism (heteroassociative write + cue-space
> decorrelation) is now built as tested modules and reproduces the real-data transfer.
> Modules: `src/energy_memory/phase4/hetero_write.py` (7/7), `decorrelator.py` (4/4).
> Design: `notes/emergent-codebook/phase-4-heteroassociative-write-design.md`.

## What was built

- **`CueDecorrelator`** (`phase4/decorrelator.py`): a **batch ZCA whitening** transform
  `P = Σ^{-1/2}` over the cue subspace, learned offline from the closed cue covariance
  `Σ = K^H K / N` and applied uniformly. The rank-deficient null space (N<D leaves D−N
  zero eigenvalues) is **projected out** (inv-sqrt → 0), not amplified. `fit` records the
  cue off-diagonal correlation before/after so the decorrelation is auditable.
- The decorrelator composes with the existing `HeteroConsolidationBuffer` +
  `heteroassociative_write` + `recall_top_index` to form the integrated batch-offline
  surgical consolidation: fit the decorrelator on the closed cue buffer → apply → write.

## Validation (module reproduces the real-data transfer)

Wiring `CueDecorrelator` into the real-corpus harness (`experiments/50`, sparse cue
`observed=1`, D=512, N≈454) reproduces Report 050 Part 2 exactly:

| | store-as-is | raw write | **write + CueDecorrelator** | floor |
|---|---|---|---|---|
| sparse cue | 0.024 | 0.058 | **0.590** | 0.047 |

The production module matches the inline-whitening validation — the surgical mechanism
works on real data where the project's store-as-is fails.

## Anti-homunculus

The decorrelator is an **offline/batch statistic** of the cue distribution (a fixed
transform applied uniformly, no per-cue metric-gating) — exempt under the filter, the
same class as the project's existing batch `d_eff` (`torch_fhrr.py`) and C.3 kernel-trick
`eigvalsh`. The write retains its prior AH review (PASS-with-conditions, discharged): a
closed seed-fixed buffer, a precommitted swap-negative (never `sims.argmax`), a
`top_index_hits` read (no ΔE / min-over-branches). No new arbitration is introduced.

## Finding: the online anti-Hebbian (FEP) form is impractical here

An initial implementation used the **single-phase anti-Hebbian / FEP rule**
(`arxiv:2505.22749`, carded transfers-with-caveats): `P ← P + η(I − PΣP^H)P`, whose fixed
point is `Σ^{-1/2}`. It **does not converge practically** on real cues: the cue covariance
is **hugely ill-conditioned** (measured λmax≈198, condition number ~1e5 — the
frequent-token / shared direction concentrates variance), so the gradient/anti-Hebbian
iteration's convergence rate (set by the condition number) is impractically slow, and a
ridge large enough to stabilize the null space under-whitens the small signal directions.
The **closed-form batch whitening reaches the same fixed point instantly** and is
anti-homunculus-clean as a batch statistic, so it is the production form. An accelerated
online variant (Newton-Schulz, or a preconditioned anti-Hebbian rule) is a possible future
upgrade for a fully-online substrate, but is **not** required: the consolidation write is a
batch-offline pass by design (runtime error-driven updates are BANNED), where a batch
eigendecomposition is the natural and permitted tool.

## Honest scope

- Recovery on real data is **partial and rank-bounded** (0.59 at N≈D; decays past N=D) —
  a heteroassociative structure stores ~N≤D associations. The MESH-scaffold scaling story
  (card-pending) is the path to larger capacity.
- This is the **mechanism build + real-data validation**, not a phase-graduation result:
  the integrated pipeline has not been run at production scale (D=4096, full corpus) with
  the full done-gate panel. That scaled run is the next milestone (and the first that may
  want Colab).

## Next

- Scale run: integrated pipeline at D=4096 on a full wikitext slice, headline Recall@K via
  `top_index_hits` vs store-as-is, with the full control panel (the Phase-3 graduation gate).
- MESH-scaffold form to lift the rank-bounded capacity (card `pdf:mesh-2022` from primary).
