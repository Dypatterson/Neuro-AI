# Report 120 — MESH scaling decision: RESOLVED → defer MESH; factored low-rank H is the cost fallback

**Status:** RESOLVED — closes the open dense-H-vs-MESH scaling decision (open since Report 057).
**Classification:** **DRILL-DOWN / engineering measurement** (storage/cost of the graduated H), NOT a graduation experiment.
**Date:** 2026-05-31
**Decision spec:** [phase-4-heteroassociative-write-design.md:82-94,182-187](../../notes/emergent-codebook/phase-4-heteroassociative-write-design.md) · Report 057:72-80.
**Code:** `experiments/59_mesh_h_rank_scaling.py` (reuses exp-56 `write_H`/`write_read`, byte-identical).
**Builds on:** Reports 055/056/057/058 (the graduated, integrated, wiring-confirmed dense-H mechanism).

---

## Decision

**Ship dense `H`. Defer the MESH-style fixed scaffold — it is the wrong tool for the cost problem.** The graduated `H` is intrinsically low-rank, and a **rank-`r` factored / SVD-truncated `H` reproduces recall byte-identically** at 4–130× less memory, with **zero validation risk** (it is the same linear map stored as its dominant factors — not a new mechanism). When dense `H`'s O(D²) cost ever bites, the fallback is **factored `H`**, not MESH.

MESH-scaffold's actual contribution (avoiding the catastrophic-forgetting *capacity cliff* at N ≫ D, `pdf:mesh-2022`) is a **different property the project does not need yet**: it operates at N ≤ D, and `H` is low-rank because the *value space is small*, not because of capacity pressure. MESH becomes relevant only if a future regime needs to store far more distinct associations than D allows — not on the current path. It also remains primary-unopened and card-flagged as potentially conflicting with the emergent-codebook goal, so building it now would validate an unneeded architecture against a cost problem already solved.

---

## The load-bearing question

`H` is a delta-rule accumulation, so `rank(H) ≤ N` (buffer size). The docs *asserted* a low-rank/scaffold form could scale with #associations but **never measured whether `H` is actually low-rank at the operating scale**. If `H`'s effective rank ≪ N and recall survives truncation to it, a cheap rank-`r` factored `H` (`store U[D,r], σ[r], V[D,r]; apply H k = U(σ ⊙ (Vᴴk))`, O(rD)) is behaviorally near-identical and obviates MESH for cost. This report measures it.

## Results (in-sample, the full written memory; pooled over seeds)

| corpus | D | N (assoc) | recall (full H) | eff-rank (PR) | eff/D | num-rank | r99 (energy) | dense | factored@eff-rank |
|---|---|---|---|---|---|---|---|---|---|
| repo_sample | 1024 | 349 | 0.779 | **126** | 0.12 | 170 | 159 | 8.4 MB | **2.1 MB** |
| synthetic (L=8) | 2048 | 500 | 0.686 | **8** | 0.004 | 8 | 8 | 33.6 MB | **0.26 MB** |

**Recall-under-SVD-truncation (ratio to full recall):**

| truncation rank | repo_sample D=1024 | synthetic D=2048 |
|---|---|---|
| r = D/32 | 0.875 | 1.000 |
| r = D/16 | **1.000** | 1.000 |
| r = D/8 … D | 1.000 | 1.000 |

- **`H` is highly low-rank.** Effective (participation-ratio) rank is ≈ D/8 on real text and **exactly L=8** on the topic-toy — far below both N and D. 99% of spectral energy sits in the top ~159 (real text) / 8 (toy) components.
- **Recall saturates at rank ≈ D/16.** Truncating `H` to rank D/16 gives **byte-identical recall** (ratio 1.000) on both corpora; even D/32 retains 87.5% (real text) / 100% (toy). The truncation is the rank-`r` matrix closest to `H`, so a factored store applies *exactly* this `H_r` — the ratio=1.000 is measured, not assumed.
- **Rank is bounded by VALUE-space diversity, not D or N.** The topic-toy's `H` is rank-8 because there are only L=8 target atoms; real text's ~126 reflects the effective number of distinct target directions. Dense `D²` is enormously over-provisioned.

## Why this resolves dense-vs-MESH

The design doc (line 87) conflated two distinct things under "MESH-style": (a) *"a sparse/low-rank heteroassociative component"* and (b) *"a fixed random scaffold"*. The measurement separates them:

- **(a) Low-rank factored `H`** — the cheap win. Behaviorally identical (recall ratio 1.000 at r ≈ D/16), pure storage/compute optimization, **no validation, no mechanism change, no anti-homunculus surface** (the read still terminates in `top_index_hits`; the write is unchanged; only the stored matrix is its own SVD truncation). This is what should be reached for *if* cost ever bites.
- **(b) MESH fixed scaffold** — a *different* mechanism for a *different* problem (N ≫ D cliff). Unvalidated, primary-unopened, may conflict with emergent-codebook goals. Not needed for cost; not needed for the project's current N ≤ D regime.

So the cost concern that motivated the MESH question never required MESH at all.

## Cost context (when does anything bite?)

Dense `H` = `D²·8` bytes (complex64): D=1024 → 8 MB, D=2048 → 34 MB, **D=4096 → 134 MB** (the graduation scale — comfortably fine), D=8192 → 537 MB, D=16384 → 2.1 GB. The project graduated at D=4096 with no driver to go higher. **The defer-trigger has not fired.** If it does (D ≥ 8192, or many concurrent `H`'s), apply factored `H` first — at the measured ranks it is ~4–130× smaller and byte-identical.

## Anti-homunculus / fence

Pure offline measurement. SVD truncation is a storage analysis of an already-written `H`; no runtime metric gates anything; the recall read terminates in `top_index_hits` (Phase-5′ fence respected). The recommended factored form is a behaviorally-identical re-storage of the same linear map — it introduces no controller, no arbitration, no mechanism change.

## What remains / forward

- Factored `H` is **not built** — it is the recommended *fallback*, to be implemented only when dense cost bites (deferred by the same trigger, now with a known cheap answer). If desired, a `store_factored(H, rank)` + `recall_factored` path is ~30 LOC and unit-testable against dense `H` for bit-equality at the saturating rank.
- MESH-scaffold stays deferred and, per this analysis, **unneeded for cost** — revisit only under an N ≫ D capacity-cliff regime (not on the current path).

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python experiments/59_mesh_h_rank_scaling.py \
  --corpus-source repo_sample --D 1024 --seeds 2 --observed 2 \
  --out reports/120_mesh_scaling_decision/repo_D1024.json
PYTHONPATH=src .venv/bin/python experiments/59_mesh_h_rank_scaling.py \
  --corpus-source synthetic --D 2048 --N 500 --seeds 2 --observed 2 \
  --out reports/120_mesh_scaling_decision/synth_D2048.json
```

Result JSON is gitignored (`reports/**/*.json`); this `report.md` is the durable record.
