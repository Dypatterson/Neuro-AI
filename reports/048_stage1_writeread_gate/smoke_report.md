# Stage-1 write-then-read gate — preliminary smoke (DRILL-DOWN, not graduation)

> **Status:** drill-down smoke. **Not** a graduation result — 1 config, 5 seeds,
> and two refinements (below) are needed before the fork verdict is licensed.
> Harness: `experiments/48_stage1_writeread_gate.py`. Spec:
> `notes/emergent-codebook/phase-3-consolidation-write-design.md`.
> Raw: `smoke_hard_regime_5seed.json`.

## Experiment preamble

- **Active phase:** Phase 3 — consolidation-write sub-program (Stage-1 toy, pre-Phase-5′, fence down).
- **Headline per [phase-3-consolidation-write-design.md §Headline]:** value-codebook `top_index_hits` Selectivity-Δ, two-floor Wilson rule.
- **Required controls per [§Required controls]:** role-pairing-shuffle, no-negatives, random-codebook, perfect-cue upper bound.
- **Last verified result:** Reports 062–066 null; bundle-first tix=3072 (R1 read works).
- **Why now:** G-A/G-D are the cheapest-decisive adjudicators of the surgical-vs-rebuild fork.

## Config

`D=512, K=6 roles, C=24 content atoms, observed=1 (key-only cue), N_train=N_test=112, epochs=3, 5 seeds, CPU`. Chance = 1/C = 0.042.

## Regime validation (self-check)

`hard_regime = true`: store-as-is recoverability **0.157** vs perfect-cue ceiling **1.0** (chance 0.042). The toy reproduces the partial-recovery hard regime; the easy regime (`observed=K−1`) gives store-as-is ≈ 0.99 and is rejected by the self-check (confirms the R1 value-codebook read works when the cue is rich — consistent with bundle-first tix=3072).

## Result

| Read | true-role | role-shuffled | Δ | note |
|---|---|---|---|---|
| store-as-is | 0.157 | 0.046 | 0.111 | binding is role-selective even with no write |
| baseline-contrastive (self-mined neg) | 0.157 | 0.046 | 0.110 | **= store-as-is** |
| swap-negative (precommitted) | 0.157 | 0.046 | 0.110 | **= store-as-is** |
| pull-only (no negatives) | 0.157 | 0.046 | 0.110 | **= store-as-is** |
| random-codebook | 0.037 | 0.036 | 0.001 | collapses, as required ✓ |

**G-A frozen-refit:** native 0.157, refit **0.052 (≈chance)**, `refit − native = −0.10`; random-codebook control gap **0.015 (≈0)** ✓.

## Reading (preliminary)

1. **No readout defect (G-A).** The refit readout does **not** exceed native — it falls to ≈chance while native holds 0.157, and the random-codebook control gap is ≈0. So the partial failure is **not** hiding written-but-unreadable structure. This points **away** from the readout-fix branch (no GSBC un-park on this evidence) and **toward** the surgical-write program — consistent with the elimination-reached diagnosis.
2. **The write is inert in this regime.** All three write rules equal store-as-is to 4 decimals. Diagnostically (not a bug): with a 1-role cue the bottleneck is **scene-MHN retrieval**, which a *content*-codebook write cannot fix. The random-codebook arm correctly collapses, so the readout is sound.

## Two refinements this smoke surfaced (do before the fork verdict)

1. **Selectivity-Δ needs a vs-no-write anchor.** Raw Δ (0.11) is confounded by baseline binding role-selectivity (store-as-is already gives Δ=0.111). The write's contribution is `Δ(write) − Δ(store-as-is)` ≈ **0** here — that, not raw Δ, is the headline that isolates "did the write deposit structure." Add it as the reported headline.
2. **Target the bottleneck.** A content-codebook write can't move scene-retrieval-limited recovery. The write must either (a) operate where the bottleneck is, or (b) the toy must isolate the content-cleanup read (richer cue) so the write has a lever. Re-run across a cue-richness sweep so the write is measured where it *can* act.

## Next gates (cards ready)

- **G-0 (BTSP key-only):** cards in `docs/ground-truth/source_cards/2026-05-30-gate-cards.md` (BTSP rule `Δw=P(1−w)−Dw`, plateau-gated/exogenous = anti-homunculus clean; transfers-with-caveats — substrate is sparse-binary, the FHRR port is the question).
- **G-B (swap-negative):** behind the bundle-first Cell-A/Cell-B margin-existence gate; A/B vs phase3b Fail-Rate-1.0.
- **G-C (N-primary capacity surface):** N↓ primary, D↑ secondary, D↓ predicted-null; D grid 512/1024/2048/4096.
