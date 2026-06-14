# Gate 0 (Frame A) — matched-world difference-in-differences

> **GATE / DIAGNOSTIC — does NOT graduate Phase 3 under any outcome.** Confirms the gauge finding (arm E) and selects the Frame B build path via the precommit branch table.

- precommit: `notes/notes/2026-05-28-gate0-frame-a-valid-control-precommit.md`
- operating point: D=4096 β=10.0 K=5 window=8 vocab=1002 landscape=64 cons_events=1000 θ′=default (requested both; DiD is θ′-invariant)
- Path C stack: lr_pull=0.1 lr_push=0.05 α_anti=0.01 repulsion=0.05
- corpus: `wikitext` (wikitext-2-raw-v1, vocab_cap=1000, eff_vocab=1002)
- n_seeds = 10 (graduation-scale ≥ 10)

## VERDICT: `G0->weak`

## Primary — consolidation corpus-specificity (DiD)

`DiD_s = [Recall_A − Recall_C] − [Recall_B − Recall_D]`

- mean DiD = **+0.0191 [-0.1207, +0.1590]** (n=10)
- clause 1 — CI > 0: **False**
- clause 2 — per-seed robustness ≥ 70%: **False** (6/10 positive)
- **DiD passes (both clauses): False**
- per-seed DiD: [-0.334, 0.1406, -0.2305, 0.0371, 0.0469, 0.3555, -0.0059, -0.041, 0.0527, 0.1699]

## Secondary reads

- (A)−(B) whole-pipeline corpus-sensitivity: +0.0068 [-0.1298, +0.1435] (CI>0: False)
- (A)−(C) consolidation benefit on real: +0.0320 [-0.0742, +0.1383] (CI>0: False)

## Gauge confirmation (arm E) — predicts Δ ≈ 0

- **4a** identity-permutation byte-identical to A[0]: **True**
- **4b** per-seed Δ = Recall(gauge control) − Recall(A) over 10 seeds: +0.0285 [-0.0673, +0.1243]; CI contains 0 (consistent with Δ=0): **True**
  - *precommit deviation:* fixed-atom×perm 4b replaced by per-seed gauge — E_π at fixed X ≠ identity recall, so the precommit's ≈0 prediction was mis-specified; see anchor note / Gate 0 precommit 4b amendment.
- gauge arm passes (4a ∧ 4b): **True** — if False → `G0->confound`, STOP and re-derive

## Pre-committed branch (all route to Frame B)

| verdict | meaning | next |
|---|---|---|
| G0->pass | consolidation is corpus-specific | build Frame B on the Path C stack; Γ1/Γ2/Γ3 stay closed |
| G0->weak | real but underpowered | escalate n before deciding |
| G0->null-cons | landscape carries structure, cons does not | redesign targeting consolidation, in Frame B |
| G0->dead | pipeline captures no structure | deeper redesign |
| G0->confound | gauge arm E failed | STOP, re-derive Part 1 |
