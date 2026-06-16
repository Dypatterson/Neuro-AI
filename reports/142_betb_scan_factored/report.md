# Report 142 — Bet B / SCAN Stage 1 iterate 2: architectural factorization → consolidation becomes LOAD-BEARING (robust partial positive)

**Status:** robust **partial positive** — the first time in the program a brain-shaped consolidation
is load-bearing in a regime where the simple method fails. Below the pre-registered graduation bar
(2026-06-15). **Experiment:** `experiments/89_betb_scan_factored.py` (+ probes). **Builds on:**
[Report 141](../141_betb_scan_consolidation/report.md) (vanilla consolidation NULL) and
[Report 140](../140_betb_scan_gap/report.md) (Stage 0 gap). **Charter:**
[notes/betb-scan-stage1-consolidation-precommit.md](../../notes/betb-scan-stage1-consolidation-precommit.md).

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer, discriminating regime — the thesis test.
- **Headline:** add-`jump` exact-match, consolidation − baseline, n=5. Pre-registered PASS =
  Δ CI-disjoint > 0 **AND** absolute ≥ 0.30. **Result: Δ robust (5/5) but absolute below 0.30 → PARTIAL.**
- **Why now:** Report 141 showed post-hoc consolidation can't manufacture composition because role and
  identity are entangled absent a factored representation. This iterate builds the factorization in.

## The fix tested (architectural role⊕filler)

`embedding = [role ; filler]`; **encoder runs on ROLE only** (structure is role-driven); the decoder picks
*which* verb-action from the attended **filler** (identity), while *when* to emit a verb + the turns come
from role. So identity is read from a SEPARATE channel than structure → aligning `jump`'s role need not break
its identity. Consolidation = the Report-141 offline pass, but aligning only the **role** sub-embedding of the
peer set (computed in code, asserted `{jump,walk,run,look}` — anti-homunculus BUILD CONDITION 1).

## Result

| | baseline (arch only) | + consolidation | Δ |
|---|---|---|---|
| mean (n=5) | 0.087 | **0.166** | **+0.079** |
| per-seed Δ | — | — | +0.097, +0.019, +0.092, +0.015, +0.174 (**5/5 positive**) |

Supporting probes (n=1, 30 epochs):
- **Competence** (in-distribution): **1.000** — the factored architecture learns SCAN perfectly (not crippled).
- **Premise** (swap `jump`'s role←`walk`, KEEP its filler, real `I_JUMP` gold): **0.26** — vs **0.000** for the
  vanilla model's role-replace (Report 141). The factorization genuinely separates role from identity… *partially*.
- **Vanilla contrast:** the identical consolidation on the non-factored model nulled (0.003→0.001, Report 141).

## Reading (honest scope)

- **The robust, real claim:** within a factored architecture, a role-alignment **consolidation is
  load-bearing** — it lifts compositional generalization ~2× in **5/5 seeds** (Δ CI-disjoint > 0), where the
  *same consolidation* on the vanilla architecture produces *nothing*. This is the **first robust evidence for
  the Bet-B thesis-shape**: a restructuring consolidation manufactures composition the simple method can't.
  The architecture alone (baseline 0.087) does NOT solve it — consolidation does real work on top.
- **Why it's PARTIAL, not a graduation:**
  1. Absolute level **0.166 ≪ 0.30** (pre-registered bar). High seed variance (baseline 0.015–0.217).
  2. The factorization is **leaky** — the premise ceiling is only **0.26** (a clean factorization would be ~1.0).
     The leak is consistent with attention-blur contaminating the filler channel (`ctx_fill` mixes the verb's
     filler with function-word fillers), so identity is only partially recovered. Consolidation climbs toward
     this leaky ceiling; it cannot exceed it.
  3. The architecture is a **designed inductive bias** (not emergent). Known architectural SCAN fixes
     (equivariance) reach 0.8+ *alone*; we are weaker — so this is NOT a match to a known fix, but the claim is
     narrow (about the consolidation *mechanism* being load-bearing), not about beating SCAN.

## Disposition — the ceiling is the next lever

The wall from Report 141 has *moved*, not been re-hit: role/identity went from inseparable (premise 0.000) to
partially separable (0.26), and within that, consolidation became load-bearing (5/5). **The result strengthens
exactly as far as the factorization is cleaned.** Next:
1. **Raise the premise ceiling** — sharpen the filler channel so identity is recovered cleanly (e.g. aggregate
   filler over verb positions only / sharper attention / larger or better-regularized `fill_dim`). If the
   premise → ~0.9, the consolidation result should follow and could clear the 0.30 bar.
2. **Then** multi-seed at the cleaned config + the GECA redundancy guard + the second-split generality check
   (BUILD CONDITION 2).

## Discipline notes

- 5 seeds for the headline; probes n=1 (consistent with the headline). Δ positive in 5/5 — the effect is robust
  even if the absolute level is modest. Anti-homunculus PASS pre-build (`a0c17e48621a908db`); CONDITION 1 honored.
- Cosmetic: `experiments/89` `train()` epoch-loss print shows 0.000 (the accumulator is never summed); training
  is unaffected (loss/competence confirmed). Fix before the next run.
