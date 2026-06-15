# Report 140 — Bet B / SCAN discriminating regime, STAGE 0: the compositional gap reproduces

**Status:** Stage-0 VALIDITY gate **PASS** (2026-06-15). Not a graduation — the analog of
[Report 136](../136_betb_transfer_gap_diag/report.md)'s gap-diagnostic, on the real benchmark.
**Experiment:** `experiments/87_betb_scan_gap.py`. **Charter:**
[notes/betb-scan-discriminating-regime-precommit.md](../../notes/betb-scan-discriminating-regime-precommit.md).

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B continual compounding-transfer ([CONTEXT-B.md §8](../../CONTEXT-B.md))
  in the discriminating / compositional regime ([RETROSPECTIVE §6](../../notes/RETROSPECTIVE-two-bets-2026-06-06.md)).
- **Headline (Stage 0, a VALIDITY diagnostic):** exact-match accuracy on the SCAN
  add-primitive-`jump` split vs the random (simple) split — must reproduce the documented
  failure (random ≈100%, jump ≈0) at our small scale, per
  [precommit §2 GATE](../../notes/betb-scan-discriminating-regime-precommit.md).
- **Control:** the simple (random) split is the sanity ceiling — same model/config must score ~100%.
- **Why now:** the modular-arithmetic version of this regime was NULL (toy substrate too
  brittle — [precommit §8.1](../../notes/betb-compositional-discriminating-regime-precommit.md));
  SCAN is the canonical "simple methods fail to compose" regime, and Stage 1 is gated on this gap.

## Result

| split | train structure | test exact-match (n=3 seeds) |
|---|---|---|
| **simple** (random) | i.i.d. command/action | **0.9947** [0.9842, 1.0000, 0.9998] |
| **add-prim `jump`** | `jump` ONLY isolated; 0 train compositions | **0.0031** [0.0005, 0.0031, 0.0057] |

**Gate (simple ≥ 0.90 ∧ jump ≤ 0.05): PASS.** A ~99.5-point gap from one model/config.

## Reading

- The model **can** learn SCAN essentially perfectly (random split ~99.5%) — so the templates,
  modifiers, and the action grammar are all learned. The jump-split failure is **not** a capacity
  or training-budget problem; it is a **systematic-generalization** failure: `jump`, seen only as
  the isolated primitive (`jump → I_JUMP`, 1467× in train, 0 compositions), is never pulled into
  the verb-slot-filler role the other verbs occupy, so the decoder's template machinery does not
  apply to it. Textbook reproduction of Lake & Baroni 2018.
- **This is the headroom the toy lacked.** Massive separation (0.995 vs 0.003), unambiguous,
  multi-seed-consistent. Unlike the modular-arithmetic regime (compositions bimodal, multi-task
  grokking collapse), the discriminating regime exists cleanly here.

## Scope / honesty

- n=3 seeds; the separation is so large (0.995 vs 0.003) that CIs are unnecessary to call the gate,
  but Stage 1's headline (closing the gap) will use ≥8 seeds + bootstrap CIs per the charter.
- Config: 1-layer GRU enc-dec + Luong attention, embed=64, hidden=200, 30 epochs, tf-ratio=0.5,
  Adam 1e-3, grad-clip 5. Attention did **not** let the model cheat the jump split (the §5 worry) —
  the failure is clean.

## Disposition

**Stage 1 LICENSED.** The question that the whole Bet-B arc was built to ask is now testable in a
regime where the simple method demonstrably fails: **does a restructuring consolidation (+ replay)
manufacture the compositional generalization (`jump` → templates) that replay/protection cannot?**
The consolidation recipe is the load-bearing unsolved design (must be a fixed local objective,
anti-homunculus-clean, and survive the GECA/known-fix redundancy guard) — designed next in a
Stage-1 precommit. Connection: this is the Bet-A king~queen **substitutability** problem
(make `jump` substitutable for `walk`), now behaviorally testable and under Bet-B's lifted rules.
