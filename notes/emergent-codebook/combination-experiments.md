# Combination experiments (cross-capability) — first-class as of 2026-06-01

*The artifact the old linear-phase framing implicitly forbade. Per [CONTEXT.md](../../CONTEXT.md) §4,
capabilities are a coupled DAG, not a sequence; a mechanism that NULLs in **isolation** can work **in
combination**. This file registers cross-capability experiments. Each still gets a normal frozen
precommit under `notes/emergent-codebook/` before it runs — this file is the registry + the shared
discipline, not the precommit.*

## Why this exists (the false-negative trap)
The 121-127 arc tested **Codebook-growth alone** — iterate a *static* operator over a flat code — and
exhausted that family (the local-vs-global bound, capability-level). But that arc never used the
**Replay** capability's distinctive lever (the *schedule* of what gets replayed, interleaved with what).
The CLS literature (McClelland; the completeness critic's Reframe A) says paradigmatic/generalizable
structure is **manufactured by the interleaving dynamics of replay**, not latent in any static operator.
So the arc's nulls may be **false negatives of isolation-testing**. Combination experiments test that.

## Shared discipline (binds every combination experiment)
- **Anti-homunculus is sharper here.** A combination is *not* a license to hand-wire. Any cross-capability
  coupling must be a **local geometric/energy dynamic or a measurement of one** — e.g. the replay schedule
  must EMERGE from a *local replay-priority signal* (surprise / tension / settling-residual), never a
  hand-set "alternate king/queen contexts" curriculum (that is the banned controller/homunculus).
- **Batch-offline** (sleep/wake) — replay is offline replay over a frozen buffer; no runtime error-driven
  writes.
- **The same gate machinery** — frozen precommit, the calibration anchor, the **within-set label-shuffle
  B-KILL** (the pair-specificity headline that killed the 119/126/127 false positives), multi-seed,
  gauge-free specificity. A combination experiment that drops these is not exempt.
- **A combination must be compared to BOTH isolated arms** — the headline is the *interaction* (does
  growth × replay beat growth-alone AND replay-alone), not just "it's positive."

---

## CE-1 (LEAD) — Codebook-growth × emergent Replay-interleaving
**Depends on:** Consolidation-write (FLOOR, cleared). **Status:** SPEC — un-built. **Capabilities:**
Codebook-growth ⇄ Replay.

**The one question:** does an **EMERGENT, pattern-separated replay interleaving** of contexts —
re-ordering *which* windows the growth dynamic consolidates, by a *local* replay-priority signal — make
paradigmatic (king/queen) structure emerge where the same growth rule on the **as-is** (corpus-order)
stream NULLs (the 121-127 wall)?

**Sketch (substrate-free, behind the Abstraction build-gate; reuse `experiments/61/65/68`):**
- Same WikiText-2 `build_S` / `grow_G` setup, same n=40 SimLex non-cooc pairs, same gauge-free para-vs-
  random + **label-shuffle B-KILL**, 5 seeds, calibration anchor (+0.1092/kq 0.222).
- **Arm A (isolated control):** the growth dynamic on the **as-is** stream (= the 121-127 null, reproduced).
- **Arm B (the combination):** the same growth dynamic fed a **replay-reordered** stream where replay
  priority is an **emergent local signal** (e.g. settling-residual / surprise / co-activation tension —
  NOT a hand-set king/queen schedule), with **pattern-separation** (interleave dissimilar contexts).
- **Controls:** the within-set label-shuffle B-KILL (headline); a **stream-shuffle gauge** (does the
  *reordering itself*, stripped of structure, manufacture the signal — the artifact check, noting the
  2nd-order gauge-leak caveat from 125); the as-is arm (the isolation baseline).

**Gates (to freeze in the precommit):** B-KILL pair-specific residual CI-lo > 0 in ≥4/5 seeds, **AND**
Arm B beats Arm A (the as-is isolation null) by the pre-set margin, **AND** the emergent-priority signal
(not a hand-set schedule) is the operative ingredient (a *random* replay reorder must NOT reproduce it).

**Kill criterion (decisive either way):** if emergent-interleaved replay **nulls** (B-KILL ≈ Arm A), the
**flat-code-from-small-text program is CLOSED** — the bound is banked as capability-level *and*
replay-invariant, and the next move is an **Abstraction-node build-gate decision** (a substrate/data
change — the user's to make), NOT another oracle. If it **passes**, replay-as-structure-generator is the
real Phase-4 capability — write its grounding + a faithful precommit (the emergent-priority dynamic
becomes the thing to specify and card).

**Anti-homunculus check (must pass before building):** who decides the replay order? → a *local* priority
scalar (surprise/tension/residual) per stored trace, not a supervisor; the interleaving is a *measurement
of a local dynamic*, not an arbitration. If the only way to get a pass is to hand-pick the schedule, the
result is void.
