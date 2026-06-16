# PRE-COMMIT — Bet B: escalate to a regime where the KNOWN method also fails (the 144 lesson, front-loaded)

*Status: DRAFT for user review + anti-homunculus check (2026-06-16). Gated on
[Report 144](../reports/144_betb_scan_geca_guard/report.md) — the GECA redundancy guard FALSIFIED the
distinctiveness of the SCAN add-jump consolidation graduation (GECA **beats** it, 0.984/0.9998 vs 0.868).
Charter: [CONTEXT-B.md §8](../CONTEXT-B.md) + [betb-scan-discriminating-regime-precommit.md](betb-scan-discriminating-regime-precommit.md).
User decision (2026-06-16): escalate to a **harder regime (MCD/COGS)** — the only route on which "consolidation
beats known methods" stays testable.*

---

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** Bet-B continual compounding-transfer — re-asking the thesis test in a regime where the
> *known* compositional-generalization fix (GECA) **also fails**, after [144](../reports/144_betb_scan_geca_guard/report.md)
> showed SCAN add-jump is GECA-solvable (so a positive there is redundant by construction).
> **Headline (unchanged in shape, harder regime):** held-out compositional exact-match, consolidation lifted
> CI-disjointly above the no-consolidation baseline **AND above a faithful GECA arm** (the new bar — beat, not
> match, the known method).
> **Controls:** vanilla floor · GECA (now a **Stage-0 GATE**, not a final guard) · the §2 mechanism arm ·
> random/simple ceiling · ablations (TBD with the mechanism).
> **Last verified:** Report 144 (GECA beats the add-jump consolidation → REDUNDANT).
> **Why now:** STATUS — guard #1 falsified the claim; the program's recurring redundancy (133/139/144) is the
> task-selection confound (RETROSPECTIVE §4); the fix is a regime that defeats known methods.

---

## 1. Why this exists — the one structural change from the SCAN Stage-1 design

Four Bet-B results in a row (133 SGNS≈SVD, 139 EWC≈Benna-Fusi, 144 GECA>consolidation) are **REDUNDANT** for the
same reason: the regime was solvable by a simple/known method, so any brain-shaped positive was matched-or-beaten.
[RETROSPECTIVE §4](RETROSPECTIVE-two-bets-2026-06-06.md) named this the **task-selection confound**. SCAN add-jump
discriminates against *vanilla seq2seq* (0.003) but **not** against the known compositional-augmentation fix
(GECA 0.984). The fix is not a better mechanism — it is a **regime where the known method also fails.**

**THE CENTRAL DESIGN PRINCIPLE (the 144 lesson, operationalized): GECA moves to STAGE 0.**
In the SCAN Stage-1 design GECA was the *last* guard — run after the mechanism graduated, where it falsified the
claim. Here it is the **first gate.** We do **not** design or build any consolidation mechanism until Stage 0 has
empirically established **both**:
- **(0a)** vanilla seq2seq fails the regime (the gap is real — as before), AND
- **(0b)** a *faithful* GECA (best-shot, computed-from-buffer, generous-to-GECA per the [144 method](../reports/144_betb_scan_geca_guard/report.md)) **also fails** the regime (≪ ceiling).

If (0b) fails — i.e. GECA solves it — the regime is **not** discriminating against known methods and we do not
invest a mechanism in it (we make it harder, or move to the next tier). This is the single most important change;
it makes the redundancy verdict cost ~20 min of the machinery we already have (`exp87` vanilla + `exp91` GECA),
not a multi-hour mechanism build that gets falsified at the end.

---

## 2. The mechanism open-problem (load-bearing — do NOT assume the exp90 recipe transfers)

**The exp90 consolidation recipe does NOT carry over.** It aligns a structurally-identified **peer set of
one-token-command primitives** ({jump,walk,run,look}) so a *held-out primitive* lands in the verb-slot subspace.
A GECA-resistant regime (MCD / compound-divergence) holds out **novel compounds of seen atoms**, not a held-out
primitive — there is no single off-manifold token to align. So:

- **The Stage-1 mechanism for this regime is UNDESIGNED.** It is the load-bearing design problem, deferred to a
  Stage-1 precommit **after** Stage 0 validates the arena. Candidate shapes (all must pass anti-homunculus first):
  a fixed local restructuring loss that factorizes encoder states into **(slot ⊕ filler)** so novel
  slot×filler *combinations* compose; an iterative consolidation that decorrelates compound representations into
  their atomic constituents. **None of these is to be built until §1 Stage 0 passes.**
- **Anti-homunculus is mandatory before any mechanism build** (the `anti-homunculus-reviewer` agent, as for exp88).
  No "if compound-held-out then …"; the restructuring must be a fixed content-blind local loss.

---

## 3. Regime tiers — start cheap, escalate only if needed (each gated on §1 Stage 0)

| tier | regime | GECA-resistance | data friction | mechanism fit |
|---|---|---|---|---|
| **A (start here)** | a **custom compound-hold-out split built from the SCAN data in hand** (hold out specific modifier-compounds, e.g. `around right` / `opposite left`, while keeping every atom + every primitive in train) | **empirically gated** — run the `exp91` GECA arm; require it to FAIL (§0b). If GECA solves it, harden the split or go to tier B | **none** (data in hand; reuse `exp87`/`exp91`) | needs slot⊕filler factorization, not primitive-alignment |
| **B** | **canonical SCAN-MCD** (mcd1/2/3, Keysers 2020 — constructed to defeat compositional augmentation) | **theoretically guaranteed** (matched atom dist., max compound divergence) | **TFDS** (`scan/mcd1-3`; not installed) **or** generate via `dbca-splitter` | same |
| **C** | **COGS** (semantic parsing; structural-generalization splits) | strong (structural gen. unsolved by augmentation) | new data + logical-form output format | different substrate; bigger build |

**Recommendation:** run **tier A's Stage 0** first (vanilla-fails + GECA-fails on a self-built compound split,
~20 min, zero new infra). It is decisive about whether SCAN *at our scale* can even host a GECA-resistant regime.
- If tier A finds a split GECA fails → that is the arena; proceed to the Stage-1 mechanism precommit.
- If tier A finds GECA solves *every* compound split we can build at this scale → that is itself a finding
  (SCAN-at-scale is GECA-saturated), and we escalate to tier B (canonical MCD) or tier C (COGS) with eyes open.

*Note (be honest about the rising bar):* even in MCD/COGS the relevant "known methods" are not only GECA —
transformer and meta-learning baselines do well on some splits. The bar for "beats known methods" keeps rising;
Stage 0 should report the *strongest cheap known baseline available*, not only GECA.

---

## 4. Headline & controls (shape unchanged; GECA promoted to a Stage-0 gate)

- **Stage 0 (gate, build FIRST):** vanilla exact-match ≪ ceiling (0a) **AND** faithful-GECA exact-match ≪ ceiling
  (0b), ≥3 seeds. PASS ⇒ the regime discriminates against known methods ⇒ a mechanism is licensed.
- **Stage 1 (only if Stage 0 passes):** headline = held-out compound exact-match, **consolidation − baseline**,
  ≥8 seeds, bootstrap CI; **PASS = CI-lo > 0 AND consolidation ≥ the faithful-GECA arm** (the new, stricter bar —
  beat-or-match the known method, never lose to it as in 144). Drill-downs: ceiling no-regression; the
  "zero-composed-data" contrast (does consolidation reach it without manufacturing the held-out compounds?).
- **Controls:** vanilla floor · faithful GECA · random/simple ceiling · mechanism ablations (TBD) · BUILD
  CONDITION 1 (all structural reads computed-from-buffer, asserted, never hardcoded).

---

## 5. Build checklist

- [ ] **Tier-A Stage 0 (next, cheap):** build a custom compound-hold-out SCAN split (a loader that partitions the
      existing `simple` data by a held-out modifier-compound; assert atoms+primitives all present in train).
      Run `exp87` vanilla + the `exp91` GECA arm on it. Gate: vanilla fails AND GECA fails.
- [ ] If Stage 0 fails (GECA solves it): harden the split (more divergent compounds) or escalate to tier B/C.
- [ ] If Stage 0 passes: **Stage-1 mechanism precommit** (the slot⊕filler restructuring consolidation) →
      **anti-homunculus review** → build → ≥8-seed headline vs baseline AND vs GECA.
- [ ] Report under `reports/`; STATUS + memory update.

## 6. Open questions before freeze

- **Custom-split GECA-resistance is empirical, not guaranteed** — full (fragment) GECA may manufacture some
  held-out compounds; that is exactly what Stage-0b measures. Use the strongest GECA we can implement, not the
  weakest, so a Stage-0b "GECA fails" is trustworthy.
- **Data tier** — tier A (custom split, tonight) vs committing to tier B (TFDS-MCD infra) up front. Recommend A
  first; only stand up TFDS/dbca if A shows SCAN-at-scale is GECA-saturated.
- **Mechanism** — the real design work; deferred until the arena is validated. Do not pre-build.
