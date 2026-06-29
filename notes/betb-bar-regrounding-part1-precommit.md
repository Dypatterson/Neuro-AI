# PRE-COMMIT — Bet B: re-ground the bar (PART 1, today) + the HELD entropy reopen (PART 2)

*Status: PART 1 = ready-to-run, **bank-respecting** (trains nothing, decides nothing — needs no
reopen consent). PART 2 (exp100) = **HELD** pending (a) explicit user bank-lift AND (b) the redesign
below. Authored 2026-06-29 after fast-forwarding the 2026-06-22 cloud branch and seam-auditing it
(7-agent workflow `wf_54a7cfe6-f9c`). Supersedes the entropy/ReCOGS framing in
[verification-round-2-and-seams.md §"Mechanisms worth testing"](../brainstorm-workspace/2026-06-22-grown-hierarchy-mechanisms/verification-round-2-and-seams.md)
where it conflicts. Charter: [CONTEXT-B.md](../CONTEXT-B.md). Program is BANKED
([RETROSPECTIVE-program-close-2026-06-21.md](RETROSPECTIVE-program-close-2026-06-21.md)).*

---

## 0. Why this doc exists — the seam audit

The 2026-06-22 cloud session's R2 verification ranked two "next moves" above its own demoted
neuron-splitting headline: **(1) re-ground the bar on ReCOGS/SLOG + treat MCD as covariate-shift**, and
**(2) test the Wold-2025 entropy soft-prior**. The 2026-06-29 seam audit found both rankings are
**not today-feasible as written**, for reasons the cloud session missed:

- **Rec (1) is the *least* today-feasible move on this repo.** There is **no COGS / ReCOGS / SLOG / CFQ
  data anywhere** under `data/` (only SCAN simple / addprim_jump / mcd1-3). The harness
  ([experiments/87_betb_scan_gap.py](../experiments/87_betb_scan_gap.py)) parses only the SCAN
  `IN: … OUT: …` flat-action format with string exact-match; COGS/ReCOGS/SLOG are logical-form targets
  needing a new tokenizer **and** ReCOGS's semantic-equivalence-modulo-variable-renaming eval. Honest
  estimate: **3–5 working days**, not the "cheap, highest-value" the R2 doc claimed (its buildability
  audit only checked the GRU splitting operator, never the dataset/harness).
- **Rec (2) is `clause-aug` relabeled by construction.** Verified in-code across three agents: the SCAN
  single-clause universe is **~105 forms; `clause-aug` (exp96) already injects 102** of them ×10, and
  **100% of mcd1 test multi-clause commands already have BOTH clauses covered** by that set. The
  residual MCD gap is *pure top-level recombination* — the maximum-compound-divergence wall Reports
  146–150 already banked. A domain-expert opened the Wold/Charpentier/Simon 2025 PDF (arXiv 2505.13089)
  and confirmed: Wold's entropy is over a **single embedded-clause verb slot**, and on MCD that slot is
  **saturated** (novel test atoms = ∅; 44/46 test bigrams already in train). MCD's divergence axis is
  *compound-template* — exactly the axis Wold's single-slot entropy does **not** control. The literature
  therefore predicts an MCD **null**; Wold's own paper says high entropy "did not meet requirement (iii)"
  and "systematic generalization remains an open problem" even on his easy synthetic CFG.
- **The 0.297 "atomic floor" is itself partly LEAKED.** `clause-aug` is sourced from
  `tasks_train_simple` **and `tasks_test_simple`**, so ~17 single-clause mcd1 *test* inputs appear
  verbatim with gold actions, and ~98% of test compounds have both clauses verbatim in the augmentation.
  So Report 149's "0.158 → 0.297" is not a clean no-leakage baseline; PART 1 quantifies this.

**The salvageable, today-feasible piece of rec (1):** the **covariate-shift characterization** of our
*own* mcd1/2/3 — analysis of existing data, no new dataset. That is PART 1. It honestly addresses **only**
the covariate-shift seam (MCD = worst-case compound covariate-shift); the *ReCOGS decoder-artifact* seam
(COGS-LF string/length confound) **does not exist on SCAN's flat-action exact-match** and remains
un-addressable today — state that explicitly, do not claim PART 1 "re-grounds the bar" wholesale.

---

## PART 1 — covariate-shift / compound-divergence characterization of mcd1/2/3 (TODAY, no training)

> **Type:** characterization / drill-down (NOT a graduation experiment — nothing trains, no gate moves).
> Bank-respecting; anti-homunculus-clean (offline batch statistic, CLAUDE.md rule 6 exempt).
> **Active capability:** none reopened — this *strengthens the banked finding* by quantifying the
> compound-divergence wall and bounding clause-aug headroom.
> **Why now:** it (a) is the honest today-feasible salvage of "re-ground the bar," (b) mechanically
> bounds how much headroom any non-leaky atom/clause prior could ever reach above the floor, and (c) its
> result *predicts the exp100 null* — so it tells us whether PART 2 is even worth a GPU run.

**Reuse:** [experiments/analyze_mcd_divergence_axis.py](../experiments/analyze_mcd_divergence_axis.py)
(already computes the three divergence axes across mcd1/2/3) + `exp96.split_clauses` /
`exp96.load_single_clause_aug`.

**Compute (all per split mcd1/2/3, train vs test):**
1. **Atom vs compound divergence (the covariate-shift signature).** Unigram (atom) distribution
   divergence ≈ 0 (already verified: novel test atoms = ∅) vs clause-template / (primitive, template)
   compound-distribution divergence (large). Report total-variation or Chernoff α per Keysers 2020.
   Names the shift as **compound, not atom**.
2. **Clause-aug atomic-coverage cross-reference (the headroom bound).** For each test command, mark
   whether each of its clauses is covered by the 102 single-clause forms `clause-aug` injects. Tabulate
   the fraction of test compounds with BOTH clauses covered (expected ≈ 100% on mcd1). This bounds the
   ceiling any non-leaky atom/clause prior (incl. the entropy soft-prior) could reach above the floor:
   **near-zero headroom ⇒ exp100 ≈ clause-aug ⇒ predicted null.**
3. **Leakage quantification of the 0.297 floor.** Count (a) verbatim test inputs present in
   `clause-aug` (the test_simple leak), (b) test clauses whose gold decode is handed over. Report it —
   it re-scopes Report 149's "atomic injection ~doubles vanilla" as *partly leakage-driven*.
4. **Leakage-invariant pre-registration (in code).** Define and assert the template-level invariant any
   future augmentation must satisfy: shares atoms/short sub-clauses with test (free — MCD shares them by
   construction) but **zero** augmentation example whose top-level clause-skeleton template matches a
   held-out test template. Surface-string equality is *not* enough — the real MCD leak is
   surface-distinct-but-template-identical.

**Deliverable:** a short `reports/151_betb_mcd_covariate_shift/`-style markdown with per-split tables.
No model trained. **Done-gate:** this is a characterization, so the 5-gate "experiment done" bar applies
only as (4) report exists + (5) STATUS updated; there is no headline-with-CI/control because nothing is
trained — label it a characterization explicitly so it is not mistaken for a graduation result.

---

## PART 2 — exp100 entropy soft-prior (HELD; do NOT run without the two gates below)

**Gate A — user must explicitly lift the bank.** The program is BANKED by the user's 2026-06-21 call;
reopen is the user's to make. No build until that consent is on record.

**Gate B — redesign so it is NOT clause-aug relabeled.** If reopened, exp100 is interpretable ONLY as
**Wold's sample-size-controlled entropy *sweep***, not "add atoms":
- **Fixed total token budget**, ≥4 occupancy-entropy levels (Wold Incremental-Support over *where*
  atoms/sub-clauses appear), with `H_atom` / `H_template` before/after reported as a **manipulation
  check** (SCAN's ~13-token input may cap the achievable entropy — if the knob barely moves, report
  "manipulation too weak," not a mechanism null).
- **Non-leaked atomic-floor control:** source the clause/atom augmentation **only from the mcd TRAIN
  split's own clauses**, NOT from `tasks_test_simple` — so the floor does not leak 17 verbatim test
  commands nor the gold decode of every test clause.
- **Required guard arms:** `VOL` (volume-matched, marginal-preserving — "more data" vs "higher-entropy
  data") and `SHUF` (structure-destroying — must NOT help). Plus a verbatim whole-command leakage
  assertion (return zero) and the template-level audit from PART 1.4.
- **Sole headline = (ENT − ATOM) CI-disjoint**, ≥8 seeds, bootstrap CI — *not* (ENT − A), which would
  let a beat-vanilla-but-not-clause-aug result rationalize as success (the 133/139/144 redundant-positive
  pattern). Non-redundancy bar = the **published fragment-GECA ≈ 0.51 on mcd1** (Report 146 / Conklin
  2021), NOT the in-house token-GECA ≈ 0.
- **Honest pre-registered prior:** the literature predicts a **null** (Wold's axis is saturated in MCD).
  A clean null hardens the bank and files entropy with the redundant-positive pile; that is an
  acceptable, decisive outcome — but it is ~84 min single-process / ~21 min sharded of GPU to confirm
  what PART 1 can largely *predict* for free. Run only if PART 1 reveals real atom/clause headroom that
  clause-aug failed to supply.
- Compute: thin reuse file `experiments/100_betb_scan_mcd_entropy.py` over exp87/exp94 helpers; no new
  model class, no consolidation loop.

---

## The real higher-validity move (medium-term, not today): ReCOGS / SLOG re-grounding

If the goal is genuinely to re-ground the bar against the decoder-artifact confound, that lives on COGS
logical forms, not SCAN. It is a **3–5 day** build: (1) fetch ReCOGS (Wu/Manning/Potts TACL 2023) + SLOG
(Li et al. EMNLP 2023) + provenance-pin; (2) new LF tokenizer + longer max_len; (3) wire ReCOGS's
**semantic-exact-match-modulo-variable-renaming** eval (the artifact fix) and validate it against the
paper's reference numbers; (4) reproduce a published vanilla seq2seq floor at ≥8 seeds before any
mechanism claim. This is benchmark-validity infrastructure, not a mechanism test — scope it as its own
multi-day effort if/when the user wants the cleaner arena.
