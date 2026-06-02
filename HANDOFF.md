# HANDOFF — 2026-06-01 (PM-9; end of a long session; fresh-session orientation)

**Branch:** `audit/coupled-null-reeval`. Memory edits live in `~/.claude/`. Oracle/experiment `_*.json` and
`reports/**/*.json` are gitignored (data); the markdown docs + experiment `.py` are the durable record.

**One line:** Two things happened. (1) The **coupled null-audit** (the session's task) is DONE — 64 nulls
re-classified under the §4 capability-DAG, triage table + short list in
[null-audit-coupled.md](notes/emergent-codebook/null-audit-coupled.md). (2) The audit's #1 lead (CE-1 ⊗ 127)
was pursued through a frozen grilled precommit → built experiments → an exciting "a local writer reaches
paradigmatic structure past the LINEAR bound" screen positive → **a 7-agent adversarial verification + a
same-operator diagnostic CAUGHT it as a 1st-order-representation artifact (the 3rd false positive the
discipline has caught this year, after 126/127) → RETRACTED.** The clean banked finding (n=10): on the
CORRECT operator `build_S`, **127 replicates** (nonlinear partition reaches it, linear ~0) but **locality
is NOT free** — a genuinely bounded-memory LOCAL writer falls short of the global one → the GLOBAL pass is
load-bearing → **replay / the Oracle-E TEM writer re-enter as the gap-closers with a MEASURED gap
(+0.179 → +0.224).** The durable win is the methodology, not a graduation.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — charter (note the new §3 **"structural priors are not homunculi"** clause:
   a fixed problem-generic scaffold is legal; only outcome-arbitration + global backprop + answer-shaped
   scaffolds are banned).
2. **[STATUS.md](STATUS.md)** — bookmark (PM-9 = the CE-1 arc; PM-8 = the audit).
3. This file.
4. **[null-audit-coupled.md](notes/emergent-codebook/null-audit-coupled.md)** (the audit + short list) and the
   **§7 run-log of [the CE-1 ⊗ 127 precommit](notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md)**
   (the full exp69→72 arc, the verification verdict, the retraction, the banked locality-cost).

## 1. Where we are RIGHT NOW
- **The audit is the solid deliverable.** 64 nulls → 47 NO (73%) on three route-invariant bound families
  (paradigmatic subdominant-mode; 1/√D FHRR crosstalk floor; bind(k,v) algebra); short list of MAYBE/UNTESTED
  led by CE-1 ⊗ 127, then Oracle-E TEM local writer (#2), 052 Pair-#2 drift-pressure, 013 tag_count→u_k,
  Oracle-C eligibility×surprise.
- **CE-1 ⊗ 127 arc CLOSED (banked at n=10):** on `build_S` (the precommit/127 operator, anchor-valid
  +0.1092/0.222): offline k-WTA ≈ k-means ≈ **+0.22** (127 REPLICATED at the hubness-immune B-KILL + breadth
  22/40), `grow_G` linear floor ≈ **0**. The accumulating/global online writer reaches the ceiling (1.01);
  the genuinely **bounded-memory LOCAL writer falls short** (bounded +0.179, ratio 0.80; converged−bounded
  across-seed CI **[+0.042,+0.055]>0** = locality cost ROBUST; bounded−frozen lo +0.104 = learning matters).
  → **locality is NOT free; the global pass is load-bearing.**
- **What was RETRACTED:** exp70's screen positive ("local writer past the linear bound") — it ran the
  partition arms on the WRONG (1st-order `l2rows(sppmi)`) operator while the floor/ceiling used `build_S`;
  on `build_S` the locality cost appears. Caught pre-banking by the verification.

## 2. The next move (un-built) — the GAP-CLOSER
The locality cost gives replay a **measured target**: can a mechanism make a bounded-memory LOCAL writer
close **+0.179 → +0.224** on `build_S`? Two candidates (the user wants replay kept alive):
1. **The emergent-replay-schedule experiment** (CE-1's actual premise, now well-motivated): a *local*
   replay-priority (surprise/novelty × settling-residual) + pattern-separation re-orders/re-weights which
   windows the bounded-memory writer consolidates — does it close the gap? Reuse exp72; the anti-homunculus
   rule (priority must EMERGE from a local scalar; random-reorder gauge must NOT reproduce) is in the precommit §1.
2. **The Oracle-E TEM local writer** (short-list #2): an online Hebbian path-integration slot-factorization
   writer (NOT reducible to k-means; its *dynamic* differs), per Report 125 §5.
Optional remaining verification controls (now low-priority — the headline is a locality-COST, not a positive
needing defense): single-pass locality, the 127 permutation-breadth battery.

## 3. Invariants the user holds (do not violate)
- **LOCAL growth is the MECHANISM, non-negotiable.** Global SVD/NMF/k-means = diagnostics/controls only.
- **Anti-homunculus, sharpened (CONTEXT.md §3, new):** a fixed problem-GENERIC scaffold (layers, k-WTA cap,
  D) is LEGAL; banned = a supervisor arbitrating outcomes, global backprop, or a scaffold hand-shaped to the
  answer (a design-time homunculus).
- **Substrate-free oracles + combination experiments are in-scope** (rung-1, no gate); a new-substrate BUILD
  is the Abstraction-node build-gate, the user's to lift. The fidelity ladder (precommit §4.5): a screen-pass
  ≠ a behavior claim; escalate rung-1 → FHRR-port → integrated, with the over-claim guards.
- **The load-bearing habit (paid off 3× this year):** the verdict-bearer is the adversarial control, never
  the headline arm. The within-set label-shuffle B-KILL + the competent (k-means) AND incompetent
  (frozen-random) controls + across-seed CIs caught 126, 127, and now exp70.

## 4. Banked side-findings (don't relitigate)
- **127 replicates on `build_S`** at the hubness-immune B-KILL level (k-WTA ≈ k-means ≈ +0.22; grow_G ~0).
  The operative bound is LINEAR-PROJECTION-vs-NONLINEAR-PARTITION (k-WTA NOT special vs k-means on `build_S`).
- **Locality is NOT free** on `build_S` (bounded-memory local writer < global, n=10, robust). The global/
  accumulated pass is load-bearing.
- **Operator/representation matters:** on 1st-order `l2rows(sppmi)`, k-means FAILS the B-KILL (CI-lo<0) while
  k-WTA succeeds (a 1st-order quirk); on the correct 2nd-order `build_S` they tie (127). Always run the
  partition on `build_S`, with grow_G/NMF on the SAME operator.
- Magnitudes are partition-inflated — never claim "k-WTA/partition beats SVD/NMF."

## 5. Artifacts (uncommitted — this session)
- Docs: [null-audit-coupled.md], the [CE-1 ⊗ 127 precommit] (FROZEN + §7 run-log), CONTEXT.md §3 clause,
  STATUS.md (PM-9 + Active-deliverable update; PM/PM-2 migrated to [status-log/2026-05.md]), this HANDOFF.
- Experiments: `experiments/69` (planted smoke), `70` (WikiText head-to-head + verification target),
  `71` (operator/representation diagnostic), `72` (decisive online-on-`build_S`, n=10).
- Data (gitignored): `reports/_exp69_planted_smoke.json`, `_exp70_wikitext_headtohead.json`,
  `_exp71_operator_repr.json`, `_exp72_online_build_s.json`, `_exp72_online_build_s_n10.json`.
- Pre-existing untracked (leave): `brainstorm-workspace/2026-05-30-research-grounded-plan/_wf{1,2}_raw.json`,
  `reports/gate0_2026-05-28/`.
