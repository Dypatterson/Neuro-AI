# Report 140 — Harness extraction + the discriminating compositional regime

**Status:** INFRASTRUCTURE + REGIME BUILT — **NO SCIENCE RUN.** The harness is green
end-to-end but has only been exercised at `--tiny`. No claim about the regime is made here.
**Type:** engineering + regime construction. Not a graduation attempt, not a drill-down on any
mechanism. **Date:** 2026-07-25. **Charter:** [CONTEXT-B.md](../../CONTEXT-B.md).

## Why this session happened

A full-repo review (6 dimensions, each adversarially verified, 13 agents) was asked for. Its
load-bearing finding was not a bug list — it was that **the program's stated next step had zero
work behind it, and the reason was structural.**

[The retrospective](../../notes/RETROSPECTIVE-two-bets-2026-06-06.md) §4 named the
task-selection confound as "the big one": every task in both bets was solvable by simple means,
so "the brain mechanism added nothing" is *expected by construction*. Its §6 specified the fix —
a compositional regime where transfer requires recombining primitives. Seven weeks later
`T1∘T2` appeared exactly once in the repo: in the retrospective itself.

The confound was **in the code**. `make_task`
([`experiments/80_betb_continual_transfer.py:50-65`](../../experiments/80_betb_continual_transfer.py))
implements `add` and `sub` mod p and raises on anything else, and it was baked into the harness
that produced Reports 134-139. Changing the task meant forking a ~300-line script — which is why
it never happened.

## What the review found (verified, corrected)

Two of the review's own claims did not survive checking, and are recorded here because the
correction is the point:

- **"exp85's headline arm runs unverified forked code."** exp85 defines `train_task_bf` as a
  hand-copy of `exp83.train_task`. Diffed: the *only* deltas are dropping `freeze_mlp` (correct —
  BF never freezes) and adding `bf.diffuse()`. **The fork is faithful; Report 139's numbers are
  not compromised.** It is a drift hazard, not a defect.
- **"0/196 reports achieve a complete experiment preamble."** Actually **5 of the 11 attempts**
  carry all five fields — 128, 129, 135, 138, 139, i.e. the entire live line. **The preamble is
  working and was kept.** What had rotted was the mandatory `file:line` citation.

Findings that did survive and drove this session:

| Finding | Evidence |
|---|---|
| Live line imports nothing from `src/` | 6 of 7 `betb_*` experiments; 84/85 chain to 83 via `spec_from_file_location` **by filename** |
| 37.6% duplicated lines across `experiments/79-85` | `ContinualNet` ×3, `main` ×7, `run_arm`/`aggregate` ×5 |
| No published number is on disk | `find reports -type f ! -name '*.md'` → **14** files (all PNG/stderr) across 196 reports; `.gitignore:22` excluded every `reports/**/*.json` |
| Nothing enforced anything | `.git/hooks/` empty (hooks are never cloned), no `.github/`, no CI |
| 6 tests permanently red | all on a fixture `git log --all --diff-filter=A` shows was **never tracked** |
| Discovery broken | no index over 152 numbered reports; CLAUDE.md's canonical grep example returns **0 hits**; Benna-Fusi was rebuilt in an experiment script while a 965-line, 26-test implementation sat in `phase4/consolidation.py` |
| Setting was Task-IL and mostly undisclosed | [`experiments/83:74-77`](../../experiments/83_betb_two_timescale.py) per-task head indexed by task id; stated once at `reports/135:31`, absent from 137/138/139 scope lines |
| 3 of 5 §8-mandatory controls never run; 1 never implemented | `grep -iE 'joint\|in_context\|scramb'` over exps 83/84/85 finds only "CI-disjoint"; `reports/134:15` lists `frozen-in-context` as used |
| No external referent | `BWT\|FWT\|Lopez-Paz\|van de Ven\|Avalanche\|Mammoth\|GEM` → **0 hits** repo-wide |

## What was built

**`src/energy_memory/betb/`** — one harness, extracted from exp83.

- **`tasks.py`** — the task family is an **injected parameter**.
  `ModularArithmeticFamily` reproduces the 134-139 stream (anchor parity asserted by test).
  `CompositionalAffineFamily` is the discriminating regime: affine operators
  `o_i(x) = m_i x + c_i` on Z_p; uniform arity 3 (a primitive is a composition with identity, so
  "T3 is harder because it has more inputs" is not a confound); composition
  `o_j(o_i(x))` has slope `m_j m_i` and intercept `m_j c_i + c_j`, so it is neither primitive's
  parameters. **A fraction of `(i,j)` pairs is held out and never trained on by any arm** — the
  bar replay and weight-anchoring cannot clear by construction, since they preserve what was
  learned rather than inducing the factorization.
- **`consolidators.py`** — `BennaFusi` and `EWCAnchor` lifted from exp85, plus
  `SubspaceRestructure`: the first **restructuring** recipe in the program (damped power
  iteration contracting the circuit toward its dominant subspace — compression, not preservation;
  fence-clean, no closed-form SVD). **Untested; no result behind it.**
- **`runner.py`** — provenance envelope (git SHA, argv, seeds, config, **scenario**,
  torch/platform) and `Provenance.verify()`, which **raises** when a declared control was not
  executed. Merge refuses shards whose configs disagree; normalizes exp82's list-vs-dict
  `per_seed_raw` divergence.

**`experiments/86_betb_compositional.py`** — regime *validation*, explicitly labelled not a
graduation. Headline: the held-out composition gap for `replay_only` and
`replay_plus_consol(ewc)`. Implements the two controls that never existed — **joint-train
ceiling** and a **frozen-feature probe** (named precisely: an MLP has no in-context mechanism, so
the honest analogue of "frozen-model-in-context" is a frozen-feature linear probe, and saying so
is the difference between a control and a claim). Single shared head, task inferred from input.

**Both outcomes are informative.** Large gap → the regime discriminates and becomes the venue for
testing whether restructuring beats protection. Arms saturate → the regime is *not* discriminating
and must be made harder before any mechanism claim is run against it.

## Two bugs the new tests caught

1. **The scramble control was not input-matched.** Drawing scramble parameters from the shared
   generator advanced the RNG, silently reshuffling the train/test split — so the control differed
   from the real task on an axis other than the one it isolates. Fixed with a dedicated generator.
   This is the Report-134 invalid-scramble failure mode in a different guise.
2. **A schedule test that depended on training dynamics.** Rewritten to test
   `eval_checks()` directly.

Also fixed: the FTSR **quantization bug** flagged in Reports 138 and 139 and never addressed. A
fixed `eval_every=100` grid rounds numerator and denominator of a *ratio* onto the same coarse
lattice; at FTSR ~12 the denominator is a few grid points. `eval_schedule='geometric'` holds
relative resolution ~constant. The fixed grid remains available for anchor reproduction.

## Housekeeping

- `CLAUDE.md` cut to **6 measurement rules**. Justification: all nine false positives in the
  retrospective's catalogue were caught by a measurement rule; **none** by a read order, a
  `file:line` citation, a byte budget, or a triage label. Session start went 4 reads → 2, and now
  points at the **live** bet (it previously pointed at Bet-A phase notes). The preamble was kept
  (it is followed on the live line) but now cites by **section header** — three root-charter line
  citations had rotted, including one inside the rule written to prevent that.
- `reports/INDEX.md` generated (150 entries) by `scripts/build_reports_index.py`, with `--check`
  wired into CI. Surfaces 11 colliding report numbers. Keyed on full path — nine directories share
  the basename `02_phase2_retrieval_baseline.md`.
- CI added (`.github/workflows/tests.yml`): suite + harness smoke + index freshness.
- `reports/**/headline*.json` un-ignored; shards stay ignored.
- The 6 permanently-red tests now **skip honestly** with a regeneration pointer. They were not
  repaired with synthetic fixtures on purpose: their assertions check `passes_all_criteria` and
  derived counts, so a hand-built fixture would manufacture the answer rather than test it.
- `brainstorm-workspace/` and `notes/notes/` were **not** relocated despite the review's
  recommendation: 81 files link into them, and the search-surface tax was a *rule*, not a
  filesystem fact. The rule is gone; the record is intact.

## Test suite

**620 pass / 6 skip / 0 fail** (~108 s, CPU). Was 594 pass / 6 **error**. 20 new tests cover
anchor parity, held-out-pair non-leakage, scramble validity, the provenance guard, shard-merge
config mismatch, and the anti-homunculus contract — the last asserted mechanically
(`step()` must take no arguments, so it cannot be handed a metric) rather than in a docstring,
because three `legacy/` modules claim compliance in prose while violating it in code
(`replay_loop.py:800-826` reads a threshold and branches to delete, behind a guard that only fires
when `coverage_lambda > 0`, which defaults to `0.0`, while `experiments/18:444` calls it
unconditionally).

## ADDENDUM (same session) — the v1 compositional design is NOT LEARNABLE, and was replaced

Before running the regime at scale, a learnability probe was run. **The v1
`CompositionalAffineFamily` fails.** The model memorizes the training set and never generalizes,
so the regime would have measured nothing.

The decisive control was running the *known-grokking* modular task through the **same harness,
same optimizer, same split** — isolating "is the task broken?" from "is the harness broken?":

| Task (202 train / 87 test, wd=1.0, AdamW 1e-3) | TRAIN acc | Test acc @20k steps |
|---|---|---|
| `(a+b) mod 17` — control | 1.000 | **1.000** (groks: 0.18 @3k → 0.94 @6k → 1.00 @10k) |
| v1 compositional primitives, n_ops=17 | 1.000 | **0.000** (never groks) |

**Diagnosis.** The modular task is *one* global rule with 289 examples. The v1 design is
**17 independent random affine maps** `o_i(x) = m_i x + c_i` with `(m_i, c_i)` drawn
independently per operator — effectively 17 separate grokking problems at 1/17 the data each,
and, decisively, **nothing ties operator token `i` to its parameters**. Generalizing to an unseen
`(op_i, x)` pair is therefore impossible in principle, not merely hard. Train accuracy of 1.000
with test accuracy of 0.000 is the signature of pure memorization against an unlearnable rule.

**This is the same trap the regime was built to escape, inverted.** The retrospective's
requirement is a regime where the simple method *fails but the structure is learnable*. A regime
where **nothing** learns is exactly as uninformative as one that saturates: in both cases the
mechanism comparison is decided by construction rather than by the mechanism. Had this been run
at n=8 without the probe, it would have produced a confident, meaningless "the simple method does
not saturate."

**Cost of catching it here:** ~3 minutes of calibration versus several hours of 8-seed compute
and a banked non-result. The competent-control rule (`CLAUDE.md` measurement rule 2) is what
caught it — the control was a *task* rather than a mechanism, but it did the same job.

### The replacement — five designs built, trained, and measured

Five candidate task families were designed **and empirically trained** (not
argued), each scored against three properties, with the one claiming viability
independently re-run at 3× the steps. Property 2 is what separates them — three
candidates produce a "gap" only because nothing learns at all, which is the v1
failure wearing a different hat.

| Design | P1 primitives generalize | P2 trained pairs | P3 held-out | Gap | Verdict |
|---|---|---|---|---|---|
| **S5 coordinatewise** | **0.977** | **0.961** | 0.319 | **+0.642** | **VIABLE** |
| signed_add | 0.837 | 0.572 | 0.416 | +0.156 | PARTIAL — P2 far from saturation |
| SharedCircuit φ=t³ | 0.951 | **0.146** | 0.125 | +0.021 | P2 FAILS → P3 vacuous |
| binop-tokens | 0.074 | 0.071 | 0.087 | −0.021 | never left chance |
| FV-MAC | 0.006 | 0.007 | 0.041 | −0.034 | never left chance |

**S5 survived adversarial re-run.** Bit-exact reproduction, then extended to
60k steps: the gap **widens** to +0.669 [0.492, 0.834]. Held-out is flat
(0.319 → 0.320) while trained-pair climbs 0.961 → 0.989. No seed trends toward
closure; the riskiest seed *declined*. So this is a stable plateau, not slow
convergence — the failure mode that would have invalidated it.

**The abelian matched control is what makes the gap mean something.** Swapping
`S_5` for `(Z_5)^3` makes composition *pooling*, so the shortcut is correct:
held-out saturates at 0.908 and the gap collapses to **+0.060 with a CI including
zero**. Same architecture, same sizes, same step budget. Without this, a large gap
would only show the task is hard. Contrast `signed_add`, whose own control passes
all three properties *more cleanly than the design it controls* — which is why
properties 1+2+3 alone do not certify discrimination.

**Two integrity bugs, both found by the adversarial re-run, both fixed:**

1. **Primitive splits were keyed on the cell, not the operator.** `(op, IDENT, x)`
   and `(IDENT, op, x)` compute the same function; independent splits put **69% of
   primitive-test rows** into training under the mirrored slot order. True primitive
   ceiling is ~0.86–0.92, not 0.98.
2. **Operator sets were not rejection-sampled.** ~5% of held-out cells had a
   composite equal to a memorized primitive, answerable without composing at all
   (0.428 accuracy on those cells vs 0.224 on genuine ones). Acceptance rate ~0.36,
   so the fix is free.

Both are now asserted by tests, not just fixed. Note both bugs *inflated* the
original numbers, so correcting them lowers primitive accuracy and **widens** the
gap.

**A measurement constraint that changes the statistics.** Per-cell accuracy within
a single run ranges 0.000 to 0.856, so the effective n is the **8 held-out cells**,
not the 1000 held-out rows. Bootstrapping over rows would understate the CI by
roughly 11×. `Task.heldout_groups` and `ArmResult.heldout_cells_final` carry that
structure, and both halves of the gap are read off the same end-of-stream model.

## What is NOT established

- **Nothing about the compositional regime.** The only run is `--tiny` (p=5, n_ops=3, K=4, n=2,
  120 steps), where train accuracy is ≈ chance. Its `regime_is_discriminating=False` is a
  plumbing check, **not a result**.
- `SubspaceRestructure` has never been run.
- The Task-IL scope caveat has not been added to reports 137/138/139 or the retrospective.
- `reports/134:15`'s unrun-control claim has not been corrected.

## Next

1. Run `experiments/86` at n≥8, real scale (`--p 17 --K 10 --seeds 8`). Decide from the headline
   whether the regime discriminates.
2. If it does: `SubspaceRestructure` vs `EWCAnchor` **at matched protection strength** — the
   competent-control bar that deflated Benna-Fusi.
3. If it does not: harden the regime (more operators, deeper composition, tighter held-out
   fraction) before running any mechanism against it.
