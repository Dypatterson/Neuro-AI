# Report 141 — Bet B: the compositional regime IS discriminating (regime validation)

**Status:** COMPLETE. **Verdict:** **The regime discriminates.** On non-abelian
operator composition, replay does **not** saturate held-out operator pairs
(gap **+0.383** [+0.346, +0.418], 8/8 seeds positive), while the **matched abelian
control collapses the gap to −0.006** [−0.041, +0.033]. This is the first regime in
this program's history where the simple method demonstrably fails on a task it can
otherwise learn. **Type:** REGIME VALIDATION — explicitly **not** a graduation
attempt and **not** a mechanism result. **Date:** 2026-07-25.
**Charter:** [CONTEXT-B.md](../../CONTEXT-B.md). **Experiment:**
[`experiments/86_betb_compositional.py`](../../experiments/86_betb_compositional.py).
**Artifact:** [`headline.json`](headline.json) (committed).

## Preamble (per CLAUDE.md §Experiment preamble)

- **Active capability:** Bet B continual compounding-transfer (`CONTEXT-B.md`
  §"THE RE-GROUNDED TARGET").
- **Headline metric** per [RETROSPECTIVE](../../notes/RETROSPECTIVE-two-bets-2026-06-06.md)
  §"What would actually test the central thesis": the **held-out composition gap**
  `comp_test(trained pairs) − heldout_pair_acc`, both read off the same
  end-of-stream model, bootstrapped over **seed × held-out cell**.
- **Required controls** per `CONTEXT-B.md` §"THE TRACER BULLET": scratch
  denominator; 2×2 factorial; **abelian matched control**; joint-train ceiling;
  frozen-feature probe. All five executed (the provenance envelope enforces this).
- **Last verified result:** [Report 140](../140_harness_extraction_and_compositional_regime/report.md)
  — harness extracted, regime designed and standalone-validated.
- **Why now:** STATUS.md blocker (3) and the retrospective's §4 task-selection
  confound. Every mechanism result in this program is uninterpretable until a
  regime exists where the simple method fails.

## Result

n=8 seeds, K=2 (primitives → composition), 8000 steps/task, Class-IL (shared head),
64 seed×cell samples per arm. `git 8353c02`.

| Arm | trained-pair acc | held-out acc | **GAP** |
|---|---|---|---|
| floor (no replay, no consol) | 0.447 | 0.077 | **+0.370** [+0.340, +0.402] |
| **replay_only** | **0.520** | **0.137** | **+0.383** [+0.346, +0.418] |
| consol_only (EWC λ=0.01) | 0.154 | 0.083 | +0.071 [+0.039, +0.099] |
| replay_plus_consol | 0.189 | 0.121 | +0.068 [+0.033, +0.098] |
| **ABELIAN matched control** (replay_only) | **0.847** | **0.854** | **−0.006** [−0.041, +0.033] |

**Both verdict clauses hold.** The primary clause (replay_only gap CI-lo > 0.10)
passes with margin. The control clause — the abelian gap must *not* stay large —
passes decisively: it is centered on zero.

Per-seed, the symmetric gap is positive in **8/8** seeds (0.304, 0.344, 0.347,
0.362, 0.379, 0.422, 0.428, 0.475). The abelian gap is ≤0 in 6/8 and centered on
zero. Nothing here rests on a lucky seed.

## Why the abelian control is the whole result

A large gap on its own would only show the task is hard. The control changes
**exactly one thing** — `S_5` → `(Z_5)^3`, permutation action → translation — which
makes "pool the two operator embeddings" the *correct* algorithm. Same
architecture, same sizes, same step budget, same splits, same seeds.

Under that swap the model reaches **0.847 on trained pairs and 0.854 on held-out**
— it saturates both, and the gap vanishes. So:

- the architecture **can** learn two-operator composition within this budget;
- it **can** generalize to unseen operator pairs when the structure permits pooling;
- and it **fails** to do so precisely when generalization requires respecting
  non-commutative order.

That is the discriminating property, isolated. It is not available from the
symmetric arm alone.

## Honest scope — three limits, one of them my error

**1. Trained-pair accuracy is 0.520, not the 0.961 of the standalone validation.**
Report 140's design validation trained jointly for 20 000 steps and reached 0.961
on trained pairs. Here, in the continual setting at 8 000 steps/task, replay_only
reaches 0.520. So **property 2 is only partially met in this run**, and part of the
+0.383 gap is "composition is hard" rather than purely "recombination fails."

This is exactly the criticism that disqualified the `signed_add` candidate in
Report 140 (trained-pair 0.572), and it must not be waved away now that it is
inconvenient. What blunts it: the abelian control reaches **0.847** on the *same*
architecture and budget, so the shortfall is specific to non-abelian composition
rather than a general capacity limit. The clean fix is more steps, and the gap
would be expected to widen (as it did 20k → 60k standalone), not close. **Re-run at
20 000 steps before any mechanism claim is made against this regime.**

**2. The consolidation arms are degenerate — the 2×2 interaction is NOT readable.**
`consol_only` and `replay_plus_consol` reach only 0.154 and 0.189 on trained pairs.
The EWC anchor at λ=0.01, tuned on the modular toy, over-constrains this harder
regime and prevents the composition task from being learned at all. Their small
gaps (+0.071, +0.068) are the **"nothing learns, so nothing saturates"** artifact,
not evidence about consolidation. **No interaction claim is licensed by this run.**
λ needs re-tuning for this regime, matched to a protection strength that still
permits learning.

**3. The joint-train ceiling is INVALID as run.** I configured it at
`max_steps // 8` = 1000 steps against arms that ran 8000 — 1/8 the budget of what
it is supposed to bound. Its 0.215 is an under-training artifact and **must not be
cited as a ceiling.** Fix the config and re-run before using it.

The frozen-feature probe is valid and informative: held-out 0.084, at the floor's
level, so the learned circuit does **not** already answer held-out compositions.
The "frozen model already does it" escape is closed.

## What this does and does not establish

**Establishes:** a continual regime exists, in this repo, where a simple method
(replay) learns the trained distribution but **fails to generalize compositionally**,
and where a matched control confirms the failure is about recombination rather than
difficulty. Per the retrospective §4/§6 this is the missing precondition for every
mechanism comparison the program has run since Report 121 — the task-selection
confound is, for this regime, addressed.

**Does not establish:** anything about any mechanism. No consolidation recipe was
validly tested here (limit 2). `SubspaceRestructure` has still never been run.
"Replay + soft anchor do not saturate it" is measured for replay; the anchor arm
was degenerate, so that half remains inference.

## Next

1. **Re-run at 20 000 steps/task** to lift trained-pair accuracy toward the 0.961
   standalone level and remove limit 1. Fix the ceiling budget while doing so.
2. **Re-tune EWC λ** for this regime — matched protection that still permits
   learning — so the 2×2 becomes readable.
3. **Then, and only then:** `SubspaceRestructure` vs `EWCAnchor` at matched
   protection. That is the first genuine test of "does a restructuring
   consolidation manufacture what protection cannot," which `CONTEXT-B.md` §8 has
   recorded as unbuilt since the fork.
