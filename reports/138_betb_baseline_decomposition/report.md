# Report 138 — Bet B: what is the baseline's compounding transfer MADE OF? (decomposition drill-down)

**Status:** COMPLETE (n=64; adversarially verified — design audit `ws9bfc18f` + report verify `wayu4dyid`).
**Verdict:** on the **new-alphabet (x-block) forward-transfer** headline, **interleaved replay** is the only
POSITIVE ingredient; the offline "consolidation" pass adds **nothing positive** (never POSITIVE, slight −lean) —
*but it is not globally inert: it helps same-alphabet (within-block) learning.* Scope: rehearsal recipes only.
**Type:** DRILL-DOWN / characterization — **NOT a graduation attempt.** **Charter:** [CONTEXT-B.md](../../CONTEXT-B.md)
§5/§8 (the Report-137 NEXT). **Experiment:**
[experiments/84_betb_baseline_decomposition.py](../../experiments/84_betb_baseline_decomposition.py) (`exp84`).
**Design-audited before trusting numbers:** workflow `ws9bfc18f` (4 lenses + synthesis).

## Preamble (per CLAUDE.md)

- **Active capability:** Bet B — continual compounding-transfer ([CONTEXT-B.md §8](../../CONTEXT-B.md)).
- **Headline metric per [CONTEXT-B.md:234-238](../../CONTEXT-B.md) + the Report-137 NEXT at
  [CONTEXT-B.md:330-335](../../CONTEXT-B.md):** the new-alphabet (x-block) **FTSR**, *decomposed* across arms.
  Because the metric is a **decomposition** of the §8 FTSR headline (not a new graduation gate), this is a
  **drill-down**, labelled as such.
- **Required controls per [CONTEXT-B.md:240-244](../../CONTEXT-B.md):** `scratch` (FTSR denominator); the new
  arms ARE the isolation controls (see Arms). Anchor: `plain` reproduces Report 137. Anti-homunculus: all arms
  are fixed schedules with content-blind replay/rehearsal.
- **Last verified result:** [Report 137](../137_betb_two_timescale/report.md) — two-timescale FREEZE NO
  GRADUATION; the baseline (`plain` = replay + offline consolidation) **compounds** (x-block FTSR 1.6→10.4,
  7/8 seeds).
- **Why now:** Report 137's NEXT. The on-target positive lives in the *simple* baseline; before testing any
  further brain-distinctive recipe (the §5 anti-redundancy guard), we must know **what** in the baseline carries
  the compounding.

## The question

`plain` (the 137 baseline that compounds) bundles **three** ingredients:

1. **Interleaved replay** during fast per-task learning (buffered examples mixed into the new-task loss).
2. An **offline consolidation pass** before each task (full-batch CE over the buffer).
3. The **extra optimizer compute** that offline pass spends (~`consol_steps` MLP-trainable steps/task, **not**
   counted in the FTSR denominator).

Scope note (audit blocker #2): the harness's "consolidation" is **rehearsal, not restructuring**. So this
experiment can only decompose the baseline's compounding into **rehearsal** ingredients. It **cannot** test
whether a *restructuring* consolidator manufactures structure, and it **does not license abandoning the
modular-arithmetic domain.** That is the genuinely-next experiment (a restructuring Stage-2 recipe swapped into
this same harness — [CONTEXT-B.md:161-170](../../CONTEXT-B.md)).

## Method

One model, a K=10 add/sub-on-rotating-blocks stream (p=17; new alphabet every 2 tasks). Five arms; dynamics
bit-identical to `exp83` via import, so `plain` reproduces Report 137 (the design audit verified byte-identity
empirically — identical steps, retention matrix, scratch denominator, per-k FTSR).

| arm | interleaved replay | offline pass | role |
|---|---|---|---|
| `scratch` | — | — | FTSR denominator (fresh model per task) |
| `plain` | ✓ | full-batch | the 137 baseline (**anchor**) |
| `replay_matched` | ✓ | minibatch, **compute-matched** to `plain` | isolates *full-batch-offline structure* from *raw rehearsal compute* |
| `replay_no_consol` | ✓ | ✗ | isolates the whole offline pass |
| `no_replay_no_consol` | ✗ | ✗ | no-rehearsal floor |

**Additive decomposition** of x-block FTSR (`plain − floor = Δ_replay + Δ_compute + Δ_structure`):

- `Δ_replay = FTSR(replay_no_consol) − FTSR(floor)` — interleaved replay's own contribution.
- `Δ_compute = FTSR(replay_matched) − FTSR(replay_no_consol)` — effect of equal-compute extra rehearsal.
- **PRIMARY** `Δ_structure = FTSR(plain) − FTSR(replay_matched)` — does the full-batch offline pass beat
  *equal-compute* minibatch rehearsal? (the compute-controlled question).
- `Δ_consol = FTSR(plain) − FTSR(replay_no_consol) = Δ_compute + Δ_structure` — offline-pass total
  (**compute-confounded**; reported as a drill-down only).

**Statistics (the methodological fix this report introduces).** FTSR is a heavy-tailed **ratio** (an arm
converging in one eval period → a huge speedup), and speedups are inherently **multiplicative**. So all
inferential statistics are computed in **log-FTSR** space (natural log of `scratch_steps/arm_steps`), with
paired within-seed bootstrap CIs; raw FTSR means are reported alongside for continuity with Report 137. The
equivalence margin is **EQUIV = ±0.22 log ≈ within a 1.25× ratio**. Three-state verdict per delta:

- **POSITIVE** — CI-lo > 0 (ingredient adds x-block transfer).
- **EQUIVALENT** — CI fully within ±EQUIV (genuinely inert at this resolution).
- **INCONCLUSIVE** — straddles 0 but extends beyond EQUIV (under-powered; report MDE, add seeds — **do not**
  conclude inert). [The false-negative guard the design audit required; the inverse of 137's n=2 false-PASS.]

**n = 64** (16 shards × 4 seeds, CPU, single-threaded, merged; the n=48 sub-sample gave the identical picture).
The log-scale half-width is 0.19 at n=48 and **0.17 at n=64** — below EQUIV, enough to resolve EQUIVALENT vs a
real sign rather than land in INCONCLUSIVE.

**"Compute-matched" = optimizer-STEP-matched, not buffer-coverage- or FLOP-matched** (a scope limit the
verification surfaced). `replay_matched`'s minibatch pass (batch=256) sees a *shrinking* fraction of the growing
buffer per step (≈63% at k=2 down to ≈14% by k=9) vs the full-batch pass's 100%. So `Δ_structure` EQUIVALENT
means *at matched optimizer steps*; a full-batch advantage that lives specifically in per-step buffer coverage
(a lower-variance, all-task-balanced gradient) would not be captured here. This does not threaten the x-block
headline — `replay_no_consol` (no offline pass at all) is already the *strongest* x-block arm — but it bounds
what the EQUIVALENT licenses.

## Design audit (workflow `ws9bfc18f`, run BEFORE trusting numbers)

Four independent lenses (harness/anchor, confound, statistics, interpretation/charter) + synthesis. Outcome:

- **Harness/anchor: PASS, verified empirically.** `plain` is byte-identical to `exp83` (the 137 anchor
  reproduces); arms are RNG-independent; gates correct.
- **Two blockers on *honoring the verdict* (not the harness), both fixed in this revision:**
  1. *Under-powered green-light* — the original frozen rule fired the hard-to-reverse "abandon the domain"
     action on the low-power straddle-0 branch (n=8 misses a true +0.5 effect 63–84% of the time). **Fix:** n≥24
     (run uses 64), log-scale statistics, explicit INCONCLUSIVE state + equivalence margin, MDE reported.
  2. *Recipe→domain over-reach* — a null on a rehearsal-only recipe cannot license "the domain can't
     discriminate consolidation." **Fix:** verdict scoped to "this rehearsal recipe is inert"; no domain-abandon
     trigger; no "linearly reachable by rehearsal" causal claim; the real test (restructuring consolidator) named
     as NEXT.
  3. *Compute confound* — `plain` gets ~`consol_steps` extra uncounted MLP steps/task. **Fix:** the
     `replay_matched` compute-matched arm makes `Δ_structure` (not `Δ_consol`) the primary, compute-controlled
     headline.

## Results (n = 64; 16 shards × 4 seeds; the n=48 sub-sample gave the identical picture)

**Decomposition of x-block FTSR (log-FTSR scale; paired within-seed bootstrap; EQUIV = ±0.22 log ≈ within 1.25×):**

| delta | log [95% CI] | geomean ratio | state |
|---|---|---|---|
| **Δ_replay** (replay_no_consol − floor) | **+0.66 [+0.52, +0.80]** | **×1.94 [1.69, 2.24]** | **POSITIVE** |
| Δ_compute (replay_matched − replay_no_consol) | −0.08 [−0.27, +0.11] | ×0.92 [0.77, 1.11] | INCONCLUSIVE (near-null, slight −lean) |
| **Δ_structure** (plain − replay_matched) — **PRIMARY** | **+0.06 [−0.11, +0.22]** | **×1.06 [0.89, 1.24]** | **EQUIVALENT** |
| Δ_consol (plain − replay_no_consol; compute-confounded) | −0.02 [−0.22, +0.17] | ×0.98 [0.80, 1.19] | INCONCLUSIVE (near-null) |

Achieved MDE (Δ_structure half-width) ≈ 0.17 log < EQUIV. **Caveat (verification):** the EQUIVALENT call sits at
the *upper edge* of the band (CI-hi +0.217 vs +0.22, slack 0.003) and the point estimate flipped sign n=48
(−0.009) → n=64 (+0.058) while the state held. So the robust claim is **"Δ_structure is never POSITIVE"** (the
offline pass's full-batch structure never *beats* equal-compute minibatch rehearsal), not a hard equivalence;
`eval_every=100` step-quantization is a second resolution floor on fast late tasks.

**Anchor — PASS.** `plain` raw x-block FTSR = **5.51 [4.60, 6.56]**, in-band with Report 137's ≈5.44; per-k
1.79 → 9.21 reproduces 137's 1.6 → 10.4 shape. Harness fidelity confirmed (the design audit had already verified
byte-identity empirically).

**Compounding survives the offline pass being removed.** The 137 signature (last-block FTSR ≫ first-block,
CI-disjoint) holds in **every** arm, including `replay_no_consol` (per-k raw FTSR **1.69 → 11.68**; last−first
log +1.50 [+1.16, +1.81]). So the compounding does not need the offline pass.

**Retention.** `replay_no_consol` 0.979 ≈ `plain` 0.980 (Δ = −0.002, held); `replay_matched` 0.984; the
no-rehearsal floor collapses to **0.196 ≈ chance** — interleaved replay alone carries retention; the offline
pass adds nothing to it.

**Within-block (same-alphabet) — the offline pass is NOT globally inert (paired n=64).** On the within-block
tasks (sub-after-add, *same* alphabet) the offline rehearsal pass *helps*: `Δ_compute` = +0.20 [+0.08, +0.31]
(×1.22) **POSITIVE**, and the whole offline pass leans positive (`Δ_consol` = +0.11 [−0.02, +0.23], ×1.11,
INCONCLUSIVE-leaning-positive); `Δ_structure` within-block = −0.09 [−0.23, +0.05] (≈null — so again the benefit
is *rehearsal compute*, not full-batch *structure*). Raw within-block FTSR: plain 30.4, replay_matched 32.4,
replay_no_consol 27.2, floor 11.8. **Reading:** the offline pass reinforces *same-operation* skill (it rehearses
buffered same-alphabet examples) but does **not** manufacture *new-alphabet* transfer — which is exactly why it
is positive within-block yet adds nothing to the x-block headline. The target behavior (cross-alphabet transfer)
comes specifically from interleaved replay's exposure, not offline rehearsal.

## Verdict

**On the new-alphabet (x-block) forward-transfer headline, interleaved replay is the only POSITIVE ingredient;
the offline "consolidation" pass adds nothing positive — never POSITIVE, slight negative lean.** (Scope:
rehearsal recipes only — see below.) Evidence: (1) interleaved replay roughly doubles new-task speedup over the
no-rehearsal floor (Δ_replay POSITIVE ×1.94) and both the 1.6→10.4 compounding and the retention survive with
the offline pass *removed* — `replay_no_consol` is in fact the *strongest* x-block arm (6.33 ≥ plain 5.51);
(2) the offline pass's full-batch *structure* never beats equal-compute minibatch rehearsal at matched optimizer
steps (PRIMARY Δ_structure EQUIVALENT/never-POSITIVE, ×1.06); (3) its compute and total x-block contribution
center near zero with a slight *negative* lean (extra rehearsal mildly hurts new-alphabet convergence on some
seeds).

**But the offline pass is not globally inert:** within-block (same-alphabet) it is POSITIVE (Δ_compute +0.20).
So the precise claim is: **offline rehearsal consolidates same-operation skill but does not manufacture
new-alphabet transfer; that transfer is carried by interleaved replay.**

This is a clean drill-down null **on the rehearsal form of consolidation** (for new-alphabet transfer), reported
with the false-negative guards the design audit required (log-scale + n=64 + equivalence margin + explicit
INCONCLUSIVE state + MDE) and a bank-time verification pass. No PASS/GRADUATES token; the metric is a
decomposition of the §8 FTSR headline, so this is a drill-down.

## What this licenses / does NOT license

- **Licenses:** *"Adding an offline full-batch rehearsal pass over the buffer manufactures no new-alphabet
  forward transfer beyond interleaved replay in this domain; the baseline's compounding is interleaved replay."*
  And: any future *restructuring* Stage-2 recipe must beat **interleaved replay alone** (`replay_no_consol`,
  x-block FTSR ≈6.3, the 1.69→11.68 compounding) — not the full `plain` arm, which is no stronger.
- **Does NOT license:** "the modular-arithmetic domain cannot discriminate consolidation," nor any "structure is
  linearly reachable by rehearsal" causal claim, nor "the offline pass is *globally* inert" (it is POSITIVE
  within-block). The null is specifically on **new-alphabet transfer**. Only the *rehearsal* form of
  consolidation was tested; the *restructuring/distilling* consolidator the charter proposes
  ([CONTEXT-B.md:161-170](../../CONTEXT-B.md)) was not. This result does **not** support abandoning the domain —
  if anything it sharpens the domain's value (the compounding is real, robust, and now attributed to interleaved
  replay, with a clean bar for the next test).

## NEXT — the genuinely-new test

Swap a **restructuring** Stage-2 consolidator (not rehearsal) into this same harness and re-measure
`Δ_structure` against `plain` ([CONTEXT-B.md:161-170, 248-251](../../CONTEXT-B.md)). The rehearsal-only ablation
here cannot speak for it either way; that is the actual open question.
