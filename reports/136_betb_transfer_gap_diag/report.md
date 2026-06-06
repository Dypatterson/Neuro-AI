# Report 136 — Bet B diagnostic: WHY do new-alphabet tasks get zero head start?

**Status:** BANKED — **STAGE-2 TARGET CONFIRMED (the good outcome): the reusable circuit exists; ordinary training destroys it.**
**Bet:** B (CONTEXT-B.md). **Type:** DIAGNOSTIC (gap drill-down for Report 135) — picks the Stage-2 lever; not a graduation run.

## TL;DR

Report 135 showed new-alphabet tasks get ~zero head start. This diagnostic finds **why**, and the answer is the *good* one. **A reusable "+" circuit already exists in the stream's shared MLP:** freeze it onto a brand-new alphabet and the new task groks **3.3× faster than from scratch** [CI 1.94, 4.94], **5.2× faster than a frozen *random* MLP** [3.21, 7.14] (n=8). The random-frozen control is *slower* than scratch (0.65×), so the speed-up is the MLP's *content*, not a freezing trick. **But ordinary training destroys it:** leave that exact MLP *trainable* (warm-start) and transfer collapses to 1.13× ≈ scratch — freezing beats warm-starting **2.91×** [1.90, 4.06]. So the zero new-symbol transfer in Report 135 was never a missing abstraction — it's a **protection failure**: fresh random embeddings dump large early gradients into the shared circuit and drag it off its clean "+" before the embeddings converge. (Bonus: the *plain*, no-replay stream's MLP is as useless as random — only the replay-maintained `raw_full` MLP stays a clean circuit. Keeping the circuit needs replay during the stream **and** protection during new-task learning.) **→ Stage 2 is now precisely specified and pre-validated with measurable headroom (3.3× available, the stream captures ~0): a two-timescale split — protect/slow the shared circuit, fast-adapt only the new embeddings — does it manufacture the missing cross-block transfer and beat plain replay?** Robustness: freeze_full beats scratch in 7/8 seeds (one outlier seed produced a poor stream circuit); freeze_joint > 1 in 7/8.

---

## Preamble (per CLAUDE.md)

- **Active capability:** Continual compounding-transfer (Bet-B target, CONTEXT-B §8). This is a **diagnostic** that decides *which* Stage-2 mechanism (CONTEXT-B §5) to build — not a graduation experiment.
- **Headline metric per [CONTEXT-B.md:161-170](../../CONTEXT-B.md) (§5 Stage-2 framing):** new-block **transfer ratio** T = steps-to-crit(scratch) / steps-to-crit(arm). The load-bearing contrasts: (a) **reusable** = frozen trained MLP beats a frozen random MLP (content ratio CI-lo > 1); (b) **protection** = freezing the MLP beats warm-starting it trainable (warm-start/freeze CI-lo > 1). Multi-seed (n=8), bootstrap CIs.
- **Why now:** Report 135 found new-alphabet tasks transfer ≈ 0 (FTSR ≈ 1). Before building a Stage-2 consolidator, find the bottleneck: is a reusable "+" circuit (a) present-but-destroyed-by-training (→ protect it, two-timescale) or (b) never-formed (→ a different lever)? The user explicitly chose "diagnose the gap first."
- **Last verified result:** Report 135 (exp81): surface rehearsal is the retention floor; cross-block transfer ≈ 0; FTSR high only within an add/sub pair.

## The probe — a frozen-MLP transfer ladder

Model = shared `embed(token)` → shared `MLP` ("± circuit") → per-task head. For a held-out **new alphabet** block, only the token→circle map is genuinely new; the MLP *could* transfer. Learn that new block with the MLP sourced/frozen different ways, vs full scratch:

| arm | MLP source | MLP trainable? | isolates |
|---|---|---|---|
| `scratch` | fresh | yes | denominator |
| `freeze_rand` | random init | **frozen** | the freezing penalty per se |
| `freeze_plain` | plain sequential stream | **frozen** | reusable content (no-replay stream) |
| `freeze_full` | raw_full sequential stream | **frozen** | reusable content (best-retained stream) |
| `freeze_joint` | joint-trained (all blocks) | **frozen** | reusable content (best-case circuit) |
| `warmstart_full` | raw_full stream | yes (trainable) | does *ordinary* training keep or lose it? |
| `warmstart_joint` | joint | yes (trainable) | same, from the best circuit |

**Decision logic:**
- frozen trained MLP ≫ frozen random ⇒ **a reusable circuit exists.**
- frozen ≫ warm-start ≈ scratch ⇒ **ordinary training destroys it ⇒ Stage 2 = protect/slow the circuit (two-timescale).** Headroom = the freeze-vs-unfrozen gap.
- even joint frozen ≈ random ⇒ **no reusable circuit ⇒ Stage 2 needs a different lever** (shared embedding geometry / circle prior), not MLP protection.

Drill-down: cross-block embedding **alignment** (do per-block circles share a 2-D subspace?) — explains *how* any transfer is/ isn't carried.

## Method

- Substrate & stream identical to `exp81` (CONTEXT-B §8): shared embed(64) + 2-layer ReLU MLP(256) + per-task head; K=10 add/sub on rotating disjoint blocks; AdamW wd=1.0 full-batch grokking recipe; p=17, frac=0.7, crit=0.90 held-out.
- Held-out probe block = a fresh alphabet never in the stream (base = n_blocks·p).
- `max_steps=8000` (probes grok reliably so the ratio denominator isn't censored), `joint_steps=12000` (a 10-task joint objective needs a generous budget to be a fair best-case). n=8 seeds, bootstrap CIs. Sharded (`--seed-start`) + merge parallel runner. MPS/CPU.
- Anti-homunculus: pure measurement; no mechanism added.

## Results (K=10, n=8, p=17, max_steps=8000, joint_steps=12000, chance=0.059)

New-block (held-out fresh alphabet) learning, steps-to-crit and transfer ratio T = scratch/arm:

| arm | MLP source / mode | steps-to-crit | T = scratch/arm [95% CI] |
|---|---|---|---|
| scratch | fresh, all trainable | 5238 | 1.00 (denominator) |
| freeze_rand | random init, **frozen** | 8000 (cap) | **0.65** [0.56, 0.75] |
| freeze_plain | plain stream, **frozen** | 8000 (cap) | 0.65 [0.57, 0.75] |
| **freeze_full** | raw_full stream, **frozen** | **2700** | **3.31** [1.94, 4.94] |
| freeze_joint | joint-trained, **frozen** | 3500 | 1.62 [1.30, 1.97] |
| warmstart_full | raw_full stream, **trainable** | 4850 | 1.13 [0.94, 1.31] |
| warmstart_joint | joint, **trainable** | 4462 | 1.24 [1.05, 1.43] |

**Reusable-circuit test** (freeze_rand / freeze_arm — both frozen, isolates MLP *content*; > 1 ⇒ reusable):
- freeze_full = **5.18× [3.21, 7.14]** · freeze_joint = 2.53× [2.03, 3.07] · freeze_plain = 1.00× (no better than random).

**Protection test** (warmstart / freeze — same MLP source; > 1 ⇒ freezing beats adapting ⇒ protect it):
- full = **2.91× [1.90, 4.06]** · joint = 1.36× [1.06, 1.83].

**Cross-block embedding alignment** (1 = blocks share a plane; ~0 = independent circles): raw_full 0.301, joint 0.111, plain 0.108 — low for all; the MLP transfer is **not** carried by aligned embedding geometry (it's a subspace-flexible "+" operation). Secondary drill-down.

**Per-seed robustness:** freeze_full T = [3.4, 2.9, 8.3, 1.4, 4.1, 0.7, 3.9, 1.9] → beats scratch 7/8 (seed 5 outlier = a poor stream circuit that run); freeze_joint > 1 in 7/8.

## Verdict

**STAGE-2 TARGET CONFIRMED — protection failure, not a missing circuit.** Two CI-clean facts at n=8: (1) a reusable "+" circuit exists (frozen stream MLP transfers 3.3× vs scratch, 5.2× vs a frozen random MLP; the random-frozen control being *slower* than scratch rules out a freezing artifact); (2) ordinary training destroys it (the same MLP left trainable transfers only 1.13× ≈ scratch; freezing wins 2.91×). The zero new-symbol transfer in Report 135 is therefore a **protection failure**, with large measurable headroom (3.3× available, the unfrozen stream captures ~0). Mechanism: at new-task onset the embeddings are random, so the frozen-correct MLP produces wrong outputs and large gradients flow into the *shared circuit*, pulling it off its clean "+" before the new embeddings converge — joint training is fragile in exactly the way a fast/slow split would fix.

**Discipline:** controls are load-bearing here — `freeze_rand` (the freezing-penalty control, 0.65× < scratch) is what licenses reading freeze_full's 3.3× as content; `warmstart_full` (same MLP, trainable) is what isolates protection from content. `freeze_plain ≈ random` adds the honest nuance that the circuit also needs replay to *survive the stream* in the first place. Not over-claimed: one seed (5) shows no transfer (poor circuit that run) — banked as variance, CI-lo still 1.94.

## What this licenses — Stage 2, now precisely specified

A **two-timescale consolidation** experiment (CONTEXT-B §5), pre-validated by this diagnostic:
- **Mechanism:** the shared "± circuit" (MLP) is the **slow** component, **protected** during new-task learning (frozen or strongly LR-slowed); the **embeddings (+head)** are the **fast** component, adapting per new alphabet. An offline consolidation pass updates the slow circuit *gently* over replayed tasks so it still integrates new structure without catastrophic churn. Anti-homunculus: fixed architectural timescales / a smooth slow-LR, no supervisor.
- **Headline (the §8 graduation bar):** does the two-timescale split **manufacture** cross-block forward transfer — new-alphabet FTSR > 1, CI-disjoint, **beating a plain replay buffer** (which this report shows captures ~0) — while holding retention? The fully-frozen case (3.3×) is the easy upper bound; the real test is whether a *slowly-consolidating* circuit keeps most of that transfer **and** still learns.
- **Guard the redundancy trap (§5):** the win must be a measured delta over plain replay / a single-timescale ablation, and the slow circuit must still *improve* (a permanently-frozen circuit is not "consolidation"). Report the no-protection and full-freeze brackets so the slow-consolidation arm is read against both.

## Reproduce

```
for S in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python \
  experiments/82_betb_transfer_gap_diag.py --p 17 --K 10 --max-steps 8000 --joint-steps 12000 \
  --seeds 1 --seed-start $S --device cpu --out reports/136_betb_transfer_gap_diag/shards/seed$S.json & done; wait
PYTHONPATH=src .venv/bin/python experiments/82_betb_transfer_gap_diag.py --merge \
  reports/136_betb_transfer_gap_diag/shards/seed*.json --out reports/136_betb_transfer_gap_diag/k10_n8.json
```

