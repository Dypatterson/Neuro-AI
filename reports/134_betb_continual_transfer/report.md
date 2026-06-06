# Report 134 — Bet B tracer bullet: continual compounding-transfer

**Status:** BANKED — **PARTIAL (easy regime) → FIRST POSITIVE (tight-buffer regime, see ADDENDUM).**
Forward transfer is REAL+strong+retained but a **known phenomenon**, and sleep-v1 ≈ replay where the
unlimited buffer already saturates. **But starve the buffer and replay BREAKS (forgets to chance), and
self-generated "pseudo-sleep" beats it on retention (+0.45 [0.39,0.53]) AND transfer (FTSR 11.7 vs 3.8),
n=8** — the session's first win for the brain-shaped mechanism (a *known* mechanism: pseudo-rehearsal).
Scramble control invalid (redesign). Charter: [CONTEXT-B.md §8](../../CONTEXT-B.md). Experiment:
[`experiments/80_betb_continual_transfer.py`](../../experiments/80_betb_continual_transfer.py).

## Preamble

- **Active capability:** Continual compounding-transfer (the re-grounded Bet-B target, CONTEXT-B §8). NOT representation.
- **Headline (CONTEXT-B §8):** Forward-Transfer Speedup Ratio FTSR_k = from-scratch steps-to-criterion / sequential-with-sleep steps-to-criterion. PASS = positive transfer (FTSR>1 CI-disjoint on both transfer tasks) AND sleep beats replay-only (delta CI-lo>0) AND T1/T2 retention ≥ 90%.
- **Controls:** from-scratch (denominator), joint (ceiling), **replay-only (load-bearing)**, scrambled-T3′, frozen-in-context.
- **Why:** first falsifiable test of "experience compounds into transferable skill" on the actual target.

## Result (modular arithmetic, p=17, AdamW wd=1.0 full-batch grokking regime, n=8 seeds)

| FTSR (from-scratch steps / arm steps) | T1 | T2 (sub, shares embedding) | T3 (add, new symbols) |
|---|---|---|---|
| **sleep** (surprise replay) | 0.99 | **19.1** [13.6, 25.0] | **3.6** [2.2, 5.2] |
| **replay-only** (uniform) | 0.99 | 18.3 [12.3, 24.8] | 5.6 [1.4, 13.0] |
| scramble-T3′ | 1.17 | 15.8 | 2.2 [1.3, 3.2] |

- **sleep − replay delta:** T2 [0.0, 1.9] (straddles 0); T3 [−8.4, 2.0] (straddles 0; replay T3 high-variance).
- **Retention end-of-stream:** T1 1.00, T2 1.00, T3 0.94. **Joint ceiling:** 0.96/0.95/0.96 (matched).
- **Gates:** positive_transfer ✅ | retains_90 ✅ | **beats_replay_only ❌** | beats_replay_T3_only ❌.

## Verdict — three honest layers

**1. The tracer bullet WORKS, and forward transfer is real.** A single small model learns 3 tasks in
sequence, genuinely generalizes (grokking), retains all three (matches the joint ceiling), and
**compounds**: a related task groks ~18× faster after the first. That is the re-grounded target's core
property — *experience making future learning faster* — demonstrated cleanly for the first time. The
harness is sound and reusable.

**2. But the transfer is a KNOWN result — not a discovery.** Grokking-transfer between modular-arithmetic
tasks via shared embeddings is published (GrokTransfer arXiv:2504.13292 reports ~5× acceleration of a
second arithmetic task; "Small Models, Smarter Learning" arXiv:2505.18369 shows the joint-training
version). Our ~18× on the closely-related pair is consistent-or-stronger, but the *phenomenon* is not
new. So this is the SAME shape as Report 133 (SGNS): a clean, valid, **redundant** confirmation of a
known effect. It establishes the baseline; it is not the prize.

**3. The project's distinctive bet NULLS (v1).** The one thing that would have been genuinely novel —
**emergent consolidation ("sleep") manufacturing transfer beyond a plain replay buffer** — is not
supported: sleep ≈ replay on both transfer tasks (deltas straddle 0). The strong transfer is carried by
**replay + the shared representation**, NOT by the surprise-prioritized sleep recipe. Per CONTEXT-B §8
and the user's binding note, **this is iterate-fuel: swap the sleep recipe, not abandon the direction.**

## Issues found (do not paper over)

- **Scramble control INVALID.** Permuting T3′'s token-values doesn't break the *operation* transfer (the
  thing that actually carries to T3), so it still shows ~16× — it cannot certify structural-vs-confound.
  The T2 transfer (18×, same tokens, related op) is structurally unambiguous regardless, but T3's
  structurality is **unverified**. Redesign: T3′ must break the reusable operation (e.g., a structured
  non-group function), not relabel an isomorphic addition.
- **Gate bug (fixed):** the original "compounding = T3 faster than T2" was wrong — T2 has the largest
  transfer by design (shared embedding). Corrected to "positive speedup on both transfer tasks."
- The auto-verdict correctly reads PARTIAL (no over-claim; discipline held — the 2-seed "sleep>replay on
  T3" was noise, killed at 8 seeds, same pattern as 126/133).

## What this licenses (the iterate path)

The distinctive question is still open and the harness is ready. Next recipes for the **swappable** sleep
mechanism, and the conditions that would let sleep beat plain replay:
1. A **harder regime where plain replay is insufficient** — longer task streams (loss-of-plasticity
   territory), tighter memory budgets, or tasks where naive replay shows negative transfer. Sleep can
   only "win" where replay alone doesn't already saturate the transfer (here it does).
2. A consolidation recipe that **restructures** rather than rehearses — e.g., generative/abstracting
   replay, or replay that compresses toward the shared latent factor.
3. A **redesigned scramble** that breaks operation-transfer, so any sleep advantage is certified structural.

Banked as: a working tracer bullet that confirms (known) strong forward transfer and **nulls the
sleep-beats-replay bet in v1** — a clean, honest negative on the distinctive claim, on the real target.

---

## ADDENDUM (iterate path, 2026-06-06) — where the buffer breaks, self-generated "sleep" WINS

Report 134's null was "sleep ≈ plain replay" *in the easy regime where replay already saturates the
transfer.* Per the charter's iterate rule, we hunted the regime where plain replay **breaks**.

**Diagnostic — replay DOES break (tight buffer).** Cap the replay buffer at 2–5 stored examples per
task: plain replay (and the v1 surprise-sleep, which just re-samples the same tiny buffer)
**catastrophically forget** the old tasks — retention on T1/T2 collapses to **~0.03–0.06 (chance =
0.059)** while only the most-recent task survives. So the easy-regime null was an artifact of an
unlimited buffer; starve it and there is real room to win.

**Result — self-generated rehearsal ("pseudo-sleep") fills the gap (cap=2, n=8 seeds).** New swappable
recipe (`--sleep`-class, arm `pseudo`): before each task the model **generates its own rehearsal data** —
random problems for the old tasks, labeled by its *own current prediction* (pseudo-rehearsal, Robins
1995 / Sleep-Replay-Consolidation; anti-homunculus: the model just queries itself), storing **zero** raw
examples.

| arm (cap=2) | T1 retain | T2 retain | FTSR T2 | FTSR T3 |
|---|---|---|---|---|
| replay (raw, 2 stored) | 0.05 | 0.03 | 3.8 | **0.9 (negative!)** |
| sleep v1 (raw, surprise) | 0.05 | 0.06 | 3.8 | 1.0 |
| **pseudo (self-generated)** | **0.47** | **0.49** | **11.7** | **1.5** |

- **Retention gain pseudo − replay = +0.41 [0.30, 0.54] (T1), +0.45 [0.39, 0.53] (T2)** — CIs clear of 0.
- **Transfer bonus:** pseudo's preserved structure transfers ~3× better (FTSR T2 11.7 vs 3.8); raw replay
  even shows **negative** transfer on T3 (forgetting hurts new learning), pseudo stays positive.

**Verdict: FIRST POSITIVE for the distinctive direction.** In the regime where the simple baseline
breaks, a brain-shaped consolidation — *regenerate your past from your own weights instead of storing
it* — beats raw replay on **both retention and transfer**, multi-seed, CI-backed. It validates the
strategy: test the brain mechanism where the baseline *fails*, not where it already wins.

**Caveats (keep it sized right):** (1) **Known mechanism** — pseudo-rehearsal / generative replay is
decades old; this is "known-but-positive," not a discovery (same redundancy honesty as the SGNS run).
(2) **Partial** — retention ~0.48, not 1.0 (self-labels degrade as the model drifts). (3) **Tiny toy** —
3 modular-arithmetic tasks, ~10⁴-param model; the win is on the **storage** axis (pseudo stores 0 raw vs
replay's 2; it spends more *sleep-compute* to regenerate — which is the point, but state it).

**Next:** push pseudo toward full retention (more/better generation; interleave generation *during*
learning, not just before); test **longer task streams** (loss-of-plasticity territory) and whether
pseudo's advantage grows as the stream lengthens; redesign the scramble; eventually a harder domain.
