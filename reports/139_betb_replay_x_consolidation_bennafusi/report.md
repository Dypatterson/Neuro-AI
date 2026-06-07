# Report 139 — Bet B: replay × CONSOLIDATION interaction, recipe 1 = Benna-Fusi (graduation attempt)

**Status:** COMPLETE (n=48 × 3-g; adversarially verified `wuybsd91u` + matched-protection control). **Verdict:**
**FIRST PASS of the §8 interaction gate** — additive, recombinant, scope-limited (RC beats both parents at all 3 g;
*not* "consolidation manufactures structure replay can't"). **Type:** GRADUATION ATTEMPT for the CONTEXT-B §8
interaction headline (the first real test of Bet B's distinctive thesis). **Charter:** [CONTEXT-B.md](../../CONTEXT-B.md)
§5/§8 + the Terminology block. **Experiment:**
[experiments/85_betb_replay_x_consolidation.py](../../experiments/85_betb_replay_x_consolidation.py) (`exp85`).
**Recipe chosen by the user:** Benna-Fusi multi-timescale synaptic consolidation (the domain-expert grounding's
#1 corpus mechanism; the principled graded fix for why [Report 137](../137_betb_two_timescale/report.md)'s binary
freeze nulled).

## Preamble (per CLAUDE.md)

- **Active capability:** Bet B continual compounding-transfer ([CONTEXT-B.md §8](../../CONTEXT-B.md)).
- **Headline per [CONTEXT-B.md:267-272](../../CONTEXT-B.md):** the replay×consolidation **interaction** in
  log-FTSR over new-alphabet (x-block) tasks — `d1 = logFTSR(RC) − logFTSR(replay_only)` CI-lo>0 **AND**
  `d2 = logFTSR(RC) − logFTSR(consol_only)` CI-lo>0 (RC beats **both** parents) **AND** RC compounds **AND**
  retention held. This is a graduation gate, not a drill-down.
- **Controls per [CONTEXT-B.md:240-246](../../CONTEXT-B.md):** the full 2×2 (`floor / replay_only / consol_only
  / replay_plus_consol`) + `scratch` denominator. Anchor: `replay_only` reproduces exp84's `replay_no_consol`
  (x-block FTSR ≈6.3 at K=10/n≥48). Anti-homunculus: BF diffusion is a fixed local linear relaxation (no metric
  read, no task identity); engaging at a task boundary is a fixed schedule (as 137's freeze was). Fence: no
  SVD/eig — iterative local updates only.
- **Last verified:** [Report 138](../138_betb_baseline_decomposition/report.md) — interleaved replay carries the
  baseline compounding; offline *rehearsal* adds nothing to transfer. Bar to beat = `replay_only` (≈6.3).
- **Why now / why this recipe:** user-selected first consolidation recipe; Benna-Fusi is the graded, emergent
  version of the binary freeze that nulled in 137.

## The mechanism (Benna-Fusi, on the shared MLP)

Each shared-MLP weight `w` is the visible variable `u₁` of a chain `u₁…u_m` with geometric capacitances
`C_k = 2^(k-1)` and fixed coupling `g`. After every optimizer step (which injects plasticity into `u₁` via Adam),
the chain diffuses one step — `du_k = (g/C_k)[(u_{k-1}−u_k)+(u_{k+1}−u_k)]`, reflective boundaries — so slower
variables protect old structure and pull `u₁` back toward the consolidated circuit *without hard-freezing*
(137's failure mode), and the circuit still improves. It is a **continuous during-learning weight dynamic**, not
an offline pass; "consolidation ON" = the shared MLP is a Benna-Fusi synapse. Embeddings + heads stay fast (the
two-timescale split). It **engages after the bootstrap** (task 0 trains plain; the slow chain is then initialized
to the *formed* circuit) — protect-a-circuit-that-exists, the 136/137 framing. Anti-homunculus clean (a fixed
local linear relaxation); fence-clean (iterative, no SVD).

*Tuning history (n=3 regime-finding):* g=0.1 continuous-from-init **over-protected** — diffusion pulled weights
back toward random init faster than Adam could form the circuit, capping the bootstrap. Fix: engage BF *after*
the bootstrap. With that fix, the mechanism became cleanly active across g (see Results).

## Method

5 arms, one model, K=10 add/sub-on-rotating-alphabets stream (p=17); leaf functions imported from exp83 so the
C-off arms (`floor`, `replay_only`) reproduce exp84 byte-identically (the anchor). Stats in **log-FTSR** (paired
within-seed bootstrap; speedups are heavy-tailed/multiplicative — Report 138), equivalence margin ±0.22.

| arm | replay | Benna-Fusi | role |
|---|---|---|---|
| `scratch` | — | — | FTSR denominator |
| `floor` | ✗ | ✗ | no-rehearsal floor (anchor = exp84 no_replay_no_consol) |
| `replay_only` (R) | ✓ | ✗ | the bar (anchor = exp84 replay_no_consol ≈6.3) |
| `consol_only` (C) | ✗ | ✓ | BF protection alone (non-degenerate — BF needs no replayed data) |
| `replay_plus_consol` (RC) | ✓ | ✓ | the mechanism arm |

## Pre-registration (locked before the full run)

- **g-profile:** {0.01, 0.03, 0.1}; n=48 per g (12×4 sharded; extend to 64 if borderline).
- **Per-g interaction:** `d1` CI-lo>0 AND `d2` CI-lo>0 AND RC compounds (last-block logFTSR ≫ first) AND
  retention(RC) ≥ retention(replay_only) − 0.05.
- **ROBUST GRADUATION** = both legs CI-disjoint at **all 3 g** with no g showing RC<R; **PARTIAL** = holds at
  some g; **NULL** = none. The full g-profile is reported regardless (guards tuning-to-pass — the g was NOT
  chosen to maximize the headline).
- **Synergy hypothesis (n=3, to confirm at n=48):** `consol_only` has HIGH transfer but LOW retention (no
  replay); `replay_only` has retention but lower transfer; `RC` has BOTH — each ingredient supplies what the
  other lacks. If true, this is a genuine superadditive interaction.
- **Mandatory checks:** anchor reproduction; the synergy retention pattern; **per-k FTSR degradation** (does RC
  transfer degrade late, the 137 freeze crossover?); adversarial verification before banking.

## Results (n=48, g-profile; log-FTSR, paired bootstrap, EQUIV ±0.22)

| g | floor | R (replay) | C (BF-only) | RC | **d1 (RC−R)** | **d2 (RC−C)** | d_super | ret(C) | ret(RC) |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.85 | 6.07 | 4.61 | 12.75 | **+0.84 [+0.67,+1.01]** | **+0.84 [+0.67,+1.02]** | +0.20 [−0.02,+0.40] INCONCL | 0.26 | 0.98 |
| 0.03 | 1.85 | 6.07 | 8.45 | 14.10 | **+0.96 [+0.78,+1.14]** | **+0.56 [+0.29,+0.82]** | −0.09 [−0.37,+0.19] INCONCL | 0.37 | 0.99 |
| 0.1 | 1.85 | 6.07 | 11.02 | 14.00 | **+0.79 [+0.56,+1.03]** | **+0.43 [+0.09,+0.76]** | −0.22 [−0.59,+0.16] INCONCL | 0.40 | 0.98 |

**The dual-leg interaction gate clears at all 3 g** — RC beats *both* replay-only (d1) and BF-only (d2),
CI-disjoint; RC compounds (per-k raw FTSR grows, e.g. g=0.03: 6.0→14.1→16.8→19.5, **no 137-style late
crossover**); retention held (RC ~0.98). **Anchor:** `replay_only` x-FTSR = 6.07 [4.50,8.13] reproduces exp84's
`replay_no_consol` (the n=64 value was 6.33). **Verification:** the stats lens independently re-merged all 60
shards and reproduced every headline at atol < 1e-9; the harness is fence-clean and anti-homunculus-clean (only
metric-driven control flow is the eval early-stop, identical to exp83; `bf_start_task` is a fixed schedule).

**Attribution — multi-timescale depth IS load-bearing (matched-protection control, g=0.03 then higher g):**
m=1 (no slow chain) is a provable no-op (RC = R byte-identical, d1 = 0 — validates the harness). For a single
slow store (m=2), **increasing g does not increase protection — it collapses it** (C-FTSR 4.53→2.30→1.61 at
g=0.03/0.1/0.3; d1 +0.20→−0.02→−0.26): one slow variable at strong coupling equilibrates fast and destabilizes
learning. Only the deep chain (m=4) sustains strong *stable* protection (C-FTSR 8.45–11.02). So m=4 ≫ m=2 is
**not** merely "more total protection" — a single store cannot enter the protection regime at any coupling.
*(Residual hedge: a frozen-reference EWC with independently-tuned strength is untested, so this shows depth is
load-bearing within the Benna-Fusi family, not that no single-timescale mechanism could ever work.)*

## Verdict — FIRST PASS of the §8 interaction gate, scope-limited and recombinant

**This is the first empirical PASS of the CONTEXT-B §8 replay×consolidation interaction gate in the Bet-B arc:**
on a toy K=10 modular-arithmetic continual stream, adding Benna-Fusi multi-timescale weight consolidation (m=4) to
the shared MLP *on top of* interleaved replay yields new-alphabet forward transfer that beats **both** replay-alone
and consolidation-alone (CI-disjoint, all 3 g), with retention held and compounding preserved. Stated at exactly
the strength the data + adversarial verification (`wuybsd91u`) support:

- **It is an ADDITIVE "beats both parents" interaction, NOT super-additive** (`d_super` inconclusive at g=0.01,
  sub-additive at g=0.03/0.1). The gate is "neither ingredient alone suffices," not "the whole exceeds the sum."
- **It is largely RECOMBINANT.** The compounding and retention come **entirely from replay** (Report 138): RC
  compounds *less* than replay alone (last−first logFTSR +1.16 vs +1.48), and C-only does **not** compound. BF
  contributes a roughly **constant per-task protected-circuit transfer multiplier** — the *graded* continuation of
  Report 136's frozen-reusable-circuit effect (C-only x-FTSR rises with protection: 4.6→8.5→11.0). **The one
  genuinely-new piece over 136/137: graded protection delivers that transfer multiplier *without* 137's
  hard-freeze late-degradation crossover** (RC per-k grows to end-of-stream).
- **Depth is load-bearing** (single slow store can't reach the protection regime — see Attribution), within the BF
  family.
- **g=0.1's d2 is a marginal convergence-rescue**, not a speed win (CI-lo +0.09; on converged-only cells RC≈C at
  g=0.1; the leg is carried by cells where C-only hits the step cap while RC succeeds — a conservative direction).
- **Retention is replay's**, not a synergy: R alone is already ~0.978 (near ceiling) and RC−R is ~+0.005; the only
  retention-relevant evidence is that C-alone *forgets* (ret 0.26–0.40), so replay is needed for retention while BF
  supplies transfer.
- **Scope:** toy modular-arithmetic, K=10, n=48; raw x-FTSR magnitudes (12–14) are partly eval-resolution-limited
  (lead with the d1/d2 log-CIs, which are censoring-immune).

**What it is NOT:** "consolidation manufactures structure/transfer replay cannot." It is: *the first mechanism in
the arc where a brain-distinctive consolidation, combined with replay, beats both replay-alone and
consolidation-alone on the continual-transfer headline — additive, recombinant, with graded-protection
late-stability as the novel piece.*

## NEXT
- The verification's remaining load-bearing control: a **matched weight-EMA / frozen-reference EWC** baseline on
  `replay_only` (tunable strength) — to fully separate "multi-timescale consolidation" from "any weight
  regularizer replay lacks" (m=2-by-g-tuning already shows single-store can't reach the regime, but a frozen
  reference with independent strength is the cleaner test).
- A **hard-freeze (137) bracket arm** in this harness, so BF's gain reads explicitly as "delta over a hard freeze"
  (the late-stability novelty), n=48.
- Then **GERM** (the design workflow's #1 pick) as the second swappable recipe, same 2×2 harness.
