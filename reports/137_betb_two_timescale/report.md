# Report 137 — Bet B Stage 2: does a two-timescale split MANUFACTURE transfer?

**Status:** BANKED — **NO GRADUATION (two-timescale freeze nulls + hurts) — but the BASELINE shows robust compounding (the on-target finding).**
**Bet:** B (CONTEXT-B.md §5/§8). **Type:** GRADUATION ATTEMPT — the first test of "consolidation manufactures what rehearsal can't."

## TL;DR

The pre-registered headline **fails**: the two-timescale split (freeze the shared circuit during fast learning, consolidate offline) does **not** manufacture transfer beyond a plain replay+consolidation baseline — the gain is marginal (x-block FTSR delta +6.5, **CI-lo +0.95 ≈ 0**) and **retention is lost** (protect 0.932 vs plain 0.978, Δ −0.045 [−0.089,−0.002]). The per-task curve is a **crossover**: `protect` transfers great early (k2=24×) but **degrades** to failure over the stream (k8=2.8×; 7/8 seeds slower at k8 than k2, often hitting the step cap), while **`plain` improves** (k2=1.6× → k8=10.4×). An n=2 smoke that stopped at k6 sat just before the crossover and showed a *false* "GRADUATES" — **caught by n=8 + the absolute-step + per-k drill-downs.** The genuinely positive, robust drill-down: **`plain` (replay + offline consolidation) exhibits real compounding forward transfer** — new alphabets learned ~6× faster by end-of-stream (FTSR 1.6→10.4, **7/8 seeds**; scratch flat at ~5800 ⇒ genuine accumulated transfer, not task-difficulty). **Read:** the brain-distinctive two-timescale freeze NULLS (echo of Report 134-main, sleep≈replay) and even *hurts* late; the simple replay+consolidation already manufactures the target compounding. The Report-136 "protection failure" was a **no-replay-warmstart artifact** — in-stream, replay implicitly protects the circuit, so explicit freezing is unnecessary and over-constrains as the buffer grows.

---

## Preamble (per CLAUDE.md)

- **Active capability:** Continual compounding-transfer (Bet-B target). This is the **graduation attempt** for the §5 two-timescale consolidation claim.
- **Headline metric per [CONTEXT-B.md:161-170](../../CONTEXT-B.md):** cross-block **FTSR** (steps-to-crit ratio scratch/arm) on the **new-alphabet** tasks (k=2,4,6,8), `protect − plain`, bootstrap CI, n=8. GRADUATES if protect manufactures transfer plain does not (delta CI-lo > 0 AND protect CI-lo > 1) AND end-stream retention(protect) ≥ retention(plain) − 0.05.
- **Why now:** Report 136 diagnosed the new-symbol zero-transfer as a **protection failure** (reusable circuit exists; ordinary training destroys it). Stage 2 tests the implied fix: protect the circuit during fast learning, consolidate it slowly offline.
- **Last verified result:** Report 136 (exp82): frozen stream MLP transfers 3.3× to a new alphabet; warm-start (trainable) ≈ scratch.

## Mechanism (anti-homunculus: a fixed schedule, no supervisor)

- **FAST timescale:** new task arrives → **freeze the MLP**, train only embeddings(+head). New embeddings snap onto the existing circuit.
- **SLOW timescale:** offline → **unfreeze and consolidate** the MLP over replayed tasks (it keeps improving).
- **Baseline (`plain`):** identical schedule, MLP never frozen during task-learning (single-timescale).
- **Drill-down (`freeze_forever`):** MLP frozen after a task-0 bootstrap, no consolidation — isolates whether the circuit must keep improving (the §5 guard: a permanently-frozen circuit is not consolidation).

## Honesty guards committed up front

- **Novelty:** freezing a feature extractor + slow consolidation = a recombination of **known** parts (linear-probing / complementary learning systems). If it graduates, the honest verdict is **PASS-but-known**, not novel — like Reports 133/134-main.
- **Fairness control (run before banking):** a `slow_mlp` arm = MLP at a small *nonzero* LR during fast-learn (not frozen). If `slow_mlp ≈ protect`, "freeze" isn't special — it's just low MLP LR; if `slow_mlp ≈ plain`, full protection is load-bearing. Either way reported.
- **Magnitude caveat:** FTSR magnitude is granularity-limited (eval_every) and depends on the `plain` baseline's own transfer — the headline is the **delta** + retention, not the absolute ×.
- **Compounding vs transfer:** does protect's new-block FTSR *grow* with k (compounding) or stay flat (constant transfer)? Reported per-k.

## Method

- Substrate/stream = exp81/exp82 (K=10 add/sub on rotating disjoint blocks; shared embed(64)+MLP(256)+per-task head; AdamW wd=1.0 full-batch grokking; p=17, frac=0.7, crit=0.90). max_steps=8000, consol_steps=400, replay_frac=0.5, n=8, bootstrap CIs. Sharded+merge parallel runner.
- New-alphabet (cross-block) tasks = even k≥2; within-block (sub-after-add) tasks = odd k (sanity).

## Results (K=10, n=8, p=17, max_steps=8000, consol_steps=400, chance=0.059)

Cross-block FTSR (new-alphabet tasks k=2,4,6,8), within-block FTSR (sub-after-add, sanity), end-stream retention:

| arm | x-block FTSR [95% CI] | within-block FTSR | end retention [95% CI] |
|---|---|---|---|
| plain (single-timescale, all-trainable + consol) | 5.44 [3.66, 7.14] | 31.5 | **0.978** [0.972, 0.983] |
| **protect (two-timescale freeze + consol)** | 11.97 [6.93, 17.47] | 23.5 | 0.932 [0.885, 0.976] |
| freeze_forever (freeze, NO consol) | 38.6 [32.6, 44.0] | 0.69 | 0.523 (collapsed) |

- **HEADLINE — x-block FTSR protect − plain = +6.53 [+0.95, +12.40]** (CI-lo barely > 0). **Retention protect − plain = −0.045 [−0.089, −0.002]** → fails the ≥ −0.05 bar. **PASS = False.**
- **Per-new-block FTSR vs k (the crossover):** protect `[k2=23.8, k4=13.3, k6=7.9, k8=2.8]` (degrades, 7/8 seeds slower at k8; frequently hits the 8000 cap = *fails to learn* later alphabets) · plain `[k2=1.6, k4=3.5, k6=6.3, k8=10.4]` (compounds, 7/8 seeds faster at k8). scratch steps flat across tasks (~5500–6300) ⇒ tasks equally hard cold ⇒ plain's speed-up is genuine accumulated transfer.
- **`freeze_forever`** (frozen task-0 add-circuit, no consolidation): huge x-block transfer (38.6× — a frozen *add* circuit is perfect for new *add* alphabets) but within-block (sub) collapses to 0.69× and retention to 0.52 → degenerate; confirms consolidation is required for retention and for serving both operations.

## Verdict

**NO GRADUATION (pre-registered headline) + a robust on-target drill-down.** The two-timescale freeze recipe does not manufacture transfer beyond the baseline (marginal, CI-lo ≈ 0) and *loses* retention; worse, it **degrades over the stream** (the circuit, frozen during fast-learn and only weakly re-consolidated over a growing buffer, fails on later new alphabets) while the baseline **improves**. So the brain-distinctive mechanism NULLS — and is actively harmful late — a sharper echo of Report 134-main (sleep ≈ replay).

The robust, genuinely-on-target finding lives in the baseline: **plain replay + offline consolidation exhibits compounding forward transfer** (new-alphabet learning ~6× faster by end-of-stream, 7/8 seeds, scratch-flat-controlled). This is the first time in the arc the *target behavior itself* (experience compounding into faster learning) is clearly present and robust — it is simply **not** carried by the brain-distinctive add-on.

**Mechanistic correction to Report 136:** the diagnosed "protection failure" (warm-start ≈ scratch) was measured **without replay** on a post-stream held-out block. In the actual stream, interleaved replay implicitly protects the shared circuit (it keeps the MLP serving old tasks during new-task learning), so the circuit is *not* destroyed — and explicit freezing, far from helping, over-constrains it as the buffer grows. The 136 diagnostic was correct that a frozen good circuit transfers; it over-generalized "ordinary training destroys it" by omitting replay.

**Discipline:** the n=2 smoke (K=8, stopping at k6) returned a *false* "GRADUATES" (protect 18.7 vs plain 3.9, retention held) — driven by lucky seeds and by stopping before the crossover. **n=8 + absolute-step + per-k drill-downs caught it.** This is the same failure mode the project has caught before (single-/few-seed false positives); the pre-registered multi-seed bar held.

## What this licenses

- **The two-timescale FREEZE recipe is banked as a null (iterate, don't repeat).** Not "two-timescale is impossible" — this specific freeze-hard + fixed-budget-consolidation recipe degrades over the stream. Live iterates: **softer/scheduled protection** (a `slow_mlp` arm at a small nonzero LR; or freeze-early/release-late), **consolidation budget that scales with the buffer** (the degradation tracks a fixed 400-step consolidation spread ever-thinner). These are recipe swaps on the same harness.
- **The higher-value redirect — characterize the baseline's compounding.** plain (replay + offline consolidation) already shows the target behavior. Open, decision-relevant questions: is the compounding driven by **replay or by the offline consolidation** (add a replay-only, no-consolidation arm)? Does it keep growing past K=10, or saturate? Is it "genuine abstraction reuse" or just "a bigger/cleaner circuit"? This may be where the real positive of the whole project actually lives — and it is a *baseline*, which means the honest contribution would be characterizing/strengthening it, not a novel mechanism on top.
- **Honest novelty status unchanged:** no brain-distinctive mechanism has yet beaten a simple baseline on the continual target (133 redundant, 134-main null, 134-ADDENDUM known+non-scaling [135], 137 null). The durable lesson holds: test where the baseline fails — and so far, on this toy, the baseline does not fail at the target.

## Reproduce

```
for S in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python \
  experiments/83_betb_two_timescale.py --p 17 --K 10 --max-steps 8000 --consol-steps 400 \
  --seeds 1 --seed-start $S --device cpu --out reports/137_betb_two_timescale/shards/seed$S.json & done; wait
PYTHONPATH=src .venv/bin/python experiments/83_betb_two_timescale.py --merge \
  reports/137_betb_two_timescale/shards/seed*.json --out reports/137_betb_two_timescale/k10_n8.json
```

