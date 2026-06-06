# Report 135 — Bet B drill-down: does self-generated "pseudo-sleep" survive a LONG stream?

**Status:** BANKED — **degradation finding (real-but-insufficient), redirects to Stage 2.**
**Bet:** B (CONTEXT-B.md). **Type:** DRILL-DOWN on the Report 134 ADDENDUM lead mechanism — NOT a graduation run.

## TL;DR

Last night's first positive — self-generated "pseudo-sleep" beats a starved replay buffer — was measured at K=3 tasks. **Stretched to K=10, the *chronic* form (re-dream the whole past from current weights each round) rots toward chance** (end retention 0.21; oldest task 0.86→0.10), its advantage over the starved buffer shrinking ~6× (+0.45 at K=3 → **+0.058 [+0.016,+0.104]** at K=10). The pre-registered "delta CI-lo > 0 = survives" gate technically passes, but that one-word verdict is a misleading near-miss and is **not** the headline. **A snapshot control isolates the cause:** dream each task *once when fresh* and freeze it → retention 0.68, oldest task **flat at ~0.65, no rot** (snapshot − chronic = **+0.465 [+0.380,+0.538]**, n=8). So the rot is a re-dreaming death-spiral, not a fundamental limit — **but** the fixed form (a) still sits **0.30 below store-everything** (0.98) and (b) now stores 128 *synthetic* examples/task, surrendering the zero-storage advantage that was pseudo's whole point (real replay at equal count has exact labels and would dominate). **Net: rehearsing surface examples — real or dreamed — is the retention FLOOR; none of it manufactures forward transfer (FTSR shows only local within-pair transfer, zero compounding). The prize requires consolidating a compact shared *schema*, not surface episodes → Stage 2 (CONTEXT-B §5).** Discipline this session: caught the auto-gate's over-claim; added the snapshot control that pre-empted the obvious "you implemented pseudo wrong" critique (and turned out to carry the finding).

---

## Preamble (per CLAUDE.md)

- **Active capability:** Continual compounding-transfer (Bet-B re-grounded target, CONTEXT-B §8) — a **drill-down** on the Report 134 ADDENDUM mechanism (self-generated "pseudo-sleep"), not a graduation experiment.
- **Headline metric per [CONTEXT-B.md:234-238](../../CONTEXT-B.md):** the §8 *graduation* headline is FTSR-beats-replay (compounding). This drill-down's question is upstream of graduation: **retention-at-length delta (pseudo − raw_replay_tight) after a K=10 stream**, bootstrap CI, n=8. FTSR_k-vs-k (the §8 compounding metric) reported as a drill-down curve.
- **Required controls per [CONTEXT-B.md:240-244](../../CONTEXT-B.md):** from-scratch (FTSR denominator); raw-replay-tight (cap=2/task — the ADDENDUM baseline that broke); **raw-replay-FULL (uncapped) = retention ceiling** (validity check: must retain ≈1.0 or the harness is broken); plain-sequential (lower bound). Matched offline-consolidation compute + replay-fraction across all replay arms — only the buffer *source* differs (stored-real-capped vs stored-real-full vs self-generated/dreamed). Multi-seed (n=8) + bootstrap CIs.
- **Last verified result:** Report 134 ADDENDUM (exp80, K=3, cap=2): pseudo retention 0.47/0.49 vs raw-replay ~0.05; delta +0.45 [0.39, 0.53]; FTSR_T2 11.7 vs 3.8.
- **Why now:** Report 134's own "Next" list item #1 (longer streams) + its partial-retention caveat ("self-labels degrade as the model drifts"). Gates whether to push *naive* pseudo-rehearsal or pivot to the Stage-2 restructuring / two-timescale consolidator (CONTEXT-B §5) — the genuinely-new piece.

## The fork (why this run is decision-relevant)

Pseudo-sleep = the model regenerates its own past from current weights (stores **zero** raw examples; pseudo-rehearsal, Robins 1995 / generative replay). The known failure mode of generative replay is **self-label rot**: each consolidation round the model rehearses slightly-wrong dreams, errors compound, retention decays over a long stream. At K=3 you cannot see this. At K=10:

- **If pseudo-sleep holds** (delta vs starved buffer stays CI-lo > 0, retention roughly flat) → the ADDENDUM positive is real and scalable; worth pushing toward full retention.
- **If pseudo-sleep degrades** (retention decays toward chance as the stream grows) → naive pseudo-rehearsal does **not** scale → motivates the Stage-2 restructuring consolidator, which is the genuinely-novel piece anyway. **Honest null, not a dead end** (CONTEXT-B §8; [[feedback_null_ideation_is_iterate_fuel]]).

Both outcomes redirect the next move. That is what makes it worth running.

## Method

- **Substrate (legal under Bet B):** shared `nn.Embedding(vocab, 64)` + shared 2-layer ReLU MLP (256) + **per-task linear head** (Task-IL: task id given at eval). Backprop, AdamW, full-batch — the grokking recipe (weight-decay 1.0) so each task **generalizes** (held-out), not memorizes.
- **Stream:** K=10 modular-arithmetic tasks = **alternating add/sub on rotating disjoint alphabet blocks** (task t: op=add if t even else sub; block = t//2; tokens = block·p + i; label = (i±j) mod p). Chosen because only add (c=1) and sub (c=16) grok within budget on this substrate — *varied coefficients do not* (c=2/3/5/8/11/15 all stall < 0.7 held-out at 6000 steps; verified pre-run). Varying the alphabet (not the coefficient) keeps every task grokkable while preserving shared structure (within-block embedding + the shared ±-circuit MLP) and real cross-task interference. This is the faithful K-fold generalization of exp80's T1=add@blk0 / T2=sub@blk0 / T3=add@blk1.
- **Arms:** `scratch` (fresh model/task = FTSR denominator) · `plain` (no replay) · `raw_tight` (cap=2 real ex/task — the ADDENDUM baseline) · `raw_full` (uncapped real = retention ceiling / validity check) · `pseudo` (**chronic** self-generated rehearsal: regenerate the *whole past* from *current* weights each round, 0 raw stored) · `pseudo_snap` (**snapshot** self-generated rehearsal: dream each task **once when fresh** — right after learning it, ~crit-correct — and freeze it; isolates whether the chronic-re-dream death-spiral is the source of rot vs. a fundamental limit of dreamed rehearsal).
- **Matched consolidation:** every replay arm gets the same offline "sleep" (400 full-batch steps over its buffer) before each task + the same interleaved replay-fraction (0.5) during task learning. Only the buffer **source** differs → isolates *what you rehearse* (stored-real vs dreamed).
- **Config:** p=17, frac=0.7 (held-out 30%), max_steps=6000, crit=0.90 held-out, n=8 seeds, bootstrap CIs (4000 resamples). MPS.
- **Anti-homunculus:** pseudo = the model querying itself; no supervisor decides what to rehearse. The fast/slow split + replay schedule is a fixed dynamic.

## Results (K=10, n=8, p=17, max_steps=6000, chance = 1/17 = 0.059)

End-of-stream mean retention (held-out acc averaged over all 10 tasks after the full stream):

| arm | end-mean retention [95% CI] |
|---|---|
| raw_full (store everything = ceiling / validity check) | **0.980** [0.969, 0.989] |
| **pseudo_snap** (freeze each dream when fresh) | **0.676** [0.628, 0.728] |
| plain (no replay) | 0.207 [0.193, 0.222] |
| pseudo (chronic re-dream from current weights) | 0.211 [0.166, 0.262] |
| raw_tight (cap=2 real ex/task) | 0.153 [0.147, 0.160] |

**Validity:** `raw_full` retains 0.980 → tasks grok and retention is real; `raw_tight` collapses to 0.153 → the starved baseline genuinely breaks. Harness sound.

**Headline (pre-registered) — retention-at-length delta pseudo (chronic) − raw_tight = +0.058 [+0.016, +0.104].**
The auto-gate prints "survives" (CI-lo > 0). **I am overriding that as the headline — it is a misleading near-miss.** The advantage has collapsed ~6× across stream length (+0.45 at K=3 → +0.18 at K=5 → +0.058 at K=10), chronic pseudo (0.211) is barely above doing *nothing* (plain 0.207), and the oldest task has rotted to chance. The honest verdict is **degradation**, not survival.

**Degradation shape — oldest task (T1) retention vs stream position:**
- chronic pseudo: `[0.86, 0.66, 0.41, 0.29, 0.17, 0.16, 0.14, 0.13, 0.12, 0.10]` → **monotonic rot to chance.**
- snapshot pseudo: `[0.86, 0.66, 0.56, 0.57, 0.63, 0.62, 0.63, 0.63, 0.67, 0.68]` → **flat plateau ~0.65, no rot.**
- raw_full: `[0.86, 0.91, 0.95, 0.96, 0.97, 0.97, 0.98, 0.97, 0.99, 0.98]` → climbs to ceiling.

**The rot is a re-dreaming death-spiral, not a fundamental limit (the snapshot control):**
- snapshot − chronic end retention = **+0.465 [+0.380, +0.538]** (n=8). Freezing fresh dreams removes essentially all the rot.
- Mechanism: chronic regenerates task-j's rehearsal data from the *current* (already-drifting) model, so it dreams a half-forgotten task and consolidates on the garbage → errors compound. Snapshot dreams each task once at ~criterion-correct and freezes it.

**But snapshot is not the answer either:**
- snapshot − raw_full = **−0.303 [−0.343, −0.259]** → still ~0.30 below store-everything (frozen dreams are only ~crit-correct, ~80–90%).
- snapshot stores **128 synthetic examples/task** — it has *given up* pseudo's zero-storage advantage. At comparable count, **real** storage has exact labels and retains better (raw_full at ~200 real/task = 0.98). So dreamed rehearsal's only unique niche is the zero-storage (chronic) form — which is the one that rots.

**No compounding (FTSR_k vs k, chronic pseudo):** `[1.0, 10.0, 1.2, 9.4, 1.0, 12.9, 0.9, 11.7, 0.9, 0.8]`. High only on the odd tasks (sub-after-add, sharing the just-learned block embedding = within-pair transfer ~10×); ≈1.0 on every new block. The transfer is **local to the add/sub pair, not accumulating across the stream** — the §8 compounding headline (FTSR growing with k) is not met, consistent with Report 134-main.

## Verdict

**REAL-BUT-INSUFFICIENT (degradation), redirecting to Stage 2.** The Report 134 ADDENDUM positive (chronic pseudo-sleep beats a starved buffer) does **not scale**: stretched from 3 to 10 tasks it rots toward the floor, beating the starved buffer by a vanishing +0.058. A snapshot control proves the rot is a re-dreaming artifact (snapshot holds flat at ~0.65, +0.465 over chronic) — *but* the fixed form still trails store-everything by 0.30 and forfeits the zero-storage advantage. **Conclusion: rehearsing surface examples (real or dreamed) is the retention FLOOR. It does not manufacture forward transfer, and the cheap-storage variants don't reach the retention ceiling.** This re-confirms, from the continual-learning side, the wall Bet A mapped on the representation side: *structure has to be built by an iterative/restructuring consolidation, not read off the surface.*

**Discipline notes (both directions):**
- *Caught an over-claim:* the script's `survives` gate (delta CI-lo > 0) flips True on a +0.058 near-miss; the full curve (6× shrinkage, oldest→chance, far below ceiling, barely above no-replay) makes "survives" wrong as a verdict. Headlined the honest read.
- *Pre-empted a critique with a control:* added `pseudo_snap` to test whether the rot was my chronic-re-dream implementation vs. fundamental. It was largely the former — a materially more honest and informative result than "pseudo degrades, period."
- *Not over-negative:* snapshot pseudo IS a real, large, CI-clean retention mechanism (0.68 vs starved 0.15) — dreamed rehearsal works when you don't re-dream from a drifting model. It is simply dominated by real storage at equal count and below the ceiling, so it is not the prize.

## What this licenses

- **The next experiment is Stage 2 (CONTEXT-B §5): a restructuring / two-timescale consolidation that distills a compact shared *schema*, not surface episodes.** Headline must be the §8 graduation bar — does the slow consolidator *manufacture* forward transfer (FTSR-beats-plain-replay, compounding) and/or hold retention with far less stored than raw_full — i.e., beat what surface rehearsal (this report's floor) cannot. Guard the redundancy trap (§5): the win must be a measured delta over a plain replay buffer / a single-timescale ablation, not "structure appears."
- **Harness is reusable:** `experiments/81` now has a sharded (`--seed-start`) + merge (`--merge`) parallel runner; the K-task add/sub-on-rotating-blocks stream groks uniformly and exposes both retention and transfer. Stage-2 mechanisms swap in as new arms.
- **Two cheap follow-ups noted, not blocking:** (1) a `raw_mid` arm (cap=128 real) to confirm by direct measurement that real@128 dominates dreamed@128; (2) longer streams / loss-of-plasticity (plain's steps-to-crit vs k) — out of scope for the retention question here.

## Reproduce

```
# 8-way parallel (CPU shards, ~20 min) then merge:
for S in 0 1 2 3 4 5 6 7; do OMP_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python \
  experiments/81_betb_longstream_pseudo.py --p 17 --K 10 --max-steps 6000 --sleep-steps 300 \
  --seeds 1 --seed-start $S --device cpu --out reports/135_betb_longstream_pseudo/shards/seed$S.json & done; wait
PYTHONPATH=src .venv/bin/python experiments/81_betb_longstream_pseudo.py --merge \
  reports/135_betb_longstream_pseudo/shards/seed*.json --out reports/135_betb_longstream_pseudo/k10_n8.json
# single-process equivalent (MPS, ~40 min): same flags, --seeds 8 (no --seed-start/--merge)
```

