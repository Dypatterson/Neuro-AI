# HANDOFF — 2026-06-02 (PM-10; CE-1 gap-closer RAN + ESCALATED + RESOLVED; fresh-session orientation)

**Branch:** `audit/coupled-null-reeval`. Memory edits live in `~/.claude/`. Oracle/experiment `_*.json` and
`reports/**/*.json` are gitignored (data); the markdown docs + experiment `.py` are the durable record.

**One line:** The coupled null-audit's #1 lead (CE-1 ⊗ 127) is now run to completion AND escalated. The
locality-cost banking (PM-9) gave replay a measured target (+0.179 → +0.224 on `build_S`); the **gap-closer
(exp73) → NULL-BUT-RISING**, then the **pre-registered escalation (exp74) → RESOLVED it
([Report 128](reports/128_emergent_replay_gapcloser/report.md)):** an emergent, LOCAL, surprise-*targeted*
**pair-replay** is a **real, SUBSTRATE-CONFIRMED local gap-closer** — it closes the gap (102%, 10/10,
gauge-discriminated) AND the gap-close **survives the rung-2 FHRR port** (D=4096 ported B-KILL +0.227 ≈
exact +0.226; 1/√D crosstalk floor only at D=512), so it is **not an idealized-Euclidean artifact**. The
**operative lever is a ONE-SHOT surprise-targeted reweight** (≈ a PMI-style reweight, §1 guard → a *modest*
claim; magnitudes partition-inflated, NOT "beats SVD"); the two-timescale **LOOP is inessential** (`g-static`
fails at n=10 AND at the heavier 40-epoch budget). Then **§8 Phase B (TinyStories, a different domain) → the
gap-close GENERALIZES** (~3× stronger) and one-shot-suffices is domain-robust — with one honest caveat: the
anti-homunculus **gauge's inertness is sparse-signal-specific** (on the paradigmatically-saturated TinyStories,
random replay also helps, but targeting still wins ~3:1). **Then §8 sharpening DEFLATED the mechanism: a
codes-INDEPENDENT inverse-frequency reweight (`freq`) closes the gap as well as B (B−freq CI [−0.009,+0.012];
freq−A [+0.034,+0.052]) → the distinctive emergent-surprise-replay hypothesis is NOT supported; the gap-close
is a MUNDANE frequency/PMI reweight** (the §1 surprise≈PMI guard, empirically confirmed; 4th narrowing this
year after 126/127/exp70). **Final, fully-controlled CE-1 verdict: a LOCAL pair-reweight closes the locality
gap (real, substrate-confirmed, domain-general) — but the mechanism is "up-weight the rare pairs the
recency-bias drops" ≈ PMI, NOT codes-derived surprise, NOT a two-timescale loop.** The thread is RESOLVED;
next = a fork (§2) — the user's.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — charter (§3 "structural priors are not homunculi" clause).
2. **[STATUS.md](STATUS.md)** — bookmark (PM-10 = the gap-closer; PM-9 = locality cost; PM-8 = the audit).
3. This file.
4. **[Report 128](reports/128_emergent_replay_gapcloser/report.md)** (the gap-closer result + the honest
   two-finding split) and the **§7 run-log + §3.5 re-anchored gate** of
   **[the CE-1 ⊗ 127 precommit](notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md)**.

## 1. Where we are RIGHT NOW
- **The audit (PM-8) is the solid deliverable** ([null-audit-coupled.md](notes/emergent-codebook/null-audit-coupled.md)):
  64 nulls → 47 NO (73%) on three route-invariant bound families; short list led by CE-1 ⊗ 127, then the
  Oracle-E TEM local writer (#2), 052 Pair-#2, 013 tag_count→u_k, Oracle-C.
- **CE-1 ⊗ 127 arc, RUN + ESCALATED + RESOLVED:** PM-9 banked the **locality cost** (bounded-memory LOCAL
  writer +0.179 < global ceiling +0.224 on `build_S`, n=10, robust). PM-10 ran the **gap-closer** (exp73):
  emergent LOCAL replay = per-pair priority (novelty×surprise vs current codes) re-presenting OBSERVED
  co-occurrence **pairs** (SPPMI cancels per-row reweights → must be pair/episode-level). rung-1 = 6/7 gate
  PASS (B +0.225 closes 102%, 10/10, gauge does NOT close it), 1 FAIL = `g-static`. Then exp74 escalation:
  - **Rung-2 FHRR port PASS** — the gap-close SURVIVES the substrate: D=4096 ported B-KILL +0.227 ≈ exact
    +0.226, (B−A) across-seed CI-lo>0 at D∈{1024,2048,4096}, B-KILL 10/10; only D=512 straddles 0 = the
    **1/√D crosstalk floor** (§4.5 divergence protocol = a finding, not a kill). → **real on the substrate.**
  - **`g-static` STILL fails at 40 epochs** ((B−static) CI [−0.023,+0.044]; B 40ep +0.213 ≤ 20ep +0.225) →
    the **two-timescale LOOP is inessential; the operative lever is a ONE-SHOT surprise-targeted reweight.**
  - **BANKED (modest, deflationary):** a real, substrate-confirmed LOCAL gap-closer; one-shot ≈ a PMI-style
    reweight (§1 guard); magnitudes partition-inflated (NOT "beats SVD/NMF").
- **§8 Phase B generality (exp75 probe → exp73 on TinyStories, n=10):** PTB dead (HF loading-script removed),
  ag_news VALID-but-weak (own anchor +0.062), **TinyStories VALID+STRONG** (own anchor +0.234/kq0.350). On
  TinyStories: **(B−A) CI [+0.134,+0.194] 10/10 (gap-close GENERALIZES, ~3× stronger); `g-static` fails again
  (one-shot-suffices domain-robust). CAVEAT: `g-gauge` FAILS** ((gauge−A) [+0.042,+0.074]>0) — random replay
  also helps on the paradigmatically-SATURATED corpus, **but targeting wins ~3:1** (B−A +0.165 vs gauge−A
  +0.058). The gauge's inertness tracks signal SPARSITY (clean on WikiText, partial on TinyStories).
- **§8 sharpening (DONE) → DEFLATION.** `freq` (codes-INDEPENDENT inverse-frequency reweight) closes the gap
  as well as B (B−freq CI [−0.009,+0.012]; freq−A [+0.034,+0.052]) → surprise/novelty/loop/pattern-sep all
  INESSENTIAL; the gap-close is a frequency/PMI reweight. ag_news gauge-test INCONCLUSIVE (partition signal
  degenerate; ceiling −0.017; valid SVD anchor ≠ partition signal). **The CE-1 thread is fully resolved.**

## 2. The next move — a FORK (the CE-1 thread is RESOLVED + fully controlled; the user's call)
The banked claim is now MODEST: a LOCAL pair-reweight closes the locality gap (real, substrate-confirmed,
domain-general) but the **mechanism is a mundane frequency/PMI reweight, NOT the distinctive emergent-surprise
replay** (deflated). Pick one:
1. **Commit & pause** — the session's work (audit + CE-1 rung-1/2 + generality + the deflation) is a complete,
   honestly-controlled arc. Commit + push, stop. *(The natural endpoint; the user said "then commit".)*
2. **Pivot** — the audit short-list's other leads: Oracle-E TEM local writer (#2, a genuinely different local
   dynamic — its slot-factorization is not a frequency reweight, so it could still beat the freq baseline),
   or 052 Pair-#2 / 013 / Oracle-C.
3. **Rung-3 — full integrated confirm** (§4.5): wire the (now-known-frequency) pair-reweight into the real
   consolidation/replay path. Lower value now — integrating a mundane frequency reweight is a modest payoff.

## 3. Invariants the user holds (do not violate)
- **LOCAL growth is the MECHANISM, non-negotiable.** Global SVD/NMF/k-means = diagnostics/controls only.
- **Anti-homunculus, sharpened (CONTEXT.md §3):** a fixed problem-GENERIC scaffold (layers, k-WTA cap, D)
  is LEGAL; banned = a supervisor arbitrating outcomes, global backprop, or a scaffold hand-shaped to the
  answer. *(exp73's replay PASSES: priority is a local scalar from the codes over observed pairs; the
  gauge proves it's not hand-shaped to king/queen.)*
- **BUILD-GATE pressure is OFF** (user, 2026-06-02): substrate/latent/competitive builds are no longer
  fenced — but the **fidelity ladder discipline stays** (a rung-1 screen-pass ≠ a behavior claim; escalate
  rung-1 → FHRR-port → integrated with the over-claim guards, precommit §4.5).
- **The load-bearing habit (paid off 4× this year):** the verdict-bearer is the adversarial control, never
  the headline arm. exp73's `g-static`/`g-gauge` + the §8 `freq` control are exactly why the exciting "emergent
  surprise replay" framing did NOT survive — each control narrowed the claim (126, 127, exp70, now the freq
  deflation).

## 4. Banked side-findings (don't relitigate)
- **A LOCAL pair-reweight CAN close a measured gap to the global pass** on the paradigmatic axis (first time) —
  the gap-close is real, **survives the rung-2 FHRR port** (substrate-confirmed, 1/√D floor only at D=512),
  and is **domain-general** (TinyStories). The random gauge fails → it IS *targeting* (rare pairs), not mass.
- **The MECHANISM is a mundane frequency/PMI reweight, NOT codes-derived surprise (deflated §8) and NOT a
  two-timescale loop (deflated §6).** `freq` (inverse-frequency, codes-independent) ≈ B; `g-static` fails. The
  distinctive emergent-surprise-replay hypothesis is **not supported**; "up-weight the rare pairs the
  recency-bias drops ≈ PMI" is the whole effect. Do NOT re-frame this as an emergent-replay win.
- The bounded writer **contracts** over epochs on the recency-biased operator (0.215→0.179); replay resists it.
- **Generality (TinyStories):** the gap-close + one-shot-suffices GENERALIZE to a different domain (gap-close
  ~3× stronger); the **gauge's inertness is sparse-signal-specific** (random replay helps on a saturated
  corpus, but targeting wins ~3:1) — "targeting beats random" holds everywhere, "*only* targeting" doesn't.
- FHRR bundling differentially degrades the **denser** global operator (ported ceiling < ported-A at D=4096)
  — a substrate/crosstalk property, doesn't affect the same-density B−A headline.
- Magnitudes partition-inflated — NOT "beats SVD/NMF" (B/static exceeding the offline ceiling = reweight
  up-weighting of rare pairs, a known inflation, not a structural claim).
- (PM-9) 127 replicates on `build_S` (k-WTA≈k-means≈+0.22; grow_G~0); operator/representation matters
  (always run the partition on `build_S`, with grow_G/NMF on the SAME operator).

## 5. Artifacts (this session)
- Docs: [Report 128](reports/128_emergent_replay_gapcloser/report.md) (§6 escalation + §7 generality + §8
  sharpening/deflation + §9 final verdict); the precommit §3.5 + §7 run-log (exp73/74/75 + TinyStories +
  sharpening entries) + §8; STATUS.md (PM-10 + Active-deliverable; PM-3/PM-4 migrated to
  [status-log/2026-05.md](notes/status-log/2026-05.md)); this HANDOFF.
- Experiments: `experiments/73` (rung-1 gap-closer; additive `return_op` + `--fast-arms` + per-corpus
  calib-band args + the `freq` inverse-frequency control arm — all default-off → WikiText A/B/gauge/static
  path byte-identical); `experiments/74` (rung-2 FHRR port D-sweep + heavier-budget g-static); `experiments/75`
  (Phase-B corpus probe).
- Core-adjacent edit: `exp61.load_corpus` gained additive `tinystories`/`ag_news` branches (existing
  wikitext/repo_sample/synthetic_planted paths unchanged). No `src/` core edits.
- Data (gitignored): `reports/_exp73_*.json/.log`, `_exp74_*.json/.log`, `_exp75_corpus_probe.log`,
  `_exp73_tinystories_*.json/.log`.
- Pre-existing untracked (leave): `brainstorm-workspace/2026-05-30-research-grounded-plan/_wf{1,2}_raw.json`,
  `reports/gate0_2026-05-28/`.
