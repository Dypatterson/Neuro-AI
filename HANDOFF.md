# HANDOFF — 2026-06-01 (PM-6; end of day; fresh-session orientation)

**Branch:** `consolidation/role-structure` · **HEAD:** the Report-124/125 commit (`2ecd597`) +
today's UNCOMMITTED Report 126 work (about to be committed). Memory edits live in `~/.claude/...`.
Oracle `_*.json` are data; `.stderr` run logs are committed. Pre-existing untracked left alone (§5).

**One line:** The **flat-code growth family is EXHAUSTED** (121/123/124/125, 7 operators,
route-invariant for LOCAL single-projection dynamics) **AND the bound is CAPABILITY-level, not a
codebook-cosine metric artifact** — today's cheap **Reframe-B behavioral probe** ([Report 126](reports/126_behavioral_substitutability/report.md))
NULLed: the graduated 055-058 memory does NOT behaviorally admit paradigmatic substitutes as
completions beyond para-set hubness. **The "wrong-metric" escape is closed.** NEXT = the structural
escapes the bound demands — the **iterated-TEM local-reachability oracle** (the one-pass version was
found mis-shapen) and/or the **eligibility-gated** family. Clean stopping point; nothing mid-run.

## 0. Read first (in order)
1. **[CONTEXT.md](CONTEXT.md)** — the charter (bet, gates, invariants).
2. **[STATUS.md](STATUS.md)** — the bookmark (Active deliverable = the full 121→126 chain).
3. This file.
4. **[Report 126](reports/126_behavioral_substitutability/report.md)** (the behavioral probe NULL +
   the label-shuffle methodology) and **[Report 125](reports/125_escape_route_triage/report.md)** (the
   4-oracle triage; E = the TEM flashlight-positive that motivates the next oracle). Frozen precommit
   for 126: [phase-3-behavioral-substitutability-probe-precommit.md](notes/emergent-codebook/phase-3-behavioral-substitutability-probe-precommit.md).
   Design space: [brainstorm-workspace/2026-06-01-nonflat-phase3/whats_missing.md](brainstorm-workspace/2026-06-01-nonflat-phase3/whats_missing.md) (§1 eligibility family; §4 the reframes).

## 1. Where we are RIGHT NOW (today, PM-6)
Verified Reports 124/125 reproduce bit-identically (re-ran the harness; the §3 guardrails reproduce to
3 dp). Then, BEFORE building the planned TEM oracle, the mandatory grounding + a domain-expert
adjudication (grounded in Whittington 2020) found the **planned one-pass-Hebbian TEM oracle is
mis-shapen**: TEM's *generalizing* factor is BACKPROP-trained; the *Hebbian* part is its
NON-generalizing memory — so a one-pass Hebbian writer tests the wrong half and is predicted-NULL by
the bound. The domain-expert also surfaced a **cheaper, more fundamental probe** (Reframe-B), which the
user chose to run first. Built + ran it ([Report 126](reports/126_behavioral_substitutability/report.md),
`experiments/66`):
- **Reframe-B = NULL (5 seeds, PASS 0/5).** Does the graduated 055-058 memory behaviorally admit queen
  as a completion of king-contexts (bypassing codebook cosine)? para admission is real (B1 5/5) but a
  within-para **label-shuffle** kill-test shows it is **para-set hubness, not pair-specific**
  (para_shuffled +0.0041 ≈ halfway rand→para; pair-specific residual fails the both-variants bar 4/5;
  king/queen too sparse). **Decisive: the 121-125 bound is CAPABILITY-level, not a metric artifact.**
- **Methodological win:** the label-shuffle + multi-seed caught a single-seed false-PASS (B1∧B2∧B3 pass
  in the best seed). Without them this would have over-claimed "the bound is a metric artifact."

## 2. The genuine next move — the structural escapes (un-built)
The reframe is closed; the bound is real at the capability level → the escape needs NEW STRUCTURE. Two
adjudicated leads, both substrate-free-oracle-able first:
- **(a) The ITERATED-TEM local-reachability oracle (the domain-expert's corrected design).** NOT the
  one-pass Hebbian writer (mis-shapen). The faithful minimal mechanism: fixed random structural slots +
  a **bounded (1-2 pass) MHN-settling content→structure equilibration** (backprop-free; the equilibration
  is the existing settle dynamic, anti-homunculus-clean) — does a LOCAL writer with the equilibration
  recover E's +0.19, or null like B/C/D? **The iterated equilibration crosses toward a stripped-TEM build
  ⇒ the Phase-5 fence (the user's to lift).** Controls: calibration anchor; **E re-run in-harness as the
  recovery yardstick**; random-feature-slots null; no-iteration ablation; freq-matched / label-shuffle;
  NOT g3 (dead at n=40).
- **(b) The eligibility-gated / three-factor consolidation family** (`whats_missing.md` §1) — the only
  proposal that attacks the failing `corr(cooc,drift)` quantity by construction (surprise is
  anti-correlated with frequency). Cheap test: reweight cooc by a surprise factor BEFORE `grow_G`; must
  beat plain SPPMI (else = 123). The cheap PMI-surrogate already nulled as Oracle C (125); the non-trivial
  claim is the two-timescale eligibility×surprise *interaction* (needs the MHN settling-residual as the
  local surprise signal). Batch-offline by construction (eligibility traces bridge to the sleep pass).
- **Write a frozen precommit first** (the discipline held all session). Reuse `experiments/65` + `exp63`
  + `experiments/66` machinery.

## 3. Invariants the user holds (do not violate)
- **LOCAL growth is the MECHANISM, NON-NEGOTIABLE.** Global computations (SVD/NMF/word2vec) are
  DIAGNOSTIC FLASHLIGHTS only. "Do it right — no thesis-compromising shortcuts." [[do-it-right]]
- Anti-homunculus (Report 126's reads are pure DIAGNOSTIC measurement — nothing branches on them);
  batch-offline (online TD/error-driven banned); FHRR-native; the 055-058 FLOOR untouched (read-only).
- **Phase fence:** a TEM/latent BUILD is Phase-5 architecture — the user's to lift; substrate-free
  oracles + grounding are in-scope. A BUILD is a separate gate even on an oracle PASS.
- **Anti-rationalization:** the discipline KEEPS paying off — today the within-para label-shuffle +
  5 seeds killed a single-seed false-PASS; Reports 124/125 caught 4 metric subtleties. Stay this
  suspicious. The verdict-bearer is the adversarial control, never the headline arm alone.

## 4. Banked side-findings (don't relitigate)
- **TEM's Hebbian part is its NON-generalizing memory; the structural factor is backprop-trained**
  (Whittington 2020, opened this session, `link_only` ⇒ motivating-only until carded). A one-pass
  Hebbian-only TEM writer therefore can't test TEM's mechanism — the faithful test needs iteration.
- **Report 126 capacity finding:** the heteroassociative memory saturates at ≈14×D keys (write_l2
  recall 0.32 < store-as-is 0.67) ⇒ a "metric-artifact positive" appears ONLY in the over-capacity
  (broken) regime and vanishes in the clean regime — further evidence it's not a real capability.
- **Report 126 precommit-spec correction:** the frozen C0 "write_l2 ≥ store-as-is (raw recall)" clause
  was mis-justified as "the 055/056 write-marginal" (that is a SELECTIVITY-Δ, not raw recall). Did NOT
  affect the verdict (clean regime passes C0 decisively). Logged in Report 126 §6.
- Banked from 124/125 (unchanged): the locality trap; g3 dead at n=40 for STATIC reads (valid for
  `grow_G` drift); don't claim "E>SVD"; predict-context objectives are SPPMI-factorizers in disguise.

## 5. Artifacts (uncommitted — today)
- Report **126** (`reports/126_behavioral_substitutability/`: report.md + `_probe_seed{0..4}.json` +
  `.stderr` + `_probe_D2048.json`). `experiments/66_behavioral_substitutability_probe.py` (deterministic).
- Frozen precommit: `notes/emergent-codebook/phase-3-behavioral-substitutability-probe-precommit.md`.
- STATUS (Active deliverable walk-back + PM-6 entry), this HANDOFF, memory (`~/.claude/...`).
- Pre-existing untracked (leave them): `brainstorm-workspace/2026-05-30-research-grounded-plan/_wf{1,2}_raw.json`, `reports/gate0_2026-05-28/`.
