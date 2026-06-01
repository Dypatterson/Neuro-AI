# FROZEN PRE-COMMIT — Phase-3 behavioral substitutability probe (Reframe-B)

*Frozen 2026-06-01 BEFORE running. After the flat-code growth family was exhausted
(Reports 121/123/124/125, 7 operators, route-invariant for LOCAL single-projection
dynamics) and the planned TEM local-reachability oracle was found mis-shapen (a one-pass
Hebbian writer keeps the NON-generalizing half of TEM; the faithful version is iterative ⇒
Phase-5), the domain-expert + completeness-critic (`brainstorm-workspace/2026-06-01-nonflat-phase3/whats_missing.md` §4 "Reframe B") surfaced a cheaper, more fundamental probe: the
entire 121→125 arc measures paradigmatic structure as **codebook-vector cosine**, but
CONTEXT.md defines the capability as **contextual completion**. Substitutability may already
exist BEHAVIORALLY in the graduated 055-058 memory, invisible to codebook cosine. This probe
tests that directly, reusing existing machinery. Harness: `experiments/66_behavioral_substitutability_probe.py` (reuses exp61/63 + phase2/phase4 verbatim).*

## 0. Preamble — DRILL-DOWN reframe probe (NOT a graduation experiment)

> **Active phase:** Phase 3 (FOUNDATION; floor 055-058 is the SUBJECT of the probe, untouched/read-only).
> **Headline metric (this probe, frozen below):** behavioral **partner-admission specificity** —
> does the graduated write+L2 heteroassociative memory admit a paradigmatic partner (queen) as a
> low-energy / high-rank completion of its mate's contexts (king-contexts), above frequency-matched
> random tokens and above non-paradigmatic rand-pairs. This is **NOT** the phase-graduation headline
> (the role-Selectivity-Δ of 055/056); it is a **reframe diagnostic** on the existing memory, so per
> CLAUDE.md it is explicitly labeled a DRILL-DOWN, not a graduation experiment.
> **Why this metric is NOT codebook cosine:** the 121-125 bound (+0.109 global / ≤+0.021 local,
> route-invariant) is stated ENTIRELY in static codebook-vector cosine. This probe never forms a
> codebook centroid; it reads the COMPLETION landscape of the heteroassociative memory. The
> admission of b under a-contexts via H reduces to the key-space overlap of a-contexts with b's
> training contexts (second-order context similarity read through completion), which is not subject
> to the subdominant-modes bound (it is a property of H's geometry, not of a grown codebook vector).
> **Required controls (frozen below):** memory-calibration C0 (INVALID gate); frequency-matched
> random tokens; non-paradigmatic rand-pairs (the para-vs-rand differential = the real headline);
> context-specificity (a-contexts vs random contexts); collo contrast (report-only). Two read
> variants (score/energy AND rank/recall) must AGREE IN SIGN.
> **Last verified result:** Report 125 (flat-code family exhausted; TEM mis-shapen). The 055-058
> memory's role-Δ graduated and is BIT-IDENTICAL end-to-end (Report 058) — verified reproducible
> this session.
> **Why now:** the cheapest, highest-leverage open question — whether the bound is a METRIC artifact
> (codebook geometry) or a CAPABILITY ceiling — reusing existing machinery, no new substrate, no fence.

## 1. Anti-homunculus / discipline

The probe is a **pure diagnostic MEASUREMENT** of the existing completion dynamic — it reads
admission scores/ranks of the settled completion; **nothing branches, gates, selects, or writes on
these reads.** No supervisor, no `if-score-then-X`. Per the anti-homunculus filter, "a measurement
of a local geometric dynamic" is the explicitly exempt case. **Tension flagged honestly:** the
G-D panel (`experiments/56`) deliberately terminates its READOUT in a top_index equality count and
NOT an energy/ΔE (the "Phase-5′ fence") — because the MECHANISM must not arbitrate via energy. This
probe reads the completion score landscape, which is energy-adjacent, but as a DIAGNOSTIC, never as
a mechanism. To stay on the right side of that line, the **rank/recall variant** (percentile rank of
the partner among completions — "measured by recall, not cosine," `whats_missing.md`:71) is reported
as co-headline alongside the score variant. The graduated memory itself is read UNCHANGED (write+L2,
the 055-058 mechanism); the floor is not modified. Batch-offline write over a frozen buffer (the
existing `hetero_write.freeze` gate); no online/error-driven write introduced.

## 2. Build (frozen)

- **Corpus:** WikiText-2 train, W=6, center-masked windows, max_vocab=2000 (V=2002 — the EXACT vocab
  of the 121-125 arc, so the comparison "codebook-cosine NULL vs behavioral" is apples-to-apples).
- **Pairs (frozen, from `exp63.select_pairs`):** para = SimLex≥5 with cooc≤2 (n=40, non-co-occurring
  ⇒ a positive cannot be co-occurrence); rand = `random_matched_pairs` matched low-cooc (n=40, the
  NEGATIVE control); collo = SimLex≥5 with cooc>2 (n=5, report-only contrast); kq = king/queen
  (named probe, king=111/queen=31 center-occ, cooc=2). SimLex sha frozen in the run JSON.
- **Memory (the graduated 055-058 mechanism, read-only):** value codebook = decodable tokens
  (chance = 1/n_decode). Train buffer = all windows centered on a tracked token (para∪rand∪collo∪kq)
  ∪ a background sample of `background_n` windows centered on NON-tracked tokens (a realistic
  full-corpus memory with interference). Keys = `encode_cue(true)` (observed=all W−1 context slots).
  Fit the L2/ZCA `CueDecorrelator` on the buffer; `heteroassociative_write` over the FROZEN buffer
  (lr=0.5, epochs=20) ⇒ H. This is `write_H(decorr="l2")` from `experiments/56` verbatim.
- **Cue set:** for each tracked token a, its center-context cues (capped at `max_ctx_per_token`=100
  for balanced power). recalled = `decorr(cue) @ Hᵀ / D`; score over value_cb = `(normalize(recalled)
  @ value_cb.conj().T).real / D`.

## 3. Metric (frozen — two variants, both reported, must agree in sign)

For paradigmatic pair (a,b), over a's cue contexts:
- **score variant (energy framing):** `Δ_score(a→b) = mean_ctx[ s[b] ] − mean_{r∈freqmatch(b)} mean_ctx[ s[r] ]`,
  freqmatch(b) = `n_freq`=10 value tokens drawn (seed-fixed) from b's log-center-frequency bin
  (±1 decile), excluding a,b. Symmetrize `Δ_score(pair) = ½[Δ_score(a→b)+Δ_score(b→a)]`.
- **rank variant (recall framing):** `Δ_rank(a→b) = mean_ctx[ percentile_rank_of_b_in s ] − 0.5`;
  symmetrize. (0.5 = a uniformly-random token; >0 = b is admitted above median.)
- **HEADLINE = para minus rand-pairs**, both variants: `H_score = mean_para Δ_score − mean_rand Δ_score`
  and `H_rank = mean_rank` likewise, with bootstrap CIs over pairs (n_boot=4000, seed=7).

## 4. Gates (frozen; PASS = C0 valid ∧ B1 ∧ B2 ∧ B3)

- **C0 — memory calibration (INVALID gate, not NULL):** the memory must complete its own contexts.
  Pooled true-target top_index recall on tracked-token cues ≥ **5× chance**, AND write_l2 true-rate
  ≥ store-as-is true-rate on this corpus (the 055/056 write-marginal direction). If C0 fails the
  memory is broken at this scale ⇒ INVALID; fix and rerun. Nothing else is read.
- **B1 — para admission positive:** `mean_para Δ` bootstrap CI-lo > 0 for BOTH variants.
- **B2 — para > rand (THE HEADLINE):** `H_score` and `H_rank` bootstrap CI-lo > **0** (paradigmatic
  partners admitted strictly above non-paradigmatic rand-pairs). The behavioral analog of the arc's
  para-vs-random specificity.
- **B3 — context-specificity (label-shuffle analog):** partner b admitted MORE under a-contexts than
  under random non-{a,b} contexts: `Δ(a-ctx) − Δ(random-ctx)` CI-lo > 0 (pooled). Guards against "b
  is a global completion hub."
- **B4 — collo contrast (REPORT-ONLY, not a gate):** collo (high-cooc) admission reported to locate
  para between rand (≈0) and collo (high, trivially context-sharing).

## 5. Disposition (frozen)

- **PASS (C0 valid ∧ B1 ∧ B2 ∧ B3):** the graduated memory ALREADY exhibits behavioral paradigmatic
  substitutability that the codebook-cosine bound declared unreachable ⇒ **the bound is a METRIC
  artifact (codebook geometry), not a capability ceiling.** Reframe the Phase-3 paradigmatic target
  from codebook-cosine to completion-admission; re-scope 121-125 as "the wrong metric for this
  capability." Surface to user (large claim; would redirect the whole escape program). Open+card the
  CLS/completion primaries before any architecture move.
- **NULL (C0 valid; H_score ≈ H_rank ≈ 0, para ≈ rand):** the bound is CAPABILITY-level — the memory
  does not behaviorally substitute paradigmatic pairs either ⇒ codebook-cosine was measuring the
  right thing ⇒ the escape genuinely needs new structure (TEM/eligibility families). Decisive kill of
  the "wrong-metric" hypothesis; banks that 121-125 is not a metric artifact.
- **PARTIAL (B1 yes, B2 no — para admitted but ≈ rand):** admission is a frequency/hub artifact, not
  paradigmatic ⇒ treat as NULL for the headline; report the artifact.
- **INVALID:** C0 fails (memory broken at scale) ⇒ adjust D / background_n / corpus, rerun.

## 6. Build checklist
- [ ] Reuse via import: `phase2.encoding.{build_position_vectors,encode_cue-equivalent,mask_positions}`,
      `phase4.hetero_write.{HeteroConsolidationBuffer,heteroassociative_write,recall_top_index,
      batched_hopfield_topindex}`, `phase4.decorrelator.CueDecorrelator`, `phase3.basin_readout.top_index_hits`,
      `exp61.{load_corpus,build_cooccurrence}`, `exp63.select_pairs`, `exp62._boot_diff`.
- [ ] New code only: the stratified train-buffer builder, the per-pair admission read (score+rank),
      the freq-matched-random draw, the context-specificity arm.
- [ ] C0 calibration printed EVERY run; INVALID if it fails.
- [ ] Planted/small SMOKE (D=512, small background) validates pipeline + C0 before the WikiText run.
- [ ] Freeze BEFORE the run: W=6, max_vocab=2000, n_para=40, lr=0.5, epochs=20, beta=10, mi=12,
      max_ctx_per_token=100, n_freq=10, n_boot=4000, boot_seed=7, the 5×-chance C0 bar, the B2 CI-lo>0
      headline bar, the SimLex sha.
