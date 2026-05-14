# Report 027 — Full repo audit & synthesis

**Date:** 2026-05-14
**Scope:** All experiments, reports, notes, brainstorm, research, design specs.
**Method:** 5 parallel sub-agents over `experiments/`, `reports/`, `notes/`,
`brainstorm-workspace/` + `research/`, and the specific Phase 3+4 integration claim.

This is the audit the session of 2026-05-13/14 should have started with. Findings
are organized so you can read top-down: executive summary → what's working →
what's not → gaps → concrete next steps.

---

## Executive summary

The project has a substantial body of validated infrastructure (Phase 0/1/2),
one cleanly validated Phase 3 result (Hebbian wins on settled synergy + R@K),
and **one barely-verified Phase 4 result** (ΔR@10 +0.010 CI [+0.001, +0.022]
at step 2000, rt=0.85, drift=0.30, 5 seeds, random-codebook control passes).

Where the project is weaker than this session's reports implied:

1. **Phase 3+4 integration was tested once at single seed, with no drift
   and Hebbian firing on only 1.8% of cues.** It validated the *mechanism*
   (Phase 4 replay produces 126 candidates instead of 2 when the codebook
   doesn't collapse) but not the *headline behavior*. Every multi-seed
   verification this session used the architecturally-incomplete setup
   (frozen codebook + synthetic drift, no Phase 3 dynamics).

2. **The Phase 4 unified design has ~7 components specified, all
   implemented, but at least 3 have not been exercised at run scale**:
   the death mechanism is structurally vacuous, the SQ-HN sparse update
   principle is uninstrumented, and the bidirectional u_k coupling at
   replay-cycle cadence happens but its effect has never been ablated.

3. **The architecture has documented commitments to a long list of
   mechanisms from the literature that have never been built**:
   consolidation-geometry regime classifier, Spisak-Friston self-
   orthogonalization, LSR kernel substrate option, IDP plasticity, FPE
   temporal binding, frequency-weighted Benna-Fusi coupling, GIB
   synergy as Phase 5 headline. Some of these were specifically named
   as pre-Phase-3 or pre-Phase-5 spikes; none have been done.

4. **Most architectural beliefs are correct.** The audit didn't surface
   anything that contradicts the project's framing. The corrections
   are about *scope of what's been verified*, not *which direction is
   right*.

---

## Part I — What is verified (across the whole repo)

### Substrate (Phase 0/1)
- FHRR round-trip at D=4096: exact (1.0). Bundle normalization is
  mandatory (discovered 2026-05-05); wrapper in place.
- MPS speedup 1.3-2.0× at L≥1024 (report 014).
- Torch and reference backends agree byte-for-byte (report 015).

### Temporal substrate
- Permutation-indexed slots beat bag encoding 1.000 vs 0.262 on synthetic
  directional recall (reports 009, 010). Multi-seed.
- Joint content+temporal energy rescues family-blend failure (report 003).

### Phase 2 (static contextual completion)
- 504-row full matrix on WikiText-2 complete. Recall@1 baseline 0.089
  [0.060, 0.125] (multi-seed pooled). Report 017 corrected the
  earlier single-seed 0.205 overclaim.

### Phase 3 (codebook learning)
- **Settled synergy** is the correct headline metric, not Recall@1.
  Phase 3c reconstruction codebook produces ~125× baseline synergy
  (report 018, 6 seeds, n=2191).
- **Hebbian is the winner on triple criteria**: settled synergy,
  R@K, and median rank — all disjoint CIs vs reconstruction,
  error-driven, random (report 019, 6 seeds).
- **Online error-driven updates are structurally unsafe** on any
  init (report 022 session summary): random init → chance,
  pretrained → 80% degradation. Hebbian is preservation-safe.

### Phase 4 (replay + consolidation)
- **First verified positive headline (report 026)**: ΔR@10 at step
  2000 = +0.010, CI [+0.001, +0.022], 5/5 seeds non-negative, rt=0.85,
  drift=0.30, 5 seeds.
- **Random-codebook control passes**: 0 candidates, Δ = 0.000 exactly
  at every checkpoint, every metric, every scale. The mechanism is
  causally tied to learned codebook structure.
- u_k Benna-Fusi graduation pattern matches design expectation
  (report 026 §u_k drill-down).
- Multi-scale (W=2,3,4) summed-score combines beat bigrams ~2× across
  5 seeds (report 024).
- HAM-arithmetic ties summed at β=30; wins narrowly with variance
  collapse at β=10 W=8 (report 022, 5 seeds). Adopted as policy:
  β=30 + summed for retrieval, β=10 + HAM-arith for replay
  diagnostics.

### Architectural commitments operationalized
- Anti-homunculus filter (2026-04-23 → 2026-05-09 → 2026-05-13). All
  mechanisms this session passed.
- FEP audit checklist (2026-05-13). Active.
- Headline-vs-drill-down metric framework (2026-05-09). Active —
  enforces single metric per phase + controlled comparison.

---

## Part II — What is not working / open

### Phase 3 atom-collapse pathology (real, unresolved)
- All three W=8 learned codebooks (Hebbian, reconstruction,
  error-driven) collapse to mean pairwise similarity ≈0.41 vs random
  ≈0.00 (report 019).
- Per-scale codebooks (W=2,3,4) stay orthogonal (~0.005) but **fail to
  retrieve at native W** — under-trained, not structurally better
  (report 021).
- Implication: substrate compensates for collapse far more than
  expected. Spisak-Friston orthogonalization redirection deferred.

### Phase 4 cap-coverage second headline (not robust)
- The Phase 4 design names *two* headlines: ΔR@K and Δcap-coverage.
- ΔR@K verified at +0.010. Δcap-coverage NOT verified: all 5 seeds'
  CIs cross zero; per-seed std is 5× larger than ΔR@K's; some seeds
  show -0.099, others +0.067. One of two design-spec headlines clears
  the CI bar.

### Phase 4 exit criterion "lower entropy" failed
- PROJECT_PLAN.md Phase 4 exit criterion: "Repeated solution paths
  become faster and lower entropy." At W=2 entropy is flat
  (Δ ≈ +0.003 throughout); at W=4 it drops early then **rises above
  baseline by step 2000** (Δ +0.057). Phase 4 *adds ambiguity* at W=4
  late in the run.

### Phase 4 death mechanism structurally vacuous
- `death_window=1000` requires 1000 consecutive consolidation steps
  (= ~50,000 cues at replay_every=50); default `death_window=100`
  needs 5,000 cues. Every Phase 4 run this session had only 40
  consolidation steps. Death cannot fire.
- Even if it could, `death_threshold=0.005` is below the Benna-Fusi
  equilibrium mean_strength ≈0.026. Active patterns never qualify.
- **Every Phase 4 result this session was produced without pattern
  death firing.** The "pattern replacement" narrative in earlier
  session reports was wrong; the mechanism is pure accumulation.

### Phase 4 W=4 seed-variance split (unexplained)
- Per-seed W=4 candidate count: {154, 3, 280, 213, 248}. Two orders
  of magnitude variation across seeds. No diagnostic explains why
  some seeds produce 3 candidates and others 280.

### Phase 3+4 integration partially tested
- **Exp 19 was run, single-seed (17), no drift**. Hebbian fired at
  1.8% of cues — effectively static codebook. Phase 4 replay produced
  126 candidates vs 2 (with broken error-driven) → 63× more discovery
  activity. Single-seed cap_t_05 +0.009.
- **Multi-seed Phase 3+4 integration with active drift has never been
  run.** This is the architecturally-intended test.

### Specified mechanisms not implemented / tested
From the cross-audit of notes, brainstorm, and design specs:

| Mechanism | Spec source | Status |
| --- | --- | --- |
| Consolidation-geometry regime classifier (d̄, d_eff per atom) | consolidation-geometry-diagnostic.md (2026-05-08) | Not built. Recommended pre-Phase-3 spike. |
| Empirical θ′(β) calibration | 2026-05-09 | Not built. Recommended pre-Phase-3. |
| Spisak-Friston self-orthogonalization | 2026-05-13 brainstorm | Cited, not implemented. Deferred per report 021. |
| LSR kernel phase diagram at D=4096 | 2026-05-13 brainstorm idea 2 | Probe started (report 008), full diagram never mapped. |
| Input-Driven Plasticity (IDP) | 2026-05-13 brainstorm idea 3 | Not built. |
| Frequency-weighted Benna-Fusi coupling | 2026-05-13 brainstorm idea 5 | Not built. Key for "compression → abstraction" claim. |
| FPE / permutation-indexed temporal slots in Phase 3 | brainstorm idea 6 | Permutation slots tested at substrate level (reports 009, 010), not integrated into Phase 3 codebook learning. |
| Reverse-replay credit assignment (TD along trajectory) | brainstorm idea 9 | Not built. |
| Pattern age distribution drill-down | phase-4-unified-design.md | Specified, not instrumented. |
| Cap-coverage at θ ∈ {0.3, 0.5, 0.7} as standing metric | 2026-05-09 | cap_t_05 + cap_t_03 added; θ=0.7 missing. |
| Meta-stable rate as standing core metric | 2026-05-09 | Added to exp 18, never aggregated across phases. |
| NC1 / separability pair tracking | 2026-05-09 | Not built. |
| GIB synergy estimator (Phase 5 headline) | brainstorm idea 8 | Not built. Synergy diagnostic at Phase 4 exists (report 011). |
| HDC binding for structural content | 2026-05-03 | Cited as candidate, not built. Embedding-space question open. |
| Diagnostic-actuator dynamic-form session | 2026-05-09 | Identified as "next major threshold," never held. |
| FPE temporal binding | brainstorm idea 6 | Cited (arXiv:2412.00488), not built. |
| REM-phase generative replay | brainstorm idea 7 | Phase 5+, deferred. |
| Cross-attention LLM workspace interface | brainstorm idea 10 | Phase 7, deferred. |

### Architectural open questions documented as critical
- **SONAR anisotropy risk** (flagged 2026-05-01 as critical). Never
  empirically tested. If confirmed, may require substrate switch
  (Nomic Embed).
- **Embedding-space structural encoding** (flagged 2026-05-03 as
  CRITICAL). Can rules + experiential reasoning coexist in the
  same landscape? HDC binding candidate but unproven.
- **Per-source priors formation** (2026-04-16 open). How do
  per-source priors get into landscape?
- **Raw snag origins** (2026-04-16 open). Pre-verbal generator
  for meta-loop is unspecified.

---

## Part III — Where the audit corrects session narrative

This session produced reports 019-026. The full audit surfaced these
corrections to what was said earlier:

| Session claim | Correction |
| --- | --- |
| "Phase 4 fades by step 2000" (report 025) | Wrong metric. ΔR@K *grows* through the run; fades by step 2000 was a top1 artifact. |
| "30 candidates per checkpoint" (reports 022-025) | Seed-17 W=2 only. Mean across 5 seeds at W=2: 9.0 ± 11.1. **W=4 is the actual workhorse: 179.6 ± 97.7**. |
| "death_window=1000 prunes 10× faster" (report 022) | Mechanism is structurally vacuous; no pruning ever fired. |
| "rt=0.85 makes Phase 4 drift-immune" (report 024) | Single-seed (17) — outlier. 5-seed picture is a real but modest +0.010 ΔR@K mid-run lift. |
| "Phase 4 fails at default drift=0.15" (report 022) | Correct as far as it goes — but drift=0.15 is the no-gap regime; Phase 4 needs damaged baseline to have room to repair. |
| "Phase 3+4 integration not tested" (report 026 §A) | *Partially* tested in exp 19 — single seed, no drift. Mechanism validated, headline not. |

---

## Part IV — Concrete next steps, prioritized

The headline question right now is: *what is the smallest set of
experiments that would either confirm Phase 4 as done-enough-to-move-
on, or surface a real architectural blocker?*

### Tier 1 — close out Phase 4 against the design spec (3-4 sessions)

1. **5-seed Phase 3+4 integration with active drift.** Use exp 19
   with `--updater-kind hebbian`, 5 seeds, plus an explicit drift
   mechanism. Two options:
   - (a) Apply the same synthetic perturbation as exp 18 (cleanest
     comparison to verified Phase 4 result).
   - (b) Let Hebbian drive the drift naturally — but Hebbian fires
     at 1.8% of cues, so over 1500 cues that's ~27 updates total.
     Need to either run much longer (n_cues = 50,000+) or increase
     Hebbian fire rate via threshold tuning.
   - Recommend: (a) for the headline, log Hebbian fire-rate as a
     drill-down. ~2-3h for 5 seeds.

2. **Fix death mechanism and re-run rt=0.85 5-seed.** Tune
   `death_threshold ≈ 0.05` (above equilibrium mean_strength) and
   `death_window ≤ 20` (fits in 40-step run). Tests whether actual
   pattern turnover changes the +0.010 ΔR@K result. ~2-3h.

3. **Recompute drift sweep against ΔR@K.** Existing JSON has the
   data; just re-extract. Frame the drift-magnitude story against
   the design-spec headline. ~10 min.

4. **rt=0.85 at drift=0.15, 5 seeds.** No-gap-regime check;
   confirms whether the +0.010 ΔR@K is drift-conditioned. ~90 min.

5. **Per-scale Recall@K aggregation across the 5 seeds we have.**
   Cheap. Tells us whether W=4 entropy-rise corresponds to W=4
   retrieval degradation. ~10 min.

### Tier 2 — address pre-Phase-3 and Phase 4 architectural gaps (4-5 sessions)

6. **Diagnostic-actuator dynamic-form session** (2026-05-09's "next
   major threshold"). For each diagnostic — high drift, high mean
   dispersion, bimodality, metastability, low cap-coverage — write
   the slow-timescale dynamic it's a snapshot of. Output: a design
   document that satisfies the FEP audit for each pair. ~1
   working session.

7. **Consolidation-geometry regime classifier** (pre-Phase-3 spec
   from 2026-05-08). Build the d̄ and d_eff per-atom diagnostic.
   ~1 session for implementation + integration into exp 18 / 19.

8. **Empirical θ′(β) calibration spike** (recommended pre-Phase-3).
   Vangara-Gopinath E1 protocol on FHRR substrate. ~2 days per
   the spec doc.

9. **Frequency-weighted Benna-Fusi coupling** (brainstorm idea 5).
   Implement `α_eff(pattern) = α_base × (1 + λ × normalized_retrieval_count)`.
   Test whether slow variables exhibit the under-capacity compression
   behavior Dury predicts. **Key experiment for the
   "compression → abstraction" claim.** ~1 session.

### Tier 3 — surface architectural risk (1 session)

10. **SONAR anisotropy test** (flagged 2026-05-01 as critical, never
    tested). Encode 100 random unrelated sentences, compute pairwise
    similarities. If mean is in [0.5, 0.8] cone, escalate
    substrate-switch discussion. ~30 min.

### Tier 4 — Phase 5 prep (after Tier 1-2)

11. **Layer-2 admission tightening** (recurrence-across-related-
    traces). From 2026-05-13 session summary's Tier 1.
12. **GIB synergy estimator** as Phase 5 headline candidate.
13. **PAM/Dury temporal co-occurrence** integration design pass.

### Deferred indefinitely (don't matter until earlier items land)

- Spisak-Friston orthogonalization (closed in report 021; reopen only
  if Phase 4 hits a collapse-linked limit).
- REM-phase generative replay (Phase 5+).
- Cross-attention LLM workspace interface (Phase 7).
- LSR kernel full phase diagram, IDP (Phase 2 alternates; not blocking
  Phase 4 verification).

---

## Part V — Recommended single next step

If the goal is "confirm Phase 4 is done enough to move on" with the
least amount of compute, the right single next step is **Tier 1 item
#1 — 5-seed Phase 3+4 integration with drift, using exp 19.**

This is the architecturally-intended test that has never been run
multi-seed. If ΔR@K survives multi-seed under Phase 3+4 concurrency,
Phase 4 graduates and the work moves to the diagnostic-actuator
dynamic-form session (Tier 2 #6) to set up Phase 5. If ΔR@K falls
apart, we know the +0.010 verified in report 026 is a property of
"frozen codebook + synthetic drift," not of the architecture's
intended use, and Phase 4 needs another design pass before Phase 5
becomes meaningful.

The other Tier 1 items (fix death, sweep drift, no-gap check) are
useful but don't address the integration gap that the design spec
actually cares about.

---

## Appendix A — Reading the corrected reports list

For someone catching up, this is the reading order:

| # | Topic | Status |
| --- | --- | --- |
| 017 | Phase 3 codebook comparison (multi-seed) | Load-bearing |
| 018 | Settled synergy as Phase 3 metric | Load-bearing |
| 019 | Atom-collapse pathology (3 codebooks all collapse) | Load-bearing |
| 021 | Per-scale codebooks orthogonal because under-trained | Load-bearing |
| 022 | HAM 5-seed validation | Load-bearing (2-regime policy) |
| 026 | Phase 4 verification against design-spec headlines | Current Phase 4 best |
| 027 | This audit | Synthesis |

Reports 023, 024, 025 are useful in context but were partly walked
back by 026's design-spec re-framing. The drift sweep findings in 023
are still real, just need recomputation against ΔR@K.

## Appendix B — Brainstorm ideas with concrete proposals

From 2026-05-13 brainstorm, prioritized:

1. FEP audit checklist ✓ done (notes/notes/2026-05-13-fep-audit-checklist.md)
2. LSR kernel - probe started, full diagram pending (report 008)
3. IDP - not started
4. Replay gate upgrades:
   - 4a tag_count ✓ done
   - 4b graded u_1 - partially done (gate computed, not used)
   - 4c inhibition of return - not done
5. Frequency-weighted Benna-Fusi coupling - **high leverage, not done**
6. FPE / permutation temporal slots - substrate done, Phase 3 integration not done
7. REM generative replay - Phase 5+
8. GIB synergy as Phase 5 headline - not built
9. Reverse-replay credit assignment - not done
10. Cross-attention LLM interface - Phase 7

## Appendix C — Anti-homunculus / FEP audit on the audit itself

This document is pure measurement of project state. No mechanisms
added; no inspect-and-trigger branches. The categorization "verified
/ open / specified-but-not-built" is a passive reading. Passes the
FEP audit trivially.
