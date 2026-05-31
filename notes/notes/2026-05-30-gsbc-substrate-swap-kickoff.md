---
name: gsbc-substrate-swap-kickoff
date: 2026-05-30
project: personal-ai
status: KICKOFF SEED — PIVOTED 2026-05-30 to the learning-objective question (option B). GSBC PARKED downstream. To be finalized into a binding Stage-0 precommit by the next session (with the user).
branch: consolidation/role-structure (renamed from GSBC/substrate-swap on 2026-05-30; off codex/…@86cc9e4. Active first move is the consolidation/learning objective; GSBC parked downstream in this seed.)
parent: brainstorm-workspace/2026-05-24-unconsidered-paths/research/01-alternative-vsa-algebras.md (GSBC spec — parked)
tags: [notes, subject/substrate, subject/consolidation, subject/learning-objective, gsbc-parked]
---

# Substrate-swap → learning-objective (PIVOTED kickoff seed)

**Read first (session-start):** [STATUS.md](../../STATUS.md), then
[Report 116](../../reports/116_frameb_corr_ac_bd_slope_drop.md) →
[117](../../reports/117_frameb_leveldid_feasibility_consolidation_variance.md) →
[118](../../reports/118_frameb_landscape_sweep_variance_irreducible.md), then this note.
Seed, not binding — the next session finalizes Stage 0 with the user.

## The pivot — why GSBC is NOT the first move

This branch began as a GSBC substrate swap. A 2026-05-30 grilling established that the
swap's case rests on the Phase-5 role-binding failure being **overlap** (role signal
present but drowned by FHRR's 1/√D crosstalk floor). **The record says the failure is
FLAT / absence, not graded / overlap.** A cleaner readout makes *written* structure
legible; it cannot write structure consolidation never laid down. So **GSBC is PARKED**
as a downstream legibility upgrade, gated behind first showing role-discriminative
structure *can be written*. The active first move is the **consolidation/learning
objective**.

### Evidence (flat, not graded)
- `hit_role = 0.000` **exactly**, across FOUR retrieval families — D1 (storage), D3
  (coupling), E1 (landscape), M1 (weighted-MHN) — every seed × cue
  ([062](../../reports/062_phase5_spikes_d1_d3_local_smoke.md):45,
  [063](../../reports/063_phase5_spike_e1_centered_log_prior.md):52,
  [064](../../reports/064_phase5_m1_retrieval_smoke_cross_seed_null.md):76).
- `rank_role` ≈ 205–498 of 1064 atoms; the best anywhere was 263.7 with hit_role 0.02
  and *negative* ΔE ([058](../../reports/058_phase5_cross_seed_cue_regime_sweep.md):72,121).
  Random-middle is ~532 → at best a faint ~2× elevation, nowhere near recoverable.
- **Anti-overlap signature:** interventions that *sharpen* the representation made it
  WORSE — D1 pseudo-inverse rank 205→403; E1 sharper landscape 205→498, ΔE monotone
  negative (063:99–101). Overlap predicts sharpening surfaces the signal; it degraded it.
- Pre-committed routing already fired: "if D1/D3 don't help, the gap is substrate
  **training**, not retrieval mechanism" (062:10); 063/064 implicate the consolidation
  dynamics as the cause (064:100).
- **Convergent, from Phase 3 today:** the variance is *in the consolidation lift* (why the
  slope was shelved), re-injected regardless of landscape (Reports 117/118). Both ends
  point at the **learning step**.

### Corrected GSBC framing (for when/if it un-parks)
- **Exchange-rate, not impossibility.** The floor is 1/√D — separation is buyable with
  dimension / orthogonalization / cleaner unbinding. GSBC may buy it more cheaply
  (confined vs global crosstalk). That justifies an efficiency swap, NOT "FHRR can't."
- **Zeros degrade under bundling.** Structural zeros are a clean-one-hot property. The
  operating regime is consolidated/bundled/renormalized, where superposed GSBC products
  are entangled (that's *why* the BCF factorizer exists). GSBC relocates crosstalk into
  within-block competition; it does not eliminate it. Smaller, more specific win.
- Net: GSBC only pays off in the overlap world — un-park only if the learning step writes
  role structure and the bottleneck becomes legibility.

## Active first move — the learning-objective question

**Core question:** does the consolidation objective write role-discriminative targets into
the substrate at all — and if not, what objective would, *without a homunculus*?

Why it's open: current consolidation writes atoms as-is / co-occurrence (Path C, the C.2.x
dynamics); the Phase-5 nulls say it never lays down a role→target association (the
role-target sits where a random atom sits). A different *objective*, not a cleaner readout,
is the candidate fix.

### Constraints (binding)
- The objective must be a **local energy / geometry / settling / consolidation dynamic** —
  role basins emerge as a local energy minimum, never an `if role then write` rule or a
  tagging convention (anti-homunculus filter).
- Online error-driven codebook updates stay **BANNED at runtime**; error-driven only in
  batch-offline passes (STATUS live policy).
- Keep the pure-Python reference backend; don't make the LLM the source of identity
  (PROJECT_PLAN non-negotiables).

### Candidate levers to weigh (Stage-0 input, NOT decided)
- **Range-shape the replay buffer** (brainstorm top-rec #1; Dorrell/Whittington ICLR 2025:
  rectangular joint role×content support *forces* modularization). Data/learning-side, cheap.
  **Caution: a range-shaped replay was partially tried — STATUS records it "closed as
  novelty-without-retrieval" ([Report 111](../../reports/111_phase5_prime_range_replay_downstream_viability.md)).
  Re-read 111 before re-proposing.**
- A **contrastive / error-driven offline consolidation** term that raises role-target
  similarity and lowers competitor similarity (batch-offline energy term; anti-homunculus-clean).
- A **role-aware replay objective** that writes a role→target association as a basin.

### Staged plan
- **Stage 0 — precommit:** define operationally what "role-discriminative structure written
  into the substrate" means + a **write-then-read** test (can a toy consolidation write a
  role basin that a *native* FHRR readout recovers above chance?), pick ONE candidate
  objective, state the anti-homunculus check. → `/anti-homunculus-reviewer` before any
  mechanism lands.
- **Stage 1 — minimal write-then-read (the go/no-go):** does the candidate objective write a
  *recoverable* role basin (hit_role ≫ 0 on a native readout) in a toy? If no objective can
  write a basin, the readout question (GSBC) is moot.
- **Stage 2+ — scale / integrate**, only if Stage 1 writes structure. GSBC un-parks here
  *iff* the bottleneck has become legibility.

## Parked: GSBC (downstream)
Retained for when/if legibility becomes the bottleneck. **The binding lesson:** GSBC needs
**native ℓ∞ primitives + a structural-zero test from line 1; never the FHRR-similarity
Hopfield** (the [Report 066] GHRR confound). Spec: research/01-alternative-vsa-algebras.md
Idea A (D=4096, B=64, L=64, block-circular bind, ℓ∞ similarity).

## Branch facts
- `substrate/` has no interface/ABC (FHRR only). Phase 5′ stays **paused**; this branch
  doesn't touch it. Branch renamed `GSBC/substrate-swap` → `consolidation/role-structure`
  (2026-05-30) to match the learning-objective scope; GSBC parked. (Filename keeps the
  `gsbc` slug for git continuity.)
