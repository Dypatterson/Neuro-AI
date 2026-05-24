---
date: 2026-05-24
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: session-close
session-closes: phase5-tier0-spike-wave
---

# Phase 5 — Session Close and Next-Move Options (2026-05-24)

Single-source pickup document for the next session. Captures what was
ruled out, what remains, the three concrete options, and the
recommended next move with reasoning. Reading this + STATUS.md is
sufficient to continue without re-reading the brainstorm or individual
spike notes.

## What was done this session

In order:

1. **Rescue brainstorm.** Six parallel literature angles (resonator networks; active inference; Modern Hopfield variants; slot attention; TEM/hippocampal; EBM training). Converged on one architectural diagnosis. Synthesis with 11 candidate mechanisms across three tiers. Anti-homunculus reviewer audit: 5 PASS unconditional, 6 CONDITIONAL with named commit gates. Artifact: [`brainstorm-workspace/2026-05-24-phase5-rescue/`](../../brainstorm-workspace/2026-05-24-phase5-rescue/).

2. **P1 dynamic-form spec.** [`2026-05-24-phase5-p1-role-bank-dynamic-form.md`](2026-05-24-phase5-p1-role-bank-dynamic-form.md). Per-role coupled MHN codebooks; role bank = FHRR position vectors (architectural); atom-to-role attribution = soft frequency normalization (no thresholded ownership). Reviewer-audited CONDITIONAL PASS; gated on two pre-implementation spikes (S1 trace schema, S2 numerical Lyapunov).

3. **Four Tier-0 spikes**:
   - S1 (read-only inspection): replay trace lacks encoder-time (atom, role) provenance. P1 needs ~30 LOC schema extension. Note: [`2026-05-24-spike-S1-replay-trace-schema.md`](2026-05-24-spike-S1-replay-trace-schema.md).
   - D3 Lyapunov (analytical): brainstorm's multiplicative form is NOT a gradient field; additive form `(1-μ)π_k + μα_k` IS gradient flow on joint energy. D3 unblocks in additive form only. Note: [`2026-05-24-spike-D3-lyapunov-analytical.md`](2026-05-24-spike-D3-lyapunov-analytical.md).
   - D1 + D3 smoke ([Report 062](../../reports/062_phase5_spikes_d1_d3_local_smoke.md)): 3 seeds × 30 cues × 3 conditions. `hit_role = 0.000` for all 270 evaluations. D1 worsens `rank_role` (205→403); D3 drives ΔE negative 0/3 seeds.
   - E1 smoke "Path C done right" ([Report 063](../../reports/063_phase5_spike_e1_centered_log_prior.md)): 3 seeds × 30 cues × 4 λ values. `hit_role = 0.000` for all 360 evaluations. `rank_role` monotonically worsens with λ (205→498). Field is noise-dominated on this FHRR substrate.

## Combined diagnosis

Three architecturally distinct retrieval-mechanism families have been smoke-tested and all returned null on `hit_role`:

| Family | Spike | hit_role | rank_role direction |
|---|---|---:|---|
| Storage rule change (Hebb → pseudo-inverse) | D1 | 0.000 | worsens 205→403 |
| Branch coupling (independent → cross-K softmax additive) | D3 | 0.000 | ΔE negative 0/3 seeds |
| Landscape reshaping (uniform → asymmetric field) | E1 | 0.000 | monotone 205→498 |

No retrieval-mechanism family extracts role-target basin retrieval from the current substrate. **The substrate as consolidated does not contain role-target basins.** Combined with Report 058 (ΔE controlled by content_distortion, not role channel) and Report 061 (Path-C one-hot is arbitration-shape under no-schema-store amplification), the retrieval-mechanism family for Phase 5 rescue is exhausted.

## The four phase-5 paths, post-spike

Per STATUS.md "Active blockers":

| Path | Status post-spike |
|---|---|
| **Path A** — capacity-proportional scale-down probe | Closure-paper evidence only; SNR-invariance finding (2026-05-23) already showed it doesn't rescue graduation |
| **Path B'** — close Phase 5 + pivot to surprise/PE-driven replay | Accepts the null; reframes Phase 5 as not-graduated, moves on |
| **Path C** — continue log-prior diagnostics | **Exhaustively characterized**. One-hot form (Report 061) is arbitration-shape; asymmetric-field form (Report 063 E1) is null and worsens with strength. No retrieval-mechanism variant of Path C remains untested |
| **Path D** — Tier-2 training-time intervention | **Only path not ruled out at smoke scale.** Two main candidates (M2 and M1) below |

## The three next-move options

### Option 1 — Commit to Path D / M2 (RECOMMENDED for least-dependencies progress)

**What.** Implement M2 = P4 (Equilibrium Propagation consolidation pass) + role-shuffled negatives + DSM (denoising score matching) warm-start, per the [EBM training research brief](../../brainstorm-workspace/2026-05-24-phase5-rescue/research/06-ebm-training.md) and the [brainstorm M2 entry](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md#idea-m2-eqprop--replay-shuffled-negatives--dsm-warm-start).

**Why this is the recommended Path D entry point**:
- **Anti-homunculus reviewer PASS** (no conditions; role-shuffled negatives are pure data augmentation, EqProp's free/nudged phases are pure settling on the substrate, DSM is partition-function-free).
- **No prerequisite spikes.** Unlike M1, M2 does not depend on the P1 schema extension (S1's open gate) or the Lyapunov check on weighted MHN (S2). It can land directly.
- **Addresses the substrate-level diagnosis directly.** The spike wave's clear conclusion was that the substrate lacks role basins. EqProp + role-shuffled negatives carves those basins via contrastive consolidation; DSM gives a partition-function-free warm start. This is the literature's standard answer to "the EBM landscape doesn't have the basins you want."
- **Replay buffer is already 80% of a PCD buffer** per the EBM brief. The infrastructure addition is the negative-sample generator (role-shuffle of replay tuples) and the EqProp two-phase settle, not a new buffer subsystem.
- **The project's "Hebbian for runtime, error-driven only in batch offline passes" rule** is satisfied: EqProp runs as an offline consolidation pass; runtime retrieval is unchanged.

**Pre-implementation work needed** (none of these are blocking gates):
- Sample-design note: how to construct role-shuffled negatives from existing replay tuples (likely a small `~20 LOC` extension)
- Smoke-scale numerical sanity: EqProp's λ (nudging strength) needs a calibration spike on the existing substrate; ~half-day work

**Estimated total**: 1-2 working weeks for first results.

**Falsification**: if EqProp + role-shuffled negatives runs to convergence but `hit_role` and `rank_role` don't improve materially, Path D is also closed and the project should consider Path B' (closure paper) or a deeper architectural change.

### Option 2 — Commit to Path D / M1 (more ambitious; longer chain)

**What.** Implement M1 = P1 (per-role coupled MHN codebooks) + D3 additive cross-K softmax + P3 (IDP saliency). Per the [P1 spec](2026-05-24-phase5-p1-role-bank-dynamic-form.md), the [brainstorm M1 entry](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md#idea-m1-per-role-codebooks--cross-k-slot-competition--idp-saliency), and the [D3 Lyapunov pass](2026-05-24-spike-D3-lyapunov-analytical.md).

**Why this is the architecturally richer option**:
- Per-role energies give the substrate the role basins the spike wave proved it lacks.
- Cross-K coupling (D3 additive form, now Lyapunov-validated) keeps K branches non-redundant during settling.
- IDP saliency (Betteti et al. 2025) reshapes the energy landscape per-cue without arbitration shape.
- All three components anti-homunculus-clean; HAMUX-style additive Lagrangians compose them as one Lyapunov function.

**Pre-implementation gate chain**:
1. **S1 follow-on**: implement the ~30 LOC backward-compatible replay-trace schema extension (encoder-time provenance plumbing). Half-day to one day.
2. **S2 Lyapunov spike**: numerical check that `diag(w)·X` MHN preserves Lyapunov + basin-magnitude on a small synthetic substrate. Half-day. If S2 fails, fall back to row-normalized `diag(w)·X` (in-LSE, anti-homunculus preserved); explicitly forbid the logit-prior fallback (Path-C-shape recurrence).
3. **P1 main implementation**: ~120 LOC per the spec, after S1 and S2 close.
4. **D3 implementation**: ~50 LOC (additive form per Lyapunov pass).
5. **P3 (IDP) implementation**: ~50 LOC.
6. **Stack into M1 with HAMUX-style additive Lagrangians**.

**Estimated total**: 3-4 working weeks. Longer chain; richer if it works.

**Why I don't recommend this as the entry point**: the prerequisite chain has more places to break, and M2 directly addresses the substrate-level diagnosis without needing per-role-codebook bookkeeping. M1 is a sensible Path D follow-on if M2 succeeds and the project wants to push further. Or a sensible Path D entry if the user prioritizes architectural richness over time-to-first-result.

### Option 3 — Phase 5 closure paper

**What.** Treat Reports 061 + 062 + 063 + the brainstorm + the anti-homunculus reviewer audits as a publishable negative result on retrieval-mechanism interventions for the FHRR + Modern Hopfield + emergent-codebook architecture's structural retrieval failure. Pair with the consolidation-geometry framing from Vangara & Gopinath 2026 and the EBM-training framing from the brainstorm's EBM brief. Output: a written-up closure of Phase 5, framing the negative result as a *substrate-training* finding rather than a *retrieval-mechanism* finding.

**Why this is a real option**:
- Three independent retrieval-mechanism families all null is publishable.
- The brainstorm + reviewer audits constitute a complete architectural sweep against the project's anti-homunculus filter.
- The smoke evidence is stronger than the single Report 061 baseline because it's three orthogonal null mechanisms, not one.
- The project gains a clean closure rather than indefinite extension.

**What would be needed**: a written-up report (probably ~5-10 pages), citing the project's own brainstorm + spike work and tying to the relevant 2024-2025 literature (Saighi HRR/replay, Kymn associative memory of structured knowledge, Betteti IDP, Santos Hopfield-Fenchel-Young, etc.). Estimated 2-3 days of writing.

**Why this might be the right move**:
- If the user's goal is the project's research output rather than continuing to chase Phase 5 graduation
- If the substrate-training intervention (M2 / M1) is itself a multi-month commit that the user doesn't want to make
- If the project pivots to a different phase (e.g. Phase 6 multi-timescale or a deferred architectural item)

## My recommendation

**Option 1 (Path D / M2).** Reasoning:
- Shortest path to a real result that would either falsify the brainstorm's training-time diagnosis or surface structural retrieval for the first time on this architecture.
- No prerequisite spike chain; lands without the S1 + S2 + P1 main + D3 + P3 + stack work that M1 needs.
- Anti-homunculus PASS without conditions.
- Even if M2 nulls, that's a strong scientific result (training-time intervention also failed) that strengthens Option 3 (closure paper).

**Option 3 is the right call if** the user does not want to invest the 1-2 weeks Option 1 requires. The closure-paper-now path is honest, defensible, and frees up architectural attention for other phases.

**Option 2 (M1) is the right call if** the user has strong prior conviction that per-role energies are load-bearing (not just training-time pressure). The brainstorm leans toward this conviction; the spike wave's clean substrate-level null leans away from it.

## What the next session should do, mechanically

1. **Read STATUS.md** (the bookmark; updated this session).
2. **Read this note** (single-source synthesis).
3. **If the user gives a direction (Option 1 / 2 / 3)**: proceed per that path. The artifacts needed are in the brainstorm doc, the P1 spec, and the EBM training brief.
4. **If the user asks "what should I do"**: surface the three options + my recommendation (Option 1, M2). Do not re-derive the analysis from scratch.
5. **Do NOT re-run D1 / D3 / E1 / S1 / D3-Lyapunov** — those are exhaustively characterized and the conclusions are in Reports 062 + 063 and the spike notes.
6. **Do NOT redo the brainstorm** — six angles, 11 mechanisms, anti-homunculus reviewer audit already done.

## Open spec / blocker items to track if Option 1 or 2 is chosen

- If Option 1 (M2): a small sample-design note for role-shuffled negatives (~½ day pre-implementation)
- If Option 2 (M1): S1 schema extension PR (~30 LOC), S2 Lyapunov numerical spike (~½ day), then P1 + D3 + P3 implementation (~120 + ~50 + ~50 LOC), then HAMUX-style stack
- If Option 3 (closure paper): no implementation; writing only

## Linked artifacts

- [STATUS.md](../../STATUS.md) — current bookmark
- [Brainstorm](../../brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md)
- [P1 dynamic-form spec](2026-05-24-phase5-p1-role-bank-dynamic-form.md)
- [S1 spike note](2026-05-24-spike-S1-replay-trace-schema.md)
- [D3 Lyapunov spike note](2026-05-24-spike-D3-lyapunov-analytical.md)
- [Report 062 — D1+D3 smoke null](../../reports/062_phase5_spikes_d1_d3_local_smoke.md)
- [Report 063 — E1 smoke null (Path C closure)](../../reports/063_phase5_spike_e1_centered_log_prior.md)
- [EBM training research brief (M2's literature backing)](../../brainstorm-workspace/2026-05-24-phase5-rescue/research/06-ebm-training.md)
- [Active inference research brief (P3's literature backing)](../../brainstorm-workspace/2026-05-24-phase5-rescue/research/02-active-inference.md)

## Session-close metadata

- Phase 5 path decision: still open, but narrowed to three options (D / D / closure paper)
- STATUS.md bookmark: updated with E1 result and Path D framing
- Active blockers: Phase 5 path decision (this note proposes a resolution); P1 spec gates (relevant only if Option 2 chosen)
- No commits beyond this note; no source-code changes pending
