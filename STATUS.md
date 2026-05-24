# Project STATUS

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

## Current state (2026-05-24)

- **Active phase:** 5 ([phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md)). HAM × energy-guided structural branching on the post-death substrate.
- **Headline metric per [phase-5-unified-design.md:269-292](notes/emergent-codebook/phase-5-unified-design.md):** `ΔE = E_content-prior − E_role-prior`, paired per cue. Magnitude floor `5.5e-3`; CI must be strictly above zero.
- **Last verified result:** [Report 061](reports/061_phase5_log_prior_gain1_required_controls.md) — Path C (Varner-style log-prior spike) gain 1 produces `mean ΔE = +0.0089` above the floor on the slow-store substrate, but no-schema-store amplifies to `+0.0915` (arbitration-shape positive control, not Path-C evidence) and role-target basin retrieval remains absent. **Phase 5 not graduated.**
- **Active blockers:**
  1. Phase 5 path decision still open. Original three options were Path A (capacity-proportional scale-down probe) / Path B' (close + pivot to surprise/PE-driven replay) / Path C (continue log-prior diagnostics). The 2026-05-24 rescue brainstorm + D1/D3/S1 Tier-0 spike results add **Path D = Tier-2 training-time intervention** (M2 EqProp+role-shuffled-negatives+DSM warm-start, or M1 P1+D3+P3 stack). D1 (pseudo-inverse storage swap) and D3 (cross-K softmax) both returned null at smoke; storage-rule-only and branch-coupling-only routes are closed. No commitment in-session without user.
  2. Phase 5 spec [phase-5-unified-design.md:269-292](notes/emergent-codebook/phase-5-unified-design.md) must be edited *first* if any path other than continued Path-C diagnostics is chosen.
  3. **P1 spec gates** (if Path D pursues M1): replay-trace schema extension (~30 LOC, per [S1 spike](notes/notes/2026-05-24-spike-S1-replay-trace-schema.md)) + Spike S2 numerical Lyapunov check on weighted MHN must land before P1 main implementation.

## Recent updates

Newest first. Detail in [notes/status-log/2026-05.md](notes/status-log/2026-05.md); per-banner cap going forward is ~5 lines (link out for detail).

- **2026-05-24** — Phase 5 rescue brainstorm + Tier-0 spikes D1/D3/S1 ([brainstorm](brainstorm-workspace/2026-05-24-phase5-rescue/brainstorm-phase5-rescue.md), [Report 062](reports/062_phase5_spikes_d1_d3_local_smoke.md), [P1 spec](notes/notes/2026-05-24-phase5-p1-role-bank-dynamic-form.md)). Six literature angles converge on one architectural diagnosis; 11 mechanisms anti-homunculus-audited (5 PASS, 6 CONDITIONAL). D1 and D3 smoke at n=3×30 returns **null** (`hit_role=0.000` all conditions). S1 reveals replay trace lacks encoder-time provenance — P1 needs ~30 LOC schema extension. Per the brainstorm's decision recipe, routes to **Path D = Tier-2 training-time intervention** (M2/M1). ([archive](notes/status-log/2026-05.md#2026-05-24--phase5-rescue-brainstorm-and-tier0-spikes))
- **2026-05-24** — C-first log-prior n=10 Colab confirmation ([Report 060](reports/060_phase5_log_prior_n10_colab_confirmation.md)). Gain 1 diagnostic soft pass; n=10 mean ΔE `+0.0089`, CI `[+0.00645,+0.01164]`, `10/10` seeds positive, `1.615×` floor. Random-prior caveat (`random_lowest` 0.339→0.372) and basin retrieval still absent. Not graduation. ([archive](notes/status-log/2026-05.md#2026-05-24--c-first-log-prior-n10-colab-confirmation))
- **2026-05-24** — C-first log-prior local smoke ([Report 059](reports/059_phase5_log_prior_spike_local_smoke.md)). Gain 1 cleanest cell at n=3 seeds × 50 cues; gains 2/4 move more energy but worsen random-prior on seeds 11/23. ([archive](notes/status-log/2026-05.md#2026-05-24--c-first-log-prior-local-smoke))
- **2026-05-23** — Path-A SNR walk-back + nested chain back to 2026-05-20 pair-#4 falsification. SNR-invariance finding: signal scales 50× from N=12→N=1064 but noise scales 95×, so SNR ≈ invariant in N — Path A probably does not rescue graduation. Path C re-elevated to first-priority. Tier 0 diagnostic trio results (Fisher diagnostic correction, magnitude-floor verification, pair-#4 audit) and all prior 2026-05 work archived. ([archive](notes/status-log/2026-05.md#2026-05-23-and-earlier--path-a-snr-walk-back-through-2026-05-20-pair-4-trail))

## Banner discipline (binding)

- Each "Recent updates" entry: one-line summary + link to report + link to archive section. Cap ~5 lines.
- Long-form narrative / walk-back chains / nested audits live in the per-session report or in the monthly archive under [notes/status-log/](notes/status-log/), never inlined here.
- If a session walks back something currently in this file, that walk-back is the **first** edit of the session — and it goes in the bookmark above, not stacked into "Recent updates."
- Monthly archive files are append-only (newest at top within a file, oldest months link forward chronologically).

---

## Live operational policies

- **HAM regime split** (from report 022): β=30 + summed scores for retrieval; β=10 + HAM-arithmetic for replay diagnostics.
- **Online error-driven codebook updates: BANNED.** Use Hebbian for runtime, error-driven only in batch offline passes.
- **Drift sources sanctioned by design** (phase-4-unified-design.md:296-309): periodic batch retrains, online Hebbian reinforcement, or simulated synthetic perturbation.

---

## Pre-phase commitments still open (deferred but documented)

These were specified as required before claiming a phase done; they have not been.

- Consolidation-geometry regime classifier (d̄, d_eff per atom) — pre-Phase-3 spec, not built. [consolidation-geometry-diagnostic.md](notes/emergent-codebook/consolidation-geometry-diagnostic.md)
- Empirical θ′(β) calibration spike — recommended pre-Phase-3 (2026-05-09), not done.
- High-leverage brainstorm idea 5 (frequency-weighted Benna-Fusi α) — never built; named as the key experiment for the architecture's "compression → abstraction" claim. [brainstorm doc](brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md)

If a phase graduates without addressing these, document why in that phase's graduation report.

---

## Update rule for this file

- Update at the end of every working session.
- Promote a "blocker" to "done" only when there is a report with multi-seed CI evidence.
- If a session walks back something currently in this file, that walk-back is the **first** edit of the session.
