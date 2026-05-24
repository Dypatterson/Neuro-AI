# Project STATUS

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

## Current state (2026-05-24)

- **Active phase:** 5 ([phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md)). HAM × energy-guided structural branching on the post-death substrate.
- **Headline metric per [phase-5-unified-design.md:269-292](notes/emergent-codebook/phase-5-unified-design.md):** `ΔE = E_content-prior − E_role-prior`, paired per cue. Magnitude floor `5.5e-3`; CI must be strictly above zero.
- **Last verified result:** [Report 061](reports/061_phase5_log_prior_gain1_required_controls.md) — Path C (Varner-style log-prior spike) gain 1 produces `mean ΔE = +0.0089` above the floor on the slow-store substrate, but no-schema-store amplifies to `+0.0915` (arbitration-shape positive control, not Path-C evidence) and role-target basin retrieval remains absent. **Phase 5 not graduated.**
- **Active blockers:**
  1. Branch-local path decision: `phase5-m1-role-energy-stack` now implements Path D / M1 as the active branch scope. M2 is deferred on this branch. Original Paths A / B' / C remain exhausted: Path A is closure-paper-only per the 2026-05-23 SNR-invariance walk-back; Path C is characterized (Report 061 one-hot arbitration-shape, Report 063 asymmetric-field null); Reports 062 + 063 closed retrieval-only storage / branch-coupling / landscape-reshaping routes.
  2. **M1 implementation status:** S1 provenance plumbing landed (`encode_window_with_provenance`, optional `TrajectoryTrace.encoder_terms`, replay preservation, Phase 4 integrated experiment plumbing). S2/P1/D3/P3 first implementation surface landed in `src/energy_memory/phase5/m1_role_energy.py`, with a synthetic smoke script at `scripts/phase5_m1_local_smoke.py`. This is not yet a Phase 5 evidence run.
  3. **Remaining blocker before any graduation claim:** rerun/retrain a provenance-bearing substrate, then run the Phase 5 control matrix at n>=10 with `hit_role`, `rank_role`, `random_lowest`, per-seed behavior, and G1 W=3 non-regression. Existing A+B+A1' snapshots predate S1 provenance and cannot by themselves provide P1's encoder-time role counts.

## Recent updates

Newest first. Detail in [notes/status-log/2026-05.md](notes/status-log/2026-05.md); per-banner cap going forward is ~5 lines (link out for detail).

- **2026-05-24** — M1 branch implementation started on `phase5-m1-role-energy-stack`. S1 provenance extension is backward-compatible; M1 role-energy stack primitives are in `phase5/m1_role_energy.py` (P1 weighted per-role MHN, D3 additive cross-K, P3 fixed saliency). Focused unittest pass and synthetic smoke pass. No Colab/n>=10 evidence yet. ([archive](notes/status-log/2026-05.md#2026-05-24--m1-role-energy-branch-implementation-start))
- **2026-05-24** — Spike E1 "Path C done right" closure ([Report 063](reports/063_phase5_spike_e1_centered_log_prior.md)). Zero-mean role/content asymmetric logit field swept λ ∈ {0, 0.25, 0.5, 1.0} on same 3 seeds × 30 cues. **`hit_role=0.000` at every λ across all 360 evaluations**; `rank_role` *worsens* monotonically 205→498 with λ. Field is noise-dominated on FHRR substrate. Combined with Report 062, three retrieval-mechanism families (storage / branch-coupling / landscape-reshaping) are now all null. **Path D is the only path not ruled out at smoke scale.** ([archive](notes/status-log/2026-05.md#2026-05-24--spike-e1-path-c-done-right-closure))
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
