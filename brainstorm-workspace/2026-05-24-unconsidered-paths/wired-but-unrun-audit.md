# Audit: the "wired-but-unrun knobs" claim

> Generated 2026-05-24 after the freq-α walk-back. The original
> verification report claimed five `ConsolidationConfig` knobs were
> "wired-but-unrun." Codex caught that `alpha_freq_lambda` was actually
> run in Report 040. This audit checks the other four with the same
> discipline: read the reports.

## TL;DR

**All five knobs were run.** The "wired-but-unrun" framing was a grep-by-
default-value artifact. The actual production picture is more useful:

| Knob | Reports | Verdict | Production state |
|------|---------|---------|------------------|
| `alpha_freq_lambda` | 040 | **Empirically null** (λ ∈ {0, 0.5, 1.0} indistinguishable; λ=2.0 supercritical) | Default 0.0 (Phase 5 design closed the path; mass-death is the schema filter) |
| `inhibition_gain` (Saighi A_k) | 034, 035, 036 | **Falsified at n=10** (orthogonal to the dominant signal) | Default 0.0 |
| `coverage_lambda` (Candidate A) | 045, 046, 047, 048, 049, 053-064 | **Integrated** | `coverage_lambda=1.0` baked into the production a1prime/M1 substrate |
| `retrieval_weight_epsilon/tau` (Candidate B / Step 3) | 046, 048, 049, 053, 054 | **Empirically inert** on A1' substrate (uniform bias, softmax-shift-invariant after the Report 054 bugfix) | Defaults retained but mechanism inactive |
| `metastability_obs_rate` (Pair #4) | 052 | **Falsified at n=1 smoke** under BOTH operationalizations (Path 1 fixed-point and Path 3 trajectory-based); pivoted per design pre-commit | Default 0.0 |

**Net retraction.** The verification report's section "Structural finding —
wired-but-unrun knobs" and its claim that "the project has a recurring
pattern of 'wire-the-knob, default-it-off, never-run-the-confirmation'"
are **wrong**. The actual pattern is: wire-the-knob, default-it-off,
**run a numbered Colab/n≥1 smoke**, **falsify or integrate**, leave
the default at 0.0 for backward compat with prior substrates. That's
the discipline working as designed, not a structural backlog.

---

## Per-knob detail

### 1. `inhibition_gain` (Saighi & Rozenberg 2025, A_k self-inhibition)

- **Default:** `0.0` ([consolidation.py:72](../../src/energy_memory/phase4/consolidation.py))
- **Reports:**
  - [Report 034](../../reports/034_saighi_ak_seed1_prototype.md) — seed-1 prototype, positive (Δtop1 +0.009, ΔR@10 +0.022, Δcap +0.022)
  - [Report 035](../../reports/035_saighi_ak_n10_falsification.md) — **n=10 falsification of report 034**. CI crosses zero.
  - [Report 036](../../reports/036_decay_sweep_and_mass_death_finding.md) — A_k decay sweep + seed-17 trajectory. Title: "A_k is orthogonal to the dominant signal."
- **Status:** closed. The design-spec headlines move (or don't) because of mass death, not because of A_k self-inhibition. The Phase 4 graduation in [Report 038](../../reports/038_phase4_d1_graduation.md) was achieved with `inhibition_gain=0.0`.

### 2. `coverage_lambda` (Candidate A from 2026-05-20)

- **Default in code:** `0.0` ([consolidation.py:94](../../src/energy_memory/phase4/consolidation.py))
- **Default in production:** **`1.0`** — baked into [scripts/run_phase5_ab_pilot_seed17.sh](../../scripts/run_phase5_ab_pilot_seed17.sh) and downstream Phase-5 pilots
- **Reports:**
  - [Report 045](../../reports/045_phase5_ab_pilot_seed17.md) — A+B 1-seed pilot retrain. Pre-committed knobs: `alpha_anti=1.0`, `coverage_lambda=1.0`, `coverage_ema_rate=0.01`.
  - [Report 047](../../reports/047_phase5_ab_branch_divergence_failure.md) — K-branch divergence FAIL at n=1 on A+B+step3 substrate.
  - [Report 048-049](../../reports/048_phase5_a1_pilot_seed17.md) — A1 and A1' pilots: discovery atoms' max effective_strength fell 12.1 → 9.6 → 0.029 (99.8% reduction).
  - Reports 053-064: continued use of `coverage_lambda=1.0` as part of the active a1prime/M1 substrate.
- **Status:** integrated. The current Phase-5 work is **on top of** Candidate A. Default 0.0 in code is for backward compat with pre-2026-05-20 baselines.

### 3. `retrieval_weight_epsilon` / `retrieval_weight_tau` (Candidate B / Step 3)

- **Defaults:** `epsilon=0.05`, `tau=0.02` ([consolidation.py:111-112](../../src/energy_memory/phase4/consolidation.py)); active when `coverage_lambda > 0`.
- **Reports:**
  - [Report 046](../../reports/046_phase5_ab_pilot_seed17_step3.md) — A+B+step3 pilot.
  - [Report 048-049](../../reports/048_phase5_a1_pilot_seed17.md) — A1 and A1' pilots.
  - [Report 054](../../reports/054_phase5_step3_smoke_seed17.md) — **bug uncovered**: reports 047-053 had `score_bias` never threaded through `experiments/40_phase5_branching.py::settle_branch_with_prior`. Walk-back patch in commit landed for Report 054. **Post-bugfix verdict** at n=20: "step 3 empirically inert on A1' substrate (uniform bias, softmax-shift-invariant)."
- **Status:** tested and empirically inert on the current substrate. Mechanism survives in code (286 tests pin its bit-identity at zero-bias) but does not move the headline.

### 4. `metastability_obs_rate` (Pair #4 from 2026-05-20)

- **Default:** `0.0` ([consolidation.py:121](../../src/energy_memory/phase4/consolidation.py))
- **Reports:**
  - [Report 052](../../reports/052_phase5_pair4_smoke_falsification.md) — **Falsified at n=1 smoke under BOTH operationalizations**: Path 1 fixed-point (`c_i = w_i · (1 − max_w)`) AND Path 3 trajectory-based (`c_i^(traj) = max_t w_i^(t) − w_i^(final)`).
- **Status:** closed at smoke gate. Per the design's pre-committed escalation rule, the response was pivot to **pair #2** (drift/replay-pressure), not a third c_i reformulation. The pair #2 pivot is what feeds the 2026-05-23 SNR-walk-back chain.

### 5. `alpha_freq_lambda` (already audited)

See the freq-α walk-back and [Report 040](../../reports/040_freq_weighted_alpha_sweep.md). λ ∈ {0, 0.5, 1.0} indistinguishable at n=10×40 production scale; λ=2.0 supercritical. Phase 5 design doc closed the path.

---

## The actual pattern

The five knobs share one shape: each gates a 2026-05 (or earlier) design
note's named mechanism. Each was tested at the level the design called for
(n=1 smoke for Pair #4, n=10 for A_k and freq-α, integrated production for
Candidates A+B). Four were falsified or inert; one (Candidate A) is in the
production substrate.

The verification report's mistake was reading **default 0.0** as **never
exercised**. The default exists to make the *prior* substrate
reproducible — not because the knob is unused. The codex audit on freq-α
showed this exact pattern; this audit shows it across all four remaining
knobs.

## Net effect on the brainstorm

Two earlier-listed "wired-but-unrun" items are gone:

| Original brainstorm framing | Corrected framing |
|---|---|
| "Run Candidate A + B at the same conditions. Already wired." | A is in production; B is post-bugfix-inert. Nothing to run. |
| "The project has a recurring 'wire-the-knob, default-it-off, never-run-the-confirmation' pattern." | The discipline is working: every wired knob has a numbered report and a verdict. |

What survives as next-session work (post-audits):

1. **MQAR as a tracked benchmark** — still the strongest finding. Hopfield key-only recall at 0% from N=128 is the geometric explanation for the four Phase-5 retrieval-mechanism nulls.
2. **Range-shaped replay** (Dorrell-Whittington ICLR 2025) — genuinely unbuilt; prototype works; 3-5 days to ship.
3. **Residue HDC / GSBC substrate-algebra swap** — the substrate change the MQAR result motivates.
4. **Re-examine bundle-style storage as the Phase-5/6 primitive** — the MQAR smoke shows HRR-bundle gives 100% top-1 key-only recall at N=128 on the same substrate where Hopfield gets 0%. The Phase 5 architecture has been targeting Hopfield-with-roles; bundle-with-disambiguation may be the natural shape.
5. **Open architectural questions** from the dated synthesis notes (HDC/SONAR embedding-space coexistence, trajectory-trace meta-loop) — still unbuilt; not affected by either audit.
