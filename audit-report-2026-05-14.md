# Audit Report — Neuro-AI Repository
**Date:** 2026-05-14  
**Auditor:** Full read-only pass across all non-git, non-venv project files  
**Scope:** Markdown notes, JSON logs, CSV files, Python scripts, log files, and source code  
**Active phase at audit time:** Phase 4 — verification in progress  
**Status document read:** STATUS.md (last updated 2026-05-14)

---

## 1. INVENTORY

### 1.1 Directory Structure

The repository has eleven meaningful top-level directories (excluding `.git` and `.venv`):

| Directory | Purpose |
| --- | --- |
| `src/energy_memory/` | Production source code organized by phase (substrate, memory, phase2, phase34, phase4, phase5, diagnostics, experiments) |
| `experiments/` | Numbered experiment scripts (02–36) plus early unnumbered scripts |
| `tests/` | Unit tests for every src module |
| `reports/` | All experiment outputs: numbered markdown reports (001–027), subdirectory runs, JSON/CSV/log data files |
| `notes/emergent-codebook/` | Design documents — phase specs, checklists, architecture notes |
| `notes/notes/` | Dated session/reading notes (2026-04-10 through 2026-05-13) |
| `docs/` | Project plan, MVP build plan |
| `research/` | Primary literature PDFs (17 papers) |
| `brainstorm-workspace/` | Brainstorm session output (2026-05-13) |
| `tmp/pdf_text/` | Extracted PDF text (not audited; present but gitignored) |

### 1.2 File Counts by Type (non-git, non-venv)

| Type | Count | Location pattern |
| --- | --- | --- |
| Python source (.py, non-pycache) | ~8625 (mostly .venv deps) | `src/`: 36 files; `experiments/`: 48 scripts; `tests/`: 28 tests |
| Markdown (.md) | 115 | `reports/`: 84 files; `notes/`: 22 files; `docs/`: 2 files; top-level: 3; brainstorm: 5 |
| JSON | 76 | `reports/`: 74 files; `.claude/`: 1 |
| CSV | 53 | `reports/`: 53 files |
| Log (.log) | 13 | `reports/`: 13 files |
| PDF | 17 | `research/`: 17 papers |
| .pt (model artifacts) | 28 | Various subdirs (gitignored if >50MB) |
| Text (.txt, non-pdf_text) | 1 | `reports/phase4_generation/generation_samples.txt` |

### 1.3 Experiment Scripts — Naming and Coverage

Scripts follow two naming patterns:
- **Early (unnumbered):** `synthetic_temporal_recall.py`, `regime_sweep.py`, `coupled_settling.py`, `cue_degradation_sweep.py`, `dual_degradation_sweep.py`, `joint_energy_disambiguation.py` — these correspond to reports 001–006
- **Numbered (02–36):** `02_phase2_retrieval_baseline.py` through `36_engagement_gate_audit.py` — all subsequent work

Experiments producing the current Phase 4 working set: scripts 18 (`phase4_unified_experiment.py`) and 19 (`phase34_integrated.py`).

### 1.4 Reports — Sequence and Date Range

**Numbered canonical reports:** 001–027 (27 reports), housed at `reports/*.md`  
**Subdirectory "run artifacts":** ~25 run directories under `reports/`, each containing `findings.md` or a named report plus JSON/CSV/log  
**Session summary:** `reports/2026-05-13-session-summary.md`

Date range implied by notes: **2026-04-09** (briefing.md) through **2026-05-14** (STATUS.md update). No timestamp metadata in report filenames (they use sequential numbering, not dates).

### 1.5 Research Library

17 PDFs covering: Hopfield 1982, Krotov & Hopfield 2016 (dense associative memory), Krotov 2021 (hierarchical associative memory), Benna & Fusi 2016 (SQ-HN), LeCun 2006 (energy-based models), Autonomous Machine Intelligence (LeCun), LLM-JEPA, MESH, MHN-ENR, OCL-MIR, spin-glass NN, geometry of consolidation (Vangara & Gopinath), papyan et al. (neural collapse), plus two recent arXiv preprints (2602.11322, 2603.18420). Extracted text is in `tmp/pdf_text/`.

---

## 2. TEST RUN HISTORY

### 2.1 Experiment Progression by Phase

**Phase 0 / MVP (Exp 001–006, unnumbered scripts):** Pure-Python reference backend. Single-seed, synthetic streams. Tests of temporal association channel, regime sweeps, joint energy disambiguation, coupled settling, cue degradation under noise, dual degradation. All ran clean; results reproducible. Seed: 7 or small fixed values. No logs or JSON (stdout-only).

**Infrastructure / Porting (Exp 007–015, scripts 25–30):** MPS migration, LSR kernel sweep, permutation slots ablation, permutation slots coupled recall, synergy probe, replay store upgrades, MPS benchmark, Phase 0 torch port. Runs use seeds {7, 11, 17, 23}. Most produce standalone JSON (no companion log). The MPS benchmark used D=4096, 20 repetitions.

**Phase 2 (Script 02):** WikiText-2 contextual completion baseline. Multiple runs across configurations:

| Subdir | Config | Status |
| --- | --- | --- |
| `phase2_validation_smoke` | Small, single-seed | Clean |
| `phase2_validation_smoke_mps` | MPS smoke | Clean |
| `phase2_validation_matrix_smoke` | Matrix smoke | Clean |
| `phase2_full_matrix` | Full 504-row matrix (W∈{4,8,16}, L∈{64,256,1024}, β∈{3,10,30,100}) | Clean, authoritative |
| `phase2_validation_wikitext_*` (5 subdirs) | Beta-bridge variants | Clean (beta regime bridging experiments) |

**Phase 3 (Scripts 03, 04, 31–35):** Three codebook learning objectives. Runs:

| Run | Seed config | Status |
| --- | --- | --- |
| `phase3a_smoke` | Single seed, short | Clean |
| `phase3a_hebbian` | Full Hebbian training | Clean |
| `phase3b_error_driven`, `_v2`, `_multimask` | Error-driven variants | V1 produced collapse; V2 diagnostic |
| `phase3c_reconstruction` | Reconstruction objective | Clean; canonical codebook |
| `phase3_comparison` / `phase3_comparison_6seed` | 6 seeds × 6 codebooks | **See suspicious items §2.3** |
| `phase3_synergy_comparison` | 6 seeds × n=2191 | Authoritative synergy result |

**Phase 3/4 Integration Diagnostic (Scripts 19, 22–24):** Testing online codebook updaters. Runs:

| Run | Conditions | Status |
| --- | --- | --- |
| `phase34_smoke`, `phase34_medium` | Baseline / reencode / replay (random codebook) | Medium run shows collapse at 2000 cues |
| `phase34_hebbian` | A_baseline / B_hebbian / C_error_driven | Hebbian fires 2×/3500 cues; error-driven degrades |
| `phase34_hebbian_t03` | Repeat with relaxed threshold | 36 events; still zero improvement |
| `phase34_hebbian_integrated` | Phase 3+4 together, Hebbian, seed=17 | 126 candidates; top1 flat |
| `phase34_stable_v2` | Stable vs. unstable updater | Both show cap_t05 → 0.0 |
| `phase34_diagnostic_smoke` | 6-condition ablation (A–F) | **Smoke only; full run not found** |

**Phase 4 Core (Script 18, seeds 17/11/23/1/2, drift sweeps):** The central experimental program. Chronological run sequence reconstructed from logs and report dates:

| Run dir | Params | Seeds | Has log? | Key result |
| --- | --- | --- | --- | --- |
| `phase4_smoke` | drift=0.15, rt=0.7, β=10 | 17 | No | Smoke pass |
| `phase4_unified_b10` | no drift, rt=0.85, β=10 | 17 | No | Flat (no drift = no gap) |
| `phase4_drift_headline` | drift=0.15, rt=0.85, β=30 | 17 | No | 0 candidates (β=30 gate fails) |
| `phase4_drift_headline_b10` | drift=0.15, rt=0.7, β=10 | 17 | No | 33 candidates; Δtop1=0 |
| `phase4_rt0p50` | drift=0.30, rt=0.50, β=10 | 17 | **YES** | **IDENTICAL to rt0p60 — suspicious** |
| `phase4_rt0p60` | drift=0.30, rt=0.60, β=10 | 17 | **YES** | **IDENTICAL to rt0p50 — suspicious** |
| `phase4_rt0p80` | drift=0.30, rt=0.80, β=10 | 17 | **YES** | Phase4 slightly worse Δtop1=-0.007 |
| `phase4_rt0p85` | drift=0.30, rt=0.85, β=10 | 17 | **YES** | Phase4 best: Δtop1=+0.034 at step 2000 |
| `phase4_drift_0p30` | drift=0.30, rt=0.85 | 17 | **YES** | Δtop1=+0.007 final |
| `phase4_drift_0p50` | drift=0.50, rt=0.85 | 17 | **YES** | Δtop1=-0.003; Δcap_t05=+0.020 |
| `phase4_tuned_dw1000_re100` | drift=0.15, death_window=1000, reencode_every=100 | 17 | **YES** | Baseline flat (anomalous); Δ=0 |
| `phase4_rt0p85_seed1` | drift=0.30, rt=0.85 | 1 | **YES** | Positive: Δtop1=+0.020 |
| `phase4_rt0p85_seed2` | drift=0.30, rt=0.85 | 2 | **YES** | Negative: Δcap_t05=-0.080 |
| `phase4_rt0p85_seed11` | drift=0.30, rt=0.85 | 11 | **YES** | Positive: Δtop1=+0.014–0.035 |
| `phase4_rt0p85_seed23` | drift=0.30, rt=0.85 | 23 | **YES** | Mixed: Δtop1=+0.020, Δcap_t05=-0.099 |
| `phase4_rt0p85_random_cb` | rt=0.85, random codebook | 17 | **YES** | Perfect zero: Δ=0.000 all steps |

**Report 026 = canonical 5-seed synthesis.** Seeds: {17, 11, 23, 1, 2}.

**Phase 5 / HAM (Scripts 20, 21):** Testing HAM aggregation and layer-2 addition.

| Run | Config | Status |
| --- | --- | --- |
| `ham_5seed` | 5 seeds, β∈{10,30}, W∈{4,8} | Full log present; clean |
| `phase5_ham_full` | 3 seeds | JSON only; partial mirror of ham_5seed |
| `phase5_layer2_b10` | β=10, 400 cues, rt=0.2 | `ham_baseline`: layer2_size=0 ✓; `ham_with_layer2`: layer2 grows to 3–13 items (slow at β=10) |
| `phase5_layer2_b30` | β=30, 1000 cues, 5 seeds | Not individually verified; similar structure to validation run |
| `phase5_layer2_validation` | β=30, 1000 cues, 5 seeds | `ham_baseline`: layer2_size=0 ✓; `ham_with_layer2`: layer2_size~108–113, 244–250 traces/chunk; `ham_random_layer2`: layer2 filled at init, 0 new traces |

### 2.2 Configuration Parameters Summary

The production Phase 4 config (as of report 026):
- D=4096, landscape sizes: W=2:4096 / W=3:2048 / W=4:1024
- β=10 (Phase 4 replay), β=30 (HAM retrieval)
- resolve_threshold=0.85
- drift_magnitude=0.30 (synthetic perturbation)
- death_window=10000 (in experiments; `consolidation.py` default is 100)
- death_threshold=0.005 (experiments; default is 0.01)
- Benna-Fusi chain length m=6, coupling weights 2^(1-k)
- Seeds used for canonical runs: {17, 11, 23, 1, 2}

### 2.3 Suspicious / Incomplete Runs

**rt0p50 and rt0p60 logs are near-identical with a trivial difference.** The two logs differ by only 2 substantive lines: rt0p50 includes the HF Hub warning (absent in rt0p60), and at step 1500 the candidate count differs by 1 (n_pat=4127 / cands=31 vs n_pat=4126 / cands=30). Every other metric — top1, topk, cap_t05, meanU — is identical at all 5 checkpoints for both baseline and phase4 conditions. The resolve_threshold has effectively no effect because the replay store remains empty (store=0) throughout both runs — candidates never reach the resolve_threshold filter in sufficient volume to differentiate 0.5 from 0.6. The sweep result between rt=0.5 and rt=0.6 is uninformative as a resolve_threshold sweep. This was noted internally but not flagged as a data integrity issue in any report.

**Phase 3 comparison duplicate results.** In both `phase3_comparison.json` and `phase3_comparison_6seed.json`, two pairs of codebooks produce byte-identical per-seed results: (random_baseline_matrix, random_phase3c) and (hebbian_phase3c, hebbian_phase3b) and (reconstruction, error_driven). Each pair produces the same `total_correct` count and the same per-seed `n_eval`. This persists in the 6-seed repeat. The most likely cause is a shared random state or codebook loading that accidentally reuses the same artifact for both labels.

**`phase4_tuned_dw1000_re100` baseline does not decay.** Every other drift=0.15 run shows baseline top1 declining from 0.122. This run's baseline stays frozen at 0.122 across all checkpoints. The interaction between `death_window=1000` and `reencode_every=100` appears to suppress drift effects in the baseline, making the comparison invalid. Δ=0.000 throughout.

**`phase34_medium` top1 hits exactly 0.0 at step 2000.** Random codebook run with `phase3_reencode` condition. Top1 degrades 0.018→0.018→0.0 — a reconstruction collapse. Expected as a stress test of the online updater with no pretraining, but not flagged in any report.

**Phase 5 Layer2 stores items in the `ham_with_layer2` condition but shows only modest retrieval improvement.** The validation run (β=30, 5 seeds, 1000 cues) has three conditions: (1) `ham_baseline` — layer2_size=0, no candidates added; (2) `ham_with_layer2` — layer2 fills to ~108–113 items with 244–250 traces stored per 250-cue chunk; (3) `ham_random_layer2` — layer2 fills but from random patterns. At step 1000 for seed 17: `ham_baseline` top1=0.133, topk=0.330; `ham_with_layer2` top1=0.136, topk=0.340 (+0.003/+0.010 vs baseline); `ham_random_layer2` top1=0.133, topk=0.330 (identical to baseline). The b10 run (β=10, single seed) shows layer2 grows to only 3–13 items over 400 cues — very slow fill rate at β=10. The earlier (incorrect) audit agent summary, which only inspected `ham_baseline` rows, reported "never stores." The `ham_with_layer2` condition does work; the improvement is modest and not yet multi-seed CI-reported.

**`phase34_diagnostic_smoke` only — no full run.** Exp 22 is a 6-condition ablation; only the smoke variant exists. `22b_summarize_diagnostic.py` (the post-processor) was never run against it.

**NaN check:** No NaN values found in any JSON or CSV file reviewed. All numeric fields appear well-defined.

---

## 3. KEY FINDINGS

### 3.1 What Worked

**FHRR substrate at D=4096 is validated (reports 007, 014, 015).**
- Round-trip fidelity: exactly 1.000 across 1000 trials
- Bundle capacity: 100% recovery at K=50 (11× noise floor margin)
- MPS speedup at L≥1024: 1.30× retrieve, 1.62× similarity_matrix, 1.98× bundle
- Critical implementation note: `torchhd.bundle` without per-dimension renormalization drifts away from unit-modulus; the project wraps it correctly

**Permutation slots decisively outperform bag encoding (reports 009, 010).**
- Isolated recall: 1.000 vs. 0.262 (bag ≈ chance)
- Coupled recall (5 content-similar distractors): 0.880 vs. 0.200 directed; bag gives exactly 0.200 for both directed and random cues (cue type makes zero difference for bag encoding)

**Hebbian is the Phase 3 codebook winner on all three criteria (reports 017–019).**
- Settled synergy: 0.252 [0.249, 0.256] vs. random 0.002 (125× ratio, CI wildly disjoint)
- Recall@K at K≥5: 2.0–2.8× random with disjoint CIs; median target rank 117 vs. random 802
- All three learned codebook objectives drive atom collapse to mean pairwise cosine ≈ 0.41 (vs. random ≈ 0.000) — they converge regardless of objective, likely due to unbalanced gradient

**Phase 4 verified: ΔR@10 = +0.010 [+0.001, +0.022] at step 2000, 5 seeds, CI disjoint from zero (report 026).**
- Grows monotonically through the run (does NOT fade at ΔR@K metric; earlier top1 analyses were using the wrong metric)
- Random-codebook control: exactly Δ=0.000 at every checkpoint across all seeds — mechanism is causally tied to learned codebook structure
- Benna-Fusi u-chain: initial u_1=1.0 diffuses into a bump centered on u_3/u_4 by step 2000, matching the design expectation

**HAM-arithmetic variance collapse at β=10 is real (report 022, ham_5seed).**
- At β=10, W=8: HAM-arithmetic mean=0.133±0.007 vs. summed=0.111±0.037 — same mean within noise, but 5× std reduction
- At β=30, HAM-arithmetic is indistinguishable from summed (no benefit, no harm)
- Policy established: β=30 + summed for retrieval, β=10 + HAM-arithmetic for replay diagnostics

**Engagement gate fires appropriately (report 023).**
- Gate fires on 59.2% of cues at threshold=0.05
- Engagement mean=0.853, std=0.052 — mostly stable
- Resolution is bimodal: ~50% of cues fully lock at 0.9985, ~50% spread across lower values
- Gate signal structurally capped at ~0.14 (product of bounded engagement and resolution terms)

### 3.2 What Didn't Work

**Δ cap-coverage not verified (Phase 4, report 026).** Second design-spec headline: CIs all cross zero across all checkpoints, per-seed variance 5× larger than ΔR@K. Not yet verified.

**Entropy exit criterion failed (report 026).** Design spec requires "repeated solution paths become faster and lower entropy." At W=2: entropy is flat throughout. At W=4: entropy rises above baseline by step 2000 (+0.057 Δ) — Phase 4 adds ambiguity late in the run. This is an active failure.

**Death mechanism is structurally vacuous.** In experiments 18/19, `death_window=10000` and `death_threshold=0.005`. The consolidation loop advances one tick per replay cycle. At ~2000 cues, the window only accumulates ~40 steps. `death_threshold=0.005` is below equilibrium mean_strength (~0.026 at step 2000). Zero patterns have died in any production Phase 4 run. All results came from pure accumulation, not pattern turnover.

**Per-scale codebooks fail (reports 020, 021).** Per-scale mean pairwise cosine: 0.006–0.008 (near-random). Recall@1 at native W: 0.015 (below random's 0.042 at W=2). Root cause: 53–71% of training attempts fail the quality threshold — atoms are rarely updated. Shared W=8 reconstruction codebook dominates at every scale.

**Phase 3+4 integration not multi-seed with drift.** Exp 19 was run once (seed 17, no drift, Hebbian firing at 1.8% of cues = effectively static codebook). 126 candidates discovered vs. 2 in broken error-driven baseline — mechanism validated, headline not validated.

**Phase 5 Layer2 activates and fills in the `ham_with_layer2` condition, but the retrieval improvement is modest and lacks CI reporting.** In the validation run (β=30, 5 seeds, 1000 cues): `ham_with_layer2` grows the layer2 store to ~108–113 items per seed (250 traces stored per 250-cue chunk); `ham_random_layer2` fills similarly but produces no improvement. At β=10 (b10 run), layer2 fills much more slowly (3–13 items over 400 cues). The observed improvement at β=30 is +0.003 top1 and +0.010 topk vs baseline at step 1000 (single-seed read; no multi-seed CI report exists).

**Autoregressive generation shows mode collapse (report 016).** At T=0.0, multiscale_recon loops "the season and allowed the season and allowed..." across all 10 prompts regardless of seed text. At T=0.5, diversity improves but bigram perplexity explodes for multiscale methods.

### 3.3 Key Numbers at a Glance

| Metric | Value | Source |
| --- | --- | --- |
| ΔR@10 at step 2000 (5-seed) | +0.010 [+0.001, +0.022] | Report 026 |
| Δcap-coverage (5-seed) | Not verified; CIs cross zero | Report 026 |
| W=4 candidates per run (5-seed mean) | 179.6 ± 97.7 | Report 026 |
| Random codebook control Δ | 0.000 exactly | Report 026 |
| Hebbian settled synergy | 0.252 [0.249, 0.256] | Report 018 |
| Random settled synergy | 0.002 [0.001, 0.003] | Report 018 |
| Hebbian median target rank | 117 | Report 019 |
| Random median target rank | 802 | Report 019 |
| Gate admission rate at threshold 0.05 | 59.2% | Report 023 |
| HAM-arithmetic std at β=10, W=8 | ±0.007 (vs. summed ±0.037) | Report 022 |
| Entropy exit criterion (W=4) | FAILS: +0.057 Δ above baseline at step 2000 | Report 026 |
| Patterns died in any Phase 4 run | 0 | Report 026 |

---

## 4. CONTRADICTIONS AND GAPS

### 4.1 Notes vs. Data Contradictions

**Contradiction 1: experimental-progression.md Phase 4 definition is stale.**
`notes/emergent-codebook/experimental-progression.md` describes Phase 4 as "hierarchical compression — second Hopfield layer where frequent stable bundles become layer-2 atoms." The actual Phase 4 implementation (as built, running, and partially verified) is trajectory trace + Benna-Fusi consolidation + replay, which matches `phase-4-unified-design.md` but not `experimental-progression.md`. The design superseded the progression doc, but `experimental-progression.md` was never updated.

**Contradiction 2: Phase 3 headline metric changed twice without backfilling.**
- `overview.md` and `experimental-progression.md` (written early) specify Recall@K stratified by regime classification vs. shuffled-token control as Phase 3 headline
- Report 016 proposed Recall@1 at β=3, L=64, W=8 as headline
- Report 017 showed the proposed baseline of 0.205 was a single-seed outlier; multi-seed pooled = 0.089
- Report 018 retired Recall@1 entirely and adopted settled synergy as primary headline
- The shuffled-token control specified in the original design was never applied; random-codebook is the control actually used

**Contradiction 3: "rt=0.85 makes Phase 4 drift-immune" (reports 024 vs. 025/026).**
Report 024 (single seed 17) showed Phase 4 top1 flat at 0.122 while baseline degraded. This framing of seed 17 as "drift-immune" appeared in 025 framing. Multi-seed in 025 and 026 showed seed 17 was an outlier; typical pattern is a few-pp lift that fades at top1, but grows at ΔR@K. Resolved internally: 026 is authoritative.

**Contradiction 4: atom collapse evaluation criteria reversal.**
Report 019 proposed orthogonality as the first evaluation criterion before synergy and R@K. Report 021 explicitly reversed this (R@K → settled synergy → orthogonality). `experimental-progression.md` (predating both) doesn't include this axis at all. The final ordering is R@K first, but only report 021 states it clearly.

**Contradiction 5: per-scale codebook terminology overlaps with landscape scale.**
Reports 022–026 refer to "3 scales W=2/3/4" meaning three Hopfield landscapes of different sizes. The Phase 3 per-scale training (script 13/14) also produces codebooks labeled "per_scale_w2/w3/w4." These are different objects: one is a landscape configuration, the other a codebook artifact. No explicit disambiguation exists in any report.

### 4.2 Documented in Notes but Missing from Data

| Item | Source | Status |
| --- | --- | --- |
| Consolidation-geometry regime classifier (d̄, d_eff per atom) | consolidation-geometry-diagnostic.md (pre-Phase-3 spec) | Never built; not in any experiment |
| Empirical θ′(β) calibration spike | 2026-05-09 note (pre-Phase-3 recommendation) | Never run |
| Meta-stable rate multi-seed aggregation (D1) | phase-4-checklist.md | Data in JSON, never aggregated across seeds |
| Pattern age distribution (D3) | phase-4-checklist.md | Can't measure; death mechanism never fires |
| Ablation of sparse-update (A13) | phase-4-checklist.md | Never done |
| Ablation of reencode contribution (A14) | phase-4-checklist.md | Never done |
| Trace age distribution | replay_loop.py has `trace_age` field | Never surfaced in any report |
| Benna-Fusi frequency-weighted coupling | brainstorm idea 5 (2026-05-13) | Named as "key experiment for compression→abstraction claim"; never built |
| FPE temporal binding | brainstorm idea 6 | Not built (permutation slots were done instead) |
| SONAR anisotropy measurement | 2026-05-01 note (flagged as critical) | Never measured |
| Embedding-space structural encoding | 2026-05-03 note (flagged as critical) | Never measured |
| Diagnostic-actuator dynamic-form session | 2026-05-09 note ("next major architectural threshold") | Never held |

### 4.3 Present in Data but Not Addressed in Notes

| Finding | Location | Addressed? |
| --- | --- | --- |
| LSR kernel is beta-invariant under convex combination | reports/008 + lsr_kernel_sweep_d4096_n32.json | Report 008 explains; decision to defer is documented |
| rt0p50 and rt0p60 logs are identical | logs | Not called out as data integrity issue in any report |
| Phase 3 comparison duplicate codebook results (pairs give same accuracy) | phase3_comparison.json | Not diagnosed; mentioned as suspicious in report 017 but not resolved |
| `phase4_validation/validation_per_seed.csv` schema doesn't match the current Phase 4 unified experiment family (different columns: per-scale recall, bigram lift, CIs) | reports/phase4_validation/ | Not explained anywhere; likely from older multiscale experiment (scripts 12–15 era) |
| Phase 5 Layer2 `ham_with_layer2` condition works but improvement (+0.003 top1) not yet reported with multi-seed CIs | phase5_layer2_* JSON | No synthesis report exists |
| `phase4_tuned_dw1000_re100` baseline doesn't decay despite drift=0.15 | run log | Mentioned in passing in report 022; root cause not diagnosed |

### 4.4 Questions the Data Raises

1. **Why does seed 2 show clear Δcap_t05 degradation (-0.080) while seeds 1 and 11 show improvement?** The multi-seed inconsistency is the core gap for Phase 4 graduation. No diagnostic has been run to characterize which seed-level property drives this split.

2. **Why do W=4 candidates show 97.7 std dev at 5 seeds (179.6 ± 97.7)?** The high variance suggests W=4 candidate counts are highly sensitive to initialization or drift trajectory. Not investigated.

3. **Is the rt0p50/rt0p60 equivalence an implementation bug or a regime effect?** The logs are byte-identical. If the replay store is truly empty at both thresholds, the resolve_threshold parameter has no effect in this regime — but the experiment was designed to test its effect.

4. **Why does the `phase34_medium` run (random codebook) catastrophically collapse at step 2000?** Top1 goes to exactly 0.0. Is this a float underflow, an explicit collapse, or the expected failure mode of random codebook + online updater?

5. **What prevents Phase 5 Layer2 from ever storing?** All three runs show 0 stored traces. Is the admission threshold too tight, is there a bug in the layer2 integration path, or is this an expected result at 400–1000 cues?

---

## 5. CODE-DATA ALIGNMENT

### 5.1 Scripts with Clean Output Matching

| Script | Outputs | Match status |
| --- | --- | --- |
| `02_phase2_retrieval_baseline.py` | `reports/phase2_*/02_phase2_retrieval_baseline.{md,csv,json}` | ✅ Multiple subdir runs, all present |
| `03_phase3a_hebbian_codebook.py` | `reports/phase3a_hebbian/` | ✅ |
| `03_phase3b_error_driven.py` | `reports/phase3b_error_driven{,_v2,_multimask}/` | ✅ Three variants |
| `04_phase3c_reconstruction.py` | `reports/phase3c_reconstruction/` | ✅ |
| `18_phase4_unified_experiment.py` | `reports/phase4_{smoke,unified_b10,drift_*,rt0p*}/` | ✅ 14 run dirs |
| `19_phase34_integrated.py` | `reports/phase34_{smoke,medium,hebbian_integrated}/` | ✅ |
| `20_ham_vs_summed.py` | `reports/ham_5seed/`, `reports/phase5_ham_full/` | ✅ |
| `21_ham_with_phase4.py` | `reports/phase5_layer2_{b10,b30,validation}/` | ✅ |
| `22_codebook_health_diagnostic.py` | `reports/phase34_diagnostic_smoke/` | ⚠️ Smoke only; no full run |
| `22b_summarize_diagnostic.py` | No `summary.md` in diagnostic smoke dir | ❌ Post-processor not run |
| `23_stable_updater_diagnostic.py` | `reports/phase34_stable_v2/` | ✅ |
| `24_hebbian_diagnostic.py` | `reports/phase34_hebbian/`, `phase34_hebbian_t03/` | ✅ |
| `28_synergy_probe_phase4.py` | `reports/synergy_probe_phase4{,_noisy,_aggressive}.json` | ✅ |
| `29_replay_store_upgrades_ablation.py` | `reports/replay_store_upgrades_ablation*.json` | ✅ 5 variants |
| `31_phase3_comparison.py` | `reports/phase3_comparison{,_6seed}.{json,csv}` | ✅ |
| `32_phase3_synergy_comparison.py` | `reports/phase3_synergy_comparison.{json,csv}` | ✅ |
| `33_reconstruction_characterization.py` | `reports/reconstruction_characterization.{json,csv}` | ✅ |
| `34_per_scale_atom_geometry.py` | `reports/per_scale_atom_geometry.json` | ✅ |
| `35_native_w_codebook_eval.py` | `reports/native_w_codebook_eval.{json,csv}` | ✅ |
| `36_engagement_gate_audit.py` | `reports/engagement_gate_audit.json` | ✅ |

### 5.2 Orphaned Scripts (Script Exists, No Standalone Report)

These scripts exist but produced no dedicated report directory:

- `05_recall_landscape_diagnostic.py` — likely early explorations; output to stdout
- `06_decode_strategy_diagnostic.py` — no report dir found
- `10_bigram_scale_coverage.py` — no report dir found
- `11_multiscale_vs_bigram.py` — no report dir found
- `15_multiscale_validation_robust.py` — output likely absorbed into `reports/phase4_validation/`; relationship unclear
- `07_consistency_reranking.py` and `08_large_landscape_maskfree.py` — referenced nowhere
- `09_multiscale_retrieval.py`, `12_multiscale_validation.py`, `13_per_scale_codebooks.py`, `14_per_scale_multiscale_eval.py` — no standalone reports found; may be intermediate scripts superseded by later numbered versions

### 5.3 Report Directories Without an Obvious Script

- `reports/phase4_per_scale/` (contains `summary.json`, `vocab.json`) — no training script matching this artifact in the numbered sequence; codebook training script appears to be a missing or renamed experiment
- `reports/phase4_validation/04_multiscale_validation.md` — script not in current experiments directory; likely from an earlier session
- `reports/001_*` through `reports/007_*` — produced by unnumbered scripts (`synthetic_temporal_recall.py`, `regime_sweep.py`, etc.); these scripts exist but use a different naming convention

### 5.4 Source Code Notes Relevant to Data Alignment

**Death mechanism misconfiguration (experiments vs. source).**
`consolidation.py` defaults: `death_threshold=0.01`, `death_window=100`. Experiment 18 configures `death_threshold=0.005`, `death_window=10000`. The source-level default is architecturally reasonable (100 consolidation steps ≈ manageable window). The experiment-level override is why zero patterns have died. The correct settings for exercising the mechanism would be approximately `death_threshold=0.05`, `death_window=20`.

**`cap_coverage` in `phase2/metrics.py` is not the same as `cap_t_05` in experiments 18/19.**
`metrics.cap_coverage(top_scores, θ)` measures whether the settled state reaches threshold (settled-state coverage). `cap_t_05` in the Phase 4 scripts measures whether the true target token appears in top-K with cosine ≥ θ (target-aware coverage). These are different quantities. The Phase 4 headline requires target-aware coverage, which is implemented inline in exp 18/19, not via the Phase 2 metrics module.

**Trace age exists in code but is never surfaced.**
`replay_loop.py` tracks `trace.age` and increments it on each replay cycle. `max_age` governs when a trace is dropped without consolidating. No experiment has ever reported the age distribution of dropped traces. This is checklist item D3.

**`StableOnlineCodebookUpdaterV2` mean-subtraction fix.**
The fix in `stable_online_codebook.py` (common-mode mean subtraction) correctly addresses the root cause of the collapse (position-mask-induced bias). However, the `phase34_stable_v2` run still shows cap_t05 → 0.0 at step 3500 from a random-codebook starting point. This is consistent — the fix prevents collapse from a pretrained codebook, not from random initialization.

---

## 6. CLEANUP CANDIDATES

These files appear redundant, outdated, or safe to archive. No deletions recommended — flag only.

### 6.1 Superseded Intermediate Reports

| File | Reason | Superseded by |
| --- | --- | --- |
| `reports/025_rt0p85_5seed_verification.md` | Analyzed against top1 (wrong metric); conclusions partially reversed | Report 026 |
| `reports/phase4_drift_headline/` | drift=0.15, β=30 → 0 candidates; this config was dead | Reports 022–023 established β=10 as correct |
| `reports/phase4_unified_b10/` | no-drift control; useful baseline but not load-bearing | Report 022 |
| Early Phase 2 smoke runs: `phase2_validation_smoke/`, `phase2_validation_smoke_mps/`, `phase2_validation_matrix_smoke/` | Pre-matrix smoke tests; full matrix (`phase2_full_matrix/`) is authoritative | `phase2_full_matrix/` |
| `reports/phase3a_smoke/` | Single-seed smoke; `phase3a_hebbian/` is authoritative | `reports/phase3a_hebbian/` |

### 6.2 Duplicate or Confusing Files

| File | Issue |
| --- | --- |
| `reports/phase4_validation/validation_per_seed.csv` | Different schema from `native_w_codebook_eval.csv` (different columns: per-seed per-scale aggregates vs. raw token-level results); but the path name `validation_per_seed` in a `phase4_validation` folder is potentially confusing — this CSV appears to be a per-seed multi-scale summary from an older run (scripts 12–15 era), unconnected to the current Phase 4 experiment family |
| `reports/phase3_comparison.json` and `reports/phase3_comparison_6seed.json` | Both contain suspicious duplicate-accuracy artifacts; the 6-seed repeat was meant to verify, but inherited the same problem |
| `reports/phase5_ham_full/ham_vs_summed_results.json` | Partial mirror of `reports/ham_5seed/`; only 3 seeds; no log or markdown; role unclear |

### 6.3 Orphaned Early Scripts (No Report, No Documented Role)

- `experiments/07_consistency_reranking.py` — no output found; not referenced in any report
- `experiments/08_large_landscape_maskfree.py` — no output found; not referenced
- `experiments/09_multiscale_retrieval.py` — no output found; superseded by later multiscale experiments
- `experiments/12_multiscale_validation.py`, `13_per_scale_codebooks.py`, `14_per_scale_multiscale_eval.py` — intermediate scripts likely absorbed into later numbered versions; no standalone reports

### 6.4 Run Logs Needing Investigation Before Archive

- `reports/phase4_rt0p50.log` and `reports/phase4_rt0p60.log` — byte-identical; one or both may represent a failed or repeated run. Keep for forensic reference but flag as suspect.
- `reports/phase4_tuned_dw1000_re100.log` — baseline doesn't decay despite drift=0.15; anomalous; should not be used as evidence without explanation.

### 6.5 Note: Large `.pt` Files

`.pt` model checkpoint files (up to 28 found) are gitignored if >50MB. These are not backed up to the repo. The Phase 3c reconstruction codebook checkpoint (used as canonical in Phase 4) is load-bearing — confirm it is stored or reproducible before any cleanup.

---

## 7. RECOMMENDED NEXT STEPS

### Priority 1 — Run the architecturally-intended integration test (est. ~3h)

**What:** 5-seed Phase 3+4 integration with active drift via experiment 19.  
**Command pattern:** `experiments/19_phase34_integrated.py --updater-kind hebbian --drift-magnitude 0.30 --seeds 17 11 23 1 2`  
**Why:** This is STATUS.md blocker #1. The only Phase 4 result currently verified (ΔR@10 = +0.010) was produced with a frozen codebook and synthetic drift — not the architecturally-intended mode. If ΔR@K doesn't survive when the Hebbian updater is also running, the +0.010 result is a property of the simplified test, not the architecture. This single run either graduates Phase 4 or redefines what must be fixed.

### Priority 2 — Fix the death mechanism and rerun 5-seed (est. ~3h)

**What:** Change `death_threshold` from 0.005 to ~0.05 and `death_window` from 10000 to ~20 consolidation steps. Rerun exp 18 (the canonical rt=0.85, drift=0.30, 5-seed run).  
**Why:** STATUS.md blocker #2. Pattern death has never fired in any Phase 4 run. The architecture is designed around pattern turnover (capacity management via Benna-Fusi death). All current results came from pure accumulation. The death mechanism affects cap-coverage (stale patterns compete and lower it). Fixing it is prerequisite to verifying the Δcap-coverage headline.

### Priority 3 — Diagnose the W=4 candidate variance split and the rt0p50/rt0p60 identity (est. ~1–2h)

**What:** (a) For the seed-variance issue: run exp 18 with seed 2 separately and add a per-seed candidate-count and drift-trajectory log. (b) For the rt0p50/rt0p60 identity: verify whether the resolve_threshold parameter is actually being passed through to the replay store filter, or if there is a code path where the store is drained before the filter is checked.  
**Why:** The W=4 candidate variance (179.6 ± 97.7) may explain why seed 2 shows cap_t05 degradation. The rt0p50/rt0p60 identity is a data integrity question — if the parameter has no effect in the current code path, the entire resolve_threshold sweep is uninformative and report 024 must be reframed.

### Priority 4 — Aggregate D1 (meta-stable rate) and run the diagnostic-actuator design session (est. ~4h total)

**What:** (a) Write a 30-minute aggregation pass: pull `metastable_rate` from existing Phase 4 JSON checkpoints and compute 5-seed CIs. (b) Hold the diagnostic-actuator dynamic-form session specified in the 2026-05-09 synthesis note — this is the architectural design work that maps diagnostics to slow-timescale response dynamics.  
**Why:** D1 (meta-stable rate) is listed in the checklist as ❌ (data in JSON, never aggregated). The diagnostic-actuator session is STATUS.md blocker #4 and was flagged as "next major architectural threshold" — it is the design work needed for the architecture to self-regulate without a supervisor.

### Priority 5 — Investigate Phase 5 Layer2 admission failure and the entropy rise at W=4 (est. ~2h)

**What:** Run `21_ham_with_phase4.py` (`ham_with_layer2` condition) at β=30 with 5 seeds and compute multi-seed Wilson CIs on Δtop1 and Δtopk vs. `ham_baseline`. Separately, investigate the W=4 entropy rise at step 2000 — check whether it correlates with the candidate count variance (179.6 ± 97.7) or with specific seeds.  
**Why:** The Phase 5 layer2 mechanism is working (108–113 items fill at β=30; +0.003 top1 observed at seed 17), but there is no multi-seed synthesis report with CIs. A 5-seed run would either confirm a real improvement or explain the positive single-seed result. The W=4 entropy rise is checklist item E3 (currently ❌) and is one of the four items preventing Phase 4 graduation.

---

## Appendix A: Data Integrity Summary

| Issue | Severity | Location |
| --- | --- | --- |
| rt0p50 and rt0p60 logs near-identical (differ by 1 candidate at step 1500); resolve_threshold has no effect in current regime | MEDIUM | `reports/phase4_rt0p50.log`, `phase4_rt0p60.log` |
| Phase 3 comparison: codebook pairs produce identical accuracy (random=random, hebbian_b=hebbian_c, recon=error_driven) | HIGH | `reports/phase3_comparison{,_6seed}.json` |
| `reports/phase4_validation/validation_per_seed.csv` has a confusing path — it is a per-seed multi-scale summary from an older run family, not a Phase 4 unified experiment output | LOW | `reports/phase4_validation/` |
| `phase4_tuned_dw1000_re100` baseline not decaying at drift=0.15 | MEDIUM | `reports/phase4_tuned_dw1000_re100.log` |
| Death mechanism never fires (wrong parameter values in experiments) | MEDIUM | All exp 18/19 runs |
| Phase 5 Layer2 `ham_with_layer2` condition works (108–113 items, β=30) but +0.003 top1 improvement not yet multi-seed CI-reported | LOW | `reports/phase5_layer2_validation/` |
| `22b_summarize_diagnostic.py` never run | LOW | `reports/phase34_diagnostic_smoke/` |
| Exp 22 full run not found (only smoke) | LOW | `reports/` |

## Appendix B: Metric Reference

| Metric name | Definition | Where computed |
| --- | --- | --- |
| Recall@K (R@K) | Fraction of test queries where true target in top-K decoded results | `phase2/metrics.py` |
| cap_coverage | Fraction of retrievals where settled top_score ≥ θ (settled-state version) | `phase2/metrics.py` |
| cap_t_05 / cap_t_03 | Fraction where true target appears in top-K with cosine ≥ θ (target-aware, **different from above**) | Inline in exp 18/19 |
| settled synergy | Mean cos(settled_state, bundle_of_context_tokens) − cos(raw_cue, bundle) | `diagnostics/synergy.py` |
| meta_stable_rate | Fraction of retrievals where top_score < 0.95 (didn't commit) | `phase2/metrics.py` |
| gate_signal | engagement × (1 − resolution) = H(softmax weights) × (1 − final_top_score) | `phase4/trajectory.py` |
| effective_strength | Weighted sum of u_k chain: Σ 2^(1-k) × u_k | `phase4/consolidation.py` |
| ΔR@K | R@K(phase4) − R@K(baseline) at same checkpoint | Computed in report synthesis |
| Δcap_t_05 | cap_t_05(phase4) − cap_t_05(baseline) | Computed in report synthesis |
