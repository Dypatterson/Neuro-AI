# Empirical & Code Inventory: Neuro-AI FHRR + Hopfield + Codebook + Replay/Consolidation

**Date:** 2026-05-24  
**Scope:** Reports 001–064 + src/energy_memory/ directory tree  
**Purpose:** Map tested mechanisms, code implementations, and gaps between empirical validation and production code.

---

## OUTPUT A: EMPIRICAL MECHANISMS (Reports 001–064)

### Phase 0 Substrate & Coupled Recall (Reports 001–010)

| # | Name | Mechanism | Result |
|---|------|-----------|--------|
| 001 | temporal_recall | Temporal association memory recovery vs. content nearest-neighbor on ordered stream | POSITIVE: 1.0 vs. 0.388 |
| 002 | regime_sweep | Beta × window sweep: content/temporal separation regime boundary | POSITIVE: robust_temporal at beta 4+ |
| 003 | joint_energy_disambiguation | Joint content+temporal energy score vs. post-content temporal read | POSITIVE: 1.0 vs. 0.0 on family blends |
| 004 | coupled_settling | Iterative coupled loop trajectory under strong/weak temporal cues | POSITIVE: converges, entropy trajectory captures engagement |
| 005 | cue_degradation_sweep | Temporal cue degradation (partial/noisy/wrong) with exact content | POSITIVE: robustness, high-beta flip behavior |
| 006 | dual_degradation_sweep | Content + temporal cue joint degradation | NULL/REGRESSION: wrong temporal + high beta flips to wrong anchor |
| 007 | mps_migration | Torch/MPS backend availability & reference backend preservation | POSITIVE: MPS available, kept both backends |
| 008 | lsr_kernel_sweep | LSR kernel (Epanechnikov) vs. softmax; β-phase diagram | NULL: LSR β-invariant under normalized updates |
| 009 | permutation_slots_ablation | Permutation-indexed offsets vs. unordered context bags | POSITIVE: 1.0 vs. 0.262 on offset recovery |
| 010 | permutation_slots_coupled_recall | Permutation slots in coupled recall under mirrored-neighborhood isolation | POSITIVE: 0.88 vs. 0.20 top-1 with directed cue |

**Phase 0 Summary:** Coupled content+temporal energy validated as core primitive. Permutation offsets = critical upgrade over bags. LSR ruled out under current update rule.

---

### Phase 1–2 Infrastructure & Phase 3 Objectives (Reports 011–020)

| # | Name | Mechanism | Result |
|---|------|-----------|--------|
| 011 | synergy_probe_phase4 | GIB synergy as Phase 5 headline metric on real Phase 4 settled bindings | POSITIVE: synergy elevation validated |
| 012 | replay_store_upgrades_ablation | Replay-store tag_count × IoR ablation (measurement-only) | POSITIVE: tag_count increases candidate discovered count |
| 013 | replay_store_upgrades_candidates_on | Replay-store upgrade with active candidate handler | POSITIVE: discovery channel fires ~80 candidates/run |
| 014 | mps_benchmark_d4096 | MPS vs. CPU speed at D=4096 (Phase 4/5 regime) | POSITIVE: 1.98× bundle, 2.31× similarity_matrix |
| 015 | phase0_torch_port | Phase 0 sweeps ported to Torch/MPS @ D=4096 | POSITIVE: all conditions reproduce on hot path |
| 016 | phase2_audit_and_phase3_objective | Phase 2 full-matrix audit; Phase 3 objective formalized | POSITIVE: objective = beat 0.205 random baseline (settled synergy) |
| 017 | phase3_codebook_comparison | Hebbian vs. error-driven vs. reconstruction codebooks (Recall@1) | NULL: all fail; synergy shows signal |
| 018 | phase3_synergy_comparison | Phase 3 codebook comparison via settled synergy (not Recall@1) | POSITIVE: Hebbian wins, first verified Phase 3 result |
| 019 | reconstruction_characterization | Atom collapse pathology across all three Phase 3 learning rules | NEGATIVE: all three collapse to 1-2 atoms |
| 020 | per_scale_atom_geometry | Multi-scale vs. single-scale atom collapse | POSITIVE: collapse isolated to single-scale, multi-scale orthogonal |

**Phase 1–2 Summary:** Phase 2 baseline clear, Phase 3 first winner = Hebbian (settled synergy), atom collapse is real. Multi-scale design validated.

---

### Phase 3–4 Integration & Phase 4 Graduation (Reports 021–040)

| # | Name | Mechanism | Result |
|---|------|-----------|--------|
| 021 | native_w_codebook_eval | Per-scale codebooks under-trained vs. properly trained | POSITIVE: validates Phase 4 multi-scale choice |
| 022 | phase4_tuning_and_ham_validation | Phase 4 death_window/reencode knobs + HAM arithmetic | NULL: knobs inert, HAM ties at β=30 |
| 023 | drift_sweep_and_gate_audit | Phase 4 at drift ∈ {0.15, 0.30, 0.50} + engagement gate telemetry | POSITIVE: +0.031 top-1 at drift=0.30, gate fires |
| 024 | resolve_threshold_sweep | resolve_threshold=0.85 drift-immunity test | POSITIVE: rt=0.85 locks top-1 at pre-drift |
| 025 | rt0p85_5seed_verification | rt=0.85 @ 5 seeds, frozen codebook + synthetic drift | POSITIVE: ΔR@10 +0.016 [0.001, 0.022], high per-seed variance |
| 026 | phase4_verification_design_spec | Phase 4 frozen codebook + synthetic drift, design-spec controls | POSITIVE: ΔR@10 +0.010 [+0.001, +0.022], **first verified Phase 4** |
| 027 | full_repo_audit_synthesis | Comprehensive Phase 0–4 audit; gaps mapped | POSITIVE/META: identifies infrastructure + blocker list |
| 028 | phase34_integration_5seed | Phase 3+4 with Hebbian online drift, 5 seeds | NULL: drift did not actualize; discovery channel healthy |
| 029 | phase34_integration_st03 | Phase 3+4 st03 variant, online Hebbian drift | POSITIVE: ΔR@10 +0.0145, 4/5 seeds positive, scales with drift |
| 030 | phase34_rfix_5seed | Phase 3+4 with discovered-pattern reencoding fix | POSITIVE/MIXED: ΔR@10 +0.0109 (smaller), top-1 regression = Phase-3 |
| 032 | phase34_n10_verification | Phase 3+4 n=10 continuation of 029/030 | (diagnostic continuation) |
| 033 | phase4_death_mechanism_diagnostic | Threshold-based death firing diagnostic | NULL: death mechanism not firing |
| 034 | saighi_ak_seed1_prototype | Saighi & Rozenberg A_k self-inhibition, single seed | SIGNAL: prototype positive, single-seed artifact |
| 035 | saighi_ak_n10_falsification | Saighi A_k @ n=10 | NEGATIVE: falsifies 034; headline collapses |
| 036 | decay_sweep_and_mass_death_finding | A_k decay sweep; A_k orthogonal to signal | REGRESSION: A_k doesn't explain headline variance |
| 037 | seed3_collapse_diagnostic | Seed-3 collapse root cause via A_nz | NEUTRAL/DIAGNOSTIC: A_nz predicts survivor count, death is real problem |
| 038 | phase4_d1_graduation | Phase 4 D1 (meta-stable rate) @ st03 regime | POSITIVE: **Phase 4 graduates on D1 baseline** |
| 039 | phase3_codebook_comparison_integrity | Data integrity of phase3_comparison.json | NEUTRAL/META: byte-identical anomaly found, doesn't invalidate Phase 4 |
| 040 | freq_weighted_alpha_sweep | Benna-Fusi α × retrieval frequency scaling | NULL: production-scale null on Phase 4 headlines |

**Phase 3–4 Summary:** Phase 4 graduates on ΔR@10. Hebbian drift + discovery channel functional. Death mechanism is real problem; A_k doesn't help. Multi-seed variance high; seed 23 idiosyncratic.

---

### Phase 5 Attempts & Closure (Reports 041–064)

| # | Name | Mechanism | Result |
|---|------|-----------|--------|
| 041 | phase5_de_n5_partial | Phase 5 headline ΔE, n=5 | SIGNAL/PARTIAL: n=5 only, not graduation |
| 042 | phase5_branching_collapse_diagnostic | K-branches collapse onto single attractor | NEGATIVE: branching mechanism broken |
| 043 | phase5_substrate_scale_diagnostic | Death collapses substrate dimensionality | NEGATIVE: death ~10× compression |
| 044 | consolidation_geometry_diagnostic | Consolidation + death role | NEGATIVE/DIAGNOSTIC: death is real bottleneck |
| 045 | phase5_ab_pilot_seed17 | A+B 1-seed pilot | NEGATIVE: FAIL at n=1 |
| 046 | phase5_ab_pilot_seed17_step3 | A+B+step3 | NEGATIVE: FAIL at n=1 |
| 047 | phase5_ab_branch_divergence_failure | A+B+step3 K-branch divergence | NEGATIVE: FAIL at n=1 |
| 048 | phase5_a1_pilot_seed17 | A1 (r_ema geometric init) | NEGATIVE: FAIL at n=1 |
| 049 | phase5_a1prime_pilot_seed17 | A1' (max-reduction r_inst) | SIGNAL: survives smoke |
| 050 | phase5_beta_smoke_seed17 | β (path-3) on A1' substrate | NEUTRAL: smoke-level |
| 051 | phase5_tier1_disambiguation | Three fidelity metric variants | NEGATIVE: all FAIL |
| 052 | phase5_pair4_smoke_falsification | Pair #4 smoke gate | NEGATIVE: both operationalizations fail |
| 053 | phase5_headline_n10_directional_subnoise | ΔE Headline n=10 | SIGNAL/NULL: directional but sub-noise |
| 054 | phase5_step3_smoke_seed17 | Step-3 walk-back smoke | NEUTRAL: diagnostic |
| 055 | phase5_headline_beta_sweep_seed17 | Headline β sweep n=30 | SIGNAL/PARTIAL: corrected in 056 |
| 056 | phase5_headline_beta_sweep_K1_seed17 | Headline β sweep K=1 n=200 (corrects 055) | SIGNAL/PARTIAL: K=1 locked |
| 057 | phase5_cross_seed_beta_sweep_K1 | Cross-seed β sweep K=1 n=10×200 | SIGNAL: mean ΔE positive, high variance |
| 058 | phase5_cross_seed_cue_regime_sweep | Cross-seed cue-regime K=1 β=10 | SIGNAL: regime dependence |
| 059 | phase5_log_prior_spike_local_smoke | Log-prior spike mechanism | SIGNAL: fires |
| 060 | phase5_log_prior_n10_colab_confirmation | Log-prior n=10 confirmation | SIGNAL: reproducible, controls incomplete |
| 061 | phase5_log_prior_gain1_required_controls | Log-prior full ablation | NULL/REGRESSION: dominant but insufficient, no graduation |
| 062 | phase5_spikes_d1_d3_local_smoke | D1+D3 spikes smoke | NULL: neither helps → Tier-2 training intervention required |
| 063 | phase5_spike_e1_centered_log_prior | E1 centered log-prior (path C closure) | NULL: monotone worsening; all retrieval-only routes exhausted |
| 064 | phase5_m1_retrieval_smoke_cross_seed_null | M1 retrieval cross-seed | NULL: cross-seed null, path graveyard |

**Phase 5 Summary:** Branching mechanism fails. Death collapses structure. All retrieval-only routes exhausted. Log-prior signal exists but insufficient. Requires Tier-2 training-time intervention.

---

## OUTPUT B: CODE SURFACE (src/energy_memory/)

### Directory Tree & Module Assignments

```
src/energy_memory/
├── substrate/
│   ├── fhrr.py                      → Reference FHRR backend (Python)
│   └── torch_fhrr.py                → MPS/Torch FHRR backend (hot path)
├── memory/
│   ├── hopfield.py                  → Classical Hopfield (Python ref)
│   ├── torch_hopfield.py            → TorchHopfieldMemory (main retrieval)
│   ├── temporal.py                  → TemporalAssociationMemory (Python ref)
│   ├── torch_temporal.py            → TorchTemporalAssociationMemory (coupled recall)
│   ├── torch_temporal_slots.py       → PermutationSlotTemporalMemory (offsets)
│   ├── _math.py                     → Math utilities (ref backend)
│   └── _torch_math.py               → Torch math utilities
├── phase2/
│   ├── codebook_learner.py          → Static codebook learning
│   ├── error_driven_learner.py       → Error-driven rule
│   ├── reconstruction_learner.py     → Reconstruction objective
│   ├── corpus.py                    → Vocabulary + ngram baseline
│   ├── encoding.py                  → Encoding helpers
│   ├── metrics.py                   → Retrieval evaluation
│   └── persistence.py               → File I/O
├── phase34/
│   ├── online_codebook.py           → OnlineCodebookUpdater (base)
│   ├── stable_online_codebook.py    → StableOnlineCodebookUpdater (v1, v2)
│   ├── hebbian_online.py            → HebbianOnlineCodebookUpdater
│   └── reencoding.py                → Reencoding pipeline
├── phase4/
│   ├── replay_loop.py               → ReplayStore, UnifiedReplayMemory
│   ├── consolidation.py             → ConsolidationConfig, ConsolidationState
│   ├── trajectory.py                → TrajectoryTrace, TracedHopfieldMemory
│   └── snapshot.py                  → State snapshots
├── phase5/
│   ├── ham_aggregator.py            → HAMAggregator (multi-scale retrieval)
│   ├── ham_with_layer2.py           → HAMWithLayer2 (role-filler bindings)
│   ├── m1_role_energy.py            → M1 role-binding energy computation
│   └── role_fidelity.py             → Fidelity metrics
├── diagnostics/
│   ├── synergy.py                   → GIB synergy estimator
│   └── metrics.py                   → Diagnostic metrics
└── experiments/
    ├── synthetic_worlds.py          → Test data generation
    └── __init__.py
```

### Phase-by-Phase Implementation Status

#### **Substrate (Phase 0):**
- **FHRR:** Reference (Python) + Torch/MPS backends ✓
  - Operations: random, perturb, bind/unbind, bundle, similarity, top_k, cleanup, permute
  - Permutation operator added (report 009) ✓
- **TorchFHRR:** Full feature parity ✓
- **Status:** Production-ready for D=4096

#### **Memory Primitives (Phase 0–2):**
- **Hopfield (content):** Reference + Torch backends ✓
  - Softmax retrieval kernel ✓
  - LSR kernel (tested, β-invariant under convex normalization, kept as option)
- **Temporal Association (coupled recall):** Reference + Torch backends ✓
  - Bag encoding (default) ✓
  - Permutation-slot encoding (added, report 009–010) ✓
  - Coupled settling loop ✓
- **Status:** Both backends feature-complete

#### **Phase 2 (Static Codebook Learning):**
- CodebookLearner ✓
- ErrorDrivenLearner ✓
- ReconstructionLearner ✓
- Corpus/Vocabulary ✓
- Metrics (Recall@K, Synergy) ✓
- **Status:** Full matrix on MPS (report 016) ✓

#### **Phase 3 (Online Codebook Learning):**
- OnlineCodebookUpdater (base class) ✓
- StableOnlineCodebookUpdater (v1, v2) ✓
- HebbianOnlineCodebookUpdater (report 018 winner) ✓
- Reencoding pipeline ✓
- **Status:** Hebbian validated on settled synergy (report 018), atom-collapse pathology real (report 019)

#### **Phase 4 (Replay + Consolidation):**
- ReplayStore ✓
  - tag_count × IoR upgrades (reports 012–013) ✓
  - Candidate discovery + storage ✓
  - Death mechanism (threshold-based, not firing per report 033)
- UnifiedReplayMemory ✓
- ConsolidationConfig / ConsolidationState ✓
- TracedHopfieldMemory (trajectory tracking) ✓
- TrajectorySnapshot ✓
- **Status:** D1 baseline graduated (report 038), ΔR@10 verified. Death mechanism broken.

#### **Phase 5 (Schema + Layer-2 Binding):**
- HAMAggregator (multi-scale retrieval) ✓
  - Summed vs. max scoring (report 022)
  - K-branch divergence mechanism (tested, fails reports 042, 045–047)
- HAMWithLayer2 (role-filler layer-2) — code present but untested in Phase 5 context
- M1RoleEnergy (m1 variant, tested smoke-only, report 064)
- RoleFidelity (multiple fidelity metrics, reports 051, 061)
- **Status:** Code infrastructure built; all retrieval-only routes fail. Requires Tier-2 training-time intervention (per report 062–064).

#### **Diagnostics:**
- Synergy estimator ✓ (GIB mutual information, reports 011, 018)
- Metrics (per-window, aggregate) ✓
- **Status:** Synergy = Phase 5 candidate headline (report 011)

---

### Modules & Classes Summary

| Module | Class | Purpose | Tested | Latest Report |
|--------|-------|---------|--------|----------------|
| substrate.fhrr | FHRR | Ref backend | ✓ | 007 |
| substrate.torch_fhrr | TorchFHRR | Hot path | ✓ | 014–015 |
| memory.hopfield | HopfieldMemory | Content retrieval | ✓ | 026 |
| memory.torch_hopfield | TorchHopfieldMemory | Torch retrieval | ✓ | 026 |
| memory.temporal | TemporalAssociationMemory | Ref coupled | ✓ | 010 |
| memory.torch_temporal | TorchTemporalAssociationMemory | Torch coupled | ✓ | 010 |
| memory.torch_temporal_slots | PermutationSlotTemporalMemory | Offset encoding | ✓ | 010 |
| phase2.codebook_learner | CodebookLearner | Static learning | ✓ | 016 |
| phase2.error_driven_learner | ErrorDrivenLearner | ED rule | ✓ | 017 |
| phase2.reconstruction_learner | ReconstructionLearner | Reconstruction | ✓ | 018 |
| phase34.online_codebook | OnlineCodebookUpdater | Base online | ✓ | 021 |
| phase34.hebbian_online | HebbianOnlineCodebookUpdater | Hebbian | ✓ | 018, 028–030 |
| phase34.stable_online_codebook | StableOnlineCodebookUpdater | Stable variants | ✓ | 021 |
| phase4.replay_loop | UnifiedReplayMemory | Replay + store | ✓ | 026, 038 |
| phase4.replay_loop | ReplayStore | Store mechanics | ✓ | 012–013 |
| phase4.consolidation | ConsolidationConfig | Config | ✓ | 038 |
| phase4.trajectory | TracedHopfieldMemory | Traced retrieval | ✓ | 026 |
| phase5.ham_aggregator | HAMAggregator | Multi-scale | ✓ | 022, 041–064 |
| phase5.ham_with_layer2 | HAMWithLayer2 | Role-filler | CODE ONLY | untested |
| phase5.m1_role_energy | M1 variants | Role energy | ✓ | 064 (null) |
| phase5.role_fidelity | RoleFidelity | Fidelity metrics | ✓ | 051, 061 |
| diagnostics.synergy | SynergyEstimator | GIB synergy | ✓ | 011, 018 |

---

## OUTPUT C: GAPS (Code vs. Empirical)

### A. Code Present, Minimally/Not Tested

1. **phase5.ham_with_layer2** (HAMWithLayer2)
   - Code: Complete Layer2Attractor, Layer2State, HAML2Result classes
   - Tests: None in main Phase 5 reports (041–064)
   - Gap: Layer-2 role-filler binding is architecturally required but never integrated into Phase 5 gradient or training loop
   - Impact: Phase 5 design assumes two-layer binding; code exists but flow untested
   - Status: Code-only artifact

2. **phase5.role_fidelity** (RoleFidelity)
   - Code: Multiple fidelity-metric variants
   - Tests: Reports 051, 061 (all FAIL smoke gates)
   - Gap: Fidelity metrics implemented but no working variant; metric choice is open question
   - Impact: Phase 5 headline metric undefined
   - Status: Experimental dead-end

3. **phase4.consolidation** (ConsolidationConfig/State)
   - Code: Config + state classes
   - Tests: Report 038 (D1 baseline only); no investigation of consolidation dynamics proper
   - Gap: Consolidation architecture specified but not characterized
   - Impact: Unknown how consolidation state evolves under Hebbian drift
   - Status: Minimal coverage

4. **memory.hopfield** (LSR kernel option)
   - Code: LSR kernel parameter + tests (test_torch_hopfield_lsr.py)
   - Tests: Report 008 (β-invariance finding; ruled out under normalized updates)
   - Gap: Kernel kept as flag; not integrated into hot path
   - Impact: Dead-end option; no Phase 4/5 experiment uses it
   - Status: Planted flag, not active

5. **phase34.reencoding** (reencoding pipeline)
   - Code: Present
   - Tests: Report 030 (reencoding fix tested; effect: C−B advantage shrinks)
   - Gap: Reencoding is tunable pipeline; only one variant tested
   - Impact: Could explore other reencoding schedules, not done
   - Status: Minimal coverage

---

### B. Empirically Validated, Code Path Unclear / Missing

1. **Permutation-slot temporal encoding (reports 009–010)**
   - Code: PermutationSlotTemporalMemory ✓
   - Validation: 4.4× over bags in coupled recall
   - Status: **Code path complete** (added in report 009)
   - Note: Default is still `encoding="bag"` in memory/__init__; promoted to default recommendation but not enforced

2. **Synergy as Phase 5 headline (report 011)**
   - Code: synergy.py ✓
   - Validation: GIB estimator works on Phase 4 settled states
   - Status: **Code exists**, recommendation pending Phase 5 decision
   - Impact: Synergy computed at every step but not primary Phase 5 loss/objective

3. **Saighi A_k self-inhibition (reports 034–036)**
   - Code: Likely in phase4/replay_loop.py or consolidation.py (not explicitly named)
   - Tests: Falsified (reports 035–036); A_k doesn't explain headline variance
   - Status: **Implemented but ruled out** by experiment
   - Impact: Dead mechanism; should be removed or marked deprecated

4. **Death mechanism (threshold-based, reports 033, 037)**
   - Code: In ReplayStore (phase4/replay_loop.py)
   - Tests: Report 033 (not firing); Report 037 (A_nz predicts survivor, mechanism is real problem)
   - Status: **Code present, mechanism broken**, needs redesign
   - Impact: Phase 4 blocker on per-seed variance reduction

5. **Frequency-weighted Benna-Fusi α (report 040)**
   - Code: Likely in phase4 or phase34 codebook modules
   - Tests: Null on Phase 4 headlines
   - Status: **Tested and nullified**, implementation may or may not be in codebase
   - Impact: Ruled out as bridge to Phase 5

---

### C. Experimentally Blocked, Code May Exist But Not Viable

1. **HAMWithLayer2 integration (phase5.ham_with_layer2)**
   - Blocker: K-branches collapse (report 042); no working fidelity metric (reports 051, 061)
   - Code: Present
   - Status: Architectural dead-end without solving branch divergence
   - Impact: All Phase 5 retrieval-only routes fail (reports 062–064)

2. **K-branch schema store mechanisms (reports 042, 045–064)**
   - Code: HAMAggregator K-branch logic + Layer2Attractor
   - Tests: Universal failure across reports 042–064
   - Status: Core mechanism does not work
   - Impact: Phase 5 main design path fails; requires Tier-2 retraining

3. **Log-prior spike mechanism (reports 059–061)**
   - Code: Likely in phase5/m1_role_energy.py or similar
   - Tests: Signal exists (reports 059–060); full ablation shows insufficient (report 061)
   - Status: **Partial success**: mechanism fires but doesn't enable graduation
   - Impact: Helps but not enough alone; requires training intervention

---

### D. Summary Table: Code vs. Empirical Alignment

| Artifact | Code Present | Tested | Result | Status |
|----------|---------|--------|--------|--------|
| FHRR substrate | ✓ | ✓ | POSITIVE | Production |
| Coupled recall (bag) | ✓ | ✓ | POSITIVE | Production |
| Coupled recall (permutation) | ✓ | ✓ | POSITIVE | Recommend default |
| Hopfield (softmax) | ✓ | ✓ | POSITIVE | Production |
| Hopfield (LSR kernel) | ✓ | ✓ | NULL | Flag only |
| Phase 2 learners | ✓ | ✓ | POSITIVE | Production |
| Phase 3 Hebbian | ✓ | ✓ | POSITIVE | Validated winner |
| Phase 3 atom collapse | ✓ | ✓ | NEGATIVE | Real pathology |
| Phase 4 replay/store | ✓ | ✓ | POSITIVE | D1 graduated |
| Phase 4 death (threshold) | ✓ | ✓ | NULL | Broken |
| Phase 4 A_k self-inhibition | ✓ | ✓ | NEGATIVE | Remove/deprecate |
| Phase 4 Benna-Fusi α | ✓ | ✓ | NULL | Ruled out |
| Phase 5 HAMWithLayer2 | ✓ | ✗ | UNTESTED | Dead-end |
| Phase 5 role-fidelity metrics | ✓ | ✓ | NEGATIVE | All FAIL |
| Phase 5 log-prior spike | ✓ | ✓ | SIGNAL/INSUFFICIENT | Partial |
| Synergy estimator | ✓ | ✓ | POSITIVE | Candidate headline |

---

## UNCONSIDERED PATHS & RESEARCH QUESTIONS

### What Was Never Tried But Code Exists For:
1. **HAMWithLayer2** with explicit layer-2 training objective (code exists; Phase 5 design assumes it but never integrated)
2. **Consolidation dynamics** under Hebbian drift (config exists; trajectory tracing exists; systematically characterized never)
3. **Reencoding schedule variants** (pipeline exists; only one tested in report 030)
4. **Phase 4 death mechanism redesign** (current threshold broken; alternative inertial designs not explored)

### What Showed Signal But Never Reached Graduation:
1. **Log-prior spike** (reports 059–061): monotone worsening under full ablation; mechanism fires but insufficient alone
2. **K-branch schema storage** (reports 042, 045–064): universal failure; no variant succeeded
3. **Phase 4 cap-coverage second headline** (reports 025–030): high variance, seed-23 outlier, never cleaned up

### Research Decisions Explicitly Deferred:
1. **LSR kernel with unnormalized gradient-flow update** (report 008, decision C): deferral justified; coexistence regime not reproduced under current rule; could try unnormalized variant but lower priority
2. **Phase 5 training-time intervention (Tier-2)** (reports 062–064): only unblocked path remaining; requires new loss objective or curriculum design

---

## ANTI-HOMUNCULUS VIOLATIONS & DECISIONS

### Mechanisms That Violated or Were Close to Violating FEP Discipline:
- **Report 006 (dual degradation):** high temporal beta can flip to wrong anchor → points to temperature as architecture flaw, not controller knob
- **Report 008 (LSR kernel):** kernel parameter set at construction; no controller issue ✓
- **Report 022 (Phase 4 tuning):** death_window/reencode knobs prescribed but untested before report; testing revealed they don't help (not homunculus, just not load-bearing)
- **Phase 5 log-prior spike (report 059):** log-multiplicity boost per pattern; pattern is supplied by caller, not inferred ✓

### All Validated Mechanisms Pass Anti-Homunculus Check:
- Permutation offset is temporal property, not controller decision ✓
- Engagement gate fires on local trajectory geometry (engagement × (1−resolution)) ✓
- Hebbian update rule is deterministic local function of (context, reconstruction error, learning rate) ✓
- Synergy is computed observation, not control signal ✓

---

