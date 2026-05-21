# Project STATUS

**Last updated:** 2026-05-21 (**Strategic reframe after sanity-check session. The Phase 5 work has been chasing K-branch state_divergence — a *drill-down diagnostic* (per [phase-5-unified-design.md:226-252](notes/emergent-codebook/phase-5-unified-design.md)) — for two sessions, treating it as the headline. The actual design-spec headline is `ΔE = E_content-prior − E_role-prior` ([phase-5-unified-design.md:256-281](notes/emergent-codebook/phase-5-unified-design.md)), which has been directionally validated at n=1 (Decision-5 spike: +2.6e-5 with 98% positive at K=1 on a 12-atom post-death substrate; [design doc:501-534](notes/emergent-codebook/phase-5-unified-design.md)) but never run at n=10 multi-seed, and never on the improved A+B+A1' substrate. Four parallel zoom-out agents independently surfaced this finding — see [reports/053_phase5_sanity_check.md] (pending). Process changes: CLAUDE.md updated to require re-skim of active phase's design spec at session start (not just STATUS.md), and any numbered report must cite the design-spec line number of the headline metric it advances (forcing function against future drift). **Pre-commit:** new note [notes/notes/2026-05-21-phase5-headline-magnitude-floor.md](notes/notes/2026-05-21-phase5-headline-magnitude-floor.md) derives a magnitude floor from the substrate's energy noise scale: at D=4096, β=10, N=1064, the FHRR noise-energy scale is 5.5e-3. The graduation criterion becomes: `mean ΔE ≥ 5.5e-3 AND bootstrap 95% CI lower bound > 0`. The Decision-5 spike's +2.6e-5 was actually *below* the noise scale at N=12 (5.8e-5); the design spec's prediction that "larger N_schemas should yield larger magnitude" is now binding. **Next:** Colab notebook to run substrate-construction + headline experiment for n=10 seeds × 4 required controls (random-schema, K=1, no-prior γ=0, no-schema-store). All four conditions per [design spec:285-293](notes/emergent-codebook/phase-5-unified-design.md). The n=10 ΔE result will write report 053 (if pass) or 053 (graduation-unattained with structural finding). **Pair #4 (now closed/falsified):** at smoke gate under BOTH operationalizations ([report 052](reports/052_phase5_pair4_smoke_falsification.md)). Pair #2 ALSO predicted to fail its analogous smoke gate per quantitative estimate (observed codebook drift magnitudes 1e-6 to 4e-5, vs gate 0.05 — 1000× below). The "softmax-derived per-atom variance" mechanism class is foreclosed by the substrate's geometry, but the ΔE headline measures an *energy comparison* (global quantity), not per-atom variance, so it is on a different metric class. Earlier:** 1-seed Colab smoke for Path 3 trajectory c_i: m_max=0.0014 (vs pre-committed gate m_max > 0.05); |Δ meta_stable_w3|=0.0000 (vs gate > 0.01). Both pair4_active and kappa0_control conditions bit-identical, with identical m_i evolution. **Path 3 m_max was actually *smaller* than Path 1's 0.0082** — counter-intuitive but mechanistically explained: at D=4096 with the trained codebook, retrievals are winner-take-all from iteration 1 (FHRR crosstalk noise floor 1/√D ≈ 0.016 makes non-winner softmax weights ≈ 5e-5 from the first iteration; the trajectory has no diffuse phase to extract a "lost out atom" signal from). Path 1 captures the fixed-point noise-floor mass `w_i · (1 - max_w) ≈ 5e-5 · 0.1`; Path 3 captures only the trajectory gap `max_t - final` ≈ 0 (no gap exists). **Same root cause as β-path falsification ([report 050](reports/050_phase5_beta_smoke_seed17.md)) and Tier-1 fidelity falsification ([report 051](reports/051_phase5_tier1_disambiguation.md))**: the substrate has been engineered for clean sharp retrievals, which forecloses ANY mechanism whose signal is softmax-derived per-atom variance. The mechanism class "per-atom softmax-derived signal feeds an EMA into replay priority" is **architecturally incompatible** with the substrate's D=4096 sharp-basin geometry. This is not a calibration miss — it is a structural finding. Per audit constraint #10: **the response is the pair #2 pivot, NOT a Path-4 c_i reformulation.** Pair #2 (drift / replay-pressure) is the only open pair structurally immune to this failure mode because its signal `δ_i = ||p_i(t) - p_i(t - Δt_i)||²` operates at the consolidation-step timescale, not the retrieval timescale — it does not read softmax weights, does not depend on β or settling sharpness, and is set by codebook re-encoding + Hebbian update magnitude over the inter-replay interval. The pair #4 substrate-side scaffolding (m_i EMA on ConsolidationState, ReplayStore primary_atom indexing, κ multiplier, payback on sample, exp 19 CLI knobs) is fully reusable for pair #2 — only the signal source changes. **Substrate-principle refinement (from report 052):** "substrate measurements that depend on softmax-derived per-atom variance collapse on substrates engineered for clean retrieval; mechanisms relying on such signals must verify viability at the smoke-gate layer before graduation attempt." This applies to pair #5 (cap-coverage) by extension; pair #2 is exempt. **Next:** draft pair #2 design note → anti-homunculus audit → implementation (mostly reusable scaffolding) → smoke + n=10. Earlier this session: **Path 3 trajectory c_i landed.** Pair #4 fixed-point c_i operationalization (`w_i · (1 − max_w)`) was structurally near-zero on the A+B+A1' substrate (sharp self-retrieving basins → max_w ≈ 1 → c_i ≈ 0). 1-seed Colab smoke confirmed m_max=0.0082 → priority bias 1.6% → below multinomial sampling variance → bit-identical κ=0 vs κ=2 training trajectories. **Path 3 reformulation** per HEN (Kashyap 2024) §rank-reduction findings: `c_i^(traj) = max_{t < T} w_i^(t) − w_i^(final)` — the "lost-out atom" signal, computed inside retrieve()'s settling loop via a running max (one elementwise op per iteration, no extra CPU sync). [Design addendum](notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md) §ADDENDUM with full anti-homunculus self-check; **anti-homunculus reviewer PASS** with 3 new Path-3-specific implementation constraints (#8 max update inside settling loop; #9 c_i computed once after loop; #10 smoke-gate falsification binding — failure means pivot to pair #2, NOT another c_i reformulation). Local validation: c_i^(traj) max = 0.19 on random retrievals at D=512, vs 0.008 fixed-point — ~24× larger magnitudes. κ=2.0, μ_obs=0.05, μ_rep=0.5 **unchanged** (no retuning). Implementation: TorchRetrievalResult gains `metastability_contribution` field; running_max_weights tracked in both `TorchHopfieldMemory.retrieve` and `TracedHopfieldMemory.retrieve_with_trace`; `ConsolidationState.update_metastability(contribution)` signature changed to consume the pre-computed c_i. 276 → 282 tests, 0 regressions. New tests in `TestTrajectoryMetastabilityContribution` validate: contribution shape/bounds, winner-atom has small c_i, mixture-query produces positive c_i, end-to-end retrieve→update pipeline. **Pre-committed smoke gate (binding):** the 1-seed Colab smoke must show `m_max > 0.05` AND `|Δ meta_stable_w3| > 0.01` before n=10 launches; failure means pivot to pair #2 (drift / replay-pressure), which is immune to this entire class of failure modes by operating at consolidation timescale rather than retrieval timescale. The substrate-wide architectural principle established: **softmax fixed-point measurements collapse on sharp-basin substrates; trajectory reformulations recover the signal.** This transfers to pair #5 (cap-coverage / restructuring) in Phase 6 — its `c_i = cos(retrieve(cue_i), p_i)` will need its own trajectory reformulation. Pair #2 (drift / replay-pressure) is immune by construction. **Next:** user fires the updated Colab notebook → 1-seed smoke → cell 8 verifies gate → if PASS, 9 remaining seeds for graduation verdict. Earlier this session: **Pair #4 implementation landed + 12 new tests pass (274 total).** Commit incoming. Library additions: `TorchRetrievalResult.weights_tensor` (pre-CPU-sync softmax tensor — added to both `memory/torch_hopfield.py` and `phase4/trajectory.py` retrieve paths); `ConsolidationConfig.metastability_obs_rate` (μ_obs); `ConsolidationState.metastability_ema` per-atom tensor + `update_metastability(weights)` method + `metastability_payback(idx, factor)` method + init-at-zero on `add_pattern` + kept in sync through `remove_pattern`; `ReplayConfig.metastability_gain` (κ) and `metastability_replay_decay` (μ_rep); `ReplayStore.__init__` accepts optional consolidation handle + κ + μ_rep; `ReplayStore.add` accepts `primary_atom_idx` keyword (set from each retrieval's top_index, audit constraint #5: pure key lookup); `ReplayStore._priorities()` multiplies in `(1 + κ · m_trace)` — bit-identical to baseline at κ=0; `ReplayStore.sample()` triggers `m_{i*(trace)} ← (1−μ_rep)·m_{i*(trace)}` inside the multinomial block (audit constraint #3); `UnifiedReplayMemory.retrieve_and_observe` calls `consolidation.update_metastability(result.weights_tensor)` after each retrieve (audit constraint #1: same tensor retrieve() already produced, no second pass); experiment 19 CLI gains `--metastability-obs-rate`, `--metastability-gain`, `--metastability-replay-decay`; death-diag JSON output surfaces `metastability_ema_{mean,max,std}` per eval row. **12 new tests in [tests/test_phase5_metastability.py](tests/test_phase5_metastability.py)** including `TestKappaZeroPreservesPriority::test_kappa_zero_priority_bit_identical_to_baseline` (the critical falsification-control non-regression), `test_new_atom_enters_with_zero_metastability` (audit constraint #7), EMA update correctness, pay-down on sampling, sharp-vs-diffuse retrieval c_i behavior, end-to-end UnifiedReplayMemory wiring. **Pre-262 → 274 tests, 0 regressions.** Colab notebook at [scripts/colab_phase5_pair4_n10.ipynb](scripts/colab_phase5_pair4_n10.ipynb): 10 seeds × 2 conditions (κ=0 control + pair4_active at κ=2.0, μ_obs=0.05, μ_rep=0.5; same seeds, identical CLI except κ — load-bearing for the falsification control); n_cues=3000; A+B substrate prerequisites included (`alpha_anti=1.0`, `coverage_lambda=1.0`, `repulsion_step_size=100.0`); ~8–12 min wall-clock for 20 parallel workers on H100; aggregator cell computes Δ meta_stable_rate at W=3 with bootstrap 95% CI + drill-downs for m_i CV non-uniformity, d_eff non-regression. **Next:** user fires the Colab notebook → graduation verdict against pre-committed criteria (1) Δ ≤ −0.1 CI-disjoint, (2) D1 non-regression Δms_w3 ≤ −0.5, (3) m_i CV > 0.1 within 500 steps, (4) d_eff ≥ 25 at step 1800. Earlier this same date: **Pair #4 design note drafted + anti-homunculus reviewer PASS.** [notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md](notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md): per-atom metastability EMA `m_i ← (1−μ_obs)·m_i + μ_obs·c_i` where `c_i = w_i · (1 − max_j w_j)` is computed from `retrieve()`'s existing softmax weights vector; pay-down `m_i ← (1−μ_rep)·m_i` on replay sampling; priority composition gains `(1 + κ · m_trace)` factor — bit-identical to baseline at κ=0 (load-bearing for the falsification control). Reviewer audit verdict PASS with 7 binding implementation constraints (no metastability re-evaluation pass / no `if m_trace > X` branch / pay-down inside `sample()` after multinomial / κ,μ never adapted from observed `meta_stable_rate` / `i*(trace)` is pure key lookup / κ=0 same code path / m_i initialised at zero on `add_pattern`). Pre-committed falsification criteria: D1 non-regression Δms_w3 ≤ −0.5 CI-disjoint n=10; Phase 5 headline `Δ meta_stable_rate at W=3` CI-disjoint from zero (Δ ≤ −0.1, upper bound < 0) vs κ=0 control on same seeds; m_i distribution non-uniform (std/mean > 0.1) within 500 steps; d_eff ≥ 25 at step 1800. **Next:** implementation (~½ day) — surface `weights_tensor` on `TorchRetrievalResult`, add `metastability_ema` to `ConsolidationState`, plumb `(1 + κ · m_trace)` into `ReplayStore._priorities()`, ~4 new tests including `test_kappa_zero_preserves_priority`. Then Colab n=10 retrain for graduation attempt. **Brainstorm + Tier-1 disambiguation complete (earlier this session). Phase 5 pivots to pair #4 (metastability ~ replay-prioritization).** Full /brainstorm at [brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md](brainstorm-workspace/2026-05-20-phase5-graduation/brainstorm-phase5-graduation.md) — 5 parallel context agents + 5 parallel research agents, 12 ranked ideas across 3 tiers. Tier 1 (cheap rescue paths) at [report 051](reports/051_phase5_tier1_disambiguation.md): **all three alternative fidelity metrics FAIL the pre-committed `std > 0.02` rescue threshold** on the existing A1' seed-17 substrate. `compute_role_fidelity` (existing): std 0.0000. `compute_role_fidelity_decode_margin` (Ganesan-style, vocabulary = substrate's own patterns): std 0.0012 (17× below threshold) — has qualitative discrimination (original atoms 2× higher margin than discovery; top-10 all originals; discovery atoms tied at FP precision) but absolute magnitudes are ~100× below what's achievable for sharp role-binding. `compute_role_fidelity_settled` (post-Hopfield-settling): std 0.0000 — substrate has *self-retrieving* basins (a positive A1' side-effect) so settling is a no-op for f_i. Library additions at [src/energy_memory/phase5/role_fidelity.py](src/energy_memory/phase5/role_fidelity.py); 6 new tests; 262 tests pass. **The metric category "intrinsic per-atom fidelity on existing A1' substrate" is exhausted.** A1' did its job correctly (decode-margin's qualitative discrimination confirms substrate is what was designed); the K-branch state_divergence headline under role-prior is just the wrong operationalization for this substrate at D=4096. **Phase 5 pivots to pair #4** (research D analysis at [brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md](brainstorm-workspace/2026-05-20-phase5-graduation/research/D-unexplored-actuator-pairs.md)): the substrate already computes `meta_stable_rate` every retrieve() call and throws it away; actuator is a Benna-Fusi/Saighi-shape per-atom EMA `m_i` multiplied into the existing `ReplayStore` priority; new headline metric `Δ meta_stable_rate at W=3` is the same substrate-pure metric class that just graduated Phase 4 on D1 ([report 038](reports/038_phase4_d1_graduation.md): Δms_w3 = -0.7920, CI [-0.94, -0.65], 10/10). **Next:** draft pair #4 design note → anti-homunculus audit → ~½ day implementation → Colab n=10 retrain for graduation. Earlier this same date: **β (path-3) implementation + smoke test on A1' substrate.** Library at [src/energy_memory/phase5/role_fidelity.py](src/energy_memory/phase5/role_fidelity.py) with `compute_role_fidelity` + `fidelity_weighted_prior`; 10 new tests pass; integrated into [experiments/40_phase5_branching.py](experiments/40_phase5_branching.py) as `prior_type='fidelity_weighted'` (full-substrate, no top-k pre-filter) + new headline conditions `fid_K1_q0` and `fid_K1_q1`. 256 tests pass (commit [87b03ae](https://github.com/Dypatterson/Neuro-AI/commit/87b03ae)). **Smoke test against A1' seed-17 substrate ([report 050](reports/050_phase5_beta_smoke_seed17.md)): β does not fix headline at n=1.** Empirical finding: **all 1064 atoms have IDENTICAL role-fidelity f_i = 0.9858 (std = 0.0000)** — at D=4096, the FHRR unbind crosstalk noise dominates and `f_i = mean(1 - |G_jk|)` is structurally near `1 - 1/√D ≈ 0.984` regardless of pattern content. Consequence: `f_i^q` is a uniform constant across all schemas, so the β prior at q=1 equals the β prior at q=0 up to a uniform scaling, and ΔE(q=0 → q=1) = 0 by construction. β at q=0 is also slightly worse than `content_K1` (continuous weighted sum dilutes the sharpest cue-alignment vs categorical top-1). **The four-step debugging journey (A+B → A1 → A1' → β) has correctly closed each layer's failure**, but the binding limit has reached substrate-encoding-level structure: the substrate doesn't have a measurable per-atom role-fidelity signal at D=4096. **Recommended next decision (user's call):** (1) stop chasing K-branch state_divergence and re-scope Phase 5's headline metric; (2) reformulate f_i (β': codebook-correlation instead of pairwise-distance); (3) investigate lower-dimension substrates (research direction); (4) defer to Colab n=5 to confirm the uniform-f_i finding generalizes. Earlier this same date: **A1' (max-reduction r_inst) implementation + 1-seed pilot.** Design note at [2026-05-20-r-inst-measure-dynamic-form.md](notes/notes/2026-05-20-r-inst-measure-dynamic-form.md), anti-homunculus reviewer PASS. One-line patch at commit [9a66879](https://github.com/Dypatterson/Neuro-AI/commit/9a66879): `_coverage_redundancy_instantaneous` reduction changed from `sqrt(mean_{j≠i} |G_ij|²)` to `max_{j≠i} |G_ij|`. Closes the "wrong reduction operator" failure mode (failure mode 3, dual of failure modes 1 and 2 from the 2026-05-09 framing). 246 tests pass (245 prior + 1 new sparse-duplicate regression test). Pilot at [reports/phase5_a1prime_pilot_seed17/](reports/phase5_a1prime_pilot_seed17/); analysis at [report 049](reports/049_phase5_a1prime_pilot_seed17.md). **Verdict at n=1: criteria #1 and #2 STILL FAIL but with dramatic mechanism-level progress.** Discovery atoms' max effective_strength fell from 12.1 (no A1) → 9.6 (A1) → **0.029 (A1')**, a 99.8% reduction. The top-1 atom is now an *original Phase-3 atom* (idx 713) instead of a discovery atom. K-branch state_divergence moved from 0 to 0.000897 (still below band [0.024, 0.044], but non-zero for the first time). d_eff preserved at 35.23 (criterion #3 PASS). **The failure mode has migrated** from "substrate construction" (A+B+step3) → "measurement operationalization" (A1) → "selector-layer ties at noise floor" (A1'): the top-K-by-strength selector picks from 7 FP-identical discovery atoms tied at strength 0.029 (each entered with novelty_strength=1.0 in the last few hundred cues; A1' throttled their accumulation; their bumped-then-decayed u-chain leaves them tied at the same effective_strength). A1+A1' has correctly solved the substrate construction + measurement levels. **Next recommended:** β (path-3) implementation — continuous role-fidelity-weighted prior displaces the categorical top-K selector that is now the binding failure. β was already designed + anti-homunculus PASS in [the cue-regime note](notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md). Earlier this same date: **A1 implementation landed + 1-seed pilot run.** Library implementation at commit [f037e48](https://github.com/Dypatterson/Neuro-AI/commit/f037e48): `add_pattern(r_ema_init=...)` kwarg + `_compute_r_inst_for_init()` helper in `replay_loop.py`; 4 new tests in `test_phase5_ab_death_dynamic.TestA1DiscoveryChannelInit`; 245 tests pass (241 prior + 4 new, 0 regressions). Pilot run at [reports/phase5_a1_pilot_seed17/](reports/phase5_a1_pilot_seed17/); analysis at [report 048](reports/048_phase5_a1_pilot_seed17.md). **Verdict at n=1: criterion #1 FAIL** (top-8 pairwise sim = 1.0000, band [0.10, 0.60]); **criterion #2 FAIL** (state_divergence = 0, target [0.024, 0.044]); **criterion #3 PASS** (d_eff = 35.19, ≥ 25 bar); criteria #5–#6 PASS by construction. Falsification SIGNAL at n=1 (not the formal n=5 verdict). **A1 *partially* worked** (discovery atoms' max effective_strength reduced 12.11→9.65; mean 0.51→0.34; top-5 reduced by 50-70%) but the first-added discovery atom (idx 1044) still dominates at strength 9.65 — 100× the top original atom (0.0915). **Root cause identified:** the implementation's `_coverage_redundancy_instantaneous` proxy (Gram-row RMS) under-measures sparse duplicates — for an atom that's a perfect duplicate of one other atom but orthogonal to N-2 others, the mean-over-others RMS gives r_inst ≈ sqrt(1/(N-1)) ≈ 0.03, not 1.0 as projection-magnitude would. A1 draws its init from this proxy, so the first-added duplicate sees a near-zero init and accumulates strength essentially unthrottled. **A1 is correct in shape; the measurement is too weak.** Three sketched redesign directions (each needs design note + anti-homunculus audit): A1' (replace RMS with max-over-others — smallest patch), A1'' (streaming low-rank SVD per the original design note), A1''' (path-3 β / role-fidelity-weighted prior as the alternative axis). **No implementation commitment yet — the next "separate decision" gate is open.** Earlier this same date: **A1 design note drafted + anti-homunculus reviewer PASS.** [notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md](notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md): substrate-derived `r_ema` initial condition for new discovery-channel atoms — `r_ema[new] ← _coverage_redundancy_instantaneous(P ∪ {new})` at add-time, replacing the implicit `r_ema = 0` (an implementer-set constant that encodes the categorical claim "new = novel"). Closes report 047's fix-selection decision: A1 is the cleanest dynamic-form move; W1 (discovery-channel gate), W2 (diversified schema-store selection / MMR / DPP), W3 (scheduled strength rebalancing) explicitly rejected as wrong-shape. The anti-homunculus reviewer confirmed: A1 has no schedule (the add event is itself a dynamic of the discovery channel, not a wall-clock trigger), no threshold, no comparator; both implementation shapes (sentinel + deferred init in step_dynamics, or pattern_matrix passed synchronously to add_pattern) preserve the dynamic-form reading. A1 is the "failure mode 2" dual of the binary-death failure: failure mode 1 was a controller reading a metric and triggering an action (closed by A+B), failure mode 2 is an implementer hard-coding a constant where a measurement belongs (A1 closes this for A's r_ema lifecycle). Six pre-committed falsification criteria: (1) top-8 schema pairwise similarity in band [0.10, 0.60] at n=5; (2) K-branch state_divergence within 30% of pre-death at n=5; (3) d_eff ≥ 25 still holds at step 1800; (4) Phase 4 D1 non-regression Δms_w3 ≤ -0.5 n=10; (5) r_ema init computation done once at add, not scheduled (code-level check); (6) coverage_ema_rate NOT retuned to compensate. **No implementation commitment yet — the user's "separate decision" gate is open.** Earlier this same date: **A+B death-mechanism dynamic-form implementation landed.** Continuous coverage-weighted reinforcement (A) + −α·log(d_eff) repulsion in substrate energy (B) replace the binary death step at the library level. Changes: `src/energy_memory/phase4/consolidation.py` adds per-atom `r_ema` EMA + `coverage_lambda`/`coverage_ema_rate` config + modulated `reinforce()` + `_coverage_redundancy_instantaneous` helper; `src/energy_memory/substrate/torch_fhrr.py` adds `alpha_anti` constructor arg + `d_eff()` + `substrate_energy_anti()` + `repulsion_force()` (autograd-based, complex Wirtinger-correct); `src/energy_memory/phase4/replay_loop.py` adds `repulsion_step_size` config + `_step_substrate_dynamics()` (called every replay cycle, applies the substrate's slow-timescale gradient) + a `garbage_collect()` no-op guard when `coverage_lambda > 0` (closes the audit's operational caveat). 17 new tests in `tests/test_phase5_ab_death_dynamic.py` cover: r_ema EMA dynamics, modulation behavior, off-by-default bit-identical, H_anti monotonicity in d_eff, single-step d_eff increase, full-loop d_eff increase across cycles, garbage_collect guard. **235 tests pass (218 baseline + 17 new, 0 regressions).** **Anti-homunculus reviewer PASS** on the implementation — A's r_ema is a continuous per-atom running estimate (not a scheduled global recompute); B's α is set once at substrate construction (not adapted from observed d_eff); no membership flag for "active atoms" introduced; H_anti is integrated as the substrate's slow-timescale gradient flow; config-guards are architecture spec, not runtime arbitration. Default config (`alpha_anti=0`, `coverage_lambda=0`, `repulsion_step_size=0`) preserves bit-identical baseline. **Next: 1-seed pilot retrain on Colab to verify d_eff preservation and Phase 4 D1 non-regression before n=10.** Earlier in this same date: **Audit-driven pre-A+B fixes landed.** Commit [ec3b95b](https://github.com/Dypatterson/Neuro-AI/commit/ec3b95b): CFL clamp at consolidation.py:_step_dynamics (`_CFL_MAX_ALPHA_EFF=0.5`, derived strict bound — audit's 0.24 was overly conservative and broke existing tests); 2 new regression tests pin the constant and assert finite u at extreme λ; .env added to .gitignore; lockfile (requirements-lock.txt, torch 2.11.0 / numpy 2.4.4); torch>=2.0,<3 + numpy>=1.24,<3 in pyproject.toml; HAMAggregator.retrieve refactored to deferred-sync pattern (~12 fewer MPS stalls per call, bit-identical converged state via torch.where freeze). AUDIT_REPORT.md committed. Earlier in this same date: **Path (a′) prerequisites both closed.** Research-literature review + notes audit reframed path (a) → path (a′): continuous-rate death as slow-timescale dynamic of which d_eff is a fast-timescale snapshot. Prereq 1 ([report 044](reports/044_consolidation_geometry_diagnostic.md)): built [scripts/consolidation_geometry_diagnostic.py](scripts/consolidation_geometry_diagnostic.py), ran on n=5 pre+post-death snapshots. **Substrate d_eff collapses ~10× across all 5 seeds** (pre-death ~40, post-death ~3–6 of 4096 dims); per-atom k-NN d_eff drops modestly (3.7 → 3.0). With K=4 branches and d_eff~5 post-death, branches cannot occupy distinct subspaces — the geometric mechanism for K-branch collapse from report 042. Companion [MESH-style memory-cliff check](reports/phase5_memory_cliff/README.md) (run by subagent): no cliff exists; n_atoms=6 retrieves perfectly. Capacity is fine; substrate effective-dim is the issue. Prereq 2 ([notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md](notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md), closes STATUS blocker #4): three candidate continuous local dynamics enumerated — A (coverage-weighted reinforcement rate), B (-α log(d_eff) repulsion in substrate energy), C (redundancy-coupled inhibition, Saighi-variant). Recommended: A+B combined. Anti-homunculus reviewer audited; caught a controller-in-disguise (step-3 hysteresis-on-ε flag) and 3 other wording slips. Four fixes applied: continuous-running-estimate r_i (A), substrate-energy-everywhere α-fixed (B), n≥10 pre-registered bimodality test (C), continuous E_i-weighted retrieval (A+B step 3), α-not-tuned pre-commitment. Now PASS. Earlier this session: [report 043](reports/043_phase5_substrate_scale_diagnostic.md) (substrate-scale discrimination); [report 042](reports/042_phase5_branching_collapse_diagnostic.md) (K-branch collapse + γ/K_main sweeps); [report 041](reports/041_phase5_de_n5_partial.md) (n=5 partial headline); Phase 4 graduated on D1 ([report 038](reports/038_phase4_d1_graduation.md)). 216 tests pass.)

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

---

## Active phase

**Phase 5 — implementation unblocked** ([phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md)).
HAM × energy-guided structural branching wrapped around the post-death
substrate. Schema source resolved by [report 040](reports/040_freq_weighted_alpha_sweep.md):
the 5–30 surviving atoms per seed at W=2 (post mass-death) are already
filtered by retrieval frequency through the binary death mechanism, so no
freq-α layer is needed. Branching seeds K_main schemas + 1 surprise branch;
combines via energy-weighted bundle + re-settle; atom-splitting diagnostic
on joint criterion (similar low energies AND substantial state divergence).

### Next session entry point (2026-05-20, late session)

**A+B+step3 1-seed pilot PASSES mechanism-validity gate**
([report 046](reports/046_phase5_ab_pilot_seed17_step3.md), supersedes
[report 045](reports/045_phase5_ab_pilot_seed17.md) which preserved as
the without-step-3 baseline):
W=4 step 1800 d_eff = **35.20** (vs 35.23 without step 3; both ≥ 25
target). Step 3 implementation landed at commit
[2a94f5f](https://github.com/Dypatterson/Neuro-AI/commit/2a94f5f) with
anti-homunculus reviewer PASS. The smooth-sigmoidal
`w_i = σ((|E_i| − ε)/τ)` retrieval weighting (ε=0.05, τ=0.02) is
implemented as `softplus((ε − |E_i|)/τ)` score-bias composed additively
with the existing Saighi A_k bias.

Consolidation drill-downs (`r_ema`, `mean_strength`, `dead_ready`) are
bit-identical between the two pilots — step 3 only affects retrieval
weights, not substrate evolution. Phase 4 top1/capt5 identical to FP
precision; Hebbian succ_rate diverges by ~3-5% in late training
(marginal-case retrievals differ).

**Death-as-asymptotic-limit is now fully expressed** at both
consolidation (atoms decay continuously without binary deletion) and
retrieval (low-E_i atoms contribute infinitesimally via the sigmoidal
weighting). The architecture has its first complete
diagnostic-actuator pair in dynamic form
([2026-05-09 prescription](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md)
threshold-crossing #1).

**K-branch state-divergence diagnostic FAILS at n=1**
([report 047](reports/047_phase5_ab_branch_divergence_failure.md)).
Falsification SIGNAL on mechanism-validity criterion #2 (n=5 needed
for the formal verdict). The failure is NOT caused by A+B+step3's
substrate mechanics — d_eff = 35 holds, the broader 1064 atoms are
geometrically diverse. The failure is caused by **a substrate-
construction interaction**: A's (1 − r_ema) modulation initializes
new atoms with r_ema=0, advantaging the Phase-4 discovery-channel
atoms over original Phase-3 atoms in strength accumulation. The
discovery atoms (added via convergent retrieval settles) are near-
duplicates of each other; they end up as the top-8 by effective_strength
with FP-precision-identical pairwise similarity (1.0000 vs baseline's
0.36). The schema store is therefore degenerate; the K-branch
mechanism produces n_branches=1 and state_divergence=0.

Three sketched fixes ([report 047 §"Three possible fixes"](reports/047_phase5_ab_branch_divergence_failure.md)):
- **A1**: substrate-aware r_ema initialization for new atoms (smallest patch).
  **Design note + anti-homunculus PASS + implementation + 1-seed pilot complete**
  ([report 048](reports/048_phase5_a1_pilot_seed17.md)). Partial
  fix; root-cause traced to wrong reduction operator in r_inst.
- **A1'**: max-over-others reduction operator (the report-048 follow-up).
  **Design note + anti-homunculus PASS + implementation + 1-seed pilot complete**
  ([report 049](reports/049_phase5_a1prime_pilot_seed17.md), commit
  [9a66879](https://github.com/Dypatterson/Neuro-AI/commit/9a66879)).
  Discovery-atom strength reduced 99.8%; top-1 is now an original atom
  (idx 713); state_divergence non-zero (0.0009) for the first time;
  criteria #1 and #2 still FAIL at n=1 because the top-K selector
  picks from 7 FP-identical discovery atoms tied at the noise floor
  (0.029). **The failure has migrated from substrate-construction →
  measurement → selector-layer.** Substrate-level mechanisms are now
  correctly solved.
- **β (path-3)**: continuous role-fidelity-weighted prior. Design +
  anti-homunculus PASS in
  [2026-05-20-cue-regime-role-prior-dynamic-form.md](notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md).
  **Implemented + smoke-tested ([report 050](reports/050_phase5_beta_smoke_seed17.md), commit
  [87b03ae](https://github.com/Dypatterson/Neuro-AI/commit/87b03ae)).**
  β doesn't fix headline at n=1 — the substrate's role-fidelity
  measure has zero variance at D=4096 (all 1064 atoms have
  f_i = 0.9858, std = 0.0000) because FHRR unbind crosstalk noise
  dominates the signal. ΔE(q=0 → q=1) = 0 by construction.
  **Binding limit has reached substrate-encoding-level structure**;
  the substrate doesn't have a measurable per-atom role-fidelity
  signal at this dim. β remains available as a library + experiment
  condition; full n=5 Colab run will confirm the uniform-f_i finding
  generalizes (or surface a seed where it doesn't).
- **A1'' (streaming low-rank SVD)**: deprioritized — A1' has already
  driven discovery atoms to the noise floor; a more-precise
  measurement won't break the selector-layer FP-identical-duplicate
  tie. Remains the principled-formulation long-term goal.
- **A2**: schema-store selection by combined strength + diversity.
  **Rejected as wrong-shape (W2 in A1 design note)** — an argmax over
  a population-level strength+diversity objective is a chooser at the
  schema-store boundary; the canonical anti-homunculus failure shape.
- **A3**: path-3 β continuous role-fidelity weighting (already designed in
  [2026-05-20-cue-regime-role-prior-dynamic-form.md](notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md));
  remains the deferred successor design, sequenced **after** A1 either
  passes or fails its K-branch criterion.

**A1 implementation is the next "separate decision" gate per the
design note's discipline.** Two implementation shapes available
(sentinel + deferred init in step_dynamics; or pattern_matrix passed
synchronously to add_pattern); anti-homunculus reviewer confirmed both
preserve the dynamic-form reading. n=10 Colab retrain remains
BLOCKED pending A1 implementation + 1-seed pilot re-run + n=5
K-branch ΔE diagnostic against the six falsification criteria.

The 1-seed pilot script (ready to run):

1. **Pre-committed config** (all four binding values now set in code per
   [reports/phase5_ab_calibration.json](reports/phase5_ab_calibration.json) +
   the design note's load-bearing constraint):
   - `substrate.alpha_anti = 1.0` (natural unit scale: H_anti = -log d_eff)
   - `consolidation.config.coverage_lambda = 1.0` (formal Candidate A)
   - `consolidation.config.coverage_ema_rate = 0.01`
     (EMA halflife ≈ 100 steps; matches the legacy `death_window=100`
     timescale)
   - `replay.config.repulsion_step_size = 100.0`
     (one-shot calibration: smallest step whose median Δd_eff across
     the 5 post-death seed snapshots is ≥ 0.05 — the conservative end
     of the calibration band; per-tick phase change ~1–2°)
   - Existing knobs (alpha_freq_lambda, inhibition_gain) remain at
     their previous values; A+B is additive.

2. **Run with** `bash scripts/run_phase5_ab_pilot_seed17.sh`. Uses
   `experiments/19_phase34_integrated.py` with the 4 A+B flags wired
   in; snapshots at {500, 1500, 1700, 1800} so the
   consolidation-geometry diagnostic produces a directly-comparable
   d_eff trajectory against the pre-existing
   [reports/phase5_snapshots_local/seed17/](reports/phase5_snapshots_local/seed17/)
   snapshots. Pipeline smoke-tested end-to-end on small corpus; no
   crashes.

3. **Verify mechanism-validity criteria** (NOT graduation criteria):
   - d_eff ≥ 25 at step 1800 (pre-committed; failure = falsification).
   - K-branch state_divergence within 30% of pre-death (matches
     report 043's pre-death ratio).
   - 1-seed only; pass before scaling to n=10.

4. **If 1-seed passes:** run n=10 retrain on Colab, then re-attempt
   the Phase 5 A1 headline on the new substrate at n=5 then n=10.
   **If 1-seed fails:** the candidate is wrong-shaped; back to design.

C then B executed earlier this session ([report 041](reports/041_phase5_de_n5_partial.md)).
The K4 headline at n=5 does not graduate — CI includes zero, 3/5 seeds
positive, seed 1 dominates the mean negatively. K1 (B2 control)
outperforms K4 with 5/5 seeds positive and pooled CI excludes zero;
B2's gratuitous-branching prediction fires at this substrate scale.

Path (a′) prerequisites both closed earlier this session. Mechanism cause
identified at the geometric level (d_eff ~10× collapse, [report 044](reports/044_consolidation_geometry_diagnostic.md));
candidate continuous local dynamics enumerated and anti-homunculus
audited ([2026-05-20 diagnostic-actuator note](notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)).

Pre-committed falsification criteria for A+B retrain run (per the
design note §"Pre-committed falsification criteria"):

1. **d_eff preservation:** at the substrate's stable steady state
   (formerly step 1800), substrate d_eff ≥ 25 across 5 seeds.
2. **K-branch state_divergence within 30% of pre-death** across 5
   seeds (matching report 043's pre-death ratio).
3. **Phase 4 D1 non-regression:** Δms_w3 ≤ -0.5, CI-disjoint, n=10
   (preserves [report 038](reports/038_phase4_d1_graduation.md)).
4. **Phase 5 A1 attempted on n=10:** the graduation criterion;
   passes 1-3 are mechanism-validity gates, not graduation gates.
5. **α and λ are fixed once before first retrain** from theoretical
   considerations; not tuned to land d_eff in target range.

Still open / not addressed by this session:
- ~~The bimodal-ΔE-across-seeds issue (report 043)~~ — **path-3 design
  pre-commitment closed 2026-05-20** ([cue-regime / role-prior note]
  (notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md));
  anti-homunculus reviewer PASS; Candidate β (per-schema role-fidelity
  continuous weighting via `prior = Σ_i (cue·s_i)^p · f_i^q · s_i`)
  + γ (cue-regime distribution averaging at evaluation) recommended.
  **Contingent on A+B pilot outcome.** Implementation only after A+B
  pilot passes mechanism-validity AND a separate decision. Pre-committed
  falsification criteria: (1) per-seed `mean(f_i)` spread within 30%
  cross-seed; (2) fidelity-weighted ΔE > 0 CI-disjoint at n_seeds ≥
  10; (3) ΔE(q) monotonic per-seed; (4) cue distribution + q pre-committed
  before observation (H1).
- θ′(β) calibration spike (pre-phase commitment) — would inform the
  tight/spread regime classifier, but A+B doesn't strictly require
  it.

**Pre-commitments still binding:**
- No n=10 on W=4 post-death without mechanism revision.
- No cherry-picking pre-death seeds 17, 23 as the headline set (H4).
- No new combiner / death mechanism without a design note + anti-
  homunculus check first (A+B has both; check passed).
- No using cue-regime sensitivity sweep results to *select* a
  graduation-passing cue regime (H1).
- α not tuned to land d_eff in target range; first retrain miss = falsification.

Snapshots on Drive: `Neuro-AI-Snapshots/phase5_snapshots_seed{N}/`
for N ∈ {17, 11, 23, 1, 2}, each with step ∈ {500, 1500, 1700, 1800} ×
scale ∈ {2, 3, 4}. 60 snapshots total, all captured via Colab notebook
`scripts/colab_phase5_snapshots.ipynb`. Five more seeds needed for n=10.

235 tests passing; 0 skipped. Working tree dirty
(`reports/phase5_snapshots_local/seed{1,2,11,23}/`,
`reports/phase5_headline_n5/`, `reports/041_phase5_de_n5_partial.md`,
`scripts/aggregate_phase5_de.py`, A+B implementation diff in
`src/energy_memory/{phase4/consolidation.py,phase4/replay_loop.py,substrate/torch_fhrr.py}`,
new test file `tests/test_phase5_ab_death_dynamic.py`, walk-back edit
in `notes/emergent-codebook/phase-5-checklist.md`).

**Phase 4** remains graduated on D1 ([report 038](reports/038_phase4_d1_graduation.md));
no regression. Open next-step questions for Phase-4-revision (gradient
death, top1 regression mechanism, cross-corpus generalization) are parked
behind Phase 5 implementation.

---

## Current headline metric (post-pivot per reports 036 + 037 + [2026-05-16 discipline note](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md))

> **Δ meta-stable rate at W=3 (D1) under active codebook drift, multi-seed,
> CI disjoint from zero.**

Rationale: the design-spec headlines (R@K, cap-coverage) are *readouts*
over a substrate that varies wildly per seed under binary mass death.
D1 measures basin-geometry of whatever substrate survives — a substrate-pure
property invariant to which atoms happen to be reinforced. Report 026
already showed Δ=−0.51 at W=3 with per-seed std=0.038 in Phase 4 isolation.
D1 at the integration regime is the in-flight verification.

Drill-downs: ΔR@K (still reported; report 026's +0.010 stands as evidence
of record), Δcap-coverage (variance-bound), Δtop1 (drill-down per discipline
note), D1 at W=2 and W=4 (scale variation), D1 C−B (Phase 4 above
phase3-only).

## Required controls (per [phase-4-unified-design.md:318-323](notes/emergent-codebook/phase-4-unified-design.md))

- No-replay baseline — present in every exp 18 / exp 19 run ✓
- Random-codebook control — verified once at rt=0.85, drift=0.30 (report 026) ✓

## Last verified results

**[Report 026](reports/026_phase4_verification_design_spec.md)** (Phase 4 in
isolation, frozen Phase 3c codebook + synthetic drift, rt=0.85, drift=0.30,
β=10, 5 seeds {17, 11, 23, 1, 2}):
- Δ Recall@10 at step 2000: **+0.010, CI [+0.001, +0.022], 5/5 ≥ 0** ✓
- Δ cap-coverage @ τ=0.5 at step 2000: -0.004, CI [-0.030, +0.022] ✗
- Random-codebook control: 0 candidates, Δ = 0.000 exactly ✓

**[Report 028](reports/028_phase34_integration_5seed.md)** (Phase 3+4
integration via online Hebbian @ success_threshold=0.5, 5 seeds, Colab A100):
- Δ Recall@10 at step 1500: +0.004, CI (t,4df) [-0.012, +0.021] ✗ (includes 0)
- The Hebbian updater fired on ~1% of cues, codebook drift was effectively
  zero (mean 1.6e-6). The test did not exercise drift; superseded by 029.

**[Report 029](reports/029_phase34_integration_st03.md)** (Phase 3+4
integration via online Hebbian @ success_threshold=0.3, 5 seeds, Colab A100):
- Δ Recall@10 at step 1500: **+0.0145, CI (t,4df) [-0.005, +0.034]**, 4/5
  seeds ≥ 0. Pooled-Wilson Δ = +0.0144 (388/1111 vs 372/1111). Monotonically
  growing trajectory. ✓ (mean exceeds report 026's +0.010)
- Δ cap-coverage @ τ=0.5: +0.006, 3/5 ≥ 0 ✗
- **Δtop1: −0.026, 4/5 NEGATIVE** ✗ — initially attributed to stale
  discovered patterns; report 030 falsified that hypothesis (see next).

**[Report 032](reports/032_phase34_n10_verification.md)** (n=10 verification,
same config as 029):
- ΔR@10 at step 1500, n=10: **+0.009, 95% CI [−0.003, +0.021]**, 5+/3(0)/2−.
  CI still includes 0. Report 029's +0.0145 was sample-lucky; new 5 seeds
  produced +0.004 alone.
- Δtop1: −0.018, 7/10 negative. Robust Phase-3-driven regression.
- Pooled-Wilson R@10: A 0.329 [0.310, 0.349], C 0.339 [0.320, 0.359].
  Windows overlap heavily.
- C − B R@10: +0.004 mean — the clean architectural test of Phase 4 over
  phase3-only. Positive sign, CI includes 0.
- Bimodal seed distribution: strong-positives {1, 7}, near-zeros middle 6,
  negatives {3, 23}. Variance is dynamics-trajectory, not substrate.
- **Implication:** Phase 4's R@10 contribution under sanctioned online
  drift is real but small (+0.005 to +0.015 plausible range). Architecture
  is not falsified; effect size is just smaller than report 029 suggested.

**[Report 030](reports/030_phase34_rfix_5seed.md)** (st=0.3 + `--reencode-discovered`
fix, 5 seeds, Colab A100):
- Δ Recall@10: +0.0109 (DOWN from 029's +0.0145; fix erodes the headline)
- Δtop1: −0.0229 (essentially unchanged; regression persists)
- Δcap_t05: −0.0043 (flipped negative; fix made capt5 worse)
- **Verdict: fix was wrong-shaped.** Re-settling discovered queries pulls
  them toward existing attractors, destroying the geometric property the
  discovery channel was meant to capture. Default flipped to off; flag kept
  as opt-in for future selective-refresh variants.
- **Re-frame:** condition B (phase3-only, no Phase 4) *also* shows top1
  collapse under drift (seed 11: B_top1 0.110→0.081). So the top1
  regression is a **Phase 3 / Hebbian-codebook-reshaping** property, not
  a Phase 4 stale-pattern property. The Phase 4 discovery channel adds a
  small *additional* top1 cost on top, but the bulk is upstream.
- Hebbian fired ~17% (slightly higher than 029's 14.5%), drift comparable.

**[D1 5-seed aggregation](reports/d1_metastable_5seed.json)** from existing
report-026 JSON checkpoints (not re-run, drilling into existing data):
- W=3 meta-stable rate: baseline 0.67±0.19 → phase4 0.16±0.21; **Δ = -0.51,
  per-seed std = 0.038** (every seed shows ~-0.5 reduction).
- W=4 meta-stable rate: Δ = -0.57±0.21.
- W=2: ≈0 in both (already committed).
- Read: Phase 4 makes higher-scale retrievals significantly more decisive.
  Unreported in 026; load-bearing drill-down.

---

## Active blockers / must-do before Phase 4 can graduate

| # | Item | Why | Source |
| - | --- | --- | --- |
| 1 | ~~D1 at Phase 3+4 integration regime~~ — **CLOSED, graduates** | Δms_w3 = −0.7920, CI [−0.9376, −0.6463], 10/10 seeds → 0. Also C−B = −0.720, CI-disjoint. R@10 +0.0089 replicates report 026. Phase 4 graduates. | [Report 038](reports/038_phase4_d1_graduation.md) |
| 2 | ~~A_k mechanism~~ — **CLOSED** | Falsified at gain=0.01 across decay ∈ {0.0, 0.02, 0.05, 0.10, 0.20}; orthogonal to the dominant death-driven substrate-shaping force. Mechanism kept implemented but not used in graduation runs. | Reports 026, 030, 033, 034, 035, 036 |
| 2a | ~~Option 1: diagnose seed 3 collapse~~ — **CLOSED** | Same mass-death survivor mechanism as seed 17; outcome direction set by whether the 5–32 surviving atoms cover the test set. A_nz at step 1500 predicts step-2000 survivor count exactly. Corpus-order variance under binary death; not tunable. | Report 037 |
| 6 | ~~Fix stale-discovered-patterns reencoding gap~~ | **CLOSED wrong-shaped** by report 030: the rfix variant erodes ΔR@10 (+0.0145 → +0.0109) and flips Δcapt5 negative (+0.006 → −0.004). Code kept as opt-in (default off); not the right primitive. | Report 030 |
| 6′ | **Top1 regression is Phase 3 not Phase 4 — and A_k AMPLIFIES it** | Report 030 §re-frame: condition B (phase3-only, no Phase 4) also shows top1 collapse under drift. The regression is a Hebbian-codebook-reshaping property, not a Phase 4 architectural gap. Action: characterize whether the top1 regression is an inherent online-Hebbian tradeoff (decide accept) or a fixable issue (decide investigate). **2026-05-15:** Ganesan-style FHRR unitarity audit closed (invariant holds within 2.4e-7); regression is a real mechanism property. **2026-05-16 (report 035):** A_k at gain=0.01, decay=0.0 at n=10 produced Δtop1 = −0.051 (CI strictly negative, 1/10 positive) — *worse* than baseline (~−0.018). So basin-narrowing mechanisms aren't a cure for #6'; they make it worse. Now investigating whether rank-1-vs-neighborhood is a fundamental tradeoff at this corpus size. | Reports 030, 035 + 2026-05-15 notes |
| 3 | ~~Δcap-coverage second headline~~ — **REFRAMED as drill-down** | Per [2026-05-16 discipline note](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md) + report 037: cap-coverage variance is downstream of binary death (corpus-order survival), not tunable without a death-mechanism redesign. Still reported as drill-down; no longer graduation-gating. | Reports 026, 028, 029, 037 |
| 4 | ~~Diagnostic-actuator dynamic-form session~~ — **CLOSED at design + implementation** | Design note held 2026-05-20 with anti-homunculus PASS ([2026-05-20 diagnostic-actuator note](notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)); library implementation landed same session (A in `consolidation.py`, B in `torch_fhrr.py`, wiring in `replay_loop.py`); anti-homunculus reviewer PASS on the diff; 235 tests pass. Empirical validation (1-seed retrain) is the next session, not a blocker — the architecture's first diagnostic-actuator pair in dynamic form exists. | [2026-05-09 paper synthesis](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md), [2026-05-20 design note](notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md) |
| 5 | Seed-23 diagnostic | **Three** independent runs (026, 028, 029) all identify seed 23 as the cap_t05 / R@10 outlier. Idiosyncratic geometry, not noise. Discipline problem — continued tolerance without diagnosis is the bottleneck on tightening CIs. | Reports 026, 028, 029 |
| 7 | ~~Phase 3 codebook-comparison data integrity~~ — **CLOSED** | [Report 039](reports/039_phase3_codebook_comparison_integrity.md): labeling bug, not data integrity. Phase3b and phase3c each load phase3a's `random` and `learned` artifacts and save them back unchanged for per-directory self-containment, so the comparison script enumerates 6 condition labels backed by only 4 distinct tensors. Audit overclaimed on `reconstruction ≡ error_driven` (they are genuinely distinct). **Phase 4 uses `phase3c_codebook_reconstruction.pt` which is byte-distinct from every other Phase 3-era codebook** — graduation result unaffected. Report 017 headline stands; §3 interpretation corrected. Low-priority cleanup: deduplicate-by-tensor-hash in `experiments/31_phase3_comparison.py` so future runs aren't misleading. | audit-report-2026-05-14.md §2.3, §Appendix A; report 039 |

---

## Audits passed / failed for current phase

| Audit | Status | Evidence |
| --- | --- | --- |
| FEP / anti-homunculus on Phase 4 mechanisms | Passed | Report 026 §FEP audit |
| Random-codebook control | Passed | Report 026 |
| Multi-seed Phase 3+4 integration (drift-effective) | **Conditionally passed** — report 029 (st=0.3) produces real drift; ΔR@10 4/5 positive, mean +0.014; top1 regresses due to stale-discovered-patterns gap (blocker #6) | Report 029 |
| Shuffled-token control | Not run (Phase 3 discipline; Phase 4 design doesn't require) | — |
| Pattern death (architectural component) | **Not exercised** in any session run; report 033 mechanism diagnostic explains why and reshapes blocker #2 | Reports 026, 033 |
| Entropy exit criterion ("repeated paths lower entropy") | **Failed** at W=4 | Report 026 §Engagement / entropy |

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

If you graduate Phase 4 without these, document why.

---

## Reading order for someone catching up

1. This file (STATUS.md).
2. [notes/emergent-codebook/phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md) — **Phase 5 design**. HAM × energy-guided structural branching; decision #1 closed (schema source = post-death substrate).
3. [reports/040_freq_weighted_alpha_sweep.md](reports/040_freq_weighted_alpha_sweep.md) — **closes the freq-α bridge experiment**; resolves Phase 5 decision #1; documents the supercritical λ ceiling.
4. [reports/038_phase4_d1_graduation.md](reports/038_phase4_d1_graduation.md) — **Phase 4 graduation result**. n=10 integration regime, Δms_w3 = −0.7920 CI-disjoint; 10/10 seeds. Trajectory analysis (seeds 17, 3) shows mechanism heterogeneity.
3. [reports/037_seed3_collapse_diagnostic.md](reports/037_seed3_collapse_diagnostic.md) — **closes option 1**: seed 3 collapse is same death-survivor mechanism as seed 17; corpus-order variance under binary death. Establishes the rationale for the D1 pivot.
4. [reports/036_decay_sweep_and_mass_death_finding.md](reports/036_decay_sweep_and_mass_death_finding.md) — A_k decay sweep + seed-17 trajectory; identifies death as the dominant substrate-shaping force.
5. [notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md](notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md) — **binding standing discipline**: top1 demoted to drill-down; substrate-pure metrics (D1) preferred when readout × substrate variance is large.
4. [reports/035_saighi_ak_n10_falsification.md](reports/035_saighi_ak_n10_falsification.md) — n=10 verification of A_k at decay=0; falsified report 034. (Framed pre-discipline-shift around Δtop1; reading order item 3 reframes.)
5. [reports/034_saighi_ak_seed1_prototype.md](reports/034_saighi_ak_seed1_prototype.md) — A_k seed-1 prototype (falsified; preserved as the optimistic single-seed precedent).
6. [notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md](notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md) — paper synthesis that proposed A_k + FHRR audit.
7. [reports/033_phase4_death_mechanism_diagnostic.md](reports/033_phase4_death_mechanism_diagnostic.md) — blocker #2 mechanism diagnostic.
8. [reports/032_phase34_n10_verification.md](reports/032_phase34_n10_verification.md) — n=10 baseline (no A_k); comparison for report 035.
3. [reports/030_phase34_rfix_5seed.md](reports/030_phase34_rfix_5seed.md) — rfix did not work; hypothesis revised; death mechanism now load-bearing.
3. [reports/029_phase34_integration_st03.md](reports/029_phase34_integration_st03.md) — ΔR@10 conditionally verified under real drift via st=0.3.
4. [reports/028_phase34_integration_5seed.md](reports/028_phase34_integration_5seed.md) — st=0.5 run; drift didn't fire; superseded by 029.
4. [reports/027_full_repo_audit_synthesis.md](reports/027_full_repo_audit_synthesis.md) — full audit.
5. [reports/026_phase4_verification_design_spec.md](reports/026_phase4_verification_design_spec.md) — Phase 4 isolation verified.
6. [reports/d1_metastable_5seed.json](reports/d1_metastable_5seed.json) — D1 drill-down (Phase 4 collapses W=3/W=4 meta-stable rate by ~50pp).
7. [notes/emergent-codebook/phase-4-checklist.md](notes/emergent-codebook/phase-4-checklist.md) — exit criteria as line items.
8. [notes/emergent-codebook/phase-4-unified-design.md](notes/emergent-codebook/phase-4-unified-design.md) — original design spec.
9. [CLAUDE.md](CLAUDE.md) — agent working rules.

Anything older than report 022 is context, not load-bearing for current Phase 4 work.

---

## Update rule for this file

- Update at the end of every working session.
- Promote a "blocker" to "done" only when there is a report with multi-seed CI evidence.
- If a session walks back something currently in this file, that walk-back is the **first** edit of the session.
