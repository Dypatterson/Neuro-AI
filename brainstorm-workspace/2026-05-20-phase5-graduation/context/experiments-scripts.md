# Experiments + scripts inventory — context for Phase 5 graduation brainstorm

Date: 2026-05-20. Scope: `experiments/` (driver scripts) and `scripts/`
(analyzers, pilot launchers, Colab notebooks). The goal of this note is
to tell the brainstorm which pieces of test infrastructure already
exist (so a "let's try X" idea is cheap) and which ones would need new
drivers (more expensive).

------------------------------------------------------------------------
## 1. The two active drivers

Two experiment scripts are the load-bearing entry points for Phase 5
work right now. Everything else in `experiments/` is either historical
(Phase 2 / Phase 3 / Phase 4 baselines) or a one-off diagnostic.

### 1a. `experiments/19_phase34_integrated.py` (1080 lines) — substrate producer

**What it produces.** A trained Phase 3+4 substrate (memory + per-scale
consolidation + codebook), snapshotted at `--snapshot-steps` to
`<output-dir>/snapshots/phase3_phase4_w{W}_step{N}.pt`. This is the
file that everything downstream (`scripts/inspect_topk_diversity.py`,
`scripts/consolidation_geometry_diagnostic.py`, exp 40, exp 41) loads.

**Headline metric the script reports per checkpoint.** All four are
printed and saved to `phase34_results.json`:

- `top1` — masked-token top-1 accuracy on the held-out test set
- `topk` (renamed in aggregators to **R@10**) — top-10 hit rate
- `cap_t_05` (aka **capt5**) — fraction with target score ≥ 0.5
- `cap_t_03` — fraction with target score ≥ 0.3
- `meta_stable_w{2,3,4}` — per-scale meta-stable rate from
  `phase2.metrics.meta_stable_rate`
- `mean_top_score_w2` — for surprise diagnostics
- Phase 4 drill-downs at the final checkpoint: `candidates_total`,
  `consolidations`, `deaths_total`, `n_patterns_alive_w{2,3,4}`,
  `codebook_drift_from_initial`

**Three conditions are run in one invocation** (identical seeds, cue
stream, RNG state restored between):

- **A. baseline_static** — frozen codebook, no replay
- **B. phase3_reencode** (called `phase3_only` in code header) —
  codebook learns online, no Phase 4 replay
- **C. phase3_phase4** — codebook learns online + Phase 4 replay +
  consolidation + periodic re-encoding of stored patterns

Only condition C produces snapshots (the others have no phase4_units).

**Phase 5 A+B+A1+A1' knobs already wired through `--`-args**
(defaults preserve baseline bit-identical when 0):

- `--alpha-anti` (B): substrate energy `H_anti = -α·log(d_eff)` term
- `--coverage-lambda` (A): `(1 − λ·r_ema)` modulation on `reinforce()`
- `--coverage-ema-rate` (A): per-step EMA rate on r_ema (default 0.01)
- `--repulsion-step-size` (B): per-replay-cycle substrate gradient step
- `--death-threshold` / `--death-window` (legacy binary-death; no-op'd
  when `coverage_lambda > 0`)
- `--inhibition-gain` / `--inhibition-decay` (Saighi A_k self-inhibition)
- `--snapshot-steps "500,1500,1700,1800"` (the standard schedule)
- `--snapshot-scales` (defaults to all scales, can restrict)

**Production defaults (binding for A+B+A1+A1' pilot).** Set by
`scripts/run_phase5_a{1,1prime,b}_pilot_seed17.sh`:
- `--n-cues 1800` (mass-death window for wikitext-2 lands ~1500–1800)
- `--success-threshold 0.3`
- `--updater-kind hebbian`
- `--alpha-anti 1.0 --coverage-lambda 1.0 --coverage-ema-rate 0.01
   --repulsion-step-size 100.0`
- `--dim 4096`
- A1 / A1' are library-level changes (no new CLI flag); selected at
  HEAD via what code is checked in.

### 1b. `experiments/40_phase5_branching.py` (1756 lines) — K-branch tester + β driver

**What it does.** Loads a snapshot from exp 19 (or builds a synthetic
substrate for smoke), generates K branches (one surprise + K_main from
the schema store), settles each with prior-biased energy
`E_k(q) = -logsumexp(β·X·q*) - γ·Re(⟨q, p_k⟩)`, combines via three
strategies (bundle, greedy, Boltzmann), reports the **headline ΔE**.

**Three operating modes** (`--mode`):

- **`smoke`** — synthetic raw substrate (random patterns), 12 atoms,
  D=256, fixed `SMOKE_CONDITIONS` grid. Used to sanity-test changes
  to the driver without paying for a real substrate. *Cheap variant
  test slot.*
- **`decision5_spike`** — `(formulation × γ)` grid: per_pattern × global_pull
  crossed with γ ∈ {0.25, 0.5, 1.0}. Used to settle the
  formulation question (closed in favor of per_pattern). *Still
  available for future tuning if needed.*
- **`headline`** — real role-binding cues, paired role-vs-content ΔE per
  cue, random-schema control, γ=0 control, and (post-report-049) the β
  conditions `fid_K1_q0` / `fid_K1_q1`. Aggregate across n_seeds × n_cues
  with 95% CI. This is the actual graduation experiment.

**Cue generators that exist.** `generate_role_binding_cue()` produces
role-binding cues with two tunable knobs:

- `--binding-noise-std` (default 0.05) — noise added to extracted role
  fillers (structural distortion)
- `--content-distortion` (default 0.6) — weight of `content_distractor`
  mixed into the cue. 0.0 = no content disagreement, 1.0 = full
  disagreement. This is *the* "role-vs-content tension" axis.

There is **no separate masked-window cue path inside exp 40** — those
live in exp 19's `evaluate_combined()` (target = `window[masked_pos]`,
single-token contextual completion). If the brainstorm wants to test
masked-window completion on a Phase 5 substrate, the path is: snapshot
from exp 19 → load into exp 40 → either reuse `_build_role_binding_cues`
or write a masked-window cue builder against the same loaded
(memory, consolidation, positions, codebook) tuple.

**Headline conditions registered in `--mode headline`**:

- `role_K{km}` — role-binding similarity prior
- `content_K{km}` — content similarity prior
- `random_K{km}` — random-prior control
- `role_K{km}_g0`, `content_K{km}_g0` — γ=0 (prior off) controls
- `fid_K1_q0`, `fid_K1_q1` — β (continuous role-fidelity-weighted)
  conditions; K=1 because β collapses K into one continuous prior
- `role_K1`, `content_K1` — extra K=1 controls if k_main > 1

**Prior types implemented** (`prior_type=`):

- `"content"` — top-k by cue-schema FHRR cosine
- `"role"` — top-k by mean-of-best-match role-binding similarity
  (requires `cue_bindings` + `schema_bindings`)
- `"random"`
- `"fidelity_weighted"` — β prior, continuous weighting over the full
  pattern matrix (not top-k)

**Per-branch diagnostics logged but NOT used for selection**
(anti-homunculus discipline):
`energy_unbiased`, `energy_biased`, `energy_drop`, `prior_alignment`,
`score_entropy_initial/final`, `entropy_collapse`,
`final_state_divergence`, `recall_support`, `cap_coverage_t05`,
`meta_stable`, `structural_match`, `converged`.

**Atom-splitting joint diagnostic.** `split_eligible`,
`n_in_low_energy_set`, `max_state_distance_in_low_energy_set` — gated
by `--delta-energy 0.1` and `--delta-state 0.3`.

------------------------------------------------------------------------
## 2. Phase-5 analyzers / diagnostics (scripts that already exist)

### `scripts/consolidation_geometry_diagnostic.py` (287 lines) — substrate d_eff

**Inputs**: one substrate snapshot. **Outputs**: substrate-level d̄
(mean pairwise FHRR distance) + d_eff (participation ratio of pattern
covariance via N×N Gram-matrix trick, fast even at D=4096), plus
per-atom k-NN d̄_t / d_eff_t. Regime classifier at β=10 (θ' = 1/β = 0.1).

Used as a library function by `analyze_phase5_ab_pilot.py` (via
`from consolidation_geometry_diagnostic import diagnose`). Production
working value: `--k-nn 5 --beta 10.0`.

### `scripts/analyze_phase5_ab_pilot.py` (207 lines) — d_eff gate runner

**Inputs**: a pilot directory (default `reports/phase5_ab_pilot_seed17`).
**Outputs**: pilot d_eff trajectory across (W ∈ {2,3,4}) × (steps ∈
{500,1500,1700,1800}); comparison vs pre-A+B baseline at
`reports/phase5_snapshots_local/seed17/`; pass/fail on the pre-committed
**mechanism-validity gate** `d_eff ≥ 25 at step 1800 on W=4`.

This script encodes the binding criterion in code; it does not adjust
any parameter based on output (anti-homunculus discipline). PASS →
"ship to Colab for n=10 retrain"; FAIL → "back to design, NOT
re-tune". Currently passing at 35.20 (A+B+step3) and 35.23 (A+B+A1').

### `scripts/inspect_topk_diversity.py` (97 lines) — A1 criterion #1 measurer

**Inputs**: one snapshot. **Outputs**: top-8 atoms by
`effective_strength`, their pairwise FHRR off-diagonal similarity
mean/min/max, with pre-committed band [0.10, 0.60]. Per-seed report;
n=5 aggregation done by the caller.

This is the script that diagnosed the report-047/048/049 failure (top-8
similarity collapsing to 1.0000 because all top-8 atoms are FP-identical
discovery-channel duplicates).

### `scripts/aggregate_phase5_de.py` — n=N ΔE aggregator

Reads per-seed JSONs from `experiments/40 --mode headline` and emits:
per-seed mean ΔE per tag (K4, K1, K4_g0), across-seed mean ± 95% t-CI,
sign test, pooled per-cue bootstrap CI, LOSO sensitivity, seed-23
side-by-side. **No mechanism reads its output** — graduation rules in
`phase-5-checklist.md` sections A, D are applied by reading this output,
not by the script.

### `scripts/calibrate_repulsion_step_size.py` — one-shot step-size pick

Loaded the post-death W=4 step-1800 snapshots, computed repulsion force
at α=1.0 for `step ∈ {0.01, 0.1, 1, 10, 100, 1000}`, picked the value
whose median Δd_eff is ≈ 0.5 in the collapsed regime. **One-shot, not
iterative**; 100.0 is the locked-in pre-commitment for A+B retrains.

### Phase 3+4 (pre-Phase-5) aggregators — still useful for D1 non-regression

- `aggregate_phase34_5seed.py` — pooled-Wilson + per-seed t CIs for
  ΔR@10, Δcapt5, Δtop1 vs baseline
- `aggregate_phase34_st03.py` — st=0.3 variant aggregator
- `aggregate_phase34_n10.py` — n=10 version
- **`aggregate_phase34_d1_n10.py`** — the **D1 graduation aggregator**.
  Headline: Δ meta_stable_w3 at final checkpoint, condition C minus
  condition A. Drill-downs: Δms_w2, Δms_w4, ΔR@10, Δcap_t05, Δtop1
  (demoted to drill-down per 2026-05-16 discipline note).
  **This is the script that gates "Phase 4 D1 non-regression" for the
  A+B+A1+A1' pilot's criterion #4.**
- `aggregate_d1_metastable.py` — older 5-seed D1 aggregator
- `aggregate_phase34_saighi_5seed.py` — A_k variant aggregator
- `aggregate_phase34_d1_n10.py` — already mentioned, D1 graduation gate

### Other in-tree diagnostics

- `scripts/profile_exp19.py` — cProfile around the exp 19 hot path
  (useful if a brainstorm move adds substrate cost)
- `experiments/22_codebook_health_diagnostic.py`,
  `experiments/24_hebbian_diagnostic.py`,
  `experiments/36_engagement_gate_audit.py`,
  `experiments/37_seed_diagnostic.py` — Phase 3-era audits
- `experiments/28_synergy_probe_phase4.py` — Phase 4 synergy probe
- `experiments/41_memory_cliff_diagnostic.py` — the MESH-style
  memory-cliff check (n_atoms vs retrieval quality; closed: no cliff)
- `experiments/29_replay_store_upgrades_ablation.py` — Phase 4 ablation
- `experiments/cue_degradation_sweep*.py`,
  `experiments/dual_degradation_sweep.py` — cue noise sweeps (older,
  Phase 2-era infrastructure but cue generators may be reusable)
- `experiments/content_vs_temporal_distractors.py` — content/temporal
  distractor sweep (relevant if the brainstorm wants to test
  content-distortion as an axis beyond the role-binding 0.6 default)

------------------------------------------------------------------------
## 3. Colab notebooks — what's set up, what needs work

### Already wired and parameterized

- **`colab_phase5_snapshots.ipynb`** (14 KB, 2026-05-19 22:31). The
  n=5 substrate-snapshot capture notebook. `SEEDS = [17, 11, 23, 1, 2]`,
  `N_CUES`, `SNAPSHOT_STEPS = "500,1500,1700,1800"`, parallel launch
  per-seed-as-worker pattern. Includes an optional cell 8 that runs
  one `--mode headline` smoke against `phase3_phase4_w4_step1800.pt`
  for a sanity check (n_cues=50, k=10). **This is the n=5 retrain
  notebook for A+B / A1 / A1' / β / future variants.** Currently
  configured for a pre-A+B canonical capture; for A+B+ retrains, the
  cell 5 `python experiments/19_phase34_integrated.py` invocation
  needs `--alpha-anti 1.0 --coverage-lambda 1.0 --coverage-ema-rate
  0.01 --repulsion-step-size 100.0` appended. **Minor edit, not new
  notebook.**

- **`colab_phase34_5seed.ipynb`** (6.5 KB). The pre-A+B n=5 sweep,
  st=0.3 + death, hebbian. Useful as a template; not directly
  relevant to Phase 5 substrate retrains.

- **`colab_phase34_5more_seeds.ipynb`** (6.2 KB). Adds 5 more seeds
  (3, 5, 7, 13, 19) to reach n=10. Same template pattern.

- **`colab_phase34_integration_n10_n3k.ipynb`**. **The n=10
  D1-graduation notebook.** `SEEDS = [17, 11, 23, 1, 2, 3, 5, 7, 13,
  19]`, `n_cues=3000`, hebbian st=0.3, A_k off, death on, no-reencode-
  discovered. Includes parallel batching. **This is the notebook
  that produces the n=10 dataset for `aggregate_phase34_d1_n10.py`**
  (Phase 5 mechanism-validity criterion #4). Same comment: for an
  A+B+ retrain at n=10, the cell-5 invocation needs the four
  pre-committed knobs appended.

- **`colab_phase34_saighi_n10_n3k.ipynb`** — same as above but with
  `--inhibition-gain 0.01` (A_k on). Closed by report 036 as
  orthogonal; useful as a parallel-launcher template.

### Bottom line on n=5 / n=10 readiness

The notebooks **exist** and the launcher / drive-copy / aggregator
plumbing is **complete**. To run A+B+A1+A1' (or any future Phase 5
substrate variant) at n=5 or n=10, the only edit is to append the
four pre-committed knobs (`--alpha-anti 1.0 --coverage-lambda 1.0
--coverage-ema-rate 0.01 --repulsion-step-size 100.0`) to the
`subprocess` call inside cell 5. No new notebook is needed for any
variant that lives behind exp 19's existing CLI surface.

The K-branch n=5 headline aggregation also has its plumbing in
`aggregate_phase5_de.py`; the bottleneck is *which substrate snapshots
to feed it*, not the aggregator.

------------------------------------------------------------------------
## 4. Cheap-variant slots — where smoke / spike modes exist

For variants that don't need a full 1800-cue retrain:

- **`experiments/40_phase5_branching.py --mode smoke`** — synthetic
  raw substrate, 12 atoms, D=256. Used to sanity-test changes to the
  branching driver without a snapshot. Sub-second on CPU. Use this
  when iterating on prior formulations, schema selection rules,
  branch combinators, energy diagnostics, etc.
- **`experiments/40_phase5_branching.py --mode decision5_spike`** —
  formulation × γ grid against a synthetic encoded-window substrate.
  Mid-cost (still synthetic, but more realistic structure).
- **`experiments/40_phase5_branching.py --mode headline --n-cues 50`**
  (as in colab cell 8) — real substrate snapshot but only 50 cues. A
  one-snapshot smoke that completes in minutes on CUDA and gives a
  preview of the headline ΔE shape before committing to n=5 × full
  n_cues.
- **`scripts/inspect_topk_diversity.py`** — sub-second per snapshot,
  no compute. Re-run after every code-level change to the discovery-
  channel / r_ema / schema-store machinery; tells you immediately
  whether the top-8 selector has degenerated.

For *substrate-changing* variants (anything that touches consolidation
dynamics or substrate energy), there is currently **no cheap mode** —
they require a 1-seed pilot via `bash scripts/run_phase5_*_pilot_seed17.sh`
on the local MPS box (~minutes-to-tens-of-minutes) before going to
Colab for n=5/n=10. The pilot shell scripts (`run_phase5_ab_pilot…`,
`run_phase5_a1_pilot…`, `run_phase5_a1prime_pilot…`) are templates;
copying one and changing the `--alpha-anti 1.0 …` line is the cheapest
way to add a new substrate variant.

------------------------------------------------------------------------
## 5. Summary table for the brainstorm

| Want to test… | Driver | Mode | Cost | Files to edit |
|---|---|---|---|---|
| Different prior shape (β variant, role-fidelity reformulation, schema-selection rule) | exp 40 | smoke / headline | seconds to minutes | exp 40 source + smoke condition list |
| K-branch ΔE on existing snapshot | exp 40 | `--mode headline` with `--substrate-snapshot …` | minutes/snapshot | none (CLI already there) |
| Substrate-level mechanism (new A/B/C-class dynamic) | exp 19 | full run | 1-seed pilot ~5-15min MPS; n=5 / n=10 Colab | exp 19 CLI args + library + new pilot shell |
| d_eff trajectory of any snapshot | consolidation_geometry_diagnostic.py | — | seconds | none |
| Top-k schema diversity of any snapshot | inspect_topk_diversity.py | — | sub-second | none |
| Phase 4 D1 non-regression on a new substrate variant | colab n=10 notebook | full | hours on Colab A100 | one-line cell-5 edit |
| Phase 5 ΔE headline with paired ΔE_content − ΔE_role | exp 40 + aggregate_phase5_de.py | `--mode headline` × n_seeds | hours on Colab + seconds aggregate | colab_phase5_snapshots.ipynb cell 8 generalized |
| Content-distortion sweep (role-vs-content tension) | exp 40 | headline; vary `--content-distortion` | hours per sweep point | none (existing CLI) |
| Masked-window contextual completion on Phase 5 substrate | none yet | — | need new driver | new evaluator that calls `evaluate_combined()`-style readout on a loaded snapshot |
| Cue noise / role-binding noise sweep | exp 40 | headline; vary `--binding-noise-std` | as above | none |

------------------------------------------------------------------------
## 6. Notable absences (would need new code)

- **No driver evaluates masked-window top1/capt5/ms_w on a Phase 5
  substrate snapshot.** Exp 19's `evaluate_combined()` is wired to the
  in-training codebook and scale slots; exp 40 only does K-branch ΔE.
  A snapshot-loading masked-window evaluator would be a new ~150-line
  script that reuses `phase2.encoding.encode_window`,
  `decode_position`, and the substrate snapshot loader. If the
  brainstorm wants the "does the Phase 5 substrate still serve
  Phase 2-style contextual completion" question, this is the missing
  piece.

- **No driver does multi-snapshot longitudinal d_eff or top-k
  diversity over the whole 0→1800 trajectory.** Current pattern is
  one-snapshot, multiple step files. A small wrapper that runs
  `consolidation_geometry_diagnostic.diagnose()` and
  `inspect_topk_diversity.inspect_topk()` over a directory and emits
  a trajectory CSV/JSON would be ~50 lines.

- **No "branch interpretability" diagnostic.** Per-branch energy
  diagnostics are logged, but there's nothing that asks "which atom
  did each branch land on, and how often do different cues converge
  to the same K branches?" — a clustering-style analysis on the
  `q_settled` states across cues.

- **No automated falsification / pre-registration runner.** The
  shell pilots have pre-committed criteria in their comments, and
  `analyze_phase5_ab_pilot.py` hardcodes criterion #3 (d_eff ≥ 25),
  but criteria #1, #2, #4 each need their own analyzer call. A
  single "run all six criteria, print PASS/FAIL grid" script would
  consolidate the workflow.
