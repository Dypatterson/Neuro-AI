# Neuro-AI Audit Report — 2026-05-20

**Auditor:** Full read-only pass across all src/, tests/, experiments/, scripts/, docs/, and notes/  
**Active phase at audit time:** Phase 5 — implementation unblocked  
**Prior audit:** `audit-report-2026-05-14.md` (Phase 4 era). This report is a fresh, independent pass.  
**Test count per STATUS.md:** 216 passing, 0 skipped  
**Files analyzed:** 38 src/, 30 tests/, 51 experiments/, 8 scripts/, plus all config/doc files  

---

## Executive Summary

- **Architecture is clean and well-layered.** The dependency graph is strictly forward-only (substrate → memory → phase2 → phase34 → phase4 → phase5), with zero circular imports and explicit state management throughout.
- **Security posture is excellent for a research repo.** Zero hardcoded secrets, no `eval`/`exec`, no bare `except:`, no wildcard imports. The only preventive gap is `.env` not in `.gitignore`.
- **The CFL stability boundary in frequency-weighted consolidation (`consolidation.py:246-253`) is the most technically dangerous finding.** With `alpha_freq_lambda > ~0.5`, the explicit Euler integration of the u-chain diffusion becomes unstable for frequently-retrieved patterns. No clamp exists.
- **`weights_only=False` in `torch.load` (`snapshot.py:148`)** is a pickle deserialization risk if snapshots are ever shared.
- **GPU pipeline stalls in settling loops** are the biggest performance bottleneck. `torch_temporal.py:coupled_recall()` syncs CPU 4–5 times per iteration (~60 stalls per retrieval); `ham_aggregator.py:155` and `ham_with_layer2.py:266` do the same. The `torch_hopfield.py:retrieve()` method already demonstrates the correct deferred-sync pattern.
- **No dependency lockfile exists** (`requirements.txt`, `uv.lock`, `poetry.lock`). `torch` and `numpy` are unpinned. Builds are not reproducible across environments.
- **Testing is strong for Phase 4/5 but thin at the foundation:** FHRR has 2 tests, pure-Python Hopfield has 1 test. Phase 4 consolidation has 15+ tests — excellent.
- **The pure-Python and Torch backends have already diverged** (`TorchFHRR.permute()` has no `FHRR` equivalent; `FHRR.top_k()` has no `TorchFHRR` equivalent), violating the project's own "do not remove the pure-Python reference backend" rule by drift.
- **README lacks setup instructions.** A stranger cannot clone and run without reverse-engineering the venv setup.
- **One-line verdict: Solid bones with specific gaps — the research architecture is well-designed and the code is unusually disciplined for a solo research project, but dependency management, foundation-layer test coverage, and settling-loop performance need attention before scaling.**

---

## Findings by Category

### 2.1 Repository Structure & Hygiene

- **[Note]** Layout is sensible and well-organized: `src/energy_memory/` (library code organized by phase), `tests/`, `experiments/` (numbered scripts), `scripts/` (aggregation + Colab notebooks), `reports/` (experiment outputs with JSON + markdown), `docs/`, `notes/` (design documents), `research/` (PDFs, gitignored).

- **[Low]** `.gitignore` covers the essentials (`__pycache__/`, `.venv/`, `.DS_Store`, `*.pt`, `tmp/`, `research/`, `.claude/`) but **does not include `.env`**.
  - Evidence: `.gitignore` — no `.env` entry
  - Why it matters: If a `.env` file is ever created (e.g., for API keys when adding an inference server), it would be tracked by default.
  - Suggested fix: Add `.env` and `.env.*` to `.gitignore`.

- **[Note]** No large files checked into git. `*.pt` is gitignored. `git ls-files '*.pt'` returns empty. Model weights live in `reports/` subdirectories on disk but are excluded from version control.

- **[Low]** 8 Colab notebooks are tracked in git (`scripts/colab_*.ipynb`) but are not present in the working tree. They exist only in the git history/HEAD. This is a mild inconsistency — they're tracked but deleted from disk.
  - Evidence: `git ls-files 'scripts/*.ipynb'` returns 8 files; `find scripts/ -name '*.ipynb'` returns 0
  - Suggested fix: Either restore them or `git rm` them and document their Drive/Colab location.

- **[Note]** The `brainstorm-workspace/` directory contains dated brainstorm outputs from the Cowork brainstorm skill. These are research artifacts, not orphaned code.

- **[Low]** `experiments/__pycache__/` and `scripts/__pycache__/` exist on disk despite `__pycache__/` being in `.gitignore`. Not tracked in git; just local artifacts.

### 2.2 Architecture & Design

- **[Note]** **Module graph is strictly layered with zero circular imports.**
  ```
  substrate (Layer 0) → memory (Layer 1) → diagnostics, phase2 (Layer 2)
    → phase34 (Layer 3) → phase4 (Layer 4) → phase5 (Layer 5)
  ```
  Every import flows downward. `TYPE_CHECKING` guards protect against eager torch imports in `phase2/codebook_learner.py`, `phase2/error_driven_learner.py`, `phase2/reconstruction_learner.py`, `phase2/encoding.py`.

- **[Note]** **Codebook / Hopfield separation is clean.** Codebook learners update codebook vectors and never directly mutate Hopfield memory. Hopfield stores/retrieves patterns agnostically. The encoding layer (`phase2/encoding.py`) bridges tokenization to HD representation. Phase boundaries are respected.

- **[Medium]** **`encode_window` has outgrown its `phase2` home.** This function is imported by 5 files across 3 different phase packages: `phase2/error_driven_learner.py:22`, `phase2/reconstruction_learner.py:26`, `phase34/reencoding.py:31`, `phase5/ham_aggregator.py:45`, `phase5/ham_with_layer2.py:44`. It's a cross-cutting utility that should live in a shared location.
  - Suggested fix: Move `encode_window` and `build_position_vectors` to `substrate/encoding.py` or a top-level `encoding.py`, with a re-export from `phase2.encoding` for backward compatibility.

- **[Medium]** **Pure-Python and Torch backends share no formal interface contract.** `FHRR` and `TorchFHRR` provide matching method names but no shared Protocol, ABC, or base class. They have already diverged: `TorchFHRR` has `permute()` (line 65) which `FHRR` does not; `FHRR` has `top_k()` (line 94) which `TorchFHRR` does not. Same pattern for `HopfieldMemory` vs `TorchHopfieldMemory`.
  - Evidence: `substrate/fhrr.py` vs `substrate/torch_fhrr.py` method lists; `memory/hopfield.py` vs `memory/torch_hopfield.py`
  - Why it matters: The project's CLAUDE.md rule "Do not remove the pure-Python reference backend" implies the backends should stay in sync. Without a Protocol, drift is invisible.
  - Suggested fix: Add a `SubstrateProtocol` (Python 3.8+ `typing.Protocol`) in `substrate/__init__.py` that both `FHRR` and `TorchFHRR` must satisfy. Add a test that asserts API parity.

- **[Note]** **No FastAPI or server code exists.** The `pyproject.toml` mentions no web framework. The provider-agnostic inference layer described in the project vision has not been built yet. This is not a gap — it's future work.

- **[Note]** **State management is explicit and well-localized.** All mutable state lives in clearly-typed class attributes (`TorchHopfieldMemory._patterns`, `ConsolidationState.u`, `Layer2State.attractors`). Persistence is through explicit `save/load` functions with path arguments. No global mutable state exists in `src/`.

- **[Low]** **`phase34/reencoding.py` directly mutates `memory._patterns[idx]`** (lines 60, 101), bypassing the `store()` API. The module does call `memory.invalidate_cache()`, but any future side-effects added to `store()` would be silently missed.
  - Suggested fix: Add a `replace_pattern(idx, new_pattern)` method to `TorchHopfieldMemory` that encapsulates the mutation + cache invalidation.

### 2.3 ML / Research-Specific Concerns

- **[High]** **Frequency-weighted alpha can violate the CFL stability condition (`consolidation.py:246-253`).** The explicit Euler step `new_u = u + alpha_eff * laplacian` is stable only when `alpha_eff * max_eigenvalue ≤ 1`. For the 1D discrete Laplacian, max eigenvalue ≈ 4. With `alpha=0.25`, the product is exactly 1.0 (marginally stable). When `alpha_freq_lambda > 0`, `alpha_eff = 0.25 * (1 + lambda * norm_count)` exceeds 0.25 for frequently-retrieved patterns, making the scheme unstable. At `alpha_freq_lambda=1.0` with the most-retrieved pattern at `norm_count=1.0`, `alpha_eff=0.5`, giving `|1 + 0.5*(-4)| = 1.0` — right at the boundary. Any higher lambda diverges.
  - Evidence: `src/energy_memory/phase4/consolidation.py:245-255`
  - Why it matters: If A+B mechanism (per STATUS.md) uses frequency-weighted alpha, u-chain oscillation or divergence could produce silent numerical corruption.
  - Suggested fix: Add `alpha_eff = torch.clamp(alpha_eff, max=0.24)` before the Euler step, or switch to implicit/semi-implicit integration for the Laplacian term.

- **[Note]** **Hopfield energy and update are correct.** The softmax energy `E = -logsumexp(β·scores)/β` and the update rule `softmax(β·scores)` weighted sum match Ramsauer et al. 2020. Beta is exposed with a positivity check. The settling loop defers all CPU syncs to end-of-loop — this is the gold-standard implementation in the codebase.
  - Evidence: `memory/torch_hopfield.py:99-151, 165-166`

- **[Note]** **FHRR operations are mathematically correct.** Bind = element-wise complex product (phase addition), unbind = multiply by conjugate (phase subtraction), bundle = sum + normalize to unit circle, similarity = `mean(real(conj(a)*b))`. Normalization uses `clamp_min(1e-12)` to avoid division by zero.

- **[Note]** **Benna-Fusi consolidation dynamics are faithful.** The Laplacian coupling `Δu_i = α(-2u_i + u_{i-1} + u_{i+1})` with zero-boundary conditions matches the paper. Binary death is a clean latch: counter increments when strength < threshold, resets to 0 when above, death triggers at counter ≥ window.

- **[Medium]** **No codebook utilization perplexity metric.** The codebase measures geometric collapse (`max_pairwise_similarity` in `codebook_learner.py:137`) and has frequency buckets in `metrics.py`, but lacks the standard VQ-VAE diagnostic `exp(H(usage_distribution))` that detects *functional* collapse (many atoms exist but few are selected).
  - Evidence: `phase2/codebook_learner.py:137-145`, `phase2/metrics.py` — no perplexity function
  - Why it matters: Codebook collapse is the project's stated concern (entire Phase 3 is about learning good codes). Geometric health ≠ functional health.
  - Suggested fix: Add a `codebook_utilization_perplexity(usage_counts: Tensor) -> float` to `phase2/metrics.py` and call it in the codebook training loop.

- **[Medium]** **No global `torch.manual_seed` or determinism flags in experiment scripts.** Randomness is controlled through per-instance `TorchFHRR(seed=...)` and `sample_windows(seed=...)`, which is good. But no experiment script sets `torch.manual_seed()`, `torch.backends.cudnn.deterministic = True`, or `torch.use_deterministic_algorithms(True)`. Any torch op using the global RNG (e.g., dropout, `torch.randn` without explicit generator) would be non-deterministic.
  - Evidence: grep for `torch.manual_seed` across `experiments/` — only found in `experiments/40_phase5_branching.py`
  - Suggested fix: Add a `set_deterministic(seed)` helper in `src/energy_memory/` that sets all global seeds, and call it at the top of every experiment script.

- **[Medium]** **No config versioning system.** Parameters are hardcoded as argparse defaults in each experiment script. No Hydra, OmegaConf, or YAML configs. Comparing parameter choices across experiments requires reading each script's source.
  - Why it matters: Parameter archaeology — figuring out what config produced a given result — is already hard. Experiment reports do serialize to JSON, which partially mitigates this.
  - Suggested fix: For new experiments, adopt a minimal YAML config that gets saved alongside the report JSON.

- **[Low]** **Validation = test set in the repo_sample fallback corpus.** `corpus.py:110` sets `test = list(validation)`. This is documented as a small fallback for environments without WikiText-2; real experiments use WikiText-2 with proper splits.

- **[Low]** **No dataset version pinning.** WikiText-2 is loaded by name (`wikitext-2-raw-v1`) which is version-stable via HuggingFace, but there's no hash verification.

### 2.4 Code Quality

- **[Note]** **Zero anti-patterns found.** No bare `except:`, no wildcard imports, no `eval`/`exec`, no mutable default arguments, no global mutable state, no `print()` in `src/`.

- **[Medium]** **Type annotations are incomplete on tensor-heavy interfaces.** `TorchFHRR` methods (`random_vector`, `bind`, `unbind`, `normalize`, `bundle`, `weighted_bundle`, `similarity_matrix`, `top_k`) lack return type annotations. `encode_window` and `build_position_vectors` also lack them. Methods using Python-native types are well-annotated.
  - Evidence: `substrate/torch_fhrr.py` — most methods have no `-> torch.Tensor` return annotation
  - Suggested fix: Add `-> torch.Tensor` return annotations to all `TorchFHRR` public methods and `encoding.py` functions.

- **[Note]** **Docstrings are strong where they matter.** Module-level docstrings with architectural context and anti-homunculus checks are present on every Phase 3+ module. `consolidation.py` includes the full mathematical formulation with paper citations. Individual method docstrings are present for non-obvious behavior; trivial one-liners are left undocumented — appropriate for a research codebase.

- **[Note]** **Config values use dataclasses with named fields consistently.** `ConsolidationConfig`, `ReplayConfig`, `HAMConfig`, `Layer2Config` — no magic numbers in computation. Standard constants (π, ε=1e-12, z=1.96) are used inline appropriately.

### 2.5 Documentation

- **[Medium]** **README lacks setup/install instructions.** No mention of how to create the venv, install dependencies, or which Python version to use. Lists 10+ experiment commands but a stranger would need to reverse-engineer the environment setup.
  - Evidence: `README.md` — no "Installation" or "Setup" section
  - Suggested fix: Add a "Getting Started" section: `python -m venv .venv && .venv/bin/pip install -e '.[torch,phase2]'`

- **[Low]** **No standalone `ARCHITECTURE.md`.** The architecture is described across `docs/PROJECT_PLAN.md` (which has a Mermaid diagram and phased roadmap), `notes/emergent-codebook/overview.md`, and various phase design notes. Acceptable for a research project, but raises onboarding cost.
  - Suggested fix: A one-page `ARCHITECTURE.md` that says "start here" and points to the relevant notes in reading order. Could be extracted from `STATUS.md`'s reading-order section.

- **[Note]** **`docs/PROJECT_PLAN.md` is strong.** Clear one-sentence goal, first-principles list, architecture diagram, energy-term decomposition, phased roadmap, non-negotiable design rules.

- **[Note]** **`STATUS.md` is exceptional.** Includes active phase, headline metric, required controls, verified results with CIs, active blockers table, audit status, reading order, and update rules. This is the best status-tracking document I've seen in a solo research project.

- **[Note]** **8 Colab notebooks exist in git** with experiment-specific titles. They are self-contained (each installs dependencies in cell 1).

### 2.6 Dependencies & Environment

- **[High]** **No lockfile exists.** No `requirements.txt`, `uv.lock`, or `poetry.lock`. The only dependency spec is `pyproject.toml` with unpinned extras.
  - Evidence: No lockfile found in repo root or subdirectories
  - Why it matters: `pip install -e '.[torch]'` today installs torch 2.7; in 6 months it installs torch 3.x which may break MPS codepaths or change softmax precision.
  - Suggested fix: Run `pip freeze > requirements-lock.txt` from the working venv and commit it. Or adopt `uv` and commit `uv.lock`.

- **[High]** **`torch` and `numpy` are unpinned in `pyproject.toml`.**
  - Evidence: `pyproject.toml:10-12` — `"numpy", "torch"` with no version constraints
  - Suggested fix: Pin to major version: `"torch>=2.0,<3"`, `"numpy>=1.24,<3"`.

- **[Low]** **Python version loosely declared.** `pyproject.toml` says `requires-python = ">=3.9"` but the active venv uses Python 3.13. No `.python-version` file exists.
  - Suggested fix: Add `.python-version` with `3.13` (or whatever the dev version is).

- **[Note]** **Dependency surface is minimal.** `src/` imports only `torch` (optional) beyond stdlib. No transitive dependency surprises. `fastapi`, `uvicorn`, `mlx` — none are present or imported.

- **[Note]** **MLX is not used anywhere.** The project targets MPS (Metal Performance Shaders via PyTorch), not MLX directly.

### 2.7 Testing

- **[Medium]** **FHRR foundation tests are thin.** `test_fhrr.py` has only 2 tests (29 lines). No coverage for `permute()`, `perturb()`, `weighted_bundle()`, `similarity_matrix()`, or edge cases (zero-vector, NaN input).
  - Evidence: `tests/test_fhrr.py` — 2 test methods in 29 lines
  - Why it matters: FHRR is the foundational substrate. Every higher layer depends on bind/unbind/bundle correctness. If a refactor breaks `normalize()`, dozens of experiments silently produce wrong results.
  - Suggested fix: Add tests for every `FHRR` and `TorchFHRR` public method, including round-trip properties (bind then unbind = identity), normalization invariants, and similarity bounds.

- **[Medium]** **Pure-Python Hopfield tests are thin.** `test_hopfield.py` has 1 test (24 lines). No coverage for empty-memory error, convergence behavior, energy trace, beta sensitivity, or the LSR kernel.
  - Evidence: `tests/test_hopfield.py` — 1 test method in 24 lines
  - Suggested fix: Add tests for: store/retrieve round-trip, convergence in ≤ max_iter, energy monotonicity, empty-memory raise, beta=0 edge case.

- **[Note]** **Phase 4 testing is exemplary.** `test_phase4_consolidation.py` (358 lines, 15+ tests) covers dynamics propagation, bidirectional coupling, inhibition, removal, edge cases, and backward-compatibility guarantees.

- **[Note]** **Phase 5 testing is thorough.** `test_phase5_branching.py` (1549 lines) is the largest test file. `test_phase5_ham.py` and `test_phase5_layer2.py` cover the HAM aggregator and layer-2 attractors.

- **[Note]** **All torch-dependent tests use `@unittest.skipIf(torch is None, ...)`** — clean conditional skip pattern. No tests are unconditionally skipped or commented out.

- **[Low]** **No `tests/__init__.py`.** `unittest discover` works, but some pytest path-resolution modes may behave unexpectedly.

### 2.8 Security & Secrets

- **[Note]** **Zero hardcoded secrets.** Comprehensive grep for `sk-`, `Bearer`, `api_key`, `password`, `OPENAI_`, `ANTHROPIC_`, `HF_TOKEN`, AWS/Azure/GCP patterns — all clean.

- **[Low]** **`.env` not in `.gitignore`** (preventive — no `.env` file exists today).

- **[Medium]** **`weights_only=False` in `torch.load` (`snapshot.py:148`).** This allows arbitrary pickle deserialization. The reason is that `pattern_labels` can be arbitrary Python objects. For a single-user repo this is acceptable, but it's a security risk if snapshots are ever shared.
  - Evidence: `src/energy_memory/phase4/snapshot.py:148`
  - Suggested fix: Document the risk in a comment. If labels are always strings/ints, switch to `weights_only=True` and serialize labels separately as JSON.

- **[Note]** **No `eval`, `exec`, `subprocess`, or network calls in `src/`.** The codebase is a pure library with no I/O beyond file reads/writes.

### 2.9 Performance & Hardware Fit

- **[High]** **GPU pipeline stalls in `torch_temporal.py:coupled_recall()`.** 4–5 `.cpu()` / `float()` / `int()` syncs per iteration inside the settling loop: `int(torch.argmax(...).detach().cpu())` (line 126), `float(weights[...].detach().cpu())` (line 132), `float(joint_scores[...].detach().cpu())` (line 134), `torch_normalized_entropy(weights)` which syncs internally (line 127), convergence delta `float(...)` (line 138). With `max_iter=12`, that's ~60 GPU pipeline stalls per retrieval.
  - Evidence: `src/energy_memory/memory/torch_temporal.py:120-141`
  - Why it matters: Per CLAUDE.md's own GPU performance rule, each sync costs ~7ms on MPS. This makes coupled retrieval ~5× slower than necessary.
  - Suggested fix: Follow the pattern established by `torch_hopfield.py:retrieve()` — accumulate all per-iteration states as tensors, do a single batched `.cpu()` sync after the loop, then select the converged state retrospectively.

- **[High]** **GPU pipeline stalls in HAM settling loops.** `ham_aggregator.py:155` has `float((...).detach().cpu())` inside the convergence check per iteration. `ham_with_layer2.py:266` has the same pattern.
  - Evidence: `src/energy_memory/phase5/ham_aggregator.py:155`, `src/energy_memory/phase5/ham_with_layer2.py:266`
  - Suggested fix: Same deferred-sync pattern as `torch_hopfield.py`.

- **[Medium]** **LSR kernel path in `torch_hopfield.py:_weights()` syncs mid-loop** (`float(total.detach().cpu())` at line 189). The softmax path has no such sync.
  - Suggested fix: Replace with a tensor comparison: `if total <= 0.0:` works on a 0-dim tensor without `.cpu()`.

- **[Medium]** **`reencoding.py:reencode_discovered_patterns()` invalidates cache inside per-pattern loop** (line 103), forcing N pattern-matrix rebuilds instead of 1.
  - Suggested fix: Move `memory.invalidate_cache()` after the loop.

- **[Medium]** **`store_sequence()` in `torch_temporal.py:83-98` uses Python for-loop** to build temporal contexts one item at a time. For large sequences this is O(N×W) Python iterations.
  - Suggested fix: Vectorize with `torch.roll` + batched bundling.

- **[Note]** **Hopfield `retrieve()` is single-query, not batched.** For experiment drivers probing many cues, this means N serial calls. Batched retrieval would help throughput but is an optimization, not a correctness issue.

- **[Note]** **No unnecessary `.to(device)` round-trips found.** `TorchFHRR` creates vectors on CPU (for RNG determinism) then transfers once. All computation stays on-device thereafter.

### 2.10 Reproducibility & Onboarding

- **[High]** **6-month comeback risk: dependency resolution.** Without a lockfile, `pip install -e '.[torch]'` in 6 months will install whatever torch version is current. MPS behavior and API may have changed.
  - Suggested fix: Commit a lockfile (see §2.6).

- **[Medium]** **Time-to-first-successful-run for a new collaborator: ~30 minutes of detective work.** The README has experiment commands but no setup instructions. A contributor would need to: (1) guess Python version, (2) discover the `.[torch,phase2]` extras, (3) figure out MPS requires running outside the sandbox with `.venv/bin/python`, (4) discover `PYTHONPATH=src` is needed.
  - Suggested fix: Add a "Getting Started" section to README with exact commands.

- **[Note]** **Shell scripts exist for multi-seed runs.** `scripts/run_phase34_5seed.sh` and `scripts/run_phase34_saighi_5seed.sh` automate the most common experiment configurations.

- **[Note]** **The Colab notebooks provide a self-contained cloud execution path.** Each installs dependencies in cell 1, so Colab runs are reproducible independent of the local environment.

---

## Prioritized Action List

Ranked by (impact × ease):

1. **[High]** Add `.env` to `.gitignore`. One-line edit, prevents future secret leaks. (`.gitignore`)
2. **[High]** Generate and commit a dependency lockfile. Run `pip freeze > requirements-lock.txt` from the working venv. Or adopt `uv` and commit `uv.lock`. (~5 min)
3. **[High]** Pin torch/numpy version ranges in `pyproject.toml`. Change to `"torch>=2.0,<3"`, `"numpy>=1.24,<3"`. (~2 min, `pyproject.toml:10-12`)
4. **[High]** Add CFL stability clamp to frequency-weighted alpha. Add `alpha_eff = torch.clamp(alpha_eff, max=0.24)` at `consolidation.py:252`. (~5 min, prevents silent numerical corruption)
5. **[High]** Fix GPU sync in `ham_aggregator.py:155` and `ham_with_layer2.py:266`. Defer convergence check to post-loop, following `torch_hopfield.py:retrieve()` pattern. (~30 min per file)
6. **[High]** Fix GPU sync in `torch_temporal.py:coupled_recall()`. Refactor to deferred-sync pattern per `torch_hopfield.py`. (~1 hr)
7. **[Medium]** Add codebook utilization perplexity metric to `phase2/metrics.py`. Standard VQ-VAE diagnostic: `exp(H(usage_distribution))`. (~20 min)
8. **[Medium]** Expand FHRR test suite. Add tests for every public method, round-trip properties, normalization invariants. (~1 hr, `tests/test_fhrr.py`)
9. **[Medium]** Expand Hopfield test suite. Add store/retrieve round-trip, convergence, energy monotonicity, edge cases. (~1 hr, `tests/test_hopfield.py`)
10. **[Medium]** Add "Getting Started" section to README. Exact venv/install/run commands. (~15 min)
11. **[Medium]** Move `invalidate_cache()` call outside the per-pattern loop in `reencoding.py:103`. (~5 min)
12. **[Medium]** Add `SubstrateProtocol` to enforce API parity between FHRR and TorchFHRR. (~30 min)
13. **[Medium]** Add return type annotations to `TorchFHRR` and `encoding.py` public methods. (~30 min)
14. **[Low]** Document the `weights_only=False` risk in `snapshot.py:148` with a comment explaining why and when to revisit. (~2 min)
15. **[Low]** Add a `set_deterministic(seed)` helper for experiment scripts. (~15 min)

---

## Notes / Open Questions

1. **No SDM implementation found.** The project vision mentions Sparse Distributed Memory (Kanerva-style), but no SDM code exists in `src/`. The Modern Hopfield Network is the current memory substrate. Was SDM deferred or replaced by the Hopfield approach? If replaced, updating the project description would help onboarding.

2. **No FastAPI / inference layer.** The project vision describes a provider-agnostic inference server, but no server code exists. This is clearly future work — just confirming it hasn't been started.

3. **The `alpha_freq_lambda` default is 0.0**, so the CFL issue (finding #4) is not triggered by any current experiment. It becomes live only when the A+B mechanism (STATUS.md's pending decision) is implemented. The fix should be applied before that implementation begins.

4. **Colab notebooks are tracked in git but deleted from the working tree.** Is this intentional? If they're canonical artifacts, they should be restorable. If they've moved permanently to Drive, `git rm` them.

5. **The `experiments/` directory has 51 scripts, some numbered (02–41) and some unnumbered (early exploratory scripts).** The unnumbered ones (`coupled_settling.py`, `cue_degradation_sweep.py`, etc.) predate the numbering convention. They still run but their results are superseded by later numbered experiments. Worth a cleanup pass if onboarding matters.

6. **`torch_hopfield.py:retrieve()` has exemplary deferred-sync GPU code.** The three other settling loops (coupled_recall, HAM aggregator, HAM layer2) should be refactored to match this pattern. This would be a good "one pattern, three applications" refactoring session.

7. **The prior audit (`audit-report-2026-05-14.md`) identified a Phase 3 codebook-comparison data integrity issue.** Per STATUS.md blocker #7, this was resolved in report 039 — labeling bug, not data integrity. The graduation result is unaffected.
