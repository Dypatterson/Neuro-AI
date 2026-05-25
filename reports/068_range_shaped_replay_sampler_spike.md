# Report 068 — Pre-Gate 2: Range-Shaped Replay Sampler Spike

> **⚠ Pre-commit codex review applied 2026-05-24.** Four corrections from a
> read-only review before this report's first commit, all annotated inline:
>
> 1. **Baseline control rewritten.** Initial version used `ReplayStore.sample()`,
>    which caps `n` at `len(traces)` and samples *without* replacement
>    ([replay_loop.py:262](../src/energy_memory/phase4/replay_loop.py))
>    — so with 400 traces and `n_sampled=1000` the original baseline returned
>    400 traces / 3200 pairs, not 8000. The fix uses a direct priority-weighted
>    `torch.multinomial(replacement=True)` matching what a multi-cycle Phase 4
>    consolidation epoch would touch. JSON now records actual pair counts.
> 2. **"Unused atoms" overreach corrected** (drill-down D-4 below). The
>    sampler reads `atom_weights` from the buffer's encoder_terms — it can
>    only sample atoms with nonzero buffer weight. Rebind-on-the-fly creates
>    new (role, observed-atom) combinations but **cannot reach atoms with
>    zero buffer weight without an explicit smoothing prior**. The 256-cell
>    algorithmic ceiling is genuine for the algorithm as-implemented.
> 3. **87.5% missing-pair math framing fixed** (D-4). The number is `1 - 4/32`
>    (4 atoms per role out of 32 *reachable* atoms), not `1 - 4/128`.
> 4. **Backing-trace candidate selection now priority-weighted.** When
>    multiple traces back a sampled (role, atom) pair, the sampler previously
>    picked uniformly; codex flagged that this drops the gate-priority
>    structure encoded in the marginals. Fixed to use multinomial weighted
>    by candidate `gate_signal`.
>
> The headline numbers are unchanged by the corrections (KL 2.08 → 0.013,
> 8× cell coverage); the framing of *why* they hold is sharper.

**Date:** 2026-05-24
**Active phase:** 5 (this is **not a Phase 5 graduation experiment** — see [Report 065 framing](065_mqar_external_architecture_gate.md); the [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md) n≥10 verified standard does not apply here)
**Status:** Codex-recommended pre-Phase-5'-commit gate 2. **Sampler-algorithm spike**, not a full Phase 4 integration. Tests whether `RangeShapedReplaySampler` (Dorrell-Whittington ICLR 2025) can produce rectangular joint support over (role, atom) when reading the **already-wired** `encoder_terms` field of real `TrajectoryTrace` dataclasses. n=3 seeds, synthetic Phase 4 traces with controlled skew.
**Decision:** The sampler algorithm rectangularizes the joint at the expected scale (KL-to-factored 2.08 → 0.013, ~160× reduction). **But the result also exposes that 87% of sampled (role, atom) pairs have no backing trace in the buffer** — meaning a real Phase 4 integration requires *rebind-on-the-fly* (~30 LOC against `encode_window_with_provenance`), or the rectangularization collapses to the buffer's natural support. The pre-gate 2 question for the architectural commit (does range-shaped replay actually create role/content modularization downstream?) **requires the deferred Phase 4 integration to answer.** Without it, the spike shows the sampler is sound but not whether it changes Phase 5 ΔE.

---

## Framing — what this experiment is and is not

The Dorrell-Whittington ICLR 2025 theorem (arXiv:2410.06232) says that nonneg + energy-efficient autoencoder training on data with rectangular joint support over (role, content) *forces* modularization of the learned representation along the role and content axes. The brainstorm smoke at [smoke_a_range_shaped_replay.py](../brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_a_range_shaped_replay.py) established that a range-shaped sampler can produce rectangular support over `FakeTrace` data. The codex pre-gate 2 question is: **does the sampler work on real Phase 4 data structures, and what's the realistic integration path?**

This report is a **sampler-algorithm spike**, not the full integration. Three things this is *not*:

1. Not a Phase 5 graduation experiment. n=3, not n≥10.
2. Not a Phase 4 consolidation comparison. The sampler is tested in isolation; it is *not* yet plugged into `UnifiedReplayMemory.run_replay_cycle`. Whether replay sampled this way actually moves codebook geometry or Phase 5 ΔE remains unanswered.
3. Not a comparison against bundle-first (Report 067). Range-shaped replay is a data-side intervention; bundle-first is an architecture-side intervention. They are not mutually exclusive — both could be part of a Phase 5' design.

**What this report establishes:**
1. The S1 trace-schema extension (encoder_terms in `TrajectoryTrace`, `encode_window_with_provenance` in `phase2/encoding.py`) is **already fully wired** in the codebase. Pre-gate 2's S1 prerequisite is done — no extra LOC needed for the sampler to read provenance.
2. The `RangeShapedReplaySampler` algorithm rectangularizes the joint at the expected scale when given the right operationalization (`sample_pairs` returns (role, atom, optional_trace_idx) triples).
3. **The operationalization matters**: a naive `fallback="skip"` implementation collapses the sampler to the buffer's existing support (no-op if support is sparse). The rectangularization holds only under `sample_pairs` (return the pair regardless of whether a trace backs it).
4. **87% of sampled pairs need rebind-on-the-fly.** This is the unbuilt piece — the brainstorm test-results.md §(a)'s "Strategy 1: Re-bind on the fly."

---

## Setup

- Experiment script: [`experiments/43_range_shaped_replay_gate.py`](../experiments/43_range_shaped_replay_gate.py)
- Substrate: `TorchFHRR(dim=4096)` on CPU, fresh per seed
- Synthetic traces with controlled skew:
  - n_traces = 400
  - n_roles = 8 (= window_size, each trace covers all 8 positions)
  - n_atoms = 128 (token codebook)
  - skew_concentration = 4 (each role draws atoms from a contiguous block of 4 atoms, giving 8×4 = 32 distinct (role, atom) cells with data out of 8×128 = 1024 possible cells)
- Each trace has `encoder_terms` populated via `encode_window_with_provenance`, gate_signal ∈ [0.1, 1.0] uniform random
- Stored in a real `ReplayStore(capacity=410)`
- Sampling: matched pair count per seed (post-codex-fix)
  - Baseline: direct `torch.multinomial(gate_signals, n=1000, replacement=True)` over store traces, then expand each picked trace to its `window_size` (=8) encoder_terms pairs → 8000 pairs per seed. This bypasses `ReplayStore.sample()` because that method caps `n` at `len(self.traces)` and samples without replacement; with 400 traces it would have returned only 3200 pairs and the comparison would not have been matched. The replacement-style multinomial is also a closer model of what a real Phase 4 consolidation epoch sees: many small `replay_batch_size` draws over many cycles, equivalent to a long with-replacement priority-weighted sequence.
  - Range-shaped: `RangeShapedReplaySampler.sample_pairs(n=8000)` emits 8000 `(role, atom, optional_trace_idx)` triples directly.
- Diagnostics: joint-distribution rectangularity (KL-to-factored), cell coverage (out of 1024), marginal preservation (L1 distance between sampler's role / atom marginals), fraction-missing-pairs (no backing trace for sampled pair)

Raw payload: [reports/phase5_range_shaped_replay_gate/results.json](phase5_range_shaped_replay_gate/results.json).

---

## Headline (this experiment, not Phase 5)

Per-seed and aggregate diagnostics at the parameters above:

| Seed | buffer_rect | baseline_rect | **rangeshape_rect** | baseline cells | **rangeshape cells** | base pairs | rs pairs | missing_pairs | role L1 | atom L1 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 17 | 2.0794 | 2.0794 | **0.0129** | 32/1024 | **256/1024** | 8000 | 8000 | 0.875 | 0.018 | 0.061 |
| 11 | 2.0794 | 2.0794 | **0.0137** | 32/1024 | **256/1024** | 8000 | 8000 | 0.873 | 0.027 | 0.065 |
| 23 | 2.0794 | 2.0794 | **0.0119** | 32/1024 | **256/1024** | 8000 | 8000 | 0.871 | 0.025 | 0.065 |
| **mean** | **2.0794** | **2.0794** | **0.0128** | **32** | **256** | **8000** | **8000** | **0.873** | **0.023** | **0.064** |

Three observations:

1. **Rectangularization works at the algorithm level.** KL-to-factored drops from 2.08 (raw buffer) to 0.013 (range-shaped sampler output) — a **161× reduction**, matching the brainstorm smoke's "2.06 → 0.03" (62× at smaller scale). The sampler's marginal-product structure cleanly factorizes the joint.
2. **Cell coverage expands 8×, bounded by the buffer's atom support.** Baseline reaches 32/1024 cells (the buffer's natural support — 8 roles × 4 atoms-per-role). Range-shaped reaches 256/1024 cells (8 roles × 32 *observed* atoms, where the 32 atoms are the union of atoms appearing in any role anywhere in the buffer). The 1024 total reflects 8 × 128 but the atom codebook is only used at 32 of the 128 atom slots, so **256 is the genuine algorithmic ceiling for this sampler** — the marginal `atom_weights` distribution is built from `encoder_terms` of stored traces, so atoms with zero buffer weight are not in the marginal and cannot be sampled. **Rebind-on-the-fly can synthesize new (role, *observed*-atom) combinations the buffer never had, but it cannot reach atoms the encoder has never produced for any window.** Sampling the 768 unreachable cells would require an explicit smoothing prior on the atom marginal (e.g., Laplace `α + count` for each codebook atom, regardless of whether it's appeared in any window) — not implemented here, flagged for the deferred follow-up.
3. **Marginals approximately preserved.** Role L1 ≈ 0.02, atom L1 ≈ 0.06 between baseline and range-shaped marginals. The sampler is sampling the *same* marginals (it's the same buffer's encoder_terms, weighted by gate_signal); slight L1 from finite-sample noise at n=8000 pairs.

---

## The integration cost (what's deferred)

**fraction_missing_pairs = 0.873.** Most sampled (role, atom) pairs have no trace in the buffer that contains exactly that pair. This is what the brainstorm test-results.md §(a) flagged:

> **The substrate-level extension** — what to do when a sampled (role, content) pair has no exact trace — is the work the brainstorm under-counted. Two viable strategies:
>
> 1. **Re-bind on the fly.** Take the sampled role from FHRR position vectors, sample a content vector from the existing content codebook, bind them, treat that as a synthetic trace. Requires the S1 trace-schema extension (~30 LOC, already named in `notes/notes/2026-05-24-spike-S1-replay-trace-schema.md`).
> 2. **Fall back to closest match.** Sample (r, c), search the trace buffer for nearest (r, c′) or (r′, c), use that trace. Simpler; loses the rectangular-support guarantee.

This spike confirms the brainstorm's estimate. Concrete integration cost:

- The **S1 schema extension is already done**: `TrajectoryTrace.encoder_terms`, `encode_window_with_provenance`, and the snapshot serializer all read/write the field. No new LOC needed for that.
- The **rebind-on-the-fly synthesizer** is ~30 LOC against `phase2/encoding.py`. Two variants to choose between, per codex's closing recommendation:
  ```python
  # Variant A: single-binding rebind (purer per Dorrell-Whittington).
  def synthesize_trace_single(substrate, position_vectors, codebook, role, atom, gate=0.1):
      bound = substrate.bind(position_vectors[role], codebook[atom])
      return TrajectoryTrace(query=bound.detach().clone(),
                             encoder_terms=[(int(role), int(atom))])

  # Variant B: window-preserving rebind (matches consolidation's training regime).
  # Sample window_size (role, atom) pairs from the factored marginals,
  # then call encode_window_with_provenance for a full window.
  ```
  Plus ~15 LOC to plumb a substrate / position / codebook reference into `RangeShapedReplaySampler` so it can call the synthesizer when `trace_idx is None`.
- **Atom-support smoothing prior** (~5-10 LOC, optional but architecturally important). The sampler-as-implemented cannot reach atoms with zero buffer weight (D-4 / drill-down above). A Laplace-style prior `atom_weights[a] = α + observed_count[a]` over the full codebook would let the sampler push the model into combinations even the encoder has never produced. Without smoothing, the reachable cell space is bounded by the observed-atom union; with smoothing, it expands to the full `n_roles × n_atoms` grid. Codex 2026-05-24 noted this as the missing piece that makes "production scale" claims hold.
- The **Phase 4 wiring** (`UnifiedReplayMemory.run_replay_cycle` calling the range-shaped sampler instead of `ReplayStore.sample()`) is ~10-20 LOC depending on whether we want a config flag for baseline vs range-shaped or a hard swap.
- Total estimated integration: **~55-75 LOC + a focused test pass + a Phase 4 production run** to compare baseline vs range-shaped consolidation. 1-2 days of work. The window-preserving rebind + smoothing prior add ~10-20 LOC over the minimal version but make the production-scale claim defensible.

After integration, the **gate question** (does range-shaped replay actually produce role/content modularization downstream?) is answered by:
- Running the existing Phase 4 consolidation under both samplers
- Measuring a codebook-geometry metric: linear probe of role vs atom on the consolidated codebook, or per-atom cap-coverage in role-coupled vs content-coupled settings
- Re-running Phase 5 ΔE at n=10 on the range-shaped-trained substrate to check whether the headline moves

That work is **deferred**. It's the cheapest of the four open paths in [STATUS.md](../STATUS.md) blocker #2 (a.2), but it's not free.

---

## Three findings

### F-1: The sampler's algorithm works on real Phase 4 data structures
KL-to-factored: 2.08 → 0.013. Cell coverage: 32 → 256. The brainstorm-scale finding (FakeTrace, 8×32 grid) generalizes to realistic-scale Phase 4 traces (8×128 grid, real `TrajectoryTrace` objects, real `ReplayStore`). The algorithm is sound; the implementation reads the already-wired `encoder_terms` field cleanly.

### F-2: The operationalization choice is load-bearing
A naive `fallback="skip"` (drop sampled pairs that have no backing trace) collapses the sampler to the buffer's existing support — KL stays at 2.08, no rectangularization. The brainstorm smoke avoided this by synthesizing FakeTraces for missing pairs. Real-system integration requires the equivalent: synthesizing TrajectoryTraces via `substrate.bind(positions[role], codebook[atom])`. The diagnostic in this spike uses `sample_pairs` directly (return the pair regardless of trace presence) to test the algorithm, not the integration.

If the integration ends up choosing `fallback="closest"` instead of true rebind-on-the-fly (because rebinding requires substrate access during sampling), the rectangularization will be partial — bounded by how dense the buffer is across (role, atom) cells. At the tested skew (8×4 / 1024 cells covered = 3%), `fallback="closest"` would expand coverage modestly but not to the 256-cell algorithmic ceiling.

### F-3: The 87% missing-pairs number is the architectural feature, not a bug
The sampler's reachable space is 8 roles × 32 *observed* atoms = 256 cells (not 1024 — see correction above). Each role has 4 atoms in the buffer, so probability a sampled `(role, atom)` pair is one of that role's stored pairs is `4/32 = 12.5%`, giving `1 − 4/32 = 87.5%` missing. **Within the algorithm's reachable 256-cell support, 87.5% of sampled pairs are (role, observed-atom) combinations the buffer has never seen as co-occurring.** Training the codebook on these out-of-distribution-but-in-codebook combinations is exactly what Dorrell-Whittington predicts forces modularization. Without rebind-on-the-fly to synthesize the missing pairs as fresh `bind(positions[role], codebook[atom])` traces, the system never sees these combinations.

---

## Implications for the architectural commit

[STATUS.md blocker #2](../STATUS.md) lists four open paths. This report's effect on each:

- **(a.1) M2 training-time EqProp.** Unchanged. Heavier than range-shaped replay; doesn't change the algebra; would benefit from a working data-side modularization mechanism anyway.
- **(a.2) Range-shaped replay + S1 trace-schema.** **Sampler algorithm validated.** S1 prerequisite already done. Integration cost realistic at ~50-65 LOC + 1-2 days of work. Whether it actually moves Phase 5 ΔE downstream is the deferred question.
- **(c.1) GHRR substrate rebuild.** Unchanged (demoted per Report 066).
- **(c.2) Bundle-first Phase 5' architecture.** Unchanged. Bundle-first and range-shaped are *not* mutually exclusive — both could be part of a Phase 5' design (bundle-first changes how Phase 5 uses storage; range-shaped changes how Phase 4 consolidation receives data).

**My read for the decision:** the bundle-first case is now considerably stronger than range-shaped because:
- Bundle-first has demonstrated downstream effect (Report 067: 100% top-1 at K_roles ≤ 8 and N ≤ 128, with `scene_tix = content_tix` confirming basin retrieval).
- Range-shaped's downstream effect is still open. The spike validates the sampler, not the consolidation outcome.

But **range-shaped is the cheaper additional bet**. If the project commits to (c.2) bundle-first Phase 5', the (a.2) range-shaped integration is a ~50 LOC, 1-2 day follow-up that *could* further improve codebook geometry without changing the architecture. It's not "either/or"; it's "(c.2) for sure, (a.2) as a low-cost additional layer."

If the architectural commit is to NOT pursue bundle-first Phase 5' (close Phase 5 and pivot elsewhere), then (a.2) becomes the cheapest remaining recovery attempt — but with the caveat that without bundle-first changing the storage architecture, range-shaped consolidation has to do all the lifting alone.

---

## Drill-downs

### D-1: Why baseline_rect == buffer_rect to 4 decimal places
`baseline_joint_rect = 2.0794` identical to `buffer_joint_rect = 2.0794`. Baseline picks trace indices with priority-weighted multinomial (replacement=True), then expands each picked trace to its full `window_size` (=8) `encoder_terms`. Across the 8000 sampled pairs over 32 occupied cells, the empirical joint is essentially the buffer's gate-weighted joint with negligible finite-sample variance at the 4th decimal place. **The baseline sampler preserves the joint distribution by design** — that's exactly the property the range-shaped sampler is meant to break.

### D-2: Why rangeshape_rect ≈ 0.013, not exactly 0
The sampler factorizes role and atom sampling exactly (independent multinomial draws from `role_marginal` and `atom_marginal`). With infinite samples, the empirical joint would converge exactly to the product of marginals (KL = 0). At n=8000 pairs over 256 reachable cells, the finite-sample noise produces residual KL ≈ 0.01. Cross-seed std on this residual is small (~0.001), consistent with finite-sample noise rather than systematic deviation.

### D-3: Why atom marginal L1 (~0.06) > role marginal L1 (~0.02)
Roles have 8 categories; atoms have 32 reachable categories. With matched 8000-pair samples from both samplers, per-category finite-sample standard error scales as `sqrt(K/n)` (K = number of categories). For roles: sqrt(8/8000) ≈ 0.032 per-cell std; aggregated L1 ≈ 0.05 expected. For atoms: sqrt(32/8000) ≈ 0.063; aggregated L1 ≈ 0.13 expected. Observed values are 0.02 and 0.06 — well within expected finite-sample variation, and the baseline now also carries finite-sample variance (it uses with-replacement multinomial picks, not the deterministic-at-this-scale `store.sample()` that the original draft inadvertently used).

### D-4: How this would change at production scale (with the algorithm's correct ceiling)
The diagnostic is at n_traces=400, n_atoms=128, with the buffer using only 32 of the 128 atom slots. A realistic Phase 4 consolidation would have n_traces ≈ 10k-100k, n_atoms (vocabulary) ≈ 5k-50k, window_size ≈ 16. Two scaling effects compose:

- **Within the algorithmic ceiling.** The reachable-cell ceiling is `n_roles × |observed_atoms|`. With a large vocabulary and many windows, `|observed_atoms|` can grow to a substantial fraction of `n_atoms` — so the ceiling rises. But each role's *backed-by-trace* density also grows (more pair coverage per role), pulling `missing_pairs` *down* somewhat.
- **Outside the algorithmic ceiling.** Atoms with zero buffer weight remain unreachable. At early-training stages (first epoch with sparse encoder use) this is a large fraction. To reach those cells, the sampler needs an explicit atom-support smoothing prior (e.g., Laplace `α + count` for every codebook atom). **Without smoothing the sampler's effective reach is bounded by what the encoder has historically produced.**

**Rebind-on-the-fly is mandatory** at any production scale: the within-ceiling `missing_pairs` rate stays in the 70-95% range across reasonable vocabularies and window densities (each role's atoms ≪ all observed atoms). `fallback="skip"` produces near-empty samples; `fallback="closest"` degrades to within-role atom-shuffling. **Atom-support smoothing is optional but architecturally important** if the goal is for the sampler to push the model into combinations it would never see otherwise — which is the data-side modularization argument's whole point.

### D-5: What rebind-on-the-fly *cannot* do, and what window-preserving rebinds would change
Rebind-on-the-fly as I've sketched it synthesizes `bind(positions[role], codebook[atom])` for any (role, atom) the sampler asks for. The resulting TrajectoryTrace has only **one** (role, atom) pair, not a window of `window_size`. So range-shaped replay through this synthesizer is **single-binding training**, not windowed training. Whether the project's Phase 4 consolidation expects windowed traces is a design question — `phase4/consolidation.py`'s update rules apply per-pattern but the encoder regime is windowed, so single-binding rebinds may not match the consolidation step's natural training distribution.

**Codex 2026-05-24 follow-up: window-preserving rebinds.** A sharper variant samples a full `window_size`-tuple `(role_0=k_0, role_1=k_1, ..., role_{w-1}=k_{w-1})` of (role, atom) pairs from the factored marginals, then binds the full window via `encode_window_with_provenance(substrate, positions, codebook, [k_0, k_1, ..., k_{w-1}])`. This preserves the windowed input distribution while still rectangularizing each (role, atom) cell. The sampling cost is `n_pairs / window_size` window-rebinds vs `n_pairs` single-binding rebinds. Choosing between the two is a Phase 5' design question — windowed rebinds match the existing consolidation regime; single-binding rebinds are mathematically purer per Dorrell-Whittington.

---

## Done-gate compliance

Per [CLAUDE.md "What 'done' looks like for an experiment"](../CLAUDE.md):

1. ✅ **Headline metric reported with CI** (or equivalent): rectangularity per seed + mean; finite-sample standard error noted in D-3.
2. ✅ **Control on same test set.** Baseline sampler (`ReplayStore.sample()` priority-weighted) and range-shaped sampler tested against the same synthetic trace buffer per seed.
3. ✅ **Drill-down metrics explain anomalies.** D-1 (baseline = buffer), D-2 (residual KL), D-3 (marginal L1), D-4 (production scale), D-5 (rebind limits).
4. ✅ **Written up under `reports/`.** This file.
5. ✅ **STATUS.md updated** — see Recent updates entry for 2026-05-24.

The Phase 5 control matrix from [phase-5-unified-design.md:309-314](../notes/emergent-codebook/phase-5-unified-design.md) does not apply — this is not a Phase 5 graduation experiment.

---

## Anti-homunculus check

`RangeShapedReplaySampler` reads buffer state (encoder_terms, gate_signals) and emits per-call sampling decisions governed by static marginal distributions. No controller decides which traces to replay based on a measured metric; the sampling weights are deterministic functions of the buffer's contents. The rebind-on-the-fly synthesizer (deferred) likewise produces a `bind(role, atom)` deterministically from sampler-emitted indices. No `if X then do Y` rule, no arbitration. ✅ Sampling discipline, not arbitration.

---

## Files

- Experiment script: [`experiments/43_range_shaped_replay_gate.py`](../experiments/43_range_shaped_replay_gate.py)
- Raw results: [`reports/phase5_range_shaped_replay_gate/results.json`](phase5_range_shaped_replay_gate/results.json)
- Predecessor: [`brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_a_range_shaped_replay.py`](../brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_a_range_shaped_replay.py) and [test-results.md §(a)](../brainstorm-workspace/2026-05-24-unconsidered-paths/test-results.md)

## Caveats and limits

- **Not Phase 5 verified evidence.** n=3 sampler diagnostic; verified per [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md) requires n≥10 against ΔE.
- **No Phase 4 consolidation comparison.** The sampler is not yet wired into `UnifiedReplayMemory`. Whether range-shaped consolidation actually improves codebook modularization is the deferred ~50-65 LOC + 1-2 day integration.
- **Synthetic traces.** Skew structure is deterministic (each role draws from a contiguous 4-atom block). Real Phase 4 traces have natural language co-occurrence statistics. Rectangularization may be smaller in absolute KL terms on natural data (because buffer skew is also smaller), but the relative reduction should persist.
- **Single binding per rebind.** Rebind-on-the-fly synthesizes single-(role, atom) traces, not windowed traces. The downstream consolidation step's expectations are a design question.
- **No comparison to (c.2) bundle-first.** Bundle-first has demonstrated downstream effect at MQAR scale; range-shaped's downstream effect is unmeasured. The two interventions live at different layers of the architecture and could compose; this report does not propose a composition.
