# MVP 0 Build Plan - Energy Memory Kernel

## Purpose

MVP 0 tests the smallest version of the core thesis:

> A local energy-memory substrate can retrieve by lived temporal association,
> distinguish temporal association from content similarity, expose ambiguity as
> trajectory dynamics, and run efficiently enough to become the base of later
> phases.

This is not a chatbot. It is the memory kernel underneath the future system.

## MVP 0 Principles

- Keep the reference backend dependency-free.
- Use controls before adding complexity.
- Prefer small synthetic worlds where the expected answer is known.
- Treat unexpected failure as architectural information.
- Avoid controller fixes; solve with coupled energy terms or local dynamics.
- Keep the MPS path optional until it is stable.

## Components

### 1. FHRR Reference Substrate

File: `src/energy_memory/substrate/fhrr.py`

Implemented operations:

- `random_vector`
- `random_vectors`
- `perturb`
- `bind`
- `unbind`
- `inverse`
- `normalize`
- `bundle`
- `weighted_bundle`
- `similarity`
- `top_k`
- `cleanup`

Purpose: provide an inspectable reference implementation of complex unit-circle
FHRR operations.

### 2. Hopfield Reference Memory

File: `src/energy_memory/memory/hopfield.py`

Implemented:

- Store pattern vectors.
- Retrieve by iterative softmax settling.
- Track scores, weights, entropy, top pattern, energy trace, convergence.

Purpose: content-addressable associative memory over FHRR states.

### 3. Temporal Association Memory

File: `src/energy_memory/memory/temporal.py`

Implemented:

- Store each item with a bundle of nearby temporal neighbors.
- Post-content temporal recall.
- One-pass joint content+temporal recall.
- Iterative coupled content+temporal settling.
- Trajectory trace with top anchor, weight, entropy, and joint score.

Purpose: test lived association as a retrieval axis distinct from content
similarity.

### 4. Torch/MPS Optional Backend

Files:

- `src/energy_memory/substrate/torch_fhrr.py`
- `src/energy_memory/memory/torch_temporal.py`
- `src/energy_memory/memory/torch_hopfield.py`

Implemented:

- Torch FHRR operations.
- Matrix similarity scoring.
- Torch temporal coupled recall.
- Torch Hopfield retrieval for Phase 2.

Purpose: scale the hot path to MPS while preserving the pure-Python reference.

## Experiments

### MVP 0.0 - Synthetic Temporal Recall

File: `experiments/synthetic_temporal_recall.py`

Question: can the memory retrieve temporal neighbors from an ordered experience
stream?

Control: shuffle temporal order while preserving item vectors.

Result:

- Ordered temporal Recall@4: `1.000`
- Shuffled temporal Recall@4: `0.388`
- Delta: `0.613`

Report: `reports/001_temporal_recall.md`

### MVP 0.1 - Content Similarity Distractors

File: `experiments/content_vs_temporal_distractors.py`

Question: can temporal recall beat content similarity when content similarity
points to the wrong items?

Setup:

- `stair`, `ladder`, `ramp`, `step`, and `escalator` are intentionally similar.
- Only `stair` is lived near `slip`, `doctor`, `table`, and `candle`.

Result:

- Content nearest-neighbor Recall@4: `0.000`
- Temporal association Recall@4: `1.000`

Report: `reports/001_temporal_recall.md`

### MVP 0.2 - Regime Sweep

File: `experiments/regime_sweep.py`

Question: where does retrieval blend, commit, or fail?

Swept:

- beta
- temporal window
- semantic-family tightness

Key finding:

- Clean temporal recall is robust at beta `4+`.
- Very tight content families (`family_noise=0.10`) cannot be disambiguated by
  post-content temporal recall.
- Wider families become recoverable at lower beta.

Reports:

- `reports/002_regime_sweep.csv`
- `reports/002_regime_sweep.md`

### MVP 0.3 - Joint Energy Disambiguation

File: `experiments/joint_energy_disambiguation.py`

Question: can temporal context help choose the right content anchor before the
temporal read?

Result on tight family failure:

- Content nearest-neighbor Recall@4: `0.000`
- Post-content temporal Recall@4: `0.000`
- Joint content+temporal Recall@4: `1.000`

Report: `reports/003_joint_energy_disambiguation.md`

Architectural conclusion: temporal association must be part of the energy read,
not only a value read after content settling.

### MVP 0.4 - Coupled Settling

File: `experiments/coupled_settling.py`

Question: can content and temporal context iteratively update each other?

Result:

- Strong temporal evidence commits immediately.
- Weak temporal evidence keeps the right anchor on top but remains partially
  contested.

Example weak-cue trace:

```text
iter=1 top=stair weight=0.478 entropy=0.472
iter=6 top=stair weight=0.540 entropy=0.443
```

Report: `reports/004_coupled_settling.md`

Architectural conclusion: trajectory exposes resolution/ambiguity even when
endpoint recall succeeds.

### MVP 0.5 - Temporal Cue Degradation

File: `experiments/cue_degradation_sweep.py`

Question: what happens when temporal cue quality degrades?

Key finding:

- Temporal degradation alone is too easy because content cue is still exact.
- Wrong temporal cue can flip the anchor at high temporal beta.
- Pure temporal noise does not hurt when exact content anchors the state.

Reports:

- `reports/005_cue_degradation_sweep.csv`
- `reports/005_cue_degradation_sweep.md`

### MVP 0.6 - Dual Degradation

File: `experiments/dual_degradation_sweep.py`

Question: what happens when content and temporal cues both degrade?

Key findings:

- Noisy content is recoverable with at least partially correct temporal evidence.
- Ambiguous family-blend content is recoverable only with correct temporal
  evidence.
- Wrong content can be rescued by correct temporal evidence at beta `4+`.
- Pure noise content cannot be reliably rescued by temporal evidence alone.
- High temporal beta makes wrong temporal context dangerous.

Reports:

- `reports/006_dual_degradation_sweep.csv`
- `reports/006_dual_degradation_sweep.md`

Architectural conclusion: the landscape needs an anchor. The system should not
magically recover identity from no contact with the target.

### MVP 0.7 - MPS Smoke Path

Files:

- `experiments/check_mps.py`
- `experiments/mps_coupled_smoke.py`
- `reports/007_mps_migration.md`

Finding:

- System Python has no Torch.
- Local `.venv` has Torch and NumPy.
- MPS is hidden inside the Codex sandbox.
- MPS works outside the sandbox.

Verified MPS result:

```text
device: mps
mps: True
top anchor: stair
temporal items: doctor, slip, table, candle
```

## Test Suite

Run reference backend tests:

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests
```

Expected:

```text
OK (skipped=2)
```

The skipped tests are optional Torch tests when using system Python.

Run full suite in local `.venv`:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests
```

Expected:

```text
OK
```

## Commands

Run all MVP 0 probes:

```bash
PYTHONPATH=src:. python3 experiments/synthetic_temporal_recall.py
PYTHONPATH=src:. python3 experiments/content_vs_temporal_distractors.py
PYTHONPATH=src:. python3 experiments/regime_sweep.py
PYTHONPATH=src:. python3 experiments/joint_energy_disambiguation.py
PYTHONPATH=src:. python3 experiments/coupled_settling.py
PYTHONPATH=src:. python3 experiments/cue_degradation_sweep.py
PYTHONPATH=src:. python3 experiments/dual_degradation_sweep.py
```

Run MPS smoke test outside the sandbox:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/mps_coupled_smoke.py
```

## MVP 0 Completion Criteria

MVP 0 is complete when:

- Reference backend passes all tests.
- Torch backend passes all tests in `.venv`.
- MPS smoke test passes outside sandbox.
- Temporal recall beats shuffle control.
- Temporal recall beats content similarity distractors.
- Joint energy read rescues entangled content anchors.
- Coupled settling exposes ambiguity through entropy and top-weight trajectory.
- Dual degradation confirms realistic limits without adding controllers.

Current status: all criteria are satisfied except large-scale MPS benchmarking.

## Known Limitations

- Synthetic worlds are tiny.
- Current Torch temporal path is a hot-path implementation, not yet a full
  replacement for every reference experiment.
- MPS requires escalated execution in Codex because sandboxed commands cannot
  see GPU availability.
- Fixed beta values are still explicit knobs. Later phases should make
  effective temperature a property of query geometry and evidence coherence.
- Temporal context is still local-window based. Later phases need trajectory and
  replay-based temporal structure.

## Next Work After MVP 0

1. Benchmark Torch/MPS at `D=4096` and larger stored memories.
2. Port the main sweeps to Torch to avoid Python-loop bottlenecks.
3. Finish the Phase 2 static contextual-completion matrix.
4. Use Phase 2 results to define the first growing-codebook update rule.
5. Keep the reference backend as the correctness oracle.

