# Neuro-AI MVP 0 Energy Memory Kernel

This workspace now contains a dependency-free first kernel for testing the
project's core memory bet: temporally lived association should be recoverable
from an energy-memory substrate, and that signal should collapse when temporal
order is shuffled.

## Planning Docs

- [Full project plan](docs/PROJECT_PLAN.md)
- [MVP 0 build plan](docs/MVP0_BUILD_PLAN.md)

## What Is Implemented

- `FHRR` substrate with random unit complex vectors, binding, unbinding,
  normalized bundling, weighted bundling, cosine-like similarity, and cleanup.
- Modern Hopfield-style retrieval with iterative softmax settling, entropy, top
  pattern, and energy trace diagnostics.
- Temporal association memory that stores each item with a bundle of nearby
  experience-stream neighbors.
- Synthetic temporal recall experiment with temporal shuffle control.
- Standard-library `unittest` tests. No NumPy, Torch, or pytest required.

## Run Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Run The First Experiment

```bash
PYTHONPATH=src python3 experiments/synthetic_temporal_recall.py
```

## Run The Similarity-Distractor Experiment

```bash
PYTHONPATH=src python3 experiments/content_vs_temporal_distractors.py
```

This checks whether temporal recall can recover lived neighbors even when a
plain content nearest-neighbor baseline is pulled toward intentionally similar
distractors.

## Run The Regime Sweep

```bash
PYTHONPATH=src:. python3 experiments/regime_sweep.py
```

Outputs:

- `reports/002_regime_sweep.csv`
- `reports/002_regime_sweep.md`

## Run The Joint Energy Disambiguation Probe

```bash
PYTHONPATH=src:. python3 experiments/joint_energy_disambiguation.py
```

This checks whether a temporal-context cue can rescue the tight content-family
failure found by the regime sweep.

## Run The Coupled Settling Probe

```bash
PYTHONPATH=src:. python3 experiments/coupled_settling.py
```

This turns the joint score into an iterative loop and records the settling
trajectory.

## Run The Cue-Degradation Sweep

```bash
PYTHONPATH=src:. python3 experiments/cue_degradation_sweep.py
```

Outputs:

- `reports/005_cue_degradation_sweep.csv`
- `reports/005_cue_degradation_sweep.md`

## Run The Dual-Degradation Sweep

```bash
PYTHONPATH=src:. python3 experiments/dual_degradation_sweep.py
```

Outputs:

- `reports/006_dual_degradation_sweep.csv`
- `reports/006_dual_degradation_sweep.md`

## Check MPS Availability

```bash
python3 experiments/check_mps.py
```

The current backend is pure Python unless this check reports Torch with MPS
available.

The local `.venv` has PyTorch installed. Because this app's sandbox hides MPS,
run MPS smoke tests outside the sandbox:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/mps_coupled_smoke.py
```

For a Phase 2 style contextual-completion probe on the same backend:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/mps_contextual_completion_smoke.py
```

For the first small Phase 2 retrieval-baseline validation run:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/02_phase2_retrieval_baseline.py
```

To aim that same driver at WikiText-2 once the Phase 2 dataset dependency is
installed:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/02_phase2_retrieval_baseline.py --corpus-source wikitext
```

Expected successful MPS pattern:

```text
device: mps
mps: True
top anchor: stair
temporal items: doctor, slip, table, candle
```

Expected pattern from the initial run:

- Ordered stream: high temporal Recall@4
- Shuffled stream: much lower temporal Recall@4
- Positive shuffle delta: evidence that recall is using temporal structure, not
  just item identity

## Next Brick

The next useful step is wiring the Phase 2 experimental driver around the
shared retrieval core, using the pure-Python path as the reference backend and
the optional Torch/MPS path for larger validation runs.
