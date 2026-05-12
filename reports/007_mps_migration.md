# MPS Migration Note

The current MVP 0 kernel is not using MPS. It is a pure standard-library Python
reference implementation.

That was intentional for the first few probes because the active system Python
does not have Torch, NumPy, or pytest installed. The reference kernel keeps the
math inspectable and dependency-free while the experiment shape is still moving.

## Current Check

Command:

```bash
python3 experiments/check_mps.py
```

System Python result:

```text
torch: not installed
mps: unavailable
current kernel: pure Python reference backend
```

Local `.venv` result inside the sandbox:

```text
torch: 2.11.0
mps: False
```

Local `.venv` result outside the sandbox:

```text
torch: 2.11.0
mps: True
mps dot smoke test: passed
```

Interpretation: MPS works on this machine, but sandboxed commands cannot see
it. Use escalated execution for MPS smoke tests and performance runs.

## Implemented

The optional Torch path now exists:

```text
src/energy_memory/substrate/torch_fhrr.py
src/energy_memory/memory/torch_temporal.py
experiments/mps_coupled_smoke.py
```

Verified outside the sandbox:

```text
device: mps
mps: True
top anchor: stair
temporal items: doctor, slip, table, candle
```

## When To Move

Move to Torch/MPS after the dual-degradation sweep. At that point the experiment
shape will be stable enough that performance matters more than editability.

## Migration Shape

Keep the current code as the reference backend and use the second substrate for
hot-path experiments:

```text
src/energy_memory/substrate/fhrr.py          # reference backend
src/energy_memory/substrate/torch_fhrr.py    # MPS backend
```

The Torch backend should preserve the same conceptual operations:

- random unit complex vectors
- perturb
- bind / unbind
- normalized bundle
- weighted bundle
- cosine similarity
- top-k cleanup

Expected wins:

- matrix scoring for all stored patterns at once
- batched sweep conditions
- larger dimensions (`4096`) and larger memories without waiting on Python loops

Do not remove the reference backend. It is useful for tests, debugging, and
proving behavior independent of Torch/MPS quirks.
