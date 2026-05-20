# Phase 5 — MHN Memory-Cliff Diagnostic (Experiment 41)

**Question.** Does MHN recall break catastrophically as `n_atoms` shrinks on
this project's substrate (`dim=4096`, FHRR complex phasors, `beta=10`)?
Specifically: do the W=4 post-death substrate sizes (`n_atoms in {6, 10, 12}`)
sit below a Sharma-Chandra-Fiete-style capacity cliff, such that no
death-mechanism redesign could possibly recover separable retrieval?

**TL;DR.** No cliff at small `n_atoms`. The post-death substrate
(`n_atoms in {6, 10, 12}`) sits **solidly above any recall cliff**: in the
task-spec noise range (0.1 to 0.9) every cell from `n_atoms=2` through
`n_atoms=1024` is at perfect recall=1.000. Within MHN, the only "cliff"
that exists with `dim=4096` runs in the *opposite* direction from the
hypothesis: more patterns means lower noise tolerance, not less. K-branch
collapse on the post-death substrate is therefore **not bounded by an MHN
capacity limit**; the room for a death-mechanism redesign to help is the
full distance between the observed K-branch collapse and the perfect single-
pattern recall the substrate already supports.

This finding is descriptive (anti-homunculus): it bounds what death-side
work *could* achieve, but does not license a phase decision by itself.

## Setup

- substrate: `TorchFHRR(dim=4096, device='cpu')`
- memory: `TorchHopfieldMemory`, beta=10, max_iter=10
- 20 trials per seed, 3 seeds (1, 11, 23) per cell
- query corruption: complex Gaussian noise added to the unit phasor, then
  `substrate.normalize` restores unit magnitude per dim
- noise sweep: task-spec range {0.1, 0.3, 0.5, 0.7, 0.9} **plus** an
  extended range {1.5, 3.0, 6.0, 12.0, 24.0} added because the spec range
  was uniformly saturated at recall=1.000

## Recall table (mean over 3 seeds x 20 trials = 60 trials per cell)

Spec range (left half) and extended range (right half):

| n_atoms | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.5 | 3.0 | 6.0 | 12.0 | 24.0 |
|--------:|----:|----:|----:|----:|----:|----:|----:|----:|-----:|-----:|
|       2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
|       4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |
|       6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |
|      12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.767 |
|      24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.600 |
|      48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.400 |
|      96 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.533 | 0.050 |
|     256 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.383 | 0.033 | 0.000 |
|    1024 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.233 | 0.000 | 0.000 | 0.000 |

## Cliff location

Within the **task-spec noise range (0.1 to 0.9)**: there is no cliff at any
`n_atoms`. Every cell is 1.000.

Within the **extended range**, the half-recall point (the noise scale at
which recall first crosses 0.5 going down) moves **with `n_atoms`**:

| n_atoms | noise at recall ~ 0.5 |
|--------:|----------------------:|
|       2 | > 24.0                |
|       4 | > 24.0                |
|       6 | > 24.0                |
|      12 | > 24.0 (still 0.767 at 24) |
|      24 | between 12 and 24     |
|      48 | between 12 and 24     |
|      96 | between 6 and 12      |
|     256 | around 6.0            |
|    1024 | around 3.0            |

The cliff that exists runs **opposite to the hypothesis**: at fixed
`dim=4096` and `beta=10`, smaller `n_atoms` is *more* robust to query
noise, not less. This matches the standard MHN picture (exponential capacity
in `D`; far below capacity, recall is dominated by per-pair pattern
separation which depends on the worst-case interferer count).

## Where the project's regimes sit

- Post-death (`n_atoms in {6, 10, 12}`): **far above any recall cliff**.
  At every spec-range noise level these sizes are at 1.000; even at the most
  extreme tested corruption (noise=24, which all but destroys the phase
  signal) they still recall correctly ~77-90% of the time.
- Pre-death (`n_atoms = 1024`): **above the cliff in the spec range**
  (1.000 across 0.1 to 0.9 and out to 1.5), with degradation appearing
  only at noise >= 3.0 — well outside any biologically meaningful
  corruption level.

## Interpretation for Phase 5

K-branch collapse on the post-death substrate **cannot** be attributed to an
MHN capacity floor at small `n_atoms`. The substrate at `n_atoms=6` is
trivially separable; if K-branch retrieval still collapses to one attractor
there, the failure is in the branch-bundle-resettle dynamics or in the
relative geometry the death event leaves behind, not in raw retrieval
capacity. This experiment does not by itself license a redesign decision,
but it *removes* the "death-mechanism is hopeless because we're below the
cliff" alternative: any redesign that improves branch separation on this
substrate has room to land in a regime the substrate can support.

## Artifacts

- script: `experiments/41_memory_cliff_diagnostic.py`
- raw results: `reports/phase5_memory_cliff/results.json`
- wall time: ~30 s on CPU
