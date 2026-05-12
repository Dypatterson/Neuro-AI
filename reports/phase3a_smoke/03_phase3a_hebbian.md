# Phase 3a: Hebbian Codebook Learning Results

## Training Config

- corpus: `repo_sample`
- device: `mps`
- dim: `512`
- epochs: `3`
- eval_betas: `[10.0]`
- eval_landscape_sizes: `[32]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center']`
- eval_window_sizes: `[4]`
- lr: `0.01`
- lr_decay: `0.85`
- mps: `True`
- repulsion_strength: `0.05`
- repulsion_threshold: `0.7`
- seed: `17`
- test_samples: `16`
- train_window_size: `8`
- train_windows: `4073`
- vocab_size: `2050`

## Training Log

| Epoch | LR | Mean Drift | Max Sim | Repulsion |
|---:|---:|---:|---:|---:|
| 0 | 0.01000 | 0.001275 | 0.1505 | 0 |
| 1 | 0.00850 | 0.001084 | 0.1506 | 0 |
| 2 | 0.00723 | 0.000922 | 0.1507 | 0 |

## Side-by-Side Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random Acc | Learned Acc | Delta | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 32 | 10 | 0.000 | 0.000 | +0.000 | 0.500 |
| masked_token | memorization | 4 | 1 | center | 32 | 10 | 1.000 | 1.000 | +0.000 | 0.267 |
| next_token | generalization | 4 | - | - | 32 | 10 | 0.000 | 0.000 | +0.000 | 0.154 |
| next_token | memorization | 4 | - | - | 32 | 10 | 1.000 | 1.000 | +0.000 | 0.143 |

## Generalization Summary

- **masked_token**: random=0.0000  learned=0.0000  delta=+0.0000  bigram=0.5000
- **next_token**: random=0.0000  learned=0.0000  delta=+0.0000  bigram=0.1538
