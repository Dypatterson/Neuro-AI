# Phase 3a: Hebbian Codebook Learning Results

## Training Config

- corpus: `repo_sample`
- device: `cpu`
- dim: `128`
- epochs: `2`
- eval_betas: `[10.0]`
- eval_landscape_sizes: `[16]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center']`
- eval_window_sizes: `[4]`
- lr: `0.01`
- lr_decay: `0.85`
- mps: `False`
- repulsion_strength: `0.05`
- repulsion_threshold: `0.7`
- seed: `17`
- test_samples: `10`
- train_window_size: `4`
- train_windows: `17181`
- vocab_size: `130`

## Training Log

| Epoch | LR | Mean Drift | Max Sim | Repulsion |
|---:|---:|---:|---:|---:|
| 0 | 0.01000 | 0.006257 | 0.2170 | 0 |
| 1 | 0.00850 | 0.005309 | 0.2162 | 0 |

## Side-by-Side Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random Acc | Learned Acc | Delta | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 16 | 10 | 0.000 | 0.000 | +0.000 | 0.000 |
| masked_token | memorization | 4 | 1 | center | 16 | 10 | 1.000 | 0.800 | -0.200 | 0.200 |
| next_token | generalization | 4 | - | - | 16 | 10 | 0.000 | 0.000 | +0.000 | 0.000 |
| next_token | memorization | 4 | - | - | 16 | 10 | 0.500 | 1.000 | +0.500 | 0.500 |

## Generalization Summary

- **masked_token**: random=0.0000  learned=0.0000  delta=+0.0000  bigram=0.0000
- **next_token**: random=0.0000  learned=0.0000  delta=+0.0000  bigram=0.0000
