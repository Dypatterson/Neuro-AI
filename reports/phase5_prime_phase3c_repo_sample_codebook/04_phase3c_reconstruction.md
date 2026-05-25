# Phase 3c: Reconstruction-Loss Codebook Results

## Training Config

- consolidation_k: `10`
- corpus: `repo_sample`
- device: `cpu`
- dim: `128`
- epochs: `1`
- eval_betas: `[10.0]`
- eval_landscape_sizes: `[16]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center']`
- eval_window_sizes: `[4]`
- lr_pull: `0.1`
- lr_push: `0.05`
- quality_threshold: `0.15`
- seed: `17`
- test_samples: `10`
- train_beta: `10.0`
- train_landscape_size: `16`
- train_probe_size: `40`
- train_window_size: `4`
- vocab_size: `130`

## Reconstruction Training Log

| Consolidation | Buffer | Pulled | Pushed | Mean Q | Total Pos | Total Fail | Fail Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 3 | 0.0268 | 4 | 3 | 0.7500 |

## Three-Way Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random | Hebbian | Reconstruction | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 16 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| masked_token | memorization | 4 | 1 | center | 16 | 10 | 0.800 | 0.800 | 0.800 | 0.200 |
| next_token | generalization | 4 | - | - | 16 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| next_token | memorization | 4 | - | - | 16 | 10 | 0.500 | 0.500 | 1.000 | 0.500 |

## Generalization Summary

- **masked_token**: random=0.0000 hebbian=0.0000 reconstruction=0.0000 bigram=0.0000
- **next_token**: random=0.0000 hebbian=0.0000 reconstruction=0.0000 bigram=0.0000
