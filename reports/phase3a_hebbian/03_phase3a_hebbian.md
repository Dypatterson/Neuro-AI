# Phase 3a: Hebbian Codebook Learning Results

## Training Config

- corpus: `wikitext`
- device: `mps`
- dim: `4096`
- epochs: `15`
- eval_betas: `[10.0, 30.0]`
- eval_landscape_sizes: `[64, 256]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center', 'end']`
- eval_window_sizes: `[4, 8]`
- lr: `0.1`
- lr_decay: `1.0`
- mps: `True`
- repulsion_strength: `0.2`
- repulsion_threshold: `0.7`
- seed: `17`
- test_samples: `64`
- train_window_size: `8`
- train_windows: `219591`
- vocab_size: `2050`

## Training Log

| Epoch | LR | Mean Drift | Max Sim | Repulsion |
|---:|---:|---:|---:|---:|
| 0 | 0.10000 | 0.069507 | 0.1223 | 0 |
| 1 | 0.10000 | 0.068983 | 0.2273 | 0 |
| 2 | 0.10000 | 0.068548 | 0.3285 | 0 |
| 3 | 0.10000 | 0.068204 | 0.4222 | 0 |
| 4 | 0.10000 | 0.067726 | 0.5067 | 0 |
| 5 | 0.10000 | 0.066984 | 0.5814 | 0 |
| 6 | 0.10000 | 0.065962 | 0.6461 | 0 |
| 7 | 0.10000 | 0.064680 | 0.6912 | 1 |
| 8 | 0.10000 | 0.063175 | 0.6914 | 3 |
| 9 | 0.10000 | 0.061472 | 0.6898 | 2 |
| 10 | 0.10000 | 0.059592 | 0.6986 | 2 |
| 11 | 0.10000 | 0.057567 | 0.6899 | 3 |
| 12 | 0.10000 | 0.055424 | 0.6997 | 3 |
| 13 | 0.10000 | 0.053189 | 0.6926 | 5 |
| 14 | 0.10000 | 0.050888 | 0.6987 | 4 |

## Side-by-Side Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random Acc | Learned Acc | Delta | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 64 | 10 | 0.000 | 0.044 | +0.044 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 64 | 30 | 0.022 | 0.044 | +0.022 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 10 | 0.022 | 0.067 | +0.044 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 30 | 0.022 | 0.000 | -0.022 | 0.156 |
| masked_token | generalization | 4 | 1 | end | 64 | 10 | 0.060 | 0.080 | +0.020 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 64 | 30 | 0.060 | 0.040 | -0.020 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 10 | 0.000 | 0.040 | +0.040 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 30 | 0.080 | 0.100 | +0.020 | 0.140 |
| masked_token | generalization | 8 | 1 | center | 64 | 10 | 0.075 | 0.019 | -0.057 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 64 | 30 | 0.057 | 0.000 | -0.057 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 10 | 0.038 | 0.057 | +0.019 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 30 | 0.019 | 0.019 | +0.000 | 0.226 |
| masked_token | generalization | 8 | 1 | end | 64 | 10 | 0.045 | 0.000 | -0.045 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 64 | 30 | 0.045 | 0.000 | -0.045 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 10 | 0.000 | 0.159 | +0.159 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 30 | 0.000 | 0.000 | +0.000 | 0.273 |
| masked_token | memorization | 4 | 1 | center | 64 | 10 | 1.000 | 1.000 | +0.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 64 | 30 | 1.000 | 1.000 | +0.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 256 | 10 | 0.809 | 0.213 | -0.596 | 0.234 |
| masked_token | memorization | 4 | 1 | center | 256 | 30 | 0.957 | 1.000 | +0.043 | 0.234 |
| masked_token | memorization | 4 | 1 | end | 64 | 10 | 0.981 | 0.981 | +0.000 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 64 | 30 | 0.981 | 0.981 | +0.000 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 256 | 10 | 0.929 | 0.119 | -0.810 | 0.167 |
| masked_token | memorization | 4 | 1 | end | 256 | 30 | 0.952 | 0.952 | +0.000 | 0.167 |
| masked_token | memorization | 8 | 1 | center | 64 | 10 | 1.000 | 1.000 | +0.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 64 | 30 | 1.000 | 1.000 | +0.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 256 | 10 | 1.000 | 1.000 | +0.000 | 0.327 |
| masked_token | memorization | 8 | 1 | center | 256 | 30 | 1.000 | 1.000 | +0.000 | 0.327 |
| masked_token | memorization | 8 | 1 | end | 64 | 10 | 1.000 | 1.000 | +0.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 64 | 30 | 1.000 | 1.000 | +0.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 256 | 10 | 1.000 | 1.000 | +0.000 | 0.095 |
| masked_token | memorization | 8 | 1 | end | 256 | 30 | 1.000 | 1.000 | +0.000 | 0.095 |
| next_token | generalization | 4 | - | - | 64 | 10 | 0.040 | 0.080 | +0.040 | 0.120 |
| next_token | generalization | 4 | - | - | 64 | 30 | 0.020 | 0.020 | +0.000 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 10 | 0.020 | 0.060 | +0.040 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 30 | 0.080 | 0.100 | +0.020 | 0.120 |
| next_token | generalization | 8 | - | - | 64 | 10 | 0.045 | 0.000 | -0.045 | 0.273 |
| next_token | generalization | 8 | - | - | 64 | 30 | 0.045 | 0.000 | -0.045 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 10 | 0.000 | 0.114 | +0.114 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 30 | 0.000 | 0.023 | +0.023 | 0.273 |
| next_token | memorization | 4 | - | - | 64 | 10 | 0.981 | 0.981 | +0.000 | 0.192 |
| next_token | memorization | 4 | - | - | 64 | 30 | 0.981 | 0.981 | +0.000 | 0.192 |
| next_token | memorization | 4 | - | - | 256 | 10 | 0.952 | 0.405 | -0.548 | 0.214 |
| next_token | memorization | 4 | - | - | 256 | 30 | 0.952 | 0.952 | +0.000 | 0.214 |
| next_token | memorization | 8 | - | - | 64 | 10 | 1.000 | 1.000 | +0.000 | 0.116 |
| next_token | memorization | 8 | - | - | 64 | 30 | 1.000 | 1.000 | +0.000 | 0.116 |
| next_token | memorization | 8 | - | - | 256 | 10 | 1.000 | 1.000 | +0.000 | 0.071 |
| next_token | memorization | 8 | - | - | 256 | 30 | 1.000 | 1.000 | +0.000 | 0.071 |

## Generalization Summary

- **masked_token**: random=0.0341  learned=0.0418  delta=+0.0077  bigram=0.1987
- **next_token**: random=0.0314  learned=0.0495  delta=+0.0182  bigram=0.1964
