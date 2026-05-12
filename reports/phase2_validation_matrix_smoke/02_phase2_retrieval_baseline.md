# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `repo_sample`
- vocab size: `514`
- train windows: `W4=8142, W8=4071`
- validation windows: `W4=3428, W8=1714`
- window sizes: `4, 8`
- mask counts: `1, 2`
- mask positions: `center, edge`
- test samples: `4`
- landscape sizes: `8`
- betas: `1`
- device: `cpu`
- mps: `False`

## Condition Summaries

| Objective | Retrieval | W | Mask | Position | Landscape | Beta | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 8 | 1 | 0.000 | [0.000, 0.562] | 0.667 | 0.000 | 1.000 | 0.036 | -2.513 |
| masked_token | generalization | 4 | 1 | edge | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 0.000 | 1.000 | 0.008 | -2.513 |
| masked_token | generalization | 4 | 2 | center | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 0.000 | 1.000 | 0.065 | -2.513 |
| masked_token | generalization | 8 | 1 | center | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 1.000 | 1.000 | 0.008 | -2.423 |
| masked_token | generalization | 8 | 1 | edge | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 1.000 | 1.000 | 0.022 | -2.423 |
| masked_token | generalization | 8 | 2 | center | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 1.000 | 1.000 | 0.028 | -2.423 |
| masked_token | generalization | 8 | 2 | edge | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 1.000 | 1.000 | 0.020 | -2.423 |
| masked_token | memorization | 4 | 1 | center | 8 | 1 | 0.333 | [0.061, 0.792] | 0.333 | 0.000 | 1.000 | 0.036 | -2.513 |
| masked_token | memorization | 4 | 1 | edge | 8 | 1 | 0.000 | [0.000, 0.490] | 0.000 | 0.000 | 1.000 | 0.008 | -2.513 |
| masked_token | memorization | 4 | 2 | center | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 0.000 | 1.000 | 0.065 | -2.513 |
| masked_token | memorization | 4 | 2 | edge | 8 | 1 | 0.000 | [0.000, 0.793] | 0.000 | 0.000 | 1.000 | 0.051 | -2.513 |
| masked_token | memorization | 8 | 1 | center | 8 | 1 | 0.000 | [0.000, 0.562] | 0.000 | 1.000 | 1.000 | 0.008 | -2.423 |
| masked_token | memorization | 8 | 1 | edge | 8 | 1 | 0.000 | [0.000, 0.562] | 0.333 | 1.000 | 1.000 | 0.022 | -2.423 |
| masked_token | memorization | 8 | 2 | center | 8 | 1 | 0.000 | [0.000, 0.562] | 0.000 | 1.000 | 1.000 | 0.028 | -2.423 |
| masked_token | memorization | 8 | 2 | edge | 8 | 1 | 0.000 | [0.000, 0.562] | 0.000 | 1.000 | 1.000 | 0.020 | -2.423 |
| next_token | generalization | 4 | - | - | 8 | 1 | 0.000 | [0.000, 0.658] | 0.000 | 0.000 | 1.000 | 0.012 | -2.513 |
| next_token | generalization | 8 | - | - | 8 | 1 | 0.000 | [0.000, 0.658] | 0.000 | 1.000 | 1.000 | 0.004 | -2.423 |
| next_token | memorization | 4 | - | - | 8 | 1 | 0.000 | [0.000, 0.562] | 0.333 | 0.000 | 1.000 | 0.012 | -2.513 |
| next_token | memorization | 8 | - | - | 8 | 1 | 0.000 | [0.000, 0.562] | 0.000 | 1.000 | 1.000 | 0.004 | -2.423 |

## Best Generalization Rows

| Objective | W | Mask | Position | Landscape | Beta | Accuracy | Bigram | Gap |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masked_token | 4 | 2 | center | 8 | 1 | 0.000 | 0.000 | 0.065 |
| next_token | 4 | - | - | 8 | 1 | 0.000 | 0.000 | 0.012 |

## Frequency Buckets

### masked_token:generalization:W4:M1:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pcenter:L8:beta1
- `q1_most_frequent`: 0.333

### masked_token:memorization:W4:M1:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M1:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M1:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pcenter:L8:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pedge:L8:beta1
- `q1_most_frequent`: 0.000

### next_token:generalization:W4:M-:P-:L8:beta1
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L8:beta1
- `q1_most_frequent`: 0.000

### next_token:memorization:W4:M-:P-:L8:beta1
- `q1_most_frequent`: 0.000

### next_token:memorization:W8:M-:P-:L8:beta1
- `q1_most_frequent`: 0.000
