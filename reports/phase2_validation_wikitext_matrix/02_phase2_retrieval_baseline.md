# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `wikitext`
- vocab size: `2050`
- train windows: `219591`
- validation windows: `23008`
- window size: `8`
- mask count: `1`
- mask position: `center`
- test samples: `32`
- landscape sizes: `64, 256`
- betas: `1, 10`
- device: `mps`
- mps: `True`
- dataset config: `wikitext-2-raw-v1`

## Condition Summaries

| Objective | Retrieval | Landscape | Beta | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 64 | 1 | 0.071 | [0.020, 0.226] | 0.214 | 1.000 | 1.000 | 0.012 | -4.357 |
| masked_token | generalization | 64 | 10 | 0.000 | [0.000, 0.121] | 0.214 | 0.000 | 0.000 | 0.161 | -1.001 |
| masked_token | generalization | 256 | 1 | 0.143 | [0.057, 0.315] | 0.214 | 1.000 | 1.000 | 0.019 | -5.719 |
| masked_token | generalization | 256 | 10 | 0.000 | [0.000, 0.121] | 0.214 | 0.000 | 0.000 | 0.123 | -1.005 |
| masked_token | memorization | 64 | 1 | 0.042 | [0.007, 0.202] | 0.125 | 1.000 | 1.000 | 0.012 | -4.357 |
| masked_token | memorization | 64 | 10 | 1.000 | [0.862, 1.000] | 0.125 | 0.000 | 0.000 | 0.282 | -1.001 |
| masked_token | memorization | 256 | 1 | 0.050 | [0.009, 0.236] | 0.150 | 1.000 | 1.000 | 0.019 | -5.719 |
| masked_token | memorization | 256 | 10 | 1.000 | [0.839, 1.000] | 0.150 | 0.000 | 0.000 | 0.280 | -1.002 |
| next_token | generalization | 64 | 1 | 0.043 | [0.008, 0.210] | 0.087 | 1.000 | 1.000 | 0.175 | -4.357 |
| next_token | generalization | 64 | 10 | 0.087 | [0.024, 0.268] | 0.087 | 0.000 | 0.000 | 0.258 | -1.001 |
| next_token | generalization | 256 | 1 | 0.043 | [0.008, 0.210] | 0.087 | 1.000 | 1.000 | 0.031 | -5.719 |
| next_token | generalization | 256 | 10 | 0.000 | [0.000, 0.143] | 0.087 | 0.000 | 0.000 | 0.078 | -1.005 |
| next_token | memorization | 64 | 1 | 0.241 | [0.122, 0.421] | 0.241 | 1.000 | 1.000 | 0.175 | -4.357 |
| next_token | memorization | 64 | 10 | 1.000 | [0.883, 1.000] | 0.241 | 0.000 | 0.000 | 0.280 | -1.001 |
| next_token | memorization | 256 | 1 | 0.143 | [0.050, 0.346] | 0.286 | 1.000 | 1.000 | 0.031 | -5.719 |
| next_token | memorization | 256 | 10 | 1.000 | [0.845, 1.000] | 0.286 | 0.000 | 0.000 | 0.276 | -1.002 |

## Best Generalization Rows

| Objective | Landscape | Beta | Accuracy | Bigram | Gap |
|---|---:|---:|---:|---:|---:|
| masked_token | 256 | 1 | 0.143 | 0.214 | 0.019 |
| next_token | 64 | 10 | 0.087 | 0.087 | 0.258 |

## Frequency Buckets

### masked_token:generalization:L64:beta1
- `q1_most_frequent`: 0.071

### masked_token:generalization:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:L256:beta1
- `q1_most_frequent`: 0.143

### masked_token:generalization:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:memorization:L64:beta1
- `q1_most_frequent`: 0.042

### masked_token:memorization:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:L256:beta1
- `q1_most_frequent`: 0.050

### masked_token:memorization:L256:beta10
- `q1_most_frequent`: 1.000

### next_token:generalization:L64:beta1
- `q1_most_frequent`: 0.043

### next_token:generalization:L64:beta10
- `q1_most_frequent`: 0.087

### next_token:generalization:L256:beta1
- `q1_most_frequent`: 0.043

### next_token:generalization:L256:beta10
- `q1_most_frequent`: 0.000

### next_token:memorization:L64:beta1
- `q1_most_frequent`: 0.241

### next_token:memorization:L64:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:L256:beta1
- `q1_most_frequent`: 0.143

### next_token:memorization:L256:beta10
- `q1_most_frequent`: 1.000
