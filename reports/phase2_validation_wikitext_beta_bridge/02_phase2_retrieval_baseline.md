# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `wikitext`
- vocab size: `2050`
- train windows: `219591`
- validation windows: `23008`
- window size: `8`
- mask count: `1`
- mask position: `center`
- test samples: `128`
- landscape sizes: `64, 256, 1024`
- betas: `0.3, 1, 3, 10, 30`
- device: `mps`
- mps: `True`
- dataset config: `wikitext-2-raw-v1`

## Condition Summaries

| Objective | Retrieval | Landscape | Beta | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 64 | 0.3 | 0.050 | [0.021, 0.111] | 0.158 | 1.000 | 1.000 | 0.006 | -14.059 |
| masked_token | generalization | 64 | 1 | 0.050 | [0.021, 0.111] | 0.158 | 1.000 | 1.000 | 0.012 | -4.357 |
| masked_token | generalization | 64 | 3 | 0.050 | [0.021, 0.111] | 0.158 | 0.000 | 1.000 | 0.035 | -1.593 |
| masked_token | generalization | 64 | 10 | 0.020 | [0.005, 0.069] | 0.158 | 0.000 | 0.000 | 0.166 | -1.001 |
| masked_token | generalization | 64 | 30 | 0.020 | [0.005, 0.069] | 0.158 | 0.000 | 0.000 | 0.199 | -1.000 |
| masked_token | generalization | 256 | 0.3 | 0.059 | [0.028, 0.124] | 0.158 | 1.000 | 1.000 | 0.020 | -18.655 |
| masked_token | generalization | 256 | 1 | 0.059 | [0.028, 0.124] | 0.158 | 1.000 | 1.000 | 0.019 | -5.719 |
| masked_token | generalization | 256 | 3 | 0.059 | [0.028, 0.124] | 0.158 | 0.000 | 1.000 | 0.005 | -2.036 |
| masked_token | generalization | 256 | 10 | 0.000 | [0.000, 0.037] | 0.158 | 0.000 | 0.000 | 0.136 | -1.005 |
| masked_token | generalization | 256 | 30 | 0.010 | [0.002, 0.054] | 0.158 | 0.000 | 0.000 | 0.206 | -1.000 |
| masked_token | generalization | 1024 | 0.3 | 0.000 | [0.000, 0.037] | 0.158 | 0.000 | 1.000 | 0.005 | -23.272 |
| masked_token | generalization | 1024 | 1 | 0.000 | [0.000, 0.037] | 0.158 | 0.000 | 1.000 | 0.005 | -7.102 |
| masked_token | generalization | 1024 | 3 | 0.000 | [0.000, 0.037] | 0.158 | 0.000 | 1.000 | 0.006 | -2.497 |
| masked_token | generalization | 1024 | 10 | 0.010 | [0.002, 0.054] | 0.158 | 0.000 | 0.000 | 0.041 | -1.035 |
| masked_token | generalization | 1024 | 30 | 0.010 | [0.002, 0.054] | 0.158 | 0.000 | 0.000 | 0.166 | -1.000 |
| masked_token | memorization | 64 | 0.3 | 0.067 | [0.023, 0.179] | 0.111 | 1.000 | 1.000 | 0.006 | -14.059 |
| masked_token | memorization | 64 | 1 | 0.067 | [0.023, 0.179] | 0.111 | 1.000 | 1.000 | 0.012 | -4.357 |
| masked_token | memorization | 64 | 3 | 0.067 | [0.023, 0.179] | 0.111 | 0.000 | 1.000 | 0.036 | -1.593 |
| masked_token | memorization | 64 | 10 | 1.000 | [0.921, 1.000] | 0.111 | 0.000 | 0.000 | 0.282 | -1.001 |
| masked_token | memorization | 64 | 30 | 1.000 | [0.921, 1.000] | 0.111 | 0.000 | 0.000 | 0.282 | -1.000 |
| masked_token | memorization | 256 | 0.3 | 0.097 | [0.052, 0.174] | 0.204 | 1.000 | 1.000 | 0.020 | -18.655 |
| masked_token | memorization | 256 | 1 | 0.097 | [0.052, 0.174] | 0.204 | 1.000 | 1.000 | 0.019 | -5.719 |
| masked_token | memorization | 256 | 3 | 0.097 | [0.052, 0.174] | 0.204 | 0.000 | 1.000 | 0.005 | -2.036 |
| masked_token | memorization | 256 | 10 | 1.000 | [0.960, 1.000] | 0.204 | 0.000 | 0.000 | 0.280 | -1.003 |
| masked_token | memorization | 256 | 30 | 1.000 | [0.960, 1.000] | 0.204 | 0.000 | 0.000 | 0.281 | -1.000 |
| masked_token | memorization | 1024 | 0.3 | 0.000 | [0.000, 0.039] | 0.200 | 0.000 | 1.000 | 0.005 | -23.272 |
| masked_token | memorization | 1024 | 1 | 0.000 | [0.000, 0.039] | 0.200 | 0.000 | 1.000 | 0.005 | -7.102 |
| masked_token | memorization | 1024 | 3 | 0.000 | [0.000, 0.039] | 0.200 | 0.000 | 1.000 | 0.006 | -2.497 |
| masked_token | memorization | 1024 | 10 | 1.000 | [0.961, 1.000] | 0.200 | 0.000 | 0.000 | 0.279 | -1.011 |
| masked_token | memorization | 1024 | 30 | 1.000 | [0.961, 1.000] | 0.200 | 0.000 | 0.000 | 0.281 | -1.000 |
| next_token | generalization | 64 | 0.3 | 0.102 | [0.056, 0.178] | 0.133 | 1.000 | 1.000 | 0.161 | -14.059 |
| next_token | generalization | 64 | 1 | 0.102 | [0.056, 0.178] | 0.133 | 1.000 | 1.000 | 0.175 | -4.357 |
| next_token | generalization | 64 | 3 | 0.102 | [0.056, 0.178] | 0.133 | 0.000 | 1.000 | 0.237 | -1.593 |
| next_token | generalization | 64 | 10 | 0.041 | [0.016, 0.100] | 0.133 | 0.000 | 0.000 | 0.254 | -1.001 |
| next_token | generalization | 64 | 30 | 0.051 | [0.022, 0.114] | 0.133 | 0.000 | 0.000 | 0.251 | -1.000 |
| next_token | generalization | 256 | 0.3 | 0.102 | [0.056, 0.178] | 0.133 | 1.000 | 1.000 | 0.039 | -18.655 |
| next_token | generalization | 256 | 1 | 0.102 | [0.056, 0.178] | 0.133 | 1.000 | 1.000 | 0.031 | -5.719 |
| next_token | generalization | 256 | 3 | 0.102 | [0.056, 0.178] | 0.133 | 0.000 | 1.000 | 0.003 | -2.036 |
| next_token | generalization | 256 | 10 | 0.000 | [0.000, 0.038] | 0.133 | 0.000 | 0.000 | 0.081 | -1.005 |
| next_token | generalization | 256 | 30 | 0.000 | [0.000, 0.038] | 0.133 | 0.000 | 0.000 | 0.202 | -1.000 |
| next_token | generalization | 1024 | 0.3 | 0.102 | [0.056, 0.178] | 0.133 | 0.000 | 1.000 | 0.004 | -23.272 |
| next_token | generalization | 1024 | 1 | 0.010 | [0.002, 0.056] | 0.133 | 0.000 | 1.000 | 0.000 | -7.102 |
| next_token | generalization | 1024 | 3 | 0.010 | [0.002, 0.056] | 0.133 | 0.000 | 1.000 | 0.007 | -2.497 |
| next_token | generalization | 1024 | 10 | 0.000 | [0.000, 0.038] | 0.133 | 0.000 | 0.000 | 0.013 | -1.034 |
| next_token | generalization | 1024 | 30 | 0.010 | [0.002, 0.056] | 0.133 | 0.000 | 0.000 | 0.154 | -1.000 |
| next_token | memorization | 64 | 0.3 | 0.182 | [0.102, 0.303] | 0.236 | 1.000 | 1.000 | 0.161 | -14.059 |
| next_token | memorization | 64 | 1 | 0.182 | [0.102, 0.303] | 0.236 | 1.000 | 1.000 | 0.175 | -4.357 |
| next_token | memorization | 64 | 3 | 0.182 | [0.102, 0.303] | 0.236 | 0.000 | 1.000 | 0.237 | -1.593 |
| next_token | memorization | 64 | 10 | 1.000 | [0.935, 1.000] | 0.236 | 0.000 | 0.000 | 0.280 | -1.001 |
| next_token | memorization | 64 | 30 | 1.000 | [0.935, 1.000] | 0.236 | 0.000 | 0.000 | 0.280 | -1.000 |
| next_token | memorization | 256 | 0.3 | 0.096 | [0.050, 0.179] | 0.217 | 1.000 | 1.000 | 0.039 | -18.655 |
| next_token | memorization | 256 | 1 | 0.096 | [0.050, 0.179] | 0.217 | 1.000 | 1.000 | 0.031 | -5.719 |
| next_token | memorization | 256 | 3 | 0.096 | [0.050, 0.179] | 0.217 | 0.000 | 1.000 | 0.003 | -2.036 |
| next_token | memorization | 256 | 10 | 1.000 | [0.956, 1.000] | 0.217 | 0.000 | 0.000 | 0.278 | -1.002 |
| next_token | memorization | 256 | 30 | 1.000 | [0.956, 1.000] | 0.217 | 0.000 | 0.000 | 0.278 | -1.000 |
| next_token | memorization | 1024 | 0.3 | 0.057 | [0.025, 0.126] | 0.125 | 0.000 | 1.000 | 0.004 | -23.272 |
| next_token | memorization | 1024 | 1 | 0.000 | [0.000, 0.042] | 0.125 | 0.000 | 1.000 | 0.000 | -7.102 |
| next_token | memorization | 1024 | 3 | 0.000 | [0.000, 0.042] | 0.125 | 0.000 | 1.000 | 0.007 | -2.497 |
| next_token | memorization | 1024 | 10 | 1.000 | [0.958, 1.000] | 0.125 | 0.000 | 0.000 | 0.276 | -1.010 |
| next_token | memorization | 1024 | 30 | 1.000 | [0.958, 1.000] | 0.125 | 0.000 | 0.000 | 0.278 | -1.000 |

## Best Generalization Rows

| Objective | Landscape | Beta | Accuracy | Bigram | Gap |
|---|---:|---:|---:|---:|---:|
| masked_token | 256 | 0.3 | 0.059 | 0.158 | 0.020 |
| next_token | 64 | 3 | 0.102 | 0.133 | 0.237 |

## Frequency Buckets

### masked_token:generalization:L64:beta0.3
- `q1_most_frequent`: 0.050

### masked_token:generalization:L64:beta1
- `q1_most_frequent`: 0.050

### masked_token:generalization:L64:beta3
- `q1_most_frequent`: 0.050

### masked_token:generalization:L64:beta10
- `q1_most_frequent`: 0.020

### masked_token:generalization:L64:beta30
- `q1_most_frequent`: 0.020

### masked_token:generalization:L256:beta0.3
- `q1_most_frequent`: 0.059

### masked_token:generalization:L256:beta1
- `q1_most_frequent`: 0.059

### masked_token:generalization:L256:beta3
- `q1_most_frequent`: 0.059

### masked_token:generalization:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:L256:beta30
- `q1_most_frequent`: 0.010

### masked_token:generalization:L1024:beta0.3
- `q1_most_frequent`: 0.000

### masked_token:generalization:L1024:beta1
- `q1_most_frequent`: 0.000

### masked_token:generalization:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:L1024:beta10
- `q1_most_frequent`: 0.010

### masked_token:generalization:L1024:beta30
- `q1_most_frequent`: 0.010

### masked_token:memorization:L64:beta0.3
- `q1_most_frequent`: 0.067

### masked_token:memorization:L64:beta1
- `q1_most_frequent`: 0.067

### masked_token:memorization:L64:beta3
- `q1_most_frequent`: 0.067

### masked_token:memorization:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:L256:beta0.3
- `q1_most_frequent`: 0.097

### masked_token:memorization:L256:beta1
- `q1_most_frequent`: 0.097

### masked_token:memorization:L256:beta3
- `q1_most_frequent`: 0.097

### masked_token:memorization:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:L1024:beta0.3
- `q1_most_frequent`: 0.000

### masked_token:memorization:L1024:beta1
- `q1_most_frequent`: 0.000

### masked_token:memorization:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:L1024:beta30
- `q1_most_frequent`: 1.000

### next_token:generalization:L64:beta0.3
- `q1_most_frequent`: 0.102

### next_token:generalization:L64:beta1
- `q1_most_frequent`: 0.102

### next_token:generalization:L64:beta3
- `q1_most_frequent`: 0.102

### next_token:generalization:L64:beta10
- `q1_most_frequent`: 0.041

### next_token:generalization:L64:beta30
- `q1_most_frequent`: 0.051

### next_token:generalization:L256:beta0.3
- `q1_most_frequent`: 0.102

### next_token:generalization:L256:beta1
- `q1_most_frequent`: 0.102

### next_token:generalization:L256:beta3
- `q1_most_frequent`: 0.102

### next_token:generalization:L256:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:L256:beta30
- `q1_most_frequent`: 0.000

### next_token:generalization:L1024:beta0.3
- `q1_most_frequent`: 0.102

### next_token:generalization:L1024:beta1
- `q1_most_frequent`: 0.010

### next_token:generalization:L1024:beta3
- `q1_most_frequent`: 0.010

### next_token:generalization:L1024:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:L1024:beta30
- `q1_most_frequent`: 0.010

### next_token:memorization:L64:beta0.3
- `q1_most_frequent`: 0.182

### next_token:memorization:L64:beta1
- `q1_most_frequent`: 0.182

### next_token:memorization:L64:beta3
- `q1_most_frequent`: 0.182

### next_token:memorization:L64:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:L64:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:L256:beta0.3
- `q1_most_frequent`: 0.096

### next_token:memorization:L256:beta1
- `q1_most_frequent`: 0.096

### next_token:memorization:L256:beta3
- `q1_most_frequent`: 0.096

### next_token:memorization:L256:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:L256:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:L1024:beta0.3
- `q1_most_frequent`: 0.057

### next_token:memorization:L1024:beta1
- `q1_most_frequent`: 0.000

### next_token:memorization:L1024:beta3
- `q1_most_frequent`: 0.000

### next_token:memorization:L1024:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:L1024:beta30
- `q1_most_frequent`: 1.000
