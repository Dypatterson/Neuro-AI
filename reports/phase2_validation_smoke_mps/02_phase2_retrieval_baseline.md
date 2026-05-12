# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `repo_sample`
- vocab size: `2050`
- train windows: `4067`
- validation windows: `1714`
- landscape windows: `64`
- device: `mps`
- mps: `True`

## Condition Summaries

| Condition | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| masked_token:generalization | 0.000 | [0.000, 0.125] | 0.074 | 0.000 | 0.000 | 0.243 | -1.000 |
| masked_token:memorization | 1.000 | [0.883, 1.000] | 0.276 | 0.000 | 0.000 | 0.235 | -1.000 |
| next_token:generalization | 0.000 | [0.000, 0.138] | 0.083 | 0.000 | 0.000 | 0.151 | -1.000 |
| next_token:memorization | 1.000 | [0.879, 1.000] | 0.321 | 0.000 | 0.000 | 0.243 | -1.000 |

## Frequency Buckets

### masked_token:generalization
- `q1_most_frequent`: 0.000
- `q2`: 0.000
- `q3`: 0.000

### masked_token:memorization
- `q1_most_frequent`: 1.000
- `q2`: 1.000

### next_token:generalization
- `q1_most_frequent`: 0.000
- `q2`: 0.000

### next_token:memorization
- `q1_most_frequent`: 1.000
