# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `wikitext`
- vocab size: `2050`
- train windows: `219591`
- validation windows: `23008`
- landscape windows: `64`
- device: `mps`
- mps: `True`
- dataset config: `wikitext-2-raw-v1`

## Condition Summaries

| Condition | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| masked_token:generalization | 0.000 | [0.000, 0.121] | 0.214 | 0.000 | 0.000 | 0.161 | -1.001 |
| masked_token:memorization | 1.000 | [0.862, 1.000] | 0.125 | 0.000 | 0.000 | 0.282 | -1.001 |
| next_token:generalization | 0.087 | [0.024, 0.268] | 0.087 | 0.000 | 0.000 | 0.258 | -1.001 |
| next_token:memorization | 1.000 | [0.883, 1.000] | 0.241 | 0.000 | 0.000 | 0.280 | -1.001 |

## Frequency Buckets

### masked_token:generalization
- `q1_most_frequent`: 0.000

### masked_token:memorization
- `q1_most_frequent`: 1.000

### next_token:generalization
- `q1_most_frequent`: 0.087

### next_token:memorization
- `q1_most_frequent`: 1.000
