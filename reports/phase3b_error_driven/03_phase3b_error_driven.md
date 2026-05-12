# Phase 3b: Error-Driven Contrastive Codebook Results

## Training Config

- consolidation_k: `100`
- corpus: `wikitext`
- device: `mps`
- dim: `4096`
- epochs: `3`
- eval_betas: `[10.0, 30.0]`
- eval_landscape_sizes: `[64, 256]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center', 'end']`
- eval_window_sizes: `[4, 8]`
- lr_pull: `0.1`
- lr_push: `0.05`
- quality_threshold: `0.5`
- seed: `17`
- test_samples: `64`
- train_beta: `10.0`
- train_landscape_size: `256`
- train_mask_count: `1`
- train_mask_position: `center`
- train_probe_size: `2000`
- train_window_size: `8`
- vocab_size: `2050`

## Error-Driven Training Log

| Consolidation | Buffer | Pulled | Pushed | Mean Q | Total Retr | Total Fail | Fail Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 66 | 25 | 0.0565 | 100 | 100 | 1.0000 |
| 2 | 100 | 69 | 26 | 0.0463 | 200 | 200 | 1.0000 |
| 3 | 100 | 74 | 26 | 0.0401 | 300 | 300 | 1.0000 |
| 4 | 100 | 65 | 17 | 0.0510 | 400 | 400 | 1.0000 |
| 5 | 100 | 77 | 23 | 0.0798 | 500 | 500 | 1.0000 |
| 6 | 100 | 70 | 16 | 0.0525 | 600 | 600 | 1.0000 |
| 7 | 100 | 73 | 15 | 0.0617 | 700 | 700 | 1.0000 |
| 8 | 100 | 79 | 16 | 0.0814 | 800 | 800 | 1.0000 |
| 9 | 100 | 68 | 13 | 0.0953 | 900 | 900 | 1.0000 |
| 10 | 100 | 72 | 17 | 0.0786 | 1000 | 1000 | 1.0000 |
| 11 | 100 | 70 | 10 | 0.0972 | 1100 | 1100 | 1.0000 |
| 12 | 100 | 72 | 14 | 0.0953 | 1200 | 1200 | 1.0000 |
| 13 | 100 | 72 | 17 | 0.0846 | 1300 | 1300 | 1.0000 |
| 14 | 100 | 70 | 16 | 0.1126 | 1400 | 1400 | 1.0000 |
| 15 | 41 | 31 | 9 | 0.1052 | 1441 | 1441 | 1.0000 |
| 16 | 100 | 66 | 24 | 0.0462 | 1541 | 1541 | 1.0000 |
| 17 | 100 | 69 | 28 | 0.0404 | 1641 | 1641 | 1.0000 |
| 18 | 100 | 74 | 29 | 0.0395 | 1741 | 1741 | 1.0000 |
| 19 | 100 | 65 | 22 | 0.0432 | 1841 | 1841 | 1.0000 |
| 20 | 100 | 77 | 22 | 0.0766 | 1941 | 1941 | 1.0000 |
| 21 | 100 | 70 | 20 | 0.0435 | 2041 | 2041 | 1.0000 |
| 22 | 100 | 73 | 18 | 0.0574 | 2141 | 2141 | 1.0000 |
| 23 | 100 | 79 | 20 | 0.0733 | 2241 | 2241 | 1.0000 |
| 24 | 100 | 68 | 16 | 0.0807 | 2341 | 2341 | 1.0000 |
| 25 | 100 | 72 | 15 | 0.0784 | 2441 | 2441 | 1.0000 |
| 26 | 100 | 70 | 12 | 0.0900 | 2541 | 2541 | 1.0000 |
| 27 | 100 | 72 | 12 | 0.0982 | 2641 | 2641 | 1.0000 |
| 28 | 100 | 72 | 14 | 0.0829 | 2741 | 2741 | 1.0000 |
| 29 | 100 | 70 | 14 | 0.1100 | 2841 | 2841 | 1.0000 |
| 30 | 41 | 31 | 11 | 0.0941 | 2882 | 2882 | 1.0000 |
| 31 | 100 | 66 | 23 | 0.0649 | 2982 | 2982 | 1.0000 |
| 32 | 100 | 69 | 25 | 0.0505 | 3082 | 3082 | 1.0000 |
| 33 | 100 | 74 | 25 | 0.0535 | 3182 | 3182 | 1.0000 |
| 34 | 100 | 65 | 19 | 0.0591 | 3282 | 3282 | 1.0000 |
| 35 | 100 | 77 | 20 | 0.0879 | 3382 | 3382 | 1.0000 |
| 36 | 100 | 70 | 19 | 0.0671 | 3482 | 3482 | 1.0000 |
| 37 | 100 | 73 | 19 | 0.0820 | 3582 | 3582 | 1.0000 |
| 38 | 100 | 79 | 19 | 0.0915 | 3682 | 3682 | 1.0000 |
| 39 | 100 | 68 | 17 | 0.0958 | 3782 | 3782 | 1.0000 |
| 40 | 100 | 72 | 15 | 0.0935 | 3882 | 3882 | 1.0000 |
| 41 | 100 | 70 | 15 | 0.1005 | 3982 | 3982 | 1.0000 |
| 42 | 100 | 72 | 14 | 0.1046 | 4082 | 4082 | 1.0000 |
| 43 | 100 | 72 | 14 | 0.0972 | 4182 | 4182 | 1.0000 |
| 44 | 100 | 70 | 15 | 0.1231 | 4282 | 4282 | 1.0000 |
| 45 | 41 | 31 | 9 | 0.1119 | 4323 | 4323 | 1.0000 |

## Three-Way Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random | Hebbian | Error-Driven | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 64 | 10 | 0.000 | 0.044 | 0.089 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 64 | 30 | 0.000 | 0.022 | 0.067 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 10 | 0.022 | 0.067 | 0.067 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 30 | 0.022 | 0.022 | 0.022 | 0.156 |
| masked_token | generalization | 4 | 1 | end | 64 | 10 | 0.060 | 0.080 | 0.040 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 64 | 30 | 0.040 | 0.020 | 0.000 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 10 | 0.020 | 0.060 | 0.040 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 30 | 0.040 | 0.060 | 0.080 | 0.140 |
| masked_token | generalization | 8 | 1 | center | 64 | 10 | 0.057 | 0.075 | 0.038 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 64 | 30 | 0.019 | 0.038 | 0.019 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 10 | 0.019 | 0.057 | 0.019 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 30 | 0.019 | 0.019 | 0.019 | 0.226 |
| masked_token | generalization | 8 | 1 | end | 64 | 10 | 0.045 | 0.000 | 0.000 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 64 | 30 | 0.023 | 0.000 | 0.023 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 10 | 0.000 | 0.159 | 0.114 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 30 | 0.023 | 0.023 | 0.000 | 0.273 |
| masked_token | memorization | 4 | 1 | center | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 256 | 10 | 0.809 | 0.191 | 0.447 | 0.234 |
| masked_token | memorization | 4 | 1 | center | 256 | 30 | 0.979 | 0.979 | 0.979 | 0.234 |
| masked_token | memorization | 4 | 1 | end | 64 | 10 | 0.981 | 0.981 | 0.981 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 64 | 30 | 0.981 | 0.981 | 0.981 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 256 | 10 | 0.929 | 0.119 | 0.333 | 0.167 |
| masked_token | memorization | 4 | 1 | end | 256 | 30 | 0.952 | 0.952 | 0.976 | 0.167 |
| masked_token | memorization | 8 | 1 | center | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.327 |
| masked_token | memorization | 8 | 1 | center | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.327 |
| masked_token | memorization | 8 | 1 | end | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.095 |
| masked_token | memorization | 8 | 1 | end | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.095 |
| next_token | generalization | 4 | - | - | 64 | 10 | 0.080 | 0.060 | 0.020 | 0.120 |
| next_token | generalization | 4 | - | - | 64 | 30 | 0.060 | 0.040 | 0.000 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 10 | 0.020 | 0.060 | 0.040 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 30 | 0.080 | 0.080 | 0.040 | 0.120 |
| next_token | generalization | 8 | - | - | 64 | 10 | 0.045 | 0.000 | 0.000 | 0.273 |
| next_token | generalization | 8 | - | - | 64 | 30 | 0.023 | 0.000 | 0.023 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 10 | 0.000 | 0.159 | 0.114 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 30 | 0.023 | 0.000 | 0.000 | 0.273 |
| next_token | memorization | 4 | - | - | 64 | 10 | 0.981 | 0.981 | 0.981 | 0.192 |
| next_token | memorization | 4 | - | - | 64 | 30 | 0.981 | 0.981 | 0.981 | 0.192 |
| next_token | memorization | 4 | - | - | 256 | 10 | 0.952 | 0.405 | 0.571 | 0.214 |
| next_token | memorization | 4 | - | - | 256 | 30 | 0.952 | 0.952 | 0.976 | 0.214 |
| next_token | memorization | 8 | - | - | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.116 |
| next_token | memorization | 8 | - | - | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.116 |
| next_token | memorization | 8 | - | - | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.071 |
| next_token | memorization | 8 | - | - | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.071 |

## Generalization Summary

- **masked_token**: random=0.0255 hebbian=0.0466 error_driven=0.0397 bigram=0.1987
- **next_token**: random=0.0414 hebbian=0.0499 error_driven=0.0295 bigram=0.1964
