# Phase 3c: Reconstruction-Loss Codebook Results

## Training Config

- consolidation_k: `100`
- corpus: `wikitext`
- device: `mps`
- dim: `4096`
- epochs: `5`
- eval_betas: `[10.0, 30.0]`
- eval_landscape_sizes: `[64, 256]`
- eval_mask_counts: `[1]`
- eval_mask_positions: `['center', 'end']`
- eval_window_sizes: `[4, 8]`
- lr_pull: `0.1`
- lr_push: `0.05`
- quality_threshold: `0.15`
- seed: `17`
- test_samples: `64`
- train_beta: `10.0`
- train_landscape_size: `256`
- train_probe_size: `2000`
- train_window_size: `8`
- vocab_size: `2050`

## Reconstruction Training Log

| Consolidation | Buffer | Pulled | Pushed | Mean Q | Total Pos | Total Fail | Fail Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 84 | 14 | 0.0377 | 198 | 100 | 0.5051 |
| 2 | 100 | 83 | 17 | 0.0365 | 406 | 200 | 0.4926 |
| 3 | 100 | 84 | 15 | 0.0355 | 627 | 300 | 0.4785 |
| 4 | 100 | 81 | 15 | 0.0386 | 853 | 400 | 0.4689 |
| 5 | 100 | 79 | 16 | 0.0383 | 1075 | 500 | 0.4651 |
| 6 | 100 | 88 | 13 | 0.0401 | 1300 | 600 | 0.4615 |
| 7 | 100 | 86 | 16 | 0.0432 | 1531 | 700 | 0.4572 |
| 8 | 3 | 3 | 3 | 0.0719 | 1536 | 703 | 0.4577 |
| 9 | 100 | 80 | 11 | 0.0482 | 1702 | 803 | 0.4718 |
| 10 | 100 | 80 | 10 | 0.0544 | 1862 | 903 | 0.4850 |
| 11 | 100 | 79 | 14 | 0.0494 | 2028 | 1003 | 0.4946 |
| 12 | 100 | 75 | 14 | 0.0559 | 2198 | 1103 | 0.5018 |
| 13 | 100 | 81 | 10 | 0.0601 | 2357 | 1203 | 0.5104 |
| 14 | 100 | 84 | 12 | 0.0510 | 2536 | 1303 | 0.5138 |
| 15 | 100 | 80 | 12 | 0.0667 | 2696 | 1403 | 0.5204 |
| 16 | 100 | 89 | 16 | 0.0556 | 2868 | 1503 | 0.5241 |
| 17 | 15 | 15 | 7 | 0.0646 | 2896 | 1518 | 0.5242 |
| 18 | 100 | 80 | 19 | 0.0621 | 3064 | 1618 | 0.5281 |
| 19 | 100 | 76 | 21 | 0.0700 | 3230 | 1718 | 0.5319 |
| 20 | 100 | 79 | 23 | 0.0725 | 3399 | 1818 | 0.5349 |
| 21 | 100 | 76 | 28 | 0.0631 | 3578 | 1918 | 0.5361 |
| 22 | 100 | 83 | 22 | 0.0764 | 3750 | 2018 | 0.5381 |
| 23 | 100 | 76 | 22 | 0.0810 | 3924 | 2118 | 0.5398 |
| 24 | 100 | 74 | 24 | 0.0806 | 4108 | 2218 | 0.5399 |
| 25 | 100 | 70 | 30 | 0.0893 | 4300 | 2318 | 0.5391 |
| 26 | 100 | 77 | 30 | 0.0872 | 4470 | 2418 | 0.5409 |
| 27 | 5 | 5 | 5 | 0.1187 | 4480 | 2423 | 0.5408 |
| 28 | 100 | 77 | 14 | 0.0696 | 4631 | 2523 | 0.5448 |
| 29 | 100 | 77 | 24 | 0.0796 | 4770 | 2623 | 0.5499 |
| 30 | 100 | 74 | 21 | 0.0808 | 4914 | 2723 | 0.5541 |
| 31 | 100 | 66 | 26 | 0.0819 | 5051 | 2823 | 0.5589 |
| 32 | 100 | 74 | 26 | 0.0825 | 5186 | 2923 | 0.5636 |
| 33 | 100 | 73 | 31 | 0.0886 | 5320 | 3023 | 0.5682 |
| 34 | 100 | 79 | 22 | 0.0890 | 5469 | 3123 | 0.5710 |
| 35 | 100 | 76 | 24 | 0.0866 | 5613 | 3223 | 0.5742 |
| 36 | 100 | 78 | 27 | 0.0843 | 5754 | 3323 | 0.5775 |
| 37 | 100 | 81 | 28 | 0.0860 | 5889 | 3423 | 0.5813 |
| 38 | 84 | 67 | 30 | 0.0800 | 6032 | 3507 | 0.5814 |
| 39 | 100 | 80 | 16 | 0.0731 | 6147 | 3607 | 0.5868 |
| 40 | 100 | 76 | 24 | 0.0771 | 6257 | 3707 | 0.5925 |
| 41 | 100 | 79 | 15 | 0.0815 | 6369 | 3807 | 0.5977 |
| 42 | 100 | 81 | 28 | 0.0784 | 6485 | 3907 | 0.6025 |
| 43 | 100 | 81 | 16 | 0.0757 | 6604 | 4007 | 0.6068 |
| 44 | 100 | 85 | 25 | 0.0783 | 6719 | 4107 | 0.6113 |
| 45 | 100 | 82 | 18 | 0.0738 | 6834 | 4207 | 0.6156 |
| 46 | 100 | 81 | 21 | 0.0754 | 6949 | 4307 | 0.6198 |
| 47 | 100 | 85 | 20 | 0.0814 | 7075 | 4407 | 0.6229 |
| 48 | 100 | 80 | 21 | 0.0839 | 7202 | 4507 | 0.6258 |
| 49 | 100 | 86 | 19 | 0.0810 | 7316 | 4607 | 0.6297 |
| 50 | 3 | 3 | 3 | 0.0288 | 7320 | 4610 | 0.6298 |

## Three-Way Comparison

| Objective | Retrieval | W | Mask | Pos | L | Beta | Random | Hebbian | Reconstruction | Bigram |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 64 | 10 | 0.000 | 0.022 | 0.044 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 64 | 30 | 0.000 | 0.022 | 0.133 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 10 | 0.000 | 0.067 | 0.089 | 0.156 |
| masked_token | generalization | 4 | 1 | center | 256 | 30 | 0.022 | 0.022 | 0.089 | 0.156 |
| masked_token | generalization | 4 | 1 | end | 64 | 10 | 0.100 | 0.080 | 0.060 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 64 | 30 | 0.080 | 0.040 | 0.040 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 10 | 0.060 | 0.040 | 0.100 | 0.140 |
| masked_token | generalization | 4 | 1 | end | 256 | 30 | 0.040 | 0.060 | 0.060 | 0.140 |
| masked_token | generalization | 8 | 1 | center | 64 | 10 | 0.038 | 0.038 | 0.057 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 64 | 30 | 0.038 | 0.000 | 0.057 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 10 | 0.038 | 0.038 | 0.094 | 0.226 |
| masked_token | generalization | 8 | 1 | center | 256 | 30 | 0.019 | 0.019 | 0.038 | 0.226 |
| masked_token | generalization | 8 | 1 | end | 64 | 10 | 0.068 | 0.000 | 0.091 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 64 | 30 | 0.045 | 0.000 | 0.023 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 10 | 0.000 | 0.159 | 0.136 | 0.273 |
| masked_token | generalization | 8 | 1 | end | 256 | 30 | 0.023 | 0.045 | 0.023 | 0.273 |
| masked_token | memorization | 4 | 1 | center | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.057 |
| masked_token | memorization | 4 | 1 | center | 256 | 10 | 0.809 | 0.191 | 0.340 | 0.234 |
| masked_token | memorization | 4 | 1 | center | 256 | 30 | 0.957 | 0.979 | 1.000 | 0.234 |
| masked_token | memorization | 4 | 1 | end | 64 | 10 | 0.981 | 0.981 | 0.981 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 64 | 30 | 0.981 | 0.981 | 0.981 | 0.192 |
| masked_token | memorization | 4 | 1 | end | 256 | 10 | 0.929 | 0.167 | 0.167 | 0.167 |
| masked_token | memorization | 4 | 1 | end | 256 | 30 | 0.952 | 0.976 | 0.976 | 0.167 |
| masked_token | memorization | 8 | 1 | center | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.340 |
| masked_token | memorization | 8 | 1 | center | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.327 |
| masked_token | memorization | 8 | 1 | center | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.327 |
| masked_token | memorization | 8 | 1 | end | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.140 |
| masked_token | memorization | 8 | 1 | end | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.095 |
| masked_token | memorization | 8 | 1 | end | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.095 |
| next_token | generalization | 4 | - | - | 64 | 10 | 0.100 | 0.060 | 0.040 | 0.120 |
| next_token | generalization | 4 | - | - | 64 | 30 | 0.080 | 0.040 | 0.040 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 10 | 0.040 | 0.060 | 0.100 | 0.120 |
| next_token | generalization | 4 | - | - | 256 | 30 | 0.080 | 0.080 | 0.040 | 0.120 |
| next_token | generalization | 8 | - | - | 64 | 10 | 0.068 | 0.000 | 0.068 | 0.273 |
| next_token | generalization | 8 | - | - | 64 | 30 | 0.045 | 0.000 | 0.023 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 10 | 0.000 | 0.159 | 0.136 | 0.273 |
| next_token | generalization | 8 | - | - | 256 | 30 | 0.000 | 0.023 | 0.023 | 0.273 |
| next_token | memorization | 4 | - | - | 64 | 10 | 0.981 | 0.981 | 0.981 | 0.192 |
| next_token | memorization | 4 | - | - | 64 | 30 | 0.981 | 0.981 | 0.981 | 0.192 |
| next_token | memorization | 4 | - | - | 256 | 10 | 0.952 | 0.405 | 0.548 | 0.214 |
| next_token | memorization | 4 | - | - | 256 | 30 | 0.952 | 0.952 | 0.976 | 0.214 |
| next_token | memorization | 8 | - | - | 64 | 10 | 1.000 | 1.000 | 1.000 | 0.116 |
| next_token | memorization | 8 | - | - | 64 | 30 | 1.000 | 1.000 | 1.000 | 0.116 |
| next_token | memorization | 8 | - | - | 256 | 10 | 1.000 | 1.000 | 1.000 | 0.071 |
| next_token | memorization | 8 | - | - | 256 | 30 | 1.000 | 1.000 | 1.000 | 0.071 |

## Generalization Summary

- **masked_token**: random=0.0357 hebbian=0.0408 reconstruction=0.0708 bigram=0.1987
- **next_token**: random=0.0517 hebbian=0.0527 reconstruction=0.0587 bigram=0.1964
