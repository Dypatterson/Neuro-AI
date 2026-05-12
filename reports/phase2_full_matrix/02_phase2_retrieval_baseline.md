# Phase 2 Retrieval Baseline

## Dataset Summary

- source: `wikitext`
- vocab size: `2050`
- train windows: `W4=439182, W8=219591, W16=109795`
- validation windows: `W4=46017, W8=23008, W16=11504`
- window sizes: `4, 8, 16`
- mask counts: `1, 2`
- mask positions: `center, edge, end`
- test samples: `64`
- landscape sizes: `64, 256, 1024`
- betas: `3, 10, 30, 100`
- device: `mps`
- mps: `True`
- dataset config: `wikitext-2-raw-v1`

## Condition Summaries

| Objective | Retrieval | W | Mask | Position | Landscape | Beta | Accuracy | 95% CI | Bigram | Cap err @0.5 | Meta-stable | Gap | Energy |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| masked_token | generalization | 4 | 1 | center | 64 | 3 | 0.022 | [0.004, 0.116] | 0.156 | 0.000 | 1.000 | 0.011 | -1.650 |
| masked_token | generalization | 4 | 1 | center | 64 | 10 | 0.000 | [0.000, 0.079] | 0.156 | 0.000 | 0.000 | 0.161 | -1.004 |
| masked_token | generalization | 4 | 1 | center | 64 | 30 | 0.022 | [0.004, 0.116] | 0.156 | 0.000 | 0.000 | 0.226 | -1.000 |
| masked_token | generalization | 4 | 1 | center | 64 | 100 | 0.067 | [0.023, 0.179] | 0.156 | 0.000 | 0.000 | 0.244 | -1.000 |
| masked_token | generalization | 4 | 1 | center | 256 | 3 | 0.089 | [0.035, 0.207] | 0.156 | 0.000 | 0.000 | 0.004 | -2.130 |
| masked_token | generalization | 4 | 1 | center | 256 | 10 | 0.022 | [0.004, 0.116] | 0.156 | 0.000 | 0.000 | 0.018 | -1.147 |
| masked_token | generalization | 4 | 1 | center | 256 | 30 | 0.022 | [0.004, 0.116] | 0.156 | 0.000 | 0.000 | 0.194 | -1.011 |
| masked_token | generalization | 4 | 1 | center | 256 | 100 | 0.022 | [0.004, 0.116] | 0.156 | 0.000 | 0.000 | 0.203 | -1.003 |
| masked_token | generalization | 4 | 1 | center | 1024 | 3 | 0.000 | [0.000, 0.079] | 0.156 | 0.000 | 1.000 | 0.000 | -2.579 |
| masked_token | generalization | 4 | 1 | center | 1024 | 10 | 0.000 | [0.000, 0.079] | 0.156 | 0.000 | 0.000 | 0.001 | -1.243 |
| masked_token | generalization | 4 | 1 | center | 1024 | 30 | 0.044 | [0.012, 0.148] | 0.156 | 0.000 | 0.000 | 0.136 | -1.021 |
| masked_token | generalization | 4 | 1 | center | 1024 | 100 | 0.044 | [0.012, 0.148] | 0.156 | 0.000 | 0.000 | 0.154 | -1.006 |
| masked_token | generalization | 4 | 1 | edge | 64 | 3 | 0.096 | [0.042, 0.206] | 0.212 | 0.000 | 1.000 | 0.003 | -1.650 |
| masked_token | generalization | 4 | 1 | edge | 64 | 10 | 0.058 | [0.020, 0.156] | 0.212 | 0.000 | 0.000 | 0.091 | -1.004 |
| masked_token | generalization | 4 | 1 | edge | 64 | 30 | 0.019 | [0.003, 0.101] | 0.212 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | generalization | 4 | 1 | edge | 64 | 100 | 0.038 | [0.011, 0.130] | 0.212 | 0.000 | 0.000 | 0.274 | -1.000 |
| masked_token | generalization | 4 | 1 | edge | 256 | 3 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.000 | 0.001 | -2.130 |
| masked_token | generalization | 4 | 1 | edge | 256 | 10 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.000 | 0.056 | -1.143 |
| masked_token | generalization | 4 | 1 | edge | 256 | 30 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.000 | 0.279 | -1.004 |
| masked_token | generalization | 4 | 1 | edge | 256 | 100 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.000 | 0.294 | -1.001 |
| masked_token | generalization | 4 | 1 | edge | 1024 | 3 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.769 | 0.001 | -2.579 |
| masked_token | generalization | 4 | 1 | edge | 1024 | 10 | 0.000 | [0.000, 0.069] | 0.212 | 0.000 | 0.000 | 0.024 | -1.240 |
| masked_token | generalization | 4 | 1 | edge | 1024 | 30 | 0.038 | [0.011, 0.130] | 0.212 | 0.000 | 0.000 | 0.188 | -1.013 |
| masked_token | generalization | 4 | 1 | edge | 1024 | 100 | 0.038 | [0.011, 0.130] | 0.212 | 0.000 | 0.000 | 0.228 | -1.002 |
| masked_token | generalization | 4 | 1 | end | 64 | 3 | 0.100 | [0.043, 0.214] | 0.140 | 0.000 | 1.000 | 0.076 | -1.650 |
| masked_token | generalization | 4 | 1 | end | 64 | 10 | 0.060 | [0.021, 0.162] | 0.140 | 0.000 | 0.000 | 0.228 | -1.004 |
| masked_token | generalization | 4 | 1 | end | 64 | 30 | 0.060 | [0.021, 0.162] | 0.140 | 0.000 | 0.000 | 0.295 | -1.000 |
| masked_token | generalization | 4 | 1 | end | 64 | 100 | 0.040 | [0.011, 0.135] | 0.140 | 0.000 | 0.000 | 0.355 | -1.000 |
| masked_token | generalization | 4 | 1 | end | 256 | 3 | 0.000 | [0.000, 0.071] | 0.140 | 0.000 | 0.000 | 0.009 | -2.130 |
| masked_token | generalization | 4 | 1 | end | 256 | 10 | 0.000 | [0.000, 0.071] | 0.140 | 0.000 | 0.000 | 0.054 | -1.145 |
| masked_token | generalization | 4 | 1 | end | 256 | 30 | 0.080 | [0.032, 0.188] | 0.140 | 0.000 | 0.000 | 0.223 | -1.005 |
| masked_token | generalization | 4 | 1 | end | 256 | 100 | 0.060 | [0.021, 0.162] | 0.140 | 0.000 | 0.000 | 0.264 | -1.001 |
| masked_token | generalization | 4 | 1 | end | 1024 | 3 | 0.100 | [0.043, 0.214] | 0.140 | 0.000 | 0.780 | 0.007 | -2.579 |
| masked_token | generalization | 4 | 1 | end | 1024 | 10 | 0.000 | [0.000, 0.071] | 0.140 | 0.000 | 0.000 | 0.011 | -1.240 |
| masked_token | generalization | 4 | 1 | end | 1024 | 30 | 0.040 | [0.011, 0.135] | 0.140 | 0.000 | 0.000 | 0.137 | -1.014 |
| masked_token | generalization | 4 | 1 | end | 1024 | 100 | 0.060 | [0.021, 0.162] | 0.140 | 0.000 | 0.000 | 0.243 | -1.002 |
| masked_token | generalization | 4 | 2 | center | 64 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 1.000 | 0.006 | -1.650 |
| masked_token | generalization | 4 | 2 | center | 64 | 10 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.043 | -1.004 |
| masked_token | generalization | 4 | 2 | center | 64 | 30 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.109 | -1.000 |
| masked_token | generalization | 4 | 2 | center | 64 | 100 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.157 | -1.000 |
| masked_token | generalization | 4 | 2 | center | 256 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.003 | -2.130 |
| masked_token | generalization | 4 | 2 | center | 256 | 10 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.014 | -1.157 |
| masked_token | generalization | 4 | 2 | center | 256 | 30 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.081 | -1.025 |
| masked_token | generalization | 4 | 2 | center | 256 | 100 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.092 | -1.007 |
| masked_token | generalization | 4 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 1.000 | 0.002 | -2.579 |
| masked_token | generalization | 4 | 2 | center | 1024 | 10 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.008 | -1.243 |
| masked_token | generalization | 4 | 2 | center | 1024 | 30 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.080 | -1.036 |
| masked_token | generalization | 4 | 2 | center | 1024 | 100 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 0.000 | 0.121 | -1.009 |
| masked_token | generalization | 4 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 1.000 | 0.002 | -1.650 |
| masked_token | generalization | 4 | 2 | edge | 64 | 10 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.054 | -1.004 |
| masked_token | generalization | 4 | 2 | edge | 64 | 30 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.177 | -1.000 |
| masked_token | generalization | 4 | 2 | edge | 64 | 100 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.240 | -1.000 |
| masked_token | generalization | 4 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.002 | -2.130 |
| masked_token | generalization | 4 | 2 | edge | 256 | 10 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.033 | -1.153 |
| masked_token | generalization | 4 | 2 | edge | 256 | 30 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.207 | -1.014 |
| masked_token | generalization | 4 | 2 | edge | 256 | 100 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.242 | -1.003 |
| masked_token | generalization | 4 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.675 | 0.002 | -2.579 |
| masked_token | generalization | 4 | 2 | edge | 1024 | 10 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.003 | -1.243 |
| masked_token | generalization | 4 | 2 | edge | 1024 | 30 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.127 | -1.020 |
| masked_token | generalization | 4 | 2 | edge | 1024 | 100 | 0.000 | [0.000, 0.088] | 0.000 | 0.000 | 0.000 | 0.133 | -1.004 |
| masked_token | generalization | 4 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 1.000 | 0.043 | -1.650 |
| masked_token | generalization | 4 | 2 | end | 64 | 10 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.144 | -1.004 |
| masked_token | generalization | 4 | 2 | end | 64 | 30 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.220 | -1.000 |
| masked_token | generalization | 4 | 2 | end | 64 | 100 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | generalization | 4 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.007 | -2.130 |
| masked_token | generalization | 4 | 2 | end | 256 | 10 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.020 | -1.153 |
| masked_token | generalization | 4 | 2 | end | 256 | 30 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.171 | -1.011 |
| masked_token | generalization | 4 | 2 | end | 256 | 100 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.199 | -1.002 |
| masked_token | generalization | 4 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 1.000 | 0.004 | -2.579 |
| masked_token | generalization | 4 | 2 | end | 1024 | 10 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.002 | -1.249 |
| masked_token | generalization | 4 | 2 | end | 1024 | 30 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.143 | -1.020 |
| masked_token | generalization | 4 | 2 | end | 1024 | 100 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 0.000 | 0.157 | -1.004 |
| masked_token | generalization | 8 | 1 | center | 64 | 3 | 0.113 | [0.053, 0.226] | 0.226 | 0.000 | 1.000 | 0.109 | -1.628 |
| masked_token | generalization | 8 | 1 | center | 64 | 10 | 0.075 | [0.030, 0.179] | 0.226 | 0.000 | 0.000 | 0.259 | -1.001 |
| masked_token | generalization | 8 | 1 | center | 64 | 30 | 0.057 | [0.019, 0.154] | 0.226 | 0.000 | 0.000 | 0.246 | -1.000 |
| masked_token | generalization | 8 | 1 | center | 64 | 100 | 0.038 | [0.010, 0.128] | 0.226 | 0.000 | 0.000 | 0.250 | -1.000 |
| masked_token | generalization | 8 | 1 | center | 256 | 3 | 0.113 | [0.053, 0.226] | 0.226 | 0.000 | 1.000 | 0.036 | -2.029 |
| masked_token | generalization | 8 | 1 | center | 256 | 10 | 0.038 | [0.010, 0.128] | 0.226 | 0.000 | 0.000 | 0.143 | -1.006 |
| masked_token | generalization | 8 | 1 | center | 256 | 30 | 0.019 | [0.003, 0.099] | 0.226 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | generalization | 8 | 1 | center | 256 | 100 | 0.019 | [0.003, 0.099] | 0.226 | 0.000 | 0.000 | 0.214 | -1.000 |
| masked_token | generalization | 8 | 1 | center | 1024 | 3 | 0.019 | [0.003, 0.099] | 0.226 | 0.000 | 1.000 | 0.002 | -2.496 |
| masked_token | generalization | 8 | 1 | center | 1024 | 10 | 0.057 | [0.019, 0.154] | 0.226 | 0.000 | 0.038 | 0.211 | -1.030 |
| masked_token | generalization | 8 | 1 | center | 1024 | 30 | 0.057 | [0.019, 0.154] | 0.226 | 0.000 | 0.000 | 0.176 | -1.000 |
| masked_token | generalization | 8 | 1 | center | 1024 | 100 | 0.057 | [0.019, 0.154] | 0.226 | 0.000 | 0.000 | 0.192 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 64 | 3 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 1.000 | 0.017 | -1.628 |
| masked_token | generalization | 8 | 1 | edge | 64 | 10 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 0.000 | 0.141 | -1.001 |
| masked_token | generalization | 8 | 1 | edge | 64 | 30 | 0.019 | [0.003, 0.099] | 0.170 | 0.000 | 0.000 | 0.193 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 64 | 100 | 0.019 | [0.003, 0.099] | 0.170 | 0.000 | 0.000 | 0.200 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 256 | 3 | 0.170 | [0.092, 0.292] | 0.170 | 0.000 | 1.000 | 0.012 | -2.029 |
| masked_token | generalization | 8 | 1 | edge | 256 | 10 | 0.019 | [0.003, 0.099] | 0.170 | 0.000 | 0.000 | 0.087 | -1.006 |
| masked_token | generalization | 8 | 1 | edge | 256 | 30 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 256 | 100 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 1024 | 3 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 1.000 | 0.011 | -2.496 |
| masked_token | generalization | 8 | 1 | edge | 1024 | 10 | 0.000 | [0.000, 0.068] | 0.170 | 0.000 | 0.000 | 0.029 | -1.031 |
| masked_token | generalization | 8 | 1 | edge | 1024 | 30 | 0.019 | [0.003, 0.099] | 0.170 | 0.000 | 0.000 | 0.135 | -1.000 |
| masked_token | generalization | 8 | 1 | edge | 1024 | 100 | 0.019 | [0.003, 0.099] | 0.170 | 0.000 | 0.000 | 0.170 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 64 | 3 | 0.205 | [0.112, 0.345] | 0.273 | 0.000 | 1.000 | 0.019 | -1.628 |
| masked_token | generalization | 8 | 1 | end | 64 | 10 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.171 | -1.001 |
| masked_token | generalization | 8 | 1 | end | 64 | 30 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.163 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 64 | 100 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.176 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 256 | 3 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 1.000 | 0.004 | -2.029 |
| masked_token | generalization | 8 | 1 | end | 256 | 10 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.088 | -1.006 |
| masked_token | generalization | 8 | 1 | end | 256 | 30 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.195 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 256 | 100 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.205 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 1024 | 3 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 1.000 | 0.001 | -2.496 |
| masked_token | generalization | 8 | 1 | end | 1024 | 10 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.017 | -1.030 |
| masked_token | generalization | 8 | 1 | end | 1024 | 30 | 0.023 | [0.004, 0.118] | 0.273 | 0.000 | 0.000 | 0.174 | -1.000 |
| masked_token | generalization | 8 | 1 | end | 1024 | 100 | 0.023 | [0.004, 0.118] | 0.273 | 0.000 | 0.000 | 0.204 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 64 | 3 | 0.026 | [0.005, 0.132] | 0.000 | 0.000 | 1.000 | 0.057 | -1.628 |
| masked_token | generalization | 8 | 2 | center | 64 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.205 | -1.001 |
| masked_token | generalization | 8 | 2 | center | 64 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.198 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 64 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.208 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 256 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.042 | -2.029 |
| masked_token | generalization | 8 | 2 | center | 256 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.196 | -1.006 |
| masked_token | generalization | 8 | 2 | center | 256 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 256 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.004 | -2.496 |
| masked_token | generalization | 8 | 2 | center | 1024 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.051 | 0.108 | -1.029 |
| masked_token | generalization | 8 | 2 | center | 1024 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.174 | -1.000 |
| masked_token | generalization | 8 | 2 | center | 1024 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.019 | -1.628 |
| masked_token | generalization | 8 | 2 | edge | 64 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.175 | -1.001 |
| masked_token | generalization | 8 | 2 | edge | 64 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 64 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.190 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.022 | -2.029 |
| masked_token | generalization | 8 | 2 | edge | 256 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.100 | -1.006 |
| masked_token | generalization | 8 | 2 | edge | 256 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.179 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 256 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.180 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.006 | -2.496 |
| masked_token | generalization | 8 | 2 | edge | 1024 | 10 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.064 | -1.032 |
| masked_token | generalization | 8 | 2 | edge | 1024 | 30 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.137 | -1.000 |
| masked_token | generalization | 8 | 2 | edge | 1024 | 100 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 1.000 | 0.026 | -1.628 |
| masked_token | generalization | 8 | 2 | end | 64 | 10 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.169 | -1.001 |
| masked_token | generalization | 8 | 2 | end | 64 | 30 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 64 | 100 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.195 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 1.000 | 0.004 | -2.029 |
| masked_token | generalization | 8 | 2 | end | 256 | 10 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.040 | -1.006 |
| masked_token | generalization | 8 | 2 | end | 256 | 30 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 256 | 100 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.183 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 1.000 | 0.002 | -2.496 |
| masked_token | generalization | 8 | 2 | end | 1024 | 10 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.033 | 0.042 | -1.029 |
| masked_token | generalization | 8 | 2 | end | 1024 | 30 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.138 | -1.000 |
| masked_token | generalization | 8 | 2 | end | 1024 | 100 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.173 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 64 | 3 | 0.042 | [0.012, 0.140] | 0.188 | 0.000 | 1.000 | 0.021 | -1.582 |
| masked_token | generalization | 16 | 1 | center | 64 | 10 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.073 | -1.001 |
| masked_token | generalization | 16 | 1 | center | 64 | 30 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.164 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 64 | 100 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.164 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 256 | 3 | 0.083 | [0.033, 0.196] | 0.188 | 1.000 | 1.000 | 0.020 | -2.002 |
| masked_token | generalization | 16 | 1 | center | 256 | 10 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.054 | -1.003 |
| masked_token | generalization | 16 | 1 | center | 256 | 30 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.128 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 256 | 100 | 0.000 | [0.000, 0.074] | 0.188 | 0.000 | 0.000 | 0.133 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 1024 | 3 | 0.000 | [0.000, 0.074] | 0.188 | 1.000 | 1.000 | 0.001 | -2.444 |
| masked_token | generalization | 16 | 1 | center | 1024 | 10 | 0.042 | [0.012, 0.140] | 0.188 | 0.021 | 0.125 | 0.057 | -0.994 |
| masked_token | generalization | 16 | 1 | center | 1024 | 30 | 0.042 | [0.012, 0.140] | 0.188 | 0.000 | 0.000 | 0.129 | -1.000 |
| masked_token | generalization | 16 | 1 | center | 1024 | 100 | 0.042 | [0.012, 0.140] | 0.188 | 0.000 | 0.000 | 0.147 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 64 | 3 | 0.000 | [0.000, 0.077] | 0.239 | 0.000 | 1.000 | 0.013 | -1.582 |
| masked_token | generalization | 16 | 1 | edge | 64 | 10 | 0.000 | [0.000, 0.077] | 0.239 | 0.000 | 0.000 | 0.027 | -1.001 |
| masked_token | generalization | 16 | 1 | edge | 64 | 30 | 0.022 | [0.004, 0.113] | 0.239 | 0.000 | 0.000 | 0.072 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 64 | 100 | 0.022 | [0.004, 0.113] | 0.239 | 0.000 | 0.000 | 0.100 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 256 | 3 | 0.043 | [0.012, 0.145] | 0.239 | 1.000 | 1.000 | 0.004 | -2.002 |
| masked_token | generalization | 16 | 1 | edge | 256 | 10 | 0.000 | [0.000, 0.077] | 0.239 | 0.000 | 0.000 | 0.165 | -1.003 |
| masked_token | generalization | 16 | 1 | edge | 256 | 30 | 0.043 | [0.012, 0.145] | 0.239 | 0.000 | 0.000 | 0.128 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 256 | 100 | 0.022 | [0.004, 0.113] | 0.239 | 0.000 | 0.000 | 0.131 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 1024 | 3 | 0.043 | [0.012, 0.145] | 0.239 | 1.000 | 1.000 | 0.010 | -2.444 |
| masked_token | generalization | 16 | 1 | edge | 1024 | 10 | 0.000 | [0.000, 0.077] | 0.239 | 0.022 | 0.065 | 0.027 | -1.003 |
| masked_token | generalization | 16 | 1 | edge | 1024 | 30 | 0.022 | [0.004, 0.113] | 0.239 | 0.000 | 0.000 | 0.140 | -1.000 |
| masked_token | generalization | 16 | 1 | edge | 1024 | 100 | 0.022 | [0.004, 0.113] | 0.239 | 0.000 | 0.000 | 0.162 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 64 | 3 | 0.091 | [0.036, 0.212] | 0.205 | 0.000 | 1.000 | 0.022 | -1.582 |
| masked_token | generalization | 16 | 1 | end | 64 | 10 | 0.000 | [0.000, 0.080] | 0.205 | 0.000 | 0.000 | 0.078 | -1.001 |
| masked_token | generalization | 16 | 1 | end | 64 | 30 | 0.023 | [0.004, 0.118] | 0.205 | 0.000 | 0.000 | 0.144 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 64 | 100 | 0.023 | [0.004, 0.118] | 0.205 | 0.000 | 0.000 | 0.143 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 256 | 3 | 0.091 | [0.036, 0.212] | 0.205 | 1.000 | 1.000 | 0.013 | -2.002 |
| masked_token | generalization | 16 | 1 | end | 256 | 10 | 0.023 | [0.004, 0.118] | 0.205 | 0.000 | 0.000 | 0.083 | -1.003 |
| masked_token | generalization | 16 | 1 | end | 256 | 30 | 0.000 | [0.000, 0.080] | 0.205 | 0.000 | 0.000 | 0.144 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 256 | 100 | 0.000 | [0.000, 0.080] | 0.205 | 0.000 | 0.000 | 0.152 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 1024 | 3 | 0.000 | [0.000, 0.080] | 0.205 | 1.000 | 1.000 | 0.005 | -2.444 |
| masked_token | generalization | 16 | 1 | end | 1024 | 10 | 0.000 | [0.000, 0.080] | 0.205 | 0.023 | 0.114 | 0.008 | -0.997 |
| masked_token | generalization | 16 | 1 | end | 1024 | 30 | 0.000 | [0.000, 0.080] | 0.205 | 0.000 | 0.000 | 0.139 | -1.000 |
| masked_token | generalization | 16 | 1 | end | 1024 | 100 | 0.000 | [0.000, 0.080] | 0.205 | 0.000 | 0.000 | 0.142 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 64 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 1.000 | 0.012 | -1.582 |
| masked_token | generalization | 16 | 2 | center | 64 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.064 | -1.001 |
| masked_token | generalization | 16 | 2 | center | 64 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.149 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 64 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.154 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 256 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.012 | -2.002 |
| masked_token | generalization | 16 | 2 | center | 256 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.057 | -1.003 |
| masked_token | generalization | 16 | 2 | center | 256 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.113 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 256 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.115 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.004 | -2.444 |
| masked_token | generalization | 16 | 2 | center | 1024 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.083 | 0.054 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 1024 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.134 | -1.000 |
| masked_token | generalization | 16 | 2 | center | 1024 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.136 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 1.000 | 0.008 | -1.582 |
| masked_token | generalization | 16 | 2 | edge | 64 | 10 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.039 | -1.001 |
| masked_token | generalization | 16 | 2 | edge | 64 | 30 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.096 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 64 | 100 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.105 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 1.000 | 1.000 | 0.008 | -2.002 |
| masked_token | generalization | 16 | 2 | edge | 256 | 10 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.112 | -1.003 |
| masked_token | generalization | 16 | 2 | edge | 256 | 30 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.132 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 256 | 100 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.145 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 1.000 | 1.000 | 0.005 | -2.444 |
| masked_token | generalization | 16 | 2 | edge | 1024 | 10 | 0.000 | [0.000, 0.110] | 0.000 | 0.032 | 0.129 | 0.024 | -0.994 |
| masked_token | generalization | 16 | 2 | edge | 1024 | 30 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.131 | -1.000 |
| masked_token | generalization | 16 | 2 | edge | 1024 | 100 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.157 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 1.000 | 0.039 | -1.582 |
| masked_token | generalization | 16 | 2 | end | 64 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.071 | -1.001 |
| masked_token | generalization | 16 | 2 | end | 64 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.126 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 64 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.130 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.027 | -2.002 |
| masked_token | generalization | 16 | 2 | end | 256 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.070 | -1.003 |
| masked_token | generalization | 16 | 2 | end | 256 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.141 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 256 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.149 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.003 | -2.444 |
| masked_token | generalization | 16 | 2 | end | 1024 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.056 | 0.091 | -1.004 |
| masked_token | generalization | 16 | 2 | end | 1024 | 30 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.138 | -1.000 |
| masked_token | generalization | 16 | 2 | end | 1024 | 100 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.150 | -1.000 |
| masked_token | memorization | 4 | 1 | center | 64 | 3 | 0.019 | [0.003, 0.099] | 0.057 | 0.000 | 1.000 | 0.011 | -1.650 |
| masked_token | memorization | 4 | 1 | center | 64 | 10 | 1.000 | [0.932, 1.000] | 0.057 | 0.000 | 0.000 | 0.410 | -1.001 |
| masked_token | memorization | 4 | 1 | center | 64 | 30 | 1.000 | [0.932, 1.000] | 0.057 | 0.000 | 0.000 | 0.411 | -1.000 |
| masked_token | memorization | 4 | 1 | center | 64 | 100 | 1.000 | [0.932, 1.000] | 0.057 | 0.000 | 0.000 | 0.411 | -1.000 |
| masked_token | memorization | 4 | 1 | center | 256 | 3 | 0.149 | [0.074, 0.277] | 0.234 | 0.000 | 0.000 | 0.004 | -2.130 |
| masked_token | memorization | 4 | 1 | center | 256 | 10 | 0.809 | [0.675, 0.896] | 0.234 | 0.000 | 0.000 | 0.343 | -1.031 |
| masked_token | memorization | 4 | 1 | center | 256 | 30 | 0.957 | [0.858, 0.988] | 0.234 | 0.000 | 0.000 | 0.405 | -1.000 |
| masked_token | memorization | 4 | 1 | center | 256 | 100 | 0.957 | [0.858, 0.988] | 0.234 | 0.000 | 0.000 | 0.405 | -1.000 |
| masked_token | memorization | 4 | 1 | center | 1024 | 3 | 0.000 | [0.000, 0.082] | 0.326 | 0.000 | 1.000 | 0.000 | -2.579 |
| masked_token | memorization | 4 | 1 | center | 1024 | 10 | 0.419 | [0.284, 0.567] | 0.326 | 0.000 | 0.000 | 0.173 | -1.149 |
| masked_token | memorization | 4 | 1 | center | 1024 | 30 | 0.930 | [0.814, 0.976] | 0.326 | 0.000 | 0.000 | 0.404 | -1.002 |
| masked_token | memorization | 4 | 1 | center | 1024 | 100 | 0.930 | [0.814, 0.976] | 0.326 | 0.000 | 0.000 | 0.404 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 64 | 3 | 0.043 | [0.012, 0.145] | 0.196 | 0.000 | 1.000 | 0.003 | -1.650 |
| masked_token | memorization | 4 | 1 | edge | 64 | 10 | 1.000 | [0.923, 1.000] | 0.196 | 0.000 | 0.000 | 0.416 | -1.001 |
| masked_token | memorization | 4 | 1 | edge | 64 | 30 | 1.000 | [0.923, 1.000] | 0.196 | 0.000 | 0.000 | 0.416 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 64 | 100 | 1.000 | [0.923, 1.000] | 0.196 | 0.000 | 0.000 | 0.416 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 256 | 3 | 0.000 | [0.000, 0.080] | 0.182 | 0.000 | 0.000 | 0.001 | -2.130 |
| masked_token | memorization | 4 | 1 | edge | 256 | 10 | 0.841 | [0.706, 0.921] | 0.182 | 0.000 | 0.000 | 0.372 | -1.019 |
| masked_token | memorization | 4 | 1 | edge | 256 | 30 | 0.955 | [0.849, 0.987] | 0.182 | 0.000 | 0.000 | 0.412 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 256 | 100 | 0.955 | [0.849, 0.987] | 0.182 | 0.000 | 0.000 | 0.412 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 1024 | 3 | 0.000 | [0.000, 0.074] | 0.229 | 0.000 | 0.792 | 0.001 | -2.579 |
| masked_token | memorization | 4 | 1 | edge | 1024 | 10 | 0.375 | [0.252, 0.516] | 0.229 | 0.000 | 0.000 | 0.164 | -1.152 |
| masked_token | memorization | 4 | 1 | edge | 1024 | 30 | 0.833 | [0.704, 0.913] | 0.229 | 0.000 | 0.000 | 0.369 | -1.000 |
| masked_token | memorization | 4 | 1 | edge | 1024 | 100 | 0.833 | [0.704, 0.913] | 0.229 | 0.000 | 0.000 | 0.369 | -1.000 |
| masked_token | memorization | 4 | 1 | end | 64 | 3 | 0.154 | [0.080, 0.275] | 0.192 | 0.000 | 1.000 | 0.076 | -1.650 |
| masked_token | memorization | 4 | 1 | end | 64 | 10 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.417 | -1.001 |
| masked_token | memorization | 4 | 1 | end | 64 | 30 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.417 | -1.000 |
| masked_token | memorization | 4 | 1 | end | 64 | 100 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.417 | -1.000 |
| masked_token | memorization | 4 | 1 | end | 256 | 3 | 0.000 | [0.000, 0.084] | 0.167 | 0.000 | 0.000 | 0.009 | -2.130 |
| masked_token | memorization | 4 | 1 | end | 256 | 10 | 0.929 | [0.810, 0.975] | 0.167 | 0.000 | 0.000 | 0.393 | -1.008 |
| masked_token | memorization | 4 | 1 | end | 256 | 30 | 0.952 | [0.842, 0.987] | 0.167 | 0.000 | 0.000 | 0.404 | -1.000 |
| masked_token | memorization | 4 | 1 | end | 256 | 100 | 0.952 | [0.842, 0.987] | 0.167 | 0.000 | 0.000 | 0.404 | -1.000 |
| masked_token | memorization | 4 | 1 | end | 1024 | 3 | 0.062 | [0.021, 0.168] | 0.104 | 0.000 | 0.729 | 0.007 | -2.579 |
| masked_token | memorization | 4 | 1 | end | 1024 | 10 | 0.396 | [0.270, 0.537] | 0.104 | 0.000 | 0.000 | 0.174 | -1.151 |
| masked_token | memorization | 4 | 1 | end | 1024 | 30 | 0.958 | [0.860, 0.988] | 0.104 | 0.000 | 0.000 | 0.408 | -1.001 |
| masked_token | memorization | 4 | 1 | end | 1024 | 100 | 0.958 | [0.860, 0.988] | 0.104 | 0.000 | 0.000 | 0.408 | -1.000 |
| masked_token | memorization | 4 | 2 | center | 64 | 3 | 0.000 | [0.000, 0.090] | 0.000 | 0.000 | 1.000 | 0.005 | -1.650 |
| masked_token | memorization | 4 | 2 | center | 64 | 10 | 0.897 | [0.764, 0.959] | 0.000 | 0.000 | 0.000 | 0.370 | -1.001 |
| masked_token | memorization | 4 | 2 | center | 64 | 30 | 0.897 | [0.764, 0.959] | 0.000 | 0.000 | 0.000 | 0.370 | -1.000 |
| masked_token | memorization | 4 | 2 | center | 64 | 100 | 0.897 | [0.764, 0.959] | 0.000 | 0.000 | 0.000 | 0.370 | -1.000 |
| masked_token | memorization | 4 | 2 | center | 256 | 3 | 0.053 | [0.015, 0.173] | 0.000 | 0.000 | 0.000 | 0.003 | -2.130 |
| masked_token | memorization | 4 | 2 | center | 256 | 10 | 0.263 | [0.150, 0.420] | 0.000 | 0.000 | 0.000 | 0.115 | -1.119 |
| masked_token | memorization | 4 | 2 | center | 256 | 30 | 0.895 | [0.759, 0.958] | 0.000 | 0.000 | 0.000 | 0.367 | -1.006 |
| masked_token | memorization | 4 | 2 | center | 256 | 100 | 0.868 | [0.727, 0.942] | 0.000 | 0.000 | 0.000 | 0.368 | -1.002 |
| masked_token | memorization | 4 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 1.000 | 0.002 | -2.579 |
| masked_token | memorization | 4 | 2 | center | 1024 | 10 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 0.000 | 0.002 | -1.241 |
| masked_token | memorization | 4 | 2 | center | 1024 | 30 | 0.806 | [0.637, 0.908] | 0.000 | 0.000 | 0.000 | 0.365 | -1.003 |
| masked_token | memorization | 4 | 2 | center | 1024 | 100 | 0.806 | [0.637, 0.908] | 0.000 | 0.000 | 0.000 | 0.371 | -1.001 |
| masked_token | memorization | 4 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 1.000 | 0.002 | -1.650 |
| masked_token | memorization | 4 | 2 | edge | 64 | 10 | 0.941 | [0.809, 0.984] | 0.000 | 0.000 | 0.000 | 0.388 | -1.001 |
| masked_token | memorization | 4 | 2 | edge | 64 | 30 | 0.941 | [0.809, 0.984] | 0.000 | 0.000 | 0.000 | 0.388 | -1.000 |
| masked_token | memorization | 4 | 2 | edge | 64 | 100 | 0.941 | [0.809, 0.984] | 0.000 | 0.000 | 0.000 | 0.388 | -1.000 |
| masked_token | memorization | 4 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.104] | 0.000 | 0.000 | 0.000 | 0.002 | -2.130 |
| masked_token | memorization | 4 | 2 | edge | 256 | 10 | 0.424 | [0.272, 0.592] | 0.000 | 0.000 | 0.000 | 0.175 | -1.097 |
| masked_token | memorization | 4 | 2 | edge | 256 | 30 | 0.848 | [0.691, 0.933] | 0.000 | 0.000 | 0.000 | 0.379 | -1.000 |
| masked_token | memorization | 4 | 2 | edge | 256 | 100 | 0.848 | [0.691, 0.933] | 0.000 | 0.000 | 0.000 | 0.379 | -1.000 |
| masked_token | memorization | 4 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.639 | 0.002 | -2.579 |
| masked_token | memorization | 4 | 2 | edge | 1024 | 10 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 0.000 | 0.003 | -1.247 |
| masked_token | memorization | 4 | 2 | edge | 1024 | 30 | 0.694 | [0.531, 0.820] | 0.000 | 0.000 | 0.000 | 0.322 | -1.004 |
| masked_token | memorization | 4 | 2 | edge | 1024 | 100 | 0.722 | [0.560, 0.842] | 0.000 | 0.000 | 0.000 | 0.328 | -1.001 |
| masked_token | memorization | 4 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.079] | 0.000 | 0.000 | 1.000 | 0.043 | -1.650 |
| masked_token | memorization | 4 | 2 | end | 64 | 10 | 0.911 | [0.793, 0.965] | 0.000 | 0.000 | 0.000 | 0.394 | -1.001 |
| masked_token | memorization | 4 | 2 | end | 64 | 30 | 0.911 | [0.793, 0.965] | 0.000 | 0.000 | 0.000 | 0.394 | -1.000 |
| masked_token | memorization | 4 | 2 | end | 64 | 100 | 0.911 | [0.793, 0.965] | 0.000 | 0.000 | 0.000 | 0.394 | -1.000 |
| masked_token | memorization | 4 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 0.000 | 0.007 | -2.130 |
| masked_token | memorization | 4 | 2 | end | 256 | 10 | 0.300 | [0.167, 0.479] | 0.000 | 0.000 | 0.000 | 0.126 | -1.102 |
| masked_token | memorization | 4 | 2 | end | 256 | 30 | 0.800 | [0.627, 0.905] | 0.000 | 0.000 | 0.000 | 0.344 | -1.005 |
| masked_token | memorization | 4 | 2 | end | 256 | 100 | 0.800 | [0.627, 0.905] | 0.000 | 0.000 | 0.000 | 0.357 | -1.000 |
| masked_token | memorization | 4 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.107] | 0.000 | 0.000 | 1.000 | 0.004 | -2.579 |
| masked_token | memorization | 4 | 2 | end | 1024 | 10 | 0.000 | [0.000, 0.107] | 0.000 | 0.000 | 0.000 | 0.003 | -1.247 |
| masked_token | memorization | 4 | 2 | end | 1024 | 30 | 0.688 | [0.514, 0.820] | 0.000 | 0.000 | 0.000 | 0.349 | -1.003 |
| masked_token | memorization | 4 | 2 | end | 1024 | 100 | 0.688 | [0.514, 0.820] | 0.000 | 0.000 | 0.000 | 0.355 | -1.001 |
| masked_token | memorization | 8 | 1 | center | 64 | 3 | 0.213 | [0.120, 0.349] | 0.340 | 0.000 | 1.000 | 0.109 | -1.628 |
| masked_token | memorization | 8 | 1 | center | 64 | 10 | 1.000 | [0.924, 1.000] | 0.340 | 0.000 | 0.000 | 0.281 | -1.001 |
| masked_token | memorization | 8 | 1 | center | 64 | 30 | 1.000 | [0.924, 1.000] | 0.340 | 0.000 | 0.000 | 0.282 | -1.000 |
| masked_token | memorization | 8 | 1 | center | 64 | 100 | 1.000 | [0.924, 1.000] | 0.340 | 0.000 | 0.000 | 0.282 | -1.000 |
| masked_token | memorization | 8 | 1 | center | 256 | 3 | 0.184 | [0.100, 0.314] | 0.327 | 0.000 | 1.000 | 0.036 | -2.029 |
| masked_token | memorization | 8 | 1 | center | 256 | 10 | 1.000 | [0.927, 1.000] | 0.327 | 0.000 | 0.000 | 0.279 | -1.002 |
| masked_token | memorization | 8 | 1 | center | 256 | 30 | 1.000 | [0.927, 1.000] | 0.327 | 0.000 | 0.000 | 0.280 | -1.000 |
| masked_token | memorization | 8 | 1 | center | 256 | 100 | 1.000 | [0.927, 1.000] | 0.327 | 0.000 | 0.000 | 0.280 | -1.000 |
| masked_token | memorization | 8 | 1 | center | 1024 | 3 | 0.000 | [0.000, 0.071] | 0.200 | 0.000 | 1.000 | 0.002 | -2.496 |
| masked_token | memorization | 8 | 1 | center | 1024 | 10 | 1.000 | [0.929, 1.000] | 0.200 | 0.000 | 0.000 | 0.276 | -1.011 |
| masked_token | memorization | 8 | 1 | center | 1024 | 30 | 1.000 | [0.929, 1.000] | 0.200 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 1 | center | 1024 | 100 | 1.000 | [0.929, 1.000] | 0.200 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 64 | 3 | 0.075 | [0.026, 0.199] | 0.250 | 0.000 | 1.000 | 0.017 | -1.628 |
| masked_token | memorization | 8 | 1 | edge | 64 | 10 | 1.000 | [0.912, 1.000] | 0.250 | 0.000 | 0.000 | 0.283 | -1.001 |
| masked_token | memorization | 8 | 1 | edge | 64 | 30 | 1.000 | [0.912, 1.000] | 0.250 | 0.000 | 0.000 | 0.283 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 64 | 100 | 1.000 | [0.912, 1.000] | 0.250 | 0.000 | 0.000 | 0.283 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 256 | 3 | 0.080 | [0.032, 0.188] | 0.160 | 0.000 | 1.000 | 0.012 | -2.029 |
| masked_token | memorization | 8 | 1 | edge | 256 | 10 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.280 | -1.002 |
| masked_token | memorization | 8 | 1 | edge | 256 | 30 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.281 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 256 | 100 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.281 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 1024 | 3 | 0.000 | [0.000, 0.092] | 0.184 | 0.000 | 1.000 | 0.011 | -2.496 |
| masked_token | memorization | 8 | 1 | edge | 1024 | 10 | 1.000 | [0.908, 1.000] | 0.184 | 0.000 | 0.000 | 0.280 | -1.010 |
| masked_token | memorization | 8 | 1 | edge | 1024 | 30 | 1.000 | [0.908, 1.000] | 0.184 | 0.000 | 0.000 | 0.282 | -1.000 |
| masked_token | memorization | 8 | 1 | edge | 1024 | 100 | 1.000 | [0.908, 1.000] | 0.184 | 0.000 | 0.000 | 0.282 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 64 | 3 | 0.140 | [0.066, 0.273] | 0.140 | 0.000 | 1.000 | 0.019 | -1.628 |
| masked_token | memorization | 8 | 1 | end | 64 | 10 | 1.000 | [0.918, 1.000] | 0.140 | 0.000 | 0.000 | 0.279 | -1.001 |
| masked_token | memorization | 8 | 1 | end | 64 | 30 | 1.000 | [0.918, 1.000] | 0.140 | 0.000 | 0.000 | 0.280 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 64 | 100 | 1.000 | [0.918, 1.000] | 0.140 | 0.000 | 0.000 | 0.280 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 256 | 3 | 0.024 | [0.004, 0.123] | 0.095 | 0.000 | 1.000 | 0.004 | -2.029 |
| masked_token | memorization | 8 | 1 | end | 256 | 10 | 1.000 | [0.916, 1.000] | 0.095 | 0.000 | 0.000 | 0.277 | -1.002 |
| masked_token | memorization | 8 | 1 | end | 256 | 30 | 1.000 | [0.916, 1.000] | 0.095 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 256 | 100 | 1.000 | [0.916, 1.000] | 0.095 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 1024 | 3 | 0.000 | [0.000, 0.073] | 0.163 | 0.000 | 1.000 | 0.001 | -2.496 |
| masked_token | memorization | 8 | 1 | end | 1024 | 10 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.276 | -1.011 |
| masked_token | memorization | 8 | 1 | end | 1024 | 30 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 1 | end | 1024 | 100 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 64 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 1.000 | 0.058 | -1.628 |
| masked_token | memorization | 8 | 2 | center | 64 | 10 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.001 |
| masked_token | memorization | 8 | 2 | center | 64 | 30 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 64 | 100 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 256 | 3 | 0.027 | [0.005, 0.138] | 0.000 | 0.000 | 1.000 | 0.042 | -2.029 |
| masked_token | memorization | 8 | 2 | center | 256 | 10 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.002 |
| masked_token | memorization | 8 | 2 | center | 256 | 30 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 256 | 100 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.104] | 0.000 | 0.000 | 1.000 | 0.004 | -2.496 |
| masked_token | memorization | 8 | 2 | center | 1024 | 10 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.274 | -1.010 |
| masked_token | memorization | 8 | 2 | center | 1024 | 30 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.276 | -1.000 |
| masked_token | memorization | 8 | 2 | center | 1024 | 100 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.276 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.121] | 0.000 | 0.000 | 1.000 | 0.019 | -1.628 |
| masked_token | memorization | 8 | 2 | edge | 64 | 10 | 1.000 | [0.879, 1.000] | 0.000 | 0.000 | 0.000 | 0.281 | -1.001 |
| masked_token | memorization | 8 | 2 | edge | 64 | 30 | 1.000 | [0.879, 1.000] | 0.000 | 0.000 | 0.000 | 0.281 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 64 | 100 | 1.000 | [0.879, 1.000] | 0.000 | 0.000 | 0.000 | 0.281 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.104] | 0.000 | 0.000 | 1.000 | 0.022 | -2.029 |
| masked_token | memorization | 8 | 2 | edge | 256 | 10 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.002 |
| masked_token | memorization | 8 | 2 | edge | 256 | 30 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 256 | 100 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.117] | 0.000 | 0.000 | 1.000 | 0.006 | -2.496 |
| masked_token | memorization | 8 | 2 | edge | 1024 | 10 | 1.000 | [0.883, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.009 |
| masked_token | memorization | 8 | 2 | edge | 1024 | 30 | 1.000 | [0.883, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | edge | 1024 | 100 | 1.000 | [0.883, 1.000] | 0.000 | 0.000 | 0.000 | 0.279 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.110] | 0.000 | 0.000 | 1.000 | 0.026 | -1.628 |
| masked_token | memorization | 8 | 2 | end | 64 | 10 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.277 | -1.001 |
| masked_token | memorization | 8 | 2 | end | 64 | 30 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 64 | 100 | 1.000 | [0.890, 1.000] | 0.000 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.114] | 0.000 | 0.000 | 1.000 | 0.004 | -2.029 |
| masked_token | memorization | 8 | 2 | end | 256 | 10 | 1.000 | [0.886, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.002 |
| masked_token | memorization | 8 | 2 | end | 256 | 30 | 1.000 | [0.886, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 256 | 100 | 1.000 | [0.886, 1.000] | 0.000 | 0.000 | 0.000 | 0.278 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 0.000 | 1.000 | 0.002 | -2.496 |
| masked_token | memorization | 8 | 2 | end | 1024 | 10 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.275 | -1.009 |
| masked_token | memorization | 8 | 2 | end | 1024 | 30 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 8 | 2 | end | 1024 | 100 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.277 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 64 | 3 | 0.085 | [0.034, 0.199] | 0.170 | 0.000 | 1.000 | 0.021 | -1.582 |
| masked_token | memorization | 16 | 1 | center | 64 | 10 | 1.000 | [0.924, 1.000] | 0.170 | 0.000 | 0.000 | 0.182 | -1.001 |
| masked_token | memorization | 16 | 1 | center | 64 | 30 | 1.000 | [0.924, 1.000] | 0.170 | 0.000 | 0.000 | 0.182 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 64 | 100 | 1.000 | [0.924, 1.000] | 0.170 | 0.000 | 0.000 | 0.182 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 256 | 3 | 0.100 | [0.043, 0.214] | 0.160 | 1.000 | 1.000 | 0.020 | -2.002 |
| masked_token | memorization | 16 | 1 | center | 256 | 10 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.185 | -1.002 |
| masked_token | memorization | 16 | 1 | center | 256 | 30 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 256 | 100 | 1.000 | [0.929, 1.000] | 0.160 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 1024 | 3 | 0.000 | [0.000, 0.077] | 0.217 | 1.000 | 1.000 | 0.001 | -2.444 |
| masked_token | memorization | 16 | 1 | center | 1024 | 10 | 1.000 | [0.923, 1.000] | 0.217 | 0.000 | 0.000 | 0.186 | -1.008 |
| masked_token | memorization | 16 | 1 | center | 1024 | 30 | 1.000 | [0.923, 1.000] | 0.217 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | memorization | 16 | 1 | center | 1024 | 100 | 1.000 | [0.923, 1.000] | 0.217 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 64 | 3 | 0.026 | [0.005, 0.132] | 0.231 | 0.000 | 1.000 | 0.013 | -1.582 |
| masked_token | memorization | 16 | 1 | edge | 64 | 10 | 1.000 | [0.910, 1.000] | 0.231 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 64 | 30 | 1.000 | [0.910, 1.000] | 0.231 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 64 | 100 | 1.000 | [0.910, 1.000] | 0.231 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 256 | 3 | 0.039 | [0.011, 0.132] | 0.118 | 1.000 | 1.000 | 0.004 | -2.002 |
| masked_token | memorization | 16 | 1 | edge | 256 | 10 | 1.000 | [0.930, 1.000] | 0.118 | 0.000 | 0.000 | 0.184 | -1.002 |
| masked_token | memorization | 16 | 1 | edge | 256 | 30 | 1.000 | [0.930, 1.000] | 0.118 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 256 | 100 | 1.000 | [0.930, 1.000] | 0.118 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 1024 | 3 | 0.020 | [0.004, 0.107] | 0.102 | 1.000 | 1.000 | 0.010 | -2.444 |
| masked_token | memorization | 16 | 1 | edge | 1024 | 10 | 1.000 | [0.927, 1.000] | 0.102 | 0.000 | 0.000 | 0.182 | -1.008 |
| masked_token | memorization | 16 | 1 | edge | 1024 | 30 | 1.000 | [0.927, 1.000] | 0.102 | 0.000 | 0.000 | 0.183 | -1.000 |
| masked_token | memorization | 16 | 1 | edge | 1024 | 100 | 1.000 | [0.927, 1.000] | 0.102 | 0.000 | 0.000 | 0.183 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 64 | 3 | 0.096 | [0.042, 0.206] | 0.231 | 0.000 | 1.000 | 0.021 | -1.582 |
| masked_token | memorization | 16 | 1 | end | 64 | 10 | 1.000 | [0.931, 1.000] | 0.231 | 0.000 | 0.000 | 0.186 | -1.001 |
| masked_token | memorization | 16 | 1 | end | 64 | 30 | 1.000 | [0.931, 1.000] | 0.231 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 64 | 100 | 1.000 | [0.931, 1.000] | 0.231 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 256 | 3 | 0.058 | [0.020, 0.156] | 0.173 | 1.000 | 1.000 | 0.013 | -2.002 |
| masked_token | memorization | 16 | 1 | end | 256 | 10 | 1.000 | [0.931, 1.000] | 0.173 | 0.000 | 0.000 | 0.186 | -1.002 |
| masked_token | memorization | 16 | 1 | end | 256 | 30 | 1.000 | [0.931, 1.000] | 0.173 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 256 | 100 | 1.000 | [0.931, 1.000] | 0.173 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 1024 | 3 | 0.000 | [0.000, 0.079] | 0.267 | 1.000 | 1.000 | 0.005 | -2.444 |
| masked_token | memorization | 16 | 1 | end | 1024 | 10 | 1.000 | [0.921, 1.000] | 0.267 | 0.000 | 0.000 | 0.189 | -1.008 |
| masked_token | memorization | 16 | 1 | end | 1024 | 30 | 1.000 | [0.921, 1.000] | 0.267 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | memorization | 16 | 1 | end | 1024 | 100 | 1.000 | [0.921, 1.000] | 0.267 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 64 | 3 | 0.000 | [0.000, 0.099] | 0.000 | 0.000 | 1.000 | 0.012 | -1.582 |
| masked_token | memorization | 16 | 2 | center | 64 | 10 | 1.000 | [0.901, 1.000] | 0.000 | 0.000 | 0.000 | 0.182 | -1.001 |
| masked_token | memorization | 16 | 2 | center | 64 | 30 | 1.000 | [0.901, 1.000] | 0.000 | 0.000 | 0.000 | 0.182 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 64 | 100 | 1.000 | [0.901, 1.000] | 0.000 | 0.000 | 0.000 | 0.182 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 256 | 3 | 0.000 | [0.000, 0.094] | 0.000 | 1.000 | 1.000 | 0.012 | -2.002 |
| masked_token | memorization | 16 | 2 | center | 256 | 10 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.184 | -1.002 |
| masked_token | memorization | 16 | 2 | center | 256 | 30 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 256 | 100 | 1.000 | [0.906, 1.000] | 0.000 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 1024 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.004 | -2.444 |
| masked_token | memorization | 16 | 2 | center | 1024 | 10 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.008 |
| masked_token | memorization | 16 | 2 | center | 1024 | 30 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | memorization | 16 | 2 | center | 1024 | 100 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.187 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 64 | 3 | 0.000 | [0.000, 0.129] | 0.000 | 0.000 | 1.000 | 0.008 | -1.582 |
| masked_token | memorization | 16 | 2 | edge | 64 | 10 | 1.000 | [0.871, 1.000] | 0.000 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 64 | 30 | 1.000 | [0.871, 1.000] | 0.000 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 64 | 100 | 1.000 | [0.871, 1.000] | 0.000 | 0.000 | 0.000 | 0.184 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 256 | 3 | 0.000 | [0.000, 0.092] | 0.000 | 1.000 | 1.000 | 0.008 | -2.002 |
| masked_token | memorization | 16 | 2 | edge | 256 | 10 | 1.000 | [0.908, 1.000] | 0.000 | 0.000 | 0.000 | 0.185 | -1.002 |
| masked_token | memorization | 16 | 2 | edge | 256 | 30 | 1.000 | [0.908, 1.000] | 0.000 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 256 | 100 | 1.000 | [0.908, 1.000] | 0.000 | 0.000 | 0.000 | 0.185 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 1024 | 3 | 0.000 | [0.000, 0.102] | 0.000 | 1.000 | 1.000 | 0.005 | -2.444 |
| masked_token | memorization | 16 | 2 | edge | 1024 | 10 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.008 |
| masked_token | memorization | 16 | 2 | edge | 1024 | 30 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 2 | edge | 1024 | 100 | 1.000 | [0.898, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 64 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 0.000 | 1.000 | 0.039 | -1.582 |
| masked_token | memorization | 16 | 2 | end | 64 | 10 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.001 |
| masked_token | memorization | 16 | 2 | end | 64 | 30 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 64 | 100 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.186 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 256 | 3 | 0.000 | [0.000, 0.096] | 0.000 | 1.000 | 1.000 | 0.027 | -2.002 |
| masked_token | memorization | 16 | 2 | end | 256 | 10 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.189 | -1.002 |
| masked_token | memorization | 16 | 2 | end | 256 | 30 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 256 | 100 | 1.000 | [0.904, 1.000] | 0.000 | 0.000 | 0.000 | 0.189 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 1024 | 3 | 0.000 | [0.000, 0.104] | 0.000 | 1.000 | 1.000 | 0.003 | -2.444 |
| masked_token | memorization | 16 | 2 | end | 1024 | 10 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.190 | -1.008 |
| masked_token | memorization | 16 | 2 | end | 1024 | 30 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.191 | -1.000 |
| masked_token | memorization | 16 | 2 | end | 1024 | 100 | 1.000 | [0.896, 1.000] | 0.000 | 0.000 | 0.000 | 0.191 | -1.000 |
| next_token | generalization | 4 | - | - | 64 | 3 | 0.100 | [0.043, 0.214] | 0.120 | 0.000 | 1.000 | 0.076 | -1.650 |
| next_token | generalization | 4 | - | - | 64 | 10 | 0.040 | [0.011, 0.135] | 0.120 | 0.000 | 0.000 | 0.263 | -1.003 |
| next_token | generalization | 4 | - | - | 64 | 30 | 0.020 | [0.004, 0.105] | 0.120 | 0.000 | 0.000 | 0.283 | -1.000 |
| next_token | generalization | 4 | - | - | 64 | 100 | 0.040 | [0.011, 0.135] | 0.120 | 0.000 | 0.000 | 0.342 | -1.000 |
| next_token | generalization | 4 | - | - | 256 | 3 | 0.000 | [0.000, 0.071] | 0.120 | 0.000 | 0.000 | 0.009 | -2.130 |
| next_token | generalization | 4 | - | - | 256 | 10 | 0.020 | [0.004, 0.105] | 0.120 | 0.000 | 0.000 | 0.113 | -1.106 |
| next_token | generalization | 4 | - | - | 256 | 30 | 0.080 | [0.032, 0.188] | 0.120 | 0.000 | 0.000 | 0.249 | -1.004 |
| next_token | generalization | 4 | - | - | 256 | 100 | 0.060 | [0.021, 0.162] | 0.120 | 0.000 | 0.000 | 0.314 | -1.001 |
| next_token | generalization | 4 | - | - | 1024 | 3 | 0.100 | [0.043, 0.214] | 0.120 | 0.000 | 0.740 | 0.007 | -2.579 |
| next_token | generalization | 4 | - | - | 1024 | 10 | 0.000 | [0.000, 0.071] | 0.120 | 0.000 | 0.040 | 0.011 | -1.237 |
| next_token | generalization | 4 | - | - | 1024 | 30 | 0.040 | [0.011, 0.135] | 0.120 | 0.000 | 0.000 | 0.155 | -1.014 |
| next_token | generalization | 4 | - | - | 1024 | 100 | 0.040 | [0.011, 0.135] | 0.120 | 0.000 | 0.000 | 0.229 | -1.003 |
| next_token | generalization | 8 | - | - | 64 | 3 | 0.205 | [0.112, 0.345] | 0.273 | 0.000 | 1.000 | 0.019 | -1.628 |
| next_token | generalization | 8 | - | - | 64 | 10 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.157 | -1.001 |
| next_token | generalization | 8 | - | - | 64 | 30 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.164 | -1.000 |
| next_token | generalization | 8 | - | - | 64 | 100 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.183 | -1.000 |
| next_token | generalization | 8 | - | - | 256 | 3 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 1.000 | 0.004 | -2.029 |
| next_token | generalization | 8 | - | - | 256 | 10 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.101 | -1.006 |
| next_token | generalization | 8 | - | - | 256 | 30 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.200 | -1.000 |
| next_token | generalization | 8 | - | - | 256 | 100 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.000 | 0.192 | -1.000 |
| next_token | generalization | 8 | - | - | 1024 | 3 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 1.000 | 0.001 | -2.496 |
| next_token | generalization | 8 | - | - | 1024 | 10 | 0.000 | [0.000, 0.080] | 0.273 | 0.000 | 0.045 | 0.017 | -1.027 |
| next_token | generalization | 8 | - | - | 1024 | 30 | 0.023 | [0.004, 0.118] | 0.273 | 0.000 | 0.000 | 0.174 | -1.000 |
| next_token | generalization | 8 | - | - | 1024 | 100 | 0.045 | [0.013, 0.151] | 0.273 | 0.000 | 0.000 | 0.200 | -1.000 |
| next_token | generalization | 16 | - | - | 64 | 3 | 0.091 | [0.036, 0.212] | 0.227 | 0.000 | 1.000 | 0.022 | -1.582 |
| next_token | generalization | 16 | - | - | 64 | 10 | 0.000 | [0.000, 0.080] | 0.227 | 0.000 | 0.000 | 0.073 | -1.001 |
| next_token | generalization | 16 | - | - | 64 | 30 | 0.023 | [0.004, 0.118] | 0.227 | 0.000 | 0.000 | 0.144 | -1.000 |
| next_token | generalization | 16 | - | - | 64 | 100 | 0.023 | [0.004, 0.118] | 0.227 | 0.000 | 0.000 | 0.143 | -1.000 |
| next_token | generalization | 16 | - | - | 256 | 3 | 0.091 | [0.036, 0.212] | 0.227 | 1.000 | 1.000 | 0.013 | -2.002 |
| next_token | generalization | 16 | - | - | 256 | 10 | 0.023 | [0.004, 0.118] | 0.227 | 0.000 | 0.000 | 0.088 | -1.003 |
| next_token | generalization | 16 | - | - | 256 | 30 | 0.000 | [0.000, 0.080] | 0.227 | 0.000 | 0.000 | 0.148 | -1.000 |
| next_token | generalization | 16 | - | - | 256 | 100 | 0.000 | [0.000, 0.080] | 0.227 | 0.000 | 0.000 | 0.150 | -1.000 |
| next_token | generalization | 16 | - | - | 1024 | 3 | 0.000 | [0.000, 0.080] | 0.227 | 1.000 | 1.000 | 0.005 | -2.444 |
| next_token | generalization | 16 | - | - | 1024 | 10 | 0.000 | [0.000, 0.080] | 0.227 | 0.023 | 0.114 | 0.008 | -0.996 |
| next_token | generalization | 16 | - | - | 1024 | 30 | 0.000 | [0.000, 0.080] | 0.227 | 0.000 | 0.000 | 0.148 | -1.000 |
| next_token | generalization | 16 | - | - | 1024 | 100 | 0.000 | [0.000, 0.080] | 0.227 | 0.000 | 0.000 | 0.153 | -1.000 |
| next_token | memorization | 4 | - | - | 64 | 3 | 0.154 | [0.080, 0.275] | 0.192 | 0.000 | 1.000 | 0.076 | -1.650 |
| next_token | memorization | 4 | - | - | 64 | 10 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.416 | -1.001 |
| next_token | memorization | 4 | - | - | 64 | 30 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.417 | -1.000 |
| next_token | memorization | 4 | - | - | 64 | 100 | 0.981 | [0.899, 0.997] | 0.192 | 0.000 | 0.000 | 0.417 | -1.000 |
| next_token | memorization | 4 | - | - | 256 | 3 | 0.000 | [0.000, 0.084] | 0.214 | 0.000 | 0.000 | 0.009 | -2.130 |
| next_token | memorization | 4 | - | - | 256 | 10 | 0.952 | [0.842, 0.987] | 0.214 | 0.000 | 0.000 | 0.402 | -1.005 |
| next_token | memorization | 4 | - | - | 256 | 30 | 0.952 | [0.842, 0.987] | 0.214 | 0.000 | 0.000 | 0.403 | -1.000 |
| next_token | memorization | 4 | - | - | 256 | 100 | 0.952 | [0.842, 0.987] | 0.214 | 0.000 | 0.000 | 0.403 | -1.000 |
| next_token | memorization | 4 | - | - | 1024 | 3 | 0.062 | [0.021, 0.168] | 0.167 | 0.000 | 0.708 | 0.007 | -2.579 |
| next_token | memorization | 4 | - | - | 1024 | 10 | 0.667 | [0.525, 0.783] | 0.167 | 0.000 | 0.021 | 0.283 | -1.086 |
| next_token | memorization | 4 | - | - | 1024 | 30 | 0.958 | [0.860, 0.988] | 0.167 | 0.000 | 0.000 | 0.408 | -1.001 |
| next_token | memorization | 4 | - | - | 1024 | 100 | 0.958 | [0.860, 0.988] | 0.167 | 0.000 | 0.000 | 0.408 | -1.000 |
| next_token | memorization | 8 | - | - | 64 | 3 | 0.140 | [0.066, 0.273] | 0.116 | 0.000 | 1.000 | 0.020 | -1.628 |
| next_token | memorization | 8 | - | - | 64 | 10 | 1.000 | [0.918, 1.000] | 0.116 | 0.000 | 0.000 | 0.279 | -1.001 |
| next_token | memorization | 8 | - | - | 64 | 30 | 1.000 | [0.918, 1.000] | 0.116 | 0.000 | 0.000 | 0.280 | -1.000 |
| next_token | memorization | 8 | - | - | 64 | 100 | 1.000 | [0.918, 1.000] | 0.116 | 0.000 | 0.000 | 0.280 | -1.000 |
| next_token | memorization | 8 | - | - | 256 | 3 | 0.024 | [0.004, 0.123] | 0.071 | 0.000 | 1.000 | 0.004 | -2.029 |
| next_token | memorization | 8 | - | - | 256 | 10 | 1.000 | [0.916, 1.000] | 0.071 | 0.000 | 0.000 | 0.277 | -1.002 |
| next_token | memorization | 8 | - | - | 256 | 30 | 1.000 | [0.916, 1.000] | 0.071 | 0.000 | 0.000 | 0.277 | -1.000 |
| next_token | memorization | 8 | - | - | 256 | 100 | 1.000 | [0.916, 1.000] | 0.071 | 0.000 | 0.000 | 0.277 | -1.000 |
| next_token | memorization | 8 | - | - | 1024 | 3 | 0.000 | [0.000, 0.073] | 0.163 | 0.000 | 1.000 | 0.001 | -2.496 |
| next_token | memorization | 8 | - | - | 1024 | 10 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.276 | -1.011 |
| next_token | memorization | 8 | - | - | 1024 | 30 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.278 | -1.000 |
| next_token | memorization | 8 | - | - | 1024 | 100 | 1.000 | [0.927, 1.000] | 0.163 | 0.000 | 0.000 | 0.278 | -1.000 |
| next_token | memorization | 16 | - | - | 64 | 3 | 0.096 | [0.042, 0.206] | 0.250 | 0.000 | 1.000 | 0.021 | -1.582 |
| next_token | memorization | 16 | - | - | 64 | 10 | 1.000 | [0.931, 1.000] | 0.250 | 0.000 | 0.000 | 0.186 | -1.001 |
| next_token | memorization | 16 | - | - | 64 | 30 | 1.000 | [0.931, 1.000] | 0.250 | 0.000 | 0.000 | 0.186 | -1.000 |
| next_token | memorization | 16 | - | - | 64 | 100 | 1.000 | [0.931, 1.000] | 0.250 | 0.000 | 0.000 | 0.186 | -1.000 |
| next_token | memorization | 16 | - | - | 256 | 3 | 0.058 | [0.020, 0.156] | 0.192 | 1.000 | 1.000 | 0.013 | -2.002 |
| next_token | memorization | 16 | - | - | 256 | 10 | 1.000 | [0.931, 1.000] | 0.192 | 0.000 | 0.000 | 0.186 | -1.002 |
| next_token | memorization | 16 | - | - | 256 | 30 | 1.000 | [0.931, 1.000] | 0.192 | 0.000 | 0.000 | 0.186 | -1.000 |
| next_token | memorization | 16 | - | - | 256 | 100 | 1.000 | [0.931, 1.000] | 0.192 | 0.000 | 0.000 | 0.186 | -1.000 |
| next_token | memorization | 16 | - | - | 1024 | 3 | 0.000 | [0.000, 0.079] | 0.244 | 1.000 | 1.000 | 0.005 | -2.444 |
| next_token | memorization | 16 | - | - | 1024 | 10 | 1.000 | [0.921, 1.000] | 0.244 | 0.000 | 0.000 | 0.189 | -1.008 |
| next_token | memorization | 16 | - | - | 1024 | 30 | 1.000 | [0.921, 1.000] | 0.244 | 0.000 | 0.000 | 0.189 | -1.000 |
| next_token | memorization | 16 | - | - | 1024 | 100 | 1.000 | [0.921, 1.000] | 0.244 | 0.000 | 0.000 | 0.189 | -1.000 |

## Best Generalization Rows

| Objective | W | Mask | Position | Landscape | Beta | Accuracy | Bigram | Gap |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| masked_token | 8 | 1 | end | 64 | 3 | 0.205 | 0.273 | 0.019 |
| next_token | 8 | - | - | 64 | 3 | 0.205 | 0.273 | 0.019 |

## Frequency Buckets

### masked_token:generalization:W4:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.022

### masked_token:generalization:W4:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 0.022

### masked_token:generalization:W4:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 0.067

### masked_token:generalization:W4:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.089

### masked_token:generalization:W4:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 0.022

### masked_token:generalization:W4:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 0.022

### masked_token:generalization:W4:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 0.022

### masked_token:generalization:W4:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.044

### masked_token:generalization:W4:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.044

### masked_token:generalization:W4:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.096

### masked_token:generalization:W4:M1:Pedge:L64:beta10
- `q1_most_frequent`: 0.058

### masked_token:generalization:W4:M1:Pedge:L64:beta30
- `q1_most_frequent`: 0.019

### masked_token:generalization:W4:M1:Pedge:L64:beta100
- `q1_most_frequent`: 0.038

### masked_token:generalization:W4:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 0.038

### masked_token:generalization:W4:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 0.038

### masked_token:generalization:W4:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.100

### masked_token:generalization:W4:M1:Pend:L64:beta10
- `q1_most_frequent`: 0.060

### masked_token:generalization:W4:M1:Pend:L64:beta30
- `q1_most_frequent`: 0.060

### masked_token:generalization:W4:M1:Pend:L64:beta100
- `q1_most_frequent`: 0.040

### masked_token:generalization:W4:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pend:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pend:L256:beta30
- `q1_most_frequent`: 0.080

### masked_token:generalization:W4:M1:Pend:L256:beta100
- `q1_most_frequent`: 0.060

### masked_token:generalization:W4:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.100

### masked_token:generalization:W4:M1:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M1:Pend:L1024:beta30
- `q1_most_frequent`: 0.040

### masked_token:generalization:W4:M1:Pend:L1024:beta100
- `q1_most_frequent`: 0.060

### masked_token:generalization:W4:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W4:M2:Pend:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.113

### masked_token:generalization:W8:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 0.075

### masked_token:generalization:W8:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 0.057

### masked_token:generalization:W8:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 0.038

### masked_token:generalization:W8:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.113

### masked_token:generalization:W8:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 0.038

### masked_token:generalization:W8:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.057

### masked_token:generalization:W8:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.057

### masked_token:generalization:W8:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.057

### masked_token:generalization:W8:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L64:beta30
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pedge:L64:beta100
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.170

### masked_token:generalization:W8:M1:Pedge:L256:beta10
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pedge:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 0.019

### masked_token:generalization:W8:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.205

### masked_token:generalization:W8:M1:Pend:L64:beta10
- `q1_most_frequent`: 0.045

### masked_token:generalization:W8:M1:Pend:L64:beta30
- `q1_most_frequent`: 0.045

### masked_token:generalization:W8:M1:Pend:L64:beta100
- `q1_most_frequent`: 0.045

### masked_token:generalization:W8:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.045

### masked_token:generalization:W8:M1:Pend:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pend:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pend:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M1:Pend:L1024:beta30
- `q1_most_frequent`: 0.023

### masked_token:generalization:W8:M1:Pend:L1024:beta100
- `q1_most_frequent`: 0.023

### masked_token:generalization:W8:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.026

### masked_token:generalization:W8:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W8:M2:Pend:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.042

### masked_token:generalization:W16:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.083

### masked_token:generalization:W16:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.042

### masked_token:generalization:W16:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.042

### masked_token:generalization:W16:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.042

### masked_token:generalization:W16:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pedge:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pedge:L64:beta30
- `q1_most_frequent`: 0.022

### masked_token:generalization:W16:M1:Pedge:L64:beta100
- `q1_most_frequent`: 0.022

### masked_token:generalization:W16:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.043

### masked_token:generalization:W16:M1:Pedge:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pedge:L256:beta30
- `q1_most_frequent`: 0.043

### masked_token:generalization:W16:M1:Pedge:L256:beta100
- `q1_most_frequent`: 0.022

### masked_token:generalization:W16:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.043

### masked_token:generalization:W16:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 0.022

### masked_token:generalization:W16:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 0.022

### masked_token:generalization:W16:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.091

### masked_token:generalization:W16:M1:Pend:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L64:beta30
- `q1_most_frequent`: 0.023

### masked_token:generalization:W16:M1:Pend:L64:beta100
- `q1_most_frequent`: 0.023

### masked_token:generalization:W16:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.091

### masked_token:generalization:W16:M1:Pend:L256:beta10
- `q1_most_frequent`: 0.023

### masked_token:generalization:W16:M1:Pend:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M1:Pend:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L64:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L64:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L64:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L256:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L256:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L256:beta100
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L1024:beta30
- `q1_most_frequent`: 0.000

### masked_token:generalization:W16:M2:Pend:L1024:beta100
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.019

### masked_token:memorization:W4:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.149

### masked_token:memorization:W4:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 0.809

### masked_token:memorization:W4:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 0.957

### masked_token:memorization:W4:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 0.957

### masked_token:memorization:W4:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.419

### masked_token:memorization:W4:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.930

### masked_token:memorization:W4:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.930

### masked_token:memorization:W4:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.043

### masked_token:memorization:W4:M1:Pedge:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pedge:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pedge:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W4:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pedge:L256:beta10
- `q1_most_frequent`: 0.841

### masked_token:memorization:W4:M1:Pedge:L256:beta30
- `q1_most_frequent`: 0.955

### masked_token:memorization:W4:M1:Pedge:L256:beta100
- `q1_most_frequent`: 0.955

### masked_token:memorization:W4:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 0.375

### masked_token:memorization:W4:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 0.833

### masked_token:memorization:W4:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 0.833

### masked_token:memorization:W4:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.154

### masked_token:memorization:W4:M1:Pend:L64:beta10
- `q1_most_frequent`: 0.981

### masked_token:memorization:W4:M1:Pend:L64:beta30
- `q1_most_frequent`: 0.981

### masked_token:memorization:W4:M1:Pend:L64:beta100
- `q1_most_frequent`: 0.981

### masked_token:memorization:W4:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M1:Pend:L256:beta10
- `q1_most_frequent`: 0.929

### masked_token:memorization:W4:M1:Pend:L256:beta30
- `q1_most_frequent`: 0.952

### masked_token:memorization:W4:M1:Pend:L256:beta100
- `q1_most_frequent`: 0.952

### masked_token:memorization:W4:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.062

### masked_token:memorization:W4:M1:Pend:L1024:beta10
- `q1_most_frequent`: 0.396

### masked_token:memorization:W4:M1:Pend:L1024:beta30
- `q1_most_frequent`: 0.958

### masked_token:memorization:W4:M1:Pend:L1024:beta100
- `q1_most_frequent`: 0.958

### masked_token:memorization:W4:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 0.897

### masked_token:memorization:W4:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 0.897

### masked_token:memorization:W4:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 0.897

### masked_token:memorization:W4:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.053

### masked_token:memorization:W4:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 0.263

### masked_token:memorization:W4:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 0.895

### masked_token:memorization:W4:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 0.868

### masked_token:memorization:W4:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 0.806

### masked_token:memorization:W4:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 0.806

### masked_token:memorization:W4:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pedge:L64:beta10
- `q1_most_frequent`: 0.941

### masked_token:memorization:W4:M2:Pedge:L64:beta30
- `q1_most_frequent`: 0.941

### masked_token:memorization:W4:M2:Pedge:L64:beta100
- `q1_most_frequent`: 0.941

### masked_token:memorization:W4:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pedge:L256:beta10
- `q1_most_frequent`: 0.424

### masked_token:memorization:W4:M2:Pedge:L256:beta30
- `q1_most_frequent`: 0.848

### masked_token:memorization:W4:M2:Pedge:L256:beta100
- `q1_most_frequent`: 0.848

### masked_token:memorization:W4:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 0.694

### masked_token:memorization:W4:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 0.722

### masked_token:memorization:W4:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pend:L64:beta10
- `q1_most_frequent`: 0.911

### masked_token:memorization:W4:M2:Pend:L64:beta30
- `q1_most_frequent`: 0.911

### masked_token:memorization:W4:M2:Pend:L64:beta100
- `q1_most_frequent`: 0.911

### masked_token:memorization:W4:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pend:L256:beta10
- `q1_most_frequent`: 0.300

### masked_token:memorization:W4:M2:Pend:L256:beta30
- `q1_most_frequent`: 0.800

### masked_token:memorization:W4:M2:Pend:L256:beta100
- `q1_most_frequent`: 0.800

### masked_token:memorization:W4:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pend:L1024:beta10
- `q1_most_frequent`: 0.000

### masked_token:memorization:W4:M2:Pend:L1024:beta30
- `q1_most_frequent`: 0.688

### masked_token:memorization:W4:M2:Pend:L1024:beta100
- `q1_most_frequent`: 0.688

### masked_token:memorization:W8:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.213

### masked_token:memorization:W8:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.184

### masked_token:memorization:W8:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.075

### masked_token:memorization:W8:M1:Pedge:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.080

### masked_token:memorization:W8:M1:Pedge:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.140

### masked_token:memorization:W8:M1:Pend:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.024

### masked_token:memorization:W8:M1:Pend:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M1:Pend:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M1:Pend:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.027

### masked_token:memorization:W8:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pedge:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pedge:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pend:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pend:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W8:M2:Pend:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W8:M2:Pend:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L64:beta3
- `q1_most_frequent`: 0.085

### masked_token:memorization:W16:M1:Pcenter:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L256:beta3
- `q1_most_frequent`: 0.100

### masked_token:memorization:W16:M1:Pcenter:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M1:Pcenter:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pcenter:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L64:beta3
- `q1_most_frequent`: 0.026

### masked_token:memorization:W16:M1:Pedge:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L256:beta3
- `q1_most_frequent`: 0.039

### masked_token:memorization:W16:M1:Pedge:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L1024:beta3
- `q1_most_frequent`: 0.020

### masked_token:memorization:W16:M1:Pedge:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pedge:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L64:beta3
- `q1_most_frequent`: 0.096

### masked_token:memorization:W16:M1:Pend:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L256:beta3
- `q1_most_frequent`: 0.058

### masked_token:memorization:W16:M1:Pend:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M1:Pend:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M1:Pend:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pcenter:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pcenter:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pcenter:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pcenter:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pedge:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pedge:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pedge:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pedge:L1024:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L64:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pend:L64:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L64:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L64:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L256:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pend:L256:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L256:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L256:beta100
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L1024:beta3
- `q1_most_frequent`: 0.000

### masked_token:memorization:W16:M2:Pend:L1024:beta10
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L1024:beta30
- `q1_most_frequent`: 1.000

### masked_token:memorization:W16:M2:Pend:L1024:beta100
- `q1_most_frequent`: 1.000

### next_token:generalization:W4:M-:P-:L64:beta3
- `q1_most_frequent`: 0.100

### next_token:generalization:W4:M-:P-:L64:beta10
- `q1_most_frequent`: 0.040

### next_token:generalization:W4:M-:P-:L64:beta30
- `q1_most_frequent`: 0.020

### next_token:generalization:W4:M-:P-:L64:beta100
- `q1_most_frequent`: 0.040

### next_token:generalization:W4:M-:P-:L256:beta3
- `q1_most_frequent`: 0.000

### next_token:generalization:W4:M-:P-:L256:beta10
- `q1_most_frequent`: 0.020

### next_token:generalization:W4:M-:P-:L256:beta30
- `q1_most_frequent`: 0.080

### next_token:generalization:W4:M-:P-:L256:beta100
- `q1_most_frequent`: 0.060

### next_token:generalization:W4:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.100

### next_token:generalization:W4:M-:P-:L1024:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:W4:M-:P-:L1024:beta30
- `q1_most_frequent`: 0.040

### next_token:generalization:W4:M-:P-:L1024:beta100
- `q1_most_frequent`: 0.040

### next_token:generalization:W8:M-:P-:L64:beta3
- `q1_most_frequent`: 0.205

### next_token:generalization:W8:M-:P-:L64:beta10
- `q1_most_frequent`: 0.045

### next_token:generalization:W8:M-:P-:L64:beta30
- `q1_most_frequent`: 0.045

### next_token:generalization:W8:M-:P-:L64:beta100
- `q1_most_frequent`: 0.045

### next_token:generalization:W8:M-:P-:L256:beta3
- `q1_most_frequent`: 0.045

### next_token:generalization:W8:M-:P-:L256:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L256:beta30
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L256:beta100
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L1024:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:W8:M-:P-:L1024:beta30
- `q1_most_frequent`: 0.023

### next_token:generalization:W8:M-:P-:L1024:beta100
- `q1_most_frequent`: 0.045

### next_token:generalization:W16:M-:P-:L64:beta3
- `q1_most_frequent`: 0.091

### next_token:generalization:W16:M-:P-:L64:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L64:beta30
- `q1_most_frequent`: 0.023

### next_token:generalization:W16:M-:P-:L64:beta100
- `q1_most_frequent`: 0.023

### next_token:generalization:W16:M-:P-:L256:beta3
- `q1_most_frequent`: 0.091

### next_token:generalization:W16:M-:P-:L256:beta10
- `q1_most_frequent`: 0.023

### next_token:generalization:W16:M-:P-:L256:beta30
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L256:beta100
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L1024:beta10
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L1024:beta30
- `q1_most_frequent`: 0.000

### next_token:generalization:W16:M-:P-:L1024:beta100
- `q1_most_frequent`: 0.000

### next_token:memorization:W4:M-:P-:L64:beta3
- `q1_most_frequent`: 0.154

### next_token:memorization:W4:M-:P-:L64:beta10
- `q1_most_frequent`: 0.981

### next_token:memorization:W4:M-:P-:L64:beta30
- `q1_most_frequent`: 0.981

### next_token:memorization:W4:M-:P-:L64:beta100
- `q1_most_frequent`: 0.981

### next_token:memorization:W4:M-:P-:L256:beta3
- `q1_most_frequent`: 0.000

### next_token:memorization:W4:M-:P-:L256:beta10
- `q1_most_frequent`: 0.952

### next_token:memorization:W4:M-:P-:L256:beta30
- `q1_most_frequent`: 0.952

### next_token:memorization:W4:M-:P-:L256:beta100
- `q1_most_frequent`: 0.952

### next_token:memorization:W4:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.062

### next_token:memorization:W4:M-:P-:L1024:beta10
- `q1_most_frequent`: 0.667

### next_token:memorization:W4:M-:P-:L1024:beta30
- `q1_most_frequent`: 0.958

### next_token:memorization:W4:M-:P-:L1024:beta100
- `q1_most_frequent`: 0.958

### next_token:memorization:W8:M-:P-:L64:beta3
- `q1_most_frequent`: 0.140

### next_token:memorization:W8:M-:P-:L64:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L64:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L64:beta100
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L256:beta3
- `q1_most_frequent`: 0.024

### next_token:memorization:W8:M-:P-:L256:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L256:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L256:beta100
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.000

### next_token:memorization:W8:M-:P-:L1024:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L1024:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W8:M-:P-:L1024:beta100
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L64:beta3
- `q1_most_frequent`: 0.096

### next_token:memorization:W16:M-:P-:L64:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L64:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L64:beta100
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L256:beta3
- `q1_most_frequent`: 0.058

### next_token:memorization:W16:M-:P-:L256:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L256:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L256:beta100
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L1024:beta3
- `q1_most_frequent`: 0.000

### next_token:memorization:W16:M-:P-:L1024:beta10
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L1024:beta30
- `q1_most_frequent`: 1.000

### next_token:memorization:W16:M-:P-:L1024:beta100
- `q1_most_frequent`: 1.000
