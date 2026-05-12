# Phase 4: Multi-Scale Retrieval Validation

Robust validation of the multi-scale retrieval claim using 5 random seeds,
N≈730 test windows per seed, bootstrap 95% CIs, with both reconstruction
and random codebook comparisons.

## Setup

- **Substrate**: TorchFHRR, D=4096, MPS
- **Codebook**: `phase3c_codebook_reconstruction.pt` (W=8 reconstruction-trained)
- **Scales**: W={2,3,4} with landscapes L={4096,2048,1024} respectively
- **Aggregation**: uniform sum of decoded scores across scales (top-20 per scale)
- **β**: 30.0
- **Test pool**: WikiText-2 validation split, 23K-46K windows
- **Seeds**: 11, 17, 23, 31, 42

## Aggregated Results

### W=4 center mask

| Method | Mean | ± Std | vs Bigram |
|---|---:|---:|---:|
| Bigram baseline | 0.056 | 0.007 | — |
| Random codebook (multi-scale) | 0.100 | 0.013 | 1.78x |
| Reconstruction codebook (multi-scale) | **0.121** | 0.013 | **2.15x** |
| Lift ratio (recon / bigram) | 2.18x | 0.35x | — |

Per-scale (reconstruction codebook):
- W=2 scale: 0.117 ± 0.016
- W=3 scale: 0.112 ± 0.004
- W=4 scale: 0.081 ± 0.010

### W=8 center mask

| Method | Mean | ± Std | vs Bigram |
|---|---:|---:|---:|
| Bigram baseline | 0.056 | 0.010 | — |
| Random codebook (multi-scale) | 0.107 | 0.020 | 1.92x |
| Reconstruction codebook (multi-scale) | **0.138** | 0.009 | **2.49x** |
| Lift ratio (recon / bigram) | 2.54x | 0.42x | — |

Per-scale (reconstruction codebook):
- W=2 scale: 0.133 ± 0.010
- W=3 scale: 0.122 ± 0.012
- W=4 scale: 0.093 ± 0.008

## Per-Seed Results

### W=4 center mask

| Seed | N | Bigram (95% CI) | Recon Combined (95% CI) | Random Combined (95% CI) | Lift |
|---:|---:|---|---|---|---:|
| 11 | 743 | 0.047 [0.032, 0.063] | 0.112 [0.090, 0.135] | 0.087 [0.067, 0.109] | 2.37x |
| 17 | 714 | 0.063 [0.046, 0.081] | 0.116 [0.092, 0.139] | 0.111 [0.087, 0.133] | 1.84x |
| 23 | 747 | 0.059 [0.043, 0.076] | 0.110 [0.087, 0.133] | 0.086 [0.066, 0.107] | 1.86x |
| 31 | 741 | 0.061 [0.045, 0.078] | 0.130 [0.107, 0.155] | 0.108 [0.086, 0.131] | 2.13x |
| 42 | 711 | 0.052 [0.037, 0.069] | 0.139 [0.115, 0.166] | 0.110 [0.086, 0.134] | 2.68x |

### W=8 center mask

| Seed | N | Bigram (95% CI) | Recon Combined (95% CI) | Random Combined (95% CI) | Lift |
|---:|---:|---|---|---|---:|
| 11 | 714 | 0.045 [0.029, 0.060] | 0.141 [0.116, 0.167] | 0.077 [0.057, 0.098] | 3.16x |
| 17 | 722 | 0.051 [0.036, 0.068] | 0.133 [0.109, 0.157] | 0.115 [0.093, 0.139] | 2.59x |
| 23 | 739 | 0.061 [0.045, 0.080] | 0.130 [0.106, 0.154] | 0.116 [0.093, 0.141] | 2.13x |
| 31 | 735 | 0.050 [0.035, 0.067] | 0.135 [0.112, 0.161] | 0.128 [0.105, 0.154] | 2.68x |
| 42 | 735 | 0.071 [0.053, 0.090] | 0.152 [0.127, 0.181] | 0.098 [0.078, 0.118] | 2.15x |

## Findings

1. **Multi-scale retrieval beats bigrams by 2-3x with high reliability.** Lift ratio
   2.18x ± 0.35x (W=4) and 2.54x ± 0.42x (W=8). All 10 seed/window combinations
   show lift > 1.84x. CIs of recon-combined and bigram never overlap in any seed.

2. **Reconstruction codebook adds 20-30% over random codebook.** This is a
   smaller but consistent effect: W=4 recon (0.121) vs random (0.100), W=8 recon
   (0.138) vs random (0.107). The codebook learning matters, but the architecture
   matters more.

3. **Architecture is the dominant factor.** Even the random codebook achieves
   1.78x-1.92x bigram lift via multi-scale aggregation alone. The multi-scale
   retrieval mechanism is doing most of the work; codebook training is an additive
   gain on top.

4. **W=2 is the strongest single scale.** At W=4 eval, the W=2 scale (0.117)
   matches the combined (0.121). At W=8 eval, W=2 (0.133) is close to combined
   (0.138) — combined adds only ~4% relative over the best single scale.

5. **W=4 scale underperforms W=2 and W=3.** At both eval window sizes, the W=4
   scale's accuracy (~0.08-0.09) is the weakest, despite being the matched scale
   for W=4 evaluation. This is consistent with the coverage hypothesis: at W=4,
   each stored pattern requires 3 context tokens to match, so coverage is much
   sparser than at W=2 (1 context token) or W=3 (2 context tokens).

## Implications

The multi-scale architecture is validated as a genuine improvement over both
single-scale FHRR retrieval and bigram statistical baselines. The 2-2.5x bigram
lift is statistically robust and reproducible across seeds.

The dominant mechanism is short-range pattern coverage (W=2 contributes most),
with longer-range scales adding marginal discrimination. This suggests the
system is functioning as an FHRR-encoded n-gram model with cross-scale
voting, rather than discovering deeper compositional structure.

Next-step candidates:
- **Autoregressive generation** — does this architecture produce coherent
  sequences when chained, or does it just give marginally better next-token
  predictions?
- **Larger D** (8192, 16384) — does increased substrate capacity raise the
  ceiling for any scale, or is the system already capacity-saturated?
- **Compositional scale aggregation** — can we make longer scales contribute
  more by using them as constraints on shorter-scale predictions rather than
  additive score evidence?
