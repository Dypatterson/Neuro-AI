# Cue-Degradation Sweep (Torch hot path)

- device: `mps`
- D: `4096`
- encoding: `bag`
- seed: `31`

## Summary

| Cue mode | First committed beta | First correct beta | Failure behavior |
|---|---:|---:|---|
| exact_pair | 4.0 | 0.5 | none |
| single_neighbor | 4.0 | 0.5 | none |
| neighbor_plus_noise | 4.0 | 0.5 | none |
| wrong_pair | none | 0.5 | flipped |
| noise_only | 4.0 | 0.5 | none |

## Full Table

| Cue mode | Temporal beta | Top label | Recall@4 | Top weight | Entropy | Iterations | Regime |
|---|---:|---|---:|---:|---:|---:|---|
| exact_pair | 0.5 | stair | 1.000 | 0.426 | 0.493 | 4 | ambiguous_correct |
| exact_pair | 1 | stair | 1.000 | 0.558 | 0.433 | 5 | ambiguous_correct |
| exact_pair | 2 | stair | 1.000 | 0.784 | 0.274 | 5 | ambiguous_correct |
| exact_pair | 4 | stair | 1.000 | 0.965 | 0.067 | 4 | committed |
| exact_pair | 8 | stair | 1.000 | 0.999 | 0.002 | 3 | committed |
| exact_pair | 16 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| exact_pair | 32 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| exact_pair | 64 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| exact_pair | 100 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| single_neighbor | 0.5 | stair | 1.000 | 0.420 | 0.495 | 4 | ambiguous_correct |
| single_neighbor | 1 | stair | 1.000 | 0.554 | 0.436 | 6 | ambiguous_correct |
| single_neighbor | 2 | stair | 1.000 | 0.781 | 0.277 | 5 | ambiguous_correct |
| single_neighbor | 4 | stair | 1.000 | 0.964 | 0.068 | 4 | committed |
| single_neighbor | 8 | stair | 1.000 | 0.999 | 0.002 | 3 | committed |
| single_neighbor | 16 | stair | 1.000 | 1.000 | 0.000 | 3 | committed |
| single_neighbor | 32 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| single_neighbor | 64 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| single_neighbor | 100 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| neighbor_plus_noise | 0.5 | stair | 1.000 | 0.415 | 0.497 | 5 | ambiguous_correct |
| neighbor_plus_noise | 1 | stair | 1.000 | 0.549 | 0.439 | 7 | ambiguous_correct |
| neighbor_plus_noise | 2 | stair | 1.000 | 0.779 | 0.279 | 6 | ambiguous_correct |
| neighbor_plus_noise | 4 | stair | 1.000 | 0.964 | 0.069 | 4 | committed |
| neighbor_plus_noise | 8 | stair | 1.000 | 0.999 | 0.002 | 3 | committed |
| neighbor_plus_noise | 16 | stair | 1.000 | 1.000 | 0.000 | 3 | committed |
| neighbor_plus_noise | 32 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| neighbor_plus_noise | 64 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| neighbor_plus_noise | 100 | stair | 1.000 | 1.000 | 0.000 | 2 | committed |
| wrong_pair | 0.5 | stair | 1.000 | 0.403 | 0.501 | 6 | ambiguous_correct |
| wrong_pair | 1 | stair | 1.000 | 0.532 | 0.447 | 10 | ambiguous_correct |
| wrong_pair | 2 | stair | 1.000 | 0.770 | 0.286 | 8 | ambiguous_correct |
| wrong_pair | 4 | ladder | 0.000 | 0.882 | 0.173 | 7 | flipped |
| wrong_pair | 8 | ladder | 0.000 | 0.997 | 0.009 | 4 | flipped |
| wrong_pair | 16 | ladder | 0.000 | 1.000 | 0.000 | 3 | flipped |
| wrong_pair | 32 | ladder | 0.000 | 1.000 | 0.000 | 2 | flipped |
| wrong_pair | 64 | ladder | 0.000 | 1.000 | 0.000 | 2 | flipped |
| wrong_pair | 100 | ladder | 0.000 | 1.000 | 0.000 | 2 | flipped |
| noise_only | 0.5 | stair | 1.000 | 0.407 | 0.500 | 6 | ambiguous_correct |
| noise_only | 1 | stair | 1.000 | 0.540 | 0.443 | 9 | ambiguous_correct |
| noise_only | 2 | stair | 1.000 | 0.776 | 0.281 | 7 | ambiguous_correct |
| noise_only | 4 | stair | 1.000 | 0.963 | 0.069 | 5 | committed |
| noise_only | 8 | stair | 1.000 | 0.999 | 0.002 | 4 | committed |
| noise_only | 16 | stair | 1.000 | 1.000 | 0.000 | 4 | committed |
| noise_only | 32 | stair | 1.000 | 1.000 | 0.000 | 3 | committed |
| noise_only | 64 | stair | 1.000 | 1.000 | 0.000 | 3 | committed |
| noise_only | 100 | stair | 1.000 | 1.000 | 0.000 | 3 | committed |
