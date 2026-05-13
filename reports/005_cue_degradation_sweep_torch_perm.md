# Cue-Degradation Sweep (Torch hot path)

- device: `mps`
- D: `4096`
- encoding: `permutation`
- seed: `31`

## Summary

| Cue mode | First committed beta | First correct beta | Failure behavior |
|---|---:|---:|---|
| exact_pair | none | none | ambiguous_low_recall, wrong_or_sparse |
| single_neighbor | none | none | ambiguous_low_recall, wrong_or_sparse |
| neighbor_plus_noise | none | none | ambiguous_low_recall, flipped, wrong_or_sparse |
| wrong_pair | none | none | ambiguous_low_recall, flipped, wrong_or_sparse |
| noise_only | none | none | ambiguous_low_recall, flipped, wrong_or_sparse |

## Full Table

| Cue mode | Temporal beta | Top label | Recall@4 | Top weight | Entropy | Iterations | Regime |
|---|---:|---|---:|---:|---:|---:|---|
| exact_pair | 0.5 | stair | 0.000 | 0.442 | 0.487 | 6 | ambiguous_low_recall |
| exact_pair | 1 | stair | 0.000 | 0.566 | 0.429 | 6 | wrong_or_sparse |
| exact_pair | 2 | stair | 0.000 | 0.785 | 0.273 | 6 | wrong_or_sparse |
| exact_pair | 4 | stair | 0.000 | 0.964 | 0.068 | 5 | wrong_or_sparse |
| exact_pair | 8 | stair | 0.000 | 0.999 | 0.002 | 4 | wrong_or_sparse |
| exact_pair | 16 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| exact_pair | 32 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| exact_pair | 64 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| exact_pair | 100 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| single_neighbor | 0.5 | stair | 0.000 | 0.442 | 0.487 | 6 | ambiguous_low_recall |
| single_neighbor | 1 | stair | 0.000 | 0.566 | 0.429 | 6 | wrong_or_sparse |
| single_neighbor | 2 | stair | 0.000 | 0.785 | 0.273 | 6 | wrong_or_sparse |
| single_neighbor | 4 | stair | 0.000 | 0.964 | 0.068 | 5 | wrong_or_sparse |
| single_neighbor | 8 | stair | 0.000 | 0.999 | 0.002 | 4 | wrong_or_sparse |
| single_neighbor | 16 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| single_neighbor | 32 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| single_neighbor | 64 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| single_neighbor | 100 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| neighbor_plus_noise | 0.5 | stair | 0.000 | 0.441 | 0.487 | 6 | ambiguous_low_recall |
| neighbor_plus_noise | 1 | stair | 0.000 | 0.565 | 0.430 | 6 | wrong_or_sparse |
| neighbor_plus_noise | 2 | stair | 0.000 | 0.784 | 0.274 | 6 | wrong_or_sparse |
| neighbor_plus_noise | 4 | stair | 0.000 | 0.964 | 0.068 | 5 | wrong_or_sparse |
| neighbor_plus_noise | 8 | stair | 0.000 | 0.999 | 0.002 | 4 | wrong_or_sparse |
| neighbor_plus_noise | 16 | stair | 0.000 | 1.000 | 0.000 | 4 | wrong_or_sparse |
| neighbor_plus_noise | 32 | stair | 0.000 | 1.000 | 0.000 | 3 | wrong_or_sparse |
| neighbor_plus_noise | 64 | stair | 0.000 | 1.000 | 0.000 | 4 | wrong_or_sparse |
| neighbor_plus_noise | 100 | step | 0.250 | 1.000 | 0.000 | 4 | flipped |
| wrong_pair | 0.5 | stair | 0.000 | 0.441 | 0.488 | 6 | ambiguous_low_recall |
| wrong_pair | 1 | stair | 0.000 | 0.564 | 0.430 | 6 | wrong_or_sparse |
| wrong_pair | 2 | stair | 0.000 | 0.784 | 0.274 | 6 | wrong_or_sparse |
| wrong_pair | 4 | stair | 0.000 | 0.964 | 0.068 | 5 | wrong_or_sparse |
| wrong_pair | 8 | stair | 0.000 | 0.999 | 0.002 | 4 | wrong_or_sparse |
| wrong_pair | 16 | stair | 0.000 | 1.000 | 0.000 | 4 | wrong_or_sparse |
| wrong_pair | 32 | ramp | 0.250 | 1.000 | 0.000 | 4 | flipped |
| wrong_pair | 64 | ramp | 0.250 | 1.000 | 0.000 | 3 | flipped |
| wrong_pair | 100 | ramp | 0.250 | 1.000 | 0.000 | 3 | flipped |
| noise_only | 0.5 | stair | 0.000 | 0.441 | 0.488 | 6 | ambiguous_low_recall |
| noise_only | 1 | stair | 0.000 | 0.565 | 0.430 | 6 | wrong_or_sparse |
| noise_only | 2 | stair | 0.000 | 0.784 | 0.274 | 6 | wrong_or_sparse |
| noise_only | 4 | stair | 0.000 | 0.964 | 0.068 | 5 | wrong_or_sparse |
| noise_only | 8 | stair | 0.000 | 0.999 | 0.002 | 4 | wrong_or_sparse |
| noise_only | 16 | stair | 0.000 | 1.000 | 0.000 | 4 | wrong_or_sparse |
| noise_only | 32 | step | 0.250 | 1.000 | 0.000 | 4 | flipped |
| noise_only | 64 | step | 0.250 | 1.000 | 0.000 | 3 | flipped |
| noise_only | 100 | step | 0.250 | 1.000 | 0.000 | 3 | flipped |
