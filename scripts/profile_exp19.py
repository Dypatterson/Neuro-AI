"""Profile experiment 19's hot path.

Runs a slim version of stream_phase34 under cProfile, then re-runs with
manual perf counters around the suspected hot spots (retrieve loop,
eval, sync points). Prints cumulative time and call counts.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_exp19.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import torch

# Import after path is set
sys.path.insert(0, "src")

from energy_memory.substrate.torch_fhrr import TorchFHRR
from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase4.trajectory import TracedHopfieldMemory


def setup(device: str, dim: int = 4096, scale: int = 2, n_patterns: int = 4096,
          vocab: int = 2050, n_cues: int = 50):
    substrate = TorchFHRR(dim=dim, device=device, seed=17)
    codebook = substrate.random_vectors(vocab)
    positions = build_position_vectors(substrate, scale)
    memory = TorchHopfieldMemory(substrate)
    traced = TracedHopfieldMemory(substrate)

    # Fill memory with random windows
    rng = torch.Generator().manual_seed(17)
    for _ in range(n_patterns):
        tokens = torch.randint(0, vocab, (scale,), generator=rng).tolist()
        vec = encode_window(substrate, positions, codebook, tokens)
        memory.store(vec, label=tuple(tokens))
        traced.store(vec, label=tuple(tokens))

    # Cue stream
    cues = []
    for _ in range(n_cues):
        tokens = torch.randint(0, vocab, (scale,), generator=rng).tolist()
        tokens[0] = 0  # masked
        cues.append((tokens, encode_window(substrate, positions, codebook, tokens)))

    return substrate, memory, traced, positions, codebook, cues


def hot_loop(memory, traced, cues, use_trace: bool, beta: float = 10.0):
    for _, cue_vec in cues:
        if use_trace:
            traced.retrieve_with_trace(cue_vec, beta=beta, max_iter=12)
        else:
            memory.retrieve(cue_vec, beta=beta, max_iter=12)


def time_section(name, fn, repeat: int = 3):
    # warmup
    fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  {name:40s} mean={sum(times)/len(times)*1000:8.2f}ms "
          f"min={min(times)*1000:8.2f}ms")
    return min(times)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")
    print(f"setup: dim=4096, n_patterns=4096, n_cues=50, max_iter=12, beta=10\n")

    substrate, memory, traced, positions, codebook, cues = setup(device)

    print("=== Wall-clock timing (synchronized) ===\n")
    print("Plain retrieve (no trace):")
    time_section(
        "  50 retrieves x 12 iters",
        lambda: hot_loop(memory, traced, cues, use_trace=False),
    )

    print("\nTraced retrieve (Phase 4 path):")
    time_section(
        "  50 retrieves with trace",
        lambda: hot_loop(memory, traced, cues, use_trace=True),
    )

    # Break down a single retrieve into substeps
    print("\n=== Single-retrieve breakdown (1 retrieve x 12 iters, 50 reps) ===\n")
    cue_vec = cues[0][1]

    def just_pattern_matrix():
        for _ in range(50):
            _ = memory._pattern_matrix()

    def just_similarity():
        patterns = memory._pattern_matrix()
        for _ in range(50):
            _ = substrate.similarity_matrix(cue_vec, patterns)

    def just_softmax_and_state():
        patterns = memory._pattern_matrix()
        for _ in range(50):
            scores = substrate.similarity_matrix(cue_vec, patterns)
            weights = torch.softmax(10.0 * scores, dim=0)
            _ = substrate.normalize((patterns * weights[:, None]).sum(dim=0))

    def just_cpu_sync():
        scores = substrate.similarity_matrix(cue_vec, memory._pattern_matrix())
        for _ in range(50):
            _ = int(torch.argmax(scores).detach().cpu())

    time_section("cache hit _pattern_matrix x50", just_pattern_matrix)
    time_section("similarity_matrix x50", just_similarity)
    time_section("similarity + softmax + state x50", just_softmax_and_state)
    time_section("argmax + .cpu() sync x50", just_cpu_sync)

    print("\n=== cProfile (plain retrieve, 50 cues) ===\n")
    profiler = cProfile.Profile()
    profiler.enable()
    hot_loop(memory, traced, cues, use_trace=False)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    profiler.disable()
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())

    print("=== cProfile (traced retrieve, 50 cues) ===\n")
    profiler = cProfile.Profile()
    profiler.enable()
    hot_loop(memory, traced, cues, use_trace=True)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    profiler.disable()
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()
