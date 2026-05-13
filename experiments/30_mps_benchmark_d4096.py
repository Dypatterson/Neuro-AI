"""MPS vs CPU benchmark at D=4096 across landscape sizes (Tier C item 1).

Project plan exit criterion for Phase 1:
  "MPS speedup is meaningful for scoring and retrieval."

This script times the hot-path FHRR/Hopfield operations at D=4096 across
landscape sizes L in {64, 256, 1024, 4096} on both CPU and MPS. Each
operation is timed over multiple repetitions; the headline is the
MPS-vs-CPU speedup ratio per (operation, landscape size).

Operations timed:
  * random_vectors(L)        -- substrate primitive
  * bind / unbind            -- per-vector substrate primitives
  * bundle(L vectors)        -- bundling
  * similarity_matrix        -- the Hopfield scoring kernel
  * retrieve(beta=10, max_iter=8) -- full settling loop

Output: a markdown table and a JSON dump under reports/. Optionally
runs only one device with --device {cpu, mps}.

Run:
    PYTHONPATH=src .venv/bin/python experiments/30_mps_benchmark_d4096.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class BenchRow:
    device: str
    landscape: int
    operation: str
    mean_ms: float
    median_ms: float
    stdev_ms: float
    n_reps: int


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover
        torch.cuda.synchronize()


def _time_op(
    fn: Callable[[], object],
    *,
    device: torch.device,
    n_warmup: int,
    n_reps: int,
) -> List[float]:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    samples: List[float] = []
    for _ in range(n_reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _summarize(
    samples: List[float],
    *,
    device: str,
    landscape: int,
    operation: str,
) -> BenchRow:
    return BenchRow(
        device=device,
        landscape=landscape,
        operation=operation,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        stdev_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        n_reps=len(samples),
    )


def _run_device(
    device_str: str,
    *,
    dim: int,
    landscapes: List[int],
    n_warmup: int,
    n_reps: int,
    beta: float,
    max_iter: int,
    seed: int,
) -> List[BenchRow]:
    substrate = TorchFHRR(dim=dim, seed=seed, device=device_str)
    device = substrate.device
    rows: List[BenchRow] = []

    for L in landscapes:
        # random_vectors(L): one-shot substrate primitive timing.
        samples = _time_op(
            lambda: substrate.random_vectors(L),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="random_vectors"))

        # Build a stable landscape and query for the following operations.
        patterns = substrate.random_vectors(L)
        roles = substrate.random_vectors(L)
        query = substrate.perturb(patterns[0], noise=0.05)

        # bind: pairwise vector multiplication, single output of dim D.
        samples = _time_op(
            lambda: substrate.bind(query, roles[0]),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="bind"))

        # unbind: same shape, conj-multiplication.
        bound = substrate.bind(query, roles[0])
        samples = _time_op(
            lambda: substrate.unbind(bound, roles[0]),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="unbind"))

        # bundle(L vectors): the project's bundle operation across the full landscape.
        pattern_list = [patterns[i] for i in range(L)]
        samples = _time_op(
            lambda: substrate.bundle(pattern_list),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="bundle"))

        # similarity_matrix: query vs L patterns.
        samples = _time_op(
            lambda: substrate.similarity_matrix(query, patterns),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="similarity_matrix"))

        # Full Hopfield retrieve: includes settling loop.
        memory = TorchHopfieldMemory[int](substrate)
        for i in range(L):
            memory.store(patterns[i], label=i)
        samples = _time_op(
            lambda: memory.retrieve(query, beta=beta, max_iter=max_iter),
            device=device,
            n_warmup=n_warmup,
            n_reps=n_reps,
        )
        rows.append(_summarize(samples, device=device_str, landscape=L, operation="retrieve"))

    return rows


def _print_table(rows: List[BenchRow]) -> None:
    by_key: Dict[tuple, Dict[str, BenchRow]] = {}
    for r in rows:
        by_key.setdefault((r.operation, r.landscape), {})[r.device] = r

    operations = ["random_vectors", "bind", "unbind", "bundle", "similarity_matrix", "retrieve"]
    landscapes = sorted({r.landscape for r in rows})
    devices = sorted({r.device for r in rows})

    print(f"{'operation':<22} {'L':>6} " + " ".join(f"{d+'_ms':>14}" for d in devices) + f" {'speedup':>10}")
    print("-" * (22 + 6 + 16 * len(devices) + 12))
    for op in operations:
        for L in landscapes:
            entry = by_key.get((op, L), {})
            if not entry:
                continue
            line = f"{op:<22} {L:>6d} "
            for d in devices:
                row = entry.get(d)
                line += f"{(row.mean_ms if row else float('nan')):>14.3f}"
            if "cpu" in entry and "mps" in entry:
                speedup = entry["cpu"].mean_ms / max(entry["mps"].mean_ms, 1e-9)
                line += f" {speedup:>10.2f}x"
            else:
                line += f" {'':>11}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--landscapes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--devices", type=str, nargs="+", default=["cpu", "mps"], choices=["cpu", "mps"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: List[BenchRow] = []
    for device_str in args.devices:
        if device_str == "mps" and not torch.backends.mps.is_available():
            print("MPS unavailable; skipping mps benchmarks.")
            continue
        print(f"\nBenchmarking device={device_str} at D={args.dim} ...")
        rows.extend(
            _run_device(
                device_str,
                dim=args.dim,
                landscapes=list(args.landscapes),
                n_warmup=args.n_warmup,
                n_reps=args.n_reps,
                beta=args.beta,
                max_iter=args.max_iter,
                seed=args.seed,
            )
        )

    print()
    _print_table(rows)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": vars(args) | {"out": str(args.out)},
            "rows": [asdict(r) for r in rows],
        }, indent=2))
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
