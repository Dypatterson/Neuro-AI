"""Iterative coupled content+temporal settling.

Run:
    PYTHONPATH=src:. python3 experiments/coupled_settling.py
"""

from __future__ import annotations

import argparse

from energy_memory.diagnostics import temporal_association_score
from energy_memory.experiments.synthetic_worlds import (
    DISTRACTOR_STREAM,
    build_memory,
    distractor_vectors,
    expected_neighbors,
)
from energy_memory.substrate import FHRR


def run(
    seed: int = 23,
    dim: int = 512,
    window: int = 2,
    family_noise: float = 0.10,
    content_beta: float = 100.0,
    temporal_beta: float = 100.0,
    feedback: float = 0.75,
    max_iter: int = 8,
    tol: float = 1e-3,
):
    substrate = FHRR(dim=dim, seed=seed)
    vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=family_noise)
    memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=window)

    query = "stair"
    expected = expected_neighbors(DISTRACTOR_STREAM, DISTRACTOR_STREAM.index(query), window)
    k = window * 2
    temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])

    one_pass = memory.joint_recall(
        vectors[query],
        temporal_query,
        content_beta=content_beta,
        temporal_beta=temporal_beta,
        top_k=k,
    )
    coupled = memory.coupled_recall(
        vectors[query],
        temporal_query,
        content_beta=content_beta,
        temporal_beta=temporal_beta,
        feedback=feedback,
        max_iter=max_iter,
        tol=tol,
        top_k=k,
    )

    one_pass_score = temporal_association_score(expected, one_pass.temporal_items, k=k)
    coupled_score = temporal_association_score(expected, coupled.temporal_items, k=k)

    print(f"dim={dim} window={window} family_noise={family_noise}")
    print(f"content_beta={content_beta} temporal_beta={temporal_beta} feedback={feedback}")
    print(f"query={query} temporal_query=[slip, doctor]")
    print(f"expected_temporal_neighbors={sorted(expected)}")
    print("one_pass_temporal=[" + ", ".join(f"{label}:{score:.2f}" for label, score in one_pass.temporal_items) + "]")
    print("coupled_temporal=[" + ", ".join(f"{label}:{score:.2f}" for label, score in coupled.temporal_items) + "]")
    print(f"one-pass Recall@{k}: {one_pass_score:.3f}")
    print(f"coupled Recall@{k}: {coupled_score:.3f}")
    print(f"coupled top anchor: {coupled.top_label}")
    print(f"coupled converged: {coupled.converged}")
    print("trace:")
    for step in coupled.trace:
        print(
            f"  iter={step.iteration} top={step.top_label} "
            f"weight={step.top_weight:.3f} entropy={step.entropy:.3f} score={step.top_joint_score:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--family-noise", type=float, default=0.10)
    parser.add_argument("--content-beta", type=float, default=100.0)
    parser.add_argument("--temporal-beta", type=float, default=100.0)
    parser.add_argument("--feedback", type=float, default=0.75)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    run(
        seed=args.seed,
        dim=args.dim,
        window=args.window,
        family_noise=args.family_noise,
        content_beta=args.content_beta,
        temporal_beta=args.temporal_beta,
        feedback=args.feedback,
        max_iter=args.max_iter,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
