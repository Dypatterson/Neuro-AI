"""Joint content+temporal energy read for entangled content anchors.

Run:
    PYTHONPATH=src python3 experiments/joint_energy_disambiguation.py
"""

from __future__ import annotations

import argparse

from energy_memory.diagnostics import temporal_association_score
from energy_memory.experiments.synthetic_worlds import (
    DISTRACTOR_STREAM,
    build_memory,
    content_neighbors,
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
):
    substrate = FHRR(dim=dim, seed=seed)
    vectors = distractor_vectors(substrate, DISTRACTOR_STREAM, family_noise=family_noise)
    memory = build_memory(substrate, DISTRACTOR_STREAM, vectors, window=window)

    query = "stair"
    expected = expected_neighbors(DISTRACTOR_STREAM, DISTRACTOR_STREAM.index(query), window)
    k = window * 2

    # Content-only receives an ambiguous stair-like cue.
    content_query = vectors[query]
    content_top = content_neighbors(substrate, query, vectors, k=k)
    content_score = temporal_association_score(expected, content_top, k=k)

    # Post-content temporal read inherits the content ambiguity and fails here.
    post_content = memory.recall(content_query, beta=content_beta, top_k=k)
    post_score = temporal_association_score(expected, post_content.temporal_items, k=k)

    # Joint read receives surrounding context as a temporal cue.
    temporal_query = substrate.bundle([vectors["slip"], vectors["doctor"]])
    joint = memory.joint_recall(
        content_query,
        temporal_query,
        content_beta=content_beta,
        temporal_beta=temporal_beta,
        top_k=k,
    )
    joint_score = temporal_association_score(expected, joint.temporal_items, k=k)

    print(f"dim={dim} window={window} family_noise={family_noise}")
    print(f"content_beta={content_beta} temporal_beta={temporal_beta}")
    print(f"query={query}")
    print(f"temporal_query=[slip, doctor]")
    print(f"expected_temporal_neighbors={sorted(expected)}")
    print("content_nearest=[" + ", ".join(f"{label}:{score:.2f}" for label, score in content_top) + "]")
    print(
        "post_content_temporal=["
        + ", ".join(f"{label}:{score:.2f}" for label, score in post_content.temporal_items)
        + "]"
    )
    print("joint_temporal=[" + ", ".join(f"{label}:{score:.2f}" for label, score in joint.temporal_items) + "]")
    print(f"content baseline Recall@{k}: {content_score:.3f}")
    print(f"post-content temporal Recall@{k}: {post_score:.3f}")
    print(f"joint temporal Recall@{k}: {joint_score:.3f}")
    print(f"joint top anchor: {joint.top_label}")
    print(f"joint entropy: {joint.entropy:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--family-noise", type=float, default=0.10)
    parser.add_argument("--content-beta", type=float, default=100.0)
    parser.add_argument("--temporal-beta", type=float, default=100.0)
    args = parser.parse_args()
    run(
        seed=args.seed,
        dim=args.dim,
        window=args.window,
        family_noise=args.family_noise,
        content_beta=args.content_beta,
        temporal_beta=args.temporal_beta,
    )


if __name__ == "__main__":
    main()

