"""Local/Colab smoke for the Phase 5 M1 role-energy stack.

Local:
    PYTHONPATH=src:. .venv/bin/python scripts/phase5_m1_local_smoke.py

Colab:
    !git clone https://github.com/Dypatterson/Neuro-AI.git
    %cd Neuro-AI
    !git checkout phase5-m1-role-energy-stack
    !python scripts/phase5_m1_local_smoke.py --device cuda --output /content/m1_smoke.json

This is a substrate sanity check, not a Phase 5 graduation run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from energy_memory.phase2.encoding import build_position_vectors  # noqa: E402
from energy_memory.phase5.m1_role_energy import (  # noqa: E402
    M1Config,
    RoleBindingStats,
    run_m1_stack,
    run_s2_weighted_mhn_check,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


def _m1_synthetic_smoke(args: argparse.Namespace) -> dict:
    substrate = TorchFHRR(dim=args.dim, seed=args.seed, device=args.device)
    patterns = substrate.random_vectors(args.n_atoms)
    roles = build_position_vectors(substrate, args.n_roles)
    term_lists = []
    for atom_id in range(args.n_atoms):
        primary_role = atom_id % args.n_roles
        term_lists.append([(primary_role, atom_id)])
        if atom_id % 5 == 0:
            term_lists.append([((primary_role + 1) % args.n_roles, atom_id)])
    stats = RoleBindingStats.from_encoder_terms(
        term_lists,
        n_atoms=args.n_atoms,
        n_roles=args.n_roles,
        device=args.device,
    )
    atom_role_weights = stats.atom_role_weights(laplace=args.laplace_count)
    target_atom = 0
    cue = substrate.bind(roles[0], patterns[target_atom])
    result = run_m1_stack(
        substrate,
        cue,
        patterns,
        roles,
        atom_role_weights,
        branch_roles=list(range(args.n_roles)),
        config=M1Config(
            beta=args.beta,
            max_iter=args.max_iter,
            d3_mix=args.d3_mix,
            p3_saliency_gain=args.p3_saliency_gain,
            laplace_count=args.laplace_count,
        ),
    )
    return {
        "target_atom": target_atom,
        "branch_count": len(result.branches),
        "joint_energy_trace": result.joint_energy_trace,
        "branches": [
            {
                "role_index": b.role_index,
                "energy": b.energy,
                "top_index": b.top_index,
                "top_score": b.top_score,
                "hit_target": b.top_index == target_atom,
            }
            for b in result.branches
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-atoms", type=int, default=48)
    parser.add_argument("--n-roles", type=int, default=4)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--d3-mix", type=float, default=0.5)
    parser.add_argument("--p3-saliency-gain", type=float, default=0.1)
    parser.add_argument("--laplace-count", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    s2 = run_s2_weighted_mhn_check(
        seed=args.seed,
        dim=args.dim,
        n_atoms=args.n_atoms,
        n_roles=args.n_roles,
        beta=args.beta,
        max_iter=args.max_iter,
    )
    payload = {
        "scope": "Phase 5 M1 local smoke; not a graduation run",
        "s2": {
            "passed": s2.passed,
            "energy_trace": s2.energy_trace,
            "final_magnitude": s2.final_magnitude,
            "max_energy_increase": s2.max_energy_increase,
        },
        "m1_synthetic": _m1_synthetic_smoke(args),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
