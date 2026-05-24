"""One-seed Phase 5 M1 smoke on a provenance-bearing snapshot.

This is a wiring/degen check only. It refuses snapshots that fail the M1
provenance audit and emits explicit smoke telemetry rather than Phase 5
evidence claims.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from energy_memory.phase4.snapshot import load_substrate_snapshot  # noqa: E402
from energy_memory.phase5.m1_role_energy import (  # noqa: E402
    M1Config,
    RoleBindingStats,
    run_m1_stack,
    weighted_patterns,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
from scripts.phase5_m1_provenance_audit import audit_snapshot  # noqa: E402


_EXP40_PATH = REPO_ROOT / "experiments" / "40_phase5_branching.py"
_spec = importlib.util.spec_from_file_location("experiments_40", str(_EXP40_PATH))
_exp40 = importlib.util.module_from_spec(_spec)
sys.modules["experiments_40"] = _exp40
_spec.loader.exec_module(_exp40)


def _load_snapshot(path: Path, device: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    patterns = state["patterns"]
    if patterns.numel() == 0:
        raise ValueError(f"snapshot {path} has zero stored patterns")
    substrate = TorchFHRR(dim=int(patterns.shape[-1]), device=device)
    mem, cons, info = load_substrate_snapshot(
        path=path, substrate=substrate, device=device,
    )
    return mem, cons, info


def _rank_target(
    *,
    substrate: TorchFHRR,
    state: "torch.Tensor",
    patterns: "torch.Tensor",
    role_weights: "torch.Tensor",
    target_idx: int,
    normalize_weighted_rows: bool,
) -> int:
    wp = weighted_patterns(
        substrate,
        patterns,
        role_weights.to(patterns.device),
        normalize_rows=normalize_weighted_rows,
    )
    scores = substrate.similarity_matrix(state.to(patterns.device), wp)
    target_score = scores[target_idx]
    return int((scores > target_score).sum().detach().cpu()) + 1


def _pick_random_prior(
    *,
    n_patterns: int,
    target_idx: int,
    content_idx: int,
    generator: "torch.Generator",
) -> int:
    order = torch.randperm(n_patterns, generator=generator).tolist()
    for idx in order:
        if idx not in {target_idx, content_idx}:
            return int(idx)
    return int(order[0])


def run_snapshot_smoke(
    *,
    snapshot: str | Path,
    output: str | Path,
    markdown_output: str | Path | None = None,
    device: str = "cpu",
    seed: int = 17,
    n_cues: int = 30,
    beta: float = 10.0,
    gamma: float = 0.5,
    max_iter: int = 12,
    d3_mix: float = 0.5,
    p3_saliency_gain: float = 0.0,
    laplace_count: float = 1.0,
    binding_noise_std: float = 0.05,
    content_distortion: float = 0.6,
) -> Dict[str, Any]:
    snapshot = Path(snapshot)
    audit = audit_snapshot(snapshot, device=device)
    if audit["status"] != "pass":
        raise ValueError(
            "snapshot failed M1 provenance audit: "
            + ", ".join(audit["failure_reasons"])
        )

    mem, _cons, info = _load_snapshot(snapshot, device=device)
    positions = info.get("positions")
    if positions is None:
        raise ValueError("snapshot smoke requires saved positions")
    role_vectors = [positions[i] for i in range(int(positions.shape[0]))]
    pattern_terms = info["pattern_encoder_terms"]
    patterns = mem._pattern_matrix()
    stats = RoleBindingStats.from_pattern_encoder_terms(
        pattern_terms,
        n_roles=len(role_vectors),
        device=device,
        require_complete=True,
    )
    atom_role_weights = stats.atom_role_weights(laplace=laplace_count)

    cue_specs = _exp40._build_role_binding_cues(
        substrate=mem.substrate,
        positions=role_vectors,
        patterns=list(mem._patterns),
        n_cues=n_cues,
        binding_noise_std=binding_noise_std,
        content_distortion=content_distortion,
        seed=seed,
    )
    rng = torch.Generator().manual_seed(seed + 991)
    cue_rows: List[Dict[str, Any]] = []
    deltas: List[float] = []
    hit_count = 0
    random_lowest_count = 0
    ranks: List[int] = []

    for cue_id, spec in enumerate(cue_specs):
        target_idx = int(spec["role_target_idx"])
        content_idx = int(spec["content_distractor_idx"])
        random_idx = _pick_random_prior(
            n_patterns=patterns.shape[0],
            target_idx=target_idx,
            content_idx=content_idx,
            generator=rng,
        )

        m1 = run_m1_stack(
            mem.substrate,
            spec["cue"],
            patterns,
            role_vectors,
            atom_role_weights,
            branch_roles=list(range(len(role_vectors))),
            config=M1Config(
                beta=beta,
                max_iter=max_iter,
                d3_mix=d3_mix,
                p3_saliency_gain=p3_saliency_gain,
                laplace_count=laplace_count,
            ),
        )
        best_branch = min(m1.branches, key=lambda b: b.energy)
        m1_min_energy = float(best_branch.energy)

        _content_state, content_tel = _exp40.settle_branch_with_prior(
            memory=mem,
            cue=spec["cue"],
            prior=patterns[content_idx],
            beta=beta,
            gamma=gamma,
            max_iter=max_iter,
            formulation="per_pattern",
        )
        _random_state, random_tel = _exp40.settle_branch_with_prior(
            memory=mem,
            cue=spec["cue"],
            prior=patterns[random_idx],
            beta=beta,
            gamma=gamma,
            max_iter=max_iter,
            formulation="per_pattern",
        )
        content_energy = float(content_tel["energy_unbiased_final"])
        random_energy = float(random_tel["energy_unbiased_final"])
        delta_e = content_energy - m1_min_energy
        hit_role = best_branch.top_index == target_idx
        rank_role = _rank_target(
            substrate=mem.substrate,
            state=best_branch.state,
            patterns=patterns,
            role_weights=atom_role_weights[:, best_branch.role_index],
            target_idx=target_idx,
            normalize_weighted_rows=False,
        )
        random_lowest = random_energy < content_energy and random_energy < m1_min_energy
        hit_count += int(hit_role)
        random_lowest_count += int(random_lowest)
        ranks.append(rank_role)
        deltas.append(delta_e)
        cue_rows.append({
            "cue_id": cue_id,
            "role_target_idx": target_idx,
            "content_distractor_idx": content_idx,
            "random_idx": random_idx,
            "content_energy": content_energy,
            "m1_min_energy": m1_min_energy,
            "random_energy": random_energy,
            "delta_e_content_minus_m1_min": delta_e,
            "hit_role": hit_role,
            "rank_role": rank_role,
            "best_branch_role": best_branch.role_index,
            "best_branch_top_index": best_branch.top_index,
            "joint_energy_trace": m1.joint_energy_trace,
            "branches": [
                {
                    "role_index": b.role_index,
                    "energy": b.energy,
                    "top_index": b.top_index,
                    "top_score": b.top_score,
                    "energy_trace": b.energy_trace,
                }
                for b in m1.branches
            ],
            "random_lowest": random_lowest,
        })

    summary = {
        "n_cues": n_cues,
        "mean_delta_e_content_minus_m1_min": (
            sum(deltas) / len(deltas) if deltas else float("nan")
        ),
        "hit_role": hit_count / n_cues if n_cues else float("nan"),
        "rank_role": sum(ranks) / len(ranks) if ranks else float("nan"),
        "random_lowest": random_lowest_count / n_cues if n_cues else float("nan"),
    }
    payload = {
        "scope": "seed-17 M1 real-substrate smoke; not Phase 5 evidence",
        "snapshot": str(snapshot),
        "seed": seed,
        "config": {
            "device": device,
            "beta": beta,
            "gamma": gamma,
            "max_iter": max_iter,
            "d3_mix": d3_mix,
            "p3_saliency_gain": p3_saliency_gain,
            "laplace_count": laplace_count,
            "binding_noise_std": binding_noise_std,
            "content_distortion": content_distortion,
        },
        "role_weight_utilization": audit["role_fractions"],
        "role_weight_entropy": audit["entropy"],
        "summary": summary,
        "audit": audit,
        "cues": cue_rows,
        "energy_comparison_note": (
            "content/random energies use the existing Phase 5 per-pattern "
            "baseline; M1 energy is the weighted role-energy branch minimum."
        ),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path = (
        Path(markdown_output)
        if markdown_output is not None
        else out_path.with_suffix(".md")
    )
    _write_markdown(payload, md_path)
    return payload


def _write_markdown(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = payload["summary"]
    lines = [
        "# Phase 5 M1 Snapshot Smoke",
        "",
        "**Scope:** seed-17 M1 real-substrate smoke; not Phase 5 evidence.",
        "",
        f"- Snapshot: `{payload['snapshot']}`",
        f"- n_cues: {s['n_cues']}",
        f"- mean DeltaE content - M1_min: {s['mean_delta_e_content_minus_m1_min']}",
        f"- hit_role: {s['hit_role']}",
        f"- rank_role: {s['rank_role']}",
        f"- random_lowest: {s['random_lowest']}",
        "",
        "## Role Weights",
        "",
        json.dumps({
            "role_weight_utilization": payload["role_weight_utilization"],
            "role_weight_entropy": payload["role_weight_entropy"],
        }, indent=2),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-cues", type=int, default=30)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--d3-mix", type=float, default=0.5)
    parser.add_argument("--p3-saliency-gain", type=float, default=0.0)
    parser.add_argument("--laplace-count", type=float, default=1.0)
    parser.add_argument("--binding-noise-std", type=float, default=0.05)
    parser.add_argument("--content-distortion", type=float, default=0.6)
    args = parser.parse_args()

    payload = run_snapshot_smoke(
        snapshot=args.snapshot,
        output=args.output,
        markdown_output=args.markdown_output,
        device=args.device,
        seed=args.seed,
        n_cues=args.n_cues,
        beta=args.beta,
        gamma=args.gamma,
        max_iter=args.max_iter,
        d3_mix=args.d3_mix,
        p3_saliency_gain=args.p3_saliency_gain,
        laplace_count=args.laplace_count,
        binding_noise_std=args.binding_noise_std,
        content_distortion=args.content_distortion,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
