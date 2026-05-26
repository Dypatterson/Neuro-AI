"""Fixed gate runner for the Report 096 natural-source mini-matrix plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from energy_memory.phase5.natural_source_protocol import (
    source_with_protocol_plan,
)
from scripts import phase5_prime_natural_source_control_cleanup_gate as cleanup_gate
from scripts import phase5_prime_nonsynthetic_native_gate as base_gate


POSITIVE_CONDITIONS = {"perfect_cue", "bundle_positive"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve(path: str) -> Path:
    return Path(path)


def _validate_preflight(preflight: dict) -> None:
    framing = preflight.get("framing", {})
    if not framing.get("preflight_only"):
        raise ValueError("mini-matrix artifact is not preflight_only")
    if not framing.get("no_candidate_control_retrieval"):
        raise ValueError("mini-matrix artifact does not forbid retrieval")
    if not preflight.get("passes_all_criteria"):
        raise ValueError("mini-matrix preflight did not pass all criteria")
    if preflight.get("cleanup_protocol_summary", {}).get("protocol_name") != (
        "non_special_unique_target_freq_le_32"
    ):
        raise ValueError("unexpected cleanup protocol")


def _run_positive_condition(
    *,
    condition: str,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cue_noise: float,
    scene_token_weight: float,
    cooccurrence: str,
    n_queries: int,
    beta: float,
    max_iter: int,
    device: str,
) -> base_gate.GateResult:
    if condition not in POSITIVE_CONDITIONS:
        raise ValueError(f"unsupported positive condition: {condition}")
    seed_key = str(seed)
    rows_raw = source["source_rows_by_seed"][seed_key]
    query_plan = source["query_plan_by_seed"][seed_key]
    if len(rows_raw) != N:
        raise ValueError(f"source rows for seed {seed} have len {len(rows_raw)} != {N}")
    if len(query_plan) != n_queries:
        raise ValueError(
            f"query plan for seed {seed} has len {len(query_plan)} != {n_queries}"
        )

    fhrr = base_gate.TorchFHRR(dim=D, seed=seed, device=device)
    roles = base_gate._native_roles(fhrr, K_roles)
    content = fhrr.random_vectors(C_codebook)
    rows = torch.tensor(rows_raw, dtype=torch.long, device=fhrr.device)
    scene_matrix = base_gate._scene_bundles(
        fhrr,
        roles,
        content,
        rows,
        scene_token_weight=scene_token_weight,
    )

    scene_idx = torch.tensor(
        [int(item["scene"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    query_role = torch.tensor(
        [int(item["query_role"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    target_atom = rows[scene_idx, query_role]
    zero_stats = torch.zeros(n_queries, device=fhrr.device)
    if condition == "bundle_positive":
        scene_state = scene_matrix[scene_idx]
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
    else:
        scene_state, scene_top_index, scene_entropy, scene_margin = (
            base_gate.EXP44._batched_hopfield_retrieve(
                fhrr,
                scene_matrix,
                scene_matrix[scene_idx],
                beta=beta,
                max_iter=max_iter,
            )
        )
    content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[query_role]))
    scene_tix = int((scene_top_index == scene_idx).sum().detach().cpu())
    content_state, content_top_index, content_entropy, content_margin = (
        base_gate.EXP44._batched_hopfield_retrieve(
            fhrr,
            content,
            content_query,
            beta=beta,
            max_iter=max_iter,
        )
    )
    content_tix = int((content_top_index == target_atom).sum().detach().cpu())
    pred = torch.argmax((content_state @ content.conj().T).real / content.shape[1], dim=1)
    n_correct = int((pred == target_atom).sum().detach().cpu())
    return base_gate.GateResult(
        condition=condition,
        D=D,
        N=N,
        K_roles=K_roles,
        cue_noise=cue_noise,
        scene_token_weight=scene_token_weight,
        source_name=base_gate.SOURCE_NAME,
        context_roles=context_roles,
        cooccurrence=cooccurrence,
        seed=seed,
        n_queries=n_queries,
        n_correct=n_correct,
        scene_tix=scene_tix,
        content_tix=content_tix,
        scene_entropy=float(scene_entropy.mean().detach().cpu()),
        content_entropy=float(content_entropy.mean().detach().cpu()),
        scene_margin=float(scene_margin.mean().detach().cpu()),
        content_margin=float(content_margin.mean().detach().cpu()),
        source_rows_available=len(rows_raw),
        source_rows_used=N,
        source_rows_invalid=0,
        source_rows_too_short=0,
        source_artifact_path=str(source_path),
        source_artifact_sha256=source_sha,
    )


def _run_cell_seed(
    *,
    cell: dict,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    config: dict,
    beta: float,
    max_iter: int,
    device: str,
) -> base_gate.GateResult:
    condition = str(cell["cell_kind"])
    gate_condition = str(cell["gate_condition"])
    kwargs = {
        "seed": seed,
        "source": source,
        "source_path": source_path,
        "source_sha": source_sha,
        "D": int(config["D"]),
        "N": int(config["N"]),
        "K_roles": int(config["K_roles"]),
        "C_codebook": int(config["C_codebook"]),
        "context_roles": int(config["context_roles"]),
        "cue_noise": float(cell["cue_noise"]),
        "scene_token_weight": float(cell["scene_token_weight"]),
        "cooccurrence": str(config["cooccurrence"]),
        "n_queries": int(config["n_queries"]),
        "beta": beta,
        "max_iter": max_iter,
        "device": device,
    }
    if condition in POSITIVE_CONDITIONS:
        return _run_positive_condition(condition=condition, **kwargs)
    return cleanup_gate._run_seed_condition(condition=gate_condition, **kwargs)


def _planned_protocol(preflight: dict) -> dict:
    summary = preflight["cleanup_protocol_summary"]
    return {
        "protocol_name": summary["protocol_name"],
        "passes_all_criteria": True,
        "required_queries_per_seed": int(preflight["config"]["n_queries"]),
        "selected_query_plan_by_seed": preflight["selected_query_plan_by_seed"],
    }


def _filter_cells(
    cells: Sequence[dict],
    *,
    conditions: Sequence[str] | None,
    cue_noise: Sequence[float] | None,
    max_cells: int | None,
) -> List[dict]:
    out = list(cells)
    if conditions:
        allowed = set(conditions)
        out = [cell for cell in out if cell["cell_kind"] in allowed]
    if cue_noise:
        allowed_noise = {float(value) for value in cue_noise}
        out = [cell for cell in out if float(cell["cue_noise"]) in allowed_noise]
    if max_cells is not None:
        out = out[:max_cells]
    if not out:
        raise ValueError("cell filters selected no planned cells")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        default="reports/phase5_prime_natural_source_mini_matrix_preflight.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_natural_source_mini_matrix_gate.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--cue-noise", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    preflight_path = Path(args.preflight)
    preflight = _load_json(preflight_path)
    _validate_preflight(preflight)
    source_path = _resolve(preflight["source_manifest"]["source_artifact_path"])
    source = _load_json(source_path)
    source_sha = base_gate._sha256(source_path)
    if source_sha != preflight["source_manifest"]["source_artifact_sha256"]:
        raise ValueError("source SHA mismatch")
    protocol = _planned_protocol(preflight)
    run_source = source_with_protocol_plan(source, protocol)
    config = preflight["config"]
    cells = _filter_cells(
        preflight["planned_cells"],
        conditions=args.conditions,
        cue_noise=args.cue_noise,
        max_cells=args.max_cells,
    )
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    print(
        f"device={device} preflight={preflight_path} cells={len(cells)} "
        f"seeds={seeds} out={args.out}"
    )

    raw: List[base_gate.GateResult] = []
    aggregates: Dict[str, dict] = {}
    for cell in cells:
        cell_results: List[base_gate.GateResult] = []
        for seed in seeds:
            result = _run_cell_seed(
                cell=cell,
                seed=seed,
                source=run_source,
                source_path=source_path,
                source_sha=source_sha,
                config=config,
                beta=args.beta,
                max_iter=args.max_iter,
                device=device,
            )
            cell_results.append(result)
            raw.append(result)
        agg = base_gate._aggregate(cell_results)
        agg["cell_kind"] = cell["cell_kind"]
        agg["gate_condition"] = cell["gate_condition"]
        agg["scene_minus_content_tix"] = int(agg["scene_tix"] - agg["content_tix"])
        agg["scene_failure_count"] = int(agg["n_total"] - agg["scene_tix"])
        aggregates[cell["cell_id"]] = agg
        print(
            f"{cell['cell_id']} top1={agg['top1_mean']:.4f} "
            f"CI=[{agg['wilson_lo']:.4f},{agg['wilson_hi']:.4f}] "
            f"scene_tix={agg['scene_tix']}/{agg['n_total']} "
            f"content_tix={agg['content_tix']}/{agg['n_total']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framing": {
            "phase": "5-prime natural-source mini-matrix gate",
            "not_graduation": True,
            "no_delta_e_headline": True,
            "no_m2": True,
            "no_full_matrix": True,
            "anti_homunculus": (
                "consumes the fixed Report 096 preflight plan; no "
                "metric-triggered routing or best-of-N selection"
            ),
        },
        "source_manifest": {
            **preflight["source_manifest"],
            "mini_matrix_preflight_path": str(preflight_path),
            "mini_matrix_preflight_sha256": base_gate._sha256(preflight_path),
        },
        "config": {
            **config,
            "device": device,
            "beta": args.beta,
            "max_iter": args.max_iter,
            "executed_seeds": seeds,
            "executed_cell_ids": [cell["cell_id"] for cell in cells],
        },
        "aggregates": aggregates,
        "raw": [
            {
                "condition": r.condition,
                "D": r.D,
                "N": r.N,
                "K_roles": r.K_roles,
                "cue_noise": r.cue_noise,
                "scene_token_weight": r.scene_token_weight,
                "source_name": r.source_name,
                "context_roles": r.context_roles,
                "cooccurrence": r.cooccurrence,
                "seed": r.seed,
                "n_queries": r.n_queries,
                "n_correct": r.n_correct,
                "top1": r.top1,
                "scene_tix": r.scene_tix,
                "content_tix": r.content_tix,
                "scene_entropy": r.scene_entropy,
                "content_entropy": r.content_entropy,
                "scene_margin": r.scene_margin,
                "content_margin": r.content_margin,
                "source_artifact_sha256": r.source_artifact_sha256,
            }
            for r in raw
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
