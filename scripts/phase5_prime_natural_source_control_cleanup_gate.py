"""Fixed gate for the Report 092 cleaned natural-source protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from scripts import phase5_prime_nonsynthetic_native_gate as base_gate


CONDITIONS = {
    "candidate",
    "random_role",
    "deranged_role",
    "fixedpoint_free_shuffled_role",
    "content_cleanup_positive",
}


def _fixedpoint_free_shuffle(seed: int, k_roles: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + k_roles)
    return torch.tensor(
        base_gate.EXP44._role_derangement(k_roles, generator=generator),
        dtype=torch.long,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _protocol_payload(preflight: dict, protocol_name: str | None) -> dict:
    selected = protocol_name or preflight.get("recommended_protocol")
    if not selected:
        raise ValueError("cleanup preflight does not name a recommended protocol")
    for protocol in preflight.get("protocols", []):
        if protocol.get("protocol_name") == selected:
            if not protocol.get("passes_all_criteria"):
                raise ValueError(f"cleanup protocol does not pass: {selected}")
            return protocol
    raise ValueError(f"cleanup protocol not found: {selected}")


def _source_with_protocol_plan(source: dict, protocol: dict) -> dict:
    out = dict(source)
    out["query_plan_by_seed"] = protocol["selected_query_plan_by_seed"]
    out["config"] = dict(source["config"])
    out["config"]["n_queries"] = int(protocol["required_queries_per_seed"])
    return out


def _run_fixedpoint_free_shuffled(
    *,
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
    known_role = torch.tensor(
        [int(item["known_role"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    query_role = torch.tensor(
        [int(item["query_role"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    target_atom = rows[scene_idx, query_role]
    role_shuffle = _fixedpoint_free_shuffle(seed, K_roles).to(fhrr.device)

    cue_role = role_shuffle[known_role]
    unbind_role = role_shuffle[query_role]
    known_atom = rows[scene_idx, known_role]
    cue = roles[cue_role] * content[known_atom]
    query_tokens = base_gate._query_context_tokens(fhrr, roles, content, rows, query_plan)
    cue = fhrr.normalize(cue + scene_token_weight * query_tokens)
    cue = base_gate.EXP44._perturb_batch(fhrr, cue, cue_noise)
    scene_state, scene_top_index, scene_entropy, scene_margin = (
        base_gate.EXP44._batched_hopfield_retrieve(
            fhrr,
            scene_matrix,
            cue,
            beta=beta,
            max_iter=max_iter,
        )
    )
    content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[unbind_role]))
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
        condition="fixedpoint_free_shuffled_role",
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


def _run_seed_condition(
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
    if condition == "fixedpoint_free_shuffled_role":
        return _run_fixedpoint_free_shuffled(
            seed=seed,
            source=source,
            source_path=source_path,
            source_sha=source_sha,
            D=D,
            N=N,
            K_roles=K_roles,
            C_codebook=C_codebook,
            context_roles=context_roles,
            cue_noise=cue_noise,
            scene_token_weight=scene_token_weight,
            cooccurrence=cooccurrence,
            n_queries=n_queries,
            beta=beta,
            max_iter=max_iter,
            device=device,
        )
    return base_gate._run_seed_condition(
        condition=condition,
        seed=seed,
        source=source,
        source_path=source_path,
        source_sha=source_sha,
        D=D,
        N=N,
        K_roles=K_roles,
        C_codebook=C_codebook,
        context_roles=context_roles,
        cue_noise=cue_noise,
        scene_token_weight=scene_token_weight,
        cooccurrence=cooccurrence,
        n_queries=n_queries,
        beta=beta,
        max_iter=max_iter,
        device=device,
    )


def _validate_preflight(preflight: dict, source_sha: str, gate_sha: str) -> None:
    manifest = preflight.get("source_manifest", {})
    if manifest.get("source_artifact_sha256") != source_sha:
        raise ValueError("cleanup preflight source SHA mismatch")
    if manifest.get("gate_artifact_sha256") != gate_sha:
        raise ValueError("cleanup preflight gate SHA mismatch")
    if not preflight.get("framing", {}).get("preflight_only"):
        raise ValueError("cleanup artifact is not marked preflight_only")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_nonsynthetic_native_context_source.json",
    )
    parser.add_argument(
        "--cleanup-preflight",
        default="reports/phase5_prime_natural_source_control_cleanup_preflight.json",
    )
    parser.add_argument(
        "--prior-gate",
        default="reports/phase5_prime_nonsynthetic_native_gate.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_natural_source_control_cleanup_gate.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "candidate",
            "random_role",
            "deranged_role",
            "fixedpoint_free_shuffled_role",
            "content_cleanup_positive",
        ],
    )
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
    unknown = sorted(set(args.conditions) - CONDITIONS)
    if unknown:
        raise ValueError(f"unknown cleanup-gate conditions: {unknown}")

    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    prior_gate = _load_json(prior_gate_path)
    source_sha = base_gate._sha256(source_path)
    prior_gate_sha = base_gate._sha256(prior_gate_path)
    _validate_preflight(cleanup_preflight, source_sha, prior_gate_sha)
    protocol = _protocol_payload(cleanup_preflight, args.protocol)
    run_source = _source_with_protocol_plan(source, protocol)
    config = cleanup_preflight["config"]["source_config"]
    n_queries = int(protocol["required_queries_per_seed"])
    seeds = [int(seed) for seed in config["seeds"]]
    cooccurrence = str(config["cooccurrence"])

    print(
        f"device={device} protocol={protocol['protocol_name']} "
        f"source={source_path} source_sha={source_sha} conditions={args.conditions}"
    )

    raw: List[base_gate.GateResult] = []
    aggregates: Dict[str, dict] = {}
    for condition in args.conditions:
        cell: List[base_gate.GateResult] = []
        for seed in seeds:
            result = _run_seed_condition(
                condition=condition,
                seed=seed,
                source=run_source,
                source_path=source_path,
                source_sha=source_sha,
                D=int(config["D"]),
                N=int(config["N"]),
                K_roles=int(config["K_roles"]),
                C_codebook=int(config["C_codebook"]),
                context_roles=int(config["context_roles"]),
                cue_noise=float(config["cue_noise"]),
                scene_token_weight=float(config["token_weight"]),
                cooccurrence=cooccurrence,
                n_queries=n_queries,
                beta=30.0,
                max_iter=10,
                device=device,
            )
            cell.append(result)
            raw.append(result)
        key = (
            f"{condition}|protocol={protocol['protocol_name']}|"
            f"D={config['D']}|K={config['K_roles']}|N={config['N']}|"
            f"noise={config['cue_noise']}|scene_token=1|"
            f"token_weight={config['token_weight']}|"
            f"token_source={base_gate.SOURCE_NAME}|"
            f"context_roles={config['context_roles']}|cooc={cooccurrence}"
        )
        agg = base_gate._aggregate(cell)
        aggregates[key] = agg
        loo_min = agg["leave_one_seed_out_top1_min"]
        loo_max = agg["leave_one_seed_out_top1_max"]
        loo_text = (
            f" loo=[{loo_min:.4f},{loo_max:.4f}]"
            if loo_min is not None and loo_max is not None
            else ""
        )
        print(
            f"{key} top1={agg['top1_mean']:.4f} "
            f"CI=[{agg['wilson_lo']:.4f},{agg['wilson_hi']:.4f}] "
            f"scene_tix={agg['scene_tix']}/{agg['n_total']} "
            f"content_tix={agg['content_tix']}/{agg['n_total']} "
            f"ent=({agg['mean_scene_entropy']:.3f},"
            f"{agg['mean_content_entropy']:.3f}) "
            f"margin=({agg['mean_scene_margin']:.4f},"
            f"{agg['mean_content_margin']:.4f}){loo_text}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framing": {
            "phase": "5-prime natural source/control cleanup gate",
            "source_name": base_gate.SOURCE_NAME,
            "source_family": base_gate.SOURCE_FAMILY,
            "source_kind": base_gate.SOURCE_KIND,
            "cleanup_protocol": protocol["protocol_name"],
            "not_graduation": True,
            "no_full_matrix": True,
            "anti_homunculus": (
                "fixed source rows, fixed Report 092 query plan, fixed controls; "
                "no metric-triggered routing or best-of-N selection"
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "cleanup_preflight_artifact_path": str(cleanup_path),
            "cleanup_preflight_artifact_sha256": base_gate._sha256(cleanup_path),
            "prior_gate_artifact_path": str(prior_gate_path),
            "prior_gate_artifact_sha256": prior_gate_sha,
        },
        "cleanup_protocol_summary": {
            "protocol_name": protocol["protocol_name"],
            "pass_criteria": protocol["pass_criteria"],
            "aggregate": protocol["aggregate"],
        },
        "config": {
            **config,
            "n_queries": n_queries,
            "beta": 30.0,
            "max_iter": 10,
            "device": device,
            "conditions": args.conditions,
            "legacy_shuffled_role_excluded": True,
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
                "source_rows_available": r.source_rows_available,
                "source_rows_used": r.source_rows_used,
                "source_rows_invalid": r.source_rows_invalid,
                "source_rows_too_short": r.source_rows_too_short,
                "source_artifact_path": r.source_artifact_path,
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
