"""Analysis-only localization for Report 090 dirty role-negative controls.

This script replays the fixed Report 090 gate deterministically to collect
per-query diagnostics. It does not change the source, query plan, controls, or
retrieval protocol, and it does not run a new decision gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence

import torch

from energy_memory.phase2.corpus import build_vocabulary, load_corpus_splits
from energy_memory.substrate.torch_fhrr import TorchFHRR
from scripts import phase5_prime_nonsynthetic_native_gate as gate


CONTROL_CONDITIONS = ["random_role", "deranged_role", "shuffled_role"]


def _load_vocab(repo_root: Path, corpus_source: str, c_codebook: int) -> dict:
    splits = load_corpus_splits(corpus_source, repo_root)
    vocab = build_vocabulary(splits["train"], max_vocab=max(2, c_codebook - 2))
    return {
        "id_to_token": list(vocab.id_to_token),
        "token_to_id": dict(vocab.token_to_id),
        "counts": dict(vocab.counts),
    }


def _token_for_atom(vocab_payload: dict, atom: int) -> str:
    id_to_token = vocab_payload["id_to_token"]
    if 0 <= atom < len(id_to_token):
        return str(id_to_token[atom])
    return f"<atom:{atom}>"


def _role_delta(query_role: int, unbind_role: int, k_roles: int) -> int:
    return int((unbind_role - query_role) % k_roles)


def _seed_atom_counts(rows: Sequence[Sequence[int]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(int(atom) for atom in row)
    return counts


def _query_details_for_condition(
    *,
    condition: str,
    seed: int,
    source: dict,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    cue_noise: float,
    scene_token_weight: float,
    n_queries: int,
    beta: float,
    max_iter: int,
    device: str,
    vocab_payload: dict,
) -> List[dict]:
    if condition not in gate.CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "content_cleanup_positive":
        raise ValueError("content_cleanup_positive has no wrong-role residual")

    seed_key = str(seed)
    rows_raw = source["source_rows_by_seed"][seed_key]
    query_plan = source["query_plan_by_seed"][seed_key]
    if len(rows_raw) != N:
        raise ValueError(f"source rows for seed {seed} have len {len(rows_raw)} != {N}")
    if len(query_plan) != n_queries:
        raise ValueError(
            f"query plan for seed {seed} has len {len(query_plan)} != {n_queries}"
        )

    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    roles = gate._native_roles(fhrr, K_roles)
    content = fhrr.random_vectors(C_codebook)
    rows = torch.tensor(rows_raw, dtype=torch.long, device=fhrr.device)
    scene_matrix = gate._scene_bundles(
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

    perm_generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + K_roles)
    role_shuffle = torch.tensor(
        gate.EXP44._role_permutation(K_roles, generator=perm_generator),
        dtype=torch.long,
        device=fhrr.device,
    )
    derange_generator = torch.Generator(device="cpu").manual_seed(seed * 5003 + K_roles)
    role_derangement = torch.tensor(
        gate.EXP44._role_derangement(K_roles, generator=derange_generator),
        dtype=torch.long,
        device=fhrr.device,
    )

    cue_role = known_role
    unbind_role = query_role
    if condition == "random_role" and K_roles > 1:
        unbind_role = (query_role + 1) % K_roles
    elif condition == "shuffled_role":
        cue_role = role_shuffle[known_role]
        unbind_role = role_shuffle[query_role]
    elif condition == "deranged_role":
        cue_role = role_derangement[known_role]
        unbind_role = role_derangement[query_role]

    known_atom = rows[scene_idx, known_role]
    cue = roles[cue_role] * content[known_atom]
    query_tokens = gate._query_context_tokens(fhrr, roles, content, rows, query_plan)
    cue = fhrr.normalize(cue + scene_token_weight * query_tokens)
    cue = gate.EXP44._perturb_batch(fhrr, cue, cue_noise)
    scene_state, scene_top_index, _scene_entropy, _scene_margin = (
        gate.EXP44._batched_hopfield_retrieve(
            fhrr,
            scene_matrix,
            cue,
            beta=beta,
            max_iter=max_iter,
        )
    )
    content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[unbind_role]))
    content_state, content_top_index, _content_entropy, _content_margin = (
        gate.EXP44._batched_hopfield_retrieve(
            fhrr,
            content,
            content_query,
            beta=beta,
            max_iter=max_iter,
        )
    )
    pred = torch.argmax((content_state @ content.conj().T).real / content.shape[1], dim=1)

    seed_counts = _seed_atom_counts(rows_raw)
    details: List[dict] = []
    for idx, item in enumerate(query_plan):
        scene = int(scene_idx[idx].detach().cpu())
        retrieved_scene = int(scene_top_index[idx].detach().cpu())
        q_role = int(query_role[idx].detach().cpu())
        u_role = int(unbind_role[idx].detach().cpu())
        t_atom = int(target_atom[idx].detach().cpu())
        p_atom = int(pred[idx].detach().cpu())
        retrieved_unbind_atom = int(rows[retrieved_scene, u_role].detach().cpu())
        same_scene_unbind_atom = int(rows[scene, u_role].detach().cpu())
        retrieved_scene_atoms = [int(atom) for atom in rows_raw[retrieved_scene]]
        observed_roles = [int(role) for role in item["observed_roles"]]
        details.append({
            "condition": condition,
            "seed": seed,
            "query_index": idx,
            "scene": scene,
            "retrieved_scene": retrieved_scene,
            "scene_hit": retrieved_scene == scene,
            "known_role": int(known_role[idx].detach().cpu()),
            "query_role": q_role,
            "cue_role": int(cue_role[idx].detach().cpu()),
            "unbind_role": u_role,
            "role_delta": _role_delta(q_role, u_role, K_roles),
            "observed_roles": observed_roles,
            "target_atom": t_atom,
            "target_token": _token_for_atom(vocab_payload, t_atom),
            "target_seed_frequency": int(seed_counts[t_atom]),
            "target_is_special": t_atom in {0, 1},
            "target_is_top16_vocab": 2 <= t_atom < 18,
            "target_is_top64_vocab": 2 <= t_atom < 66,
            "pred_atom": p_atom,
            "pred_token": _token_for_atom(vocab_payload, p_atom),
            "content_top_index": int(content_top_index[idx].detach().cpu()),
            "correct": p_atom == t_atom,
            "same_scene_unbind_atom": same_scene_unbind_atom,
            "same_scene_unbind_token": _token_for_atom(
                vocab_payload,
                same_scene_unbind_atom,
            ),
            "same_scene_unbind_matches_target": same_scene_unbind_atom == t_atom,
            "retrieved_unbind_atom": retrieved_unbind_atom,
            "retrieved_unbind_token": _token_for_atom(
                vocab_payload,
                retrieved_unbind_atom,
            ),
            "retrieved_unbind_matches_target": retrieved_unbind_atom == t_atom,
            "target_in_retrieved_scene_any_role": t_atom in retrieved_scene_atoms,
            "target_count_in_retrieved_scene": retrieved_scene_atoms.count(t_atom),
        })
    return details


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_bool(rows: Sequence[dict], key: str) -> float:
    return _fraction(sum(1 for row in rows if row[key]), len(rows))


def _frequency_stats(rows: Sequence[dict]) -> dict:
    values = [int(row["target_seed_frequency"]) for row in rows]
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def _top_atoms(rows: Sequence[dict], *, limit: int = 12) -> List[dict]:
    counts = Counter(int(row["target_atom"]) for row in rows)
    tokens = {
        int(row["target_atom"]): str(row["target_token"])
        for row in rows
    }
    out = []
    for atom, count in counts.most_common(limit):
        out.append({
            "atom": int(atom),
            "token": tokens[atom],
            "count": int(count),
            "fraction": _fraction(int(count), len(rows)),
        })
    return out


def _summary_for_condition(details: Sequence[dict]) -> dict:
    rows = list(details)
    hits = [row for row in rows if row["correct"]]
    exact_retrieved = [
        row for row in hits if row["retrieved_unbind_matches_target"]
    ]
    exact_same_scene = [
        row for row in hits if row["same_scene_unbind_matches_target"]
    ]
    exact_retrieved_all = [
        row for row in rows if row["retrieved_unbind_matches_target"]
    ]
    exact_same_scene_all = [
        row for row in rows if row["same_scene_unbind_matches_target"]
    ]
    target_in_retrieved_scene_hits = [
        row for row in hits if row["target_in_retrieved_scene_any_role"]
    ]
    role_delta_hits = Counter(int(row["role_delta"]) for row in hits)
    role_delta_all = Counter(int(row["role_delta"]) for row in rows)

    return {
        "n_total": len(rows),
        "n_correct": len(hits),
        "top1": _fraction(len(hits), len(rows)),
        "scene_hit_rate": _mean_bool(rows, "scene_hit"),
        "hit_scene_hit_rate": _mean_bool(hits, "scene_hit"),
        "all_same_scene_unbind_match_rate": _fraction(
            len(exact_same_scene_all),
            len(rows),
        ),
        "hit_same_scene_unbind_match_rate": _fraction(
            len(exact_same_scene),
            len(hits),
        ),
        "all_retrieved_unbind_match_rate": _fraction(
            len(exact_retrieved_all),
            len(rows),
        ),
        "hit_retrieved_unbind_match_rate": _fraction(
            len(exact_retrieved),
            len(hits),
        ),
        "hit_target_in_retrieved_scene_any_role_rate": _fraction(
            len(target_in_retrieved_scene_hits),
            len(hits),
        ),
        "hit_target_frequency": _frequency_stats(hits),
        "all_target_frequency": _frequency_stats(rows),
        "hit_special_fraction": _mean_bool(hits, "target_is_special"),
        "all_special_fraction": _mean_bool(rows, "target_is_special"),
        "hit_top16_vocab_fraction": _mean_bool(hits, "target_is_top16_vocab"),
        "all_top16_vocab_fraction": _mean_bool(rows, "target_is_top16_vocab"),
        "hit_top64_vocab_fraction": _mean_bool(hits, "target_is_top64_vocab"),
        "all_top64_vocab_fraction": _mean_bool(rows, "target_is_top64_vocab"),
        "hit_role_delta_counts": {
            str(delta): int(count) for delta, count in sorted(role_delta_hits.items())
        },
        "all_role_delta_counts": {
            str(delta): int(count) for delta, count in sorted(role_delta_all.items())
        },
        "hit_top_target_atoms": _top_atoms(hits),
        "all_top_target_atoms": _top_atoms(rows),
        "sample_hits": hits[:8],
    }


def _gate_expected_by_condition_seed(gate_payload: dict) -> Dict[tuple[str, int], int]:
    expected: Dict[tuple[str, int], int] = {}
    for row in gate_payload.get("raw", []):
        expected[(str(row["condition"]), int(row["seed"]))] = int(row["n_correct"])
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_nonsynthetic_native_context_source.json",
    )
    parser.add_argument(
        "--gate",
        default="reports/phase5_prime_nonsynthetic_native_gate.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_nonsynthetic_native_residual_analysis.json",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=CONTROL_CONDITIONS,
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

    source_path = Path(args.source)
    gate_path = Path(args.gate)
    source = gate._load_json(source_path)
    gate_payload = gate._load_json(gate_path)
    config = gate_payload["config"]
    source_sha = gate._sha256(source_path)
    expected_sha = gate_payload.get("source_manifest", {}).get("source_artifact_sha256")
    if expected_sha != source_sha:
        raise ValueError(
            f"source SHA mismatch: gate={expected_sha} artifact={source_sha}"
        )
    if gate_payload.get("framing", {}).get("source_name") != gate.SOURCE_NAME:
        raise ValueError("gate payload is not the non-synthetic native source")

    vocab_payload = _load_vocab(
        Path(args.repo_root),
        source.get("config", {}).get("corpus_source", "repo_sample"),
        int(config["C_codebook"]),
    )
    expected = _gate_expected_by_condition_seed(gate_payload)

    print(
        f"device={device} source={source_path} gate={gate_path} "
        f"conditions={args.conditions}"
    )

    details_by_condition: Dict[str, List[dict]] = {}
    cross_checks = []
    for condition in args.conditions:
        if condition not in CONTROL_CONDITIONS:
            raise ValueError(f"condition is not a role-negative control: {condition}")
        condition_details: List[dict] = []
        for seed in config["seeds"]:
            seed_details = _query_details_for_condition(
                condition=condition,
                seed=int(seed),
                source=source,
                D=int(config["D"]),
                N=int(config["N"]),
                K_roles=int(config["K_roles"]),
                C_codebook=int(config["C_codebook"]),
                cue_noise=float(config["cue_noise"]),
                scene_token_weight=float(config["token_weight"]),
                n_queries=int(config["n_queries"]),
                beta=float(config["beta"]),
                max_iter=int(config.get("max_iter", 10)),
                device=device,
                vocab_payload=vocab_payload,
            )
            n_correct = sum(1 for row in seed_details if row["correct"])
            expected_correct = expected[(condition, int(seed))]
            if n_correct != expected_correct:
                raise ValueError(
                    f"{condition} seed {seed} recomputed {n_correct} "
                    f"!= gate {expected_correct}"
                )
            cross_checks.append({
                "condition": condition,
                "seed": int(seed),
                "n_correct": n_correct,
                "matches_gate": True,
            })
            condition_details.extend(seed_details)
        details_by_condition[condition] = condition_details

    summaries = {
        condition: _summary_for_condition(details)
        for condition, details in details_by_condition.items()
    }

    payload = {
        "framing": {
            "phase": "5-prime non-synthetic native provenance residual analysis",
            "source_name": gate.SOURCE_NAME,
            "analysis_only": True,
            "not_graduation": True,
            "no_new_gate": True,
            "description": (
                "Deterministic replay of Report 090 controls to localize "
                "dirty role-negative hits without changing the source or "
                "candidate/control protocol."
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "gate_artifact_path": str(gate_path),
            "gate_artifact_sha256": gate._sha256(gate_path),
        },
        "config": {
            **config,
            "analysis_device": device,
            "conditions": args.conditions,
        },
        "cross_checks": cross_checks,
        "summaries": summaries,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "gate_sha256": payload["source_manifest"]["gate_artifact_sha256"],
        "summaries": {
            condition: {
                "top1": summary["top1"],
                "hit_retrieved_unbind_match_rate": summary[
                    "hit_retrieved_unbind_match_rate"
                ],
                "hit_same_scene_unbind_match_rate": summary[
                    "hit_same_scene_unbind_match_rate"
                ],
                "hit_top64_vocab_fraction": summary["hit_top64_vocab_fraction"],
            }
            for condition, summary in summaries.items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
