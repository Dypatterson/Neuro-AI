"""Preflight for a cleaned natural-source/control protocol after Report 091.

This script does not run candidate/control retrieval. It inspects the committed
Report 089 source rows and selected observed-role plans to determine whether a
stricter query/control protocol has enough support before any new gate is
allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Sequence

import torch

from energy_memory.phase2.corpus import build_vocabulary, load_corpus_splits
from scripts import phase5_prime_nonsynthetic_native_gate as gate


FREQ_CAPS: List[int | None] = [None, 512, 256, 128, 64, 32]
SPECIAL_ATOMS = {0, 1}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _cap_label(cap: int | None) -> str:
    return "none" if cap is None else f"le_{cap}"


def _load_vocab(repo_root: Path, corpus_source: str, c_codebook: int) -> dict:
    splits = load_corpus_splits(corpus_source, repo_root)
    vocab = build_vocabulary(splits["train"], max_vocab=max(2, c_codebook - 2))
    return {
        "id_to_token": list(vocab.id_to_token),
        "counts": dict(vocab.counts),
    }


def _token_for_atom(vocab: dict, atom: int) -> str:
    id_to_token = vocab["id_to_token"]
    if 0 <= atom < len(id_to_token):
        return str(id_to_token[atom])
    return f"<atom:{atom}>"


def _seed_atom_counts(rows: Sequence[Sequence[int]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(int(atom) for atom in row)
    return counts


def _role_permutation(seed: int, k_roles: int) -> List[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + k_roles)
    return [int(role) for role in gate.EXP44._role_permutation(k_roles, generator=generator)]


def _role_derangement(seed: int, k_roles: int) -> List[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed * 5003 + k_roles)
    return [int(role) for role in gate.EXP44._role_derangement(k_roles, generator=generator)]


def _fixedpoint_free_shuffle(seed: int, k_roles: int) -> List[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + k_roles)
    return [int(role) for role in gate.EXP44._role_derangement(k_roles, generator=generator)]


def _eligible_triples_for_seed(
    *,
    seed: int,
    rows: Sequence[Sequence[int]],
    selected_context_roles: Sequence[Sequence[int]],
    k_roles: int,
    cap: int | None,
) -> List[dict]:
    atom_counts = _seed_atom_counts(rows)
    triples: List[dict] = []
    for scene, row in enumerate(rows):
        observed_roles = [int(role) for role in selected_context_roles[scene]]
        for query_role in range(k_roles):
            if query_role in observed_roles:
                continue
            target_atom = int(row[query_role])
            target_frequency = int(atom_counts[target_atom])
            if target_atom in SPECIAL_ATOMS:
                continue
            if row.count(target_atom) != 1:
                continue
            if cap is not None and target_frequency > cap:
                continue
            for known_role in observed_roles:
                triples.append({
                    "scene": int(scene),
                    "known_role": int(known_role),
                    "query_role": int(query_role),
                    "observed_roles": observed_roles,
                    "target_atom": target_atom,
                    "target_frequency": target_frequency,
                })
    return triples


def _select_queries(
    *,
    seed: int,
    eligible: Sequence[dict],
    n_queries: int,
    cap: int | None,
    k_roles: int,
    n_rows: int,
) -> List[dict]:
    ordered = sorted(
        eligible,
        key=lambda item: (
            item["scene"],
            item["query_role"],
            item["known_role"],
            item["target_atom"],
        ),
    )
    if len(ordered) <= n_queries:
        return [dict(item) for item in ordered]
    cap_value = 0 if cap is None else int(cap)
    rng = random.Random(seed * 7001 + n_rows * 31 + n_queries * 17 + cap_value * 13 + k_roles)
    indices = sorted(rng.sample(range(len(ordered)), n_queries))
    return [dict(ordered[index]) for index in indices]


def _same_scene_opportunities(
    *,
    rows: Sequence[Sequence[int]],
    selected: Sequence[dict],
    seed: int,
    k_roles: int,
) -> dict:
    legacy_shuffle = _role_permutation(seed, k_roles)
    derangement = _role_derangement(seed, k_roles)
    fixedpoint_free_shuffle = _fixedpoint_free_shuffle(seed, k_roles)
    counts = Counter()
    for item in selected:
        row = rows[int(item["scene"])]
        query_role = int(item["query_role"])
        target_atom = int(item["target_atom"])
        random_role = (query_role + 1) % k_roles
        deranged_role = derangement[query_role]
        legacy_shuffle_role = legacy_shuffle[query_role]
        fixedpoint_free_shuffle_role = fixedpoint_free_shuffle[query_role]
        counts["random"] += int(int(row[random_role]) == target_atom)
        counts["deranged"] += int(int(row[deranged_role]) == target_atom)
        counts["legacy_shuffled"] += int(int(row[legacy_shuffle_role]) == target_atom)
        counts["fixedpoint_free_shuffled"] += int(
            int(row[fixedpoint_free_shuffle_role]) == target_atom
        )
        counts["legacy_shuffle_fixed_point"] += int(legacy_shuffle_role == query_role)
    total = len(selected)
    return {
        "random_exact_rate": _fraction(counts["random"], total),
        "deranged_exact_rate": _fraction(counts["deranged"], total),
        "legacy_shuffled_exact_rate": _fraction(counts["legacy_shuffled"], total),
        "fixedpoint_free_shuffled_exact_rate": _fraction(
            counts["fixedpoint_free_shuffled"],
            total,
        ),
        "legacy_shuffle_fixed_point_rate": _fraction(
            counts["legacy_shuffle_fixed_point"],
            total,
        ),
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "legacy_shuffle": legacy_shuffle,
        "fixedpoint_free_shuffle": fixedpoint_free_shuffle,
    }


def _target_summary(selected: Sequence[dict], vocab: dict) -> dict:
    atoms = [int(item["target_atom"]) for item in selected]
    frequencies = [int(item["target_frequency"]) for item in selected]
    counts = Counter(atoms)
    return {
        "special_fraction": _fraction(sum(atom in SPECIAL_ATOMS for atom in atoms), len(atoms)),
        "target_frequency": _stats([float(freq) for freq in frequencies]),
        "distinct_target_atoms": int(len(counts)),
        "top_target_atoms": [
            {
                "atom": int(atom),
                "token": _token_for_atom(vocab, int(atom)),
                "count": int(count),
                "fraction": _fraction(int(count), len(atoms)),
            }
            for atom, count in counts.most_common(12)
        ],
    }


def _protocol_for_cap(
    *,
    cap: int | None,
    source: dict,
    vocab: dict,
    n_queries: int,
) -> dict:
    config = source["config"]
    k_roles = int(config["K_roles"])
    per_seed = []
    selected_plans: Dict[str, List[dict]] = {}
    for seed in [int(seed) for seed in config["seeds"]]:
        seed_key = str(seed)
        rows = source["source_rows_by_seed"][seed_key]
        selected_context_roles = source["selected_context_roles_by_seed"][seed_key]
        eligible = _eligible_triples_for_seed(
            seed=seed,
            rows=rows,
            selected_context_roles=selected_context_roles,
            k_roles=k_roles,
            cap=cap,
        )
        selected = _select_queries(
            seed=seed,
            eligible=eligible,
            n_queries=n_queries,
            cap=cap,
            k_roles=k_roles,
            n_rows=int(config["N"]),
        )
        opportunities = _same_scene_opportunities(
            rows=rows,
            selected=selected,
            seed=seed,
            k_roles=k_roles,
        )
        duplicates = [
            int(rows[int(item["scene"])].count(int(item["target_atom"])) > 1)
            for item in selected
        ]
        seed_summary = {
            "seed": seed,
            "eligible_triples": int(len(eligible)),
            "selected_queries": int(len(selected)),
            "selected_query_fraction_of_required": _fraction(len(selected), n_queries),
            "same_row_duplicate_fraction": _fraction(sum(duplicates), len(selected)),
            "target_summary": _target_summary(selected, vocab),
            "same_scene_exact_opportunity": opportunities,
        }
        per_seed.append(seed_summary)
        selected_plans[seed_key] = [
            {
                "scene": int(item["scene"]),
                "known_role": int(item["known_role"]),
                "query_role": int(item["query_role"]),
                "observed_roles": [int(role) for role in item["observed_roles"]],
                "target_atom": int(item["target_atom"]),
                "target_frequency": int(item["target_frequency"]),
            }
            for item in selected
        ]

    required_selected = [row["selected_queries"] == n_queries for row in per_seed]
    enough_support = [row["eligible_triples"] >= n_queries for row in per_seed]
    no_special = [
        row["target_summary"]["special_fraction"] == 0.0
        for row in per_seed
    ]
    no_duplicates = [
        row["same_row_duplicate_fraction"] == 0.0
        for row in per_seed
    ]
    no_random = [
        row["same_scene_exact_opportunity"]["random_exact_rate"] == 0.0
        for row in per_seed
    ]
    no_deranged = [
        row["same_scene_exact_opportunity"]["deranged_exact_rate"] == 0.0
        for row in per_seed
    ]
    no_fixed_shuffle = [
        row["same_scene_exact_opportunity"]["fixedpoint_free_shuffled_exact_rate"] == 0.0
        for row in per_seed
    ]
    pass_criteria = {
        "eligible_triples_ge_n_queries_all_seeds": all(enough_support),
        "selected_queries_eq_n_queries_all_seeds": all(required_selected),
        "selected_targets_no_special_atoms": all(no_special),
        "selected_targets_no_same_row_duplicates": all(no_duplicates),
        "random_role_same_scene_exact_opportunity_zero": all(no_random),
        "deranged_role_same_scene_exact_opportunity_zero": all(no_deranged),
        "fixedpoint_free_shuffled_same_scene_exact_opportunity_zero": all(no_fixed_shuffle),
    }
    aggregate = {
        "eligible_triples": _stats([float(row["eligible_triples"]) for row in per_seed]),
        "selected_queries": _stats([float(row["selected_queries"]) for row in per_seed]),
        "target_frequency_mean": _stats([
            float(row["target_summary"]["target_frequency"]["mean"])
            for row in per_seed
        ]),
        "target_frequency_median": _stats([
            float(row["target_summary"]["target_frequency"]["median"])
            for row in per_seed
        ]),
        "distinct_target_atoms": _stats([
            float(row["target_summary"]["distinct_target_atoms"])
            for row in per_seed
        ]),
        "legacy_shuffle_fixed_point_rate": _stats([
            float(row["same_scene_exact_opportunity"]["legacy_shuffle_fixed_point_rate"])
            for row in per_seed
        ]),
        "legacy_shuffled_exact_rate": _stats([
            float(row["same_scene_exact_opportunity"]["legacy_shuffled_exact_rate"])
            for row in per_seed
        ]),
    }
    return {
        "protocol_name": f"non_special_unique_target_freq_{_cap_label(cap)}",
        "frequency_cap": cap,
        "required_queries_per_seed": n_queries,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(pass_criteria.values()),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "selected_query_plan_by_seed": selected_plans,
    }


def _select_recommended_protocol(protocols: Sequence[dict]) -> str | None:
    passing = [protocol for protocol in protocols if protocol["passes_all_criteria"]]
    if not passing:
        return None
    def sort_key(protocol: dict) -> tuple[int, int]:
        cap = protocol["frequency_cap"]
        return (0 if cap is not None else 1, int(cap or 10**9))
    return sorted(passing, key=sort_key)[0]["protocol_name"]


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
        "--residual",
        default="reports/phase5_prime_nonsynthetic_native_residual_analysis.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_natural_source_control_cleanup_preflight.json",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--n_queries", type=int, default=512)
    args = parser.parse_args()

    source_path = Path(args.source)
    gate_path = Path(args.gate)
    residual_path = Path(args.residual)
    source = json.loads(source_path.read_text())
    gate_payload = json.loads(gate_path.read_text())
    residual = json.loads(residual_path.read_text())

    if source.get("framing", {}).get("source_name") != gate.SOURCE_NAME:
        raise ValueError("unexpected source artifact")
    source_sha = _sha256(source_path)
    gate_sha = _sha256(gate_path)
    residual_sha = _sha256(residual_path)
    if gate_payload.get("source_manifest", {}).get("source_artifact_sha256") != source_sha:
        raise ValueError("gate source SHA does not match source artifact")
    if residual.get("source_manifest", {}).get("gate_artifact_sha256") != gate_sha:
        raise ValueError("residual analysis SHA does not match gate artifact")

    vocab = _load_vocab(
        Path(args.repo_root),
        source.get("config", {}).get("corpus_source", "repo_sample"),
        int(source["config"]["C_codebook"]),
    )
    protocols = [
        _protocol_for_cap(
            cap=cap,
            source=source,
            vocab=vocab,
            n_queries=args.n_queries,
        )
        for cap in FREQ_CAPS
    ]
    recommended = _select_recommended_protocol(protocols)
    payload = {
        "framing": {
            "phase": "5-prime natural source/control cleanup preflight",
            "source_name": gate.SOURCE_NAME,
            "preflight_only": True,
            "analysis_only": True,
            "not_graduation": True,
            "no_candidate_control_retrieval": True,
            "description": (
                "Support and exact-opportunity preflight for a stricter natural "
                "source/control protocol after Report 091. No top1 gate is run."
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "gate_artifact_path": str(gate_path),
            "gate_artifact_sha256": gate_sha,
            "residual_artifact_path": str(residual_path),
            "residual_artifact_sha256": residual_sha,
        },
        "config": {
            "n_queries": args.n_queries,
            "frequency_caps": [_cap_label(cap) for cap in FREQ_CAPS],
            "special_atoms": sorted(SPECIAL_ATOMS),
            "source_config": source["config"],
        },
        "protocols": protocols,
        "recommended_protocol": recommended,
        "decision_read": (
            "Preflight only. A passing protocol may be used to write a separate "
            "gate precommit, but does not authorize a candidate/control top1 run."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "recommended_protocol": recommended,
        "protocols": [
            {
                "protocol_name": protocol["protocol_name"],
                "passes_all_criteria": protocol["passes_all_criteria"],
                "eligible_triples_min": protocol["aggregate"]["eligible_triples"]["min"],
                "target_frequency_mean": protocol["aggregate"]["target_frequency_mean"]["mean"],
                "legacy_shuffle_fixed_point_rate_mean": protocol["aggregate"][
                    "legacy_shuffle_fixed_point_rate"
                ]["mean"],
            }
            for protocol in protocols
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
