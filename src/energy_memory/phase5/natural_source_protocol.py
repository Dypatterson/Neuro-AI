"""Reusable cleaned natural-source protocol helpers for Phase 5'.

The helpers here are deliberately data-only: callers provide source rows,
observed context roles, and protocol settings. Report scripts may load files
and build vocabularies, but this module does not know about report paths.
"""

from __future__ import annotations

import random
from collections import Counter
from statistics import mean, median, pstdev
from typing import Dict, List, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


SPECIAL_ATOMS = {0, 1}
DEFAULT_FREQUENCY_CAPS: List[int | None] = [None, 512, 256, 128, 64, 32]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "natural-source role controls require torch"
        ) from _IMPORT_ERROR


def stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def cap_label(cap: int | None) -> str:
    return "none" if cap is None else f"le_{cap}"


def seed_atom_counts(rows: Sequence[Sequence[int]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts.update(int(atom) for atom in row)
    return counts


def role_permutation(seed: int, k_roles: int) -> List[int]:
    """Report-compatible role permutation for legacy shuffled controls."""
    _require_torch()
    if k_roles <= 0:
        raise ValueError("k_roles must be positive")
    if k_roles == 1:
        return [0]
    generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + k_roles)
    perm = torch.randperm(k_roles, generator=generator).tolist()
    if all(i == p for i, p in enumerate(perm)):
        perm = perm[1:] + perm[:1]
    return [int(role) for role in perm]


def role_derangement(seed: int, k_roles: int, *, seed_multiplier: int = 5003) -> List[int]:
    """Report-compatible no-fixed-point role control."""
    _require_torch()
    if k_roles <= 0:
        raise ValueError("k_roles must be positive")
    if k_roles == 1:
        return [0]
    generator = torch.Generator(device="cpu").manual_seed(
        seed * seed_multiplier + k_roles
    )
    for _ in range(64):
        perm = torch.randperm(k_roles, generator=generator).tolist()
        if all(i != p for i, p in enumerate(perm)):
            return [int(role) for role in perm]
    return list(range(1, k_roles)) + [0]


def fixedpoint_free_shuffle(seed: int, k_roles: int) -> List[int]:
    """No-fixed-point replacement for the legacy shuffled-role control."""
    return role_derangement(seed, k_roles, seed_multiplier=4001)


def eligible_triples_for_seed(
    *,
    rows: Sequence[Sequence[int]],
    selected_context_roles: Sequence[Sequence[int]],
    k_roles: int,
    cap: int | None,
    special_atoms: set[int] | None = None,
) -> List[dict]:
    if k_roles <= 0:
        raise ValueError("k_roles must be positive")
    if len(rows) != len(selected_context_roles):
        raise ValueError("rows and selected_context_roles must have the same length")
    blocked_atoms = SPECIAL_ATOMS if special_atoms is None else set(special_atoms)
    atom_counts = seed_atom_counts(rows)
    triples: List[dict] = []
    for scene, row in enumerate(rows):
        if len(row) != k_roles:
            raise ValueError(f"row {scene} is not k_roles long")
        observed_roles = [int(role) for role in selected_context_roles[scene]]
        for role in observed_roles:
            if role < 0 or role >= k_roles:
                raise ValueError(f"observed role {role} out of range")
        for query_role in range(k_roles):
            if query_role in observed_roles:
                continue
            target_atom = int(row[query_role])
            target_frequency = int(atom_counts[target_atom])
            if target_atom in blocked_atoms:
                continue
            if [int(atom) for atom in row].count(target_atom) != 1:
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


def select_queries(
    *,
    seed: int,
    eligible: Sequence[dict],
    n_queries: int,
    cap: int | None,
    k_roles: int,
    n_rows: int,
) -> List[dict]:
    if n_queries <= 0:
        return []
    ordered = sorted(
        eligible,
        key=lambda item: (
            int(item["scene"]),
            int(item["query_role"]),
            int(item["known_role"]),
            int(item["target_atom"]),
        ),
    )
    if len(ordered) <= n_queries:
        return [dict(item) for item in ordered]
    cap_value = 0 if cap is None else int(cap)
    rng = random.Random(
        seed * 7001 + n_rows * 31 + n_queries * 17 + cap_value * 13 + k_roles
    )
    indices = sorted(rng.sample(range(len(ordered)), n_queries))
    return [dict(ordered[index]) for index in indices]


def same_scene_opportunities(
    *,
    rows: Sequence[Sequence[int]],
    selected: Sequence[dict],
    seed: int,
    k_roles: int,
) -> dict:
    legacy_shuffle = role_permutation(seed, k_roles)
    derangement = role_derangement(seed, k_roles)
    no_fixed_shuffle = fixedpoint_free_shuffle(seed, k_roles)
    counts = Counter()
    for item in selected:
        row = rows[int(item["scene"])]
        query_role = int(item["query_role"])
        target_atom = int(item["target_atom"])
        random_role = (query_role + 1) % k_roles
        deranged_role = derangement[query_role]
        legacy_shuffle_role = legacy_shuffle[query_role]
        fixedpoint_free_shuffle_role = no_fixed_shuffle[query_role]
        counts["random"] += int(int(row[random_role]) == target_atom)
        counts["deranged"] += int(int(row[deranged_role]) == target_atom)
        counts["legacy_shuffled"] += int(int(row[legacy_shuffle_role]) == target_atom)
        counts["fixedpoint_free_shuffled"] += int(
            int(row[fixedpoint_free_shuffle_role]) == target_atom
        )
        counts["legacy_shuffle_fixed_point"] += int(
            legacy_shuffle_role == query_role
        )
    total = len(selected)
    return {
        "random_exact_rate": fraction(counts["random"], total),
        "deranged_exact_rate": fraction(counts["deranged"], total),
        "legacy_shuffled_exact_rate": fraction(counts["legacy_shuffled"], total),
        "fixedpoint_free_shuffled_exact_rate": fraction(
            counts["fixedpoint_free_shuffled"],
            total,
        ),
        "legacy_shuffle_fixed_point_rate": fraction(
            counts["legacy_shuffle_fixed_point"],
            total,
        ),
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "legacy_shuffle": legacy_shuffle,
        "fixedpoint_free_shuffle": no_fixed_shuffle,
    }


def token_for_atom(vocab: dict, atom: int) -> str:
    id_to_token = vocab.get("id_to_token", [])
    if 0 <= atom < len(id_to_token):
        return str(id_to_token[atom])
    return f"<atom:{atom}>"


def target_summary(
    selected: Sequence[dict],
    vocab: dict,
    *,
    special_atoms: set[int] | None = None,
) -> dict:
    blocked_atoms = SPECIAL_ATOMS if special_atoms is None else set(special_atoms)
    atoms = [int(item["target_atom"]) for item in selected]
    frequencies = [int(item["target_frequency"]) for item in selected]
    counts = Counter(atoms)
    return {
        "special_fraction": fraction(
            sum(atom in blocked_atoms for atom in atoms),
            len(atoms),
        ),
        "target_frequency": stats([float(freq) for freq in frequencies]),
        "distinct_target_atoms": int(len(counts)),
        "top_target_atoms": [
            {
                "atom": int(atom),
                "token": token_for_atom(vocab, int(atom)),
                "count": int(count),
                "fraction": fraction(int(count), len(atoms)),
            }
            for atom, count in counts.most_common(12)
        ],
    }


def protocol_for_frequency_cap(
    *,
    cap: int | None,
    source: dict,
    vocab: dict,
    n_queries: int,
    special_atoms: set[int] | None = None,
) -> dict:
    config = source["config"]
    k_roles = int(config["K_roles"])
    per_seed = []
    selected_plans: Dict[str, List[dict]] = {}
    for seed in [int(seed) for seed in config["seeds"]]:
        seed_key = str(seed)
        rows = source["source_rows_by_seed"][seed_key]
        selected_context_roles = source["selected_context_roles_by_seed"][seed_key]
        eligible = eligible_triples_for_seed(
            rows=rows,
            selected_context_roles=selected_context_roles,
            k_roles=k_roles,
            cap=cap,
            special_atoms=special_atoms,
        )
        selected = select_queries(
            seed=seed,
            eligible=eligible,
            n_queries=n_queries,
            cap=cap,
            k_roles=k_roles,
            n_rows=int(config["N"]),
        )
        opportunities = same_scene_opportunities(
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
            "selected_query_fraction_of_required": fraction(len(selected), n_queries),
            "same_row_duplicate_fraction": fraction(sum(duplicates), len(selected)),
            "target_summary": target_summary(
                selected,
                vocab,
                special_atoms=special_atoms,
            ),
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
        row["same_scene_exact_opportunity"][
            "fixedpoint_free_shuffled_exact_rate"
        ] == 0.0
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
        "eligible_triples": stats([
            float(row["eligible_triples"]) for row in per_seed
        ]),
        "selected_queries": stats([
            float(row["selected_queries"]) for row in per_seed
        ]),
        "target_frequency_mean": stats([
            float(row["target_summary"]["target_frequency"]["mean"])
            for row in per_seed
        ]),
        "target_frequency_median": stats([
            float(row["target_summary"]["target_frequency"]["median"])
            for row in per_seed
        ]),
        "distinct_target_atoms": stats([
            float(row["target_summary"]["distinct_target_atoms"])
            for row in per_seed
        ]),
        "legacy_shuffle_fixed_point_rate": stats([
            float(
                row["same_scene_exact_opportunity"][
                    "legacy_shuffle_fixed_point_rate"
                ]
            )
            for row in per_seed
        ]),
        "legacy_shuffled_exact_rate": stats([
            float(row["same_scene_exact_opportunity"]["legacy_shuffled_exact_rate"])
            for row in per_seed
        ]),
    }
    return {
        "protocol_name": f"non_special_unique_target_freq_{cap_label(cap)}",
        "frequency_cap": cap,
        "required_queries_per_seed": n_queries,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(pass_criteria.values()),
        "aggregate": aggregate,
        "per_seed": per_seed,
        "selected_query_plan_by_seed": selected_plans,
    }


def select_recommended_protocol(protocols: Sequence[dict]) -> str | None:
    passing = [protocol for protocol in protocols if protocol["passes_all_criteria"]]
    if not passing:
        return None

    def sort_key(protocol: dict) -> tuple[int, int]:
        cap = protocol["frequency_cap"]
        return (0 if cap is not None else 1, int(cap or 10**9))

    return sorted(passing, key=sort_key)[0]["protocol_name"]


def protocol_payload(preflight: dict, protocol_name: str | None) -> dict:
    selected = protocol_name or preflight.get("recommended_protocol")
    if not selected:
        raise ValueError("cleanup preflight does not name a recommended protocol")
    for protocol in preflight.get("protocols", []):
        if protocol.get("protocol_name") == selected:
            if not protocol.get("passes_all_criteria"):
                raise ValueError(f"cleanup protocol does not pass: {selected}")
            return protocol
    raise ValueError(f"cleanup protocol not found: {selected}")


def source_with_protocol_plan(source: dict, protocol: dict) -> dict:
    out = dict(source)
    out["query_plan_by_seed"] = protocol["selected_query_plan_by_seed"]
    out["config"] = dict(source["config"])
    out["config"]["n_queries"] = int(protocol["required_queries_per_seed"])
    return out


def validate_cleanup_preflight(
    preflight: dict,
    *,
    source_sha: str,
    gate_sha: str,
) -> None:
    manifest = preflight.get("source_manifest", {})
    if manifest.get("source_artifact_sha256") != source_sha:
        raise ValueError("cleanup preflight source SHA mismatch")
    if manifest.get("gate_artifact_sha256") != gate_sha:
        raise ValueError("cleanup preflight gate SHA mismatch")
    if not preflight.get("framing", {}).get("preflight_only"):
        raise ValueError("cleanup artifact is not marked preflight_only")
