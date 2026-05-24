"""Build deterministic codebook scrambles for Phase 5 M1 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from energy_memory.phase2.persistence import load_codebook, save_codebook  # noqa: E402


DEFAULT_METHOD = "rowwise_coordinate_permutation"
DEFAULT_SEED = 4242
DEFAULT_OUTPUT = Path("/private/tmp/phase5_m1_codebook_scrambled_seed4242.pt")
DEFAULT_MANIFEST = Path(
    "/private/tmp/phase5_m1_codebook_scrambled_seed4242_manifest.json",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("codebook scrambling requires torch") from _IMPORT_ERROR


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rowwise_coordinate_permutation(
    codebook: "torch.Tensor",
    *,
    seed: int = DEFAULT_SEED,
) -> "torch.Tensor":
    """Permute coordinates independently per row while preserving row norms."""
    _require_torch()
    if codebook.ndim != 2:
        raise ValueError("codebook must be a [N, D] tensor")
    n_rows, dim = int(codebook.shape[0]), int(codebook.shape[1])
    if n_rows <= 0 or dim <= 0:
        raise ValueError("codebook must be non-empty")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scrambled = torch.empty_like(codebook)
    for row_index in range(n_rows):
        permutation = torch.randperm(dim, generator=generator, device="cpu")
        scrambled[row_index] = codebook[row_index, permutation]
    return scrambled


def scramble_codebook(
    codebook: "torch.Tensor",
    *,
    method: str = DEFAULT_METHOD,
    seed: int = DEFAULT_SEED,
) -> "torch.Tensor":
    if method != DEFAULT_METHOD:
        raise ValueError(f"unsupported scramble method: {method}")
    return rowwise_coordinate_permutation(codebook, seed=seed)


def _identity(path: Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "basename": resolved.name,
    }


def build_manifest(
    *,
    input_path: Path,
    output_path: Path,
    source: "torch.Tensor",
    scrambled: "torch.Tensor",
    method: str,
    seed: int,
) -> Dict[str, Any]:
    _require_torch()
    source_norms = source.norm(dim=1)
    scrambled_norms = scrambled.norm(dim=1)
    row_norm_abs_diff = (source_norms - scrambled_norms).abs()
    global_norm_abs_diff = abs(
        float(source.norm().detach().cpu()) - float(scrambled.norm().detach().cpu())
    )
    return {
        "schema_version": 1,
        "artifact": "phase5_m1_codebook_scramble",
        "method": method,
        "seed": int(seed),
        "input_codebook": _identity(input_path),
        "output_codebook": _identity(output_path),
        "shape": [int(dim) for dim in source.shape],
        "dtype": str(source.dtype),
        "row_norm_max_abs_diff": float(row_norm_abs_diff.max().detach().cpu()),
        "row_norm_mean_abs_diff": float(row_norm_abs_diff.mean().detach().cpu()),
        "global_norm_abs_diff": global_norm_abs_diff,
        "registry_policy": (
            "scrambled controls are intentionally not added to "
            "config/phase5_m1_codebook_registry.json; codebook_not_in_registry "
            "is the expected warning"
        ),
    }


def write_scrambled_codebook(
    *,
    input_codebook: Path,
    output: Path = DEFAULT_OUTPUT,
    manifest: Path = DEFAULT_MANIFEST,
    method: str = DEFAULT_METHOD,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    _require_torch()
    source = load_codebook(Path(input_codebook), device="cpu")
    scrambled = scramble_codebook(source, method=method, seed=seed)
    output = Path(output)
    manifest = Path(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    save_codebook(scrambled, output)
    payload = build_manifest(
        input_path=Path(input_codebook),
        output_path=output,
        source=source,
        scrambled=scrambled,
        method=method,
        seed=seed,
    )
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic rowwise coordinate-permuted codebook control.",
    )
    parser.add_argument("--input-codebook", required=True, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=[DEFAULT_METHOD])
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = write_scrambled_codebook(
        input_codebook=args.input_codebook,
        output=args.output,
        manifest=args.manifest,
        method=args.method,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
