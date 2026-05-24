"""Build deterministic random/unit codebook controls for Phase 5 M1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from energy_memory.phase2.persistence import save_codebook  # noqa: E402


DEFAULT_METHOD = "iid_unit_complex_phases"
DEFAULT_SEED = 20260524
DEFAULT_ROWS = 2050
DEFAULT_DIM = 4096
DEFAULT_OUTPUT = Path("/private/tmp/phase5_m1_codebook_random_unit_seed20260524.pt")
DEFAULT_MANIFEST = Path(
    "/private/tmp/phase5_m1_codebook_random_unit_seed20260524_manifest.json",
)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("random/unit codebook controls require torch") from _IMPORT_ERROR


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iid_unit_complex_codebook(
    *,
    rows: int = DEFAULT_ROWS,
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
) -> "torch.Tensor":
    """Generate an IID FHRR-style codebook with unit-magnitude coordinates."""
    _require_torch()
    if rows <= 0:
        raise ValueError("rows must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    phase = torch.rand(
        (int(rows), int(dim)),
        generator=generator,
        device="cpu",
    ) * (2.0 * math.pi)
    return torch.polar(torch.ones_like(phase), phase).to(torch.complex64)


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
    output_path: Path,
    codebook: "torch.Tensor",
    method: str,
    seed: int,
) -> Dict[str, Any]:
    _require_torch()
    row_norms = codebook.norm(dim=1)
    coord_abs = codebook.abs()
    return {
        "schema_version": 1,
        "artifact": "phase5_m1_random_unit_codebook_control",
        "method": method,
        "seed": int(seed),
        "output_codebook": _identity(output_path),
        "shape": [int(dim) for dim in codebook.shape],
        "dtype": str(codebook.dtype),
        "row_norm_min": float(row_norms.min().detach().cpu()),
        "row_norm_max": float(row_norms.max().detach().cpu()),
        "row_norm_mean": float(row_norms.mean().detach().cpu()),
        "coordinate_abs_min": float(coord_abs.min().detach().cpu()),
        "coordinate_abs_max": float(coord_abs.max().detach().cpu()),
        "coordinate_abs_mean": float(coord_abs.mean().detach().cpu()),
        "lineage_policy": (
            "shape-matched IID FHRR unit control; no corpus, vocabulary, "
            "Phase 3c training, or M1 snapshot rows are used"
        ),
        "registry_policy": (
            "random/unit controls are intentionally not added to "
            "config/phase5_m1_codebook_registry.json; codebook_not_in_registry "
            "is the expected warning"
        ),
    }


def write_random_unit_codebook(
    *,
    output: Path = DEFAULT_OUTPUT,
    manifest: Path = DEFAULT_MANIFEST,
    rows: int = DEFAULT_ROWS,
    dim: int = DEFAULT_DIM,
    seed: int = DEFAULT_SEED,
    method: str = DEFAULT_METHOD,
) -> Dict[str, Any]:
    _require_torch()
    if method != DEFAULT_METHOD:
        raise ValueError(f"unsupported random-control method: {method}")

    output = Path(output)
    manifest = Path(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    codebook = iid_unit_complex_codebook(rows=rows, dim=dim, seed=seed)
    save_codebook(codebook, output)
    payload = build_manifest(
        output_path=output,
        codebook=codebook,
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
        description="Create a deterministic random/unit FHRR codebook control.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    parser.add_argument("--rows", default=DEFAULT_ROWS, type=int)
    parser.add_argument("--dim", default=DEFAULT_DIM, type=int)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=[DEFAULT_METHOD])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = write_random_unit_codebook(
        output=args.output,
        manifest=args.manifest,
        rows=args.rows,
        dim=args.dim,
        seed=args.seed,
        method=args.method,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
