"""Persistence helpers for Phase 2 reproducibility artifacts."""

from __future__ import annotations

from pathlib import Path
import json

import torch

from .corpus import Vocabulary


def save_vocabulary(vocab: Vocabulary, path: Path) -> None:
    payload = {
        "id_to_token": vocab.id_to_token,
        "counts": vocab.counts,
        "unk_token": vocab.unk_token,
        "mask_token": vocab.mask_token,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_vocabulary(path: Path) -> Vocabulary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    id_to_token = list(payload["id_to_token"])
    return Vocabulary(
        id_to_token=id_to_token,
        token_to_id={token: index for index, token in enumerate(id_to_token)},
        counts={str(token): int(count) for token, count in payload["counts"].items()},
        unk_token=payload["unk_token"],
        mask_token=payload["mask_token"],
    )


def save_codebook(codebook, path: Path) -> None:
    torch.save(codebook.detach().cpu(), Path(path))


def load_codebook(path: Path, device: str | None = None):
    tensor = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
    return tensor if device is None else tensor.to(device)
