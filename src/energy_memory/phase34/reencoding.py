"""Replay-and-re-encode for stale stored patterns.

When the codebook evolves (Phase 3 dynamics), stored Hopfield patterns
were encoded with an older codebook. The bundled FHRR vector becomes
stale: bind(position, OLD codebook[token]) doesn't match
bind(position, NEW codebook[token]).

This module provides a re-encoding helper: given a list of source
windows and a current codebook, regenerate the stored patterns so they
align with the new substrate.

Anti-homunculus check: re-encoding is triggered by a passive condition
(K cycles elapsed, or total codebook drift exceeded a threshold), not
by a controller deciding when to refresh. The condition is a property
of the architecture's timeline, not a rule.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import encode_window
from energy_memory.substrate.torch_fhrr import TorchFHRR


def reencode_patterns(
    memory: TorchHopfieldMemory,
    source_windows: Sequence[Optional[tuple[int, ...]]],
    substrate: TorchFHRR,
    positions: Sequence,
    codebook: "torch.Tensor",
) -> int:
    """Re-encode each stored pattern using the current codebook.

    source_windows[i] is the original token-window that pattern i was
    built from. Entries that are None (e.g., patterns discovered via
    Phase 4 replay that don't have a source window) are skipped.

    Returns the number of patterns actually re-encoded.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("reencode_patterns requires torch") from _IMPORT_ERROR

    n = min(len(memory._patterns), len(source_windows))
    re_encoded = 0
    for idx in range(n):
        window = source_windows[idx]
        if window is None:
            continue
        fresh = encode_window(substrate, positions, codebook, window)
        memory._patterns[idx] = fresh.to(substrate.device)
        re_encoded += 1
    return re_encoded


def codebook_drift(
    codebook_a: "torch.Tensor",
    codebook_b: "torch.Tensor",
) -> float:
    """Mean cosine distance between two codebooks (row-wise).

    A useful metric for deciding when to re-encode: large drift means
    stored patterns are getting stale. Return value in [0, 2].
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("codebook_drift requires torch")
    if codebook_a.shape != codebook_b.shape:
        raise ValueError("codebooks must have matching shape")
    a = codebook_a / codebook_a.abs().clamp(min=1e-8)
    b = codebook_b / codebook_b.abs().clamp(min=1e-8)
    cos = (a.conj() * b).sum(dim=-1).real / codebook_a.shape[-1]
    return float((1.0 - cos).mean().detach().cpu())
