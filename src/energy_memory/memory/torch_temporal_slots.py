"""Permutation-indexed temporal slot memory.

Discrete variant of brainstorm Idea 6 (2026-05-13). Replaces the unordered
temporal context bag

    context(i) = bundle({ v_j : j in window around i, j != i })

with a directed slot encoding

    context(i) = bundle({ permute(v_j, j - i) : j in window, j != i })

where ``permute`` is the cyclic-shift VSA permutation operator. The bundle
now carries the signed temporal offset of each neighbor; querying at a
specific offset k applies ``permute(context, -k)`` and reads off the
nearest atom from the supplied vocabulary.

This is the discrete / exact alternative to Fractional Power Encoding —
no continuous decoder, no LCD cleanup, no extra moving parts. Algebraically
equivalent to RoPE-style positional encoding in transformers.

The 2026-05-09 paper-synthesis note documents two named failure modes of
unordered bags: temporal inaccuracy (cannot distinguish simultaneous from
sequential) and temporal fragmentation (loses relative ordering). The
directed slot encoding eliminates both by construction.

Anti-homunculus check: every binding operation is a local, deterministic
function of (atom, signed offset). No controller picks the offset; the
offset is a property of the temporal axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class OffsetQueryResult:
    offset: int
    top_label: str
    top_score: float
    ranked: List[Tuple[str, float]]


class PermutationSlotTemporalMemory:
    """Directed temporal context memory using permutation-indexed slots.

    Stores per-token contexts of the form

        context(i) = normalize( sum_{j != i in window} permute(v_j, j - i) )

    Query API:
      query_offset(context, offset, vocab_vectors, vocab_labels):
        returns the atom at the requested offset and a ranked list.
    """

    def __init__(self, substrate: TorchFHRR, window: int = 2):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "PermutationSlotTemporalMemory requires torch"
            ) from _IMPORT_ERROR
        if window <= 0:
            raise ValueError("window must be positive")
        self.substrate = substrate
        self.window = window
        self.labels: List[str] = []
        self.vectors = None  # [N, D]
        self.temporal_contexts = None  # [N, D]

    def store_sequence(self, labels: Sequence[str], vectors: Sequence) -> None:
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        self.labels = list(labels)
        self.vectors = torch.stack(list(vectors), dim=0)

        contexts = []
        for i in range(len(labels)):
            shifted: List = []
            lo = max(0, i - self.window)
            hi = min(len(labels), i + self.window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                offset = j - i  # signed: negative = before, positive = after
                shifted.append(self.substrate.permute(vectors[j], offset))
            if shifted:
                contexts.append(self.substrate.bundle(shifted))
            else:
                # Singleton window: store self as a degenerate context.
                contexts.append(vectors[i])
        self.temporal_contexts = torch.stack(contexts, dim=0)

    def context_for(self, index: int):
        if self.temporal_contexts is None:
            raise ValueError("memory is empty")
        return self.temporal_contexts[index]

    def query_offset(
        self,
        context,
        offset: int,
        vocab_vectors=None,
        vocab_labels: Sequence[str] = None,
        top_k: int = 5,
    ) -> OffsetQueryResult:
        """Probe a context for the atom at signed temporal ``offset``.

        ``permute(context, -offset)`` undoes the slot binding; the result
        is bundle-noise plus the original neighbor at that offset. A
        nearest-neighbor lookup against the vocabulary recovers the label.

        When vocab_vectors/labels are not supplied, the stored sequence
        is used as the vocabulary (useful for sequence-internal queries).
        """
        if self.vectors is None:
            raise ValueError("memory is empty")
        if offset == 0:
            raise ValueError("offset must be non-zero")

        probe = self.substrate.permute(context, -offset)
        if vocab_vectors is None:
            vocab_vectors = self.vectors
            vocab_labels = list(self.labels)
        else:
            if vocab_labels is None:
                raise ValueError("vocab_labels must be provided with vocab_vectors")

        sims = self.substrate.similarity_matrix(probe, vocab_vectors)
        count = min(top_k, len(vocab_labels))
        values, indices = torch.topk(sims, count)
        ranked = [
            (vocab_labels[int(idx)], float(val))
            for idx, val in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist())
        ]
        top_label, top_score = ranked[0]
        return OffsetQueryResult(
            offset=offset,
            top_label=top_label,
            top_score=top_score,
            ranked=ranked,
        )
