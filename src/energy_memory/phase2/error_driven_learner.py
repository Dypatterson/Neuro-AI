"""Error-driven contrastive codebook learning for FHRR substrates.

Runs masked-token retrieval on training windows.  When the decoded
slot vector is far from the correct codebook vector (q < threshold),
the failure is buffered.  Every K failures a consolidation event fires:
correct atoms are pulled toward the decoded-slot direction, wrong
predictions are pushed away.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterator, List, Sequence

import torch

if TYPE_CHECKING:
    from energy_memory.phase2.corpus import Vocabulary
    from energy_memory.substrate.torch_fhrr import TorchFHRR

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window,
    mask_positions as compute_mask_positions,
    masked_window,
)


class ErrorDrivenLearner:

    def __init__(
        self,
        substrate: "TorchFHRR",
        codebook: torch.Tensor,
        vocab: "Vocabulary",
        lr_pull: float = 0.1,
        lr_push: float = 0.05,
        consolidation_k: int = 100,
        quality_threshold: float = 0.5,
    ):
        self.substrate = substrate
        self.codebook = codebook.clone()
        self.vocab = vocab
        self.lr_pull = lr_pull
        self.lr_push = lr_push
        self.consolidation_k = consolidation_k
        self.quality_threshold = quality_threshold
        self._buffer: List[dict] = []
        self._consolidation_count = 0
        self._total_retrievals = 0
        self._total_failures = 0

    def train(
        self,
        landscape_windows: Sequence[tuple[int, ...]],
        probe_windows: Sequence[tuple[int, ...]],
        window_size: int,
        mask_count: int = 1,
        mask_position: str = "center",
        beta: float = 10.0,
    ) -> Iterator[dict]:
        """Train codebook via error-driven contrastive updates.

        Stores *landscape_windows* in a Hopfield memory, then probes
        with masked versions of *probe_windows*.  Yields diagnostics
        after each consolidation event.
        """
        positions = build_position_vectors(self.substrate, window_size)
        masked_positions = compute_mask_positions(
            window_size, mask_count, mask_position,
        )
        decode_ids = [
            i for i, t in enumerate(self.vocab.id_to_token)
            if t not in {self.vocab.unk_token, self.vocab.mask_token}
        ]
        candidate_ids = torch.tensor(decode_ids, device=self.codebook.device)

        memory = self._build_memory(landscape_windows, positions)

        for window in probe_windows:
            targets = [window[pos] for pos in masked_positions]
            if any(t == self.vocab.unk_id for t in targets):
                continue

            cue_win = masked_window(window, masked_positions, self.vocab.mask_id)
            cue = encode_window(
                self.substrate, positions, self.codebook, cue_win,
            )
            result = memory.retrieve(cue, beta=beta, max_iter=12)
            self._total_retrievals += 1

            for pos_idx, target_id in zip(masked_positions, targets):
                slot_query = self.substrate.unbind(
                    result.state, positions[pos_idx],
                )
                q = self.substrate.similarity(slot_query, self.codebook[target_id])

                if q >= self.quality_threshold:
                    continue

                candidate_matrix = self.codebook[candidate_ids]
                sims = self.substrate.similarity_matrix(
                    slot_query, candidate_matrix,
                )
                predicted_local = int(sims.argmax().cpu().item())
                predicted_id = decode_ids[predicted_local]

                self._total_failures += 1
                self._buffer.append({
                    "target_id": target_id,
                    "predicted_id": predicted_id,
                    "slot_query": slot_query.detach().clone(),
                    "quality": float(q),
                })

                if len(self._buffer) >= self.consolidation_k:
                    yield self._consolidate()

        if self._buffer:
            yield self._consolidate()

    def _build_memory(
        self,
        windows: Sequence[tuple[int, ...]],
        positions,
    ) -> TorchHopfieldMemory[str]:
        memory: TorchHopfieldMemory[str] = TorchHopfieldMemory(self.substrate)
        for idx, window in enumerate(windows):
            encoded = encode_window(
                self.substrate, positions, self.codebook, window,
            )
            memory.store(encoded, label=f"w_{idx}")
        return memory

    def _consolidate(self) -> dict:
        """Contrastive update: pull correct atoms, push wrong atoms."""
        pull_targets: dict[int, List[torch.Tensor]] = defaultdict(list)
        push_targets: dict[int, List[torch.Tensor]] = defaultdict(list)

        for entry in self._buffer:
            pull_targets[entry["target_id"]].append(entry["slot_query"])
            if entry["predicted_id"] != entry["target_id"]:
                push_targets[entry["predicted_id"]].append(entry["slot_query"])

        pulled = 0
        for tid, queries in pull_targets.items():
            avg_dir = self.substrate.normalize(torch.stack(queries).sum(dim=0))
            self.codebook[tid] = self.substrate.normalize(
                (1.0 - self.lr_pull) * self.codebook[tid]
                + self.lr_pull * avg_dir
            )
            pulled += 1

        pushed = 0
        for wid, queries in push_targets.items():
            avg_dir = self.substrate.normalize(torch.stack(queries).sum(dim=0))
            self.codebook[wid] = self.substrate.normalize(
                (1.0 + self.lr_push) * self.codebook[wid]
                - self.lr_push * avg_dir
            )
            pushed += 1

        self._consolidation_count += 1
        mean_q = (
            sum(e["quality"] for e in self._buffer) / len(self._buffer)
            if self._buffer
            else 0.0
        )
        diagnostics = {
            "consolidation": self._consolidation_count,
            "buffer_size": len(self._buffer),
            "pulled": pulled,
            "pushed": pushed,
            "mean_quality": mean_q,
            "total_retrievals": self._total_retrievals,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / max(1, self._total_retrievals)
            ),
        }
        self._buffer.clear()
        return diagnostics
