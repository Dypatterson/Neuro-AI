"""Online codebook updater for Phase 3+4 integration.

Exposes a per-observation API for codebook learning so it can be called
inline during a streaming cue loop, rather than as a batch trainer that
builds its own memory.

The consolidation logic is identical to ErrorDrivenLearner._consolidate:
contrastive pull (codebook[correct] toward avg slot_query) and push
(codebook[wrong] away from avg slot_query). The difference is API
shape: observe() is called once per retrieval, consolidate_if_ready()
runs the update when the buffer fills.

Anti-homunculus check: consolidation triggers on buffer fill, not on a
controller decision. The buffer-fill condition is a passive geometric
property (failure count crossed K), not a rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass
class _BufferedFailure:
    target_id: int
    predicted_id: int
    slot_query: "torch.Tensor"
    quality: float


class OnlineCodebookUpdater:
    """Per-observation codebook updater with periodic consolidation.

    Usage:
        updater = OnlineCodebookUpdater(substrate, codebook, ...)
        # Per retrieval:
        updater.observe(target_id=t, slot_query=q, predicted_id=p)
        # Returns True if a consolidation event fired
        diagnostics = updater.consolidate_if_ready()
    """

    def __init__(
        self,
        substrate: TorchFHRR,
        codebook: "torch.Tensor",
        lr_pull: float = 0.1,
        lr_push: float = 0.05,
        consolidation_k: int = 100,
        quality_threshold: float = 0.15,
        consolidation_state: Optional["object"] = None,
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("OnlineCodebookUpdater requires torch") from _IMPORT_ERROR
        self.substrate = substrate
        self.codebook = codebook
        self.lr_pull = lr_pull
        self.lr_push = lr_push
        self.consolidation_k = consolidation_k
        self.quality_threshold = quality_threshold
        self._buffer: List[_BufferedFailure] = []
        self._consolidation_count = 0
        self._total_observations = 0
        self._total_failures = 0
        # C.2.1 actuator handle. Optional substrate-side state container;
        # when supplied and lambda_ac > 0, _consolidate() adds the anti-
        # collapse force to each updated atom. None / lambda_ac == 0
        # leaves consolidation byte-identical to the pre-C.2.1 baseline.
        self.consolidation_state = consolidation_state

    def observe(
        self,
        target_id: int,
        slot_query: "torch.Tensor",
        predicted_id: int,
    ) -> bool:
        """Observe one (target, slot_query, predicted) tuple.

        If similarity(slot_query, codebook[target]) is below quality_threshold,
        the observation is buffered as a failure. Returns True if a
        consolidation is now ready to fire (buffer reached K).
        """
        self._total_observations += 1
        quality = float(
            self.substrate.similarity(slot_query, self.codebook[target_id])
        )
        if quality >= self.quality_threshold:
            return False
        self._total_failures += 1
        self._buffer.append(_BufferedFailure(
            target_id=target_id,
            predicted_id=predicted_id,
            slot_query=slot_query.detach().clone(),
            quality=quality,
        ))
        return len(self._buffer) >= self.consolidation_k

    def consolidate_if_ready(self) -> Optional[dict]:
        if len(self._buffer) < self.consolidation_k:
            return None
        return self._consolidate()

    def force_consolidate(self) -> Optional[dict]:
        if not self._buffer:
            return None
        return self._consolidate()

    def _consolidate(self) -> dict:
        pull_targets: dict[int, List["torch.Tensor"]] = defaultdict(list)
        push_targets: dict[int, List["torch.Tensor"]] = defaultdict(list)

        for entry in self._buffer:
            pull_targets[entry.target_id].append(entry.slot_query)
            if entry.predicted_id != entry.target_id:
                push_targets[entry.predicted_id].append(entry.slot_query)

        pulled = 0
        for tid, queries in pull_targets.items():
            avg_dir = self.substrate.normalize(
                torch.stack(queries).sum(dim=0),
            )
            self.codebook[tid] = self.substrate.normalize(
                (1.0 - self.lr_pull) * self.codebook[tid]
                + self.lr_pull * avg_dir
            )
            pulled += 1

        pushed = 0
        for wid, queries in push_targets.items():
            avg_dir = self.substrate.normalize(
                torch.stack(queries).sum(dim=0),
            )
            self.codebook[wid] = self.substrate.normalize(
                (1.0 + self.lr_push) * self.codebook[wid]
                - self.lr_push * avg_dir
            )
            pushed += 1

        self._apply_anti_collapse(pull_targets.keys() | push_targets.keys())

        self._consolidation_count += 1
        mean_q = (
            sum(e.quality for e in self._buffer) / len(self._buffer)
            if self._buffer else 0.0
        )
        diagnostics = {
            "consolidation": self._consolidation_count,
            "buffer_size": len(self._buffer),
            "pulled": pulled,
            "pushed": pushed,
            "mean_quality": mean_q,
            "total_observations": self._total_observations,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / max(1, self._total_observations)
            ),
        }
        self._buffer.clear()
        return diagnostics

    def _apply_anti_collapse(self, atom_ids) -> None:
        # C.2.1: per-atom anti-collapse force added to the existing update.
        # Early-exit preserves the κ=0 baseline byte-identically.
        cs = self.consolidation_state
        if cs is None or getattr(cs.config, "lambda_ac", 0.0) == 0.0:
            return
        for atom_id in atom_ids:
            if atom_id < 0 or atom_id >= self.codebook.shape[0]:
                continue
            current = self.codebook[atom_id]
            force = cs.anti_collapse_force(int(atom_id), current)
            if force is None:
                continue
            self.codebook[atom_id] = self.substrate.normalize(current + force)

    def stats(self) -> dict:
        return {
            "consolidations": self._consolidation_count,
            "buffer_size": len(self._buffer),
            "total_observations": self._total_observations,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / max(1, self._total_observations)
            ),
        }
