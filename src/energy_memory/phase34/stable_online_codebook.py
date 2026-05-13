"""Stable online codebook updater with the missing repulsion term.

The shipped `OnlineCodebookUpdater` implements only the error-driven half
of the Phase 3 deep-dive spec — pull/push without the anti-collapse
repulsion term. The diagnostic experiment (`experiments/22_*`) confirmed
that pull alone collapses retrieval to chance by ~3000 cues on a single
W=2 scale when started from random codebook init: atoms get pulled toward
the noise-dominated slot-query centroid of the still-untrained retrieval,
which degrades discrimination, which makes the next batch of slot-queries
even noisier — a closed feedback loop.

The Phase 3 deep-dive (`notes/emergent-codebook/phase-3-deep-dive.md`)
explicitly names this failure mode ("Codebook collapse — pairwise
similarity histogram narrows toward 1.0") and prescribes the fix:
repulsion at cosine threshold 0.7.

This updater extends `OnlineCodebookUpdater` with:

1. Per-consolidation repulsion. After pull/push, for each updated atom we
   look at its nearest non-self neighbor in the codebook. If the cosine
   exceeds `repulsion_threshold`, the pair is nudged apart along the
   line connecting them.

Anti-homunculus check: nearest-neighbor distance is a passive geometric
property of the codebook. The repulsion is a local force on a local
state (this atom against its closest neighbor), not an arbitration over
mechanisms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.phase34.online_codebook import OnlineCodebookUpdater
from energy_memory.substrate.torch_fhrr import TorchFHRR


class StableOnlineCodebookUpdater(OnlineCodebookUpdater):
    """OnlineCodebookUpdater + Phase-3-spec repulsion."""

    def __init__(
        self,
        substrate: TorchFHRR,
        codebook: "torch.Tensor",
        lr_pull: float = 0.1,
        lr_push: float = 0.05,
        consolidation_k: int = 100,
        quality_threshold: float = 0.15,
        repulsion_threshold: float = 0.7,
        repulsion_strength: float = 0.1,
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "StableOnlineCodebookUpdater requires torch"
            ) from _IMPORT_ERROR
        super().__init__(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
        )
        self.repulsion_threshold = repulsion_threshold
        self.repulsion_strength = repulsion_strength
        self._repulsion_events = 0

    def _consolidate(self) -> dict:
        # Collect the IDs that are about to be updated, so we can apply
        # repulsion to exactly the atoms that just moved.
        updated_ids: Set[int] = set()
        for entry in self._buffer:
            updated_ids.add(entry.target_id)
            if entry.predicted_id != entry.target_id:
                updated_ids.add(entry.predicted_id)

        diagnostics = super()._consolidate()

        if updated_ids and self.repulsion_strength > 0.0:
            n_repulsed = self._apply_repulsion(sorted(updated_ids))
            self._repulsion_events += n_repulsed
            diagnostics["repulsed_pairs"] = n_repulsed
            diagnostics["repulsion_events_total"] = self._repulsion_events
        else:
            diagnostics["repulsed_pairs"] = 0
            diagnostics["repulsion_events_total"] = self._repulsion_events

        return diagnostics

    def _apply_repulsion(self, updated_ids: List[int]) -> int:
        """Push each updated atom away from its nearest non-self neighbor
        if their cosine exceeds `repulsion_threshold`.

        Returns the number of (updated_atom, neighbor) pairs that fired.
        """
        n_total = self.codebook.shape[0]
        idx = torch.tensor(updated_ids, device=self.codebook.device, dtype=torch.long)
        updated = self.codebook[idx]  # [U, D] complex

        # Compute similarity from each updated atom to the full codebook
        # using the substrate's FHRR cosine: real-mean of conj * pattern.
        sims = (
            self.codebook.conj()[None, :, :] * updated[:, None, :]
        ).real.mean(dim=-1)  # [U, N]

        # Mask out self by setting the diagonal-of-the-subset to -inf so
        # the argmax never selects self. Each row corresponds to
        # updated_ids[i]; the entry at column updated_ids[i] is self.
        for i, atom_id in enumerate(updated_ids):
            sims[i, atom_id] = float("-inf")

        nn_sims, nn_idx = sims.max(dim=-1)  # [U]
        firing = nn_sims > self.repulsion_threshold
        if not bool(firing.any()):
            return 0

        n_fired = int(firing.sum().item())
        fire_updated = updated[firing]                   # [F, D]
        fire_neighbor_ids = nn_idx[firing]               # [F]
        fire_neighbors = self.codebook[fire_neighbor_ids]  # [F, D]

        # Repulsion: move each atom slightly away from its neighbor's
        # direction. Element-wise, on the unit-modulus per-coordinate
        # FHRR representation.
        new_atoms = self.substrate.normalize(
            (1.0 + self.repulsion_strength) * fire_updated
            - self.repulsion_strength * fire_neighbors
        )
        fire_atom_ids = idx[firing]
        for slot, atom_id in enumerate(fire_atom_ids.tolist()):
            self.codebook[atom_id] = new_atoms[slot]
        return n_fired

    def stats(self) -> dict:
        s = super().stats()
        s["repulsion_events_total"] = self._repulsion_events
        s["repulsion_threshold"] = self.repulsion_threshold
        s["repulsion_strength"] = self.repulsion_strength
        return s


class StableOnlineCodebookUpdaterV2(StableOnlineCodebookUpdater):
    """V2: V1 (repulsion) + common-mode mean subtraction on pull targets.

    Diagnostic finding from exp23: V1 repulsion at threshold 0.7 never
    fires because the collapse signature is not pairwise convergence
    (mean stays ~0.009, p95 only ~0.023). The collapse is driven by
    every target's pull centroid converging to approximately the same
    position-mask-induced direction:

      slot_query = unbind(retrieved_state, position_mask)
                 = retrieved_state * conj(position_mask)

    When retrieved_state is noise-dominated (random init), avg_dir for
    every target_id is approximately `rotated conj(position_mask)`. Pull
    yanks every codebook atom toward the same common-mode direction.

    The fix: before updating, subtract the buffer-wide mean slot_query
    direction from each target's avg_dir. This removes the common-mode
    bias while keeping target-specific residual structure.

    Repulsion threshold is also relaxed from 0.7 to a value in the
    observed regime (~0.05 by default) so it actually fires.
    """

    def __init__(
        self,
        substrate: TorchFHRR,
        codebook: "torch.Tensor",
        lr_pull: float = 0.02,
        lr_push: float = 0.05,
        consolidation_k: int = 100,
        quality_threshold: float = 0.15,
        repulsion_threshold: float = 0.05,
        repulsion_strength: float = 0.05,
        mean_subtract: bool = True,
    ):
        super().__init__(
            substrate=substrate, codebook=codebook,
            lr_pull=lr_pull, lr_push=lr_push,
            consolidation_k=consolidation_k,
            quality_threshold=quality_threshold,
            repulsion_threshold=repulsion_threshold,
            repulsion_strength=repulsion_strength,
        )
        self.mean_subtract = mean_subtract

    def _consolidate(self) -> dict:
        # Collect updated IDs before we mutate the buffer.
        updated_ids = set()
        for entry in self._buffer:
            updated_ids.add(entry.target_id)
            if entry.predicted_id != entry.target_id:
                updated_ids.add(entry.predicted_id)

        if not self._buffer:
            return {
                "consolidation": self._consolidation_count,
                "buffer_size": 0, "pulled": 0, "pushed": 0,
                "mean_quality": 0.0,
                "total_observations": self._total_observations,
                "total_failures": self._total_failures,
                "failure_rate": 0.0,
                "repulsed_pairs": 0,
                "repulsion_events_total": self._repulsion_events,
                "mean_subtracted": self.mean_subtract,
            }

        # Compute the buffer-wide common-mode direction. This is the
        # vector that every "pull toward avg slot_query" would carry,
        # so it's the common-mode bias we want to factor out.
        all_queries = torch.stack([e.slot_query for e in self._buffer])
        common = self.substrate.normalize(all_queries.sum(dim=0))

        from collections import defaultdict
        pull_targets: dict = defaultdict(list)
        push_targets: dict = defaultdict(list)
        for entry in self._buffer:
            pull_targets[entry.target_id].append(entry.slot_query)
            if entry.predicted_id != entry.target_id:
                push_targets[entry.predicted_id].append(entry.slot_query)

        pulled = 0
        for tid, queries in pull_targets.items():
            raw_dir = self.substrate.normalize(torch.stack(queries).sum(dim=0))
            if self.mean_subtract:
                # Element-wise subtract the common-mode then renormalize.
                # In FHRR (unit-modulus per coord) this rotates each
                # coord by the angle of (raw - common), which gives the
                # target-specific residual direction.
                avg_dir = self.substrate.normalize(raw_dir - common)
            else:
                avg_dir = raw_dir
            self.codebook[tid] = self.substrate.normalize(
                (1.0 - self.lr_pull) * self.codebook[tid]
                + self.lr_pull * avg_dir
            )
            pulled += 1

        pushed = 0
        for wid, queries in push_targets.items():
            raw_dir = self.substrate.normalize(torch.stack(queries).sum(dim=0))
            if self.mean_subtract:
                avg_dir = self.substrate.normalize(raw_dir - common)
            else:
                avg_dir = raw_dir
            self.codebook[wid] = self.substrate.normalize(
                (1.0 + self.lr_push) * self.codebook[wid]
                - self.lr_push * avg_dir
            )
            pushed += 1

        self._consolidation_count += 1
        mean_q = sum(e.quality for e in self._buffer) / len(self._buffer)
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
            "mean_subtracted": self.mean_subtract,
        }
        self._buffer.clear()

        if updated_ids and self.repulsion_strength > 0.0:
            n = self._apply_repulsion(sorted(updated_ids))
            self._repulsion_events += n
            diagnostics["repulsed_pairs"] = n
        else:
            diagnostics["repulsed_pairs"] = 0
        diagnostics["repulsion_events_total"] = self._repulsion_events
        return diagnostics
