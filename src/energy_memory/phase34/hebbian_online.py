"""Online Hebbian reinforcement updater for retrieval success.

Per phase-3-deep-dive.md, line 100:
> If q is high (above success threshold): apply Hebbian update with
> magnitude proportional to q. For each atom that participated in the
> cue, drift slightly toward the *clean context bag* (bundle of other
> atoms in the cue, no position bindings).

This is the architecturally-intended runtime update rule. It does not
require ground-truth target ids — only the cue's tokens and the
retrieval success score.

Trivial-skip thresholds (deep-dive line 102, 136):
- q > 0.95: skip (retrieval was already at ceiling, nothing to reinforce)

Failure regime (q < success_threshold): skip in this implementation.
The deep-dive splits "failure" between batch error-driven training
(off the corpus) and replay-driven consolidation (Phase 4). Online
error-driven contrastive updates from random init are out of scope —
see reports/phase34_stable_v2/findings.md for the diagnosis.

Anti-homunculus check: q is a passive geometric property (cosine
similarity of the settled state to the top stored pattern). The success
threshold is a passive filter on that property — not a rule that
decides what to do, just whether the dynamic is active at this moment.
The pull target (context bag) is a function of the cue itself, not of
any state-reading supervisor.
"""

from __future__ import annotations

from typing import Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.substrate.torch_fhrr import TorchFHRR


class HebbianOnlineCodebookUpdater:
    """Hebbian reinforcement on retrieval success."""

    def __init__(
        self,
        substrate: TorchFHRR,
        codebook: "torch.Tensor",
        lr_hebbian: float = 0.01,
        success_threshold: float = 0.5,
        trivial_skip_threshold: float = 0.95,
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "HebbianOnlineCodebookUpdater requires torch"
            ) from _IMPORT_ERROR
        self.substrate = substrate
        self.codebook = codebook
        self.lr_hebbian = lr_hebbian
        self.success_threshold = success_threshold
        self.trivial_skip_threshold = trivial_skip_threshold
        self._total = 0
        self._success = 0
        self._trivial = 0
        self._below_threshold = 0
        self._atoms_updated = 0

    def observe(
        self,
        q: float,
        cue_token_ids: Sequence[int],
    ) -> bool:
        """Apply Hebbian update if q is in the success regime.

        Returns True if an update fired, False otherwise.
        """
        self._total += 1
        if q >= self.trivial_skip_threshold:
            self._trivial += 1
            return False
        if q < self.success_threshold:
            self._below_threshold += 1
            return False
        self._success += 1
        n = len(cue_token_ids)
        if n < 2:
            return False

        magnitude = self.lr_hebbian * q
        # Build the bundled stack once: bundle of all cue atoms. Each
        # atom's context bag = bundle minus that atom, then normalized.
        # Cheaper than recomputing per atom.
        all_atoms = torch.stack(
            [self.codebook[tid] for tid in cue_token_ids], dim=0,
        )
        total_sum = all_atoms.sum(dim=0)
        for i, tid in enumerate(cue_token_ids):
            # Context bag = bundle(others) = normalize(total_sum - this_atom)
            context_bag = self.substrate.normalize(total_sum - all_atoms[i])
            self.codebook[tid] = self.substrate.normalize(
                (1.0 - magnitude) * self.codebook[tid]
                + magnitude * context_bag
            )
            self._atoms_updated += 1
        return True

    def stats(self) -> dict:
        return {
            "total_observations": self._total,
            "successes": self._success,
            "trivial_skips": self._trivial,
            "below_threshold": self._below_threshold,
            "atoms_updated": self._atoms_updated,
            "success_rate": self._success / max(1, self._total),
            "trivial_rate": self._trivial / max(1, self._total),
        }
