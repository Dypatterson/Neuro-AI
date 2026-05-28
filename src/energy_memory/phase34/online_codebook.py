"""Online codebook updater for Phase 3+4 integration.

Exposes a per-observation API for codebook learning so it can be called
inline during a streaming cue loop, rather than as a batch trainer that
builds its own memory.

Two base update mechanisms are gated by config flags, set independently
at construction:

- **Pull/push (default, use_pull_push=True):** the original error-driven
  contrastive update — pull codebook[correct] toward avg slot_query;
  push codebook[wrong] away from avg slot_query. The
  predicted_id != target_id gate at L132-133 enqueues into push_targets
  by the energy-support of ½‖codebook[predicted] − slot_query‖² over
  misclassified events.
- **Context-residual (Γ1.c, use_context_residual=True):** the Path γ
  leader candidate per the precommit at
  notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md.
  Asymmetric gradient descent on a per-event repulsion energy
  E_cr = −Σ_j 1[predicted_j ≠ target_j] · ½‖codebook[target_j] −
  codebook[predicted_j]‖² with respect to codebook[target_j] only
  (stop-gradient on codebook[predicted_j]). The indicator is realized
  as a property of ε's support (ε = 0 when predicted == target) — no
  if/branch in the runtime path (reviewer watch item W1).

Anti-homunculus check (binding per CLAUDE.md and the Γ1 precommit
anti-homunculus reviewer PASS, 2026-05-27): consolidation triggers on
buffer fill, not on a controller decision. The buffer-fill condition is
a passive geometric property (failure count crossed K), not a rule.
Both base updates compose additively with C.2.1–C.2.5 dynamics; the
composition order is preserved across mechanism swaps.
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
        *,
        use_pull_push: bool = True,
        use_context_residual: bool = False,
        lr_cr: float = 0.1,
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
        # Γ1 base-update flags (Path γ precommit, 2026-05-27). Defaults
        # (use_pull_push=True, use_context_residual=False) preserve Path C
        # byte-identity exactly. Both flags can be True simultaneously;
        # composition is additive (anti-homunculus reviewer PASS
        # 2026-05-27). The Γ1 headline condition sets
        # use_pull_push=False, use_context_residual=True.
        self.use_pull_push = use_pull_push
        self.use_context_residual = use_context_residual
        self.lr_cr = lr_cr

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
        # C.2.5: snapshot the codebook BEFORE any forces fire so end-of-event
        # update_drift_tension can compute ||current - previous|| per atom.
        # Early-exits when drift_ema_rate == 0 (κ=0 baseline).
        cs = self.consolidation_state
        if cs is not None:
            cs.snapshot_previous_codebook(self.codebook)

        pull_targets: dict[int, List["torch.Tensor"]] = defaultdict(list)
        push_targets: dict[int, List["torch.Tensor"]] = defaultdict(list)

        for entry in self._buffer:
            pull_targets[entry.target_id].append(entry.slot_query)
            if entry.predicted_id != entry.target_id:
                push_targets[entry.predicted_id].append(entry.slot_query)

        # Affected set = atoms moved by the active BASE update(s). Pull/push
        # moves both target_ids (pull) and predicted_ids on confusion (push).
        # Context-residual is asymmetric — moves only target_ids. The union
        # is used when both base updates are active simultaneously.
        affected: set = set()
        if self.use_pull_push:
            affected |= set(pull_targets.keys()) | set(push_targets.keys())
        if self.use_context_residual:
            affected |= set(pull_targets.keys())  # i.e., unique target_ids
        # C.2.2: update splitting tension BEFORE any consolidation force this
        # event so T_k reflects basin state pre-anti-collapse; modulation
        # then acts on the next event with that tension value. Early-exits
        # when mu_T == 0 (κ=0 baseline byte-identical).
        pre_states = self._snapshot_pre_states(affected)
        if cs is not None:
            cs.update_splitting_tension()

        pulled = 0
        pushed = 0
        if self.use_pull_push:
            for tid, queries in pull_targets.items():
                avg_dir = self.substrate.normalize(
                    torch.stack(queries).sum(dim=0),
                )
                self.codebook[tid] = self.substrate.normalize(
                    (1.0 - self.lr_pull) * self.codebook[tid]
                    + self.lr_pull * avg_dir
                )
                pulled += 1

            for wid, queries in push_targets.items():
                avg_dir = self.substrate.normalize(
                    torch.stack(queries).sum(dim=0),
                )
                self.codebook[wid] = self.substrate.normalize(
                    (1.0 + self.lr_push) * self.codebook[wid]
                    - self.lr_push * avg_dir
                )
                pushed += 1

        cr_updated = 0
        if self.use_context_residual:
            cr_updated = self._apply_context_residual()

        self._apply_anti_collapse(affected)
        # C.2.3: per-atom cap-coverage error force added BEFORE the
        # splitting-tension modulation so all additive forces are summed
        # before being multiplicatively attenuated. Early-exit preserves
        # κ=0 byte-identity when lambda_cc == 0.
        self._apply_cap_coverage(affected)
        # C.2.2: multiplicatively attenuate the combined (base + anti-
        # collapse + cap-coverage) per-atom update by 1/(1 + T_k / τ_T).
        # H12 binding: modulation is multiplicative, applied to the NET
        # update — not a new additive force, not a replacement of the update.
        self._apply_splitting_tension(pre_states)

        # C.2.5: EMA-update drift tension AFTER all forces (incl. C.2.1, C.2.2,
        # C.2.3 modulation) have settled the codebook for this event.
        if cs is not None:
            cs.update_drift_tension(self.codebook)

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
            "context_residual_updated": cr_updated,
            "mean_quality": mean_q,
            "total_observations": self._total_observations,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / max(1, self._total_observations)
            ),
        }
        self._buffer.clear()
        return diagnostics

    def _apply_context_residual(self) -> int:
        """Γ1.c — asymmetric gradient descent on per-event repulsion energy.

        Per the precommit at
        notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md:
        E_cr = −Σ_j 1[predicted_j ≠ target_j] · ½‖codebook[target_j] −
        codebook[predicted_j]‖². Gradient w.r.t. codebook[target_j] gives the
        update direction ε_j = codebook[target_j] − codebook[predicted_j];
        descent step is codebook[target_j] ← codebook[target_j] + lr_cr · ε_j.

        Asymmetric: codebook[predicted_j] is stop-gradient; not updated by
        this term.

        The indicator 1[predicted ≠ target] appears as a property of ε's
        support (ε = 0 when predicted == target) — no if/branch is needed
        in the runtime code path (reviewer watch item W1).

        Snapshot semantics: ε computations use the codebook state BEFORE
        any Γ1.c update fires this consolidation event. Avoids coupled
        fixed-point ambiguity when target_id of one entry overlaps with
        predicted_id of another in the same buffer.
        """
        if not self._buffer:
            return 0
        # Snapshot: all ε computed from pre-Γ1.c codebook state.
        codebook_snapshot = self.codebook.detach().clone()
        sums: dict[int, "torch.Tensor"] = {}
        counts: dict[int, int] = defaultdict(int)
        for entry in self._buffer:
            # ε = snapshot[target] − snapshot[predicted].
            # ε is the zero vector when target == predicted (energy-support
            # property of E_cr); W1 indicator-as-mask form (no if/branch).
            eps = (
                codebook_snapshot[entry.target_id]
                - codebook_snapshot[entry.predicted_id]
            )
            if entry.target_id in sums:
                sums[entry.target_id] = sums[entry.target_id] + eps
            else:
                sums[entry.target_id] = eps.clone()
            counts[entry.target_id] += 1
        updated = 0
        for tid, sum_eps in sums.items():
            mean_eps = sum_eps / counts[tid]
            self.codebook[tid] = self.substrate.normalize(
                self.codebook[tid] + self.lr_cr * mean_eps
            )
            updated += 1
        return updated

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

    def _apply_cap_coverage(self, atom_ids) -> None:
        # C.2.3: per-atom cap-coverage error force added additively.
        # Early-exit when lambda_cc == 0 preserves the κ=0 baseline.
        cs = self.consolidation_state
        if cs is None or getattr(cs.config, "lambda_cc", 0.0) == 0.0:
            return
        for atom_id in atom_ids:
            if atom_id < 0 or atom_id >= self.codebook.shape[0]:
                continue
            current = self.codebook[atom_id]
            force = cs.cap_coverage_force(int(atom_id), current)
            if force is None:
                continue
            self.codebook[atom_id] = self.substrate.normalize(current + force)

    def _snapshot_pre_states(self, atom_ids):
        # C.2.2: capture pre-update atom states so the multiplicative
        # modulation can attenuate the *net* delta after pull/push +
        # anti-collapse + cap-coverage run. Empty / no-op when actuator off.
        cs = self.consolidation_state
        if cs is None or getattr(cs.config, "mu_T", 0.0) == 0.0:
            return {}
        return {
            int(a): self.codebook[a].detach().clone()
            for a in atom_ids
            if 0 <= a < self.codebook.shape[0]
        }

    def _apply_splitting_tension(self, pre_states) -> None:
        # C.2.2: blend codebook[k] = pre + modulation * (post - pre), then
        # renormalize. Multiplicative attenuation of the net update (H12).
        # Early-exits via pre_states being empty when mu_T == 0.
        if not pre_states:
            return
        cs = self.consolidation_state
        for atom_id, pre in pre_states.items():
            modulation = cs.splitting_tension_modulation(atom_id)
            if modulation == 1.0:
                continue
            post = self.codebook[atom_id]
            blended = pre + modulation * (post - pre)
            self.codebook[atom_id] = self.substrate.normalize(blended)

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
