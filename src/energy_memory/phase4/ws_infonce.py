"""WS-InfoNCE — within-scene predictive InfoNCE codebook shaping (Phase-3, Stage-1).

Design: notes/emergent-codebook/phase-3-within-scene-predictive-jepa-design.md
Secondary/exploratory **generalization** track. The memorization mechanism
(Reports 055/056/058) is the FLOOR, not relitigated.

A COMPOSE-not-replace write: a single batch-offline InfoNCE pass shapes the FIXED
Phase-3 value codebook into ``C'``, which the graduated heteroassociative memory
``H`` + ``CueDecorrelator`` (Reports 055/056; phase4/hetero_write.py +
phase4/decorrelator.py) then read UNCHANGED. The honest prior (design §0) is that
on real text this writes a co-occurrence MEMORY (held-out-roles null), not a learner;
this module exists to *detect* that cleanly, not to assume it away.

Stage-1 (the only built default, design §2.5):
- learnable = the value-codebook atom PHASES (kept on the FHRR unit-phasor manifold);
- ``g_phi`` = identity (no predictor MLP);
- no EMA / stop-grad (the contrastive denominator + the unit-magnitude constraint
  resist collapse; the target INDEX ``t*`` is the anti-homunculus-relevant invariant).

Anti-homunculus (PASS by construction — design §6):
- The self-target ``t*`` is the TRUE masked filler index of THIS scene: a FIXED
  function of the data, the deliberate OPPOSITE of the deleted ``sims.argmax``
  thermostat (phase2/error_driven_learner.py:106). It is NEVER argmax-mined.
- The candidate / negative set is the PRECOMMITTED value codebook (full-dictionary
  InfoNCE) — no runtime negative mining. ``precommit_swap_negatives``
  (hetero_write.py:77) is available for a sampled-negative variant.
- The write is a SINGLE batch-offline pass over a FROZEN scene buffer
  (``FrozenSceneBuffer.freeze()`` must be called first). No runtime metric is read to
  gate, branch, or select. ``C'`` is produced, then ``H`` reads it: a data-shaping
  handoff, not arbitration over subsystems.

Fence (Phase-5' bright line — design §7): this module computes its OWN ``log_softmax``
over the score matmul (a permitted shared MEASUREMENT). It MUST NOT import or call
``_energy_from_scores`` / ``mhn_energy`` / ``cue_conditioned_scene_energy`` /
``delta_e_content_minus_role`` / ``raw_scene_energy_v0``. ``tau`` (InfoNCE temperature)
is NOT the read-time ``beta``. The downstream read still terminates in a ``top_index``
basin count (``recall_top_index``), never an energy -> min/argmin -> dE.

Within-scene / Inward (binding): the cue is the rest of the CURRENT scene with one
position masked; the target is the true masked filler of THIS scene only. There is no
across-scene target (that is the banned Phase-6 ``E_world``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WSInfoNCEConfig:
    """Stage-1 WS-InfoNCE hyperparameters. ``contrastive=False`` is the no-negatives
    ablation (pull-only, no softmax denominator) — design control that must collapse
    the selectivity-Δ if the contrastive push is what creates role separation."""

    lr: float = 0.05
    epochs: int = 400
    tau: float = 0.05
    contrastive: bool = True
    seed: int = 0


class FrozenSceneBuffer:
    """Closed, seed-fixed buffer of within-scene ``(window, target_local_idx)``
    observations for the batch-offline InfoNCE pass.

    ``freeze()`` must be called before shaping fires — the anti-homunculus condition:
    a single offline pass over a frozen buffer, never a streaming per-observation
    update (runtime error-driven updates are BANNED, STATUS.md live policy)."""

    def __init__(self) -> None:
        self._windows: list[list[int]] = []
        self._tgt: list[int] = []
        self._frozen = False
        self.windows: Optional[torch.Tensor] = None   # LongTensor [N, W]
        self.tgt: Optional[torch.Tensor] = None        # LongTensor [N] value-local idx

    def add(self, window, target_local_idx: int) -> None:
        if self._frozen:
            raise RuntimeError("buffer is frozen; cannot add after freeze()")
        self._windows.append([int(t) for t in window])
        self._tgt.append(int(target_local_idx))

    def freeze(self, device) -> "FrozenSceneBuffer":
        if self._frozen:
            return self
        if not self._windows:
            raise ValueError("cannot freeze an empty buffer")
        widths = {len(w) for w in self._windows}
        if len(widths) != 1:
            raise ValueError(f"ragged windows: widths {sorted(widths)}")
        self.windows = torch.tensor(self._windows, dtype=torch.long, device=device)
        self.tgt = torch.tensor(self._tgt, dtype=torch.long, device=device)
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._tgt)


def within_scene_slot_queries(substrate, positions, codebook, windows, mpos):
    """Differentiable within-scene self-cue (design §2):

        S_rest    = bundle_{p != mpos} bind(positions[p], codebook[tok_p])
        slot_query = normalize( unbind(S_rest, positions[mpos]) )
                   = normalize( S_rest * positions[mpos].conj() )   (FREE, EXACT)

    The FHRR conjugate unbind is the only inversion — this is where the Dorrell port
    survives (no learned bind-inversion). ``positions`` are FIXED roles (not learnable).

    Args:
        positions: ``[W, D]`` complex role/position vectors.
        codebook:  ``[V, D]`` complex unit-phasor atoms (may carry grad).
        windows:   ``[N, W]`` long token ids.
        mpos:      int, the masked position index.
    Returns:
        ``[N, D]`` complex slot queries (a differentiable function of ``codebook``).
    """
    w = windows.shape[1]
    ctx = [p for p in range(w) if p != mpos]
    if not ctx:
        raise ValueError("need at least one context position besides the mask")
    # terms[s, j, :] = positions[ctx_j] * codebook[windows[s, ctx_j]]
    terms = positions[ctx][None, :, :] * codebook[windows[:, ctx]]   # [N, |ctx|, D]
    s_rest = substrate.normalize(terms.sum(dim=1))                   # [N, D] (bundle)
    slot = substrate.normalize(s_rest * positions[mpos].conj())      # [N, D] (unbind)
    return slot


def shape_codebook(
    substrate,
    codebook: torch.Tensor,
    value_row_ids: torch.Tensor,
    positions: torch.Tensor,
    buffer: FrozenSceneBuffer,
    mpos: int,
    config: WSInfoNCEConfig,
):
    """Batch-offline WS-InfoNCE shaping of the value codebook.

    Optimizes the PHASES of the value-codebook rows so the within-scene slot query of
    each scene aligns with its true masked-filler atom and separates from the other
    atoms (full-dictionary InfoNCE). Atoms stay on the unit-phasor manifold, so ``C'``
    is a valid FHRR codebook the graduated ``H`` + decorrelator read unchanged.

    Args:
        codebook:      ``[V, D]`` complex unit-phasor substrate codebook.
        value_row_ids: 1-D long tensor — rows of ``codebook`` that form the value
                       codebook (the learnable + read-out atoms). Their LOCAL order
                       defines ``target_local_idx`` in ``buffer``.
        positions:     ``[W, D]`` complex FIXED roles.
        buffer:        a FROZEN :class:`FrozenSceneBuffer`.
        mpos:          masked position index.
    Returns:
        ``(c_prime_full, value_cb_prime, info)`` where ``c_prime_full`` is the full
        codebook with value rows replaced by the shaped atoms (non-value rows
        unchanged — needed to rebuild slot queries when value atoms also appear in the
        scene context, e.g. real text), ``value_cb_prime = c_prime_full[value_row_ids]``,
        and ``info`` is a diagnostics dict (REPORTED, never gating).
    """
    if not buffer.frozen:
        raise RuntimeError(
            "freeze() the scene buffer before shaping (batch-offline only; "
            "a streaming per-observation write is BANNED)")
    device = codebook.device
    positions = positions.to(device)
    value_row_ids = value_row_ids.to(device)
    d = codebook.shape[1]

    # Phase parametrization: atom = exp(i*theta) keeps |atom| == 1 (FHRR manifold).
    theta = torch.angle(codebook).clone().detach().requires_grad_(True)   # [V, D] real
    learn_mask = torch.zeros(codebook.shape[0], dtype=torch.bool, device=device)
    learn_mask[value_row_ids] = True

    opt = torch.optim.Adam([theta], lr=config.lr)
    windows = buffer.windows.to(device)
    tgt_local = buffer.tgt.to(device)

    loss_trace: list[float] = []
    for _ in range(config.epochs):
        cb = torch.exp(1j * theta)                                   # [V, D] unit-phasor
        value_cb = cb[value_row_ids]                                 # [L, D]
        slot = within_scene_slot_queries(substrate, positions, cb, windows, mpos)
        # OWN log-softmax over the score matmul (fence: never _energy_from_scores).
        logits = (slot @ value_cb.conj().T).real / d / config.tau    # [N, L]
        if config.contrastive:
            loss = torch.nn.functional.cross_entropy(logits, tgt_local)
        else:
            # no-negatives ablation: pull-only toward the positive, no denominator.
            pos = logits.gather(1, tgt_local[:, None]).squeeze(1)
            loss = -pos.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            theta.grad[~learn_mask] = 0                              # freeze non-value rows
        opt.step()
        loss_trace.append(float(loss.detach().cpu()))

    with torch.no_grad():
        # Replace ONLY the value rows; non-value (context) rows stay byte-identical to
        # the input codebook (so cues built from context atoms are isolable, and the
        # raw-C vs C' comparison differs only in the shaped value atoms).
        c_prime_full = codebook.clone()
        c_prime_full[value_row_ids] = torch.exp(1j * theta[value_row_ids]).detach()
        value_cb_prime = c_prime_full[value_row_ids]
        # Diagnostics — reported, NEVER used to gate/branch/select.
        d_eff_before = float(substrate.d_eff(codebook[value_row_ids]).detach().cpu())
        d_eff_after = float(substrate.d_eff(value_cb_prime).detach().cpu())
        drift = float((1.0 - (codebook[value_row_ids].conj()
                              * value_cb_prime).real.mean()).detach().cpu())
    info = {
        # epochs=0 is a degenerate no-op (C' == C up to the phase round-trip); keep it
        # graceful rather than crashing on an empty trace.
        "loss_start": loss_trace[0] if loss_trace else float("nan"),
        "loss_end": loss_trace[-1] if loss_trace else float("nan"),
        "d_eff_before": d_eff_before,
        "d_eff_after": d_eff_after,
        "codebook_drift": drift,
        "epochs": config.epochs,
        "tau": config.tau,
        "lr": config.lr,
        "contrastive": config.contrastive,
        "n_scenes": len(buffer),
        "n_value_atoms": int(value_row_ids.numel()),
    }
    return c_prime_full, value_cb_prime, info


__all__ = [
    "WSInfoNCEConfig",
    "FrozenSceneBuffer",
    "within_scene_slot_queries",
    "shape_codebook",
]
