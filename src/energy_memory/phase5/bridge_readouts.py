"""Fixed bridge readouts for Phase 5' bundle-first scene states."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


READOUT_ID_CUE_CONDITIONED_SCENE = "cue_conditioned_scene_energy_v1"


def cue_conditioned_scene_energy(cue, scene_state):
    """Return fixed cue-conditioned scene energy.

    E_bridge(q_scene, cue) = -Re(<q_scene, cue>) / D.

    This intentionally scores the final scene state against the fixed cue
    rather than against the scene store. It avoids the Report 104 degeneracy
    where any exact stored-scene attractor has self-similarity 1 and therefore
    raw scene-MHN energy -1 regardless of which scene was selected.

    The function accepts [D] or batched [..., D] tensors. Lower is better.
    """
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("cue_conditioned_scene_energy requires torch") from _IMPORT_ERROR
    if cue.shape[-1] != scene_state.shape[-1]:
        raise ValueError(
            f"cue and scene_state must share last dimension, got "
            f"{cue.shape[-1]} and {scene_state.shape[-1]}"
        )
    return -((scene_state.conj() * cue).real.mean(dim=-1))


def delta_e_content_minus_role(content_energy, role_energy):
    """Phase 5 bridge Delta E sign convention.

    Positive means the role-prior branch has lower energy than the content-prior
    branch under the fixed readout.
    """
    return content_energy - role_energy
