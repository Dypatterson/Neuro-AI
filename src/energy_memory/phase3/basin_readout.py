"""Basin-membership readout + Selectivity-Δ for the Phase-3 consolidation-write gate.

The canonical readout is ``top_index_hits`` (tix): a recall counts as a HIT iff
the Modern Hopfield pre-decode argmax basin equals the cued target index
(Report 066:19) — **not** whether a downstream top-1 cosine decode happened to
land on the right value (the two disagree by ~7pt on FHRR, Report 066:37).

The headline is the value-codebook ``top_index_hits`` **Selectivity-Δ** with the
**two-floor Wilson rule** (see
``notes/emergent-codebook/phase-3-consolidation-write-design.md`` §Headline):

    Δ = recoverability(true-role cue) − recoverability(role-shuffled cue)

PASS iff the absolute true-role arm Wilson lower bound > chance (1/N) **and** the
Δ Wilson interval (Newcombe) excludes 0. A Δ>0 produced by a *sub-chance* shuffled
arm is the anti-overlap signature and is **rejected**, not a pass.

Anti-homunculus: this module is a pure offline measurement fit on a held-out
split; it never selects a condition at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from energy_memory.phase2.metrics import wilson_interval


def top_index_hits(top_index, target) -> int:
    """Count basin-membership hits: positions where the settled argmax basin
    index equals the cued target index. ``top_index`` and ``target`` are 1-D
    integer tensors of equal length (the batched-Hopfield ``top_index`` output
    and the ground-truth target atom indices)."""
    if top_index.shape != target.shape:
        raise ValueError(
            f"top_index {tuple(top_index.shape)} and target "
            f"{tuple(target.shape)} must match"
        )
    return int((top_index == target).sum().detach().cpu())


def newcombe_diff_ci(
    s1: int, n1: int, s2: int, n2: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe's hybrid-score CI for the difference of two proportions
    (p1 − p2). Uses the Wilson score intervals of each arm — the project's
    standing CI convention — rather than a normal approximation, so it stays
    honest at the small counts of a toy gate."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both arm totals must be positive")
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson_interval(s1, n1, z)
    l2, u2 = wilson_interval(s2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


@dataclass(frozen=True)
class SelectivityDelta:
    true_rate: float
    true_lo: float
    true_hi: float
    shuffled_rate: float
    shuffled_lo: float
    shuffled_hi: float
    delta: float
    delta_lo: float
    delta_hi: float
    chance: float
    n: int
    two_floor_pass: bool
    anti_overlap_flag: bool

    def as_dict(self) -> dict:
        return {
            "true_rate": self.true_rate,
            "true_ci": [self.true_lo, self.true_hi],
            "shuffled_rate": self.shuffled_rate,
            "shuffled_ci": [self.shuffled_lo, self.shuffled_hi],
            "delta": self.delta,
            "delta_ci": [self.delta_lo, self.delta_hi],
            "chance": self.chance,
            "n": self.n,
            "two_floor_pass": self.two_floor_pass,
            "anti_overlap_flag": self.anti_overlap_flag,
        }


def selectivity_delta(
    *,
    true_hits: int,
    true_n: int,
    shuffled_hits: int,
    shuffled_n: int,
    chance: float,
    z: float = 1.96,
) -> SelectivityDelta:
    """Compute the Selectivity-Δ headline with the two-floor Wilson rule.

    ``two_floor_pass`` is True iff (1) the true-role arm Wilson lower bound is
    strictly above ``chance`` AND (2) the Δ Newcombe interval excludes 0 AND
    (3) the shuffled arm is not significantly sub-chance (anti-overlap guard).
    """
    if true_n <= 0 or shuffled_n <= 0:
        raise ValueError("arm totals must be positive")
    tp, sp = true_hits / true_n, shuffled_hits / shuffled_n
    tl, tu = wilson_interval(true_hits, true_n, z)
    sl, su = wilson_interval(shuffled_hits, shuffled_n, z)
    dlo, dhi = newcombe_diff_ci(true_hits, true_n, shuffled_hits, shuffled_n, z)
    # anti-overlap signature: the shuffled (destroyed-structure) arm sits
    # significantly BELOW chance, so a positive Δ is an artifact, not selectivity.
    anti_overlap = su < chance
    two_floor = (tl > chance) and (dlo > 0.0) and (not anti_overlap)
    return SelectivityDelta(
        true_rate=tp, true_lo=tl, true_hi=tu,
        shuffled_rate=sp, shuffled_lo=sl, shuffled_hi=su,
        delta=tp - sp, delta_lo=dlo, delta_hi=dhi,
        chance=chance, n=true_n,
        two_floor_pass=two_floor, anti_overlap_flag=anti_overlap,
    )


__all__ = ["top_index_hits", "newcombe_diff_ci", "SelectivityDelta", "selectivity_delta"]
