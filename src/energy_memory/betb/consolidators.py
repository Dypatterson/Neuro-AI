"""Consolidation recipes — swappable weight dynamics on the shared circuit.

Lifted 2026-07-25 from `experiments/85_betb_replay_x_consolidation.py`, where
they were defined inline. `CONTEXT-B.md` §8 makes the recipe an explicitly
swappable part: "if one consolidation recipe NULLs on the interaction headline,
that is iterate-fuel, not a dead end."

All recipes expose `.step()`, called after every optimizer step by
`continual.train_task`. One code path serves every arm — the mechanism arms
differ from the controls only by whether a consolidator is present.

**Where the evidence stands (Report 139 + its EWC addendum).** `BennaFusi` and
`EWCAnchor` both clear the §8 interaction gate on the modular toy, and the
matched-protection control showed the plain frozen-L2 anchor reproduces the
*entire* pass (d1 +1.28 ≥ BF +0.96, n=48). Multi-timescale is **not**
load-bearing; the graduating mechanism is soft weight-anchoring + replay, fully
recombinant. Both are therefore **protection**, not restructuring — which is why
`CONTEXT-B.md` §8 still records the restructuring operation as *unbuilt*, and why
`SubspaceRestructure` below is the first attempt at one.
"""

from __future__ import annotations

from typing import Iterable, List

import torch


class Consolidator:
    """Interface: a fixed local weight dynamic, stepped after each optimizer step.

    Anti-homunculus contract (`CLAUDE.md` §measurement rule 6): `step()` must not
    read a performance metric and branch on it. It sees weights, nothing else —
    no task identity, no accuracy, no loss. `tests/test_betb_consolidators.py`
    asserts this rather than trusting the docstring, because three modules in
    `legacy/` claim compliance in prose while violating it in code.
    """

    def step(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class BennaFusi(Consolidator):
    """Benna & Fusi 2016 multi-timescale synaptic consolidation.

    Each parameter `w` is the visible variable `u_1` of a chain `u_1..u_m` with
    geometric capacitances `C_k = 2^(k-1)`. After the optimizer injects
    plasticity into `u_1`, the chain relaxes one step with fixed coupling `g` and
    reflective boundaries:

        du_k = (g / C_k) * [(u_{k-1} - u_k) + (u_{k+1} - u_k)]

    Slow variables protect old structure and feed it back, so the circuit is
    protected *gradedly* rather than hard-frozen — the principled fix for why
    Report 137's binary freeze nulled and degraded late.

    Fixed local linear dynamic (anti-homunculus clean); iterative, no SVD/eig
    (fence-clean per `CONTEXT-B.md` §2).
    """

    def __init__(self, params: Iterable[torch.Tensor], m: int = 4, g: float = 0.03, device=None):
        self.params: List[torch.Tensor] = list(params)
        self.m = m
        self.g = g
        self.C = [float(2 ** k) for k in range(m)]
        self.slow = [
            p.detach().clone().unsqueeze(0).repeat(m - 1, *([1] * p.dim()))
            for p in self.params
        ]

    @torch.no_grad()
    def step(self) -> None:
        for p, s in zip(self.params, self.slow):
            u = torch.cat([p.unsqueeze(0), s], dim=0)
            up = torch.cat([u[:1], u[:-1]], dim=0)
            dn = torch.cat([u[1:], u[-1:]], dim=0)
            du = torch.stack(
                [(self.g / self.C[k]) * ((up[k] - u[k]) + (dn[k] - u[k])) for k in range(self.m)], 0
            )
            u = u + du
            p.copy_(u[0])
            s.copy_(u[1:])


class EWCAnchor(Consolidator):
    """Frozen-reference L2 weight anchor (EWC-lite, uniform / no Fisher).

    Snapshot `theta*` at engage time; after each optimizer step pull live weights
    toward it by `lam`. Single knob, tunable independently so protection strength
    can be *matched* to Benna-Fusi's.

    This is the control that overturned Report 139's Benna-Fusi-specific claim.
    Keep it in every consolidation comparison: it is the competent matched
    control, which `CLAUDE.md` names the highest-yield rule in the repo.
    """

    def __init__(self, params: Iterable[torch.Tensor], lam: float = 0.01, device=None):
        self.params: List[torch.Tensor] = list(params)
        self.lam = lam
        self.ref = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, r in zip(self.params, self.ref):
            p.add_(r - p, alpha=self.lam)


class SubspaceRestructure(Consolidator):
    """A genuine *restructuring* operation — the first in this program.

    `CONTEXT-B.md` §8 defines consolidation as an offline operation that does
    **more than re-presentation**: it restructures the representation toward a
    compact shared schema via a **non-reconstruction objective** that is itself a
    fixed local loss. Every mechanism in Reports 134-139 failed that definition —
    134-138's were replay variants, 139's were weight *protection* — which is why
    the charter still records the restructuring operation as unbuilt and why
    "consolidation manufactures structure replay can't" was never actually tested.

    This one restructures rather than protects: it applies a fixed local
    contraction toward the dominant subspace of each weight matrix's own recent
    activity, implemented as damped power iteration on `W`:

        v <- normalize(W^T (W v))        # one power step, no eig/SVD
        W <- W - eta * (W - (W v) v^T)   # contract toward the rank-1 dominant mode

    Repeated across steps this compresses the circuit toward a low-rank shared
    schema while leaving the residual free to keep adapting — compression, not
    preservation. The reference is **not** frozen (unlike `EWCAnchor`) and there
    is no slow chain replaying old values back (unlike `BennaFusi`), so any
    effect it has cannot be re-described as protection. That separation is the
    whole point: it is what makes the comparison against those two informative.

    Fence-clean: iterative power steps, never a closed-form SVD/eig
    (`CONTEXT-B.md` §2 keeps one-shot factorization as a flashlight).
    Anti-homunculus: a fixed local dynamic on weights, reading no metric.

    **Status: untested.** It has no result behind it. Treat it as recipe #1 of a
    swappable slot, and hold it to the same competent-control bar that deflated
    Benna-Fusi — in particular, report it against `EWCAnchor` at matched strength.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        eta: float = 0.01,
        rank: int = 1,
        device=None,
    ):
        # restructuring applies to weight matrices; biases carry no subspace
        self.params: List[torch.Tensor] = [p for p in params if p.dim() == 2]
        self.eta = eta
        self.rank = rank
        self.v = [torch.randn(p.shape[1], device=p.device) for p in self.params]
        for v in self.v:
            v /= v.norm().clamp_min(1e-12)

    @torch.no_grad()
    def step(self) -> None:
        for i, p in enumerate(self.params):
            v = self.v[i]
            # one damped power step toward the dominant right-singular direction
            w = p @ v                       # [out]
            v_new = p.transpose(0, 1) @ w   # [in]
            n = v_new.norm()
            if n > 1e-12:
                v = v_new / n
                self.v[i] = v
            # contract the matrix toward its rank-1 dominant mode
            w = p @ v
            p.add_(torch.outer(w, v) - p, alpha=self.eta)


RECIPES = {
    "bennafusi": BennaFusi,
    "ewc": EWCAnchor,
    "subspace": SubspaceRestructure,
}


def make_consolidator(name: str, **kwargs):
    """Return a factory `(model, device) -> Consolidator` bound to the shared MLP."""
    if name not in RECIPES:
        raise ValueError(f"unknown consolidation recipe {name!r}; have {sorted(RECIPES)}")
    cls = RECIPES[name]

    def factory(model, device):
        return cls(model.mlp.parameters(), device=device, **kwargs)

    factory.recipe_name = name
    factory.recipe_kwargs = dict(kwargs)
    return factory
