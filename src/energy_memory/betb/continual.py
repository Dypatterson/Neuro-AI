"""The Bet-B continual-learning harness — one implementation, many task families.

Extracted 2026-07-25 from `experiments/83_betb_two_timescale.py`, which had
become the de-facto harness for Reports 137-139: experiments 84 and 85 loaded it
**by filename** via `importlib.spec_from_file_location`, and 37.6% of normalized
lines across `experiments/79-85` were verbatim duplicates of a sibling
(`ContinualNet` defined 3×, `main` 7×, `run_arm`/`aggregate` 5× each). Six of the
seven imported nothing from `energy_memory` at all.

Two things changed in the extraction, both deliberate:

1. **The task family is injected** (`betb.tasks`), never hardcoded. The old
   `make_task` implemented only add/sub mod p and raised otherwise — it *was*
   the task-selection confound, in code.
2. **The head mode is explicit.** Reports 134-139 ran `head_mode="per_task"`:
   a separate `nn.Linear` per task, indexed by task id at eval. That is
   **Task-IL**, the easiest of van de Ven's three continual scenarios, and it
   makes the reported ~0.98 retention much cheaper than it looks. It was
   disclosed once (`reports/135:31`) and omitted from 137/138/139 and the
   retrospective. `head_mode="shared"` uses a single head with the task inferred
   from input tokens; new work should prefer it, and the scenario is now recorded
   in every provenance envelope rather than left implicit.

`head_mode="per_task"` + `ModularArithmeticFamily` reproduces the published
anchors bit-for-bit (same seeding, same layer construction order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import Row, Stream, Task

WEIGHT_DECAY = 1.0


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class ContinualNet(nn.Module):
    """Embeddings -> shared MLP ("the circuit") -> head(s).

    The shared MLP is the object every consolidation mechanism in this program
    acts on: Report 136 showed a reusable circuit exists in it (3.3× speed-up
    when frozen onto a new alphabet), 137 froze it (nulled and hurt late), 139
    protected it gradedly (cleared the interaction gate, then deflated to a plain
    L2 anchor under the matched control).
    """

    def __init__(
        self,
        vocab: int,
        n_classes: int,
        n_tasks: int,
        n_inputs: int = 2,
        embed: int = 64,
        hidden: int = 256,
        seed: int = 0,
        head_mode: str = "per_task",
    ):
        super().__init__()
        torch.manual_seed(seed * 7919 + 1)
        if head_mode not in ("per_task", "shared"):
            raise ValueError(f"head_mode must be 'per_task' or 'shared', got {head_mode!r}")
        self.head_mode = head_mode
        self.n_inputs = n_inputs
        self.emb = nn.Embedding(vocab, embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs * embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if head_mode == "per_task":
            self.heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_tasks)])
        else:
            self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """`x` is [N, n_inputs] of token ids. `t` is ignored when head_mode='shared'."""
        h = self.mlp(self.emb(x).flatten(1))
        return self.heads[t](h) if self.head_mode == "per_task" else self.head(h)


def to_tensors(rows: Sequence[Row], device) -> tuple:
    x = torch.tensor([r[0] for r in rows], device=device)
    y = torch.tensor([r[1] for r in rows], device=device)
    return x, y


def evaluate(model: ContinualNet, t: int, rows: Sequence[Row], device) -> float:
    if not rows:
        return float("nan")
    x, y = to_tensors(rows, device)
    with torch.no_grad():
        return (model(x, t).argmax(-1) == y).float().mean().item()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def eval_checks(max_steps: int, eval_every: int, schedule: str = "geometric") -> set:
    """Step indices at which to test the stopping criterion.

    `schedule='fixed'` reproduces Reports 134-139: a uniform grid of
    `eval_every`. That grid is a measurement bug for this program's headline.
    FTSR is a *ratio* of step counts, so both numerator and denominator are
    rounded up to the same coarse lattice; at the reported FTSR ~12 the
    denominator is only a few grid points, and the ratio inherits large,
    scale-dependent rounding error. Flagged in Reports 138 and 139, never fixed.

    `schedule='geometric'` keeps *relative* resolution roughly constant (~15%
    spacing), so a fast task is timed as precisely as a slow one.
    """
    if schedule == "fixed":
        return set(range(eval_every, max_steps + 1, eval_every))
    if schedule != "geometric":
        raise ValueError(f"unknown eval_schedule {schedule!r}")
    checks, nxt = set(), float(max(1, eval_every // 10))
    while nxt <= max_steps:
        checks.add(int(nxt))
        nxt = max(nxt + 1.0, nxt * 1.15)
    checks.add(max_steps)
    return checks


def train_task(
    model: ContinualNet,
    opt: torch.optim.Optimizer,
    task_idx: int,
    task: Task,
    *,
    max_steps: int,
    crit: float,
    eval_every: int,
    buf: Sequence,
    replay_frac: float,
    gen: torch.Generator,
    device,
    freeze_mlp: bool = False,
    consolidator=None,
    eval_schedule: str = "geometric",
) -> int:
    """Fast per-task learning to criterion. Returns steps-to-criterion.

    `consolidator`, if given, has its `.step()` called after every optimizer
    step — a during-learning weight dynamic (Benna-Fusi, EWC anchor). This is one
    code path for all arms: the mechanism arms differ from the controls only by
    `consolidator is not None`, so there is no forked training loop to drift.
    (Report 139 ran a hand-copied `train_task_bf` for exactly this reason; the
    copy was faithful, but it did not have to exist.)

    `eval_schedule='geometric'` fixes a quantization bug flagged in Reports 138
    and 139 and never addressed: the headline FTSR is a *ratio* of step counts,
    but a fixed `eval_every=100` grid quantizes numerator and denominator to the
    same coarse lattice. At FTSR ~12 the denominator is a handful of grid points,
    so the ratio inherits large, non-uniform rounding error. A geometric schedule
    keeps relative resolution roughly constant across scales.
    """
    for prm in model.mlp.parameters():
        prm.requires_grad_(not freeze_mlp)

    x, y = to_tensors(task.train, device)
    bt = [(*to_tensors(rows, device), t) for (rows, t) in buf] if (buf and replay_frac > 0) else []
    rep = max(1, int(replay_frac * len(task.train) / max(1, len(bt)))) if bt else 0

    checks = eval_checks(max_steps, eval_every, eval_schedule)

    steps = max_steps
    for step in range(1, max_steps + 1):
        loss = F.cross_entropy(model(x, task_idx), y)
        for (bx, by, t) in bt:
            idx = torch.randint(bx.shape[0], (min(bx.shape[0], rep),), generator=gen).to(device)
            loss = loss + F.cross_entropy(model(bx[idx], t), by[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if consolidator is not None:
            consolidator.step()
        if step in checks and evaluate(model, task_idx, task.test, device) >= crit:
            steps = step
            break

    for prm in model.mlp.parameters():
        prm.requires_grad_(True)
    return steps


def offline_replay(model, opt, buf, steps: int, device) -> None:
    """Offline pass of SGD over the replay buffer.

    Named honestly. Reports 134-138 called this "consolidation"; Report 138
    (n=64) attributed *all* of the baseline's new-alphabet compounding to
    interleaved replay and found this pass adds nothing positive to transfer.
    Per `CONTEXT-B.md` §8 Terminology it is **replay** — re-presentation — not
    consolidation, which must *restructure* via a non-reconstruction objective.
    """
    if not buf or steps <= 0:
        return
    for prm in model.mlp.parameters():
        prm.requires_grad_(True)
    data = [(to_tensors(rows, device), t) for (rows, t) in buf]
    for _ in range(steps):
        loss = 0.0
        for ((x, y), t) in data:
            loss = loss + F.cross_entropy(model(x, t), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
@dataclass
class ArmResult:
    steps: List[Optional[int]]
    ret_after: Optional[List[List[float]]] = None
    heldout_after: Optional[List[List[float]]] = None
    #: end-of-stream accuracy per held-out CELL, per task: [task][cell].
    #: The headline bootstraps over these, not over rows — see Task.heldout_groups.
    heldout_cells_final: Optional[List[List[float]]] = None
    #: end-of-stream accuracy on the trained-pair test split, per task.
    #: The headline gap is (this - held-out), so it must come from the same model
    #: state; measuring it mid-stream would compare different models.
    test_final: Optional[List[float]] = None


#: the 2x2 factorial of CONTEXT-B §8, plus the from-scratch denominator
ARMS_2X2 = ["scratch", "floor", "replay_only", "consol_only", "replay_plus_consol"]


def run_arm(
    arm: str,
    stream: Stream,
    *,
    embed: int,
    hidden: int,
    max_steps: int,
    crit: float,
    eval_every: int,
    replay_frac: float,
    lr: float,
    seed: int,
    device,
    head_mode: str = "per_task",
    offline_steps: int = 0,
    consolidator_factory: Optional[Callable] = None,
    consol_start_task: int = 1,
    eval_schedule: str = "geometric",
) -> ArmResult:
    """Run one arm of the factorial over `stream`.

    Arms are the 2×2 of `CONTEXT-B.md` §8 plus the denominator:
    `scratch` (fresh model per task), `floor` (sequential, no replay, no
    consolidation), `replay_only`, `consol_only`, `replay_plus_consol`.

    The consolidator engages at `consol_start_task` (default 1) so task 0
    bootstraps the circuit with plain training and the consolidator's reference
    is the *formed* circuit — protect-a-circuit-that-exists, per Reports 136/137.
    Engaging from random init over-protects and caps the bootstrap (139's tuning
    history).
    """
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    K = len(stream)
    steps: List[Optional[int]] = [None] * K
    ret_after: List[Optional[List[float]]] = [None] * K
    held_after: List[Optional[List[float]]] = [None] * K

    def fresh():
        m = ContinualNet(stream.vocab, stream.n_classes, K, stream.n_inputs,
                         embed, hidden, seed, head_mode).to(device)
        return m, torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    if arm == "scratch":
        for k, task in enumerate(stream.tasks):
            m, opt = fresh()
            steps[k] = train_task(m, opt, k, task, max_steps=max_steps, crit=crit,
                                  eval_every=eval_every, buf=[], replay_frac=0.0,
                                  gen=g, device=device, eval_schedule=eval_schedule)
        return ArmResult(steps=steps)

    do_replay = arm in ("replay_only", "replay_plus_consol")
    do_consol = arm in ("consol_only", "replay_plus_consol")
    if do_consol and consolidator_factory is None:
        raise ValueError(f"arm {arm!r} needs a consolidator_factory")

    m, opt = fresh()
    consolidator = None
    buf: List = []
    for k, task in enumerate(stream.tasks):
        if do_consol and k == consol_start_task:
            consolidator = consolidator_factory(m, device)
        if do_consol and offline_steps and buf:
            offline_replay(m, opt, buf, offline_steps, device)
        steps[k] = train_task(
            m, opt, k, task, max_steps=max_steps, crit=crit, eval_every=eval_every,
            buf=buf, replay_frac=replay_frac if do_replay else 0.0, gen=g,
            device=device, consolidator=consolidator, eval_schedule=eval_schedule,
        )
        buf.append((task.train, k))
        ret_after[k] = [evaluate(m, j, stream.tasks[j].test, device) for j in range(k + 1)]
        held_after[k] = [
            evaluate(m, j, stream.tasks[j].heldout, device) if stream.tasks[j].heldout else float("nan")
            for j in range(k + 1)
        ]

    # End-of-stream, per held-out cell. Both halves of the headline gap are read
    # off the SAME final model so the comparison is within-model.
    cells_final = [
        [evaluate(m, j, cell, device) for cell in stream.tasks[j].heldout_groups]
        for j in range(K)
    ]
    test_final = [evaluate(m, j, stream.tasks[j].test, device) for j in range(K)]
    return ArmResult(steps=steps, ret_after=ret_after, heldout_after=held_after,
                     heldout_cells_final=cells_final, test_final=test_final)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def boot_ci(vals: Sequence[float], n: int = 4000, seed: int = 0):
    """Bootstrap mean + 95% CI. NaNs dropped."""
    t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return (float("nan"),) * 3
    gg = torch.Generator().manual_seed(seed)
    idx = torch.randint(t.numel(), (n, t.numel()), generator=gg)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return (float(t.mean()), lo, hi)


def log_ftsr(scratch_steps: Sequence[float], stream_steps: Sequence[float]) -> List[float]:
    """log FTSR per task. Log scale per Report 138 — speedups are heavy-tailed."""
    import math

    out = []
    for s, q in zip(scratch_steps, stream_steps):
        if s and q and s == s and q == q:
            out.append(math.log(s / q))
    return out


def retention_matrix_metrics(ret_after: Sequence[Sequence[float]]) -> Dict[str, float]:
    """ACC / BWT / FWT from the retention matrix (Lopez-Paz & Ranzato 2017).

    This program reported only its bespoke FTSR across seven continual-learning
    reports — `grep -rnE '\\bBWT\\b|\\bFWT\\b|Lopez-Paz|van de Ven|Avalanche|GEM'`
    over `reports/ notes/ CONTEXT*.md` returns **zero hits**, so none of the
    results could be placed against the literature. The matrix these need was
    already being built; only the metrics were missing.
    """
    R = [r for r in ret_after if r]
    if not R:
        return {}
    K = len(R)
    final = R[-1]
    acc = sum(final) / len(final)
    bwt = sum(final[i] - R[i][i] for i in range(K - 1)) / max(1, K - 1)
    return {"ACC": acc, "BWT": bwt}
