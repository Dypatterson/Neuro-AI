"""Task families for the Bet-B continual-learning harness.

**Why this module exists.** Reports 134-139 all ran on one hardcoded task family:
`make_task` in `experiments/80_betb_continual_transfer.py` implements `add` and
`sub` mod p and raises on anything else. The retrospective
(`notes/RETROSPECTIVE-two-bets-2026-06-06.md` §4) names the consequence as the
program's central confound:

    every task chosen in both bets was solvable by simple means, so "the brain
    mechanism added nothing" is *expected by construction*, not a verdict on the
    mechanism.

You cannot fix that by swapping in another mechanism — only by changing the
task. So the task family is an **injected parameter** here, never a member of the
harness. Adding a regime means adding a `TaskFamily`, not forking the harness.

Two families ship:

- `ModularArithmeticFamily` — reproduces the 134-139 stream exactly (add/sub mod
  p over per-block alphabets). Kept so the published anchors stay checkable.
- `CompositionalAffineFamily` — the **discriminating regime** the retrospective
  §6 specifies: forward transfer requires *recombining* learned primitives rather
  than reusing one protected circuit, so replay + a soft weight anchor should
  demonstrably not saturate it.

## The compositional design

Operators are affine maps on Z_p: `o_i(x) = (m_i * x + c_i) mod p`. A block owns
`n_ops` of them plus a disjoint token range. Every example has the same arity —
three tokens `(op_i, op_j, x)` — because a **primitive** example is just a
composition with identity, `(op_i, IDENTITY, x)`. Uniform arity is deliberate: it
removes "T3 is harder because it has more inputs" as a confound.

Composition is genuinely recombinant. Affine composition is
`o_j(o_i(x)) = (m_j*m_i)x + (m_j*c_i + c_j)`, so the answer is not either
primitive's parameters — it is a *product* and a *cross term*. Memorizing each
`(i, j)` pair therefore does not yield the rule.

**The discriminating split.** A fraction of `(i, j)` pairs is held out and never
trained on, in any block. Systematic composition generalizes to them; pair
memorization does not. This is the property replay and weight-anchoring cannot
manufacture — they protect what was learned, they do not induce the
factorization — which is exactly what makes it a fair test rather than another
task the simple method already saturates.

**The scramble control** (`scramble=True`) assigns each `(i, j)` pair an
independent random affine map unrelated to `o_i` and `o_j`. Composition structure
is destroyed while surface statistics, arity, class balance, and example count
are preserved, so transfer must vanish. This replaces the Report-134 scramble,
which `STATUS.md` records as INVALID because it relabelled an isomorphic addition
and so never broke operation-transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

Row = Tuple[Tuple[int, ...], int]


@dataclass
class Task:
    """One task in the continual stream.

    `heldout` is empty for families without a generalization split. It is
    evaluated but **never** trained on, by any arm.
    """

    train: List[Row]
    test: List[Row]
    heldout: List[Row] = field(default_factory=list)
    kind: str = "task"
    block: int = 0
    label: str = ""


@dataclass
class Stream:
    tasks: List[Task]
    vocab: int
    n_classes: int
    n_inputs: int
    #: task indices that introduce a fresh alphabet — the compounding measure
    xblock: List[int] = field(default_factory=list)
    #: task indices that reuse the current alphabet — within-block transfer
    wblock: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)


def _split(rows: Sequence[Row], frac: float, gen: torch.Generator) -> Tuple[List[Row], List[Row]]:
    idx = torch.randperm(len(rows), generator=gen).tolist()
    n = int(frac * len(rows))
    return [rows[k] for k in idx[:n]], [rows[k] for k in idx[n:]]


class ModularArithmeticFamily:
    """The Reports 134-139 stream: `a+b mod p` / `a-b mod p` over per-block alphabets.

    Preserved verbatim so the published anchors (exp84 `replay_no_consol`
    x-block FTSR ~6.3; exp83 plain ~5.44) stay reproducible. Even tasks are
    `add` on a fresh alphabet (cross-block); odd tasks are `sub` on the alphabet
    the preceding task just introduced (within-block).
    """

    name = "modular"
    n_inputs = 2

    def __init__(self, p: int = 17):
        self.p = p

    def build(self, K: int, frac: float, gen: torch.Generator) -> Stream:
        p = self.p
        tasks: List[Task] = []
        for t in range(K):
            op = "add" if t % 2 == 0 else "sub"
            base = (t // 2) * p
            rows: List[Row] = [
                ((base + i, base + j), (i + j) % p if op == "add" else (i - j) % p)
                for i in range(p)
                for j in range(p)
            ]
            tr, te = _split(rows, frac, gen)
            tasks.append(Task(train=tr, test=te, kind=op, block=t // 2, label=f"{op}@{t // 2}"))
        return Stream(
            tasks=tasks,
            vocab=((K + 1) // 2) * p,
            n_classes=p,
            n_inputs=2,
            xblock=[t for t in range(2, K, 2)],
            wblock=[t for t in range(1, K, 2)],
        )


class CompositionalAffineFamily:
    """The discriminating regime — transfer requires recombining primitives.

    Stream layout, alternating per block `b`:

    - **even task** — *primitives*: learn block `b`'s `n_ops` affine operators,
      presented as `(op_i, IDENTITY, x)`.
    - **odd task** — *compositions*: `(op_i, op_j, x) -> o_j(o_i(x))` over the
      **train pairs** only. The held-out pairs are evaluated, never trained.

    So a composition task can only be learned quickly by *reusing* the operators
    the preceding task just taught — and it can only generalize to held-out pairs
    by learning composition *systematically*. Those are two different bars, and
    the gap between them is the measurement this whole regime exists to make.
    """

    name = "compositional"
    n_inputs = 3

    def __init__(
        self,
        p: int = 17,
        n_ops: int = 6,
        heldout_frac: float = 0.3,
        scramble: bool = False,
    ):
        if n_ops < 3:
            raise ValueError("n_ops must be >= 3 to leave any pairs held out")
        self.p = p
        self.n_ops = n_ops
        self.heldout_frac = heldout_frac
        self.scramble = scramble
        # tokens per block: p values + n_ops operators + 1 identity
        self._span = p + n_ops + 1

    # -- token layout -----------------------------------------------------
    def _val(self, block: int, x: int) -> int:
        return block * self._span + x

    def _op(self, block: int, i: int) -> int:
        return block * self._span + self.p + i

    def _identity(self, block: int) -> int:
        return block * self._span + self.p + self.n_ops

    # -- operator parameters ---------------------------------------------
    def _operators(self, block: int, gen: torch.Generator) -> List[Tuple[int, int]]:
        """`n_ops` affine maps (m, c) with m invertible mod p (m != 0, p prime)."""
        p = self.p
        ms = (torch.randint(1, p, (self.n_ops,), generator=gen)).tolist()
        cs = (torch.randint(0, p, (self.n_ops,), generator=gen)).tolist()
        return list(zip(ms, cs))

    def build(self, K: int, frac: float, gen: torch.Generator) -> Stream:
        p, n_ops = self.p, self.n_ops
        tasks: List[Task] = []
        n_blocks = (K + 1) // 2

        for b in range(n_blocks):
            ops = self._operators(b, gen)
            ident = self._identity(b)

            # ---- primitive task: (op_i, IDENTITY, x) -> o_i(x)
            prim: List[Row] = []
            for i, (m, c) in enumerate(ops):
                for x in range(p):
                    prim.append((( self._op(b, i), ident, self._val(b, x)), (m * x + c) % p))
            tr, te = _split(prim, frac, gen)
            t_idx = 2 * b
            if t_idx < K:
                tasks.append(Task(train=tr, test=te, kind="primitive", block=b,
                                  label=f"prim@{b}"))

            # ---- composition task: (op_i, op_j, x) -> o_j(o_i(x))
            t_idx = 2 * b + 1
            if t_idx >= K:
                continue
            pairs = [(i, j) for i in range(n_ops) for j in range(n_ops)]
            order = torch.randperm(len(pairs), generator=gen).tolist()
            n_held = max(1, int(self.heldout_frac * len(pairs)))
            held = {pairs[k] for k in order[:n_held]}

            # Scramble control: each pair gets an independent random affine map, so
            # no systematic composition exists. Drawn from a SEPARATE generator so
            # the scrambled stream is input-identical to the real one — same tokens,
            # same held-out pairs, same train/test split, only the labels differ.
            # Drawing from `gen` would advance the shared RNG and silently reshuffle
            # the splits, making the control unmatched on exactly the axis it is
            # supposed to isolate. (This is the failure mode that made the Report-134
            # scramble invalid, in a different guise.)
            scram = {}
            if self.scramble:
                sg = torch.Generator().manual_seed(915_587 + b)
                for (i, j) in pairs:
                    m = int(torch.randint(1, p, (1,), generator=sg).item())
                    c = int(torch.randint(0, p, (1,), generator=sg).item())
                    scram[(i, j)] = (m, c)

            train_rows: List[Row] = []
            held_rows: List[Row] = []
            for (i, j) in pairs:
                mi, ci = ops[i]
                mj, cj = ops[j]
                for x in range(p):
                    if self.scramble:
                        m, c = scram[(i, j)]
                        y = (m * x + c) % p
                    else:
                        y = (mj * ((mi * x + ci) % p) + cj) % p
                    row = ((self._op(b, i), self._op(b, j), self._val(b, x)), y)
                    (held_rows if (i, j) in held else train_rows).append(row)

            tr, te = _split(train_rows, frac, gen)
            tasks.append(Task(train=tr, test=te, heldout=held_rows, kind="composition",
                              block=b, label=f"comp@{b}"))

        tasks = tasks[:K]
        return Stream(
            tasks=tasks,
            vocab=n_blocks * self._span,
            n_classes=p,
            n_inputs=3,
            # a fresh block's primitive task is the compounding measure
            xblock=[t for t in range(2, K, 2)],
            # composition-after-primitives is the within-block reuse measure
            wblock=[t for t in range(1, K, 2)],
        )


FAMILIES = {
    "modular": ModularArithmeticFamily,
    "compositional": CompositionalAffineFamily,
}


def build_family(name: str, **kwargs):
    if name not in FAMILIES:
        raise ValueError(f"unknown task family {name!r}; have {sorted(FAMILIES)}")
    return FAMILIES[name](**kwargs)
