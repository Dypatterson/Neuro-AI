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
- `PermutationCompositionFamily` — the **discriminating regime** the retrospective
  §6 specifies: forward transfer requires *recombining* learned primitives rather
  than reusing one protected circuit. Empirically validated (Report 140): primitives
  generalize (0.977), trained composition pairs generalize (0.961), held-out pairs
  do not (0.319). Its abelian control collapses the gap to +0.060.

## The discriminating design

Operators are elements of `S_m` acting coordinatewise on `k`-tuples. Every example
is `(op_a, op_b, x_0..x_{k-1})` — a **primitive** is a composition with identity,
so primitives and compositions share one input shape and "the composition task is
harder because it has more inputs" is not available as an explanation.

**The discriminating split.** A fraction of ordered `(i, j)` operator pairs is held
out and never trained on, in any block. Systematic composition generalizes to them;
pair memorization does not. This is the property replay and weight-anchoring cannot
manufacture — they protect what was learned, they do not induce the factorization
— which is what makes it a fair test rather than another task the simple method
already saturates.

**The matched control** is `group='cyclic'`: the abelian `(Z_m)^k`, where composing
two operators *is* pooling them, so the shortcut is correct and held-out saturates.
It collapses the measured gap from +0.642 to +0.060 (CI includes 0). Run it
alongside any mechanism arm — it replaces the Report-134 scramble, which
`STATUS.md` records as INVALID because it relabelled an isomorphic addition and so
never broke operation-transfer.

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
    #: held-out rows grouped by the unit that generalization is measured over
    #: (here: one ordered operator pair). Bootstrap over THESE, not over rows —
    #: per-cell accuracy within a single run ranged 0.000 to 0.856, so the
    #: effective n is the number of cells (8), not the number of rows (1000).
    #: Treating rows as independent would understate the CI by ~11x.
    heldout_groups: List[List[Row]] = field(default_factory=list)
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


class PermutationCompositionFamily:
    """The discriminating regime — non-abelian operator composition (validated).

    Operators are elements of the symmetric group `S_m` acting **coordinatewise**
    on `k`-tuples over `m` points. A primitive applies one operator; a composition
    applies two. Some ordered operator pairs are **never trained on**, and whether
    the model gets those right is the entire measurement.

    ## Why this design and not the obvious one

    The first attempt here (`CompositionalAffineFamily`, removed 2026-07-25) used
    random affine maps `o_i(x) = m_i x + c_i` on Z_p. It is **not learnable**: with
    `(m_i, c_i)` drawn independently per operator, nothing ties operator token `i`
    to its parameters, so generalizing to an unseen `(op, x)` is impossible in
    principle. Measured against the known-grokking modular task in the same
    harness: modular reached test 1.000 by step 10k; the affine design sat at
    **0.000 after 20k with train accuracy 1.000** — pure memorization. A regime
    where nothing learns is exactly as uninformative as one that saturates.

    Five designs were then built and empirically trained (see Report 140). This is
    the only one where all three required properties hold:

    | | measured (n=5 seeds, 20k steps) |
    |---|---|
    | primitives generalize | **0.977** |
    | composition on *trained* pairs | **0.961** |
    | composition on *held-out* pairs | **0.319** |
    | **gap** | **+0.642** [0.466, 0.808] |

    Extended to 60k steps the gap *widens* to +0.669 — held-out stays flat (0.319 →
    0.320) while trained-pair climbs to 0.989. No seed trends toward closure, so
    this is a stable plateau rather than slow convergence.

    ## The matched control (`group='cyclic'`) is load-bearing

    Swapping `S_m` for the abelian `(Z_m)^k` — translations instead of permutations
    — makes the "just pool the two operators" shortcut *correct*, and held-out
    accuracy saturates at 0.908, collapsing the gap to **+0.060 with a CI including
    zero**. That is what shows the gap is about **non-commutative recombination**
    and not merely about task difficulty. Run it alongside any mechanism arm; it is
    the competent matched control the charter requires (`CLAUDE.md` rule 2).

    ## Two integrity fixes, both found by adversarial re-run

    1. **Primitive splits are keyed on the operator, not the cell.** `(i, IDENT)`
       and `(IDENT, i)` compute the *same function*. Splitting them independently
       leaked 69% of primitive-test rows into training under the mirrored slot
       order, inflating primitive accuracy from a true ~0.86-0.92 to 0.98.
    2. **Operator sets are rejection-sampled** so no composite `g_j∘g_i` equals any
       primitive `g_l` or the identity. Without it ~5% of held-out cells are
       answerable by recalling a memorized primitive — no composition required
       (measured 0.428 accuracy on those cells vs 0.224 on genuine ones).
       Acceptance rate ~0.36, so this costs nothing.

    ## Uniform arity

    Every example is `(op_a, op_b, x_0..x_{k-1})`. A primitive is a composition
    with identity, so primitives and compositions share one input shape and "the
    composition task is harder because it has more inputs" is not available as an
    explanation. Points are **split into one token per coordinate**: presenting the
    state as a single token out of `m^k` fails property 1 outright (0.133 vs 0.688
    at matched steps).
    """

    name = "permutation"

    def __init__(
        self,
        m: int = 5,
        k: int = 3,
        n_ops: int = 6,
        heldout_frac: float = 0.27,
        group: str = "symmetric",
    ):
        if group not in ("symmetric", "cyclic"):
            raise ValueError("group must be 'symmetric' (non-abelian) or 'cyclic' (abelian control)")
        if n_ops < 3:
            raise ValueError("n_ops must be >= 3 to leave any operator pairs held out")
        self.m = m
        self.k = k
        self.n_ops = n_ops
        self.heldout_frac = heldout_frac
        self.group = group
        self.n_inputs = 2 + k
        # per-block token span: (n_ops + 1) operator tokens (0 = identity) + m points
        self._span = (n_ops + 1) + m

    # -- token layout -----------------------------------------------------
    def _op_tok(self, block: int, i: int) -> int:
        """i == 0 is IDENTITY; operators are 1..n_ops."""
        return block * self._span + i

    def _pt_tok(self, block: int, x: int) -> int:
        return block * self._span + (self.n_ops + 1) + x

    # -- group elements ---------------------------------------------------
    def _compose(self, g2, g1):
        """Apply g1 then g2 (i.e. g2 ∘ g1), as a map on points."""
        if self.group == "symmetric":
            return tuple(g2[g1[x]] for x in range(self.m))
        return tuple((g2[x] + g1[x]) % self.m for x in range(self.m))

    def _identity(self):
        if self.group == "symmetric":
            return tuple(range(self.m))
        return tuple(0 for _ in range(self.m))

    def _sample_operators(self, gen: torch.Generator):
        """Rejection-sample so no composite collides with a primitive or identity."""
        ident = self._identity()
        for _ in range(2000):
            ops = []
            for _ in range(self.n_ops):
                if self.group == "symmetric":
                    g = tuple(torch.randperm(self.m, generator=gen).tolist())
                else:
                    g = tuple(torch.randint(0, self.m, (self.m,), generator=gen).tolist())
                ops.append(g)
            if len(set(ops)) != self.n_ops or ident in ops:
                continue
            bad = False
            for i in range(self.n_ops):
                for j in range(self.n_ops):
                    if i == j:
                        continue
                    c = self._compose(ops[j], ops[i])
                    if c == ident or c in ops:
                        bad = True
                        break
                if bad:
                    break
            if not bad:
                return ops
        raise RuntimeError(
            f"could not sample {self.n_ops} operators from {self.group} with m={self.m} "
            "such that no composite collides with a primitive — raise m or lower n_ops"
        )

    # -- labels -----------------------------------------------------------
    def _apply(self, g, state):
        if self.group == "symmetric":
            return tuple(g[x] for x in state)
        return tuple((x + g[x_i]) % self.m for x_i, x in enumerate(state))

    def _label(self, state) -> int:
        out = 0
        for x in state:
            out = out * self.m + x
        return out

    def _states(self):
        out, cur = [], [0] * self.k
        total = self.m ** self.k
        for n in range(total):
            v, s = n, []
            for _ in range(self.k):
                s.append(v % self.m)
                v //= self.m
            out.append(tuple(reversed(s)))
        del cur
        return out

    def build(self, K: int, frac: float, gen: torch.Generator) -> Stream:
        states = self._states()
        n_states = len(states)
        tasks: List[Task] = []
        n_blocks = (K + 1) // 2

        for b in range(n_blocks):
            ops = self._sample_operators(gen)
            ident_tok = self._op_tok(b, 0)

            def row(oa: int, ob: int, st, label_state):
                toks = (self._op_tok(b, oa), self._op_tok(b, ob)) + tuple(
                    self._pt_tok(b, x) for x in st
                )
                return (toks, self._label(label_state))

            # ---- primitive task -------------------------------------------------
            # FIX 1: split keyed on the OPERATOR, so (i, IDENT) and (IDENT, i) —
            # which compute the same function — share one split and the mirrored
            # slot order cannot leak a test fact into training.
            prim_tr: List[Row] = []
            prim_te: List[Row] = []
            for i in range(1, self.n_ops + 1):
                g = ops[i - 1]
                og = torch.Generator().manual_seed(hash((b, i, "prim")) % (2 ** 31))
                order = torch.randperm(n_states, generator=og).tolist()
                cut = int(frac * n_states)
                tr_idx, te_idx = set(order[:cut]), set(order[cut:])
                for si, st in enumerate(states):
                    out = self._apply(g, st)
                    dest = prim_tr if si in tr_idx else prim_te
                    dest.append(row(i, 0, st, out))     # (op, IDENT, x)
                    dest.append(row(0, i, st, out))     # (IDENT, op, x) — same split
                del te_idx
            t_idx = 2 * b
            if t_idx < K:
                tasks.append(Task(train=prim_tr, test=prim_te, kind="primitive",
                                  block=b, label=f"prim@{b}"))

            # ---- composition task ----------------------------------------------
            t_idx = 2 * b + 1
            if t_idx >= K:
                continue
            pairs = [(i, j) for i in range(1, self.n_ops + 1)
                     for j in range(1, self.n_ops + 1) if i != j]
            order = torch.randperm(len(pairs), generator=gen).tolist()
            n_held = max(1, int(self.heldout_frac * len(pairs)))
            held = {pairs[t] for t in order[:n_held]}

            comp_tr: List[Row] = []
            comp_te: List[Row] = []
            comp_held: List[Row] = []
            held_groups: List[List[Row]] = []
            for (i, j) in pairs:
                comp = self._compose(ops[j - 1], ops[i - 1])   # apply g_i then g_j
                if (i, j) in held:
                    cell = [row(i, j, st, self._apply(comp, st)) for st in states]
                    comp_held.extend(cell)
                    held_groups.append(cell)
                    continue
                og = torch.Generator().manual_seed(hash((b, i, j, "comp")) % (2 ** 31))
                o2 = torch.randperm(n_states, generator=og).tolist()
                cut = int(frac * n_states)
                tr_idx = set(o2[:cut])
                for si, st in enumerate(states):
                    r = row(i, j, st, self._apply(comp, st))
                    (comp_tr if si in tr_idx else comp_te).append(r)

            tasks.append(Task(train=comp_tr, test=comp_te, heldout=comp_held,
                              heldout_groups=held_groups, kind="composition",
                              block=b, label=f"comp@{b}"))

        tasks = tasks[:K]
        return Stream(
            tasks=tasks,
            vocab=n_blocks * self._span,
            n_classes=self.m ** self.k,
            n_inputs=self.n_inputs,
            xblock=[t for t in range(2, K, 2)],
            wblock=[t for t in range(1, K, 2)],
        )


FAMILIES = {
    "modular": ModularArithmeticFamily,
    "permutation": PermutationCompositionFamily,
}


def build_family(name: str, **kwargs):
    if name not in FAMILIES:
        raise ValueError(f"unknown task family {name!r}; have {sorted(FAMILIES)}")
    return FAMILIES[name](**kwargs)
