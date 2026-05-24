"""Smoke test for recommendation (a): range-shaped replay buffer.

Goal: prototype role/content-factored sampling on top of the existing
ReplayStore. The Dorrell-Whittington ICLR 2025 theorem says modularisation
in nonneg + energy-efficient autoencoders is *forced* by rectangular joint
support of role and content. Default `ReplayStore.sample()` samples whole
traces by priority. The range-shaped variant decomposes each trace into a
(role_tag, content_tag) pair and samples roles and contents independently
before pairing them.

This smoke verifies:
  1. The shape of the joint (role, content) distribution from baseline
     vs range-shaped sampling.
  2. The two samplers are different (KL divergence > 0 between joint
     distributions).
  3. The range-shaped sampler approximates the product of marginals
     (rectangular support), while baseline preserves the (often skewed)
     joint co-occurrence.
"""
from __future__ import annotations
import math, random
import torch


class FakeTrace:
    """Stand-in for TrajectoryTrace carrying role/content tags."""
    def __init__(self, role: int, content: int, gate: float):
        self.role = role
        self.content = content
        self.gate = gate

    def __repr__(self):
        return f"T(r={self.role},c={self.content})"


class BaselineReplayBuffer:
    """Mimics ReplayStore.sample(): pick whole traces by priority."""
    def __init__(self, traces):
        self.traces = list(traces)

    def sample_n(self, n, rng):
        ws = torch.tensor([t.gate for t in self.traces], dtype=torch.float32)
        idx = torch.multinomial(ws / ws.sum(), n, replacement=True, generator=rng)
        return [self.traces[int(i)] for i in idx]


class RangeShapedReplayBuffer:
    """Sample role and content independently, then look up a real trace
    with matching (role, content); if none exists, fall back to closest
    role-only match. Independence at the sampler is what creates the
    rectangular support."""
    def __init__(self, traces):
        self.traces = list(traces)
        # Marginal weights from existing traces' priorities.
        self.by_role: dict[int, float] = {}
        self.by_content: dict[int, float] = {}
        for t in self.traces:
            self.by_role[t.role] = self.by_role.get(t.role, 0.0) + t.gate
            self.by_content[t.content] = self.by_content.get(t.content, 0.0) + t.gate
        # Index for lookup.
        self.idx: dict[tuple[int, int], list[FakeTrace]] = {}
        for t in self.traces:
            self.idx.setdefault((t.role, t.content), []).append(t)
        self.roles = list(self.by_role.keys())
        self.contents = list(self.by_content.keys())

    def sample_n(self, n, rng):
        rw = torch.tensor([self.by_role[r] for r in self.roles], dtype=torch.float32)
        cw = torch.tensor([self.by_content[c] for c in self.contents], dtype=torch.float32)
        ri = torch.multinomial(rw / rw.sum(), n, replacement=True, generator=rng)
        ci = torch.multinomial(cw / cw.sum(), n, replacement=True, generator=rng)
        out = []
        for r_, c_ in zip(ri, ci):
            r = self.roles[int(r_)]; c = self.contents[int(c_)]
            cand = self.idx.get((r, c))
            if cand:
                out.append(cand[0])
            else:
                # No (r,c) trace exists — emit a synthetic pair (the actual
                # implementation would re-bind a stored content under the
                # sampled role).
                out.append(FakeTrace(role=r, content=c, gate=0.0))
        return out


def joint_histogram(traces, n_roles, n_contents):
    H = torch.zeros(n_roles, n_contents)
    for t in traces:
        H[t.role, t.content] += 1
    H = H / max(H.sum().item(), 1.0)
    return H


def kl_div(p, q, eps=1e-9):
    p = p.flatten() + eps; p = p / p.sum()
    q = q.flatten() + eps; q = q / q.sum()
    return float((p * (p.log() - q.log())).sum())


def rectangularity(H, eps=1e-9):
    """Distance from H to the outer product of its marginals (KL).
    0 = perfectly rectangular (factored); larger = correlated."""
    pr = H.sum(dim=1, keepdim=True)
    pc = H.sum(dim=0, keepdim=True)
    fact = pr @ pc
    return kl_div(H, fact)


def main():
    rng = torch.Generator().manual_seed(42)
    random.seed(42)
    # Realistic Phase-4 setting: 8 roles, 32 contents.
    # Buffer has skewed co-occurrence — say, role i mostly co-occurs with
    # contents {i, i+1, i+2}. This simulates "episodic" joint structure.
    N_ROLES, N_CONTENTS = 8, 32
    traces = []
    for _ in range(400):
        role = random.randrange(N_ROLES)
        # Content concentrated around role.
        content = (role * 4 + random.randrange(4)) % N_CONTENTS
        gate = random.uniform(0.1, 1.0)
        traces.append(FakeTrace(role, content, gate))

    baseline = BaselineReplayBuffer(traces)
    rangeshaped = RangeShapedReplayBuffer(traces)

    SAMPLE_N = 4000
    bs = baseline.sample_n(SAMPLE_N, rng)
    rs = rangeshaped.sample_n(SAMPLE_N, rng)

    H_orig = joint_histogram(traces, N_ROLES, N_CONTENTS)
    H_bs = joint_histogram(bs, N_ROLES, N_CONTENTS)
    H_rs = joint_histogram(rs, N_ROLES, N_CONTENTS)

    print(f"Original buffer joint rectangularity (KL to factored): {rectangularity(H_orig):.4f}")
    print(f"Baseline-sampled joint rectangularity:                 {rectangularity(H_bs):.4f}")
    print(f"Range-shaped-sampled joint rectangularity:             {rectangularity(H_rs):.4f}")
    print()
    print(f"KL(baseline_joint || rangeshape_joint): {kl_div(H_bs, H_rs):.4f}")
    print(f"KL(rangeshape_joint || baseline_joint): {kl_div(H_rs, H_bs):.4f}")
    print()
    # Coverage: what fraction of the (role, content) grid does each sampler
    # actually reach?
    bs_cells = float((H_bs > 0).sum())
    rs_cells = float((H_rs > 0).sum())
    print(f"Baseline reaches {bs_cells:.0f}/{N_ROLES*N_CONTENTS} (role,content) cells")
    print(f"Range-shaped reaches {rs_cells:.0f}/{N_ROLES*N_CONTENTS} cells")
    # Marginals should be preserved by range-shape (that's the point).
    rm_bs = H_bs.sum(dim=1); rm_rs = H_rs.sum(dim=1)
    cm_bs = H_bs.sum(dim=0); cm_rs = H_rs.sum(dim=0)
    print(f"Role marginal L1 diff (baseline vs rangeshape):    {(rm_bs - rm_rs).abs().sum():.4f}")
    print(f"Content marginal L1 diff (baseline vs rangeshape): {(cm_bs - cm_rs).abs().sum():.4f}")


if __name__ == "__main__":
    main()
