"""Smoke test for recommendation (b): freq-weighted Benna-Fusi α.

Cannot run the canonical Phase-4 Colab experiment without the Phase-3c
codebook .pt (Drive-only, gitignored). This is a *synthetic* end-to-end
smoke that exercises the same code path — many patterns, heterogeneous
retrieval counts, multi-step cascade — and measures the headline
drill-down `corr_u_m_retrieval_count`. Tells us whether the mechanism
produces a NON-ZERO and INTERPRETABLE correlation between retrieval
count and slow-store mass at modest scale.
"""
from __future__ import annotations
import math, random
import torch
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState


def gini(x):
    x = torch.sort(x.abs())[0]
    n = x.numel()
    if n == 0 or x.sum().item() == 0:
        return 0.0
    cum = torch.cumsum(x, dim=0)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def pearson(a, b):
    am, bm = a.mean(), b.mean()
    av, bv = a - am, b - bm
    denom = torch.sqrt((av**2).sum() * (bv**2).sum())
    return float((av * bv).sum() / denom) if denom > 0 else 0.0


def run(lam: float, seed: int, n_patterns=50, n_cues=400, m=6, alpha=0.25):
    """Mimics the inner Phase-4 loop: a pool of patterns, each cue
    reinforces one pattern (with a power-law retrieval distribution), then
    step_dynamics advances the cascade. We report the headline correlation
    + gini at the end."""
    torch.manual_seed(seed); random.seed(seed)
    cfg = ConsolidationConfig(m=m, alpha=alpha, alpha_freq_lambda=lam)
    s = ConsolidationState(cfg, device="cpu")
    for _ in range(n_patterns):
        s.add_pattern(novelty_strength=1.0)
    # Power-law retrieval (Zipf-ish) — natural Phase-4 regime.
    weights = torch.tensor([1.0 / (i + 1) for i in range(n_patterns)])
    weights = weights / weights.sum()
    for t in range(n_cues):
        # Pick one pattern proportional to weights; reinforce; step cascade.
        idx = int(torch.multinomial(weights, 1).item())
        s.reinforce(idx, magnitude=0.5)
        s.step_dynamics()
    u_m = s.u[:, -1].detach().cpu()
    rc = s.retrieval_count.detach().cpu().to(torch.float32)
    return {
        "corr_u_m_rc": pearson(u_m, rc),
        "gini_u_m": gini(u_m),
        "u_m_mean": float(u_m.mean()),
        "u_m_max": float(u_m.max()),
        "u_m_min": float(u_m.min()),
        "rc_max": int(rc.max()),
        "rc_nonzero": int((rc > 0).sum()),
        "any_nan": bool(torch.isnan(s.u).any()),
    }


if __name__ == "__main__":
    LAMBDAS = [0.0, 0.5, 1.0, 2.0]
    SEEDS = [17, 11, 23]
    print(f"{'lam':>5}  {'seed':>4}  {'corr_u_m_rc':>11}  {'gini_u_m':>9}  {'u_m_mean':>9}  {'u_m_max':>9}  {'rc_max':>6}  {'nan?':>4}")
    agg = {l: [] for l in LAMBDAS}
    for lam in LAMBDAS:
        for seed in SEEDS:
            r = run(lam=lam, seed=seed)
            agg[lam].append(r["corr_u_m_rc"])
            print(f"{lam:>5.2f}  {seed:>4d}  {r['corr_u_m_rc']:>+11.4f}  {r['gini_u_m']:>9.4f}  {r['u_m_mean']:>9.5f}  {r['u_m_max']:>9.5f}  {r['rc_max']:>6d}  {str(r['any_nan']):>4}")
    print()
    print("Per-λ corr_u_m_rc means across seeds:")
    for lam in LAMBDAS:
        vs = agg[lam]
        m_ = sum(vs) / len(vs)
        spread = max(vs) - min(vs)
        print(f"  λ={lam:>4.2f}  mean={m_:+.4f}  spread={spread:.4f}  signs={['+' if v>0 else '-' if v<0 else '0' for v in vs]}")
