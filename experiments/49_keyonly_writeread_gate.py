"""Key-only write-then-read gate (G-0 / G-B) — the decisive 065/066 probe.

Spec: notes/emergent-codebook/phase-3-consolidation-write-design.md
Plan: brainstorm-workspace/2026-05-30-research-grounded-plan/02-forward-plan-FINAL.md

The partial-context scene toy (experiments/48) is scene-MHN-retrieval-limited, so
a content write has no lever there (smoke report 048). This gate uses the KEY-ONLY
setting — the exact read Reports 065/066 found has no basin for any binding
operator — where store-as-is genuinely fails AND a write has a direct lever:

  store N (key, value) pairs; cue a KEY ALONE; recover its VALUE.

Reads (all native top_index_hits over the value codebook; chance = 1/C):
  - store_as_is   : bundle B = Σ bind(k_i, v_i); recover = cleanup(unbind(B, k_i))
                    -- reproduces the 065/066 key-only null (decays with N).
  - hebbian_W     : W = Σ v_i ⊗ conj(k_i); recover = cleanup(W k_i / D).
  - delta_W       : Widrow-Hoff error-correcting associative write (the margin
                    write; batch-offline, fixed local residual rule).
  - swap_contrastive_W : delta_W plus an anti-Hebbian push away from a PRECOMMITTED
                    seed-fixed swap-negative value (never sims.argmax) -- the
                    anti-homunculus-clean G-B negative.
  - bounded_oneshot_W : magnitude-bounded one-shot Hebbian (a BTSP-inspired proxy;
                    the faithful sparse-binary BTSP port is flagged open, card
                    biorxiv:2025.05.15.654220 transfers-with-caveats).

Controls (same test set): perfect-cue UPPER BOUND (cue == the value atom);
shuffled-key selectivity (cue a deranged key, target the true value -> chance);
random-codebook (write into a random value codebook -> collapse). Headline is the
vs-no-write anchor: Δ_write = recover(write) − recover(store_as_is).

Anti-homunculus: every W is built by a fixed offline rule over the precommitted
pairs; the delta residual is a local quantity, not a metric-gated branch; the
swap-negative is a seed-fixed draw, never sims.argmax. No runtime arbitration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from energy_memory.phase3.basin_readout import selectivity_delta, top_index_hits
from energy_memory.substrate.torch_fhrr import TorchFHRR

EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")
_retrieve = EXP44._batched_hopfield_retrieve


def _deranged(n: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 7919 + 17)
    for _ in range(2000):
        perm = torch.randperm(n, generator=g)
        if bool((perm != torch.arange(n)).all()):
            return perm.to(device)
    return ((torch.arange(n) + 1) % n).to(device)


def build_pairs(*, D: int, N: int, C: int, seed: int, device: str, key_rho: float = 0.0):
    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    keys = fhrr.random_vectors(N)        # [N, D]
    if key_rho > 0.0:
        # Correlated keys: blend in a shared component so pairwise key cosine
        # rises (more Hebbian crosstalk) while each key stays distinct (the
        # task stays well-posed -- one cue -> one value).
        shared = fhrr.random_vectors(1)  # [1, D]
        keys = fhrr.normalize((1.0 - key_rho) * keys + key_rho * shared)
    values = fhrr.random_vectors(C)      # [C, D] value codebook
    vidx = torch.randint(0, C, (N,), generator=fhrr.generator, device="cpu").to(device)
    pair_vals = values[vidx]             # [N, D]
    bundle = fhrr.normalize(torch.stack(
        [keys[i] * pair_vals[i] for i in range(N)], dim=0).sum(dim=0))  # [D]
    return fhrr, keys, values, vidx, pair_vals, bundle


def _cleanup(fhrr, values, queries, beta, max_iter):
    q = fhrr.normalize(queries)
    _, top_index, entropy, margin = _retrieve(
        fhrr, values, q, beta=beta, max_iter=max_iter)
    return top_index, entropy, margin


# ----- write rules: each returns a recall function key_batch -> query_batch ----
def _hebbian_W(keys, pair_vals, D):
    # W[a,b] = Σ_i v_i[a] conj(k_i)[b]
    return pair_vals.transpose(0, 1) @ keys.conj()           # [D, D]


def _delta_W(keys, pair_vals, D, *, lr, epochs, W0=None):
    N = keys.shape[0]
    W = torch.zeros(D, D, dtype=keys.dtype, device=keys.device) if W0 is None else W0.clone()
    for _ in range(epochs):
        R = (keys @ W.transpose(0, 1)) / D                   # [N, D] = W k_i
        resid = pair_vals - R
        W = W + lr * (resid.transpose(0, 1) @ keys.conj()) / N
    return W


def _swap_contrastive_W(keys, pair_vals, values, vidx, D, *, lr, lr_push, epochs, seed):
    """delta-rule plus an anti-Hebbian push away from a PRECOMMITTED swap-negative
    value (a different value atom, seed-fixed; never sims.argmax)."""
    C = values.shape[0]
    g = torch.Generator().manual_seed(seed * 104729 + 11)
    # precommitted negative value index per pair (different from the true value)
    neg_idx = vidx.clone()
    for i in range(len(vidx)):
        while True:
            j = int(torch.randint(0, C, (1,), generator=g))
            if j != int(vidx[i]):
                neg_idx[i] = j
                break
    neg_vals = values[neg_idx]
    N = keys.shape[0]
    W = torch.zeros(D, D, dtype=keys.dtype, device=keys.device)
    for _ in range(epochs):
        R = (keys @ W.transpose(0, 1)) / D
        resid = pair_vals - R
        W = W + lr * (resid.transpose(0, 1) @ keys.conj()) / N
        # anti-Hebbian push: reduce W's mapping of k_i toward the negative value
        W = W - lr_push * (neg_vals.transpose(0, 1) @ keys.conj()) / N
    return W


def _bounded_oneshot_W(keys, pair_vals, D, *, wmax):
    W = pair_vals.transpose(0, 1) @ keys.conj()
    mag = W.abs()
    scale = torch.clamp(wmax / mag.clamp_min(1e-12), max=1.0)
    return W * scale  # magnitude-bounded (BTSP-inspired proxy)


def _recall_W(keys, W, D):
    return (keys @ W.transpose(0, 1)) / D                    # [N, D]


def run_seed(args, seed: int) -> dict:
    dev = args.device
    fhrr, keys, values, vidx, pair_vals, bundle = build_pairs(
        D=args.D, N=args.N, C=args.C, seed=seed, device=dev, key_rho=args.key_rho)
    N, C, D = args.N, args.C, args.D
    chance = 1.0 / C
    der = _deranged(N, seed, dev)

    def read_rate(query_batch, *, shuffle=False):
        ti, ent, marg = _cleanup(fhrr, values, query_batch, args.beta, args.max_iter)
        target = vidx[der] if shuffle else vidx
        return top_index_hits(ti, target), N, float(ent.mean()), float(marg.mean())

    with torch.no_grad():
        kc = (keys @ keys.conj().T).real.abs() / D
        offdiag = kc[~torch.eye(N, dtype=torch.bool, device=dev)]
        mean_key_cos = float(offdiag.mean())
    out = {"seed": seed, "chance": chance, "mean_key_cos": mean_key_cos}

    # store-as-is (the 065/066 null)
    u_true = torch.stack([fhrr.unbind(bundle, keys[i]) for i in range(N)], dim=0)
    u_shuf = torch.stack([fhrr.unbind(bundle, keys[int(der[i])]) for i in range(N)], dim=0)
    sa_t = read_rate(u_true)
    sa_s = read_rate(u_shuf, shuffle=False)  # cue wrong key, target same true value
    # perfect cue upper bound
    pc_t = read_rate(pair_vals)
    out["store_as_is"] = {
        "true_rate": sa_t[0] / N, "shuffled_rate": sa_s[0] / N,
        "perfect_cue_rate": pc_t[0] / N, "mean_entropy": sa_t[2], "mean_margin": sa_t[3],
    }
    sa_delta = selectivity_delta(true_hits=sa_t[0], true_n=N,
                                 shuffled_hits=sa_s[0], shuffled_n=N, chance=chance)

    # write arms
    writers = {
        "hebbian_W": _hebbian_W(keys, pair_vals, D),
        "delta_W": _delta_W(keys, pair_vals, D, lr=args.lr, epochs=args.epochs),
        "swap_contrastive_W": _swap_contrastive_W(
            keys, pair_vals, values, vidx, D, lr=args.lr, lr_push=args.lr_push,
            epochs=args.epochs, seed=seed),
        "bounded_oneshot_W": _bounded_oneshot_W(keys, pair_vals, D, wmax=args.wmax),
    }
    out["arms"] = {}
    for name, W in writers.items():
        r_true = _recall_W(keys, W, D)
        r_shuf = _recall_W(keys[der], W, D)
        t = read_rate(r_true)
        s = read_rate(r_shuf, shuffle=False)
        sd = selectivity_delta(true_hits=t[0], true_n=N,
                               shuffled_hits=s[0], shuffled_n=N, chance=chance)
        out["arms"][name] = {
            **sd.as_dict(),
            "anchor_vs_store_as_is": sd.delta - sa_delta.delta,
            "true_minus_storeasis_true": t[0] / N - sa_t[0] / N,
        }

    # random-codebook control: recall through delta_W but clean up against a
    # FRESH random value codebook (structure must collapse).
    rand_values = fhrr.random_vectors(C)
    W = writers["delta_W"]
    r_true = _recall_W(keys, W, D)
    ti, _, _ = _cleanup(fhrr, rand_values, r_true, args.beta, args.max_iter)
    out["random_codebook_control"] = {"true_rate": top_index_hits(ti, vidx) / N}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--key-rho", type=float, default=0.0, dest="key_rho",
                    help="0=random keys; ->1 blends a shared component (correlated keys)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--max-iter", type=int, default=12, dest="max_iter")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--lr-push", type=float, default=0.1, dest="lr_push")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--wmax", type=float, default=2.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    results = [run_seed(args, s) for s in range(args.seeds)]

    def agg(fn):
        return sum(fn(r) for r in results) / len(results)

    arms = list(results[0]["arms"].keys())
    summary = {
        "config": vars(args), "chance": 1.0 / args.C,
        "store_as_is": {
            "true_rate": agg(lambda r: r["store_as_is"]["true_rate"]),
            "shuffled_rate": agg(lambda r: r["store_as_is"]["shuffled_rate"]),
            "perfect_cue_rate": agg(lambda r: r["store_as_is"]["perfect_cue_rate"]),
            "null_reproduced": agg(lambda r: r["store_as_is"]["true_rate"]) < 0.4 * agg(lambda r: r["store_as_is"]["perfect_cue_rate"]),
        },
        "arms": {
            a: {
                "true_rate": agg(lambda r: r["arms"][a]["true_rate"]),
                "shuffled_rate": agg(lambda r: r["arms"][a]["shuffled_rate"]),
                "delta": agg(lambda r: r["arms"][a]["delta"]),
                "anchor_vs_store_as_is": agg(lambda r: r["arms"][a]["anchor_vs_store_as_is"]),
                "true_minus_storeasis_true": agg(lambda r: r["arms"][a]["true_minus_storeasis_true"]),
                "two_floor_pass_mean": agg(lambda r: 1.0 if r["arms"][a]["two_floor_pass"] else 0.0),
            } for a in arms
        },
        "random_codebook_control_true_rate": agg(lambda r: r["random_codebook_control"]["true_rate"]),
        "per_seed": results,
    }
    print(json.dumps({k: summary[k] for k in ("chance", "store_as_is", "arms", "random_codebook_control_true_rate")}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
