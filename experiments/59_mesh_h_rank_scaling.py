"""MESH scaling decision — graduated H rank/spectrum + recall-under-truncation + cost.

CLASSIFICATION: **DRILL-DOWN / engineering measurement**, NOT a graduation experiment.
This measures storage/cost properties of the graduated heteroassociative write H
(Reports 055/056/057/058); it does not touch the headline graduation metric.

The open decision (phase-4-heteroassociative-write-design.md:82-94; Report 057:72-80):
ship the consolidation write as a **dense H** (D×D complex, ~134 MB @ D=4096, the only
primary-verified rescue) vs a **MESH-style fixed scaffold** (Sharma/Chandra/Fiete 2022,
pdf:mesh-2022; learned part scales with #associations, NOT validated, primary only
card-routed, card warns it "can conflict with emergent-codebook goals").

The load-bearing empirical question the docs assert but never measured: H is a delta-rule
accumulation so rank(H) <= N (buffer size). Is H LOW-RANK at the operating scale?
- If H's EFFECTIVE rank << N and recall survives truncation to it, then a cheap rank-r
  FACTORED H (store U[D,r],sigma[r],V[D,r]; apply H k = U(sigma*(V^H k))) is behaviorally
  near-identical, has NO validation risk, and scales O(rD) — obviating MESH for cost.
- If effective rank ~ N ~ D/2 and recall needs near-full rank, dense is necessary at this
  scale; MESH's fixed-scaffold only matters in the N >> D capacity-cliff regime the
  project does not yet hit.

This script builds H via the graduated path (exp-56 write_H, byte-identical), then for
each (corpus, D) cell reports: singular spectrum, numerical + participation-ratio rank,
recall-under-SVD-truncation curve (in-sample top_index_hits true-rate), and the dense vs
rank-r factored memory at the breakeven r = D/2.

ANTI-HOMUNCULUS: pure offline measurement. No runtime metric gates anything; SVD
truncation is an offline storage analysis of an already-written H; the recall read
terminates in top_index_hits (Phase-5' fence respected). No mechanism change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "exp56_panel", REPO / "experiments" / "56_gd_selectivity_panel.py")
exp56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp56)

from energy_memory.phase3.basin_readout import top_index_hits
from energy_memory.substrate.torch_fhrr import TorchFHRR

TopicCorpus = exp56.TopicCorpus
CorpusWindows = exp56.CorpusWindows
encode_cue = exp56.encode_cue
build_position_vectors = exp56.build_position_vectors
write_H = exp56.write_H
write_read = exp56.write_read

PRESETS = {
    "synthetic": dict(corpus_source="synthetic", W=6, L=8, Vc=16, eps=0.25, max_vocab=512),
    "repo_sample": dict(corpus_source="repo_sample", W=6, L=0, Vc=0, eps=0.0, max_vocab=512),
}

# Truncation fractions of D to probe the recall-vs-rank curve.
TRUNC_FRACS = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]


def _source(args, seed):
    if args.corpus_source == "synthetic":
        return TopicCorpus(seed=seed, W=args.W, L=args.L, Vc=args.Vc, N=args.N,
                           eps=args.eps, posdep=True)
    return CorpusWindows(seed=seed, W=args.W, N=args.N, corpus_source=args.corpus_source,
                         wikitext_name="wikitext-2-raw-v1", max_vocab=args.max_vocab,
                         repo_root=REPO)


def run_cell(args, seed, observed):
    dev = args.device
    src = _source(args, seed)
    sub = TorchFHRR(dim=args.D, seed=seed, device=dev)
    codebook, value_cb, tgt, chance = src.codebook_targets(sub)
    tgt = tgt.to(dev)
    positions = torch.stack(build_position_vectors(sub, src.W)).to(dev)
    mpos, mask_id, W = src.mpos, src.mask_id, src.W
    op = src.observed_positions(observed)

    # In-sample (the full memory): fit and read the same set — this is a storage/capacity
    # measurement of the written H, not a generalization test.
    idx = torch.arange(src.N, device=dev)
    K_true = torch.stack([encode_cue(sub, positions, codebook, w, op, mpos, W, mask_id,
                                     mode="true") for w in src.windows])
    N_assoc = src.N

    # Graduated write (byte-identical to exp-56 write_l2 arm).
    H, dec = write_H(K_true[idx], tgt[idx], value_cb, decorr="l2", dim=args.D,
                     lr=args.lr, epochs=args.epochs)

    # Full-H recall (the reference fidelity).
    ti_full, _, _ = write_read(sub, H, dec, K_true[idx], value_cb, args.beta, args.mi)
    recall_full = top_index_hits(ti_full, tgt) / N_assoc

    # Singular spectrum (move to CPU for SVD stability; D<=2048 here).
    Hc = H.detach().to("cpu")
    U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)   # U[D,D], S[D], Vh[D,D]
    s = S.double()
    s2 = s * s
    total_e = float(s2.sum())
    eff_rank = float((s.sum() ** 2 / s2.sum())) if total_e > 0 else 0.0           # participation ratio
    num_rank = int((s > s[0] * 1e-6).sum()) if float(s[0]) > 0 else 0             # numerical rank
    # smallest r capturing 90/99% of spectral energy
    csum = torch.cumsum(s2, dim=0) / s2.sum()
    r90 = int((csum < 0.90).sum()) + 1
    r99 = int((csum < 0.99).sum()) + 1

    # Recall-under-truncation: H_r = U[:, :r] diag(S[:r]) Vh[:r].
    U_d, S_d, Vh_d = U.to(dev), S.to(dev), Vh.to(dev)
    trunc = []
    for f in TRUNC_FRACS:
        r = max(1, min(args.D, int(round(f * args.D))))
        H_r = (U_d[:, :r] * S_d[:r]) @ Vh_d[:r]
        ti_r, _, _ = write_read(sub, H_r, dec, K_true[idx], value_cb, args.beta, args.mi)
        rec_r = top_index_hits(ti_r, tgt) / N_assoc
        trunc.append({"r": r, "frac_D": f, "recall": rec_r,
                      "recall_ratio_to_full": (rec_r / recall_full) if recall_full else 0.0})

    # Memory: dense D^2 vs rank-r factored 2rD (complex64 = 8 bytes/elem).
    bytes_per = 8
    dense_mb = args.D * args.D * bytes_per / 1e6
    def factored_mb(r):
        return (2 * r * args.D + r) * bytes_per / 1e6
    breakeven_r = args.D // 2  # 2rD < D^2  <=>  r < D/2

    return {
        "seed": seed, "observed": observed, "corpus": args.corpus_source, "D": args.D,
        "N_assoc": N_assoc, "chance": chance, "recall_full_insample": recall_full,
        "num_rank": num_rank, "eff_rank_participation": eff_rank,
        "r90_energy": r90, "r99_energy": r99,
        "N_over_D": N_assoc / args.D, "eff_rank_over_D": eff_rank / args.D,
        "breakeven_r_DH_over_2": breakeven_r,
        "dense_MB": dense_mb,
        "factored_MB_at_eff_rank": factored_mb(int(round(eff_rank))),
        "factored_MB_at_r99": factored_mb(r99),
        "factored_beats_dense_at_eff_rank": factored_mb(int(round(eff_rank))) < dense_mb,
        "truncation_curve": trunc,
        "top_singular": [float(x) for x in S[:5].tolist()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", choices=list(PRESETS), default="repo_sample",
                    dest="corpus_source")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--observed", type=int, nargs="+", default=[2])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    for k, v in PRESETS[args.corpus_source].items():
        setattr(args, k, v)

    cells = []
    for seed in range(args.seeds):
        for obs in args.observed:
            cells.append(run_cell(args, seed, obs))

    # Pool: mean over seeds at each observed.
    from collections import defaultdict
    by_obs = defaultdict(list)
    for c in cells:
        by_obs[c["observed"]].append(c)
    pooled = {}
    for obs, cs in sorted(by_obs.items()):
        n = len(cs)
        pooled[str(obs)] = {
            "n_seeds": n, "D": cs[0]["D"], "corpus": cs[0]["corpus"],
            "N_assoc_mean": sum(c["N_assoc"] for c in cs) / n,
            "recall_full_mean": sum(c["recall_full_insample"] for c in cs) / n,
            "num_rank_mean": sum(c["num_rank"] for c in cs) / n,
            "eff_rank_mean": sum(c["eff_rank_participation"] for c in cs) / n,
            "r90_mean": sum(c["r90_energy"] for c in cs) / n,
            "r99_mean": sum(c["r99_energy"] for c in cs) / n,
            "N_over_D_mean": sum(c["N_over_D"] for c in cs) / n,
            "eff_rank_over_D_mean": sum(c["eff_rank_over_D"] for c in cs) / n,
            "dense_MB": cs[0]["dense_MB"],
            "factored_MB_at_eff_rank_mean": sum(c["factored_MB_at_eff_rank"] for c in cs) / n,
            "factored_beats_dense": all(c["factored_beats_dense_at_eff_rank"] for c in cs),
            # recall retained at each truncation fraction (mean ratio to full)
            "trunc_recall_ratio": {
                str(TRUNC_FRACS[i]): sum(c["truncation_curve"][i]["recall_ratio_to_full"] for c in cs) / n
                for i in range(len(TRUNC_FRACS))
            },
        }

    summary = {"config": {k: getattr(args, k) for k in
                          ("corpus_source", "D", "N", "observed", "seeds", "lr", "epochs")},
               "pooled": pooled, "cells": cells}
    print(json.dumps({"config": summary["config"], "pooled": pooled}, indent=2))
    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
