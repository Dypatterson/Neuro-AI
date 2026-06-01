"""Exp 57 — END-TO-END WIRING CHECK for the integrated consolidation write.

Report 057 folded the validated heteroassociative-write + L2 cue-space decorrelator
(Reports 055 graduation / 056 G-D) into the production orchestrator
``OnlineCodebookUpdater`` (phase34/online_codebook.py). This harness validates the
*wiring* of that fold-in — NOT the mechanism, which already graduated.

CLASSIFICATION: DRILL-DOWN / integration-validation, NOT a graduation experiment.
The claim under test is "the integrated public-API path reproduces the standalone
exp-56 harness result (Reports 055/056)", not a fresh graduation.

DESIGN: head-to-head equivalence on byte-identical data.
  For each (corpus, D, seed, observed) cell, build the masked-encoding substrate, the
  value codebook, the per-window RAW masked cue K_true (the key), the per-scene
  fixed-point-free position-deranged cue K_der (the role-shuffle arm), and the
  value-codebook-local targets tgt — EXACTLY as experiments/56 builds them (we reuse
  exp-56's own TopicCorpus / CorpusWindows / encode_cue / write_H / write_read /
  build_position_vectors). Then we run the SAME data through two paths:

    Path A (standalone reference): exp56.write_H + exp56.write_read  (the code that
            produced Reports 055/056).
    Path B (integrated): OnlineCodebookUpdater.observe(cue=) x N -> consolidate_hetero()
            -> recall_hetero(cue)  (the production fold-in, Report 057).

  Both paths call the SAME phase4.hetero_write.heteroassociative_write +
  recall_top_index + CueDecorrelator with IDENTICAL defaults (lr=0.5, epochs=20,
  ridge=1e-5, beta=10, max_iter=12), so on identical data they must produce
  BIT-IDENTICAL basin indices. We assert torch.equal on the basin indices and H, and
  recompute the two-floor role-Selectivity-Δ from the integrated path's own output.

  KEY MAPPING: the integrated path writes targets via ``self.codebook[vidx]`` and reads
  ``top_index`` over ``self.codebook``, whereas exp-56 uses ``value_cb``. So Path B is
  constructed with ``codebook=value_cb`` and ``target_id`` = the value-codebook-local
  index. This is the load-bearing equivalence mapping.

In-sample only (tr==te==range(N)): this is a WIRING check by construction.
Memorization-recall is the PHASE-3 FLOOR (CLAUDE.md contextual-completion, NOT
prediction) — not the project ceiling; held-out compositional generalization
(~chance on real text) is the PREDICTED Phase-3 floor and a Phase-5 deliverable,
deferred by design (see CONTEXT.md), not relevant to a wiring-equivalence check.

Run:
  PYTHONPATH=src .venv/bin/python experiments/57_e2e_integration_wiring_check.py \
      --device cpu --out reports/058_e2e_integration_wiring/wiring_results.json
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

# exp-56's module name starts with a digit -> load by path.
_spec = importlib.util.spec_from_file_location(
    "exp56_panel", REPO / "experiments" / "56_gd_selectivity_panel.py")
exp56 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp56)

from energy_memory.substrate.torch_fhrr import TorchFHRR
from energy_memory.phase3.basin_readout import top_index_hits, selectivity_delta
from energy_memory.phase34.online_codebook import OnlineCodebookUpdater

# Reuse exp-56's exact building blocks (guarantees byte-identical data + reference path).
TopicCorpus = exp56.TopicCorpus
CorpusWindows = exp56.CorpusWindows
encode_cue = exp56.encode_cue
build_position_vectors = exp56.build_position_vectors
write_H = exp56.write_H
write_read = exp56.write_read
batched_hopfield_topindex = exp56.batched_hopfield_topindex


def build_data(corpus, D, seed, observed, W, dev, *, L, Vc, N, eps, max_vocab):
    """Replicate exactly the (seed, observed) data construction of exp56.run_cell,
    in-sample (tr==te==range(N)). Returns sub, value_cb, tgt, chance, K_true, K_der."""
    if corpus == "synthetic":
        src = TopicCorpus(seed=seed, W=W, L=L, Vc=Vc, N=N, eps=eps, posdep=True)
    else:
        src = CorpusWindows(seed=seed, W=W, N=N, corpus_source=corpus,
                            wikitext_name="wikitext-2-raw-v1", max_vocab=max_vocab,
                            repo_root=REPO)
    op = src.observed_positions(observed)
    mpos, mask_id, W = src.mpos, src.mask_id, src.W
    sub = TorchFHRR(dim=D, seed=seed, device=dev)
    codebook, value_cb, tgt, chance = src.codebook_targets(sub)
    tgt = tgt.to(dev)
    positions = build_position_vectors(sub, W)

    def cue_set(mode):
        return torch.stack([encode_cue(sub, positions, codebook, w, op, mpos, W,
                                       mask_id, mode=mode) for w in src.windows])

    K_true = cue_set("true")

    # Per-scene fixed-point-free position derangement = the role-shuffle (Control 3)
    # arm. Verbatim copy of exp56.run_cell.deranged_cue_set (same seed formula).
    def deranged_cue_set():
        keys = []
        for si, w in enumerate(src.windows):
            g = torch.Generator().manual_seed(seed * 100003 + si * 131 + 17)
            rp = list(range(W))
            if len(op) >= 2:
                while True:
                    pp = torch.randperm(len(op), generator=g).tolist()
                    if all(i != p for i, p in enumerate(pp)):
                        break
                for i, p in enumerate(op):
                    rp[p] = op[pp[i]]
            else:
                cands = [p for p in range(W) if p != op[0]]
                rp[op[0]] = cands[int(torch.randint(0, len(cands), (1,), generator=g))]
            keys.append(encode_cue(sub, positions, codebook, w, op, mpos, W,
                                   mask_id, mode="true", perm=rp))
        return torch.stack(keys)

    K_der = deranged_cue_set()
    return sub, value_cb, tgt, float(chance), K_true, K_der


def run_paths(sub, value_cb, tgt, K_true, K_der, *, D, lr, epochs, beta, mi):
    """Run the SAME data through Path A (standalone) and Path B (integrated)."""
    N = K_true.shape[0]

    # ---- Path A: the standalone exp-56 reference (the code behind Reports 055/056) ----
    H_a, dec_a = write_H(K_true, tgt, value_cb, decorr="l2", dim=D, lr=lr, epochs=epochs)
    ti_true_a = write_read(sub, H_a, dec_a, K_true, value_cb, beta, mi)[0]
    ti_der_a = write_read(sub, H_a, dec_a, K_der, value_cb, beta, mi)[0]

    # ---- Path B: the integrated production fold-in (Report 057 public API) ----
    upd = OnlineCodebookUpdater(
        sub, value_cb,
        hetero_write_enabled=True,
        decorrelator_enabled=True,
        hetero_lr=lr,
        hetero_epochs=epochs,
        hetero_contrastive=False,
        decorrelator_ridge=1e-5,
    )
    for i in range(N):
        t = int(tgt[i])
        # slot_query/predicted_id feed only the legacy pull/push gate; passing the
        # target atom keeps that path inert (quality high). The hetero write uses cue.
        upd.observe(target_id=t, slot_query=value_cb[t], predicted_id=t, cue=K_true[i])
    meta = upd.consolidate_hetero()
    ti_true_b = upd.recall_hetero(K_true, beta=beta, max_iter=mi)[0]
    ti_der_b = upd.recall_hetero(K_der, beta=beta, max_iter=mi)[0]

    # ---- Equivalence diagnostics ----
    H_equal = bool(torch.equal(H_a, upd.hetero_H))
    H_allclose = bool(torch.allclose(H_a, upd.hetero_H, atol=1e-6, rtol=1e-5))
    tix_true_equal = bool(torch.equal(ti_true_a, ti_true_b))
    tix_der_equal = bool(torch.equal(ti_der_a, ti_der_b))

    # ---- Integrated-path readout-leak control: read the integrated H + integrated
    # decorrelator against a FRESH random codebook -> must collapse to chance. ----
    rand_cb = sub.random_vectors(value_cb.shape[0])
    recalled = (upd.hetero_decorrelator.apply(K_true) @ upd.hetero_H.T) / D
    ti_rand = batched_hopfield_topindex(sub, rand_cb, recalled, beta=beta, max_iter=mi)[0]
    rand_hits = top_index_hits(ti_rand, tgt)

    return {
        "meta": meta,
        "H_equal": H_equal,
        "H_allclose": H_allclose,
        "tix_true_equal": tix_true_equal,
        "tix_der_equal": tix_der_equal,
        # raw hits for pooling (integrated Path B)
        "B_true_hits": top_index_hits(ti_true_b, tgt),
        "B_der_hits": top_index_hits(ti_der_b, tgt),
        # raw hits Path A (for the per-cell point anchor)
        "A_true_hits": top_index_hits(ti_true_a, tgt),
        "A_der_hits": top_index_hits(ti_der_a, tgt),
        "rand_hits": rand_hits,
        "n": N,
    }


# Canonical in-sample cells anchored to Reports 055/056 (the report seed/config).
PRESETS = {
    "repo_sample": dict(D=1024, N=500, W=6, max_vocab=512, L=8, Vc=16, eps=0.25,
                        seeds=[0, 1, 2], observed=[1, 2, 3]),
    "synthetic":   dict(D=2048, N=500, W=6, max_vocab=512, L=8, Vc=16, eps=0.25,
                        seeds=[0, 1, 2, 3, 4], observed=[1, 2, 3]),  # 5 seeds = Report-056 config
    # Graduation-scale convincer (Reports 055/056 WikiText-2 D=4096 config). CUDA-only
    # in practice (complex eigh + dense H @ D=4096); see notebooks/058_*.ipynb for Colab.
    "wikitext":    dict(D=4096, N=1000, W=6, max_vocab=2000, L=8, Vc=16, eps=0.25,
                        seeds=[0, 1, 2], observed=[1, 2, 3]),
}

# Report-055/056 in-sample pooled role-Selectivity-Δ targets (for the anchor print).
REPORT_TARGETS = {
    ("repo_sample", 1): 0.352, ("repo_sample", 2): 0.775, ("repo_sample", 3): 0.933,
    ("synthetic", 2): 0.592,
    # Report-056 G-D D=4096 WikiText role-Selectivity-Δ (true-pos − deranged).
    ("wikitext", 1): 0.323, ("wikitext", 2): 0.730, ("wikitext", 3): 0.907,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--corpora", nargs="+",
                    choices=["repo_sample", "synthetic", "wikitext"],
                    default=["repo_sample", "synthetic"])
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12)
    # Optional per-run overrides of the chosen preset(s) (0/None -> keep preset).
    ap.add_argument("--override-D", type=int, default=0, dest="override_D")
    ap.add_argument("--override-N", type=int, default=0, dest="override_N")
    ap.add_argument("--override-max-vocab", type=int, default=0, dest="override_max_vocab")
    ap.add_argument("--seeds", type=int, default=0, help="seed count (0 -> preset)")
    ap.add_argument("--observed", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {"config": vars(args), "cells": [], "pooled": {}}
    all_equal = True

    for corpus in args.corpora:
        p = dict(PRESETS[corpus])  # copy so overrides don't mutate the preset
        if args.override_D:
            p["D"] = args.override_D
        if args.override_N:
            p["N"] = args.override_N
        if args.override_max_vocab:
            p["max_vocab"] = args.override_max_vocab
        if args.seeds:
            p["seeds"] = list(range(args.seeds))
        if args.observed:
            p["observed"] = args.observed
        # pooled accumulators per observed level
        pool = {obs: dict(A_true=0, A_der=0, B_true=0, B_der=0, rand=0, n=0)
                for obs in p["observed"]}
        for seed in p["seeds"]:
            for obs in p["observed"]:
                sub, value_cb, tgt, chance, K_true, K_der = build_data(
                    corpus, p["D"], seed, obs, p["W"], args.device,
                    L=p["L"], Vc=p["Vc"], N=p["N"], eps=p["eps"], max_vocab=p["max_vocab"])
                r = run_paths(sub, value_cb, tgt, K_true, K_der,
                              D=p["D"], lr=args.lr, epochs=args.epochs,
                              beta=args.beta, mi=args.mi)
                cell_equal = r["tix_true_equal"] and r["tix_der_equal"]
                all_equal = all_equal and cell_equal
                cell = {"corpus": corpus, "D": p["D"], "seed": seed, "observed": obs,
                        "chance": chance, "n": r["n"],
                        "vocab": int(value_cb.shape[0]),
                        "H_equal": r["H_equal"], "H_allclose": r["H_allclose"],
                        "tix_true_equal": r["tix_true_equal"],
                        "tix_der_equal": r["tix_der_equal"],
                        "B_true_rate": r["B_true_hits"] / r["n"],
                        "B_der_rate": r["B_der_hits"] / r["n"],
                        "A_true_rate": r["A_true_hits"] / r["n"],
                        "rand_rate": r["rand_hits"] / r["n"],
                        "consolidate_meta": r["meta"]}
                results["cells"].append(cell)
                pk = pool[obs]
                pk["A_true"] += r["A_true_hits"]; pk["A_der"] += r["A_der_hits"]
                pk["B_true"] += r["B_true_hits"]; pk["B_der"] += r["B_der_hits"]
                pk["rand"] += r["rand_hits"]; pk["n"] += r["n"]
                print(f"[{corpus} D{p['D']} seed{seed} obs{obs}] "
                      f"tix_eq(true/der)={r['tix_true_equal']}/{r['tix_der_equal']} "
                      f"H_eq={r['H_equal']} B_true={cell['B_true_rate']:.3f} "
                      f"B_der={cell['B_der_rate']:.3f} rand={cell['rand_rate']:.4f} "
                      f"chance={chance:.4f}", file=sys.stderr, flush=True)

        # pooled two-floor read per observed (Path B = integrated)
        for obs, pk in pool.items():
            chance = next(c["chance"] for c in results["cells"]
                          if c["corpus"] == corpus and c["observed"] == obs)
            sd_b = selectivity_delta(true_hits=pk["B_true"], true_n=pk["n"],
                                     shuffled_hits=pk["B_der"], shuffled_n=pk["n"],
                                     chance=chance).as_dict()
            sd_a = selectivity_delta(true_hits=pk["A_true"], true_n=pk["n"],
                                     shuffled_hits=pk["A_der"], shuffled_n=pk["n"],
                                     chance=chance).as_dict()
            tgt_report = REPORT_TARGETS.get((corpus, obs))
            results["pooled"][f"{corpus}_obs{obs}"] = {
                "integrated_B": sd_b, "standalone_A": sd_a,
                "rand_rate": pk["rand"] / pk["n"], "n_pooled": pk["n"],
                "report_role_delta": tgt_report,
                "A_minus_report": (sd_a["delta"] - tgt_report) if tgt_report else None,
                "B_equals_A": abs(sd_b["delta"] - sd_a["delta"]) < 1e-9,
            }

    results["all_basin_indices_bit_identical"] = all_equal
    print(json.dumps(results["pooled"], indent=2))
    print(f"\nALL basin indices bit-identical (A==B) across every cell: {all_equal}")
    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()
