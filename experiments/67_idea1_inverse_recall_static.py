"""experiments/67 — Idea-1 collapse-check: inverse-recall STATIC semantics (DRILL-DOWN, not graduation).

Confirms the grounded reduction (grill-verified this session): Idea 1's "semantic codebook =
inverse image of the recall map", semantic_i = normalize(H_conjT @ v_i), is Report 126 re-expressed
as a STATIC cosine. Algebra (hetero_write.py:167-174): the delta-rule write over DECORRELATED keys
makes H_conjT @ v_a ~= the whitened context-centroid of token a, so cos(H*v_a, H*v_b) IS the
"do a and b share contexts" quantity Report 126 read through completion (and nulled, PASS 0/5,
label-shuffle = para-set hubness). This re-runs that quantity as a static pairwise cosine on the
SAME n=40 SimLex>=5 non-cooc pairs, on the SAME valid-regime graduated memory H, with the MANDATORY
within-para LABEL-SHUFFLE (the control the user's ladder omitted; without it a static 40-pair cosine
reproduces the 126 false positive). 5 seeds. PURE DIAGNOSTIC measurement (anti-homunculus-exempt).

Predicted: lands on the 126 NULL — para>rand (B1/B2) may appear but para ~= para_shuffled (B4 fails),
king/queen null. If it DIVERGES from 126 that is itself the finding. Reuses exp61/63 + phase4 verbatim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, REPO / "experiments" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp62 = _load("exp62", "62_exp61_oracles.py")
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp66 = _load("exp66", "66_behavioral_substitutability_probe.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.phase2.encoding import build_position_vectors, mask_positions  # noqa: E402
from energy_memory.phase3.basin_readout import top_index_hits  # noqa: E402
from energy_memory.phase4.decorrelator import CueDecorrelator  # noqa: E402
from energy_memory.phase4.hetero_write import (  # noqa: E402
    HeteroConsolidationBuffer, batched_hopfield_topindex, heteroassociative_write,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def pair_cos(sem, idx_of, pairs):
    """Real cosine of the static inverse-recall vectors for each (a,b); sem rows L2-normalized."""
    out = []
    for a, b in pairs:
        if a not in idx_of or b not in idx_of:
            continue
        sa, sb = sem[idx_of[a]], sem[idx_of[b]]
        c = (sa * sb.conj()).sum().real / (sa.norm() * sb.norm()).clamp(min=EPS)
        out.append(float(c))
    return torch.tensor(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--background-n", type=int, default=2000, dest="background_n")
    ap.add_argument("--max-ctx-per-token", type=int, default=20, dest="max_ctx_per_token")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device

    # ---- corpus / pairs / tracked / background (seed-independent; built ONCE) ----
    splits = exp61.load_corpus(args.corpus_source, args, 0)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    ids = encode_texts(splits["train"], vocab)
    windows = make_windows(ids, args.W)
    mpos = mask_positions(args.W, 1, "center")[0]
    op = [p for p in range(args.W) if p != mpos]
    mask_id = vocab.mask_id
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    decode_ids = [i for i, t in enumerate(vocab.id_to_token)
                  if t not in {vocab.unk_token, vocab.mask_token}]
    remap = {tid: i for i, tid in enumerate(decode_ids)}
    n_decode = len(decode_ids)
    chance = 1.0 / n_decode
    pair_lists = {"para": para, "rand": rand, "collo": collo}
    tracked = set()
    for pl in pair_lists.values():
        for a, b in pl:
            tracked.add(a); tracked.add(b)
    if kq:
        tracked.add(kq[0][0]); tracked.add(kq[0][1])
    tracked = {t for t in tracked if t in remap}

    ctx_by_tok = {t: [] for t in tracked}
    bg_pool = []
    for w in windows:
        c = w[mpos]
        if c in tracked:
            ctx_by_tok[c].append(w)
        elif c not in special:
            bg_pool.append(w)

    # within-para label-shuffle pairs (same para tokens, deranged partners) — seed-fixed
    para_ok = [(a, b) for (a, b) in para if a in tracked and b in tracked]
    gd = torch.Generator().manual_seed(args.boot_seed * 13 + 5)
    while True:
        perm = torch.randperm(len(para_ok), generator=gd).tolist()
        if all(i != p for i, p in enumerate(perm)):
            break
    shuf_pairs = [(para_ok[i][0], para_ok[perm[i]][1]) for i in range(len(para_ok))]
    print(f"V={V} n_decode={n_decode} tracked={len(tracked)} para={len(para)} rand={len(rand)} "
          f"collo={len(collo)} src={src}", file=sys.stderr, flush=True)

    per_seed = []
    for seed in range(args.seeds):
        g_sample = torch.Generator().manual_seed(seed * 911 + 3)
        # cap tracked contexts + background sample (seed-fixed)
        cbt = {}
        for t in tracked:
            ws = ctx_by_tok[t]
            if len(ws) > args.max_ctx_per_token:
                idx = torch.randperm(len(ws), generator=g_sample)[:args.max_ctx_per_token].tolist()
                cbt[t] = [ws[i] for i in idx]
            else:
                cbt[t] = list(ws)
        if len(bg_pool) > args.background_n:
            bidx = torch.randperm(len(bg_pool), generator=g_sample)[:args.background_n].tolist()
            bg = [bg_pool[i] for i in bidx]
        else:
            bg = list(bg_pool)

        sub = TorchFHRR(dim=args.D, seed=seed, device=dev)
        codebook = sub.random_vectors(V)
        value_cb = codebook[torch.tensor(decode_ids, device=dev)]
        positions = build_position_vectors(sub, args.W)

        def cue(w):
            return exp66.encode_cue_true(sub, positions, codebook, w, op, mpos, args.W, mask_id)

        train_ws = [w for t in tracked for w in cbt[t]] + bg
        K_train = torch.stack([cue(w) for w in train_ws])
        v_train = torch.tensor([remap[w[mpos]] for w in train_ws], device=dev)
        dec = CueDecorrelator(args.D, renorm="l2").fit(K_train)
        Kd = dec.apply(K_train)
        buf = HeteroConsolidationBuffer(args.D, dev)
        for i in range(Kd.shape[0]):
            buf.add(Kd[i], int(v_train[i]))
        buf.freeze()
        H = heteroassociative_write(buf, value_cb, lr=args.lr, epochs=args.epochs)

        # C0 (confirm same valid memory): write_l2 true recall on a 2000 subsample
        cs = min(len(train_ws), 2000)
        ci = torch.randperm(len(train_ws), generator=g_sample)[:cs]
        ti_w, _, _ = batched_hopfield_topindex(
            sub, value_cb, (dec.apply(K_train[ci]) @ H.transpose(0, 1)) / args.D,
            beta=args.beta, max_iter=args.mi)
        wl2 = top_index_hits(ti_w, v_train[ci]) / cs

        # ---- Idea 1: static inverse-recall semantics  semantic_i = normalize(H_conjT @ v_i) ----
        Hh = H.conj().transpose(0, 1)
        tracked_list = sorted(tracked)
        idx_of = {t: i for i, t in enumerate(tracked_list)}
        V_tracked = value_cb[torch.tensor([remap[t] for t in tracked_list], device=dev)]
        sem = (V_tracked @ Hh.transpose(0, 1)) / args.D          # [n_tracked, D] = H* @ v_i per row

        cos = {k: pair_cos(sem, idx_of, pl) for k, pl in pair_lists.items()}
        cos["para_shuffled"] = pair_cos(sem, idx_of, shuf_pairs)
        kq_cos = float(pair_cos(sem, idx_of, kq)[0]) if kq else float("nan")

        spec = exp62._boot_diff(cos["para"], cos["rand"], n_boot=args.n_boot, seed=args.boot_seed)
        kill = exp62._boot_diff(cos["para"], cos["para_shuffled"], n_boot=args.n_boot, seed=args.boot_seed)
        b1 = exp66.boot_mean_ci(cos["para"] - cos["rand"].mean(), args.n_boot, args.boot_seed)  # para-vs-meanrand
        cell = {
            "seed": seed, "c0_write_l2_true": wl2,
            "mean_cos": {k: float(v.mean()) for k, v in cos.items()},
            "king_queen_cos": kq_cos,
            "spec_para_minus_rand": list(spec),
            "kill_para_minus_shuffled": list(kill),
            "b2_para_gt_rand": bool(spec[1] > 0),
            "b4_pair_specific": bool(kill[1] > 0),
        }
        per_seed.append(cell)
        print(f"[seed {seed}] C0={wl2:.3f}  para={cell['mean_cos']['para']:+.4f} "
              f"shuf={cell['mean_cos']['para_shuffled']:+.4f} rand={cell['mean_cos']['rand']:+.4f} "
              f"collo={cell['mean_cos']['collo']:+.4f} kq={kq_cos:+.4f} | "
              f"spec(para-rand)={spec[0]:+.4f}CI[{spec[1]:+.4f},{spec[2]:+.4f}] "
              f"KILL(para-shuf)={kill[0]:+.4f}CI[{kill[1]:+.4f},{kill[2]:+.4f}] B4={cell['b4_pair_specific']}",
              file=sys.stderr, flush=True)

    # ---- aggregate ----
    import statistics as st
    def col(f): return [c[f] for c in per_seed]
    b2 = sum(c["b2_para_gt_rand"] for c in per_seed)
    b4 = sum(c["b4_pair_specific"] for c in per_seed)
    agg = {
        "n_seeds": len(per_seed),
        "c0_write_l2_true_mean": st.mean(col("c0_write_l2_true")),
        "mean_cos": {k: st.mean([c["mean_cos"][k] for c in per_seed])
                     for k in ["para", "para_shuffled", "rand", "collo"]},
        "king_queen_cos_mean": st.mean(col("king_queen_cos")),
        "king_queen_cos_range": [min(col("king_queen_cos")), max(col("king_queen_cos"))],
        "spec_para_minus_rand_mean": st.mean([c["spec_para_minus_rand"][0] for c in per_seed]),
        "kill_para_minus_shuffled_mean": st.mean([c["kill_para_minus_shuffled"][0] for c in per_seed]),
        "B2_para_gt_rand_frac": f"{b2}/{len(per_seed)}",
        "B4_pair_specific_frac": f"{b4}/{len(per_seed)}",
        "LANDS_ON_126_NULL": bool(b4 <= 1),
    }
    verdict = ("NULL — lands on the Report 126 NULL: Idea-1 static inverse-recall reproduces the "
               "behavioral probe's para-set-hubness, NOT pair-specific substitutability "
               f"(B4 pair-specific {b4}/{len(per_seed)} seeds). Idea-1/2/3/7 cluster CLOSED."
               if agg["LANDS_ON_126_NULL"] else
               "DIVERGES from Report 126 — static inverse-recall is pair-specific where the "
               "completion read was not; this is a metric/cleanup discrepancy worth a drill-down.")
    out = {
        "experiment": "67_idea1_inverse_recall_static (DRILL-DOWN collapse-check, NOT graduation)",
        "reduction": "Idea-1 semantic_i = normalize(H_conjT @ v_i) == Report 126 as a static cosine",
        "config": {"D": args.D, "V": V, "n_decode": n_decode, "max_vocab": args.max_vocab,
                   "background_n": args.background_n, "max_ctx_per_token": args.max_ctx_per_token,
                   "n_para": len(para_ok), "seeds": args.seeds, "chance": chance,
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "per_seed": per_seed, "aggregate": agg, "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"[AGG] C0={agg['c0_write_l2_true_mean']:.3f} | para={agg['mean_cos']['para']:+.4f} "
          f"shuf={agg['mean_cos']['para_shuffled']:+.4f} rand={agg['mean_cos']['rand']:+.4f} | "
          f"spec(para-rand)={agg['spec_para_minus_rand_mean']:+.4f} "
          f"KILL(para-shuf)={agg['kill_para_minus_shuffled_mean']:+.4f} | "
          f"B2={agg['B2_para_gt_rand_frac']} B4={agg['B4_pair_specific_frac']} | "
          f"kq={agg['king_queen_cos_mean']:+.4f}", file=sys.stderr, flush=True)
    print(f"[VERDICT] {verdict[:100]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
