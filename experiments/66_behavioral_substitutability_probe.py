"""experiments/66 — Behavioral substitutability probe (Reframe-B; DRILL-DOWN, not graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-behavioral-substitutability-probe-precommit.md.

The 121-125 arc measured paradigmatic structure as STATIC CODEBOOK-VECTOR COSINE and found a
route-invariant bound (+0.109 global / <=+0.021 local). CONTEXT.md defines the capability as
CONTEXTUAL COMPLETION, not codebook similarity. This probe asks the bound's question BEHAVIORALLY,
on the existing graduated write+L2 heteroassociative memory (055-058, read-only): does cueing the
memory with a paradigmatic token's contexts (king-contexts, the token masked) admit its partner
(queen) as a low-energy / high-rank COMPLETION, above frequency-matched random tokens and above
non-paradigmatic rand-pairs? The admission of b under a-contexts via H reduces to the key-space
overlap of a-contexts with b's training contexts (second-order context similarity read through the
completion dynamic) -- NOT subject to the subdominant-modes bound (a property of H's geometry, not
of a grown codebook vector). PURE DIAGNOSTIC MEASUREMENT: nothing branches/gates/writes on the reads
(anti-homunculus-exempt); the rank/recall variant is the "measured by recall, not cosine" co-headline.

Reuses exp61/63 + phase2/phase3/phase4 verbatim. Floor (055-058) read UNCHANGED.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from energy_memory.phase2.corpus import (  # noqa: E402
    build_vocabulary, encode_texts, make_windows,
)
from energy_memory.phase2.encoding import build_position_vectors, mask_positions  # noqa: E402
from energy_memory.phase3.basin_readout import top_index_hits  # noqa: E402
from energy_memory.phase4.decorrelator import CueDecorrelator  # noqa: E402
from energy_memory.phase4.hetero_write import (  # noqa: E402
    HeteroConsolidationBuffer, batched_hopfield_topindex, heteroassociative_write,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


# --------------------------------------------------------------------------- #
# Cue encoding: observed = ALL W-1 context slots, center masked (the richest cue;
# mirrors experiments/56 encode_cue "true" mode with observed = full context).
# --------------------------------------------------------------------------- #
def encode_cue_true(sub, positions, codebook, w, op, mpos, W, mask_id):
    keep = set(op)
    toks = [w[p] if p in keep else mask_id for p in range(W)]
    toks[mpos] = mask_id
    return sub.bundle([sub.bind(positions[p], codebook[toks[p]]) for p in range(W)])


def boot_mean_ci(x, n_boot=4000, seed=7):
    """Bootstrap 95% CI of mean(x)."""
    if x.numel() == 0:
        return (float("nan"),) * 3
    g = torch.Generator().manual_seed(seed)
    means = []
    n = x.numel()
    for _ in range(n_boot):
        means.append(float(x[torch.randint(0, n, (n,), generator=g)].mean()))
    means.sort()
    return (float(x.mean()), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--background-n", type=int, default=20000, dest="background_n")
    ap.add_argument("--max-ctx-per-token", type=int, default=100, dest="max_ctx_per_token")
    ap.add_argument("--n-freq", type=int, default=10, dest="n_freq")
    ap.add_argument("--n-randctx", type=int, default=200, dest="n_randctx")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--mi", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device
    g_sample = torch.Generator().manual_seed(args.seed * 911 + 3)

    # ---- corpus / vocab / windows ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    ids = encode_texts(splits["train"], vocab)
    windows = make_windows(ids, args.W)
    mpos = mask_positions(args.W, 1, "center")[0]
    op = [p for p in range(args.W) if p != mpos]            # observed = all context slots
    mask_id = vocab.mask_id

    # ---- pairs (frozen, from exp63.select_pairs) + cooccurrence for the selector ----
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)

    # ---- decodable value codebook + index remap + center frequency ----
    decode_ids = [i for i, t in enumerate(vocab.id_to_token)
                  if t not in {vocab.unk_token, vocab.mask_token}]
    remap = {tid: i for i, tid in enumerate(decode_ids)}    # token_id -> value_cb row
    n_decode = len(decode_ids)
    chance = 1.0 / n_decode
    centers_all = Counter(w[mpos] for w in windows if w[mpos] not in special)

    # ---- tracked tokens (must be decodable to be admissible completions) ----
    pair_lists = {"para": para, "rand": rand, "collo": collo}
    tracked = set()
    for pl in pair_lists.values():
        for a, b in pl:
            tracked.add(a); tracked.add(b)
    if kq:
        tracked.add(kq[0][0]); tracked.add(kq[0][1])
    tracked = {t for t in tracked if t in remap}            # decodable only

    # windows centered on each tracked token (capped, seed-fixed sample if over cap)
    ctx_by_tok = {t: [] for t in tracked}
    bg_pool = []
    for w in windows:
        c = w[mpos]
        if c in tracked:
            ctx_by_tok[c].append(w)
        elif c not in special:
            bg_pool.append(w)
    for t in tracked:
        ws = ctx_by_tok[t]
        if len(ws) > args.max_ctx_per_token:
            idx = torch.randperm(len(ws), generator=g_sample)[:args.max_ctx_per_token].tolist()
            ctx_by_tok[t] = [ws[i] for i in idx]
    # background sample (non-tracked centers) for a realistic full-corpus memory
    if len(bg_pool) > args.background_n:
        idx = torch.randperm(len(bg_pool), generator=g_sample)[:args.background_n].tolist()
        bg = [bg_pool[i] for i in idx]
    else:
        bg = bg_pool

    print(f"V={V} n_decode={n_decode} windows={len(windows)} tracked={len(tracked)} "
          f"para={len(para)} rand={len(rand)} collo={len(collo)} bg={len(bg)} src={src}",
          file=sys.stderr, flush=True)

    # ---- substrate + codebook + positions ----
    sub = TorchFHRR(dim=args.D, seed=args.seed, device=dev)
    codebook = sub.random_vectors(V)
    value_cb = codebook[torch.tensor(decode_ids, device=dev)]
    positions = build_position_vectors(sub, args.W)

    def cues_for(ws):
        return torch.stack([encode_cue_true(sub, positions, codebook, w, op, mpos, args.W, mask_id)
                            for w in ws]) if ws else torch.zeros((0, args.D), dtype=codebook.dtype, device=dev)

    # ---- build the graduated write+L2 memory over the FROZEN buffer ----
    train_ws = [w for t in tracked for w in ctx_by_tok[t]] + bg
    K_train = cues_for(train_ws)
    v_train = torch.tensor([remap[w[mpos]] for w in train_ws], device=dev)
    dec = CueDecorrelator(args.D, renorm="l2").fit(K_train)
    Kd = dec.apply(K_train)
    buf = HeteroConsolidationBuffer(args.D, dev)
    for i in range(Kd.shape[0]):
        buf.add(Kd[i], int(v_train[i]))
    buf.freeze()
    H = heteroassociative_write(buf, value_cb, lr=args.lr, epochs=args.epochs)

    # ---- C0 calibration: memory completes its own contexts (write_l2 vs store-as-is) ----
    # On a capped random subsample of train cues (the store-as-is scene-MHN is O(n^2); C0 is a
    # sanity check, not the headline). Subsample is seed-fixed.
    c0_sample = min(len(train_ws), 2000)
    c0_idx = torch.randperm(len(train_ws), generator=g_sample)[:c0_sample]
    K_c0 = K_train[c0_idx]
    v_c0 = v_train[c0_idx]
    c0_ws = [train_ws[i] for i in c0_idx.tolist()]
    # write_l2 true recall on the subsample:
    ti_w, _, _ = batched_hopfield_topindex(sub, value_cb, (dec.apply(K_c0) @ H.transpose(0, 1)) / args.D,
                                           beta=args.beta, max_iter=args.mi)
    wl2_true = top_index_hits(ti_w, v_c0) / c0_sample
    # store-as-is: scene-MHN over the subsample's full window encodings, unbind center, cleanup
    from energy_memory.phase2.encoding import encode_window
    full_enc = torch.stack([encode_window(sub, positions, codebook, w) for w in c0_ws])
    st = sub.normalize(K_c0)
    for _ in range(args.mi):
        sc = (st @ full_enc.conj().T).real / args.D
        st = sub.normalize(torch.softmax(args.beta * sc, dim=1).to(full_enc.dtype) @ full_enc)
    slot = sub.normalize(sub.unbind(st, positions[mpos]))
    ti_s, _, _ = batched_hopfield_topindex(sub, value_cb, slot, beta=args.beta, max_iter=args.mi)
    sa_true = top_index_hits(ti_s, v_c0) / c0_sample
    c0_valid = bool(wl2_true >= 5.0 * chance and wl2_true >= sa_true)
    print(f"[C0] write_l2 true={wl2_true:.4f} store_as_is true={sa_true:.4f} chance={chance:.5f} "
          f"5x={5*chance:.4f} VALID={c0_valid}", file=sys.stderr, flush=True)

    # ---- precompute completion-score matrices per tracked token ----
    # scores_by_tok[t] = [n_ctx, n_decode] cosine of the settled completion to every value token
    scores_by_tok = {}
    for t in tracked:
        K = cues_for(ctx_by_tok[t])
        recalled = (dec.apply(K) @ H.transpose(0, 1)) / args.D
        r = sub.normalize(recalled)
        scores_by_tok[t] = (r @ value_cb.conj().T).real / args.D       # [n_ctx, n_decode]

    # random-context score matrix (for B3 context-specificity): a pooled background cue set
    rc_ws = bg[:args.n_randctx] if len(bg) >= args.n_randctx else bg
    K_rc = cues_for(rc_ws)
    rc_scores = (sub.normalize((dec.apply(K_rc) @ H.transpose(0, 1)) / args.D)
                 @ value_cb.conj().T).real / args.D if K_rc.shape[0] else None

    # ---- frequency-matched random draw (by log-center-frequency decile) ----
    logf = {t: math.log10(centers_all.get(t, 0) + 1.0) for t in decode_ids}
    fvals = sorted(logf.values())
    def decile(x):
        return min(9, int(10 * (sum(v <= x for v in fvals) - 1) / max(1, len(fvals))))
    by_dec = {d: [] for d in range(10)}
    for tid in decode_ids:
        by_dec[decile(logf[tid])].append(tid)
    def freqmatch(b_tid, exclude, gen):
        cands = [t for t in by_dec[decile(logf[b_tid])] if t not in exclude]
        if len(cands) < args.n_freq:
            cands = [t for t in decode_ids if t not in exclude]
        idx = torch.randperm(len(cands), generator=gen)[:args.n_freq].tolist()
        return [remap[cands[i]] for i in idx]

    # ---- per-pair admission reads ----
    def admit_one(a_tid, b_tid, gen):
        """Admission of partner b under a's contexts: (score-diff, rank-diff)."""
        S = scores_by_tok[a_tid]                       # [n_ctx, n_decode]
        vb = remap[b_tid]
        fm = freqmatch(b_tid, {a_tid, b_tid}, gen)
        score_b = float(S[:, vb].mean())
        score_r = float(S[:, fm].mean())
        d_score = score_b - score_r
        pct = float((S < S[:, vb:vb + 1]).float().mean())   # percentile rank of b
        d_rank = pct - 0.5
        # context-specificity: b's admission under RANDOM contexts (same freq-match)
        if rc_scores is not None:
            rc_b = float(rc_scores[:, vb].mean()) - float(rc_scores[:, fm].mean())
        else:
            rc_b = float("nan")
        return d_score, d_rank, rc_b

    _tag_off = {"para": 0, "rand": 1, "collo": 2, "para_shuffled": 3}   # deterministic (not hash())

    def run_pairs(pairs, tag):
        ds, dr, b3 = [], [], []
        for pi, (a, b) in enumerate(pairs):
            if a not in scores_by_tok or b not in scores_by_tok:
                continue
            gen = torch.Generator().manual_seed(args.boot_seed * 7919 + pi * 17 + _tag_off.get(tag, 7) * 101)
            s_ab, r_ab, rc_ab = admit_one(a, b, gen)
            s_ba, r_ba, rc_ba = admit_one(b, a, gen)
            ds.append(0.5 * (s_ab + s_ba))
            dr.append(0.5 * (r_ab + r_ba))
            # B3: a-context admission minus random-context admission (symmetrized)
            b3.append(0.5 * (s_ab + s_ba) - 0.5 * (rc_ab + rc_ba))
        return (torch.tensor(ds), torch.tensor(dr), torch.tensor(b3))

    res = {}
    for tag, pairs in pair_lists.items():
        ds, dr, b3 = run_pairs(pairs, tag)
        res[tag] = {"n": int(ds.numel()),
                    "score_mean_ci": boot_mean_ci(ds, args.n_boot, args.boot_seed),
                    "rank_mean_ci": boot_mean_ci(dr, args.n_boot, args.boot_seed),
                    "_ds": ds, "_dr": dr, "_b3": b3}
    # ---- ADVERSARIAL kill-test: within-para LABEL-SHUFFLE (same para tokens, broken pairing).
    # Derange the partners so each a is paired with a DIFFERENT pair's b. If real-para admission
    # ≈ shuffled-para, the signal is para-set hubness, NOT pair-specific substitutability. ----
    para_ok = [(a, b) for (a, b) in para if a in scores_by_tok and b in scores_by_tok]
    shuf_pairs = []
    if len(para_ok) >= 3:
        gd = torch.Generator().manual_seed(args.boot_seed * 13 + 5)
        while True:
            perm = torch.randperm(len(para_ok), generator=gd).tolist()
            if all(i != p for i, p in enumerate(perm)):
                break
        shuf_pairs = [(para_ok[i][0], para_ok[perm[i]][1]) for i in range(len(para_ok))]
    sds, srs, _b3s = run_pairs(shuf_pairs, "para_shuffled")
    res["para_shuffled"] = {"n": int(sds.numel()),
                            "score_mean_ci": boot_mean_ci(sds, args.n_boot, args.boot_seed),
                            "rank_mean_ci": boot_mean_ci(srs, args.n_boot, args.boot_seed),
                            "_ds": sds, "_dr": srs}
    KILL_score = exp62._boot_diff(res["para"]["_ds"], sds, n_boot=args.n_boot, seed=args.boot_seed)
    KILL_rank = exp62._boot_diff(res["para"]["_dr"], srs, n_boot=args.n_boot, seed=args.boot_seed)

    kq_admit = None
    if kq:
        a, b = kq[0]
        if a in scores_by_tok and b in scores_by_tok:
            gen = torch.Generator().manual_seed(999)
            s_ab, r_ab, _ = admit_one(a, b, gen)
            s_ba, r_ba, _ = admit_one(b, a, gen)
            kq_admit = {"score": 0.5 * (s_ab + s_ba), "rank": 0.5 * (r_ab + r_ba)}

    # ---- headline gates ----
    pds, prs = res["para"]["_ds"], res["para"]["_dr"]
    rds, rrs = res["rand"]["_ds"], res["rand"]["_dr"]
    H_score = exp62._boot_diff(pds, rds, n_boot=args.n_boot, seed=args.boot_seed)   # para - rand (score)
    H_rank = exp62._boot_diff(prs, rrs, n_boot=args.n_boot, seed=args.boot_seed)    # para - rand (rank)
    B3 = boot_mean_ci(res["para"]["_b3"], args.n_boot, args.boot_seed)
    b1 = bool(res["para"]["score_mean_ci"][1] > 0 and res["para"]["rank_mean_ci"][1] > 0)
    b2 = bool(H_score[1] > 0 and H_rank[1] > 0)
    b3_pass = bool(B3[1] > 0)
    kill_pass = bool(KILL_score[1] > 0 and KILL_rank[1] > 0)   # adversarial: para > para_shuffled
    PASS = bool(c0_valid and b1 and b2 and b3_pass and kill_pass)
    verdict = ("INVALID — memory failed C0 calibration (broken at this scale); adjust D/background/corpus."
               if not c0_valid else
               ("PASS — the graduated memory BEHAVIORALLY admits paradigmatic substitutes above "
                "non-paradigmatic rand-pairs and above random contexts → the 121-125 codebook-cosine "
                "bound is a METRIC artifact, not a capability ceiling; reframe the paradigmatic target "
                "to completion-admission (surface to user)."
                if PASS else
                "NULL — no behavioral substitutability beyond rand-pairs (para ≈ rand) → the bound is "
                "CAPABILITY-level, codebook-cosine measured the right thing; the escape needs new structure."))

    out = {
        "experiment": "66_behavioral_substitutability_probe (DRILL-DOWN reframe, NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-behavioral-substitutability-probe-precommit.md",
        "config": {"corpus_source": args.corpus_source, "V": V, "n_decode": n_decode,
                   "windows": len(windows), "D": args.D, "W": args.W, "max_vocab": args.max_vocab,
                   "background_n": len(bg), "max_ctx_per_token": args.max_ctx_per_token,
                   "n_freq": args.n_freq, "chance": chance, "n_train": len(train_ws),
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "C0": {"write_l2_true": wl2_true, "store_as_is_true": sa_true, "chance": chance,
               "five_x_chance": 5 * chance, "valid": c0_valid},
        "admission": {k: {"n": res[k]["n"], "score_mean_ci": res[k]["score_mean_ci"],
                          "rank_mean_ci": res[k]["rank_mean_ci"]}
                      for k in list(pair_lists) + ["para_shuffled"]},
        "king_queen": kq_admit,
        "headline_para_minus_rand": {"score": list(H_score), "rank": list(H_rank)},
        "KILL_para_minus_shuffled": {"score": list(KILL_score), "rank": list(KILL_rank)},
        "B3_context_specificity": list(B3),
        "gates": {"C0_valid": c0_valid, "B1_para_positive": b1, "B2_para_gt_rand": b2,
                  "B3_context_specific": b3_pass, "B4_pair_specific_kill": kill_pass, "PASS": PASS},
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    for k in pair_lists:
        sc, rk = res[k]["score_mean_ci"], res[k]["rank_mean_ci"]
        print(f"[{k:5s}] n={res[k]['n']:3d} score Δ={sc[0]:+.4f} CI[{sc[1]:+.4f},{sc[2]:+.4f}] "
              f"rank Δ={rk[0]:+.4f} CI[{rk[1]:+.4f},{rk[2]:+.4f}]", file=sys.stderr, flush=True)
    ss = res["para_shuffled"]["score_mean_ci"]; sr = res["para_shuffled"]["rank_mean_ci"]
    print(f"[para_shuffled] n={res['para_shuffled']['n']:3d} score Δ={ss[0]:+.4f} CI[{ss[1]:+.4f},{ss[2]:+.4f}] "
          f"rank Δ={sr[0]:+.4f} CI[{sr[1]:+.4f},{sr[2]:+.4f}]", file=sys.stderr, flush=True)
    print(f"[HEADLINE para−rand] score={H_score[0]:+.4f} CI[{H_score[1]:+.4f},{H_score[2]:+.4f}] "
          f"rank={H_rank[0]:+.4f} CI[{H_rank[1]:+.4f},{H_rank[2]:+.4f}]", file=sys.stderr, flush=True)
    print(f"[KILL para−shuffled] score={KILL_score[0]:+.4f} CI[{KILL_score[1]:+.4f},{KILL_score[2]:+.4f}] "
          f"rank={KILL_rank[0]:+.4f} CI[{KILL_rank[1]:+.4f},{KILL_rank[2]:+.4f}]  pair_specific={kill_pass}",
          file=sys.stderr, flush=True)
    print(f"[B3 ctx-specificity] Δ={B3[0]:+.4f} CI[{B3[1]:+.4f},{B3[2]:+.4f}]", file=sys.stderr, flush=True)
    if kq_admit:
        print(f"[king/queen] score Δ={kq_admit['score']:+.4f} rank Δ={kq_admit['rank']:+.4f}",
              file=sys.stderr, flush=True)
    print(f"[VERDICT] {out['VERDICT'][:90]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
