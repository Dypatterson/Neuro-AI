"""experiments/79 — Bet B, Stage 1: streamed predict-the-future writer (SGNS) vs the local-writer floor.

CHARTER: CONTEXT-B.md (Bet B — lifted rules: backprop/global-as-mechanism ALLOWED; the one fenced
shortcut is a one-shot closed-form SVD/eig, kept as a *reference* only). Discipline KEPT in full.

THE ONE QUESTION (Stage-1 make-or-break, CONTEXT-B §5):
Does a STREAMED, gradient-trained predict-the-context writer (skip-gram negative sampling, SGNS) reach
PAIR-SPECIFIC paradigmatic king~queen structure that the nulled local-writer family (123/124/125/129)
could not — WITHOUT ever materializing the co-occurrence operator and WITHOUT a closed-form SVD?

THE ANTI-REDUNDANCY GUARD (the load-bearing control): SGNS is known to implicitly factorize shifted
PMI (Levy & Goldberg 2014). So the "you-are-just-SVD/NMF" check is mandatory: the closed-form SVD-of-
SPPMI REFERENCE (+0.109 anchor) and global NMF are read with the IDENTICAL metric, side by side. The
TEST writer earns "non-redundant" only by a measured DELTA — better pair-specific B-KILL, OR reaching
the structure from a streamed loss the closed-form route needed the full operator to reach. It NEVER
forms the operator and NEVER calls torch.svd (that is what keeps it a *mechanism*, not a re-factoring).

ARMS (all read with exp68.read_specificity on l2-row embeddings, the SAME para-vs-random + B-KILL metric):
  - SGNS            : the streamed predict-the-context writer (TEST). in-embeddings, operator never formed.
  - SGNS_nolearn    : random-init, same architecture, no training (ANTI-INFLATION floor; g4).
  - grow_G floor    : the nulled linear local writer on build_S (FLOOR; g2 anchor) — exp65.faithful_read.
  - SVD_ref         : closed-form truncated SVD of SPPMI at matched dim (REFERENCE/ceiling — the
                      "you-are-just-SVD" control; allowed to use SVD because it is the reference).
  - NMF_ref         : global NMF slots on build_S (REFERENCE — Oracle-E family).

GATES (CONTEXT-B §5; n=10; g3 DROPPED — dead at n=40, the SVD anchor fails it too):
  g1: spec CI-lo > +0.04   g2: spec_mean − grow_G floor ≥ +0.02   g4: spec_mean − SGNS_nolearn ≥ +0.02
  B-KILL: within-para label-shuffle CI-lo > 0 in ≥8/10 seeds.   PASS = g1 ∧ g2 ∧ g4 ∧ B-KILL≥8/10.
Calibration: raw-SPPMI-SVD anchor must reproduce +0.1092 / king-queen 0.222 or the run is INVALID.
Planted-corpus smoke (synthetic_planted) is mandatory before the WikiText run.

Reuses exp61/62/63/65/68 verbatim. Floor (055-058) untouched. Substrate-free read (real embeddings).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import statistics as st

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
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# The TEST writer: streamed skip-gram negative sampling (SGNS).
# Operator is NEVER materialized; NO closed-form SVD. Pure streamed gradient descent.
# =====================================================================

def build_skipgram_pairs(train_ids, V, special, *, radius, subsample_t):
    """Word2vec-faithful: subsample frequent tokens + drop specials, then emit (center, context)
    pairs within +/-radius over the FILTERED stream. Seed-independent (shared across seeds)."""
    ids = torch.tensor(train_ids, dtype=torch.long)
    counts = torch.bincount(ids, minlength=V).double()
    for s in special:
        counts[s] = 0.0
    total = counts.sum().clamp(min=1.0)
    freq = counts / total
    # word2vec subsample keep-prob: P_keep = sqrt(t/f) + t/f, clamped to 1
    ratio = (subsample_t / freq.clamp(min=1e-12))
    keep = (ratio.sqrt() + ratio).clamp(max=1.0)
    keep[counts == 0] = 0.0
    # filter the stream deterministically by a fixed keep-threshold draw (seed-independent corpus)
    g = torch.Generator().manual_seed(20240605)
    draws = torch.rand(len(ids), generator=g)
    is_special = torch.zeros(len(ids), dtype=torch.bool)
    for s in special:
        is_special |= (ids == s)
    kept_mask = (draws < keep[ids]) & (~is_special)
    stream = ids[kept_mask]
    N = len(stream)
    centers, contexts = [], []
    for off in range(1, radius + 1):
        # center[i] with context[i+off] and context[i-off], vectorized
        c = stream[:-off]; ctx = stream[off:]
        centers.append(c); contexts.append(ctx)
        centers.append(ctx); contexts.append(c)
    centers = torch.cat(centers); contexts = torch.cat(contexts)
    # negative-sampling distribution: unigram^0.75 over non-special, in-vocab tokens
    negp = counts.pow(0.75); negp = negp / negp.sum()
    return centers, contexts, negp, N


def train_sgns(centers, contexts, negp, V, d, *, neg, epochs, batch, lr, seed, device, learn=True):
    g = torch.Generator().manual_seed(seed * 2654435761 % (2**31 - 1) + 11)
    Win = ((torch.rand((V, d), generator=g) - 0.5) / d).to(torch.float32)
    if not learn:
        return l2rows(Win.to(torch.float64))                       # random-init architecture control
    Wout = torch.zeros((V, d), dtype=torch.float32)
    Win = Win.to(device).requires_grad_(True)
    Wout = Wout.to(device).requires_grad_(True)
    negp_d = negp.to(torch.float32).to(device)
    centers = centers.to(device); contexts = contexts.to(device)
    opt = torch.optim.Adam([Win, Wout], lr=lr)
    P = centers.shape[0]
    gd = torch.Generator(device="cpu").manual_seed(seed * 100003 + 5)
    for ep in range(epochs):
        perm = torch.randperm(P, generator=gd).to(device)
        for s in range(0, P, batch):
            idx = perm[s:s + batch]
            c = centers[idx]; pos = contexts[idx]
            nb = c.shape[0]
            negs = torch.multinomial(negp_d, nb * neg, replacement=True).view(nb, neg)
            vc = Win[c]                                            # (b,d)
            vpos = Wout[pos]                                       # (b,d)
            vneg = Wout[negs]                                      # (b,neg,d)
            pos_score = (vc * vpos).sum(-1)                        # (b,)
            neg_score = torch.bmm(vneg, vc.unsqueeze(-1)).squeeze(-1)  # (b,neg)
            loss = (-torch.nn.functional.logsigmoid(pos_score)
                    - torch.nn.functional.logsigmoid(-neg_score).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return l2rows(Win.detach().to("cpu").to(torch.float64))


def svd_embed(sppmi, d):
    """REFERENCE only: truncated SVD-of-SPPMI embedding U·sqrt(S) (Levy-Goldberg / the anchor embedding).
    Closed-form SVD is allowed HERE because this is the reference-to-beat-or-differ-from, not the writer."""
    U, S, _ = torch.linalg.svd(sppmi.to(torch.float64), full_matrices=False)
    return l2rows(U[:, :d] * S[:d].clamp(min=0).sqrt())


def real_d_eff(Y):
    Yc = Y - Y.mean(0, keepdim=True)
    G = Yc @ Yc.T
    tr = torch.diagonal(G).sum()
    return float((tr * tr) / (G * G).sum().clamp(min=EPS))


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=4096)                 # FHRR dim for the grow_G floor only
    ap.add_argument("--d-embed", type=int, default=300, dest="d_embed")  # SGNS / SVD-ref dim
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--subsample-t", type=float, default=1e-3, dest="subsample_t")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="")
    # grow_G floor knobs (reuse exp65.faithful_read defaults)
    ap.add_argument("--eta-grid", default="0,0.05,0.1,0.2", dest="eta_grid")
    ap.add_argument("--grow-epochs", type=int, default=20, dest="grow_epochs")
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-decay", type=float, default=0.9, dest="alpha_decay")
    args = ap.parse_args()
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x != ""]
    planted = args.corpus_source == "synthetic_planted"

    # ---- corpus / vocab / windows / SPPMI / pairs / anchor (seed-independent) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    k_sym = k_sym or 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    build_S = exp61.build_S(sppmi)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners([(a, b) for (a, b) in para], args.boot_seed)

    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool(planted or ((0.082 - hw) <= a_spec <= (0.137 + hw)
                                and abs(anchor["king_queen_cos"] - 0.222) <= 0.04))
    d = args.d_embed
    print(f"V={V} windows={len(windows)} n_para={len(para)} src={src} "
          f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    # ---- streamed skip-gram pairs (seed-independent corpus) ----
    centers, contexts, negp, n_stream = build_skipgram_pairs(
        train_ids, V, special, radius=args.radius, subsample_t=args.subsample_t)
    print(f"[sgns] filtered_stream={n_stream} pairs={centers.shape[0]} d={d} epochs={args.epochs}",
          file=sys.stderr, flush=True)

    def rd(Y, seed):
        return exp68.read_specificity(Y, para, rand, collo, shuf, args.n_boot, seed)

    # ---- REFERENCES (seed-independent: SVD closed-form; NMF is per-seed below) ----
    svd_ref_read = rd(svd_embed(sppmi, d), args.boot_seed)
    svd_ref_deff = real_d_eff(svd_embed(sppmi, d))

    per_seed = {"SGNS": [], "SGNS_nolearn": [], "grow_G_floor": [], "NMF_ref": [], "deff_sgns": []}
    for seed in range(args.seeds):
        Y = train_sgns(centers, contexts, negp, V, d, neg=args.neg, epochs=args.epochs,
                       batch=args.batch, lr=args.lr, seed=seed, device=args.device, learn=True)
        Y0 = train_sgns(centers, contexts, negp, V, d, neg=args.neg, epochs=args.epochs,
                        batch=args.batch, lr=args.lr, seed=seed, device=args.device, learn=False)
        per_seed["SGNS"].append(rd(Y, seed))
        per_seed["SGNS_nolearn"].append(rd(Y0, seed))
        per_seed["deff_sgns"].append(real_d_eff(Y))
        # grow_G linear floor on build_S (the nulled local writer) — CPU (grow_G internals are CPU-bound)
        sub = TorchFHRR(dim=args.D, seed=seed, device="cpu", alpha_anti=1.0)
        G0 = sub.random_vectors(V); deff0 = exp61.d_eff(sub, G0)
        cooc_x = (torch.log1p(exp61.cooc_counts_for_pairs(C, torch.tensor(para, dtype=torch.long)))
                  if para else None)
        fr = exp65.faithful_read(sub, G0, deff0, build_S, para, rand, cooc_x, eta_grid,
                                 args.grow_epochs, args.alpha0, args.alpha_decay, args.n_boot, args.boot_seed)
        per_seed["grow_G_floor"].append(fr["best"]["spec"] if fr["best"] else float("nan"))
        # NMF reference on build_S at matched-ish k (Oracle-E family); use k=min(32,d)
        per_seed["NMF_ref"].append(rd(exp65.nmf_slots(build_S, min(32, d), seed=seed), seed))
        sg = per_seed["SGNS"][-1]
        print(f"[seed {seed}] SGNS spec={sg['spec_para_minus_rand'][0]:+.4f} "
              f"(CIlo {sg['spec_para_minus_rand'][1]:+.4f}) kill_lo={sg['kill_para_minus_shuf'][1]:+.4f} "
              f"para_cos={sg['para_cos']:.3f} rand_cos={sg['rand_cos']:.3f} "
              f"floor={per_seed['grow_G_floor'][-1]:+.4f}", file=sys.stderr, flush=True)

    # ---- aggregate + gate ----
    def agg(rows):
        return {"spec_mean": st.mean([c["spec_para_minus_rand"][0] for c in rows]),
                "spec_ci_lo": st.mean([c["spec_para_minus_rand"][1] for c in rows]),
                "para_cos": st.mean([c["para_cos"] for c in rows]),
                "rand_cos": st.mean([c["rand_cos"] for c in rows]),
                "kill_lo_per_seed": [round(c["kill_para_minus_shuf"][1], 4) for c in rows],
                "bkill_seeds": sum(1 for c in rows if c["kill_para_minus_shuf"][1] == c["kill_para_minus_shuf"][1]
                                   and c["kill_para_minus_shuf"][1] > 0)}
    sgns = agg(per_seed["SGNS"]); nolearn = agg(per_seed["SGNS_nolearn"]); nmf = agg(per_seed["NMF_ref"])
    floor = st.mean([x for x in per_seed["grow_G_floor"] if x == x])

    g1 = bool(sgns["spec_ci_lo"] > 0.04)
    g2 = bool(sgns["spec_mean"] - floor >= 0.02)
    g4 = bool(sgns["spec_mean"] - nolearn["spec_mean"] >= 0.02)
    bkill_ok = bool(sgns["bkill_seeds"] >= 8)
    PASS = bool(g1 and g2 and g4 and bkill_ok and calib_ok)

    # anti-redundancy / "you-are-just-SVD" delta
    redundancy = {
        "svd_ref_spec": svd_ref_read["spec_para_minus_rand"][0],
        "svd_ref_kill_lo": svd_ref_read["kill_para_minus_shuf"][1],
        "sgns_minus_svd_spec": sgns["spec_mean"] - svd_ref_read["spec_para_minus_rand"][0],
        "sgns_bkill_seeds": sgns["bkill_seeds"],
        "note": "SGNS earns NON-REDUNDANT only by a delta: better B-KILL than the closed-form ref, OR "
                "reaching the structure from the streamed loss without forming the operator (it never did).",
    }

    verdict = ("INVALID — calibration anchor missed +0.109/0.222." if not calib_ok else
               ("PASS — a STREAMED gradient writer (no operator, no SVD) reaches pair-specific paradigmatic "
                "structure the nulled local-writer family could not (g1∧g2∧g4∧B-KILL≥8/10). The wall was the "
                "OLD RULES, not the data. Stage-1 graduates → Stage-2 (two-timescale consolidation) is licensed." if PASS else
                "NULL/REDUNDANT — SGNS did not clear the gate as a distinct mechanism (see g-flags + the "
                "you-are-just-SVD delta). If it merely matched the SVD reference, bank as redundant."))

    out = {"experiment": "79_betb_streamed_sgns (Bet B, Stage 1 — streamed predict-the-context writer)",
           "charter": "CONTEXT-B.md §5", "src": src,
           "config": {"V": V, "d_embed": d, "radius": args.radius, "neg": args.neg, "epochs": args.epochs,
                      "lr": args.lr, "seeds": args.seeds, "n_stream": n_stream, "n_pairs": int(centers.shape[0]),
                      "n_para": len(para), "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
           "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
           "floor_grow_G": floor,
           "SGNS": sgns, "SGNS_nolearn": nolearn, "NMF_ref": nmf,
           "SVD_ref": {"spec": svd_ref_read["spec_para_minus_rand"][0],
                       "spec_ci_lo": svd_ref_read["spec_para_minus_rand"][1],
                       "kill_lo": svd_ref_read["kill_para_minus_shuf"][1], "d_eff": svd_ref_deff},
           "deff_sgns_mean": st.mean(per_seed["deff_sgns"]),
           "gates": {"g1_spec_cilo_gt_0.04": g1, "g2_beats_floor_by_0.02": g2,
                     "g4_beats_nolearn_by_0.02": g4, "bkill_seeds": f"{sgns['bkill_seeds']}/{args.seeds}",
                     "bkill_ge_8": bkill_ok},
           "redundancy_check": redundancy,
           "PASS": PASS, "verdict": verdict}
    print(json.dumps(out, indent=2))
    print(f"\n=== {verdict}", file=sys.stderr, flush=True)
    print(f"=== SGNS spec={sgns['spec_mean']:+.4f} (CIlo {sgns['spec_ci_lo']:+.4f}) B-KILL={sgns['bkill_seeds']}/{args.seeds} "
          f"| floor={floor:+.4f} nolearn={nolearn['spec_mean']:+.4f} SVD_ref={svd_ref_read['spec_para_minus_rand'][0]:+.4f} "
          f"NMF_ref={nmf['spec_mean']:+.4f} | g1={g1} g2={g2} g4={g4} PASS={PASS}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
