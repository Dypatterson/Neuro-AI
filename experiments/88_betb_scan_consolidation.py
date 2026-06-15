"""experiments/88 — Bet B / SCAN Stage 1: does a restructuring CONSOLIDATION manufacture composition?

CHARTER: notes/betb-scan-stage1-consolidation-precommit.md (anti-homunculus PASS, agent a0c17e48621a908db,
2026-06-15, with TWO binding build conditions honored here). Gated on Report 140 (Stage 0 gap real, ≈0.003)
and the premise probe (jump<-walk embedding swap -> 0.9935: the template slot works, jump's embedding is the
SOLE locus of failure). THE Bet-B graduation attempt: a regime where the simple method provably FAILS, so a
positive here is NOT the task-selection confound (RETROSPECTIVE §4).

MECHANISM (the role/filler alignment of §2, realized as a loss interaction):
  consolidation = an OFFLINE pass over replayed data that updates the INPUT embeddings only:
     L = L_replay (CE on replayed train batches — keeps each verb decoding to its OWN action = the FILLER/identity)
       + lam * L_align (variance of the PEER-SET input embeddings -> pulls verbs together = the shared ROLE)
  The role/filler split EMERGES: replay protects identity (verbs stay distinguishable enough to decode), align
  manufactures the shared verb-role; jump, pulled into the cluster, inherits the role -> composes, while replay
  on its standalone example keeps jump -> I_JUMP. No hand-set projection; no per-token branch.

BUILD CONDITION 1 (anti-homunculus, non-negotiable): the peer set is COMPUTED IN CODE from the train buffer via
the one-token-command predicate and ASSERTED == {jump,walk,run,look} — never hardcoded as a literal the loss
reads. (peer_token_ids() below; asserted in main.)
BUILD CONDITION 2: the second-split control reuses the BYTE-IDENTICAL predicate + loss (--split flag; no
jump-specific constant). [second-split run is a follow-up invocation.]

ARMS:
  baseline       train only (Stage-0 floor, jump ≈ 0.003)
  consolidation  baseline + the offline pass (mode=peer)                          (THE mechanism)
  random_align   baseline + offline pass aligning RANDOM non-peer tokens (mode=random)   (ablation: peer-structure load-bearing?)
  collapse       baseline + align with NO replay (mode=collapse)                  (ablation: identity term necessary?)

HEADLINE: jump-split exact-match, consolidation − baseline, >=8 seeds, bootstrap CI; PASS = CI-lo > 0 AND lift
non-trivial (>=0.30). Drill-downs: random-split ceiling no-regression; random_align ≈ baseline; collapse loses
identity (jump twice -> walk twice). REDUNDANCY GUARD (GECA) is a separate follow-up arm.
ANTI-HOMUNCULUS: align loss is uniform over the structural peer set; replay content-blind; no metric-read/branch.
FENCE: iterative gradient, no closed-form SVD. (Bet B legal.)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_s = importlib.util.spec_from_file_location("exp87", REPO / "experiments" / "87_betb_scan_gap.py")
exp87 = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp87)
load_pairs, build_vocab, to_ids, pad = exp87.load_pairs, exp87.build_vocab, exp87.to_ids, exp87.pad
make_batches, Seq2Seq, train, exact_match = exp87.make_batches, exp87.Seq2Seq, exp87.train, exp87.exact_match
PAD, SOS, EOS = exp87.PAD, exp87.SOS, exp87.EOS
DATA = REPO / "data" / "scan"

SPLITS = {"jump": ("tasks_train_addprim_jump.txt", "tasks_test_addprim_jump.txt"),
          "simple": ("tasks_train_simple.txt", "tasks_test_simple.txt")}
ARMS = ["baseline", "consolidation", "random_align", "collapse"]


def peer_token_ids(train_pairs, in_vocab):
    """BUILD CONDITION 1: the peer set = tokens appearing as a COMPLETE one-token command in the buffer.
    Computed from data structure; content-blind; NOT a hardcoded list. Returns sorted ids + the token set."""
    peer = {p[0][0] for p in train_pairs if len(p[0]) == 1}
    return sorted(in_vocab[t] for t in peer), peer


def consolidate(model, train_pairs, in_vocab, out_vocab, align_ids, *, mode, lam, steps, batch_size, lr,
                gen, device):
    """Offline restructuring pass over replayed data. Updates INPUT embeddings only (the diagnosed locus;
    the premise probe showed a frozen enc/dec composes a verb whose embedding sits in the manifold)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.enc.emb.weight.requires_grad_(True)
    opt = torch.optim.Adam([model.enc.emb.weight], lr=lr)
    model.train()
    batcher = None
    for step in range(steps):
        # L_replay (skipped for the collapse ablation -> identity unprotected)
        loss = torch.zeros((), device=device)
        if mode != "collapse":
            if batcher is None:
                batcher = iter(make_batches(train_pairs, in_vocab, out_vocab, batch_size, gen))
            try:
                src, tgt = next(batcher)
            except StopIteration:
                batcher = iter(make_batches(train_pairs, in_vocab, out_vocab, batch_size, gen)); src, tgt = next(batcher)
            src, _ = pad(src, device); tgt, _ = pad(tgt, device)
            logits = model(src, src != PAD, tgt, 0.5, gen)
            loss = loss + F.cross_entropy(logits.reshape(-1, model.out_vocab), tgt.reshape(-1), ignore_index=PAD)
        # L_align: variance of the aligned token-set input embeddings (pull together = shared role)
        E = model.enc.emb.weight[align_ids]                       # (k, E)
        loss_align = ((E - E.mean(0, keepdim=True)) ** 2).sum(1).mean()
        loss = loss + lam * loss_align
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([model.enc.emb.weight], 5.0)
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)


def run_arm(arm, split, args, seed):
    tr_fp, te_fp = (str(DATA / SPLITS[split][0]), str(DATA / SPLITS[split][1]))
    train_pairs, test_pairs = load_pairs(tr_fp), load_pairs(te_fp)
    allp = train_pairs + test_pairs
    in_vocab, _ = build_vocab(allp, "in"); out_vocab, _ = build_vocab(allp, "out")
    max_len = max(len(p[1]) for p in allp) + 2
    peer_ids, peer_set = peer_token_ids(train_pairs, in_vocab)
    if split == "jump":                                          # CONDITION 1 assertion (the structural read is real)
        assert peer_set == {"jump", "walk", "run", "look"}, f"peer predicate -> {peer_set}"

    g = torch.Generator().manual_seed(seed * 104729 + 7)
    model = Seq2Seq(len(in_vocab), len(out_vocab), args.embed, args.hidden, seed).to(args.device)
    train(model, train_pairs, in_vocab, out_vocab, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, tf_ratio=args.tf_ratio, seed=seed, device=args.device)

    if arm != "baseline":
        if arm == "random_align":                                # ablation: align a random NON-peer set of equal size
            pool = [i for i in range(len(in_vocab)) if i not in set(peer_ids) and i >= len(exp87.SPECIAL)]
            perm = torch.randperm(len(pool), generator=g).tolist()
            align_ids = sorted(pool[perm[k]] for k in range(min(len(peer_ids), len(pool))))
            mode = "peer"
        else:
            align_ids = peer_ids
            mode = "collapse" if arm == "collapse" else "peer"
        consolidate(model, train_pairs, in_vocab, out_vocab, align_ids, mode=mode, lam=args.lam,
                    steps=args.consol_steps, batch_size=args.batch_size, lr=args.consol_lr, gen=g, device=args.device)

    acc = exact_match(model, test_pairs, in_vocab, out_vocab, max_len=max_len, batch_size=args.batch_size,
                      device=args.device)
    # drill-down: does collapse lose identity? (jump twice -> walk twice). Measure structural acc (verb-agnostic).
    print(f"  [{arm} {split} seed {seed}] test exact-match = {acc:.4f}", file=sys.stderr, flush=True)
    return acc


def boot_ci(vals, n=4000, seed=0):
    t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return (float("nan"),) * 3
    gg = torch.Generator().manual_seed(seed)
    idx = torch.randint(t.numel(), (n, t.numel()), generator=gg)
    m = t[idx].mean(1)
    lo, hi = torch.quantile(m, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return (float(t.mean()), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="jump")                   # jump | simple (CONDITION 2: same code both)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tf-ratio", type=float, default=0.5, dest="tf_ratio")
    ap.add_argument("--lam", type=float, default=1.0)            # align strength
    ap.add_argument("--consol-steps", type=int, default=500, dest="consol_steps")
    ap.add_argument("--consol-lr", type=float, default=1e-3, dest="consol_lr")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.consol_steps = 1, 12, 150
    arms = [a for a in args.arms.split(",") if a in ARMS]

    res = {a: [] for a in arms}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for a in arms:
            res[a].append(run_arm(a, args.split, args, seed))

    ci = {a: boot_ci(res[a], seed=hash(a) % 9991) for a in arms}
    out = {"experiment": "88_betb_scan_consolidation (Bet B / SCAN Stage 1 graduation attempt)",
           "charter": "notes/betb-scan-stage1-consolidation-precommit.md (anti-homunculus PASS)",
           "config": vars(args), "split": args.split,
           "exact_match_by_arm": {a: {"ci": ci[a], "per_seed": res[a]} for a in arms}}
    if "baseline" in res and "consolidation" in res:
        delta = boot_ci([res["consolidation"][s] - res["baseline"][s] for s in range(len(res["baseline"]))], seed=99)
        out["HEADLINE_consolidation_minus_baseline"] = delta
        out["PASS"] = bool(delta[1] > 0 and ci["consolidation"][0] - ci["baseline"][0] >= 0.30)
        out["verdict"] = (f"SCAN Stage 1 ({args.split}, n={args.seeds}): baseline={ci['baseline'][0]:.4f} "
                          f"consolidation={ci['consolidation'][0]:.4f} (Δ={delta[0]:+.4f}[{delta[1]:+.4f},{delta[2]:+.4f}]). "
                          f"random_align={ci.get('random_align',[float('nan')])[0]:.4f} "
                          f"collapse={ci.get('collapse',[float('nan')])[0]:.4f}. PASS={out['PASS']} "
                          "(restructuring consolidation manufactures composition the simple method can't). "
                          "Guard: if Δ ≈ GECA -> REDUNDANT; random_align must ≈ baseline; collapse must lose identity.")
    print(json.dumps(out, indent=2))
    print(f"\n=== {out.get('verdict','(no baseline/consolidation pair)')}", file=sys.stderr, flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
